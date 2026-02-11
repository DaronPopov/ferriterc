//! XLA TLSF Allocator Integration
//!
//! This crate provides a custom XLA allocator that uses PTX-OS TLSF
//! for O(1) GPU memory allocation for JAX/TensorFlow/XLA workloads.

use ptx_runtime::{PtxRuntime, PTXStableConfig};
use std::collections::HashMap;
use std::os::raw::c_void;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, Once, OnceLock};

mod policy;

static INIT: Once = Once::new();
static INITIALIZED: AtomicBool = AtomicBool::new(false);
static GLOBAL_RUNTIME: OnceLock<Arc<PtxRuntime>> = OnceLock::new();
static PTR_MAP: OnceLock<Mutex<HashMap<usize, Arc<ptx_runtime::GpuPtr>>>> = OnceLock::new();

unsafe extern "C" {
    fn xla_tlsf_init(device_id: i32);
    fn xla_tlsf_alloc(size: usize, alignment: usize) -> *mut c_void;
    fn xla_tlsf_free(ptr: *mut c_void, size: usize);
    fn xla_tlsf_stats();
}

fn runtime() -> Option<&'static Arc<PtxRuntime>> {
    GLOBAL_RUNTIME.get()
}

/// Initialize PTX runtime (called from C++)
#[unsafe(no_mangle)]
pub extern "C" fn tlsf_init_ffi(device_id: i32) {
    INIT.call_once(|| {
        let cfg: PTXStableConfig = policy::XlaTlsfPolicy::default().stable_config();

        match PtxRuntime::with_stable_config(device_id, Some(cfg)) {
            Ok(rt) => {
                let rt = Arc::new(rt);
                let _ = GLOBAL_RUNTIME.set(rt);
                let _ = PTR_MAP.set(Mutex::new(HashMap::new()));
                INITIALIZED.store(true, Ordering::Release);
                eprintln!("[XLA-TLSF-Rust] Runtime initialized on device {}", device_id);
            }
            Err(e) => {
                eprintln!("[XLA-TLSF-Rust] FATAL: Failed to initialize: {:?}", e);
            }
        }
    });
}

/// Allocate memory via TLSF (called from C++)
#[unsafe(no_mangle)]
pub extern "C" fn tlsf_alloc_ffi(size: usize) -> *mut c_void {
    let Some(rt) = runtime() else {
        eprintln!("[XLA-TLSF-Rust] ERROR: Runtime not initialized!");
        return std::ptr::null_mut();
    };

    match rt.alloc(size) {
        Ok(gpu_ptr) => {
            let raw_ptr = gpu_ptr.as_ptr() as usize;
            let gpu_ptr_arc = Arc::new(gpu_ptr);

            if let Some(map) = PTR_MAP.get() {
                if let Ok(mut ptr_map) = map.lock() {
                    ptr_map.insert(raw_ptr, gpu_ptr_arc);
                }
            }

            raw_ptr as *mut c_void
        }
        Err(e) => {
            eprintln!("[XLA-TLSF-Rust] Alloc failed ({} bytes): {:?}", size, e);
            std::ptr::null_mut()
        }
    }
}

/// Free memory via TLSF (called from C++)
#[unsafe(no_mangle)]
pub extern "C" fn tlsf_free_ffi(ptr: *mut c_void) {
    if ptr.is_null() {
        return;
    }

    let raw_ptr = ptr as usize;

    if let Some(map) = PTR_MAP.get() {
        if let Ok(mut ptr_map) = map.lock() {
            if let Some(gpu_ptr_arc) = ptr_map.remove(&raw_ptr) {
                drop(gpu_ptr_arc);
                return;
            }
        }
    }

    eprintln!(
        "[XLA-TLSF-Rust] WARNING: Attempted to free unknown pointer {:p}",
        ptr
    );
}

/// Print TLSF statistics (called from C++)
#[unsafe(no_mangle)]
pub extern "C" fn tlsf_print_stats_ffi() {
    if let Some(rt) = runtime() {
        let stats = rt.tlsf_stats();
        eprintln!("\n[XLA-TLSF] Statistics:");
        eprintln!("  Pool size:      {:.2} GB", stats.total_pool_size as f64 / 1e9);
        eprintln!("  Allocated:      {:.2} MB", stats.allocated_bytes as f64 / 1e6);
        eprintln!("  Peak:           {:.2} MB", stats.peak_allocated as f64 / 1e6);
        eprintln!("  Fragmentation:  {:.6}", stats.fragmentation_ratio);
        eprintln!("  Utilization:    {:.1}%\n", stats.utilization_percent);
    }
}

// Rust-friendly API for tests/examples
pub fn init_xla_allocator(device_id: usize) -> Result<(), String> {
    tlsf_init_ffi(device_id as i32);
    if INITIALIZED.load(Ordering::Acquire) {
        Ok(())
    } else {
        Err("Failed to initialize XLA allocator".to_string())
    }
}

pub fn get_tlsf_stats() -> Option<ptx_sys::TLSFPoolStats> {
    runtime().map(|rt| rt.tlsf_stats())
}

/// Initialize and use the C++ XLA allocator shim.
pub fn init_xla_cpp_allocator(device_id: i32) {
    unsafe { xla_tlsf_init(device_id) }
}

/// Allocate through the C++ shim (alignment defaults to 256 when 0 is passed).
pub fn xla_cpp_alloc(size: usize, alignment: usize) -> *mut c_void {
    let align = if alignment == 0 { 256 } else { alignment };
    unsafe { xla_tlsf_alloc(size, align) }
}

/// Free through the C++ shim.
pub fn xla_cpp_free(ptr: *mut c_void, size: usize) {
    unsafe { xla_tlsf_free(ptr, size) }
}

/// Print stats through the C++ shim.
pub fn xla_cpp_print_stats() {
    unsafe { xla_tlsf_stats() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init() {
        let result = init_xla_allocator(0);
        assert!(result.is_ok());
    }

    #[test]
    fn test_alloc_free() {
        init_xla_allocator(0).unwrap();

        let ptr = tlsf_alloc_ffi(1024);
        assert!(!ptr.is_null());

        tlsf_free_ffi(ptr);
    }
}
