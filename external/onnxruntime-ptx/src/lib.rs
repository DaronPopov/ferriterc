//! ONNX Runtime TLSF Allocator Integration
//!
//! This crate provides a custom ONNX Runtime GPU allocator that uses PTX-OS
//! TLSF for O(1) GPU memory allocation. It follows the same pattern as
//! ferrite-xla: C++ shim implementing OrtAllocator -> FFI -> Rust core.
//!
//! # Architecture
//!
//! ```text
//! ONNX Runtime session
//!   └─ OrtTLSFAllocator (C++)
//!       ├─ Alloc()  → ort_tlsf_alloc()  → Rust TLSF
//!       ├─ Free()   → ort_tlsf_free()   → Rust TLSF
//!       └─ Info()   → OrtMemoryInfo (CUDA device memory)
//! ```

use ptx_runtime::{PtxRuntime, PTXStableConfig};
use std::collections::HashMap;
use std::os::raw::c_void;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, Once, OnceLock};

pub mod policy;

static INIT: Once = Once::new();
static INITIALIZED: AtomicBool = AtomicBool::new(false);
static GLOBAL_RUNTIME: OnceLock<Arc<PtxRuntime>> = OnceLock::new();
static PTR_MAP: OnceLock<Mutex<HashMap<usize, Arc<ptx_runtime::GpuPtr>>>> = OnceLock::new();

unsafe extern "C" {
    fn ort_tlsf_create_allocator(device_id: i32);
    fn ort_tlsf_destroy_allocator();
}

fn runtime() -> Option<&'static Arc<PtxRuntime>> {
    GLOBAL_RUNTIME.get()
}

// ─── FFI entry points (called from C++ OrtTLSFAllocator) ─────────────────────

/// Initialize PTX runtime for ONNX Runtime (called from C++)
#[unsafe(no_mangle)]
pub extern "C" fn ort_tlsf_init(device_id: i32) {
    INIT.call_once(|| {
        let cfg: PTXStableConfig = policy::OrtTlsfPolicy::default().stable_config();

        match PtxRuntime::with_stable_config(device_id, Some(cfg)) {
            Ok(rt) => {
                let rt = Arc::new(rt);
                let _ = GLOBAL_RUNTIME.set(rt);
                let _ = PTR_MAP.set(Mutex::new(HashMap::new()));
                INITIALIZED.store(true, Ordering::Release);
                eprintln!("[ORT-TLSF-Rust] Runtime initialized on device {}", device_id);
            }
            Err(e) => {
                eprintln!("[ORT-TLSF-Rust] FATAL: Failed to initialize: {:?}", e);
            }
        }
    });
}

/// Allocate memory via TLSF (called from C++)
#[unsafe(no_mangle)]
pub extern "C" fn ort_tlsf_alloc(size: usize) -> *mut c_void {
    let Some(rt) = runtime() else {
        eprintln!("[ORT-TLSF-Rust] ERROR: Runtime not initialized!");
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
            eprintln!("[ORT-TLSF-Rust] Alloc failed ({} bytes): {:?}", size, e);
            std::ptr::null_mut()
        }
    }
}

/// Free memory via TLSF (called from C++)
#[unsafe(no_mangle)]
pub extern "C" fn ort_tlsf_free(ptr: *mut c_void) {
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
        "[ORT-TLSF-Rust] WARNING: Attempted to free unknown pointer {:p}",
        ptr
    );
}

/// Print TLSF statistics (called from C++)
#[unsafe(no_mangle)]
pub extern "C" fn ort_tlsf_print_stats() {
    if let Some(rt) = runtime() {
        let stats = rt.tlsf_stats();
        eprintln!("\n[ORT-TLSF] Statistics:");
        eprintln!("  Pool size:      {:.2} GB", stats.total_pool_size as f64 / 1e9);
        eprintln!("  Allocated:      {:.2} MB", stats.allocated_bytes as f64 / 1e6);
        eprintln!("  Peak:           {:.2} MB", stats.peak_allocated as f64 / 1e6);
        eprintln!("  Fragmentation:  {:.6}", stats.fragmentation_ratio);
        eprintln!("  Utilization:    {:.1}%\n", stats.utilization_percent);
    }
}

// ─── Rust-friendly public API ────────────────────────────────────────────────

/// Initialize the ORT TLSF allocator from Rust.
pub fn init_ort_allocator(device_id: usize) -> Result<(), String> {
    ort_tlsf_init(device_id as i32);
    if INITIALIZED.load(Ordering::Acquire) {
        Ok(())
    } else {
        Err("Failed to initialize ORT TLSF allocator".to_string())
    }
}

/// Get TLSF pool statistics. Returns `None` if the runtime is not initialized.
pub fn get_tlsf_stats() -> Option<ptx_sys::TLSFPoolStats> {
    runtime().map(|rt| rt.tlsf_stats())
}

/// Initialize the C++ ORT allocator shim (creates global OrtTLSFAllocator).
pub fn init_ort_cpp_allocator(device_id: i32) {
    unsafe { ort_tlsf_create_allocator(device_id) }
}

/// Destroy the C++ ORT allocator shim.
pub fn destroy_ort_cpp_allocator() {
    unsafe { ort_tlsf_destroy_allocator() }
}

/// Check if the runtime is initialized.
pub fn is_initialized() -> bool {
    INITIALIZED.load(Ordering::Acquire)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init() {
        let result = init_ort_allocator(0);
        assert!(result.is_ok());
    }

    #[test]
    fn test_alloc_free() {
        init_ort_allocator(0).unwrap();

        let ptr = ort_tlsf_alloc(1024);
        assert!(!ptr.is_null());

        ort_tlsf_free(ptr);
    }
}
