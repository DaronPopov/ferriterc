//! Candle + PTX-OS TLSF Allocator Integration
//!
//! This crate provides candle-core with all GPU allocations routed through
//! the PTX-OS TLSF allocator. It works by patching candle's `cudarc`
//! dependency to the local `cudarc-ptx` fork with the `ptx-alloc` feature,
//! so `malloc_sync()` calls automatically use TLSF. Zero candle code changes.
//!
//! # Usage
//!
//! ```no_run
//! // Initialize TLSF before any candle GPU operations
//! candle_ptx::init().expect("TLSF init failed");
//!
//! // Use candle normally — all GPU allocs go through TLSF
//! use candle_ptx::candle_core;
//! let device = candle_core::Device::new_cuda(0).unwrap();
//! let a = candle_core::Tensor::randn(0f32, 1.0, (128, 128), &device).unwrap();
//! ```

pub use candle_core;

use ptx_runtime::PtxRuntime;
use std::sync::Arc;

/// Initialize the PTX-OS TLSF runtime on device 0.
///
/// Must be called before any candle CUDA operations to ensure TLSF is active
/// when cudarc performs its first `malloc_sync()`. If the runtime is already
/// initialized (e.g. by another adapter), this is a no-op.
pub fn init() -> Result<(), String> {
    init_on_device(0)
}

/// Initialize the PTX-OS TLSF runtime on a specific device.
pub fn init_on_device(device_id: i32) -> Result<(), String> {
    match ptx_runtime::init_global_runtime(device_id) {
        Ok(_) => {
            eprintln!("[candle-ptx] TLSF runtime initialized on device {}", device_id);
            Ok(())
        }
        Err(e) => Err(format!("[candle-ptx] Failed to initialize TLSF runtime: {:?}", e)),
    }
}

/// Get a handle to the global PTX runtime (if initialized).
pub fn runtime() -> Result<Arc<PtxRuntime>, String> {
    ptx_runtime::global_runtime()
        .map_err(|e| format!("[candle-ptx] Runtime not available: {:?}", e))
}

/// Get TLSF pool statistics. Returns `None` if the runtime is not initialized.
pub fn get_tlsf_stats() -> Option<ptx_runtime::TLSFPoolStats> {
    ptx_runtime::global_runtime().ok().map(|rt| rt.tlsf_stats())
}

/// Print TLSF pool statistics to stderr.
pub fn print_tlsf_stats() {
    if let Some(stats) = get_tlsf_stats() {
        eprintln!("\n[candle-ptx] TLSF Statistics:");
        eprintln!("  Pool size:      {:.2} GB", stats.total_pool_size as f64 / 1e9);
        eprintln!("  Allocated:      {:.2} MB", stats.allocated_bytes as f64 / 1e6);
        eprintln!("  Peak:           {:.2} MB", stats.peak_allocated as f64 / 1e6);
        eprintln!("  Fragmentation:  {:.6}", stats.fragmentation_ratio);
        eprintln!("  Utilization:    {:.1}%\n", stats.utilization_percent);
    } else {
        eprintln!("[candle-ptx] Runtime not initialized, no stats available");
    }
}
