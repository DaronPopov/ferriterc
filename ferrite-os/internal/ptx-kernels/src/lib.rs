//! PTX Kernels - Candle CUDA kernel bindings for PTX-OS runtime
//!
//! This crate provides FFI bindings to optimized CUDA kernels from Candle,
//! designed to work seamlessly with the PTX-OS TLSF allocator.
//!
//! # Architecture
//!
//! - Raw CUDA kernels compiled from .cu files
//! - C launcher wrappers for kernel invocation
//! - Rust FFI bindings (this crate)
//! - Safe high-level wrappers (future: ptx-compute)
//!
//! # Memory Model
//!
//! All kernels accept raw GPU pointers allocated via PTX-OS TLSF allocator:
//! - `gpu_hot_alloc()` or `gpu_hot_alloc_async()`
//! - No framework-specific tensor types
//! - Direct pointer passing for zero overhead

#![allow(non_camel_case_types)]

pub mod test_kernels;
pub mod sha256;
pub mod guards;
pub mod candle;  // Now enabled with architecture fixes
pub mod safe_api;

// Re-export commonly used types
pub use guards::{GuardedBuffer, KernelContext, GuardError, GuardResult};
pub use safe_api::{unary, binary};  // Safe wrappers for Candle kernels

/// Re-export cudaStream_t from ptx-sys
pub use ptx_sys::cudaStream_t;

/// Check CUDA errors and panic with message
#[inline]
pub fn check_cuda(err: ptx_sys::cudaError_t, msg: &str) {
    if err != ptx_sys::CUDA_SUCCESS {
        let err_str = unsafe {
            let ptr = ptx_sys::cudaGetErrorString(err);
            std::ffi::CStr::from_ptr(ptr)
                .to_str()
                .unwrap_or("unknown error")
        };
        panic!("{}: {} (code {})", msg, err_str, err);
    }
}

/// Synchronize a CUDA stream and check for errors
#[inline]
pub fn sync_stream(stream: cudaStream_t) {
    let err = unsafe { ptx_sys::cudaStreamSynchronize(stream) };
    check_cuda(err, "Stream synchronization failed");
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_module_loads() {
        // Just verify the module compiles and links
        assert!(true);
    }
}
