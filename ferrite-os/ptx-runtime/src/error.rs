//! Error types for PTX-OS runtime operations.

use std::ffi::CStr;
use thiserror::Error;

/// Result type for PTX-OS operations.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors that can occur during PTX-OS runtime operations.
#[derive(Error, Debug)]
pub enum Error {
    /// Runtime initialization failed
    #[error("Failed to initialize PTX-OS runtime on device {device_id}")]
    InitFailed { device_id: i32 },

    /// Memory allocation failed
    #[error("Failed to allocate {size} bytes on GPU")]
    AllocationFailed { size: usize },

    /// Invalid pointer passed to operation
    #[error("Invalid GPU pointer")]
    InvalidPointer,

    /// Stream operation failed
    #[error("Stream operation failed: {message}")]
    StreamError { message: String },

    /// CUDA graph operation failed
    #[error("CUDA graph operation failed: {message}")]
    GraphError { message: String },

    /// cuBLAS operation failed
    #[error("cuBLAS error: {status}")]
    CublasError { status: i32 },

    /// CUDA error
    #[error("CUDA error {code}: {message}")]
    CudaError { code: i32, message: String },

    /// Shape mismatch
    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    /// Data type mismatch
    #[error("Data type mismatch: expected {expected:?}, got {actual:?}")]
    DTypeMismatch {
        expected: ptx_sys::PTXDType,
        actual: ptx_sys::PTXDType,
    },

    /// Runtime not initialized
    #[error("PTX-OS runtime not initialized")]
    NotInitialized,

    /// Stable runtime API failure
    #[error("PTX stable API error {status}: {message}")]
    StableApi { status: i32, message: String },

    /// Operation not supported
    #[error("Operation not supported: {message}")]
    NotSupported { message: String },

    /// Internal error
    #[error("Internal error: {message}")]
    Internal { message: String },
}

impl Error {
    /// Create a CUDA error from error code
    pub fn cuda(code: i32) -> Self {
        let message = unsafe {
            let ptr = ptx_sys::cudaGetErrorString(code);
            if ptr.is_null() {
                "Unknown CUDA error".to_string()
            } else {
                CStr::from_ptr(ptr).to_string_lossy().into_owned()
            }
        };
        Error::CudaError { code, message }
    }

    /// Create a cuBLAS error from status code
    pub fn cublas(status: i32) -> Self {
        Error::CublasError { status }
    }

    /// Check CUDA result and return error if not success
    pub fn check_cuda(code: i32) -> Result<()> {
        if code == ptx_sys::CUDA_SUCCESS {
            Ok(())
        } else {
            Err(Error::cuda(code))
        }
    }

    /// Check cuBLAS result and return error if not success
    pub fn check_cublas(status: i32) -> Result<()> {
        if status == ptx_sys::CUBLAS_STATUS_SUCCESS {
            Ok(())
        } else {
            Err(Error::cublas(status))
        }
    }

    /// Create a stable runtime API error from status code.
    pub fn stable(status: ptx_sys::PTXStableStatus) -> Self {
        let message = unsafe {
            let ptr = ptx_sys::ptx_stable_strerror(status);
            if ptr.is_null() {
                "Unknown stable API error".to_string()
            } else {
                CStr::from_ptr(ptr).to_string_lossy().into_owned()
            }
        };
        Error::StableApi {
            status: status as i32,
            message,
        }
    }

    /// Check stable API status and return error if not OK.
    pub fn check_stable(status: ptx_sys::PTXStableStatus) -> Result<()> {
        if status == ptx_sys::PTXStableStatus::Ok {
            Ok(())
        } else {
            Err(Error::stable(status))
        }
    }
}
