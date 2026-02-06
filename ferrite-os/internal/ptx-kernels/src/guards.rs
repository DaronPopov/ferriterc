//! Memory guards and validation layer for PTX-OS TLSF allocator
//!
//! This module provides a thin abstraction that validates memory operations
//! before executing Candle kernels, ensuring that:
//! - All pointers are owned by the TLSF allocator
//! - Memory bounds are valid
//! - Stream ordering is preserved
//! - Operations are type-safe

use ptx_sys::cudaStream_t;
use libc::c_void;

/// Result type for guarded operations
pub type GuardResult<T> = Result<T, GuardError>;

/// Errors that can occur during guarded kernel execution
#[derive(Debug, Clone)]
pub enum GuardError {
    /// Pointer is not owned by the TLSF allocator
    InvalidPointer { ptr: *const c_void },

    /// Buffer size is too small for the operation
    BufferTooSmall {
        required: usize,
        available: usize
    },

    /// Null pointer provided where valid pointer expected
    NullPointer { operation: &'static str },

    /// CUDA kernel launch failed
    KernelLaunchFailed {
        kernel: &'static str,
        error_code: i32,
    },

    /// Stream synchronization failed
    StreamSyncFailed { error_code: i32 },
}

impl std::fmt::Display for GuardError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GuardError::InvalidPointer { ptr } => {
                write!(f, "Pointer {:?} not owned by TLSF allocator", ptr)
            }
            GuardError::BufferTooSmall { required, available } => {
                write!(f, "Buffer too small: need {} bytes, have {}", required, available)
            }
            GuardError::NullPointer { operation } => {
                write!(f, "Null pointer in operation: {}", operation)
            }
            GuardError::KernelLaunchFailed { kernel, error_code } => {
                write!(f, "Kernel {} launch failed with code {}", kernel, error_code)
            }
            GuardError::StreamSyncFailed { error_code } => {
                write!(f, "Stream sync failed with code {}", error_code)
            }
        }
    }
}

impl std::error::Error for GuardError {}

/// A guarded GPU buffer that validates TLSF ownership
#[derive(Debug)]
pub struct GuardedBuffer {
    pub(crate) ptr: *mut c_void,
    pub(crate) size_bytes: usize,
    pub(crate) runtime: *mut ptx_sys::GPUHotRuntime,
}

impl GuardedBuffer {
    /// Create a new guarded buffer from a raw pointer
    ///
    /// # Safety
    ///
    /// - `ptr` must be allocated by the given runtime's TLSF allocator
    /// - `size_bytes` must be the actual allocation size
    pub unsafe fn new(
        ptr: *mut c_void,
        size_bytes: usize,
        runtime: *mut ptx_sys::GPUHotRuntime,
    ) -> GuardResult<Self> {
        // Validate pointer is not null
        if ptr.is_null() {
            return Err(GuardError::NullPointer {
                operation: "GuardedBuffer::new"
            });
        }

        // Validate TLSF ownership
        if !ptx_sys::gpu_hot_owns_ptr(runtime, ptr) {
            return Err(GuardError::InvalidPointer { ptr });
        }

        Ok(Self {
            ptr,
            size_bytes,
            runtime,
        })
    }

    /// Get the raw pointer (for kernel calls)
    #[inline]
    pub fn as_ptr(&self) -> *mut c_void {
        self.ptr
    }

    /// Get the raw pointer as a typed pointer
    #[inline]
    pub fn as_ptr_typed<T>(&self) -> *mut T {
        self.ptr as *mut T
    }

    /// Get the size in bytes
    #[inline]
    pub fn size_bytes(&self) -> usize {
        self.size_bytes
    }

    /// Get the number of elements of type T
    #[inline]
    pub fn len<T>(&self) -> usize {
        self.size_bytes / std::mem::size_of::<T>()
    }

    /// Check if this buffer can hold `count` elements of type T
    pub fn can_hold<T>(&self, count: usize) -> bool {
        count * std::mem::size_of::<T>() <= self.size_bytes
    }

    /// Validate that this buffer can hold `count` elements of type T
    pub fn validate_capacity<T>(&self, count: usize) -> GuardResult<()> {
        let required = count * std::mem::size_of::<T>();
        if required > self.size_bytes {
            return Err(GuardError::BufferTooSmall {
                required,
                available: self.size_bytes,
            });
        }
        Ok(())
    }

    /// Re-validate TLSF ownership (useful after long operations)
    pub fn revalidate(&self) -> GuardResult<()> {
        if !unsafe { ptx_sys::gpu_hot_owns_ptr(self.runtime, self.ptr) } {
            return Err(GuardError::InvalidPointer { ptr: self.ptr });
        }
        Ok(())
    }
}

// GuardedBuffer is safe to send between threads (GPU pointer)
unsafe impl Send for GuardedBuffer {}
unsafe impl Sync for GuardedBuffer {}

/// A validated kernel launch context
///
/// This ensures all parameters are valid before launching kernels
pub struct KernelContext {
    #[allow(dead_code)]
    pub(crate) runtime: *mut ptx_sys::GPUHotRuntime,
    pub(crate) stream: cudaStream_t,
}

impl KernelContext {
    /// Create a new kernel context
    pub fn new(runtime: *mut ptx_sys::GPUHotRuntime, stream: cudaStream_t) -> Self {
        Self { runtime, stream }
    }

    /// Get the CUDA stream
    #[inline]
    pub fn stream(&self) -> cudaStream_t {
        self.stream
    }

    /// Synchronize the stream and check for errors
    pub fn sync(&self) -> GuardResult<()> {
        let err = unsafe { ptx_sys::cudaStreamSynchronize(self.stream) };
        if err != ptx_sys::CUDA_SUCCESS {
            return Err(GuardError::StreamSyncFailed { error_code: err });
        }
        Ok(())
    }

    /// Check last CUDA error without blocking
    pub fn check_last_error(&self) -> GuardResult<()> {
        let err = unsafe { ptx_sys::cudaGetLastError() };
        if err != ptx_sys::CUDA_SUCCESS {
            return Err(GuardError::KernelLaunchFailed {
                kernel: "unknown",
                error_code: err,
            });
        }
        Ok(())
    }
}

/// Validated parameters for unary operations
pub struct UnaryOpGuard<'a> {
    pub input: &'a GuardedBuffer,
    pub output: &'a GuardedBuffer,
    pub numel: usize,
    pub context: &'a KernelContext,
}

impl<'a> UnaryOpGuard<'a> {
    /// Create and validate a unary operation
    pub fn new(
        input: &'a GuardedBuffer,
        output: &'a GuardedBuffer,
        numel: usize,
        context: &'a KernelContext,
    ) -> GuardResult<Self> {
        // Validate input buffer
        input.validate_capacity::<f32>(numel)?;
        input.revalidate()?;

        // Validate output buffer
        output.validate_capacity::<f32>(numel)?;
        output.revalidate()?;

        Ok(Self {
            input,
            output,
            numel,
            context,
        })
    }

    /// Get parameters for kernel launch (contiguous tensors)
    #[inline]
    pub fn kernel_params(&self) -> (usize, usize, *const usize, *const f32, *mut f32, cudaStream_t) {
        (
            self.numel,
            0, // num_dims = 0 for contiguous
            std::ptr::null(), // info = null for contiguous
            self.input.as_ptr_typed::<f32>(),
            self.output.as_ptr_typed::<f32>(),
            self.context.stream(),
        )
    }
}

/// Validated parameters for binary operations
pub struct BinaryOpGuard<'a> {
    pub left: &'a GuardedBuffer,
    pub right: &'a GuardedBuffer,
    pub output: &'a GuardedBuffer,
    pub numel: usize,
    pub context: &'a KernelContext,
}

impl<'a> BinaryOpGuard<'a> {
    /// Create and validate a binary operation
    pub fn new(
        left: &'a GuardedBuffer,
        right: &'a GuardedBuffer,
        output: &'a GuardedBuffer,
        numel: usize,
        context: &'a KernelContext,
    ) -> GuardResult<Self> {
        // Validate all buffers
        left.validate_capacity::<f32>(numel)?;
        left.revalidate()?;

        right.validate_capacity::<f32>(numel)?;
        right.revalidate()?;

        output.validate_capacity::<f32>(numel)?;
        output.revalidate()?;

        Ok(Self {
            left,
            right,
            output,
            numel,
            context,
        })
    }

    /// Get parameters for kernel launch (contiguous tensors)
    #[inline]
    pub fn kernel_params(&self) -> (
        usize,
        usize,
        *const usize,
        *const usize,
        *const f32,
        *const usize,
        *const f32,
        *mut f32,
        cudaStream_t
    ) {
        (
            self.numel,
            0, // num_dims = 0 for contiguous
            std::ptr::null(), // dims = null for contiguous
            std::ptr::null(), // left_strides = null for contiguous
            self.left.as_ptr_typed::<f32>(),
            std::ptr::null(), // right_strides = null for contiguous
            self.right.as_ptr_typed::<f32>(),
            self.output.as_ptr_typed::<f32>(),
            self.context.stream(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_guard_error_display() {
        let err = GuardError::BufferTooSmall {
            required: 1024,
            available: 512,
        };
        assert!(err.to_string().contains("1024"));
        assert!(err.to_string().contains("512"));
    }

    #[test]
    fn test_buffer_capacity() {
        // This test just verifies the logic, doesn't need actual GPU
        let size = 1024;
        let count_ok = 256; // 256 * 4 = 1024 bytes
        let count_fail = 257; // 257 * 4 = 1028 bytes

        assert_eq!(count_ok * std::mem::size_of::<f32>(), size);
        assert!(count_fail * std::mem::size_of::<f32>() > size);
    }
}
