//! Candle CUDA kernel FFI bindings
//!
//! These are direct bindings to the C launcher functions that invoke
//! the actual CUDA kernels from Candle.
//!
//! # Kernel Categories
//!
//! - **Unary**: Element-wise operations on single tensor
//! - **Binary**: Element-wise operations on two tensors
//! - **Ternary**: Three-input operations (where, clamp, etc.)
//! - **Reductions**: Sum, mean, max, min, etc.
//! - **Affine**: Scale and bias transformations
//!
//! # Memory Layout
//!
//! Kernels support both contiguous and strided tensors:
//! - Set `info` to `null()` for contiguous tensors
//! - Provide dims + strides for strided tensors

use libc::size_t;
use crate::cudaStream_t;

// ============================================================================
// Unary Operations (F32)
// ============================================================================

extern "C" {
    /// GELU activation (Gaussian Error Linear Unit) - Tanh approximation
    ///
    /// Formula: `x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))`
    pub fn candle_launch_ugelu_f32(
        numel: size_t,
        num_dims: size_t,
        info: *const size_t,
        inp: *const f32,
        out: *mut f32,
        stream: cudaStream_t,
    );

    /// ReLU activation (Rectified Linear Unit)
    ///
    /// Formula: `max(0, x)`
    pub fn candle_launch_urelu_f32(
        numel: size_t,
        num_dims: size_t,
        info: *const size_t,
        inp: *const f32,
        out: *mut f32,
        stream: cudaStream_t,
    );

    /// SiLU activation (Sigmoid Linear Unit) aka Swish
    ///
    /// Formula: `x / (1 + exp(-x))`
    pub fn candle_launch_usilu_f32(
        numel: size_t,
        num_dims: size_t,
        info: *const size_t,
        inp: *const f32,
        out: *mut f32,
        stream: cudaStream_t,
    );

    /// Sigmoid activation
    ///
    /// Formula: `1 / (1 + exp(-x))`
    pub fn candle_launch_usigmoid_f32(
        numel: size_t,
        num_dims: size_t,
        info: *const size_t,
        inp: *const f32,
        out: *mut f32,
        stream: cudaStream_t,
    );

    /// Absolute value
    pub fn candle_launch_uabs_f32(
        numel: size_t,
        num_dims: size_t,
        info: *const size_t,
        inp: *const f32,
        out: *mut f32,
        stream: cudaStream_t,
    );

    /// Square root
    pub fn candle_launch_usqrt_f32(
        numel: size_t,
        num_dims: size_t,
        info: *const size_t,
        inp: *const f32,
        out: *mut f32,
        stream: cudaStream_t,
    );

    /// Exponential (e^x)
    pub fn candle_launch_uexp_f32(
        numel: size_t,
        num_dims: size_t,
        info: *const size_t,
        inp: *const f32,
        out: *mut f32,
        stream: cudaStream_t,
    );

    /// Natural logarithm
    pub fn candle_launch_ulog_f32(
        numel: size_t,
        num_dims: size_t,
        info: *const size_t,
        inp: *const f32,
        out: *mut f32,
        stream: cudaStream_t,
    );

    /// Hyperbolic tangent
    pub fn candle_launch_utanh_f32(
        numel: size_t,
        num_dims: size_t,
        info: *const size_t,
        inp: *const f32,
        out: *mut f32,
        stream: cudaStream_t,
    );
}

// ============================================================================
// Binary Operations (F32)
// ============================================================================

extern "C" {
    /// Element-wise addition
    ///
    /// Formula: `out[i] = left[i] + right[i]`
    pub fn candle_launch_badd_f32(
        numel: size_t,
        num_dims: size_t,
        dims: *const size_t,
        left_strides: *const size_t,
        left_ptr: *const f32,
        right_strides: *const size_t,
        right_ptr: *const f32,
        out: *mut f32,
        stream: cudaStream_t,
    );

    /// Element-wise multiplication
    ///
    /// Formula: `out[i] = left[i] * right[i]`
    pub fn candle_launch_bmul_f32(
        numel: size_t,
        num_dims: size_t,
        dims: *const size_t,
        left_strides: *const size_t,
        left_ptr: *const f32,
        right_strides: *const size_t,
        right_ptr: *const f32,
        out: *mut f32,
        stream: cudaStream_t,
    );

    /// Element-wise subtraction
    ///
    /// Formula: `out[i] = left[i] - right[i]`
    pub fn candle_launch_bsub_f32(
        numel: size_t,
        num_dims: size_t,
        dims: *const size_t,
        left_strides: *const size_t,
        left_ptr: *const f32,
        right_strides: *const size_t,
        right_ptr: *const f32,
        out: *mut f32,
        stream: cudaStream_t,
    );

    /// Element-wise division
    ///
    /// Formula: `out[i] = left[i] / right[i]`
    pub fn candle_launch_bdiv_f32(
        numel: size_t,
        num_dims: size_t,
        dims: *const size_t,
        left_strides: *const size_t,
        left_ptr: *const f32,
        right_strides: *const size_t,
        right_ptr: *const f32,
        out: *mut f32,
        stream: cudaStream_t,
    );
}

// ============================================================================
// Safe Wrappers (Optional - for convenience)
// ============================================================================

/// Safe wrapper for unary operations on contiguous F32 tensors
pub mod unary_f32 {
    use super::*;

    /// Apply GELU activation in-place or to output buffer
    #[inline]
    pub unsafe fn gelu(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: cudaStream_t,
    ) {
        candle_launch_ugelu_f32(numel, 0, std::ptr::null(), input, output, stream);
    }

    /// Apply ReLU activation
    #[inline]
    pub unsafe fn relu(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: cudaStream_t,
    ) {
        candle_launch_urelu_f32(numel, 0, std::ptr::null(), input, output, stream);
    }

    /// Apply SiLU/Swish activation
    #[inline]
    pub unsafe fn silu(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: cudaStream_t,
    ) {
        candle_launch_usilu_f32(numel, 0, std::ptr::null(), input, output, stream);
    }

    /// Apply Sigmoid activation
    #[inline]
    pub unsafe fn sigmoid(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: cudaStream_t,
    ) {
        candle_launch_usigmoid_f32(numel, 0, std::ptr::null(), input, output, stream);
    }

    /// Compute absolute value
    #[inline]
    pub unsafe fn abs(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: cudaStream_t,
    ) {
        candle_launch_uabs_f32(numel, 0, std::ptr::null(), input, output, stream);
    }

    /// Compute square root
    #[inline]
    pub unsafe fn sqrt(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: cudaStream_t,
    ) {
        candle_launch_usqrt_f32(numel, 0, std::ptr::null(), input, output, stream);
    }

    /// Compute exponential
    #[inline]
    pub unsafe fn exp(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: cudaStream_t,
    ) {
        candle_launch_uexp_f32(numel, 0, std::ptr::null(), input, output, stream);
    }

    /// Compute natural logarithm
    #[inline]
    pub unsafe fn log(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: cudaStream_t,
    ) {
        candle_launch_ulog_f32(numel, 0, std::ptr::null(), input, output, stream);
    }

    /// Compute hyperbolic tangent
    #[inline]
    pub unsafe fn tanh(
        input: *const f32,
        output: *mut f32,
        numel: usize,
        stream: cudaStream_t,
    ) {
        candle_launch_utanh_f32(numel, 0, std::ptr::null(), input, output, stream);
    }
}

/// Safe wrapper for binary operations on contiguous F32 tensors
pub mod binary_f32 {
    use super::*;

    /// Element-wise addition (contiguous tensors)
    #[inline]
    pub unsafe fn add(
        left: *const f32,
        right: *const f32,
        output: *mut f32,
        numel: usize,
        stream: cudaStream_t,
    ) {
        candle_launch_badd_f32(
            numel,
            0,
            std::ptr::null(),
            std::ptr::null(),
            left,
            std::ptr::null(),
            right,
            output,
            stream,
        );
    }

    /// Element-wise multiplication (contiguous tensors)
    #[inline]
    pub unsafe fn mul(
        left: *const f32,
        right: *const f32,
        output: *mut f32,
        numel: usize,
        stream: cudaStream_t,
    ) {
        candle_launch_bmul_f32(
            numel,
            0,
            std::ptr::null(),
            std::ptr::null(),
            left,
            std::ptr::null(),
            right,
            output,
            stream,
        );
    }

    /// Element-wise subtraction (contiguous tensors)
    #[inline]
    pub unsafe fn sub(
        left: *const f32,
        right: *const f32,
        output: *mut f32,
        numel: usize,
        stream: cudaStream_t,
    ) {
        candle_launch_bsub_f32(
            numel,
            0,
            std::ptr::null(),
            std::ptr::null(),
            left,
            std::ptr::null(),
            right,
            output,
            stream,
        );
    }

    /// Element-wise division (contiguous tensors)
    #[inline]
    pub unsafe fn div(
        left: *const f32,
        right: *const f32,
        output: *mut f32,
        numel: usize,
        stream: cudaStream_t,
    ) {
        candle_launch_bdiv_f32(
            numel,
            0,
            std::ptr::null(),
            std::ptr::null(),
            left,
            std::ptr::null(),
            right,
            output,
            stream,
        );
    }
}
