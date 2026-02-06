//! Safe high-level API for Candle kernels with TLSF memory validation
//!
//! This module provides ergonomic wrappers around Candle kernels that:
//! - Automatically validate TLSF allocator ownership
//! - Perform bounds checking
//! - Ensure type safety
//! - Provide clear error messages

use crate::guards::{GuardedBuffer, KernelContext, UnaryOpGuard, BinaryOpGuard, GuardResult};

/// Safe unary operations on F32 tensors
pub mod unary {
    use super::*;

    /// Apply GELU activation (Gaussian Error Linear Unit)
    ///
    /// Formula: `x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))`
    ///
    /// # Safety
    ///
    /// This function validates all memory bounds and TLSF ownership before
    /// launching the kernel.
    pub fn gelu(
        input: &GuardedBuffer,
        output: &GuardedBuffer,
        numel: usize,
        context: &KernelContext,
    ) -> GuardResult<()> {
        let guard = UnaryOpGuard::new(input, output, numel, context)?;
        let params = guard.kernel_params();

        unsafe {
            crate::candle::candle_launch_ugelu_f32(
                params.0, params.1, params.2, params.3, params.4, params.5
            );
        }

        context.check_last_error()?;
        Ok(())
    }

    /// Apply ReLU activation (Rectified Linear Unit)
    ///
    /// Formula: `max(0, x)`
    pub fn relu(
        input: &GuardedBuffer,
        output: &GuardedBuffer,
        numel: usize,
        context: &KernelContext,
    ) -> GuardResult<()> {
        let guard = UnaryOpGuard::new(input, output, numel, context)?;
        let params = guard.kernel_params();

        unsafe {
            crate::candle::candle_launch_urelu_f32(
                params.0, params.1, params.2, params.3, params.4, params.5
            );
        }

        context.check_last_error()?;
        Ok(())
    }

    /// Apply SiLU/Swish activation
    ///
    /// Formula: `x / (1 + exp(-x))`
    pub fn silu(
        input: &GuardedBuffer,
        output: &GuardedBuffer,
        numel: usize,
        context: &KernelContext,
    ) -> GuardResult<()> {
        let guard = UnaryOpGuard::new(input, output, numel, context)?;
        let params = guard.kernel_params();

        unsafe {
            crate::candle::candle_launch_usilu_f32(
                params.0, params.1, params.2, params.3, params.4, params.5
            );
        }

        context.check_last_error()?;
        Ok(())
    }

    /// Apply Sigmoid activation
    ///
    /// Formula: `1 / (1 + exp(-x))`
    pub fn sigmoid(
        input: &GuardedBuffer,
        output: &GuardedBuffer,
        numel: usize,
        context: &KernelContext,
    ) -> GuardResult<()> {
        let guard = UnaryOpGuard::new(input, output, numel, context)?;
        let params = guard.kernel_params();

        unsafe {
            crate::candle::candle_launch_usigmoid_f32(
                params.0, params.1, params.2, params.3, params.4, params.5
            );
        }

        context.check_last_error()?;
        Ok(())
    }

    /// Compute absolute value
    pub fn abs(
        input: &GuardedBuffer,
        output: &GuardedBuffer,
        numel: usize,
        context: &KernelContext,
    ) -> GuardResult<()> {
        let guard = UnaryOpGuard::new(input, output, numel, context)?;
        let params = guard.kernel_params();

        unsafe {
            crate::candle::candle_launch_uabs_f32(
                params.0, params.1, params.2, params.3, params.4, params.5
            );
        }

        context.check_last_error()?;
        Ok(())
    }

    /// Compute square root
    pub fn sqrt(
        input: &GuardedBuffer,
        output: &GuardedBuffer,
        numel: usize,
        context: &KernelContext,
    ) -> GuardResult<()> {
        let guard = UnaryOpGuard::new(input, output, numel, context)?;
        let params = guard.kernel_params();

        unsafe {
            crate::candle::candle_launch_usqrt_f32(
                params.0, params.1, params.2, params.3, params.4, params.5
            );
        }

        context.check_last_error()?;
        Ok(())
    }

    /// Compute exponential (e^x)
    pub fn exp(
        input: &GuardedBuffer,
        output: &GuardedBuffer,
        numel: usize,
        context: &KernelContext,
    ) -> GuardResult<()> {
        let guard = UnaryOpGuard::new(input, output, numel, context)?;
        let params = guard.kernel_params();

        unsafe {
            crate::candle::candle_launch_uexp_f32(
                params.0, params.1, params.2, params.3, params.4, params.5
            );
        }

        context.check_last_error()?;
        Ok(())
    }

    /// Compute natural logarithm
    pub fn log(
        input: &GuardedBuffer,
        output: &GuardedBuffer,
        numel: usize,
        context: &KernelContext,
    ) -> GuardResult<()> {
        let guard = UnaryOpGuard::new(input, output, numel, context)?;
        let params = guard.kernel_params();

        unsafe {
            crate::candle::candle_launch_ulog_f32(
                params.0, params.1, params.2, params.3, params.4, params.5
            );
        }

        context.check_last_error()?;
        Ok(())
    }

    /// Compute hyperbolic tangent
    pub fn tanh(
        input: &GuardedBuffer,
        output: &GuardedBuffer,
        numel: usize,
        context: &KernelContext,
    ) -> GuardResult<()> {
        let guard = UnaryOpGuard::new(input, output, numel, context)?;
        let params = guard.kernel_params();

        unsafe {
            crate::candle::candle_launch_utanh_f32(
                params.0, params.1, params.2, params.3, params.4, params.5
            );
        }

        context.check_last_error()?;
        Ok(())
    }
}

/// Safe binary operations on F32 tensors
pub mod binary {
    use super::*;

    /// Element-wise addition: `out[i] = left[i] + right[i]`
    pub fn add(
        left: &GuardedBuffer,
        right: &GuardedBuffer,
        output: &GuardedBuffer,
        numel: usize,
        context: &KernelContext,
    ) -> GuardResult<()> {
        let guard = BinaryOpGuard::new(left, right, output, numel, context)?;
        let params = guard.kernel_params();

        unsafe {
            crate::candle::candle_launch_badd_f32(
                params.0, params.1, params.2, params.3, params.4,
                params.5, params.6, params.7, params.8
            );
        }

        context.check_last_error()?;
        Ok(())
    }

    /// Element-wise multiplication: `out[i] = left[i] * right[i]`
    pub fn mul(
        left: &GuardedBuffer,
        right: &GuardedBuffer,
        output: &GuardedBuffer,
        numel: usize,
        context: &KernelContext,
    ) -> GuardResult<()> {
        let guard = BinaryOpGuard::new(left, right, output, numel, context)?;
        let params = guard.kernel_params();

        unsafe {
            crate::candle::candle_launch_bmul_f32(
                params.0, params.1, params.2, params.3, params.4,
                params.5, params.6, params.7, params.8
            );
        }

        context.check_last_error()?;
        Ok(())
    }

    /// Element-wise subtraction: `out[i] = left[i] - right[i]`
    pub fn sub(
        left: &GuardedBuffer,
        right: &GuardedBuffer,
        output: &GuardedBuffer,
        numel: usize,
        context: &KernelContext,
    ) -> GuardResult<()> {
        let guard = BinaryOpGuard::new(left, right, output, numel, context)?;
        let params = guard.kernel_params();

        unsafe {
            crate::candle::candle_launch_bsub_f32(
                params.0, params.1, params.2, params.3, params.4,
                params.5, params.6, params.7, params.8
            );
        }

        context.check_last_error()?;
        Ok(())
    }

    /// Element-wise division: `out[i] = left[i] / right[i]`
    pub fn div(
        left: &GuardedBuffer,
        right: &GuardedBuffer,
        output: &GuardedBuffer,
        numel: usize,
        context: &KernelContext,
    ) -> GuardResult<()> {
        let guard = BinaryOpGuard::new(left, right, output, numel, context)?;
        let params = guard.kernel_params();

        unsafe {
            crate::candle::candle_launch_bdiv_f32(
                params.0, params.1, params.2, params.3, params.4,
                params.5, params.6, params.7, params.8
            );
        }

        context.check_last_error()?;
        Ok(())
    }
}
