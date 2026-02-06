//! Gradient functions for automatic differentiation.

use ptx_tensor::Tensor;
use ptx_runtime::Result;

/// Trait for gradient functions.
///
/// A gradient function computes the gradients of inputs given the gradient
/// of the output.
pub trait GradFn: Send + Sync {
    /// Compute gradients for inputs.
    ///
    /// # Arguments
    ///
    /// * `grad_output` - Gradient of the output tensor
    /// * `saved` - Tensors saved during forward pass
    ///
    /// # Returns
    ///
    /// Vector of gradients for each input. `None` means the input doesn't
    /// require gradient.
    fn backward(&self, grad_output: &Tensor, saved: &[Tensor]) -> Result<Vec<Option<Tensor>>>;

    /// Get the name of this gradient function (for debugging).
    fn name(&self) -> &'static str;
}

// ============================================================================
// Identity (no-op)
// ============================================================================

pub struct IdentityBackward;

impl GradFn for IdentityBackward {
    fn backward(&self, grad_output: &Tensor, _saved: &[Tensor]) -> Result<Vec<Option<Tensor>>> {
        Ok(vec![Some(grad_output.clone_tensor()?)])
    }

    fn name(&self) -> &'static str {
        "IdentityBackward"
    }
}

// ============================================================================
// Binary Operations
// ============================================================================

/// Gradient for addition: d/da (a + b) = 1, d/db (a + b) = 1
pub struct AddBackward;

impl GradFn for AddBackward {
    fn backward(&self, grad_output: &Tensor, _saved: &[Tensor]) -> Result<Vec<Option<Tensor>>> {
        // Both inputs get the same gradient
        Ok(vec![
            Some(grad_output.clone_tensor()?),
            Some(grad_output.clone_tensor()?),
        ])
    }

    fn name(&self) -> &'static str {
        "AddBackward"
    }
}

/// Gradient for subtraction: d/da (a - b) = 1, d/db (a - b) = -1
pub struct SubBackward;

impl GradFn for SubBackward {
    fn backward(&self, grad_output: &Tensor, _saved: &[Tensor]) -> Result<Vec<Option<Tensor>>> {
        Ok(vec![
            Some(grad_output.clone_tensor()?),
            Some(grad_output.neg()?),
        ])
    }

    fn name(&self) -> &'static str {
        "SubBackward"
    }
}

/// Gradient for multiplication: d/da (a * b) = b, d/db (a * b) = a
/// Saved: [a, b]
pub struct MulBackward;

impl GradFn for MulBackward {
    fn backward(&self, grad_output: &Tensor, saved: &[Tensor]) -> Result<Vec<Option<Tensor>>> {
        let a = &saved[0];
        let b = &saved[1];
        Ok(vec![
            Some(grad_output.mul(b)?),  // d/da = grad * b
            Some(grad_output.mul(a)?),  // d/db = grad * a
        ])
    }

    fn name(&self) -> &'static str {
        "MulBackward"
    }
}

/// Gradient for division: d/da (a / b) = 1/b, d/db (a / b) = -a/b^2
/// Saved: [a, b]
pub struct DivBackward;

impl GradFn for DivBackward {
    fn backward(&self, grad_output: &Tensor, saved: &[Tensor]) -> Result<Vec<Option<Tensor>>> {
        let a = &saved[0];
        let b = &saved[1];

        // d/da = grad / b
        let grad_a = grad_output.div(b)?;

        // d/db = -grad * a / b^2 = -grad * a * (1/b) * (1/b)
        let b_inv = b.recip()?;
        let grad_b = grad_output.neg()?.mul(a)?.mul(&b_inv)?.mul(&b_inv)?;

        Ok(vec![Some(grad_a), Some(grad_b)])
    }

    fn name(&self) -> &'static str {
        "DivBackward"
    }
}

// ============================================================================
// Unary Operations
// ============================================================================

/// Gradient for negation: d/dx (-x) = -1
pub struct NegBackward;

impl GradFn for NegBackward {
    fn backward(&self, grad_output: &Tensor, _saved: &[Tensor]) -> Result<Vec<Option<Tensor>>> {
        Ok(vec![Some(grad_output.neg()?)])
    }

    fn name(&self) -> &'static str {
        "NegBackward"
    }
}

/// Gradient for exp: d/dx exp(x) = exp(x)
/// Saved: [output] (the exp(x) value)
pub struct ExpBackward;

impl GradFn for ExpBackward {
    fn backward(&self, grad_output: &Tensor, saved: &[Tensor]) -> Result<Vec<Option<Tensor>>> {
        let output = &saved[0];  // exp(x)
        Ok(vec![Some(grad_output.mul(output)?)])
    }

    fn name(&self) -> &'static str {
        "ExpBackward"
    }
}

/// Gradient for log: d/dx log(x) = 1/x
/// Saved: [input]
pub struct LogBackward;

impl GradFn for LogBackward {
    fn backward(&self, grad_output: &Tensor, saved: &[Tensor]) -> Result<Vec<Option<Tensor>>> {
        let input = &saved[0];
        Ok(vec![Some(grad_output.div(input)?)])
    }

    fn name(&self) -> &'static str {
        "LogBackward"
    }
}

/// Gradient for sqrt: d/dx sqrt(x) = 0.5 / sqrt(x)
/// Saved: [output] (the sqrt(x) value)
pub struct SqrtBackward;

impl GradFn for SqrtBackward {
    fn backward(&self, grad_output: &Tensor, saved: &[Tensor]) -> Result<Vec<Option<Tensor>>> {
        let output = &saved[0];  // sqrt(x)
        // grad * 0.5 / sqrt(x)
        let grad_input = grad_output.mul_scalar(0.5)?.div(output)?;
        Ok(vec![Some(grad_input)])
    }

    fn name(&self) -> &'static str {
        "SqrtBackward"
    }
}

/// Gradient for tanh: d/dx tanh(x) = 1 - tanh(x)^2
/// Saved: [output] (the tanh(x) value)
pub struct TanhBackward;

impl GradFn for TanhBackward {
    fn backward(&self, grad_output: &Tensor, saved: &[Tensor]) -> Result<Vec<Option<Tensor>>> {
        let output = &saved[0];  // tanh(x)
        // grad * (1 - tanh(x)^2)
        let tanh_sq = output.sqr()?;
        let one_minus = tanh_sq.affine(-1.0, 1.0)?;  // 1 - tanh^2
        let grad_input = grad_output.mul(&one_minus)?;
        Ok(vec![Some(grad_input)])
    }

    fn name(&self) -> &'static str {
        "TanhBackward"
    }
}

// ============================================================================
// Activation Gradients
// ============================================================================

/// Gradient for ReLU: d/dx relu(x) = 1 if x > 0 else 0
/// Saved: [input]
pub struct ReluBackward;

impl GradFn for ReluBackward {
    fn backward(&self, grad_output: &Tensor, saved: &[Tensor]) -> Result<Vec<Option<Tensor>>> {
        let input = &saved[0];
        // grad * (input > 0)
        // We use clamp(0, 1) on the sign to get 0 or 1
        // Actually, we need to implement this properly with a mask
        // For now, use the input to compute the mask via affine
        // This is a workaround - proper implementation needs backward kernels
        let positive = input.clamp(0.0, f32::INFINITY)?;
        let clamped = positive.clamp(1e-8, f32::INFINITY)?;
        let mask = positive.div(&clamped)?;  // 0 or 1
        Ok(vec![Some(grad_output.mul(&mask)?)])
    }

    fn name(&self) -> &'static str {
        "ReluBackward"
    }
}

/// Gradient for Sigmoid: d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
/// Saved: [output] (the sigmoid(x) value)
pub struct SigmoidBackward;

impl GradFn for SigmoidBackward {
    fn backward(&self, grad_output: &Tensor, saved: &[Tensor]) -> Result<Vec<Option<Tensor>>> {
        let output = &saved[0];  // sigmoid(x)
        // grad * sigmoid * (1 - sigmoid)
        let one_minus = output.affine(-1.0, 1.0)?;  // 1 - sigmoid
        let grad_input = grad_output.mul(output)?.mul(&one_minus)?;
        Ok(vec![Some(grad_input)])
    }

    fn name(&self) -> &'static str {
        "SigmoidBackward"
    }
}

/// Gradient for GELU (approximate): d/dx gelu(x) ≈ complex expression
/// Saved: [input]
pub struct GeluBackward;

impl GradFn for GeluBackward {
    fn backward(&self, grad_output: &Tensor, saved: &[Tensor]) -> Result<Vec<Option<Tensor>>> {
        // Approximate GELU gradient using tanh approximation
        // This is a simplified version - proper implementation needs the backward kernel
        let input = &saved[0];

        // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        // The derivative is complex, for now use finite difference approximation
        // or implement proper backward kernel in CUDA

        // Simplified: just use sigmoid approximation for now
        let sig = input.sigmoid()?;
        let grad_input = grad_output.mul(&sig)?;
        Ok(vec![Some(grad_input)])
    }

    fn name(&self) -> &'static str {
        "GeluBackward"
    }
}

// ============================================================================
// Reduction Gradients
// ============================================================================

/// Gradient for sum: broadcast grad_output back to input shape
/// Saved: [input_shape as tensor - or we store shape separately]
pub struct SumBackward {
    pub input_shape: Vec<usize>,
    pub dim: i32,
    pub keepdim: bool,
}

impl GradFn for SumBackward {
    fn backward(&self, grad_output: &Tensor, _saved: &[Tensor]) -> Result<Vec<Option<Tensor>>> {
        // For sum, gradient is broadcast (ones) back to input shape
        // grad_input = grad_output.expand(input_shape)
        // For now, we need to implement broadcast/expand
        // Simple case: full reduction
        if self.input_shape.len() == 1 {
            // Full reduction case - broadcast scalar to all elements
            let grad_input = ptx_tensor::Tensor::full(
                &self.input_shape,
                1.0,
                grad_output.dtype(),
                grad_output.runtime(),
            )?;
            Ok(vec![Some(grad_input.mul(grad_output)?)])
        } else {
            // Partial reduction - need expand/broadcast
            Err(ptx_runtime::Error::NotSupported {
                message: "Sum backward for partial reduction not yet implemented".to_string(),
            })
        }
    }

    fn name(&self) -> &'static str {
        "SumBackward"
    }
}

/// Gradient for mean: similar to sum but scaled by 1/n
pub struct MeanBackward {
    pub input_shape: Vec<usize>,
    pub dim: i32,
    pub keepdim: bool,
}

impl GradFn for MeanBackward {
    fn backward(&self, grad_output: &Tensor, _saved: &[Tensor]) -> Result<Vec<Option<Tensor>>> {
        let n: usize = self.input_shape.iter().product();
        let scale = 1.0 / n as f32;

        // For full reduction
        if self.input_shape.len() == 1 || self.dim == -1 {
            let grad_input = ptx_tensor::Tensor::full(
                &self.input_shape,
                scale,
                grad_output.dtype(),
                grad_output.runtime(),
            )?;
            Ok(vec![Some(grad_input.mul(grad_output)?)])
        } else {
            Err(ptx_runtime::Error::NotSupported {
                message: "Mean backward for partial reduction not yet implemented".to_string(),
            })
        }
    }

    fn name(&self) -> &'static str {
        "MeanBackward"
    }
}

// ============================================================================
// Softmax Gradient
// ============================================================================

/// Gradient for softmax
/// Saved: [output] (the softmax output)
pub struct SoftmaxBackward {
    pub dim: i32,
}

impl GradFn for SoftmaxBackward {
    fn backward(&self, grad_output: &Tensor, saved: &[Tensor]) -> Result<Vec<Option<Tensor>>> {
        let output = &saved[0];  // softmax(x)

        // For softmax: d/dx softmax(x)_i = softmax(x)_i * (1_{i=j} - softmax(x)_j)
        // In practice: grad_input = output * (grad_output - sum(grad_output * output))

        // sum(grad * softmax) along dim
        let grad_times_out = grad_output.mul(output)?;
        let sum = grad_times_out.sum_keepdim(self.dim)?;

        // grad_input = softmax * (grad - sum)
        let grad_input = output.mul(&grad_output.sub(&sum)?)?;

        Ok(vec![Some(grad_input)])
    }

    fn name(&self) -> &'static str {
        "SoftmaxBackward"
    }
}

// ============================================================================
// Matmul Gradient
// ============================================================================

/// Gradient for matmul: C = A @ B
/// d/dA = grad @ B.T
/// d/dB = A.T @ grad
/// Saved: [A, B]
pub struct MatmulBackward;

impl GradFn for MatmulBackward {
    fn backward(&self, _grad_output: &Tensor, saved: &[Tensor]) -> Result<Vec<Option<Tensor>>> {
        let _a = &saved[0];
        let _b = &saved[1];

        // Need transpose for proper backward
        // d/dA = grad @ B.T
        // d/dB = A.T @ grad

        Err(ptx_runtime::Error::NotSupported {
            message: "Matmul backward requires transpose, not yet implemented".to_string(),
        })
    }

    fn name(&self) -> &'static str {
        "MatmulBackward"
    }
}
