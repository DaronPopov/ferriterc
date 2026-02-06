//! Backward implementations for reduction operations.

use ptx_tensor::Tensor;
use ptx_runtime::{Result, Error};

/// Backward for sum reduction along a dimension.
///
/// The gradient needs to be broadcast back to the input shape.
pub fn sum_backward(
    grad_out: &Tensor,
    input_shape: &[usize],
    dim: i32,
    _keepdim: bool,
) -> Result<Tensor> {
    // For sum, we need to broadcast grad_out back to input_shape
    // This requires an expand/broadcast operation

    // Handle negative dimension
    let ndim = input_shape.len() as i32;
    let dim = if dim < 0 { ndim + dim } else { dim };

    if dim < 0 || dim >= ndim {
        return Err(Error::Internal {
            message: format!("Invalid dimension {} for {} dims", dim, ndim),
        });
    }
    let _dim = dim as usize;

    // If keepdim is false, we need to unsqueeze the grad_out first
    // Then broadcast to input shape

    // For now, handle the simple case of full reduction (scalar output)
    if grad_out.elem_count() == 1 {
        // Scalar gradient - broadcast to all elements
        return Tensor::full(input_shape, 1.0, grad_out.dtype(), grad_out.runtime());
    }

    // TODO: Implement proper broadcast for partial reductions
    Err(Error::NotSupported {
        message: "sum_backward for partial reduction requires broadcast, not yet implemented".to_string(),
    })
}

/// Backward for mean reduction along a dimension.
///
/// Similar to sum but scaled by 1/n where n is the reduction size.
pub fn mean_backward(
    grad_out: &Tensor,
    input_shape: &[usize],
    dim: i32,
    _keepdim: bool,
) -> Result<Tensor> {
    let ndim = input_shape.len() as i32;
    let dim = if dim < 0 { ndim + dim } else { dim };

    if dim < 0 || dim >= ndim {
        return Err(Error::Internal {
            message: format!("Invalid dimension {} for {} dims", dim, ndim),
        });
    }

    // Get the size of the reduced dimension
    let reduce_size = input_shape[dim as usize];
    let scale = 1.0 / reduce_size as f32;

    // For scalar output (full reduction)
    if grad_out.elem_count() == 1 {
        return Tensor::full(input_shape, scale, grad_out.dtype(), grad_out.runtime());
    }

    // TODO: Implement proper broadcast for partial reductions
    Err(Error::NotSupported {
        message: "mean_backward for partial reduction requires broadcast, not yet implemented".to_string(),
    })
}

/// Backward for max reduction.
///
/// Gradient flows only to the maximum element(s).
/// This requires storing the argmax indices.
pub fn max_backward(
    _grad_out: &Tensor,
    _input: &Tensor,
    _output: &Tensor,
    _dim: i32,
) -> Result<Tensor> {
    // This requires knowing which elements were the maximum
    // Ideally we'd store argmax during forward pass

    // Simple workaround: create mask where input == output (after broadcasting)
    // This is inefficient but works for simple cases

    Err(Error::NotSupported {
        message: "max_backward requires argmax storage, not yet implemented".to_string(),
    })
}

/// Backward for min reduction.
///
/// Similar to max backward.
pub fn min_backward(
    _grad_out: &Tensor,
    _input: &Tensor,
    _output: &Tensor,
    _dim: i32,
) -> Result<Tensor> {
    Err(Error::NotSupported {
        message: "min_backward requires argmin storage, not yet implemented".to_string(),
    })
}

/// Backward for softmax.
///
/// For softmax(x)_i = exp(x_i) / sum(exp(x_j)):
/// d_softmax/dx_i = softmax_i * (1 - softmax_i) when i = j
/// d_softmax/dx_i = -softmax_i * softmax_j when i != j
///
/// In matrix form: diag(s) - s * s^T
/// For batched softmax, this simplifies to:
/// grad_input = softmax * (grad_output - sum(grad_output * softmax, dim=-1, keepdim=True))
pub fn softmax_backward(
    grad_out: &Tensor,
    output: &Tensor,  // softmax output
    dim: i32,
) -> Result<Tensor> {
    // grad_input = output * (grad_out - sum(grad_out * output, dim))
    let grad_times_out = grad_out.mul(output)?;
    let sum_grad_out = grad_times_out.sum_keepdim(dim)?;
    let grad_input = output.mul(&grad_out.sub(&sum_grad_out)?)?;

    Ok(grad_input)
}

/// Backward for log_softmax.
///
/// log_softmax(x) = x - log(sum(exp(x)))
/// d_log_softmax/dx = 1 - softmax(x)
///
/// So: grad_input = grad_output - softmax * sum(grad_output, dim)
pub fn log_softmax_backward(
    grad_out: &Tensor,
    output: &Tensor,  // log_softmax output
    dim: i32,
) -> Result<Tensor> {
    // softmax = exp(log_softmax)
    let softmax = output.exp()?;

    // sum_grad = sum(grad_output, dim, keepdim=True)
    let sum_grad = grad_out.sum_keepdim(dim)?;

    // grad_input = grad_output - softmax * sum_grad
    let grad_input = grad_out.sub(&softmax.mul(&sum_grad)?)?;

    Ok(grad_input)
}
