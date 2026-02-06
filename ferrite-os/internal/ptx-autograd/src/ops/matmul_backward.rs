//! Backward implementations for matrix multiplication.

use ptx_tensor::Tensor;
use ptx_runtime::{Result, Error};

/// Backward for matrix multiplication C = A @ B
///
/// For C = A @ B where A: (M, K), B: (K, N), C: (M, N):
/// - dL/dA = dL/dC @ B^T  (M, N) @ (N, K) = (M, K)
/// - dL/dB = A^T @ dL/dC  (K, M) @ (M, N) = (K, N)
pub fn matmul_backward(
    _grad_out: &Tensor,  // (M, N)
    _a: &Tensor,         // (M, K)
    _b: &Tensor,         // (K, N)
) -> Result<(Tensor, Tensor)> {
    // This requires transpose operations
    // grad_a = grad_out @ B^T
    // grad_b = A^T @ grad_out

    // For now, we need transpose support
    Err(Error::NotSupported {
        message: "matmul_backward requires transpose, not yet implemented".to_string(),
    })
}

/// Backward for batched matrix multiplication C[i] = A[i] @ B[i]
///
/// Same as matmul but applied per batch.
pub fn bmm_backward(
    _grad_out: &Tensor,  // (batch, M, N)
    _a: &Tensor,         // (batch, M, K)
    _b: &Tensor,         // (batch, K, N)
) -> Result<(Tensor, Tensor)> {
    Err(Error::NotSupported {
        message: "bmm_backward requires transpose, not yet implemented".to_string(),
    })
}

/// Backward for linear layer: y = x @ W^T + b
///
/// - dL/dx = dL/dy @ W
/// - dL/dW = dL/dy^T @ x = (x^T @ dL/dy)^T
/// - dL/db = sum(dL/dy, dim=0)
pub fn linear_backward(
    _grad_out: &Tensor,      // (batch, out_features)
    _input: &Tensor,         // (batch, in_features)
    _weight: &Tensor,        // (out_features, in_features)
    _bias_requires_grad: bool,
) -> Result<(Tensor, Tensor, Option<Tensor>)> {
    // This also requires transpose
    Err(Error::NotSupported {
        message: "linear_backward requires transpose, not yet implemented".to_string(),
    })
}
