//! Matrix multiplication operations via cuBLAS.

use crate::tensor::Tensor;
use crate::dtype::DType;
use crate::shape::{Shape, contiguous_strides};
use crate::storage::Storage;
use ptx_runtime::{Result, Error, increment_ops, cublas::Gemm};

impl Tensor {
    /// Matrix multiplication: C = A @ B
    ///
    /// For 2D tensors:
    /// - A: (M, K)
    /// - B: (K, N)
    /// - C: (M, N)
    ///
    /// For batched 3D tensors:
    /// - A: (batch, M, K)
    /// - B: (batch, K, N)
    /// - C: (batch, M, N)
    pub fn matmul(&self, other: &Tensor) -> Result<Tensor> {
        // Check dtypes match
        if self.dtype() != other.dtype() {
            return Err(Error::DTypeMismatch {
                expected: self.dtype().to_ptx(),
                actual: other.dtype().to_ptx(),
            });
        }

        // Check dimensions
        match (self.ndim(), other.ndim()) {
            (2, 2) => self.matmul_2d(other),
            (3, 3) => self.bmm(other),
            _ => Err(Error::NotSupported {
                message: format!(
                    "matmul not supported for shapes {:?} @ {:?}",
                    self.shape(),
                    other.shape()
                ),
            }),
        }
    }

    /// 2D matrix multiplication.
    fn matmul_2d(&self, other: &Tensor) -> Result<Tensor> {
        let m = self.shape()[0];
        let k = self.shape()[1];
        let k2 = other.shape()[0];
        let n = other.shape()[1];

        if k != k2 {
            return Err(Error::ShapeMismatch {
                expected: vec![k],
                actual: vec![k2],
            });
        }

        // Create output tensor
        let out_shape = Shape::from_slice(&[m, n]);
        let out_storage = Storage::new(m * n, self.dtype(), self.runtime())?;
        let output = Tensor::from_storage(
            out_storage,
            out_shape.clone(),
            contiguous_strides(&out_shape),
            0,
        );

        // Get or create cuBLAS handle
        let gemm = Gemm::new()?;
        let stream = self.runtime().next_stream();
        gemm.set_stream(&stream)?;

        match self.dtype() {
            DType::F32 => unsafe {
                gemm.matmul_f32(
                    self.data_ptr_typed::<f32>(),
                    other.data_ptr_typed::<f32>(),
                    output.data_ptr_typed::<f32>(),
                    m,
                    n,
                    k,
                )?;
            },
            dtype => return Err(Error::NotSupported {
                message: format!("matmul not supported for {:?}", dtype),
            }),
        }

        increment_ops();
        Ok(output)
    }

    /// Batched matrix multiplication: C[i] = A[i] @ B[i]
    ///
    /// - A: (batch, M, K)
    /// - B: (batch, K, N)
    /// - C: (batch, M, N)
    pub fn bmm(&self, other: &Tensor) -> Result<Tensor> {
        if self.ndim() != 3 || other.ndim() != 3 {
            return Err(Error::Internal {
                message: "bmm requires 3D tensors".to_string(),
            });
        }

        let batch = self.shape()[0];
        let m = self.shape()[1];
        let k = self.shape()[2];
        let batch2 = other.shape()[0];
        let k2 = other.shape()[1];
        let n = other.shape()[2];

        if batch != batch2 {
            return Err(Error::ShapeMismatch {
                expected: vec![batch],
                actual: vec![batch2],
            });
        }

        if k != k2 {
            return Err(Error::ShapeMismatch {
                expected: vec![k],
                actual: vec![k2],
            });
        }

        // Create output tensor
        let out_shape = Shape::from_slice(&[batch, m, n]);
        let out_storage = Storage::new(batch * m * n, self.dtype(), self.runtime())?;
        let output = Tensor::from_storage(
            out_storage,
            out_shape.clone(),
            contiguous_strides(&out_shape),
            0,
        );

        // Get or create cuBLAS handle
        let gemm = Gemm::new()?;
        let stream = self.runtime().next_stream();
        gemm.set_stream(&stream)?;

        match self.dtype() {
            DType::F32 => unsafe {
                gemm.bmm_f32(
                    self.data_ptr_typed::<f32>(),
                    other.data_ptr_typed::<f32>(),
                    output.data_ptr_typed::<f32>(),
                    batch,
                    m,
                    n,
                    k,
                )?;
            },
            dtype => return Err(Error::NotSupported {
                message: format!("bmm not supported for {:?}", dtype),
            }),
        }

        increment_ops();
        Ok(output)
    }

    /// Transpose the last two dimensions.
    ///
    /// For 2D: (M, N) -> (N, M)
    /// For 3D: (B, M, N) -> (B, N, M)
    pub fn transpose(&self) -> Result<Tensor> {
        match self.ndim() {
            2 => {
                let _m = self.shape()[0];
                let _n = self.shape()[1];
                // TODO: Implement actual transpose kernel
                // For now, return error
                Err(Error::NotSupported {
                    message: "transpose not yet implemented".to_string(),
                })
            }
            _ => Err(Error::NotSupported {
                message: "transpose only supports 2D tensors".to_string(),
            }),
        }
    }

    /// Linear layer: out = x @ weight.T + bias
    ///
    /// - x: (*, in_features)
    /// - weight: (out_features, in_features)
    /// - bias: (out_features,) or None
    pub fn linear(&self, _weight: &Tensor, _bias: Option<&Tensor>) -> Result<Tensor> {
        // For now, just do matmul
        // TODO: fused linear kernel for better performance

        // x: (batch, in_features) @ weight.T: (in_features, out_features) = (batch, out_features)
        // We need weight.T, but we can compute x @ weight.T as (weight @ x.T).T
        // Or we can adjust the GEMM call

        // Actually for row-major: C = A @ B.T can be done by swapping args in cuBLAS
        // Let's just error for now until we have transpose
        Err(Error::NotSupported {
            message: "linear layer requires transpose, not yet implemented".to_string(),
        })
    }
}

/// Convenience function for matrix multiply.
pub fn matmul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    a.matmul(b)
}

/// Convenience function for batched matrix multiply.
pub fn bmm(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    a.bmm(b)
}
