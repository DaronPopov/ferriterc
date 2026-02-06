//! Softmax operations.

use crate::tensor::Tensor;
use crate::dtype::DType;
use ptx_runtime::{Result, Error, increment_ops};

impl Tensor {
    /// Softmax along the last dimension.
    ///
    /// softmax(x)_i = exp(x_i) / sum(exp(x_j))
    pub fn softmax(&self, dim: i32) -> Result<Tensor> {
        // Handle negative dimension
        let ndim = self.ndim() as i32;
        let dim = if dim < 0 { ndim + dim } else { dim };
        if dim < 0 || dim >= ndim {
            return Err(Error::Internal {
                message: format!("Invalid dimension {} for tensor with {} dims", dim, ndim),
            });
        }
        let dim = dim as usize;

        // For now, we only support softmax along the last dimension
        if dim != self.ndim() - 1 {
            return Err(Error::NotSupported {
                message: "Softmax only supported along last dimension".to_string(),
            });
        }

        let output = self.empty_like()?;
        let batch: usize = self.shape()[..dim].iter().product();
        let softmax_dim = self.shape()[dim];
        let stream = self.runtime().next_stream();

        match self.dtype() {
            DType::F32 => unsafe {
                ptx_sys::ptx_tensor_softmax_f32(
                    self.data_ptr_typed::<f32>(),
                    output.data_ptr_typed::<f32>(),
                    batch,
                    softmax_dim,
                    stream.raw(),
                );
            },
            DType::F64 => unsafe {
                ptx_sys::ptx_tensor_softmax_f64(
                    self.data_ptr_typed::<f64>(),
                    output.data_ptr_typed::<f64>(),
                    batch,
                    softmax_dim,
                    stream.raw(),
                );
            },
            DType::F16 => unsafe {
                ptx_sys::ptx_tensor_softmax_f16(
                    self.data_ptr_typed::<ptx_sys::__half>(),
                    output.data_ptr_typed::<ptx_sys::__half>(),
                    batch,
                    softmax_dim,
                    stream.raw(),
                );
            },
            dtype => return Err(Error::NotSupported {
                message: format!("Softmax not supported for {:?}", dtype),
            }),
        }

        increment_ops();
        Ok(output)
    }

    /// Log softmax along the last dimension.
    ///
    /// log_softmax(x)_i = log(softmax(x)_i) = x_i - log(sum(exp(x_j)))
    pub fn log_softmax(&self, dim: i32) -> Result<Tensor> {
        // Handle negative dimension
        let ndim = self.ndim() as i32;
        let dim = if dim < 0 { ndim + dim } else { dim };
        if dim < 0 || dim >= ndim {
            return Err(Error::Internal {
                message: format!("Invalid dimension {} for tensor with {} dims", dim, ndim),
            });
        }
        let dim = dim as usize;

        // For now, we only support softmax along the last dimension
        if dim != self.ndim() - 1 {
            return Err(Error::NotSupported {
                message: "Log softmax only supported along last dimension".to_string(),
            });
        }

        let output = self.empty_like()?;
        let batch: usize = self.shape()[..dim].iter().product();
        let softmax_dim = self.shape()[dim];
        let stream = self.runtime().next_stream();

        match self.dtype() {
            DType::F32 => unsafe {
                ptx_sys::ptx_tensor_log_softmax_f32(
                    self.data_ptr_typed::<f32>(),
                    output.data_ptr_typed::<f32>(),
                    batch,
                    softmax_dim,
                    stream.raw(),
                );
            },
            dtype => return Err(Error::NotSupported {
                message: format!("Log softmax not supported for {:?}", dtype),
            }),
        }

        increment_ops();
        Ok(output)
    }
}
