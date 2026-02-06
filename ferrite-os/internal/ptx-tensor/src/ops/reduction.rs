//! Reduction operations.

use crate::tensor::Tensor;
use crate::dtype::DType;
use crate::shape::{contiguous_strides, reduction_sizes, reduce_shape};
use crate::storage::Storage;
use ptx_runtime::{Result, Error, increment_ops};

/// Reduction operation type.
#[derive(Debug, Clone, Copy)]
pub enum ReductionOp {
    Sum,
    Mean,
    Max,
    Min,
}

impl Tensor {
    /// Generic reduction operation along a dimension.
    fn reduce_op(&self, dim: i32, op: ReductionOp, keepdim: bool) -> Result<Tensor> {
        // Handle negative dimension
        let ndim = self.ndim() as i32;
        let dim = if dim < 0 { ndim + dim } else { dim };
        if dim < 0 || dim >= ndim {
            return Err(Error::Internal {
                message: format!("Invalid dimension {} for tensor with {} dims", dim, ndim),
            });
        }
        let dim = dim as usize;

        // Calculate output shape
        let out_shape = reduce_shape(self.shape(), dim, keepdim);
        let (outer, reduce, inner) = reduction_sizes(self.shape(), dim);

        // Create output tensor
        let out_storage = Storage::new(
            out_shape.iter().product(),
            self.dtype(),
            self.runtime(),
        )?;
        let output = Tensor::from_storage(
            out_storage,
            out_shape.clone(),
            contiguous_strides(&out_shape),
            0,
        );

        let stream = self.runtime().next_stream();

        match self.dtype() {
            DType::F32 => unsafe {
                let input = self.data_ptr_typed::<f32>();
                let out = output.data_ptr_typed::<f32>();

                match op {
                    ReductionOp::Sum => ptx_sys::ptx_tensor_reduce_sum_f32(
                        input, out, outer, reduce, inner, stream.raw(),
                    ),
                    ReductionOp::Mean => ptx_sys::ptx_tensor_reduce_mean_f32(
                        input, out, outer, reduce, inner, stream.raw(),
                    ),
                    ReductionOp::Max => ptx_sys::ptx_tensor_reduce_max_f32(
                        input, out, outer, reduce, inner, stream.raw(),
                    ),
                    ReductionOp::Min => ptx_sys::ptx_tensor_reduce_min_f32(
                        input, out, outer, reduce, inner, stream.raw(),
                    ),
                }
            },
            DType::F64 => unsafe {
                let input = self.data_ptr_typed::<f64>();
                let out = output.data_ptr_typed::<f64>();

                match op {
                    ReductionOp::Sum => ptx_sys::ptx_tensor_reduce_sum_f64(
                        input, out, outer, reduce, inner, stream.raw(),
                    ),
                    ReductionOp::Max => ptx_sys::ptx_tensor_reduce_max_f64(
                        input, out, outer, reduce, inner, stream.raw(),
                    ),
                    ReductionOp::Min => ptx_sys::ptx_tensor_reduce_min_f64(
                        input, out, outer, reduce, inner, stream.raw(),
                    ),
                    ReductionOp::Mean => {
                        // Implement mean as sum / count
                        ptx_sys::ptx_tensor_reduce_sum_f64(
                            input, out, outer, reduce, inner, stream.raw(),
                        );
                        // Scale by 1/reduce
                        let scale = 1.0 / reduce as f64;
                        ptx_sys::ptx_tensor_affine_f64(
                            out,
                            out,
                            outer * inner,
                            scale,
                            0.0,
                            stream.raw(),
                        );
                    }
                }
            },
            dtype => return Err(Error::NotSupported {
                message: format!("Reduction not supported for {:?}", dtype),
            }),
        }

        increment_ops();
        Ok(output)
    }

    /// Sum along a dimension.
    pub fn sum(&self, dim: i32) -> Result<Tensor> {
        self.reduce_op(dim, ReductionOp::Sum, false)
    }

    /// Sum along a dimension, keeping the dimension.
    pub fn sum_keepdim(&self, dim: i32) -> Result<Tensor> {
        self.reduce_op(dim, ReductionOp::Sum, true)
    }

    /// Mean along a dimension.
    pub fn mean(&self, dim: i32) -> Result<Tensor> {
        self.reduce_op(dim, ReductionOp::Mean, false)
    }

    /// Mean along a dimension, keeping the dimension.
    pub fn mean_keepdim(&self, dim: i32) -> Result<Tensor> {
        self.reduce_op(dim, ReductionOp::Mean, true)
    }

    /// Max along a dimension.
    pub fn max(&self, dim: i32) -> Result<Tensor> {
        self.reduce_op(dim, ReductionOp::Max, false)
    }

    /// Max along a dimension, keeping the dimension.
    pub fn max_keepdim(&self, dim: i32) -> Result<Tensor> {
        self.reduce_op(dim, ReductionOp::Max, true)
    }

    /// Min along a dimension.
    pub fn min(&self, dim: i32) -> Result<Tensor> {
        self.reduce_op(dim, ReductionOp::Min, false)
    }

    /// Min along a dimension, keeping the dimension.
    pub fn min_keepdim(&self, dim: i32) -> Result<Tensor> {
        self.reduce_op(dim, ReductionOp::Min, true)
    }

    /// Sum all elements (full reduction).
    pub fn sum_all(&self) -> Result<Tensor> {
        let flat = self.flatten()?;
        flat.sum(0)
    }

    /// Mean of all elements (full reduction).
    pub fn mean_all(&self) -> Result<Tensor> {
        let flat = self.flatten()?;
        flat.mean(0)
    }

    /// Max of all elements (full reduction).
    pub fn max_all(&self) -> Result<Tensor> {
        let flat = self.flatten()?;
        flat.max(0)
    }

    /// Min of all elements (full reduction).
    pub fn min_all(&self) -> Result<Tensor> {
        let flat = self.flatten()?;
        flat.min(0)
    }

    /// Variance along a dimension.
    ///
    /// var(x) = mean((x - mean(x))^2)
    pub fn var(&self, dim: i32) -> Result<Tensor> {
        let mean = self.mean_keepdim(dim)?;
        // Broadcast mean back to original shape
        let diff = self.sub(&mean)?;
        let sq = diff.sqr()?;
        sq.mean(dim)
    }

    /// Standard deviation along a dimension.
    pub fn std(&self, dim: i32) -> Result<Tensor> {
        self.var(dim)?.sqrt()
    }
}
