//! Core tensor type.

use std::sync::Arc;
use std::fmt;

use ptx_runtime::{PtxRuntime, Result, Error};
use crate::dtype::DType;
use crate::shape::{Shape, Strides, contiguous_strides, elem_count, is_contiguous, size_bytes, checked_elem_count};
use crate::storage::Storage;

/// A multi-dimensional array stored on the GPU.
///
/// Tensors support:
/// - Element-wise operations (add, mul, etc.)
/// - Unary operations (exp, log, relu, etc.)
/// - Reductions (sum, mean, max, etc.)
/// - Matrix operations (matmul via cuBLAS)
/// - Automatic memory management
///
/// # Example
///
/// ```no_run
/// use ptx_tensor::{Tensor, DType};
/// use ptx_runtime::PtxRuntime;
/// use std::sync::Arc;
///
/// let runtime = Arc::new(PtxRuntime::new(0).unwrap());
///
/// // Create tensors
/// let a = Tensor::zeros(&[2, 3], DType::F32, &runtime).unwrap();
/// let b = Tensor::ones(&[2, 3], DType::F32, &runtime).unwrap();
///
/// // Operations
/// let c = a.add(&b).unwrap();
/// let d = c.relu().unwrap();
/// let e = d.sum(1).unwrap();  // Sum along dimension 1
/// ```
#[derive(Clone)]
pub struct Tensor {
    /// Underlying storage (may be shared with other tensors).
    storage: Storage,
    /// Shape of the tensor.
    shape: Shape,
    /// Strides for each dimension (in elements).
    strides: Strides,
    /// Offset into storage (in elements).
    offset: usize,
    /// Data type.
    dtype: DType,
}

impl Tensor {
    /// Create a new tensor with the given storage and shape.
    pub fn from_storage(
        storage: Storage,
        shape: Shape,
        strides: Strides,
        offset: usize,
    ) -> Self {
        let dtype = storage.dtype();
        Self {
            storage,
            shape,
            strides,
            offset,
            dtype,
        }
    }

    /// Create a new contiguous tensor with the given shape.
    pub fn new(shape: &[usize], dtype: DType, runtime: &Arc<PtxRuntime>) -> Result<Self> {
        let shape = Shape::from_slice(shape);
        let strides = contiguous_strides(&shape);
        let len = checked_elem_count(&shape).map_err(|msg| Error::Internal {
            message: msg.to_string(),
        })?;
        let storage = Storage::new(len, dtype, runtime)?;
        Ok(Self {
            storage,
            shape,
            strides,
            offset: 0,
            dtype,
        })
    }

    /// Create a tensor filled with zeros.
    pub fn zeros(shape: &[usize], dtype: DType, runtime: &Arc<PtxRuntime>) -> Result<Self> {
        let shape = Shape::from_slice(shape);
        let strides = contiguous_strides(&shape);
        let len = checked_elem_count(&shape).map_err(|msg| Error::Internal {
            message: msg.to_string(),
        })?;
        let storage = Storage::zeros(len, dtype, runtime)?;
        Ok(Self {
            storage,
            shape,
            strides,
            offset: 0,
            dtype,
        })
    }

    /// Create a tensor filled with ones.
    pub fn ones(shape: &[usize], dtype: DType, runtime: &Arc<PtxRuntime>) -> Result<Self> {
        let tensor = Self::zeros(shape, dtype, runtime)?;
        if dtype == DType::F32 {
            tensor.storage.fill_f32(1.0)?;
        } else {
            return Err(Error::NotSupported {
                message: format!("ones not implemented for {:?}", dtype),
            });
        }
        Ok(tensor)
    }

    /// Create a tensor filled with a specific value.
    pub fn full(shape: &[usize], value: f32, dtype: DType, runtime: &Arc<PtxRuntime>) -> Result<Self> {
        let tensor = Self::new(shape, dtype, runtime)?;
        if dtype == DType::F32 {
            tensor.storage.fill_f32(value)?;
        } else {
            return Err(Error::NotSupported {
                message: format!("full not implemented for {:?}", dtype),
            });
        }
        Ok(tensor)
    }

    /// Create a tensor from host data.
    pub fn from_slice<T: Copy>(
        data: &[T],
        shape: &[usize],
        dtype: DType,
        runtime: &Arc<PtxRuntime>,
    ) -> Result<Self> {
        let shape = Shape::from_slice(shape);
        let strides = contiguous_strides(&shape);
        let len = checked_elem_count(&shape).map_err(|msg| Error::Internal {
            message: msg.to_string(),
        })?;

        if len != data.len() {
            return Err(Error::ShapeMismatch {
                expected: vec![len],
                actual: vec![data.len()],
            });
        }

        let storage = unsafe { Storage::from_host(data, dtype, runtime)? };
        Ok(Self {
            storage,
            shape,
            strides,
            offset: 0,
            dtype,
        })
    }

    /// Create a 1D tensor with evenly spaced values: [start, start+step, start+2*step, ...].
    pub fn arange(start: f32, end: f32, step: f32, runtime: &Arc<PtxRuntime>) -> Result<Self> {
        let n = ((end - start) / step).ceil() as usize;
        if n == 0 {
            return Self::new(&[0], DType::F32, runtime);
        }
        let tensor = Self::new(&[n], DType::F32, runtime)?;
        let stream = runtime.next_stream();
        unsafe {
            ptx_sys::ptx_tensor_arange_f32(
                tensor.data_ptr_typed::<f32>(),
                n,
                start,
                step,
                stream.raw(),
            );
        }
        stream.synchronize()?;
        Ok(tensor)
    }

    /// Create a 1D tensor with n evenly spaced values from start to end (inclusive).
    pub fn linspace(start: f32, end: f32, steps: usize, runtime: &Arc<PtxRuntime>) -> Result<Self> {
        if steps == 0 {
            return Self::new(&[0], DType::F32, runtime);
        }
        if steps == 1 {
            return Self::from_slice(&[start], &[1], DType::F32, runtime);
        }
        let step = (end - start) / (steps - 1) as f32;
        let tensor = Self::new(&[steps], DType::F32, runtime)?;
        let stream = runtime.next_stream();
        unsafe {
            ptx_sys::ptx_tensor_arange_f32(
                tensor.data_ptr_typed::<f32>(),
                steps,
                start,
                step,
                stream.raw(),
            );
        }
        stream.synchronize()?;
        Ok(tensor)
    }

    /// Create an empty tensor with the same shape and dtype as another.
    pub fn empty_like(&self) -> Result<Self> {
        Self::new(&self.shape, self.dtype, self.runtime())
    }

    /// Create a zeros tensor with the same shape and dtype as another.
    pub fn zeros_like(&self) -> Result<Self> {
        Self::zeros(&self.shape, self.dtype, self.runtime())
    }

    // ========================================================================
    // Accessors
    // ========================================================================

    /// Get the shape.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get the strides.
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// Get the number of dimensions.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Get a specific dimension size.
    pub fn dim(&self, d: usize) -> usize {
        self.shape[d]
    }

    /// Get the total number of elements.
    pub fn elem_count(&self) -> usize {
        elem_count(&self.shape)
    }

    /// Get the size in bytes.
    pub fn size_bytes(&self) -> usize {
        size_bytes(&self.shape, self.dtype.size_bytes())
    }

    /// Get the data type.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Get the runtime.
    pub fn runtime(&self) -> &Arc<PtxRuntime> {
        self.storage.runtime()
    }

    /// Get the raw data pointer (with offset applied).
    pub fn data_ptr(&self) -> *mut libc::c_void {
        let base = self.storage.as_ptr() as *mut u8;
        unsafe { base.add(self.offset * self.dtype.size_bytes()) as *mut libc::c_void }
    }

    /// Get the typed data pointer.
    pub fn data_ptr_typed<T>(&self) -> *mut T {
        self.data_ptr() as *mut T
    }

    /// Check if the tensor is contiguous.
    pub fn is_contiguous(&self) -> bool {
        is_contiguous(&self.shape, &self.strides)
    }

    /// Check if this tensor shares storage with another.
    pub fn shares_storage(&self, other: &Tensor) -> bool {
        self.storage.same_storage(&other.storage)
    }

    // ========================================================================
    // Data Transfer
    // ========================================================================

    /// Copy tensor data to host.
    pub fn to_vec<T: Copy + Default>(&self) -> Result<Vec<T>> {
        // TODO: Handle non-contiguous tensors
        if !self.is_contiguous() {
            return Err(Error::NotSupported {
                message: "to_vec requires contiguous tensor".to_string(),
            });
        }
        self.storage.to_host()
    }

    /// Copy tensor data to host as f32.
    pub fn to_vec_f32(&self) -> Result<Vec<f32>> {
        if self.dtype != DType::F32 {
            return Err(Error::DTypeMismatch {
                expected: DType::F32.to_ptx(),
                actual: self.dtype.to_ptx(),
            });
        }
        self.to_vec::<f32>()
    }

    /// Copy tensor data to host as i32.
    pub fn to_vec_i32(&self) -> Result<Vec<i32>> {
        if self.dtype != DType::I32 {
            return Err(Error::DTypeMismatch {
                expected: DType::I32.to_ptx(),
                actual: self.dtype.to_ptx(),
            });
        }
        self.to_vec::<i32>()
    }

    /// Copy tensor data to host as u32.
    pub fn to_vec_u32(&self) -> Result<Vec<u32>> {
        if self.dtype != DType::U32 {
            return Err(Error::DTypeMismatch {
                expected: DType::U32.to_ptx(),
                actual: self.dtype.to_ptx(),
            });
        }
        self.to_vec::<u32>()
    }

    /// Copy tensor data to host as u8.
    pub fn to_vec_u8(&self) -> Result<Vec<u8>> {
        if self.dtype != DType::U8 {
            return Err(Error::DTypeMismatch {
                expected: DType::U8.to_ptx(),
                actual: self.dtype.to_ptx(),
            });
        }
        self.to_vec::<u8>()
    }

    // ========================================================================
    // Reshaping
    // ========================================================================

    /// Reshape the tensor (returns a view if possible).
    pub fn reshape(&self, new_shape: &[usize]) -> Result<Self> {
        let new_elem_count: usize = checked_elem_count(new_shape).map_err(|msg| Error::Internal {
            message: msg.to_string(),
        })?;
        if new_elem_count != self.elem_count() {
            return Err(Error::ShapeMismatch {
                expected: self.shape.to_vec(),
                actual: new_shape.to_vec(),
            });
        }

        if self.is_contiguous() {
            // Can return a view
            Ok(Self {
                storage: self.storage.clone(),
                shape: Shape::from_slice(new_shape),
                strides: contiguous_strides(new_shape),
                offset: self.offset,
                dtype: self.dtype,
            })
        } else {
            // Need to make contiguous first
            let contiguous = self.contiguous()?;
            Ok(Self {
                storage: contiguous.storage,
                shape: Shape::from_slice(new_shape),
                strides: contiguous_strides(new_shape),
                offset: 0,
                dtype: self.dtype,
            })
        }
    }

    /// Flatten the tensor to 1D.
    pub fn flatten(&self) -> Result<Self> {
        self.reshape(&[self.elem_count()])
    }

    /// Squeeze dimensions of size 1.
    pub fn squeeze(&self) -> Result<Self> {
        let new_shape: Shape = self.shape.iter()
            .copied()
            .filter(|&d| d != 1)
            .collect();
        self.reshape(&new_shape)
    }

    /// Unsqueeze: add a dimension of size 1 at the given position.
    pub fn unsqueeze(&self, dim: usize) -> Result<Self> {
        let mut new_shape = self.shape.clone();
        new_shape.insert(dim, 1);
        self.reshape(&new_shape)
    }

    /// Transpose two dimensions (returns a view — no data copy).
    pub fn transpose(&self, dim0: usize, dim1: usize) -> Result<Self> {
        if dim0 >= self.ndim() || dim1 >= self.ndim() {
            return Err(Error::Internal {
                message: format!("transpose dims ({}, {}) out of range for {} dims", dim0, dim1, self.ndim()),
            });
        }
        let mut new_shape = self.shape.clone();
        let mut new_strides = self.strides.clone();
        new_shape.swap(dim0, dim1);
        new_strides.swap(dim0, dim1);
        Ok(Self {
            storage: self.storage.clone(),
            shape: new_shape,
            strides: new_strides,
            offset: self.offset,
            dtype: self.dtype,
        })
    }

    /// Transpose a 2D matrix (shorthand for transpose(0, 1)).
    pub fn t(&self) -> Result<Self> {
        if self.ndim() != 2 {
            return Err(Error::Internal {
                message: format!("t() requires 2D tensor, got {}D", self.ndim()),
            });
        }
        self.transpose(0, 1)
    }

    /// Permute dimensions (returns a view — no data copy).
    pub fn permute(&self, dims: &[usize]) -> Result<Self> {
        if dims.len() != self.ndim() {
            return Err(Error::Internal {
                message: format!("permute requires {} dims, got {}", self.ndim(), dims.len()),
            });
        }
        let mut new_shape = Shape::with_capacity(self.ndim());
        let mut new_strides = Strides::with_capacity(self.ndim());
        for &d in dims {
            if d >= self.ndim() {
                return Err(Error::Internal {
                    message: format!("permute dim {} out of range for {} dims", d, self.ndim()),
                });
            }
            new_shape.push(self.shape[d]);
            new_strides.push(self.strides[d]);
        }
        Ok(Self {
            storage: self.storage.clone(),
            shape: new_shape,
            strides: new_strides,
            offset: self.offset,
            dtype: self.dtype,
        })
    }

    /// Make the tensor contiguous (copy if necessary).
    pub fn contiguous(&self) -> Result<Self> {
        if self.is_contiguous() {
            return Ok(self.clone());
        }

        let n = self.elem_count();
        let out = Self::new(self.shape(), self.dtype, self.runtime())?;
        let stream = self.runtime().next_stream();

        // Upload shape and strides to GPU for the kernel
        let shape_bytes = self.ndim() * std::mem::size_of::<usize>();
        let shape_gpu = self.runtime().alloc(shape_bytes)?;
        let strides_gpu = self.runtime().alloc(shape_bytes)?;
        unsafe {
            shape_gpu.copy_from_host(
                self.shape().as_ptr() as *const libc::c_void,
                shape_bytes,
            )?;
            strides_gpu.copy_from_host(
                self.strides().as_ptr() as *const libc::c_void,
                shape_bytes,
            )?;
        }

        match self.dtype {
            DType::F32 => unsafe {
                ptx_sys::ptx_tensor_strided_copy_f32(
                    self.data_ptr_typed::<f32>(),
                    out.data_ptr_typed::<f32>(),
                    shape_gpu.as_ptr() as *const usize,
                    strides_gpu.as_ptr() as *const usize,
                    self.ndim() as i32,
                    n,
                    stream.raw(),
                );
            },
            DType::F64 => unsafe {
                ptx_sys::ptx_tensor_strided_copy_f64(
                    self.data_ptr_typed::<f64>(),
                    out.data_ptr_typed::<f64>(),
                    shape_gpu.as_ptr() as *const usize,
                    strides_gpu.as_ptr() as *const usize,
                    self.ndim() as i32,
                    n,
                    stream.raw(),
                );
            },
            DType::U8 => unsafe {
                ptx_sys::ptx_tensor_strided_copy_u8(
                    self.data_ptr_typed::<u8>(),
                    out.data_ptr_typed::<u8>(),
                    shape_gpu.as_ptr() as *const usize,
                    strides_gpu.as_ptr() as *const usize,
                    self.ndim() as i32,
                    n,
                    stream.raw(),
                );
            },
            _ => {
                return Err(Error::NotSupported {
                    message: format!("contiguous() not yet supported for {:?}", self.dtype),
                });
            }
        }

        stream.synchronize()?;
        Ok(out)
    }

    /// Return self if contiguous, else make a contiguous copy.
    pub fn require_contiguous(&self) -> Result<Self> {
        if self.is_contiguous() {
            Ok(self.clone())
        } else {
            self.contiguous()
        }
    }

    /// Create a deep copy of the tensor.
    pub fn clone_tensor(&self) -> Result<Self> {
        let new_storage = self.storage.deep_clone()?;
        Ok(Self {
            storage: new_storage,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            offset: self.offset,
            dtype: self.dtype,
        })
    }

    // ========================================================================
    // Synchronization
    // ========================================================================

    /// Synchronize all pending operations on this tensor.
    pub fn sync(&self) {
        self.runtime().sync_all();
    }
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Tensor(shape={:?}, dtype={}, contiguous={})",
            self.shape.as_slice(),
            self.dtype,
            self.is_contiguous()
        )
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tensor{:?}", self.shape.as_slice())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: These tests require a CUDA-capable GPU
    #[test]
    #[ignore]
    fn test_tensor_creation() {
        let runtime = Arc::new(PtxRuntime::new(0).unwrap());
        let tensor = Tensor::zeros(&[2, 3], DType::F32, &runtime).unwrap();
        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(tensor.elem_count(), 6);
        assert!(tensor.is_contiguous());
    }

    #[test]
    #[ignore]
    fn test_tensor_reshape() {
        let runtime = Arc::new(PtxRuntime::new(0).unwrap());
        let tensor = Tensor::zeros(&[2, 3], DType::F32, &runtime).unwrap();
        let reshaped = tensor.reshape(&[6]).unwrap();
        assert_eq!(reshaped.shape(), &[6]);
        assert_eq!(reshaped.elem_count(), 6);
    }
}
