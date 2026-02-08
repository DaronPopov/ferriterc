use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::Arc;

use ptx_kernels::candle::{binary_f32, scan_f32, topk_f32, unary_f32};
use ptx_runtime::{GpuPtr, PtxRuntime};
use thiserror::Error;

pub type Result<T> = std::result::Result<T, LangError>;

#[derive(Debug, Error)]
pub enum LangError {
    #[error("invalid shape {0:?}: must be non-empty and have no zero dimensions")]
    InvalidShape(Vec<usize>),
    #[error("invalid node reference: {0}")]
    InvalidNodeRef(usize),
    #[error("program has no output")]
    MissingOutput,
    #[error("shape mismatch for op {op}: left={left:?}, right={right:?}")]
    ShapeMismatch {
        op: &'static str,
        left: Vec<usize>,
        right: Vec<usize>,
    },
    #[error("expected {expected} input tensors, got {got}")]
    InputCountMismatch { expected: usize, got: usize },
    #[error("input shape mismatch at slot {index}: expected {expected:?}, got {got:?}")]
    InputShapeMismatch {
        index: usize,
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    #[error("input data len mismatch at slot {index}: expected {expected}, got {got}")]
    InputLenMismatch {
        index: usize,
        expected: usize,
        got: usize,
    },
    #[error("transfer error: {message}")]
    Transfer { message: String },
    #[error(transparent)]
    Runtime(#[from] ptx_runtime::Error),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ValueId(usize);

impl ValueId {
    pub fn index(self) -> usize {
        self.0
    }
}

#[derive(Clone, Debug)]
pub enum Op {
    Input { shape: Vec<usize> },
    Relu(ValueId),
    Tanh(ValueId),
    Sigmoid(ValueId),
    Add { lhs: ValueId, rhs: ValueId },
    Mul { lhs: ValueId, rhs: ValueId },
    CumSum { input: ValueId, dim: usize },
    TopK { input: ValueId, k: usize, dim: usize, largest: bool },
}

#[derive(Clone, Debug)]
struct Node {
    op: Op,
}

#[derive(Clone, Debug)]
pub struct Program {
    nodes: Vec<Node>,
    output: Option<ValueId>,
}

impl Program {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            output: None,
        }
    }

    pub fn input(&mut self, shape: &[usize]) -> Result<ValueId> {
        validate_shape(shape)?;
        Ok(self.push(Op::Input {
            shape: shape.to_vec(),
        }))
    }

    pub fn relu(&mut self, x: ValueId) -> ValueId {
        self.push(Op::Relu(x))
    }

    pub fn tanh(&mut self, x: ValueId) -> ValueId {
        self.push(Op::Tanh(x))
    }

    pub fn sigmoid(&mut self, x: ValueId) -> ValueId {
        self.push(Op::Sigmoid(x))
    }

    pub fn add(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId {
        self.push(Op::Add { lhs, rhs })
    }

    pub fn mul(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId {
        self.push(Op::Mul { lhs, rhs })
    }

    pub fn cumsum(&mut self, input: ValueId, dim: usize) -> ValueId {
        self.push(Op::CumSum { input, dim })
    }

    /// TopK — returns the values tensor (dim-th dimension replaced by k).
    pub fn topk(&mut self, input: ValueId, k: usize, dim: usize, largest: bool) -> ValueId {
        self.push(Op::TopK { input, k, dim, largest })
    }

    pub fn set_output(&mut self, output: ValueId) {
        self.output = Some(output);
    }

    pub fn compile(self) -> Result<CompiledProgram> {
        let output = self.output.ok_or(LangError::MissingOutput)?;
        if output.index() >= self.nodes.len() {
            return Err(LangError::InvalidNodeRef(output.index()));
        }

        let mut shapes: Vec<Vec<usize>> = Vec::with_capacity(self.nodes.len());
        let mut numels: Vec<usize> = Vec::with_capacity(self.nodes.len());
        let mut input_ids = Vec::new();

        for (i, node) in self.nodes.iter().enumerate() {
            let (shape, numel) = match &node.op {
                Op::Input { shape } => {
                    validate_shape(shape)?;
                    input_ids.push(ValueId(i));
                    (shape.clone(), numel(shape))
                }
                Op::Relu(x) | Op::Tanh(x) | Op::Sigmoid(x) => {
                    let shape = get_shape(&shapes, *x)?;
                    let n = numel(&shape);
                    (shape, n)
                }
                Op::CumSum { input, dim } => {
                    let shape = get_shape(&shapes, *input)?;
                    if *dim >= shape.len() {
                        return Err(LangError::InvalidNodeRef(*dim));
                    }
                    let n = numel(&shape);
                    (shape, n)
                }
                Op::TopK { input, k, dim, .. } => {
                    let mut shape = get_shape(&shapes, *input)?;
                    if *dim >= shape.len() {
                        return Err(LangError::InvalidNodeRef(*dim));
                    }
                    shape[*dim] = *k;
                    let n = numel(&shape);
                    (shape, n)
                }
                Op::Add { lhs, rhs } | Op::Mul { lhs, rhs } => {
                    let lhs_shape = get_shape(&shapes, *lhs)?;
                    let rhs_shape = get_shape(&shapes, *rhs)?;
                    if lhs_shape != rhs_shape {
                        let op_name = if matches!(&node.op, Op::Add { .. }) {
                            "add"
                        } else {
                            "mul"
                        };
                        return Err(LangError::ShapeMismatch {
                            op: op_name,
                            left: lhs_shape,
                            right: rhs_shape,
                        });
                    }
                    let n = numel(&lhs_shape);
                    (lhs_shape, n)
                }
            };

            shapes.push(shape);
            numels.push(numel);
        }

        Ok(CompiledProgram {
            nodes: self.nodes,
            shapes,
            numels,
            input_ids,
            output,
        })
    }

    fn push(&mut self, op: Op) -> ValueId {
        let id = ValueId(self.nodes.len());
        self.nodes.push(Node { op });
        id
    }
}

impl Default for Program {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Debug)]
pub struct CompiledProgram {
    nodes: Vec<Node>,
    shapes: Vec<Vec<usize>>,
    numels: Vec<usize>,
    input_ids: Vec<ValueId>,
    output: ValueId,
}

impl CompiledProgram {
    pub fn input_shapes(&self) -> Vec<Vec<usize>> {
        self.input_ids
            .iter()
            .map(|id| self.shapes[id.index()].clone())
            .collect()
    }

    pub fn output_shape(&self) -> &[usize] {
        &self.shapes[self.output.index()]
    }
}

#[derive(Clone, Debug)]
pub struct HostTensor {
    shape: Vec<usize>,
    data: Vec<f32>,
}

impl HostTensor {
    pub fn new(shape: Vec<usize>, data: Vec<f32>) -> Result<Self> {
        validate_shape(&shape)?;
        let expected = numel(&shape);
        if expected != data.len() {
            return Err(LangError::InputLenMismatch {
                index: 0,
                expected,
                got: data.len(),
            });
        }
        Ok(Self { shape, data })
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }
}

pub struct GpuLangRuntime {
    runtime: Arc<PtxRuntime>,
}

impl GpuLangRuntime {
    pub fn new(device_id: i32) -> Result<Self> {
        let config = ptx_runtime::PTXStableConfig {
            struct_size: std::mem::size_of::<ptx_runtime::PTXStableConfig>() as u32,
            abi_version: ptx_runtime::PTX_STABLE_ABI_VERSION,
            flags: 0,
            device_id,
            // Use a conservative default so runtime init succeeds even when VRAM is contested.
            pool_fraction: 0.70,
            fixed_pool_size: 0,
            reserve_vram: 256 * 1024 * 1024,
            max_streams: 16,
            quiet_init: 0,
            enable_leak_detection: 1,
            enable_pool_health: 1,
            _reserved0: 0,
        };
        let runtime = Arc::new(PtxRuntime::with_stable_config(device_id, Some(config))?);
        runtime.export_for_hook();
        runtime.export_context();
        Ok(Self { runtime })
    }

    pub fn runtime(&self) -> &Arc<PtxRuntime> {
        &self.runtime
    }

    pub fn execute(&self, program: &CompiledProgram, inputs: &[HostTensor]) -> Result<HostTensor> {
        if inputs.len() != program.input_ids.len() {
            return Err(LangError::InputCountMismatch {
                expected: program.input_ids.len(),
                got: inputs.len(),
            });
        }

        for (i, (id, input)) in program.input_ids.iter().zip(inputs.iter()).enumerate() {
            let expected_shape = &program.shapes[id.index()];
            if input.shape() != expected_shape {
                return Err(LangError::InputShapeMismatch {
                    index: i,
                    expected: expected_shape.clone(),
                    got: input.shape().to_vec(),
                });
            }
            let expected_numel = program.numels[id.index()];
            if input.data().len() != expected_numel {
                return Err(LangError::InputLenMismatch {
                    index: i,
                    expected: expected_numel,
                    got: input.data().len(),
                });
            }
        }

        let stream = self.runtime.next_stream();
        let mut slots: Vec<Option<GpuPtr>> = (0..program.nodes.len()).map(|_| None).collect();
        let mut input_data_map: HashMap<usize, &HostTensor> = HashMap::new();
        for (i, id) in program.input_ids.iter().enumerate() {
            input_data_map.insert(id.index(), &inputs[i]);
        }

        for (i, node) in program.nodes.iter().enumerate() {
            let numel = program.numels[i];
            let bytes = numel * std::mem::size_of::<f32>();

            let out_ptr = match &node.op {
                Op::Input { .. } => {
                    let tensor = input_data_map
                        .get(&i)
                        .expect("input id must exist during execution");
                    let gpu = self.runtime.alloc(bytes)?;
                    unsafe {
                        gpu.copy_from_host(tensor.data().as_ptr() as *const c_void, bytes)?;
                    }
                    gpu
                }
                Op::Relu(x) => {
                    let inp = slots[x.index()]
                        .as_ref()
                        .ok_or(LangError::InvalidNodeRef(x.index()))?;
                    let out = self.runtime.alloc(bytes)?;
                    unsafe {
                        unary_f32::relu(
                            inp.as_ptr_typed::<f32>(),
                            out.as_ptr_typed::<f32>(),
                            numel,
                            stream.raw(),
                        );
                    }
                    out
                }
                Op::Tanh(x) => {
                    let inp = slots[x.index()]
                        .as_ref()
                        .ok_or(LangError::InvalidNodeRef(x.index()))?;
                    let out = self.runtime.alloc(bytes)?;
                    unsafe {
                        unary_f32::tanh(
                            inp.as_ptr_typed::<f32>(),
                            out.as_ptr_typed::<f32>(),
                            numel,
                            stream.raw(),
                        );
                    }
                    out
                }
                Op::Sigmoid(x) => {
                    let inp = slots[x.index()]
                        .as_ref()
                        .ok_or(LangError::InvalidNodeRef(x.index()))?;
                    let out = self.runtime.alloc(bytes)?;
                    unsafe {
                        unary_f32::sigmoid(
                            inp.as_ptr_typed::<f32>(),
                            out.as_ptr_typed::<f32>(),
                            numel,
                            stream.raw(),
                        );
                    }
                    out
                }
                Op::Add { lhs, rhs } => {
                    let l = slots[lhs.index()]
                        .as_ref()
                        .ok_or(LangError::InvalidNodeRef(lhs.index()))?;
                    let r = slots[rhs.index()]
                        .as_ref()
                        .ok_or(LangError::InvalidNodeRef(rhs.index()))?;
                    let out = self.runtime.alloc(bytes)?;
                    unsafe {
                        binary_f32::add(
                            l.as_ptr_typed::<f32>(),
                            r.as_ptr_typed::<f32>(),
                            out.as_ptr_typed::<f32>(),
                            numel,
                            stream.raw(),
                        );
                    }
                    out
                }
                Op::Mul { lhs, rhs } => {
                    let l = slots[lhs.index()]
                        .as_ref()
                        .ok_or(LangError::InvalidNodeRef(lhs.index()))?;
                    let r = slots[rhs.index()]
                        .as_ref()
                        .ok_or(LangError::InvalidNodeRef(rhs.index()))?;
                    let out = self.runtime.alloc(bytes)?;
                    unsafe {
                        binary_f32::mul(
                            l.as_ptr_typed::<f32>(),
                            r.as_ptr_typed::<f32>(),
                            out.as_ptr_typed::<f32>(),
                            numel,
                            stream.raw(),
                        );
                    }
                    out
                }
                Op::CumSum { input, dim } => {
                    let inp = slots[input.index()]
                        .as_ref()
                        .ok_or(LangError::InvalidNodeRef(input.index()))?;
                    let shape = &program.shapes[i];
                    let outer: usize = shape[..*dim].iter().product();
                    let dim_size = shape[*dim];
                    let inner: usize = shape[*dim + 1..].iter().product();
                    let out = self.runtime.alloc(bytes)?;
                    unsafe {
                        scan_f32::cumsum(
                            inp.as_ptr_typed::<f32>(),
                            out.as_ptr_typed::<f32>(),
                            outer,
                            dim_size,
                            inner,
                            stream.raw(),
                        );
                    }
                    out
                }
                Op::TopK { input, k, dim, largest } => {
                    let inp = slots[input.index()]
                        .as_ref()
                        .ok_or(LangError::InvalidNodeRef(input.index()))?;
                    // Input shape is needed to compute outer/dim_size/inner.
                    // The compile step stored the output shape (with dim replaced by k),
                    // so we need to recover the original dim_size from the input.
                    let inp_shape = &program.shapes[input.index()];
                    let outer: usize = inp_shape[..*dim].iter().product();
                    let dim_size = inp_shape[*dim];
                    let inner: usize = inp_shape[*dim + 1..].iter().product();
                    let out = self.runtime.alloc(bytes)?;
                    unsafe {
                        topk_f32::topk(
                            inp.as_ptr_typed::<f32>(),
                            out.as_ptr_typed::<f32>(),
                            std::ptr::null_mut(), // no indices in graph mode
                            outer,
                            dim_size,
                            inner,
                            *k,
                            *largest,
                            stream.raw(),
                        );
                    }
                    out
                }
            };

            slots[i] = Some(out_ptr);
        }

        stream.synchronize()?;

        let out_id = program.output.index();
        let out_shape = program.shapes[out_id].clone();
        let out_numel = program.numels[out_id];
        let out_bytes = out_numel * std::mem::size_of::<f32>();
        let mut host = vec![0.0f32; out_numel];
        let out_gpu = slots[out_id]
            .as_ref()
            .ok_or(LangError::InvalidNodeRef(out_id))?;

        unsafe {
            out_gpu.copy_to_host(host.as_mut_ptr() as *mut c_void, out_bytes)?;
        }

        Ok(HostTensor {
            shape: out_shape,
            data: host,
        })
    }
}

fn get_shape(shapes: &[Vec<usize>], id: ValueId) -> Result<Vec<usize>> {
    shapes
        .get(id.index())
        .cloned()
        .ok_or(LangError::InvalidNodeRef(id.index()))
}

fn validate_shape(shape: &[usize]) -> Result<()> {
    if shape.is_empty() || shape.iter().any(|&d| d == 0) {
        return Err(LangError::InvalidShape(shape.to_vec()));
    }
    Ok(())
}

fn numel(shape: &[usize]) -> usize {
    shape.iter().product()
}

pub use ferrite_apps as apps;
pub mod context;
pub mod cpu_tlsf;
pub mod tensor;
#[cfg(feature = "torch")]
pub use context::gpu_anyhow;
pub use context::{cpu, gpu, CpuCtx, GpuCtx};
pub use cpu_tlsf::CpuTlsfStats;
pub use tensor::{CpuTensor, GpuTensor, ToCpu, ToGpu};

#[cfg(feature = "torch")]
pub mod cv;

#[cfg(feature = "torch")]
pub mod dataflow;

#[cfg(feature = "torch")]
pub mod torch_bridge;
