//! CUDA graph backend.
//!
//! Compiles the IR graph to a CUDA graph for efficient replay.

use std::sync::Arc;
use std::collections::HashMap;

use ptx_runtime::{PtxRuntime, GpuPtr, Result, Error};
use ptx_runtime::stream::Stream;
use ptx_tensor::DType;

use crate::ir::{Graph, Node, OpCode, TensorMeta, TensorId};
use crate::passes::MemoryPlan;
use crate::backend::CompiledGraph;

/// Compile the graph to a CUDA graph.
pub fn compile(
    graph: Graph,
    memory_plan: MemoryPlan,
    runtime: &Arc<PtxRuntime>,
) -> Result<CompiledGraph> {
    // Collect input/output metadata
    let input_metas: Vec<TensorMeta> = graph.inputs()
        .iter()
        .filter_map(|id| graph.tensor(*id).cloned())
        .collect();
    let output_metas: Vec<TensorMeta> = graph.outputs()
        .iter()
        .filter_map(|id| graph.tensor(*id).cloned())
        .collect();

    // Allocate buffers for non-input tensors
    let mut buffers = Vec::with_capacity(memory_plan.buffers.len());
    for buffer in &memory_plan.buffers {
        let ptr = Arc::new(runtime.alloc(buffer.size)?);
        buffers.push(ptr);
    }

    // Allocate input buffers (one per input tensor)
    let mut input_buffers: Vec<Arc<GpuPtr>> = Vec::with_capacity(input_metas.len());
    let mut input_buffer_map: HashMap<TensorId, usize> = HashMap::new();
    for (idx, meta) in input_metas.iter().enumerate() {
        let ptr = Arc::new(runtime.alloc(meta.size_bytes())?);
        input_buffers.push(ptr);
        input_buffer_map.insert(meta.id, idx);
    }

    // Begin CUDA graph capture
    let stream_id = 0;
    let capture = runtime.begin_capture(stream_id, "compiled_graph")?;

    // Get the stream for emitting operations
    let stream = runtime.stream(stream_id);

    // Emit operations in topological order
    for node_id in graph.topo_order() {
        let node = match graph.node(node_id) {
            Some(n) => n,
            None => continue,
        };

        // Skip input nodes (constants are emitted if possible)
        if node.is_input() {
            continue;
        }

        emit_node(
            &graph,
            node,
            &memory_plan,
            &buffers,
            &input_buffers,
            &input_buffer_map,
            runtime,
            &stream,
        )?;
    }

    // End capture
    let cuda_graph = capture.end()?;

    Ok(CompiledGraph {
        graph_id: Some(cuda_graph.id()),
        memory_plan,
        buffers,
        input_buffers,
        input_ids: graph.inputs().to_vec(),
        input_metas,
        output_ids: graph.outputs().to_vec(),
        output_metas,
        runtime: Arc::clone(runtime),
    })
}

/// Emit a single node operation.
fn emit_node(
    graph: &Graph,
    node: &Node,
    memory_plan: &MemoryPlan,
    buffers: &[Arc<GpuPtr>],
    input_buffers: &[Arc<GpuPtr>],
    input_buffer_map: &HashMap<TensorId, usize>,
    runtime: &Arc<PtxRuntime>,
    stream: &Stream,
) -> Result<()> {
    // Get output buffer
    let output_meta = graph.tensor(node.output).ok_or_else(|| Error::Internal {
        message: "Missing output tensor metadata".to_string(),
    })?;

    let output_buffer_id = memory_plan.buffer_for(node.output).ok_or_else(|| Error::Internal {
        message: "No buffer assigned for output".to_string(),
    })?;

    let output_ptr = buffers[output_buffer_id].as_ptr();
    let elem_count = output_meta.elem_count();

    // Get input pointers
    let mut input_ptrs: Vec<*mut libc::c_void> = Vec::new();
    for &input_id in &node.inputs {
        let input_meta = graph.tensor(input_id).ok_or_else(|| Error::Internal {
            message: "Missing input tensor metadata".to_string(),
        })?;

        if input_meta.is_input {
            let idx = input_buffer_map.get(&input_id).ok_or_else(|| Error::Internal {
                message: "Missing input buffer mapping".to_string(),
            })?;
            input_ptrs.push(input_buffers[*idx].as_ptr());
        } else {
            let buffer_id = memory_plan.buffer_for(input_id).ok_or_else(|| Error::Internal {
                message: "No buffer assigned for input".to_string(),
            })?;
            input_ptrs.push(buffers[buffer_id].as_ptr());
        }
    }

    // Emit based on operation type and dtype
    if node.op == OpCode::Constant {
        if let Some(value) = output_meta.constant_value {
            if output_meta.dtype != DType::F32 {
                return Err(Error::NotSupported {
                    message: "Constant fill only supported for F32".to_string(),
                });
            }
            unsafe {
                ptx_sys::ptx_tensor_fill_f32(
                    output_ptr as *mut f32,
                    elem_count,
                    value,
                    stream.raw(),
                );
            }
            return Ok(());
        }
        return Err(Error::NotSupported {
            message: "Constant nodes without value are not supported in compiler".to_string(),
        });
    }

    match output_meta.dtype {
        DType::F32 => emit_node_f32(node, &input_ptrs, output_ptr, elem_count, graph, runtime, stream)?,
        dtype => return Err(Error::NotSupported {
            message: format!("Compilation not supported for dtype {:?}", dtype),
        }),
    }

    Ok(())
}

/// Emit F32 operation.
fn emit_node_f32(
    node: &Node,
    inputs: &[*mut libc::c_void],
    output: *mut libc::c_void,
    n: usize,
    graph: &Graph,
    runtime: &Arc<PtxRuntime>,
    stream: &Stream,
) -> Result<()> {
    let out = output as *mut f32;

    match node.op {
        // Binary ops
        OpCode::Add => {
            let a = inputs[0] as *mut f32;
            let b = inputs[1] as *mut f32;
            unsafe { ptx_sys::ptx_tensor_add_f32(a, b, out, n, stream.raw()) };
        }
        OpCode::Sub => {
            let a = inputs[0] as *mut f32;
            let b = inputs[1] as *mut f32;
            unsafe { ptx_sys::ptx_tensor_sub_f32(a, b, out, n, stream.raw()) };
        }
        OpCode::Mul => {
            let a = inputs[0] as *mut f32;
            let b = inputs[1] as *mut f32;
            unsafe { ptx_sys::ptx_tensor_mul_f32(a, b, out, n, stream.raw()) };
        }
        OpCode::Div => {
            let a = inputs[0] as *mut f32;
            let b = inputs[1] as *mut f32;
            unsafe { ptx_sys::ptx_tensor_div_f32(a, b, out, n, stream.raw()) };
        }
        OpCode::Max => {
            let a = inputs[0] as *mut f32;
            let b = inputs[1] as *mut f32;
            unsafe { ptx_sys::ptx_tensor_max_f32(a, b, out, n, stream.raw()) };
        }
        OpCode::Min => {
            let a = inputs[0] as *mut f32;
            let b = inputs[1] as *mut f32;
            unsafe { ptx_sys::ptx_tensor_min_f32(a, b, out, n, stream.raw()) };
        }

        // Unary ops
        OpCode::Neg => {
            let a = inputs[0] as *mut f32;
            unsafe { ptx_sys::ptx_tensor_neg_f32(a, out, n, stream.raw()) };
        }
        OpCode::Abs => {
            let a = inputs[0] as *mut f32;
            unsafe { ptx_sys::ptx_tensor_abs_f32(a, out, n, stream.raw()) };
        }
        OpCode::Exp => {
            let a = inputs[0] as *mut f32;
            unsafe { ptx_sys::ptx_tensor_exp_f32(a, out, n, stream.raw()) };
        }
        OpCode::Log => {
            let a = inputs[0] as *mut f32;
            unsafe { ptx_sys::ptx_tensor_log_f32(a, out, n, stream.raw()) };
        }
        OpCode::Sqrt => {
            let a = inputs[0] as *mut f32;
            unsafe { ptx_sys::ptx_tensor_sqrt_f32(a, out, n, stream.raw()) };
        }
        OpCode::Rsqrt => {
            let a = inputs[0] as *mut f32;
            unsafe { ptx_sys::ptx_tensor_rsqrt_f32(a, out, n, stream.raw()) };
        }
        OpCode::Sin => {
            let a = inputs[0] as *mut f32;
            unsafe { ptx_sys::ptx_tensor_sin_f32(a, out, n, stream.raw()) };
        }
        OpCode::Cos => {
            let a = inputs[0] as *mut f32;
            unsafe { ptx_sys::ptx_tensor_cos_f32(a, out, n, stream.raw()) };
        }
        OpCode::Tanh => {
            let a = inputs[0] as *mut f32;
            unsafe { ptx_sys::ptx_tensor_tanh_f32(a, out, n, stream.raw()) };
        }
        OpCode::Ceil => {
            let a = inputs[0] as *mut f32;
            unsafe { ptx_sys::ptx_tensor_ceil_f32(a, out, n, stream.raw()) };
        }
        OpCode::Floor => {
            let a = inputs[0] as *mut f32;
            unsafe { ptx_sys::ptx_tensor_floor_f32(a, out, n, stream.raw()) };
        }
        OpCode::Round => {
            let a = inputs[0] as *mut f32;
            unsafe { ptx_sys::ptx_tensor_round_f32(a, out, n, stream.raw()) };
        }
        OpCode::Sqr => {
            let a = inputs[0] as *mut f32;
            unsafe { ptx_sys::ptx_tensor_sqr_f32(a, out, n, stream.raw()) };
        }
        OpCode::Recip => {
            let a = inputs[0] as *mut f32;
            unsafe { ptx_sys::ptx_tensor_recip_f32(a, out, n, stream.raw()) };
        }
        OpCode::Copy => {
            let a = inputs[0] as *mut f32;
            unsafe { ptx_sys::ptx_tensor_copy_f32(a, out, n, stream.raw()) };
        }

        // Activations
        OpCode::Relu => {
            let a = inputs[0] as *mut f32;
            unsafe { ptx_sys::ptx_tensor_relu_f32(a, out, n, stream.raw()) };
        }
        OpCode::Relu6 => {
            let a = inputs[0] as *mut f32;
            unsafe { ptx_sys::ptx_tensor_relu6_f32(a, out, n, stream.raw()) };
        }
        OpCode::LeakyRelu => {
            let a = inputs[0] as *mut f32;
            let alpha = node.attrs.scalar_a.ok_or_else(|| Error::Internal {
                message: "LeakyRelu missing alpha".to_string(),
            })?;
            unsafe { ptx_sys::ptx_tensor_leaky_relu_f32(a, out, n, alpha, stream.raw()) };
        }
        OpCode::Elu => {
            let a = inputs[0] as *mut f32;
            let alpha = node.attrs.scalar_a.ok_or_else(|| Error::Internal {
                message: "Elu missing alpha".to_string(),
            })?;
            unsafe { ptx_sys::ptx_tensor_elu_f32(a, out, n, alpha, stream.raw()) };
        }
        OpCode::Selu => {
            let a = inputs[0] as *mut f32;
            unsafe { ptx_sys::ptx_tensor_selu_f32(a, out, n, stream.raw()) };
        }
        OpCode::Gelu => {
            let a = inputs[0] as *mut f32;
            unsafe { ptx_sys::ptx_tensor_gelu_f32(a, out, n, stream.raw()) };
        }
        OpCode::Sigmoid => {
            let a = inputs[0] as *mut f32;
            unsafe { ptx_sys::ptx_tensor_sigmoid_f32(a, out, n, stream.raw()) };
        }
        OpCode::Silu => {
            let a = inputs[0] as *mut f32;
            unsafe { ptx_sys::ptx_tensor_silu_f32(a, out, n, stream.raw()) };
        }
        OpCode::Softplus => {
            let a = inputs[0] as *mut f32;
            unsafe { ptx_sys::ptx_tensor_softplus_f32(a, out, n, stream.raw()) };
        }
        OpCode::Mish => {
            let a = inputs[0] as *mut f32;
            unsafe { ptx_sys::ptx_tensor_mish_f32(a, out, n, stream.raw()) };
        }

        // Reductions need special handling for dimensions
        OpCode::ReduceSum | OpCode::ReduceMean | OpCode::ReduceMax | OpCode::ReduceMin => {
            let input_id = node.inputs.get(0).ok_or_else(|| Error::Internal {
                message: "Reduction missing input".to_string(),
            })?;
            let input_meta = graph.tensor(*input_id).ok_or_else(|| Error::Internal {
                message: "Missing reduction input meta".to_string(),
            })?;
            let dim = node.attrs.reduce_dim.ok_or_else(|| Error::Internal {
                message: "Reduction missing dim".to_string(),
            })?;
            let ndim = input_meta.shape.len() as i32;
            let dim = if dim < 0 { ndim + dim } else { dim } as usize;
            let (outer, reduce, inner) = ptx_tensor::shape::reduction_sizes(&input_meta.shape, dim);

            let a = inputs[0] as *mut f32;
            match node.op {
                OpCode::ReduceSum => unsafe {
                    ptx_sys::ptx_tensor_reduce_sum_f32(a, out, outer, reduce, inner, stream.raw())
                },
                OpCode::ReduceMean => unsafe {
                    ptx_sys::ptx_tensor_reduce_mean_f32(a, out, outer, reduce, inner, stream.raw())
                },
                OpCode::ReduceMax => unsafe {
                    ptx_sys::ptx_tensor_reduce_max_f32(a, out, outer, reduce, inner, stream.raw())
                },
                OpCode::ReduceMin => unsafe {
                    ptx_sys::ptx_tensor_reduce_min_f32(a, out, outer, reduce, inner, stream.raw())
                },
                _ => {}
            }
        }

        // Softmax
        OpCode::Softmax => {
            let input_id = node.inputs.get(0).ok_or_else(|| Error::Internal {
                message: "Softmax missing input".to_string(),
            })?;
            let input_meta = graph.tensor(*input_id).ok_or_else(|| Error::Internal {
                message: "Missing softmax input meta".to_string(),
            })?;
            let dim = node.attrs.reduce_dim.ok_or_else(|| Error::Internal {
                message: "Softmax missing dim".to_string(),
            })?;
            let ndim = input_meta.shape.len() as i32;
            let dim = if dim < 0 { ndim + dim } else { dim } as usize;
            if dim != input_meta.shape.len() - 1 {
                return Err(Error::NotSupported {
                    message: "Softmax only supported along last dimension".to_string(),
                });
            }
            let batch: usize = input_meta.shape[..dim].iter().product();
            let softmax_dim = input_meta.shape[dim];
            let a = inputs[0] as *mut f32;
            unsafe { ptx_sys::ptx_tensor_softmax_f32(a, out, batch, softmax_dim, stream.raw()) };
        }

        OpCode::LogSoftmax => {
            let input_id = node.inputs.get(0).ok_or_else(|| Error::Internal {
                message: "LogSoftmax missing input".to_string(),
            })?;
            let input_meta = graph.tensor(*input_id).ok_or_else(|| Error::Internal {
                message: "Missing log_softmax input meta".to_string(),
            })?;
            let dim = node.attrs.reduce_dim.ok_or_else(|| Error::Internal {
                message: "LogSoftmax missing dim".to_string(),
            })?;
            let ndim = input_meta.shape.len() as i32;
            let dim = if dim < 0 { ndim + dim } else { dim } as usize;
            if dim != input_meta.shape.len() - 1 {
                return Err(Error::NotSupported {
                    message: "Log softmax only supported along last dimension".to_string(),
                });
            }
            let batch: usize = input_meta.shape[..dim].iter().product();
            let softmax_dim = input_meta.shape[dim];
            let a = inputs[0] as *mut f32;
            unsafe {
                ptx_sys::ptx_tensor_log_softmax_f32(a, out, batch, softmax_dim, stream.raw())
            };
        }

        // Scalar/transform ops
        OpCode::Affine => {
            let a = inputs[0] as *mut f32;
            let mul = node.attrs.scalar_a.ok_or_else(|| Error::Internal {
                message: "Affine missing mul".to_string(),
            })?;
            let add = node.attrs.scalar_b.ok_or_else(|| Error::Internal {
                message: "Affine missing add".to_string(),
            })?;
            unsafe { ptx_sys::ptx_tensor_affine_f32(a, out, n, mul, add, stream.raw()) };
        }
        OpCode::Clamp => {
            let a = inputs[0] as *mut f32;
            let min_val = node.attrs.scalar_a.ok_or_else(|| Error::Internal {
                message: "Clamp missing min".to_string(),
            })?;
            let max_val = node.attrs.scalar_b.ok_or_else(|| Error::Internal {
                message: "Clamp missing max".to_string(),
            })?;
            unsafe { ptx_sys::ptx_tensor_clamp_f32(a, out, n, min_val, max_val, stream.raw()) };
        }
        OpCode::Pow => {
            let a = inputs[0] as *mut f32;
            let exp = node.attrs.scalar_a.ok_or_else(|| Error::Internal {
                message: "Pow missing exponent".to_string(),
            })?;
            unsafe { ptx_sys::ptx_tensor_powf_f32(a, out, n, exp, stream.raw()) };
        }
        OpCode::Fill => {
            let value = node.attrs.scalar_a.ok_or_else(|| Error::Internal {
                message: "Fill missing value".to_string(),
            })?;
            unsafe { ptx_sys::ptx_tensor_fill_f32(out, n, value, stream.raw()) };
        }

        // Matmul
        OpCode::Matmul => {
            let a_id = node.inputs.get(0).ok_or_else(|| Error::Internal {
                message: "Matmul missing A".to_string(),
            })?;
            let b_id = node.inputs.get(1).ok_or_else(|| Error::Internal {
                message: "Matmul missing B".to_string(),
            })?;
            let a_meta = graph.tensor(*a_id).ok_or_else(|| Error::Internal {
                message: "Missing matmul A meta".to_string(),
            })?;
            let b_meta = graph.tensor(*b_id).ok_or_else(|| Error::Internal {
                message: "Missing matmul B meta".to_string(),
            })?;
            if a_meta.shape.len() != 2 || b_meta.shape.len() != 2 {
                return Err(Error::NotSupported {
                    message: "Matmul requires 2D tensors".to_string(),
                });
            }
            let m = a_meta.shape[0];
            let k = a_meta.shape[1];
            let k2 = b_meta.shape[0];
            let ncol = b_meta.shape[1];
            if k != k2 {
                return Err(Error::ShapeMismatch {
                    expected: vec![k],
                    actual: vec![k2],
                });
            }
            let a = inputs[0] as *const f32;
            let b = inputs[1] as *const f32;

            let handle = runtime.cublas()?;
            if let Some(h) = handle.as_ref() {
                h.set_stream(stream)?;
                unsafe {
                    h.sgemm(
                        ptx_runtime::cublas::GemmOp::None,
                        ptx_runtime::cublas::GemmOp::None,
                        ncol as i32,
                        m as i32,
                        k as i32,
                        1.0,
                        b,
                        ncol as i32,
                        a,
                        k as i32,
                        0.0,
                        out,
                        ncol as i32,
                    )?;
                }
            }
        }
        OpCode::BatchMatmul => {
            let a_id = node.inputs.get(0).ok_or_else(|| Error::Internal {
                message: "BMM missing A".to_string(),
            })?;
            let b_id = node.inputs.get(1).ok_or_else(|| Error::Internal {
                message: "BMM missing B".to_string(),
            })?;
            let a_meta = graph.tensor(*a_id).ok_or_else(|| Error::Internal {
                message: "Missing bmm A meta".to_string(),
            })?;
            let b_meta = graph.tensor(*b_id).ok_or_else(|| Error::Internal {
                message: "Missing bmm B meta".to_string(),
            })?;
            if a_meta.shape.len() != 3 || b_meta.shape.len() != 3 {
                return Err(Error::NotSupported {
                    message: "BatchMatmul requires 3D tensors".to_string(),
                });
            }
            let batch = a_meta.shape[0];
            let m = a_meta.shape[1];
            let k = a_meta.shape[2];
            let batch2 = b_meta.shape[0];
            let k2 = b_meta.shape[1];
            let ncol = b_meta.shape[2];
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
            let a = inputs[0] as *const f32;
            let b = inputs[1] as *const f32;

            let handle = runtime.cublas()?;
            if let Some(h) = handle.as_ref() {
                h.set_stream(stream)?;
                unsafe {
                    h.sgemm_strided_batched(
                        ptx_runtime::cublas::GemmOp::None,
                        ptx_runtime::cublas::GemmOp::None,
                        ncol as i32,
                        m as i32,
                        k as i32,
                        1.0,
                        b,
                        ncol as i32,
                        (k * ncol) as i64,
                        a,
                        k as i32,
                        (m * k) as i64,
                        0.0,
                        out,
                        ncol as i32,
                        (m * ncol) as i64,
                        batch as i32,
                    )?;
                }
            }
        }

        op => {
            return Err(Error::NotSupported {
                message: format!("Operation {:?} not yet supported in compiler", op),
            });
        }
    }

    Ok(())
}
