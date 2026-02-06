//! Backward pass engine for automatic differentiation.


use ptx_tensor::Tensor;
use ptx_runtime::Result;

use crate::tape::{Tape, TensorId, TAPE};

/// Run backward pass from a loss tensor.
///
/// This computes gradients for all tensors that require_grad and were involved
/// in computing the loss.
pub fn backward(loss_id: TensorId, loss_tensor: &Tensor) -> Result<()> {
    TAPE.with(|tape| {
        let mut tape = tape.lock();
        backward_impl(&mut tape, loss_id, loss_tensor)
    })
}

/// Internal backward implementation.
fn backward_impl(tape: &mut Tape, loss_id: TensorId, loss_tensor: &Tensor) -> Result<()> {
    // Initialize gradient of loss to ones (scalar 1 for scalar loss)
    let grad_loss = Tensor::ones(loss_tensor.shape(), loss_tensor.dtype(), loss_tensor.runtime())?;
    tape.set_grad(loss_id, grad_loss);

    // Collect node indices to process in reverse order
    let num_nodes = tape.nodes().len();

    // Process nodes in reverse order (topological sort in reverse)
    for i in (0..num_nodes).rev() {
        // Get gradient for this node's output
        let (output_id, inputs, saved_tensors) = {
            let node = &tape.nodes()[i];
            (node.output_id, node.inputs.clone(), node.saved_tensors.clone())
        };

        let grad_output = match tape.get_grad(output_id) {
            Some(grad) => grad.clone(),
            None => continue,  // No gradient to propagate
        };

        // Compute gradients for inputs
        let input_grads = tape.nodes()[i].grad_fn.backward(&grad_output, &saved_tensors)?;

        // Accumulate gradients to inputs
        for (input_id, maybe_grad) in inputs.iter().zip(input_grads.into_iter()) {
            if let Some(grad) = maybe_grad {
                tape.accumulate_grad(*input_id, grad)?;
            }
        }
    }

    Ok(())
}

/// Run backward pass and return gradients for specific tensors.
pub fn backward_with_gradients(
    loss_id: TensorId,
    loss_tensor: &Tensor,
    tensor_ids: &[TensorId],
) -> Result<Vec<Option<Tensor>>> {
    backward(loss_id, loss_tensor)?;

    let grads = TAPE.with(|tape| {
        let tape = tape.lock();
        tensor_ids
            .iter()
            .map(|id| tape.get_grad(*id).cloned())
            .collect()
    });

    Ok(grads)
}

/// Compute gradients and return them without modifying .grad attributes.
pub fn grad(
    outputs: &[(&TensorId, &Tensor)],
    inputs: &[TensorId],
    _grad_outputs: Option<Vec<Tensor>>,
    retain_graph: bool,
    create_graph: bool,
) -> Result<Vec<Option<Tensor>>> {
    if create_graph {
        return Err(ptx_runtime::Error::NotSupported {
            message: "create_graph not yet supported".to_string(),
        });
    }

    // For now, just run backward and extract gradients
    if outputs.len() != 1 {
        return Err(ptx_runtime::Error::NotSupported {
            message: "grad only supports single output for now".to_string(),
        });
    }

    let (loss_id, loss_tensor) = outputs[0];
    backward(*loss_id, loss_tensor)?;

    let grads = TAPE.with(|tape| {
        let mut tape = tape.lock();
        let result: Vec<Option<Tensor>> = inputs
            .iter()
            .map(|id| tape.get_grad(*id).cloned())
            .collect();

        if !retain_graph {
            tape.clear();
        }

        result
    });

    Ok(grads)
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_backward_simple() {
        // This would require a GPU, so skip in unit tests
    }
}
