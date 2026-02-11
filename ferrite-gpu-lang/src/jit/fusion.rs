/// Elementwise fusion pass on the computation graph.
///
/// Detects patterns where a unary activation immediately follows a binary
/// operation and replaces the two nodes with a single fused node, eliminating
/// one intermediate VRAM allocation.
///
/// Supported patterns:
///   relu(add(a, b))  →  FusedReluAdd(a, b)
///   relu(mul(a, b))  →  FusedReluMul(a, b)
///   sigmoid(add(a, b)) → FusedSigmoidAdd(a, b)
///   tanh(add(a, b))  →  FusedTanhAdd(a, b)

use std::collections::HashSet;

use crate::{Node, Op, Program, ValueId};

/// Run fusion on a `Program`, returning a new `Program` with fused ops.
///
/// Preconditions (checked in debug builds):
/// - All ValueId references in the input program are in-bounds.
/// - The output (if set) references a valid node.
///
/// Postconditions (checked in debug builds):
/// - The fused program has the same number of nodes (dead placeholders
///   preserve index stability).
/// - The output ValueId still references a valid node.
/// - All ValueId references in fused nodes are in-bounds.
pub fn fuse(program: Program) -> Program {
    // ── precondition: input graph well-formed ────────────────────
    debug_assert!(
        verify_graph_refs(&program.nodes, program.output),
        "fusion precondition failed: input program has invalid ValueId references"
    );

    let nodes = program.nodes;
    let output = program.output;

    if nodes.len() < 2 {
        return Program { nodes, output };
    }

    // Build use-count: how many downstream nodes consume each ValueId.
    let mut use_count = vec![0u32; nodes.len()];
    for node in &nodes {
        for dep in op_deps(&node.op) {
            if dep.index() < use_count.len() {
                use_count[dep.index()] += 1;
            }
        }
    }
    // Output counts as a use.
    if let Some(out) = output {
        if out.index() < use_count.len() {
            use_count[out.index()] += 1;
        }
    }

    // Track which nodes have been absorbed into a fused node.
    let mut absorbed: HashSet<usize> = HashSet::new();

    // Build new node list, attempting fusion for each unary activation.
    let mut new_nodes: Vec<Node> = Vec::with_capacity(nodes.len());
    let mut remap: Vec<Option<ValueId>> = vec![None; nodes.len()];

    for (i, node) in nodes.iter().enumerate() {
        if absorbed.contains(&i) {
            // This node was fused into a prior node — emit a placeholder
            // that preserves index mapping (dead node, won't be executed
            // because nothing references it after remap).
            new_nodes.push(node.clone());
            remap[i] = Some(ValueId(new_nodes.len() - 1));
            continue;
        }

        // Try to fuse: unary(binary(...)) where binary has single use.
        if let Some((fused_op, binary_idx)) = try_fuse(i, &nodes, &use_count) {
            absorbed.insert(binary_idx);
            // Remap the binary op's inputs through any prior remaps.
            let remapped = remap_op(&fused_op, &remap);
            new_nodes.push(Node { op: remapped });
            remap[i] = Some(ValueId(new_nodes.len() - 1));
        } else {
            let remapped = remap_op(&node.op, &remap);
            new_nodes.push(Node { op: remapped });
            remap[i] = Some(ValueId(new_nodes.len() - 1));
        }
    }

    // Remap output.
    let new_output = output.map(|o| remap[o.index()].unwrap_or(o));

    let result = Program {
        nodes: new_nodes,
        output: new_output,
    };

    // ── postcondition: fused graph well-formed ──────────────────
    debug_assert_eq!(
        result.nodes.len(),
        nodes.len(),
        "fusion postcondition failed: node count changed ({} → {})",
        nodes.len(),
        result.nodes.len()
    );
    debug_assert!(
        verify_graph_refs(&result.nodes, result.output),
        "fusion postcondition failed: fused program has invalid ValueId references"
    );

    result
}

/// Verify that all ValueId references in the graph are in-bounds.
fn verify_graph_refs(nodes: &[Node], output: Option<ValueId>) -> bool {
    let n = nodes.len();
    for node in nodes {
        for dep in op_deps(&node.op) {
            if dep.index() >= n {
                return false;
            }
        }
    }
    if let Some(out) = output {
        if out.index() >= n {
            return false;
        }
    }
    true
}

/// Try to fuse node `i` (a unary activation) with its input (a binary op).
/// Returns the fused Op and the index of the absorbed binary node.
fn try_fuse(i: usize, nodes: &[Node], use_count: &[u32]) -> Option<(Op, usize)> {
    let node = &nodes[i];

    // Identify unary activations and their input.
    let (activation, input_id) = match &node.op {
        Op::Relu(x) => ("relu", *x),
        Op::Sigmoid(x) => ("sigmoid", *x),
        Op::Tanh(x) => ("tanh", *x),
        _ => return None,
    };

    let bin_idx = input_id.index();
    if bin_idx >= nodes.len() {
        return None;
    }

    // Only fuse if the binary op has exactly one consumer (this unary).
    if use_count[bin_idx] != 1 {
        return None;
    }

    let binary = &nodes[bin_idx];
    match (&activation, &binary.op) {
        (&"relu", Op::Add { lhs, rhs }) => Some((Op::FusedReluAdd { lhs: *lhs, rhs: *rhs }, bin_idx)),
        (&"relu", Op::Mul { lhs, rhs }) => Some((Op::FusedReluMul { lhs: *lhs, rhs: *rhs }, bin_idx)),
        (&"sigmoid", Op::Add { lhs, rhs }) => Some((Op::FusedSigmoidAdd { lhs: *lhs, rhs: *rhs }, bin_idx)),
        (&"tanh", Op::Add { lhs, rhs }) => Some((Op::FusedTanhAdd { lhs: *lhs, rhs: *rhs }, bin_idx)),
        _ => None,
    }
}

/// Extract all ValueId dependencies from an Op.
fn op_deps(op: &Op) -> Vec<ValueId> {
    match op {
        Op::Input { .. } => vec![],
        Op::Relu(x) | Op::Tanh(x) | Op::Sigmoid(x) => vec![*x],
        Op::Add { lhs, rhs } | Op::Mul { lhs, rhs } | Op::Sub { lhs, rhs } | Op::Div { lhs, rhs } => vec![*lhs, *rhs],
        Op::FusedReluAdd { lhs, rhs } | Op::FusedReluMul { lhs, rhs }
        | Op::FusedSigmoidAdd { lhs, rhs } | Op::FusedTanhAdd { lhs, rhs } => vec![*lhs, *rhs],
        Op::FillLike { like, .. } => vec![*like],
        Op::CumSum { input, .. } | Op::TopK { input, .. } => vec![*input],
    }
}

/// Remap ValueId references in an Op through the remap table.
fn remap_op(op: &Op, remap: &[Option<ValueId>]) -> Op {
    let r = |v: ValueId| -> ValueId {
        remap.get(v.index()).and_then(|m| *m).unwrap_or(v)
    };
    match op {
        Op::Input { shape } => Op::Input { shape: shape.clone() },
        Op::Relu(x) => Op::Relu(r(*x)),
        Op::Tanh(x) => Op::Tanh(r(*x)),
        Op::Sigmoid(x) => Op::Sigmoid(r(*x)),
        Op::Add { lhs, rhs } => Op::Add { lhs: r(*lhs), rhs: r(*rhs) },
        Op::Mul { lhs, rhs } => Op::Mul { lhs: r(*lhs), rhs: r(*rhs) },
        Op::Sub { lhs, rhs } => Op::Sub { lhs: r(*lhs), rhs: r(*rhs) },
        Op::Div { lhs, rhs } => Op::Div { lhs: r(*lhs), rhs: r(*rhs) },
        Op::FillLike { value, like } => Op::FillLike { value: *value, like: r(*like) },
        Op::CumSum { input, dim } => Op::CumSum { input: r(*input), dim: *dim },
        Op::TopK { input, k, dim, largest } => Op::TopK { input: r(*input), k: *k, dim: *dim, largest: *largest },
        Op::FusedReluAdd { lhs, rhs } => Op::FusedReluAdd { lhs: r(*lhs), rhs: r(*rhs) },
        Op::FusedReluMul { lhs, rhs } => Op::FusedReluMul { lhs: r(*lhs), rhs: r(*rhs) },
        Op::FusedSigmoidAdd { lhs, rhs } => Op::FusedSigmoidAdd { lhs: r(*lhs), rhs: r(*rhs) },
        Op::FusedTanhAdd { lhs, rhs } => Op::FusedTanhAdd { lhs: r(*lhs), rhs: r(*rhs) },
    }
}
