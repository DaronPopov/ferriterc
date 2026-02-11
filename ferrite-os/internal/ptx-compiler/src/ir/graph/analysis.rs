use super::*;

impl Graph {
    /// Get all tensor metadata.
    pub fn tensors(&self) -> &IndexMap<TensorId, TensorMeta> {
        &self.tensors
    }

    /// Get tensor metadata by ID.
    pub fn tensor(&self, id: TensorId) -> Option<&TensorMeta> {
        self.tensors.get(&id)
    }

    /// Get all nodes.
    pub fn nodes(&self) -> &IndexMap<NodeId, Node> {
        &self.nodes
    }

    /// Get a node by ID.
    pub fn node(&self, id: NodeId) -> Option<&Node> {
        self.nodes.get(&id)
    }

    /// Get input tensor IDs.
    pub fn inputs(&self) -> &[TensorId] {
        &self.inputs
    }

    /// Get output tensor IDs.
    pub fn outputs(&self) -> &[TensorId] {
        &self.outputs
    }

    /// Get the number of nodes.
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Get the number of tensors.
    pub fn num_tensors(&self) -> usize {
        self.tensors.len()
    }

    // ========================================================================
    // Graph analysis
    // ========================================================================

    /// Get nodes in topological order.
    pub fn topo_order(&self) -> Vec<NodeId> {
        // Simple topological sort based on insertion order
        // (since we add nodes in dependency order)
        self.nodes.keys().copied().collect()
    }

    /// Find which nodes produce tensors used by a given node.
    pub fn producers(&self, node_id: NodeId) -> Vec<NodeId> {
        let node = match self.nodes.get(&node_id) {
            Some(n) => n,
            None => return Vec::new(),
        };

        let mut producers = Vec::new();
        for input_id in &node.inputs {
            for (nid, n) in &self.nodes {
                if n.output == *input_id {
                    producers.push(*nid);
                }
            }
        }
        producers
    }

    /// Find which nodes consume a tensor.
    pub fn consumers(&self, tensor_id: TensorId) -> Vec<NodeId> {
        self.nodes
            .iter()
            .filter(|(_, n)| n.inputs.contains(&tensor_id))
            .map(|(id, _)| *id)
            .collect()
    }
}
