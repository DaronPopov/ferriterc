use tch::Tensor;

/// Fixed-capacity ring buffer for passing GPU tensors between pipeline stages.
pub struct RingBuffer {
    slots: Vec<Option<Tensor>>,
    capacity: usize,
    write_idx: usize,
    read_idx: usize,
    count: usize,
}

impl RingBuffer {
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "RingBuffer capacity must be > 0");
        Self {
            slots: (0..capacity).map(|_| None).collect(),
            capacity,
            write_idx: 0,
            read_idx: 0,
            count: 0,
        }
    }

    pub fn push(&mut self, tensor: Tensor) -> bool {
        if self.count == self.capacity {
            return false;
        }
        self.slots[self.write_idx] = Some(tensor);
        self.write_idx = (self.write_idx + 1) % self.capacity;
        self.count += 1;
        true
    }

    pub fn pop(&mut self) -> Option<Tensor> {
        if self.count == 0 {
            return None;
        }
        let tensor = self.slots[self.read_idx].take();
        self.read_idx = (self.read_idx + 1) % self.capacity;
        self.count -= 1;
        tensor
    }

    pub fn peek(&self) -> Option<&Tensor> {
        if self.count == 0 {
            return None;
        }
        self.slots[self.read_idx].as_ref()
    }

    pub fn is_full(&self) -> bool {
        self.count == self.capacity
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    pub fn len(&self) -> usize {
        self.count
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn clear(&mut self) {
        for slot in &mut self.slots {
            *slot = None;
        }
        self.write_idx = 0;
        self.read_idx = 0;
        self.count = 0;
    }
}

/// Per-stage latency metrics collected during pipeline execution.
pub struct StageMetrics {
    pub name: &'static str,
    pub total_us: u128,
    pub invocations: u64,
    pub min_us: u128,
    pub max_us: u128,
}

impl StageMetrics {
    pub fn new(name: &'static str) -> Self {
        Self {
            name,
            total_us: 0,
            invocations: 0,
            min_us: u128::MAX,
            max_us: 0,
        }
    }

    pub fn record(&mut self, us: u128) {
        self.total_us += us;
        self.invocations += 1;
        self.min_us = self.min_us.min(us);
        self.max_us = self.max_us.max(us);
    }

    pub fn avg_us(&self) -> f64 {
        if self.invocations == 0 {
            0.0
        } else {
            self.total_us as f64 / self.invocations as f64
        }
    }
}

/// Aggregate pipeline statistics for a completed run.
pub struct PipelineStats {
    pub total_ticks: u64,
    pub total_time_ms: f64,
    pub warmup_ticks: u64,
    pub result_frames: u64,
    pub stages: Vec<StageMetrics>,
}

impl PipelineStats {
    pub fn throughput_fps(&self) -> f64 {
        if self.total_time_ms <= 0.0 {
            0.0
        } else {
            self.result_frames as f64 / (self.total_time_ms / 1000.0)
        }
    }
}
