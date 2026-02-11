use super::*;

impl PtxRuntime {
    /// Get a stream by ID.
    ///
    /// # Panics
    ///
    /// Panics if `id` is negative or >= num_streams() and stream is not found in pool.
    pub fn stream(&self, id: i32) -> Stream {
        // Doc contract says "panics if id is negative or >= num_streams()"
        assert!(id >= 0, "Stream ID must be non-negative");
        assert!(
            id < self.streams.len() as i32,
            "Stream ID {} exceeds pool size {}",
            id,
            self.streams.len()
        );

        self.streams.get(id as usize).unwrap_or_else(|| {
            // SAFETY: gpu_hot_get_stream is safe when:
            // - runtime pointer is valid (guaranteed by Arc)
            // - id is in valid range [0, max_streams)
            let raw = unsafe { ptx_sys::gpu_hot_get_stream(self.inner.raw, id) };
            Stream::new(raw, id)
        })
    }

    /// Get the next stream in round-robin order.
    pub fn next_stream(&self) -> Stream {
        self.streams.next()
    }

    /// Get the number of streams in the pool.
    pub fn num_streams(&self) -> usize {
        self.streams.len()
    }

    /// Get a stream by priority.
    pub fn priority_stream(&self, priority: StreamPriority) -> Stream {
        let raw = unsafe {
            ptx_sys::gpu_hot_get_priority_stream(self.inner.raw, priority as i32)
        };
        Stream::new(raw, priority as i32)
    }

    /// Get access to the stream pool.
    pub fn stream_pool(&self) -> &StreamPool {
        &self.streams
    }

    /// Synchronize all streams.
    pub fn sync_all(&self) {
        let _timer = crate::telemetry::OpTimer::new("sync_all");
        crate::telemetry::metrics().record_stream_sync();

        tracing::trace!("Synchronizing all streams");

        // SAFETY: Synchronization is safe when runtime pointer is valid
        unsafe { ptx_sys::gpu_hot_sync_all(self.inner.raw) }
    }

    // ========================================================================
    // CUDA Graphs
    // ========================================================================

    /// Begin capturing a CUDA graph.
    ///
    /// All operations on the returned stream will be captured until `end_capture` is called.
    pub fn begin_capture(&self, stream_id: i32, name: &str) -> Result<GraphCapture> {
        GraphCapture::begin(self.inner.raw, stream_id, name)
    }

    /// Launch a previously captured graph.
    pub fn launch_graph(&self, graph: &CudaGraph, stream: &Stream) -> Result<()> {
        graph.launch(stream)
    }

    // ========================================================================
    // cuBLAS
    // ========================================================================

    /// Get or create the cuBLAS handle.
    pub fn cublas(&self) -> Result<parking_lot::MutexGuard<'_, Option<CublasHandle>>> {
        let mut guard = self.cublas.lock();
        if guard.is_none() {
            *guard = Some(CublasHandle::new()?);
        }
        Ok(guard)
    }
}
