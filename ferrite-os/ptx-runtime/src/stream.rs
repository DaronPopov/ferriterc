//! CUDA stream management.

use std::sync::atomic::{AtomicUsize, Ordering};
use crate::error::{Error, Result};

/// A CUDA stream handle.
#[derive(Debug, Clone, Copy)]
pub struct Stream {
    raw: ptx_sys::cudaStream_t,
    id: i32,
}

impl Stream {
    /// Create a new stream wrapper from a raw handle.
    pub(crate) fn new(raw: ptx_sys::cudaStream_t, id: i32) -> Self {
        Self { raw, id }
    }

    /// Create a stream wrapper from a raw CUDA stream handle.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `raw` is a valid CUDA stream handle.
    pub unsafe fn from_raw(raw: ptx_sys::cudaStream_t) -> Self {
        Self { raw, id: -1 }
    }

    /// Get the raw CUDA stream handle.
    pub fn raw(&self) -> ptx_sys::cudaStream_t {
        self.raw
    }

    /// Get the stream ID.
    pub fn id(&self) -> i32 {
        self.id
    }

    /// Synchronize this stream (wait for all operations to complete).
    pub fn synchronize(&self) -> Result<()> {
        let err = unsafe { ptx_sys::cudaStreamSynchronize(self.raw) };
        Error::check_cuda(err)
    }

    /// Check if the stream is the null/default stream.
    pub fn is_default(&self) -> bool {
        self.raw.is_null()
    }
}

// Safety: CUDA streams can be used from multiple threads
unsafe impl Send for Stream {}
unsafe impl Sync for Stream {}

/// A pool of CUDA streams for round-robin allocation.
pub struct StreamPool {
    streams: Vec<Stream>,
    next: AtomicUsize,
}

impl StreamPool {
    /// Create a new stream pool with the given streams.
    pub fn new(streams: Vec<Stream>) -> Self {
        Self {
            streams,
            next: AtomicUsize::new(0),
        }
    }

    /// Get the next stream in round-robin order.
    ///
    /// # Panics
    ///
    /// Panics if the pool is empty. An empty pool is a fatal configuration error.
    pub fn next(&self) -> Stream {
        assert!(!self.streams.is_empty(), "StreamPool::next() called on empty pool");
        let idx = self.next.fetch_add(1, Ordering::Relaxed) % self.streams.len();
        self.streams[idx]
    }

    /// Get a specific stream by index.
    pub fn get(&self, index: usize) -> Option<Stream> {
        self.streams.get(index).copied()
    }

    /// Get the number of streams in the pool.
    pub fn len(&self) -> usize {
        self.streams.len()
    }

    /// Check if the pool is empty.
    pub fn is_empty(&self) -> bool {
        self.streams.is_empty()
    }

    /// Synchronize all streams in the pool.
    pub fn synchronize_all(&self) -> Result<()> {
        for stream in &self.streams {
            stream.synchronize()?;
        }
        Ok(())
    }

    /// Acquire the next stream on behalf of a tenant, checking the tenant's stream quota.
    ///
    /// This performs a quota check against the tenant's current active stream count
    /// and quota limit. If the quota allows, it returns the next round-robin stream.
    ///
    /// # Arguments
    ///
    /// * `tenant_active_streams` - Current number of streams held by the tenant.
    /// * `tenant_max_streams` - Maximum number of streams the tenant is allowed.
    ///
    /// # Errors
    ///
    /// Returns `Error::QuotaExceeded` if the tenant has reached its stream limit.
    pub fn acquire_for_tenant(
        &self,
        tenant_id: u64,
        tenant_active_streams: u64,
        tenant_max_streams: u64,
    ) -> Result<Stream> {
        if tenant_active_streams >= tenant_max_streams {
            return Err(Error::QuotaExceeded {
                tenant_id,
                resource: "streams".to_string(),
                limit: tenant_max_streams,
                current: tenant_active_streams,
            });
        }
        Ok(self.next())
    }
}

/// Global stream counter for assigning streams to operations.
static STREAM_COUNTER: AtomicUsize = AtomicUsize::new(0);

/// Get the next stream ID for round-robin stream selection.
pub fn next_stream_id() -> i32 {
    (STREAM_COUNTER.fetch_add(1, Ordering::Relaxed) % (ptx_sys::GPU_HOT_MAX_STREAMS as usize)) as i32
}

/// Priority levels for stream selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamPriority {
    /// Real-time priority (lowest latency)
    Realtime = 0,
    /// High priority
    High = 1,
    /// Normal priority (default)
    Normal = 2,
    /// Low priority (background tasks)
    Low = 3,
}

impl Default for StreamPriority {
    fn default() -> Self {
        Self::Normal
    }
}
