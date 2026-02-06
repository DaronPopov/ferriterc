//! PTX-OS runtime wrapper with RAII semantics.

use std::sync::Arc;

use parking_lot::Mutex;

use crate::device::Device;
use crate::memory::{GpuPtr, PtxRuntimeInner};
use crate::stream::{Stream, StreamPool, StreamPriority};
use crate::graph::{CudaGraph, GraphCapture};
use crate::cublas::CublasHandle;
use crate::error::{Error, Result};

/// The main PTX-OS runtime handle.
///
/// This is the primary interface for interacting with the GPU through PTX-OS.
/// It manages memory allocation, streams, and CUDA graphs.
///
/// # Thread Safety
///
/// The runtime is thread-safe and can be shared between threads using `Arc<PtxRuntime>`.
///
/// # Example
///
/// ```no_run
/// use ptx_runtime::PtxRuntime;
///
/// let runtime = PtxRuntime::new(0).expect("Failed to init");
/// let ptr = runtime.alloc(1024).expect("Failed to alloc");
/// runtime.sync_all();
/// ```
pub struct PtxRuntime {
    inner: Arc<PtxRuntimeInner>,
    device: Device,
    streams: StreamPool,
    cublas: Mutex<Option<CublasHandle>>,
}

impl PtxRuntime {
    /// Initialize the PTX-OS runtime on the given device.
    ///
    /// Uses default configuration (all available VRAM minus headroom).
    pub fn new(device_id: i32) -> Result<Self> {
        tracing::info!(device_id, "Initializing PTX-OS runtime");
        Self::with_config(device_id, None)
    }

    /// Initialize the runtime with custom configuration.
    pub fn with_config(device_id: i32, config: Option<ptx_sys::GPUHotConfig>) -> Result<Self> {
        let cfg = config.unwrap_or_default();
        let stable_cfg = ptx_sys::PTXStableConfig {
            struct_size: std::mem::size_of::<ptx_sys::PTXStableConfig>() as u32,
            abi_version: ptx_sys::PTX_STABLE_ABI_VERSION,
            flags: ptx_sys::PTX_STABLE_CONFIG_DEFAULT,
            device_id,
            pool_fraction: cfg.pool_fraction,
            fixed_pool_size: cfg.fixed_pool_size as u64,
            reserve_vram: cfg.reserve_vram as u64,
            max_streams: cfg.max_streams,
            quiet_init: u8::from(cfg.quiet_init),
            enable_leak_detection: u8::from(cfg.enable_leak_detection),
            enable_pool_health: u8::from(cfg.enable_pool_health),
            _reserved0: 0,
        };
        Self::with_stable_config(device_id, Some(stable_cfg))
    }

    /// Initialize the runtime with the versioned stable ABI config.
    pub fn with_stable_config(device_id: i32, config: Option<ptx_sys::PTXStableConfig>) -> Result<Self> {
        let _timer = crate::telemetry::OpTimer::new("runtime_init");
        let mut stable_cfg = config.unwrap_or_default();
        stable_cfg.device_id = device_id;

        tracing::debug!(
            device_id,
            pool_fraction = stable_cfg.pool_fraction,
            max_streams = stable_cfg.max_streams,
            abi_version = stable_cfg.abi_version,
            "Initializing with stable config"
        );

        let mut stable_raw: *mut ptx_sys::PTXStableRuntime = std::ptr::null_mut();
        let init_status = unsafe { ptx_sys::ptx_stable_init(&stable_cfg, &mut stable_raw) };
        if init_status != ptx_sys::PTXStableStatus::Ok || stable_raw.is_null() {
            tracing::error!(device_id, status = init_status as i32, "Stable runtime initialization failed");
            return Err(Error::InitFailed { device_id });
        }

        let mut raw_void: *mut libc::c_void = std::ptr::null_mut();
        let raw_status = unsafe { ptx_sys::ptx_stable_get_hot_runtime(stable_raw, &mut raw_void) };
        if raw_status != ptx_sys::PTXStableStatus::Ok || raw_void.is_null() {
            let _ = unsafe { ptx_sys::ptx_stable_release(stable_raw) };
            return Err(Error::stable(raw_status));
        }
        let raw = raw_void as *mut ptx_sys::GPUHotRuntime;

        let mut runtime_stats = ptx_sys::GPUHotStats::default();
        unsafe { ptx_sys::gpu_hot_get_stats(raw, &mut runtime_stats) };
        let num_streams = if runtime_stats.active_streams > 0 {
            runtime_stats.active_streams as u32
        } else if stable_cfg.max_streams > 0 {
            stable_cfg.max_streams
        } else {
            1
        };

        tracing::info!(device_id, num_streams, "Runtime initialized successfully");

        let inner = Arc::new(PtxRuntimeInner { stable: stable_raw, raw });

        // Create stream pool with configured number of streams
        let streams: Vec<Stream> = (0..num_streams as i32)
            .map(|id| {
                let stream = unsafe { ptx_sys::gpu_hot_get_stream(raw, id) };
                Stream::new(stream, id)
            })
            .collect();

        Ok(Self {
            inner,
            device: Device::new(device_id),
            streams: StreamPool::new(streams),
            cublas: Mutex::new(None),
        })
    }

    /// Get the device this runtime is associated with.
    pub fn device(&self) -> Device {
        self.device
    }

    /// Get a reference to the inner runtime (for creating GpuPtrs).
    #[allow(dead_code)]
    pub(crate) fn inner(&self) -> Arc<PtxRuntimeInner> {
        Arc::clone(&self.inner)
    }

    /// Get the raw runtime pointer.
    ///
    /// # Safety
    ///
    /// This pointer must not outlive the PtxRuntime instance.
    /// Callers must ensure they don't use the pointer after the runtime is dropped.
    pub fn raw(&self) -> *mut ptx_sys::GPUHotRuntime {
        self.inner.raw
    }

    /// Enable CUDA allocation hooks to route all cudaMalloc calls through TLSF.
    ///
    /// This enables interception of CUDA allocations made by external libraries
    /// (like Candle) so they use the TLSF allocator instead of cudaMalloc.
    ///
    /// Call this immediately after runtime initialization if you want to marshal
    /// all CUDA allocations through TLSF.
    pub fn enable_hooks(&self, verbose: bool) {
        unsafe {
            ptx_sys::ptx_hook_init(self.inner.raw, verbose);
        }
        tracing::info!("CUDA allocation hooks enabled");
    }

    // ========================================================================
    // Memory Allocation
    // ========================================================================

    /// Allocate GPU memory.
    ///
    /// Returns a smart pointer that automatically frees the memory when dropped.
    pub fn alloc(&self, size: usize) -> Result<GpuPtr> {
        let _timer = crate::telemetry::OpTimer::new("gpu_alloc");

        let mut ptr: *mut libc::c_void = std::ptr::null_mut();
        let status = unsafe { ptx_sys::ptx_stable_alloc(self.inner.stable, size, &mut ptr) };
        if status != ptx_sys::PTXStableStatus::Ok {
            crate::telemetry::metrics().record_allocation_failure();
            tracing::warn!(size, status = status as i32, "Stable allocation failed");
            return Err(Error::stable(status));
        }

        match unsafe { GpuPtr::new(ptr, size, Arc::clone(&self.inner)) } {
            Ok(gpu_ptr) => {
                crate::telemetry::metrics().record_allocation(size);
                tracing::trace!(size, ptr = ?ptr, "GPU allocation successful");
                Ok(gpu_ptr)
            }
            Err(e) => {
                crate::telemetry::metrics().record_allocation_failure();
                tracing::warn!(size, error = ?e, "GPU allocation failed");
                Err(e)
            }
        }
    }

    /// Free GPU memory that was allocated by this runtime.
    pub fn free(&self, ptr: GpuPtr) -> Result<()> {
        drop(ptr);
        Ok(())
    }

    /// Allocate GPU memory (async, stream-ordered).
    ///
    /// The allocation itself is immediate, but the returned pointer should be
    /// used and freed on the same stream to preserve ordering guarantees.
    pub fn alloc_async(&self, size: usize, stream: &Stream) -> Result<*mut libc::c_void> {
        let ptr = unsafe { ptx_sys::gpu_hot_alloc_async(self.inner.raw, size, stream.raw()) };
        if ptr.is_null() {
            Err(Error::AllocationFailed { size })
        } else {
            Ok(ptr)
        }
    }

    /// Free GPU memory asynchronously on a given stream.
    ///
    /// This defers the actual free until all prior work on the stream completes.
    ///
    /// # Safety
    ///
    /// Caller must ensure:
    /// - `ptr` was allocated by this runtime (via alloc_async)
    /// - `ptr` is not used after this call (no use-after-free)
    /// - `ptr` is freed exactly once (no double-free)
    /// - All GPU operations using `ptr` on other streams have completed
    ///
    /// # Defensive Checks
    ///
    /// In debug builds, validates that pointer is owned by TLSF allocator.
    pub unsafe fn free_async(&self, ptr: *mut libc::c_void, stream: &Stream) {
        // SAFETY: Defensive validation in debug builds
        debug_assert!(
            !ptr.is_null(),
            "Attempted to free null pointer"
        );
        debug_assert!(
            self.owns_ptr(ptr),
            "Attempted to free pointer not owned by this runtime: {:?}",
            ptr
        );

        // SAFETY: Caller must uphold documented preconditions
        ptx_sys::gpu_hot_free_async(self.inner.raw, ptr, stream.raw())
    }

    /// Poll and release completed deferred frees.
    ///
    /// `max_drain` of 0 drains all completed events.
    pub fn poll_deferred(&self, max_drain: i32) {
        unsafe { ptx_sys::gpu_hot_poll_deferred(self.inner.raw, max_drain) }
    }

    /// Check if a size can be allocated.
    pub fn can_allocate(&self, size: usize) -> bool {
        unsafe { ptx_sys::gpu_hot_can_allocate(self.inner.raw, size) }
    }

    /// Get the maximum size that can be allocated.
    pub fn max_allocatable(&self) -> usize {
        unsafe { ptx_sys::gpu_hot_get_max_allocatable(self.inner.raw) }
    }

    /// Check if a pointer belongs to this runtime.
    pub fn owns_ptr(&self, ptr: *mut libc::c_void) -> bool {
        let mut owned = false;
        let st = unsafe { ptx_sys::ptx_stable_owns_ptr(self.inner.stable, ptr, &mut owned) };
        st == ptx_sys::PTXStableStatus::Ok && owned
    }

    // ========================================================================
    // Streams
    // ========================================================================

    /// Get a stream by ID.
    ///
    /// # Panics
    ///
    /// Panics if `id` is negative or >= num_streams() and stream is not found in pool.
    pub fn stream(&self, id: i32) -> Stream {
        // SAFETY: Defensive check - id should be in valid range
        debug_assert!(id >= 0, "Stream ID must be non-negative");
        debug_assert!(
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

    // ========================================================================
    // Statistics
    // ========================================================================

    /// Get runtime statistics.
    pub fn stats(&self) -> ptx_sys::GPUHotStats {
        let mut stable = ptx_sys::PTXStableStats::default();
        let st = unsafe { ptx_sys::ptx_stable_get_stats(self.inner.stable, &mut stable) };
        if st != ptx_sys::PTXStableStatus::Ok {
            let mut fallback = ptx_sys::GPUHotStats::default();
            unsafe { ptx_sys::gpu_hot_get_stats(self.inner.raw, &mut fallback) };
            return fallback;
        }

        ptx_sys::GPUHotStats {
            vram_allocated: stable.vram_allocated as usize,
            vram_used: stable.vram_used as usize,
            vram_free: stable.vram_free as usize,
            gpu_utilization: stable.gpu_utilization,
            active_streams: stable.active_streams as i32,
            registered_kernels: 0,
            shm_count: 0,
            total_ops: stable.total_ops,
            avg_latency_us: 0.0,
            watchdog_tripped: stable.watchdog_tripped != 0,
        }
    }

    /// Get TLSF pool statistics.
    pub fn tlsf_stats(&self) -> ptx_sys::TLSFPoolStats {
        let mut stats = ptx_sys::TLSFPoolStats::default();
        unsafe { ptx_sys::gpu_hot_get_tlsf_stats(self.inner.raw, &mut stats) };
        stats
    }

    /// Validate the TLSF pool and get a health report.
    pub fn validate_pool(&self) -> ptx_sys::TLSFHealthReport {
        let mut report = ptx_sys::TLSFHealthReport::default();
        unsafe { ptx_sys::gpu_hot_validate_tlsf_pool(self.inner.raw, &mut report) };
        report
    }

    /// Print the pool memory map (for debugging).
    pub fn print_pool_map(&self) {
        unsafe { ptx_sys::gpu_hot_print_pool_map(self.inner.raw) }
    }

    /// Export the runtime pointer for LD_PRELOAD hook integration.
    ///
    /// Sets PTX_RUNTIME_PTR environment variable so the CUDA intercept hook
    /// can reuse this runtime instead of creating a competing one.
    ///
    /// **Call this before any CUDA operations that should go through TLSF.**
    pub fn export_for_hook(&self) {
        let mut raw_void: *mut libc::c_void = std::ptr::null_mut();
        let st = unsafe { ptx_sys::ptx_stable_get_hot_runtime(self.inner.stable, &mut raw_void) };
        if st != ptx_sys::PTXStableStatus::Ok || raw_void.is_null() {
            tracing::warn!(status = st as i32, "Failed to export runtime pointer for hook");
            return;
        }
        let ptr_str = format!("{:x}", raw_void as usize);
        std::env::set_var("PTX_RUNTIME_PTR", ptr_str);
        tracing::info!(ptr = ?self.inner.raw, "Runtime pointer exported for LD_PRELOAD hook");
    }

    // ========================================================================
    // Context Management
    // ========================================================================

    /// Get the raw CUcontext pointer captured during runtime initialization.
    ///
    /// Returns `None` if no context was captured (e.g., driver API unavailable).
    pub fn context(&self) -> Option<ptx_sys::CUcontext> {
        let mut ctx: *mut libc::c_void = std::ptr::null_mut();
        let st = unsafe { ptx_sys::ptx_stable_get_context(self.inner.stable, &mut ctx) };
        if st != ptx_sys::PTXStableStatus::Ok {
            return None;
        }
        if ctx.is_null() { None } else { Some(ctx) }
    }

    /// Export the CUcontext pointer as `PTX_CONTEXT_PTR` environment variable.
    ///
    /// External consumers (e.g., cudarc) can read this to share the same context.
    pub fn export_context(&self) {
        unsafe { ptx_sys::gpu_hot_export_context(self.inner.raw) };
        tracing::info!("CUcontext exported via PTX_CONTEXT_PTR");
    }

    /// Get context hook statistics, if the hook is loaded.
    ///
    /// Returns `None` if the context hook library is not present (weak symbol).
    pub fn context_stats(&self) -> Option<ptx_sys::PTXContextStats> {
        let mut stats = ptx_sys::PTXContextStats::default();
        unsafe { ptx_sys::ptx_context_hook_get_stats(&mut stats) };
        Some(stats)
    }

    // ========================================================================
    // Pool Management
    // ========================================================================

    /// Run defragmentation on the memory pool.
    pub fn defragment(&self) {
        unsafe { ptx_sys::gpu_hot_defragment_pool(self.inner.raw) }
    }

    /// Set the warning threshold for pool utilization.
    pub fn set_warning_threshold(&self, threshold: f32) {
        unsafe { ptx_sys::gpu_hot_set_warning_threshold(self.inner.raw, threshold) }
    }

    /// Enable or disable automatic defragmentation.
    pub fn set_auto_defrag(&self, enable: bool) {
        unsafe { ptx_sys::gpu_hot_set_auto_defrag(self.inner.raw, enable) }
    }

    // ========================================================================
    // Watchdog
    // ========================================================================

    /// Set the watchdog timeout.
    pub fn set_watchdog(&self, timeout_ms: i32) {
        unsafe { ptx_sys::gpu_hot_set_watchdog(self.inner.raw, timeout_ms) }
    }

    /// Check if the watchdog has tripped.
    pub fn check_watchdog(&self) -> bool {
        unsafe { ptx_sys::gpu_hot_check_watchdog(self.inner.raw) }
    }

    /// Reset the watchdog.
    pub fn reset_watchdog(&self) {
        unsafe { ptx_sys::gpu_hot_reset_watchdog(self.inner.raw) }
    }

    // ========================================================================
    // Keepalive
    // ========================================================================

    /// Send a keepalive signal to prevent GPU idle timeout.
    pub fn keepalive(&self) {
        unsafe { ptx_sys::gpu_hot_keepalive(self.inner.raw) }
    }

    // ========================================================================
    // System State
    // ========================================================================

    /// Get the system state snapshot.
    pub fn system_snapshot(&self) -> ptx_sys::GPUHotSystemSnapshot {
        let mut snapshot = ptx_sys::GPUHotSystemSnapshot::default();
        unsafe { ptx_sys::gpu_hot_get_system_snapshot(self.inner.raw, &mut snapshot) };
        snapshot
    }

    /// Flush the task queue.
    pub fn flush_task_queue(&self) {
        unsafe { ptx_sys::gpu_hot_flush_task_queue(self.inner.raw) }
    }
}

// The runtime is thread-safe
unsafe impl Send for PtxRuntime {}
unsafe impl Sync for PtxRuntime {}

/// Global runtime instance for convenience.
static GLOBAL_RUNTIME: once_cell::sync::OnceCell<Arc<PtxRuntime>> = once_cell::sync::OnceCell::new();

/// Get or initialize the global runtime on device 0.
pub fn global_runtime() -> Result<Arc<PtxRuntime>> {
    GLOBAL_RUNTIME
        .get_or_try_init(|| PtxRuntime::new(0).map(Arc::new))
        .cloned()
}

/// Initialize the global runtime with a specific device.
pub fn init_global_runtime(device_id: i32) -> Result<Arc<PtxRuntime>> {
    GLOBAL_RUNTIME
        .get_or_try_init(|| PtxRuntime::new(device_id).map(Arc::new))
        .cloned()
}

/// Install an existing runtime as the process-global runtime.
///
/// If a global runtime was already installed, the existing instance is retained
/// and returned.
pub fn install_global_runtime(runtime: Arc<PtxRuntime>) -> Arc<PtxRuntime> {
    if GLOBAL_RUNTIME.set(Arc::clone(&runtime)).is_ok() {
        runtime
    } else {
        GLOBAL_RUNTIME
            .get()
            .expect("global runtime should exist after set failure")
            .clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: These tests require a CUDA-capable GPU and the PTX-OS library
    #[test]
    #[ignore]
    fn test_runtime_init() {
        let runtime = PtxRuntime::new(0).expect("Failed to init");
        assert!(runtime.device().is_gpu());
    }

    #[test]
    #[ignore]
    fn test_alloc_free() {
        let runtime = PtxRuntime::new(0).expect("Failed to init");
        let ptr = runtime.alloc(1024).expect("Failed to alloc");
        assert!(ptr.is_valid());
        // ptr is automatically freed when dropped
    }
}
