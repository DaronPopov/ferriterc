//! NVIDIA CUTLASS GEMM kernels with PTX-OS TLSF allocator.
//!
//! This crate provides Rust bindings to CUTLASS template-instantiated GEMM kernels
//! for tensor-core accelerated matrix multiplication on sm_80+ (Ampere/Orin).
//!
//! # Architecture
//!
//! ```text
//! Rust caller
//!   └─ CutlassGemm handle
//!       ├─ hgemm()       → cutlass_hgemm()     → CUTLASS FP16×FP16 GEMM
//!       ├─ gemm_i8()     → cutlass_gemm_i8()   → CUTLASS INT8×INT8 GEMM
//!       └─ gemm_i4()     → cutlass_gemm_i4()   → CUTLASS INT4×INT4 GEMM
//!       └─ workspace allocated via TLSF (PtxRuntime::alloc)
//! ```
//!
//! # Supported Kernels
//!
//! - **FP16 × FP16 → FP16**: Standard half-precision GEMM with FP32 accumulator
//! - **INT8 × INT8 → FP16**: Quantized GEMM with INT32 accumulator, FP16 output via epilogue
//! - **INT4 × INT4 → FP16**: Quantized GEMM with INT32 accumulator, FP16 output via epilogue
//!
//! The INT8/INT4 kernels are designed for quantized inference: both activations and
//! weights are quantized, the INT32 accumulator preserves precision, and the epilogue
//! applies a dequantization scale (alpha) and converts to FP16 output.

use ptx_runtime::{GpuPtr, PTXStableConfig, PtxRuntime};
use std::collections::HashMap;
use std::os::raw::c_void;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, Once, OnceLock};

pub mod policy;

// ─── Global state (matches onnxruntime-ptx / ferrite-xla pattern) ────────────

static INIT: Once = Once::new();
static INITIALIZED: AtomicBool = AtomicBool::new(false);
static GLOBAL_RUNTIME: OnceLock<Arc<PtxRuntime>> = OnceLock::new();
static PTR_MAP: OnceLock<Mutex<HashMap<usize, Arc<GpuPtr>>>> = OnceLock::new();

// ─── FFI declarations (implemented in cpp/cutlass_gemm.cu) ───────────────────

unsafe extern "C" {
    // FP16 x FP16 -> FP16
    fn cutlass_hgemm(
        a: *const c_void,
        b: *const c_void,
        c: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        workspace: *mut c_void,
        workspace_size: usize,
    ) -> i32;

    fn cutlass_hgemm_workspace_size(m: i32, n: i32, k: i32) -> usize;

    // FP16 x FP16 -> FP16 (column-major, cuBLAS-compatible)
    fn cutlass_hgemm_nn(
        a: *const c_void,
        b: *const c_void,
        c: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
        lda: i32,
        ldb: i32,
        ldc: i32,
        alpha: f32,
        beta: f32,
        workspace: *mut c_void,
        workspace_size: usize,
    ) -> i32;

    fn cutlass_hgemm_nn_workspace_size(m: i32, n: i32, k: i32) -> usize;

    // INT8 x INT8 -> FP16
    fn cutlass_gemm_i8(
        a: *const c_void,
        b: *const c_void,
        c: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
        scale: f32,
        beta: f32,
        workspace: *mut c_void,
        workspace_size: usize,
    ) -> i32;

    fn cutlass_gemm_i8_workspace_size(m: i32, n: i32, k: i32) -> usize;

    // INT4 x INT4 -> FP16
    fn cutlass_gemm_i4(
        a: *const c_void,
        b: *const c_void,
        c: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
        scale: f32,
        beta: f32,
        workspace: *mut c_void,
        workspace_size: usize,
    ) -> i32;

    fn cutlass_gemm_i4_workspace_size(m: i32, n: i32, k: i32) -> usize;
}

// ─── FFI entry points (called from C++ if needed, matches bridge pattern) ────

fn runtime() -> Option<&'static Arc<PtxRuntime>> {
    GLOBAL_RUNTIME.get()
}

/// Initialize PTX runtime for CUTLASS (called internally or from C++).
#[unsafe(no_mangle)]
pub extern "C" fn cutlass_tlsf_init(device_id: i32) {
    INIT.call_once(|| {
        let cfg: PTXStableConfig = policy::CutlassTlsfPolicy::default().stable_config();

        match PtxRuntime::with_stable_config(device_id, Some(cfg)) {
            Ok(rt) => {
                let rt = Arc::new(rt);
                let _ = GLOBAL_RUNTIME.set(rt);
                let _ = PTR_MAP.set(Mutex::new(HashMap::new()));
                INITIALIZED.store(true, Ordering::Release);
                eprintln!(
                    "[CUTLASS-TLSF] Runtime initialized on device {}",
                    device_id
                );
            }
            Err(e) => {
                eprintln!("[CUTLASS-TLSF] FATAL: Failed to initialize: {:?}", e);
            }
        }
    });
}

/// Allocate GPU memory via TLSF.
#[unsafe(no_mangle)]
pub extern "C" fn cutlass_tlsf_alloc(size: usize) -> *mut c_void {
    let Some(rt) = runtime() else {
        eprintln!("[CUTLASS-TLSF] ERROR: Runtime not initialized!");
        return std::ptr::null_mut();
    };

    match rt.alloc(size) {
        Ok(gpu_ptr) => {
            let raw_ptr = gpu_ptr.as_ptr() as usize;
            let gpu_ptr_arc = Arc::new(gpu_ptr);

            if let Some(map) = PTR_MAP.get() {
                if let Ok(mut ptr_map) = map.lock() {
                    ptr_map.insert(raw_ptr, gpu_ptr_arc);
                }
            }

            raw_ptr as *mut c_void
        }
        Err(e) => {
            eprintln!("[CUTLASS-TLSF] Alloc failed ({} bytes): {:?}", size, e);
            std::ptr::null_mut()
        }
    }
}

/// Free GPU memory via TLSF.
#[unsafe(no_mangle)]
pub extern "C" fn cutlass_tlsf_free(ptr: *mut c_void) {
    if ptr.is_null() {
        return;
    }

    let raw_ptr = ptr as usize;

    if let Some(map) = PTR_MAP.get() {
        if let Ok(mut ptr_map) = map.lock() {
            if let Some(gpu_ptr_arc) = ptr_map.remove(&raw_ptr) {
                drop(gpu_ptr_arc);
                return;
            }
        }
    }

    eprintln!(
        "[CUTLASS-TLSF] WARNING: Attempted to free unknown pointer {:p}",
        ptr
    );
}

/// Print TLSF statistics.
#[unsafe(no_mangle)]
pub extern "C" fn cutlass_tlsf_print_stats() {
    if let Some(rt) = runtime() {
        let stats = rt.tlsf_stats();
        eprintln!("\n[CUTLASS-TLSF] Statistics:");
        eprintln!(
            "  Pool size:      {:.2} GB",
            stats.total_pool_size as f64 / 1e9
        );
        eprintln!(
            "  Allocated:      {:.2} MB",
            stats.allocated_bytes as f64 / 1e6
        );
        eprintln!(
            "  Peak:           {:.2} MB",
            stats.peak_allocated as f64 / 1e6
        );
        eprintln!("  Fragmentation:  {:.6}", stats.fragmentation_ratio);
        eprintln!("  Utilization:    {:.1}%\n", stats.utilization_percent);
    }
}

// ─── CutlassGemm handle (Rust-friendly API) ─────────────────────────────────

/// CUTLASS GEMM handle with TLSF-backed workspace.
///
/// Manages workspace memory for CUTLASS kernels, automatically resizing
/// via the PTX-OS TLSF allocator as needed.
pub struct CutlassGemm {
    runtime: Arc<PtxRuntime>,
    workspace: Option<GpuPtr>,
    workspace_size: usize,
}

impl CutlassGemm {
    /// Create a new CUTLASS GEMM handle.
    ///
    /// Uses the global TLSF runtime if initialized, otherwise initializes
    /// a new runtime on the specified device.
    pub fn new(device_id: i32) -> Result<Self, String> {
        cutlass_tlsf_init(device_id);

        let rt = runtime()
            .ok_or_else(|| "CUTLASS TLSF runtime not initialized".to_string())?
            .clone();

        Ok(Self {
            runtime: rt,
            workspace: None,
            workspace_size: 0,
        })
    }

    /// Create a handle using an existing PtxRuntime.
    pub fn with_runtime(runtime: &Arc<PtxRuntime>) -> Self {
        // Ensure global state is set up for FFI callbacks
        let _ = GLOBAL_RUNTIME.set(runtime.clone());
        let _ = PTR_MAP.set(Mutex::new(HashMap::new()));
        INITIALIZED.store(true, Ordering::Release);

        Self {
            runtime: runtime.clone(),
            workspace: None,
            workspace_size: 0,
        }
    }

    /// Ensure workspace is at least `needed` bytes, reallocating if necessary.
    fn ensure_workspace(&mut self, needed: usize) -> Result<*mut c_void, String> {
        if needed == 0 {
            return Ok(std::ptr::null_mut());
        }

        if self.workspace_size >= needed {
            return Ok(self
                .workspace
                .as_ref()
                .map(|w| w.as_ptr())
                .unwrap_or(std::ptr::null_mut()));
        }

        // Drop old workspace before allocating new one
        self.workspace = None;
        self.workspace_size = 0;

        match self.runtime.alloc(needed) {
            Ok(gpu_ptr) => {
                let ptr = gpu_ptr.as_ptr();
                self.workspace = Some(gpu_ptr);
                self.workspace_size = needed;
                Ok(ptr)
            }
            Err(e) => Err(format!(
                "Failed to allocate workspace ({} bytes): {:?}",
                needed, e
            )),
        }
    }

    /// FP16 × FP16 → FP16 GEMM via tensor cores.
    ///
    /// Computes `C = alpha * A @ B + beta * C` where:
    /// - `A`: (M, K) row-major FP16
    /// - `B`: (N, K) column-major FP16 (transposed weight matrix)
    /// - `C`: (M, N) row-major FP16 (output, also used as input for beta != 0)
    ///
    /// # Safety
    /// Pointers must be valid device pointers with correct dimensions.
    pub unsafe fn hgemm(
        &mut self,
        a: *const c_void,
        b: *const c_void,
        c: *mut c_void,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        beta: f32,
    ) -> Result<(), i32> {
        let (m, n, k) = (m as i32, n as i32, k as i32);
        let ws_size = unsafe { cutlass_hgemm_workspace_size(m, n, k) };
        let ws_ptr = self
            .ensure_workspace(ws_size)
            .map_err(|_| -1i32)?;

        let status = unsafe {
            cutlass_hgemm(a, b, c, m, n, k, alpha, beta, ws_ptr, ws_size)
        };

        if status == 0 {
            Ok(())
        } else {
            Err(status)
        }
    }

    /// Column-major FP16 × FP16 → FP16 GEMM via tensor cores (cuBLAS-compatible).
    ///
    /// Computes `C = alpha * A @ B + beta * C` where all matrices are column-major:
    /// - `A`: (M, K) column-major FP16, leading dimension `lda`
    /// - `B`: (K, N) column-major FP16, leading dimension `ldb`
    /// - `C`: (M, N) column-major FP16, leading dimension `ldc`
    ///
    /// Accepts lda/ldb/ldc directly from cuBLAS — no layout conversion needed.
    ///
    /// # Safety
    /// Pointers must be valid device pointers with correct dimensions.
    pub unsafe fn hgemm_nn(
        &mut self,
        a: *const c_void,
        b: *const c_void,
        c: *mut c_void,
        m: usize,
        n: usize,
        k: usize,
        lda: usize,
        ldb: usize,
        ldc: usize,
        alpha: f32,
        beta: f32,
    ) -> Result<(), i32> {
        let (m_i, n_i, k_i) = (m as i32, n as i32, k as i32);
        let ws_size = unsafe { cutlass_hgemm_nn_workspace_size(m_i, n_i, k_i) };
        let ws_ptr = self
            .ensure_workspace(ws_size)
            .map_err(|_| -1i32)?;

        let status = unsafe {
            cutlass_hgemm_nn(
                a, b, c, m_i, n_i, k_i,
                lda as i32, ldb as i32, ldc as i32,
                alpha, beta, ws_ptr, ws_size,
            )
        };

        if status == 0 {
            Ok(())
        } else {
            Err(status)
        }
    }

    /// INT8 × INT8 → FP16 quantized GEMM via tensor cores.
    ///
    /// For Q8_0 quantized inference. Computes `C = scale * A_i8 @ B_i8 + beta * C`.
    /// Both activations and weights are INT8; INT32 accumulator preserves precision;
    /// epilogue applies dequant scale and converts to FP16.
    ///
    /// - `a`: (M, K) row-major INT8 quantized activations
    /// - `b`: (N, K) column-major INT8 quantized weights
    /// - `c`: (M, N) row-major FP16 output
    /// - `scale`: dequantization scale (alpha = scale_a * scale_b)
    ///
    /// # Safety
    /// Pointers must be valid device pointers with correct dimensions.
    pub unsafe fn gemm_i8(
        &mut self,
        a: *const c_void,
        b: *const c_void,
        c: *mut c_void,
        m: usize,
        n: usize,
        k: usize,
        scale: f32,
        beta: f32,
    ) -> Result<(), i32> {
        let (m, n, k) = (m as i32, n as i32, k as i32);
        let ws_size = unsafe { cutlass_gemm_i8_workspace_size(m, n, k) };
        let ws_ptr = self
            .ensure_workspace(ws_size)
            .map_err(|_| -1i32)?;

        let status = unsafe {
            cutlass_gemm_i8(a, b, c, m, n, k, scale, beta, ws_ptr, ws_size)
        };

        if status == 0 {
            Ok(())
        } else {
            Err(status)
        }
    }

    /// INT4 × INT4 → FP16 quantized GEMM via tensor cores.
    ///
    /// For Q4_0 quantized inference. Computes `C = scale * A_i4 @ B_i4 + beta * C`.
    /// Both activations and weights are INT4 (packed, 2 per byte); INT32 accumulator
    /// preserves precision; epilogue applies dequant scale and converts to FP16.
    ///
    /// - `a`: (M, K) row-major INT4 quantized activations (packed)
    /// - `b`: (N, K) column-major INT4 quantized weights (packed)
    /// - `c`: (M, N) row-major FP16 output
    /// - `scale`: dequantization scale (alpha = scale_a * scale_b)
    ///
    /// # Safety
    /// Pointers must be valid device pointers with correct dimensions.
    pub unsafe fn gemm_i4(
        &mut self,
        a: *const c_void,
        b: *const c_void,
        c: *mut c_void,
        m: usize,
        n: usize,
        k: usize,
        scale: f32,
        beta: f32,
    ) -> Result<(), i32> {
        let (m, n, k) = (m as i32, n as i32, k as i32);
        let ws_size = unsafe { cutlass_gemm_i4_workspace_size(m, n, k) };
        let ws_ptr = self
            .ensure_workspace(ws_size)
            .map_err(|_| -1i32)?;

        let status = unsafe {
            cutlass_gemm_i4(a, b, c, m, n, k, scale, beta, ws_ptr, ws_size)
        };

        if status == 0 {
            Ok(())
        } else {
            Err(status)
        }
    }

    /// Get the underlying PtxRuntime.
    pub fn runtime(&self) -> &Arc<PtxRuntime> {
        &self.runtime
    }
}

// ─── Rust-friendly public API ────────────────────────────────────────────────

/// Initialize the CUTLASS TLSF allocator from Rust.
pub fn init_cutlass_allocator(device_id: usize) -> Result<(), String> {
    cutlass_tlsf_init(device_id as i32);
    if INITIALIZED.load(Ordering::Acquire) {
        Ok(())
    } else {
        Err("Failed to initialize CUTLASS TLSF allocator".to_string())
    }
}

/// Get TLSF pool statistics. Returns `None` if the runtime is not initialized.
pub fn get_tlsf_stats() -> Option<ptx_sys::TLSFPoolStats> {
    runtime().map(|rt| rt.tlsf_stats())
}

/// Check if the runtime is initialized.
pub fn is_initialized() -> bool {
    INITIALIZED.load(Ordering::Acquire)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init() {
        let result = init_cutlass_allocator(0);
        assert!(result.is_ok());
    }

    #[test]
    fn test_alloc_free() {
        init_cutlass_allocator(0).unwrap();

        let ptr = cutlass_tlsf_alloc(1024);
        assert!(!ptr.is_null());

        cutlass_tlsf_free(ptr);
    }
}
