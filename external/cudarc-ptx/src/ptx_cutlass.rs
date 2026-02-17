//! CUTLASS GEMM Intercept for cuBLAS
//!
//! When the `ptx-cutlass` feature is enabled, this module provides a CUTLASS
//! fast-path for FP16 OP_N/OP_N GEMM calls. The CUTLASS kernel runs ~12%
//! faster than cuBLAS on Jetson Orin (3230 vs 2889 GFLOPS).
//!
//! When `ptx-cutlass` is disabled, all functions return `Err` so the caller
//! falls through to cuBLAS transparently.

#[cfg(feature = "ptx-cutlass")]
use std::sync::{Mutex, OnceLock};

#[cfg(feature = "ptx-cutlass")]
use crate::cublas::result::CublasError;

#[cfg(feature = "ptx-cutlass")]
static CUTLASS_GEMM: OnceLock<Mutex<cutlass_ptx::CutlassGemm>> = OnceLock::new();

#[cfg(feature = "ptx-cutlass")]
fn get_gemm() -> &'static Mutex<cutlass_ptx::CutlassGemm> {
    CUTLASS_GEMM.get_or_init(|| {
        // Share the existing ptx_alloc runtime — do NOT create a second TLSF pool.
        // ptx_alloc is guaranteed to be active (ptx-cutlass depends on ptx-alloc).
        let runtime = crate::ptx_alloc::get_runtime()
            .expect("[ptx-cutlass] ptx_alloc runtime not initialized yet");
        let gemm = cutlass_ptx::CutlassGemm::with_runtime(&runtime);
        eprintln!("[ptx-cutlass] CUTLASS GEMM initialized (sharing ptx_alloc TLSF pool)");
        Mutex::new(gemm)
    })
}

/// Attempt CUTLASS FP16 column-major GEMM (cuBLAS-compatible NN layout).
///
/// Called from `gemm_ex()` when transa=OP_N, transb=OP_N, all types=FP16.
/// Alpha and beta are read from raw pointers (host-side f32, matching
/// cuBLAS COMPUTE_32F convention).
///
/// Returns `Ok(())` on success, `Err` on failure (caller falls through to cuBLAS).
#[cfg(feature = "ptx-cutlass")]
pub unsafe fn cutlass_hgemm_nn(
    a: *const core::ffi::c_void,
    b: *const core::ffi::c_void,
    c: *mut core::ffi::c_void,
    m: core::ffi::c_int,
    n: core::ffi::c_int,
    k: core::ffi::c_int,
    lda: core::ffi::c_int,
    ldb: core::ffi::c_int,
    ldc: core::ffi::c_int,
    alpha: *const core::ffi::c_void,
    beta: *const core::ffi::c_void,
) -> Result<(), CublasError> {
    let alpha_f32 = unsafe { *(alpha as *const f32) };
    let beta_f32 = unsafe { *(beta as *const f32) };

    let mut gemm = get_gemm()
        .lock()
        .map_err(|_| CublasError(crate::cublas::sys::cublasStatus_t::CUBLAS_STATUS_INTERNAL_ERROR))?;

    unsafe {
        gemm.hgemm_nn(
            a, b, c,
            m as usize, n as usize, k as usize,
            lda as usize, ldb as usize, ldc as usize,
            alpha_f32, beta_f32,
        )
    }
    .map_err(|_| CublasError(crate::cublas::sys::cublasStatus_t::CUBLAS_STATUS_INTERNAL_ERROR))
}

/// Fallback: CUTLASS not available, always returns Err so caller uses cuBLAS.
#[cfg(not(feature = "ptx-cutlass"))]
pub unsafe fn cutlass_hgemm_nn(
    _a: *const core::ffi::c_void,
    _b: *const core::ffi::c_void,
    _c: *mut core::ffi::c_void,
    _m: core::ffi::c_int,
    _n: core::ffi::c_int,
    _k: core::ffi::c_int,
    _lda: core::ffi::c_int,
    _ldb: core::ffi::c_int,
    _ldc: core::ffi::c_int,
    _alpha: *const core::ffi::c_void,
    _beta: *const core::ffi::c_void,
) -> Result<(), crate::cublas::result::CublasError> {
    Err(crate::cublas::result::CublasError(
        crate::cublas::sys::cublasStatus_t::CUBLAS_STATUS_NOT_SUPPORTED,
    ))
}
