//! CUTLASS GEMM correctness and performance test.
//!
//! Tests the FP16×FP16 GEMM kernel against cuBLAS GemmEx for correctness,
//! then benchmarks both paths and reports GFLOPS.
//!
//! Run:
//!   CUDA_ARCH=sm_87 PTX_POOL_FRACTION=0.90 cargo run --release --example cutlass_gemm_test

use cutlass_ptx::CutlassGemm;
use std::os::raw::c_void;
use std::time::Instant;

// Re-export ptx_sys via the crate's dependency
extern crate ptx_sys;

const M: usize = 1024;
const N: usize = 1024;
const K: usize = 1024;
const WARMUP_ITERS: usize = 5;
const BENCH_ITERS: usize = 20;

/// Convert f32 to FP16 (IEEE 754 half-precision) bit representation.
fn f32_to_f16(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exponent = ((bits >> 23) & 0xFF) as i32;
    let mantissa = bits & 0x7FFFFF;

    if exponent == 0xFF {
        // Inf/NaN
        return (sign | 0x7C00 | if mantissa != 0 { 0x200 } else { 0 }) as u16;
    }

    let new_exp = exponent - 127 + 15;
    if new_exp >= 31 {
        return (sign | 0x7C00) as u16; // overflow -> Inf
    }
    if new_exp <= 0 {
        return sign as u16; // underflow -> zero
    }

    (sign | ((new_exp as u32) << 10) | (mantissa >> 13)) as u16
}

/// Convert FP16 bit representation to f32.
fn f16_to_f32(val: u16) -> f32 {
    let sign = ((val as u32) & 0x8000) << 16;
    let exponent = ((val as u32) >> 10) & 0x1F;
    let mantissa = (val as u32) & 0x3FF;

    if exponent == 0 {
        if mantissa == 0 {
            return f32::from_bits(sign);
        }
        // Denormalized
        let mut m = mantissa;
        let mut e = 1u32;
        while (m & 0x400) == 0 {
            m <<= 1;
            e += 1;
        }
        let exp = (127 - 15 + 1 - e) << 23;
        let man = (m & 0x3FF) << 13;
        return f32::from_bits(sign | exp | man);
    }
    if exponent == 31 {
        let bits = sign | 0x7F800000 | (mantissa << 13);
        return f32::from_bits(bits);
    }

    let exp = (exponent + 127 - 15) << 23;
    let man = mantissa << 13;
    f32::from_bits(sign | exp | man)
}

fn main() {
    eprintln!("=== CUTLASS GEMM Test ===");
    eprintln!("Matrix size: {}x{}x{}", M, N, K);
    eprintln!();

    // --- Initialize runtime ---
    let device_id = 0i32;
    cutlass_ptx::init_cutlass_allocator(device_id as usize)
        .expect("Failed to initialize CUTLASS TLSF allocator");

    let mut gemm = CutlassGemm::new(device_id).expect("Failed to create CutlassGemm");
    let rt = gemm.runtime().clone();

    // --- Generate test data on host ---
    eprintln!("Generating test matrices...");

    let a_host: Vec<u16> = (0..M * K)
        .map(|i| f32_to_f16(((i % 17) as f32 - 8.0) * 0.01))
        .collect();

    let b_host: Vec<u16> = (0..N * K)
        .map(|i| f32_to_f16(((i % 13) as f32 - 6.0) * 0.01))
        .collect();

    let c_host_init: Vec<u16> = vec![f32_to_f16(0.0); M * N];

    // --- Allocate GPU memory via TLSF ---
    let a_size = M * K * 2; // FP16 = 2 bytes
    let b_size = N * K * 2;
    let c_size = M * N * 2;

    let a_gpu = rt.alloc(a_size).expect("Failed to alloc A");
    let b_gpu = rt.alloc(b_size).expect("Failed to alloc B");
    let c_cutlass = rt.alloc(c_size).expect("Failed to alloc C (CUTLASS)");
    let c_cublas = rt.alloc(c_size).expect("Failed to alloc C (cuBLAS)");

    // --- Upload data ---
    unsafe {
        a_gpu
            .copy_from_host(a_host.as_ptr() as *const c_void, a_size)
            .expect("Failed to upload A");
        b_gpu
            .copy_from_host(b_host.as_ptr() as *const c_void, b_size)
            .expect("Failed to upload B");
        c_cutlass
            .copy_from_host(c_host_init.as_ptr() as *const c_void, c_size)
            .expect("Failed to upload C (CUTLASS)");
        c_cublas
            .copy_from_host(c_host_init.as_ptr() as *const c_void, c_size)
            .expect("Failed to upload C (cuBLAS)");
    }

    // --- Run CUTLASS HGEMM ---
    eprintln!("Running CUTLASS HGEMM...");
    unsafe {
        gemm.hgemm(
            a_gpu.as_ptr(),
            b_gpu.as_ptr(),
            c_cutlass.as_ptr() as *mut c_void,
            M,
            N,
            K,
            1.0,
            0.0,
        )
        .expect("CUTLASS HGEMM failed");

        ptx_sys::cudaDeviceSynchronize();
    }

    // --- Run cuBLAS GemmEx for comparison ---
    eprintln!("Running cuBLAS GemmEx (FP16)...");
    {
        let cublas_guard = rt.cublas().expect("Failed to get cuBLAS handle");
        let cublas = cublas_guard.as_ref().expect("cuBLAS handle not initialized");

        let alpha_f32: f32 = 1.0;
        let beta_f32: f32 = 0.0;

        // cuBLAS is column-major. For row-major C = A @ B^T:
        // We compute C^T = B @ A^T in column-major
        // B is (N,K) col-major -> no transpose needed (B = N×K in col-major is fine)
        // A is (M,K) row-major = (K,M) col-major -> transpose
        let status = unsafe {
            ptx_sys::cublasGemmEx(
                cublas.raw(),
                ptx_sys::cublasOperation_t::CUBLAS_OP_T, // B transposed
                ptx_sys::cublasOperation_t::CUBLAS_OP_N, // A no transpose
                N as i32,                                 // m (rows of result in col-major)
                M as i32,                                 // n (cols of result in col-major)
                K as i32,                                 // k
                &alpha_f32 as *const f32 as *const c_void,
                b_gpu.as_ptr() as *const c_void,
                ptx_sys::CUDA_R_16F,
                K as i32,                                 // ldb
                a_gpu.as_ptr() as *const c_void,
                ptx_sys::CUDA_R_16F,
                K as i32,                                 // lda
                &beta_f32 as *const f32 as *const c_void,
                c_cublas.as_ptr() as *mut c_void,
                ptx_sys::CUDA_R_16F,
                N as i32,                                 // ldc
                ptx_sys::CUDA_R_32F,                      // compute in FP32
                ptx_sys::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
            )
        };
        assert_eq!(
            status as i32, 0,
            "cuBLAS GemmEx failed with status {}",
            status as i32
        );

        unsafe {
            ptx_sys::cudaDeviceSynchronize();
        }
    }

    // --- Download and compare results ---
    eprintln!("Comparing results...");
    let mut c_cutlass_host = vec![0u16; M * N];
    let mut c_cublas_host = vec![0u16; M * N];

    unsafe {
        c_cutlass
            .copy_to_host(c_cutlass_host.as_mut_ptr() as *mut c_void, c_size)
            .expect("Failed to download CUTLASS result");
        c_cublas
            .copy_to_host(c_cublas_host.as_mut_ptr() as *mut c_void, c_size)
            .expect("Failed to download cuBLAS result");
    }

    let mut max_abs_diff: f32 = 0.0;
    let mut max_rel_diff: f32 = 0.0;
    let mut mismatches = 0usize;
    let tolerance = 0.05; // 5% relative tolerance for FP16

    for i in 0..M * N {
        let cutlass_val = f16_to_f32(c_cutlass_host[i]);
        let cublas_val = f16_to_f32(c_cublas_host[i]);
        let abs_diff = (cutlass_val - cublas_val).abs();
        let rel_diff = if cublas_val.abs() > 1e-6 {
            abs_diff / cublas_val.abs()
        } else {
            abs_diff
        };

        max_abs_diff = max_abs_diff.max(abs_diff);
        max_rel_diff = max_rel_diff.max(rel_diff);

        if rel_diff > tolerance && abs_diff > 1e-3 {
            mismatches += 1;
            if mismatches <= 5 {
                let row = i / N;
                let col = i % N;
                eprintln!(
                    "  Mismatch at ({},{}): CUTLASS={:.6} cuBLAS={:.6} diff={:.6}",
                    row, col, cutlass_val, cublas_val, abs_diff
                );
            }
        }
    }

    eprintln!();
    eprintln!("Correctness results:");
    eprintln!(
        "  Max absolute diff: {:.6}",
        max_abs_diff
    );
    eprintln!(
        "  Max relative diff: {:.6}",
        max_rel_diff
    );
    eprintln!(
        "  Mismatches (>{:.0}% rel): {} / {}",
        tolerance * 100.0,
        mismatches,
        M * N
    );

    if mismatches == 0 {
        eprintln!("  PASS: CUTLASS matches cuBLAS within tolerance");
    } else {
        eprintln!("  WARN: {} mismatches detected", mismatches);
    }

    // --- Benchmark ---
    eprintln!();
    eprintln!("Benchmarking ({} warmup, {} timed iterations)...", WARMUP_ITERS, BENCH_ITERS);

    let flops = 2.0 * M as f64 * N as f64 * K as f64; // 2*M*N*K FLOPs per GEMM

    // Warmup CUTLASS
    for _ in 0..WARMUP_ITERS {
        unsafe {
            gemm.hgemm(
                a_gpu.as_ptr(),
                b_gpu.as_ptr(),
                c_cutlass.as_ptr() as *mut c_void,
                M, N, K, 1.0, 0.0,
            )
            .unwrap();
        }
    }
    unsafe { ptx_sys::cudaDeviceSynchronize(); }

    // Timed CUTLASS
    let start = Instant::now();
    for _ in 0..BENCH_ITERS {
        unsafe {
            gemm.hgemm(
                a_gpu.as_ptr(),
                b_gpu.as_ptr(),
                c_cutlass.as_ptr() as *mut c_void,
                M, N, K, 1.0, 0.0,
            )
            .unwrap();
        }
    }
    unsafe { ptx_sys::cudaDeviceSynchronize(); }
    let cutlass_time = start.elapsed();
    let cutlass_avg_ms = cutlass_time.as_secs_f64() * 1000.0 / BENCH_ITERS as f64;
    let cutlass_gflops = flops / (cutlass_avg_ms * 1e6);

    // Warmup cuBLAS
    {
        let cublas_guard = rt.cublas().expect("cuBLAS");
        let cublas = cublas_guard.as_ref().expect("cuBLAS handle");
        let alpha_f32: f32 = 1.0;
        let beta_f32: f32 = 0.0;

        for _ in 0..WARMUP_ITERS {
            unsafe {
                ptx_sys::cublasGemmEx(
                    cublas.raw(),
                    ptx_sys::cublasOperation_t::CUBLAS_OP_T,
                    ptx_sys::cublasOperation_t::CUBLAS_OP_N,
                    N as i32, M as i32, K as i32,
                    &alpha_f32 as *const f32 as *const c_void,
                    b_gpu.as_ptr() as *const c_void, ptx_sys::CUDA_R_16F, K as i32,
                    a_gpu.as_ptr() as *const c_void, ptx_sys::CUDA_R_16F, K as i32,
                    &beta_f32 as *const f32 as *const c_void,
                    c_cublas.as_ptr() as *mut c_void, ptx_sys::CUDA_R_16F, N as i32,
                    ptx_sys::CUDA_R_32F,
                    ptx_sys::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
                );
            }
        }
        unsafe { ptx_sys::cudaDeviceSynchronize(); }

        // Timed cuBLAS
        let start = Instant::now();
        for _ in 0..BENCH_ITERS {
            unsafe {
                ptx_sys::cublasGemmEx(
                    cublas.raw(),
                    ptx_sys::cublasOperation_t::CUBLAS_OP_T,
                    ptx_sys::cublasOperation_t::CUBLAS_OP_N,
                    N as i32, M as i32, K as i32,
                    &alpha_f32 as *const f32 as *const c_void,
                    b_gpu.as_ptr() as *const c_void, ptx_sys::CUDA_R_16F, K as i32,
                    a_gpu.as_ptr() as *const c_void, ptx_sys::CUDA_R_16F, K as i32,
                    &beta_f32 as *const f32 as *const c_void,
                    c_cublas.as_ptr() as *mut c_void, ptx_sys::CUDA_R_16F, N as i32,
                    ptx_sys::CUDA_R_32F,
                    ptx_sys::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
                );
            }
        }
        unsafe { ptx_sys::cudaDeviceSynchronize(); }
        let cublas_time = start.elapsed();
        let cublas_avg_ms = cublas_time.as_secs_f64() * 1000.0 / BENCH_ITERS as f64;
        let cublas_gflops = flops / (cublas_avg_ms * 1e6);

        eprintln!();
        eprintln!("Performance results ({}x{}x{}, {} iters):", M, N, K, BENCH_ITERS);
        eprintln!(
            "  CUTLASS: {:.3} ms/iter, {:.1} GFLOPS",
            cutlass_avg_ms, cutlass_gflops
        );
        eprintln!(
            "  cuBLAS:  {:.3} ms/iter, {:.1} GFLOPS",
            cublas_avg_ms, cublas_gflops
        );
        eprintln!(
            "  Ratio:   {:.2}x (CUTLASS vs cuBLAS)",
            cublas_avg_ms / cutlass_avg_ms
        );
    }

    // --- Print TLSF stats ---
    cutlass_ptx::cutlass_tlsf_print_stats();

    eprintln!();
    eprintln!("=== Test complete ===");
}
