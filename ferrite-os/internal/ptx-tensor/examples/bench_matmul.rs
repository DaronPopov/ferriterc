//! PTX-OS matmul benchmark (cuBLAS).
//!
//! Run with:
//!   cargo run --release -p ptx-tensor --example bench_matmul
//!
//! Tunables:
//!   PTX_BENCH_M=1024
//!   PTX_BENCH_N=1024
//!   PTX_BENCH_K=1024
//!   PTX_BENCH_ITERS=10
//!   PTX_BENCH_WARMUP=2

use std::env;
use std::sync::Arc;
use std::time::Instant;

use ptx_runtime::PtxRuntime;
use ptx_tensor::{Tensor, DType, Result};

fn parse_usize_env(key: &str, default: usize) -> usize {
    env::var(key)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(default)
}

fn main() -> Result<()> {
    let m = parse_usize_env("PTX_BENCH_M", 1024);
    let n = parse_usize_env("PTX_BENCH_N", 1024);
    let k = parse_usize_env("PTX_BENCH_K", 1024);
    let iters = parse_usize_env("PTX_BENCH_ITERS", 10);
    let warmup = parse_usize_env("PTX_BENCH_WARMUP", 2);

    println!("=== PTX-OS Matmul Benchmark ===");
    println!("shape: [{}x{}] @ [{}x{}]", m, k, k, n);
    println!("iters: {}, warmup: {}", iters, warmup);
    println!();

    let runtime = Arc::new(PtxRuntime::new(0)?);

    let a = Tensor::full(&[m, k], 1.0, DType::F32, &runtime)?;
    let b = Tensor::full(&[k, n], 2.0, DType::F32, &runtime)?;
    runtime.sync_all();

    // Warmup
    for _ in 0..warmup {
        let _c = a.matmul(&b)?;
    }
    runtime.sync_all();

    let start = Instant::now();
    let mut last = None;
    for _ in 0..iters {
        let c = a.matmul(&b)?;
        last = Some(c);
    }
    runtime.sync_all();
    let elapsed = start.elapsed();
    drop(last);

    let flops = 2.0 * (m as f64) * (n as f64) * (k as f64) * (iters as f64);
    let secs = elapsed.as_secs_f64();
    let gflops = (flops / secs) / 1e9;

    println!("total time: {:.4} s", secs);
    println!("throughput: {:.2} GFLOP/s", gflops);

    Ok(())
}
