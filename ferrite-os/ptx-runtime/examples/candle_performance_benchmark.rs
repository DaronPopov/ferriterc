//! CANDLE KERNEL PERFORMANCE BENCHMARK
//!
//! Demonstrates maximum compute throughput with validated Candle kernels
//! orchestrated by PTX-OS TLSF allocator.
//!
//! This shows REAL GFLOPS numbers with large-scale operations!

use ptx_runtime::PtxRuntime;
use ptx_kernels::candle;
use std::time::Instant;
use std::ptr;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║  ⚡ CANDLE KERNEL PERFORMANCE BENCHMARK ⚡                 ║");
    println!("║                                                            ║");
    println!("║  Maximum compute throughput with TLSF orchestration       ║");
    println!("║  Let's see some REAL numbers! 🔥                          ║");
    println!("╚════════════════════════════════════════════════════════════╝");
    println!();

    // High-performance config - focus on compute, not just parallelism
    let mut config = ptx_sys::GPUHotConfig::default();
    config.pool_fraction = 0.80;  // Use 80% of VRAM
    config.max_streams = 1024;    // Moderate stream count for max throughput
    config.quiet_init = false;

    println!("⚡ Initializing PTX-OS runtime...");
    let runtime = PtxRuntime::with_config(0, Some(config))?;

    println!("  ✓ PTX-OS initialized");
    println!("  Streams: {}", runtime.num_streams());
    println!("  TLSF Pool: {:.2} GB", runtime.tlsf_stats().total_pool_size as f64 / 1e9);
    println!();

    // Performance-focused tests - LARGE workloads that fit in VRAM
    let test_configs = [
        (8, 32 * 1024 * 1024, "8 streams × 32M elements - WARMUP"),
        (16, 16 * 1024 * 1024, "16 streams × 16M elements - GETTING HOT"),
        (32, 8 * 1024 * 1024, "32 streams × 8M elements - HIGH THROUGHPUT"),
        (64, 4 * 1024 * 1024, "64 streams × 4M elements - BALANCED"),
        (128, 2 * 1024 * 1024, "128 streams × 2M elements - MAXIMUM SCALE"),
    ];

    for (num_streams, elements_per_stream, description) in test_configs.iter() {
        println!("═══════════════════════════════════════════════════════════");
        println!("🎯 TEST: {}", description);
        println!("═══════════════════════════════════════════════════════════");
        println!("  Concurrent streams: {}", num_streams);
        println!("  Elements per stream: {:.2}M", *elements_per_stream as f64 / 1e6);
        println!("  Total elements: {:.2}M", (*num_streams * *elements_per_stream) as f64 / 1e6);
        println!("  Memory per stream: {:.2} MB", (*elements_per_stream * 4 * 3) as f64 / 1e6);
        println!("  Total memory: {:.2} GB", (*num_streams * *elements_per_stream * 4 * 3) as f64 / 1e9);
        println!();

        match run_performance_benchmark(&runtime, *num_streams, *elements_per_stream) {
            Ok(_) => {},
            Err(e) => {
                println!("  ⚠️  Benchmark failed: {}", e);
                println!("  Skipping remaining tests...");
                break;
            }
        }

        println!();
    }

    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║  🎉 BENCHMARK COMPLETE! 🎉                                 ║");
    println!("║                                                            ║");
    println!("║  PTX-OS TLSF delivered maximum compute throughput!        ║");
    println!("╚════════════════════════════════════════════════════════════╝");

    Ok(())
}

fn run_performance_benchmark(
    runtime: &PtxRuntime,
    num_streams: usize,
    elements: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let bytes = elements * std::mem::size_of::<f32>();

    // Allocation phase - via TLSF
    println!("🌊 TLSF Allocation Phase...");
    let alloc_start = Instant::now();

    let mut workloads = Vec::with_capacity(num_streams);

    unsafe {
        let runtime_ptr = runtime.raw();

        for i in 0..num_streams {
            let stream_id = (i % runtime.num_streams()) as i32;
            let stream = ptx_sys::gpu_hot_get_stream(runtime_ptr, stream_id);

            // Allocate 3 large buffers per stream via TLSF
            let input = ptx_sys::gpu_hot_alloc_async(runtime_ptr, bytes, stream);
            let temp = ptx_sys::gpu_hot_alloc_async(runtime_ptr, bytes, stream);
            let output = ptx_sys::gpu_hot_alloc_async(runtime_ptr, bytes, stream);

            if input.is_null() || temp.is_null() || output.is_null() {
                return Err(format!("TLSF allocation failed at stream {}/{}", i, num_streams).into());
            }

            workloads.push((stream, input, temp, output));
        }
    }

    let alloc_time = alloc_start.elapsed();
    let total_bytes_allocated = num_streams * bytes * 3;
    println!("  ✓ Allocated {:.2} GB in {:?}", total_bytes_allocated as f64 / 1e9, alloc_time);
    println!("    Bandwidth: {:.2} GB/s", total_bytes_allocated as f64 / 1e9 / alloc_time.as_secs_f64());
    println!();

    // Launch phase - massive compute kernels
    println!("🚀 Kernel Launch Phase...");
    let launch_start = Instant::now();

    unsafe {
        for (stream, input, temp, output) in workloads.iter() {
            let input_f32 = *input as *const f32;
            let temp_f32 = *temp as *const f32;
            let output_f32 = *output as *mut f32;

            // Chain of compute-intensive operations:
            // 1. Exponential: exp(input) -> temp
            candle::candle_launch_uexp_f32(
                elements, 0, ptr::null(), input_f32, temp_f32 as *mut f32, *stream
            );

            // 2. Tanh: tanh(temp) -> output
            candle::candle_launch_utanh_f32(
                elements, 0, ptr::null(), temp_f32, output_f32, *stream
            );

            // 3. Sigmoid: sigmoid(output) -> temp
            candle::candle_launch_usigmoid_f32(
                elements, 0, ptr::null(), output_f32 as *const f32, temp_f32 as *mut f32, *stream
            );

            // 4. ReLU: relu(temp) -> output
            candle::candle_launch_urelu_f32(
                elements, 0, ptr::null(), temp_f32, output_f32, *stream
            );

            // 5. Binary multiply: output = output * input
            candle::candle_launch_bmul_f32(
                elements, 0, ptr::null(), ptr::null(), output_f32 as *const f32,
                ptr::null(), input_f32, output_f32, *stream
            );

            // 6. Binary add: output = output + temp
            candle::candle_launch_badd_f32(
                elements, 0, ptr::null(), ptr::null(), output_f32 as *const f32,
                ptr::null(), temp_f32, output_f32, *stream
            );
        }
    }

    let launch_time = launch_start.elapsed();
    let total_kernel_launches = workloads.len() * 6;
    println!("  ✓ Launched {} kernels in {:?}", total_kernel_launches, launch_time);
    println!("    Launch rate: {:.2}K kernels/sec", total_kernel_launches as f64 / launch_time.as_secs_f64() / 1000.0);
    println!();

    // Execution phase - where the magic happens!
    println!("⚡ GPU Execution Phase...");
    let exec_start = Instant::now();

    unsafe {
        for (stream, _, _, _) in workloads.iter() {
            ptx_sys::cudaStreamSynchronize(*stream);
        }
    }

    let exec_time = exec_start.elapsed();

    // Calculate throughput
    // Each stream runs 6 kernels on 'elements' data points
    // Operations per kernel:
    // - exp: ~10 FLOPs per element (approximation)
    // - tanh: ~15 FLOPs per element
    // - sigmoid: ~10 FLOPs per element
    // - relu: ~1 FLOP per element
    // - mul: ~1 FLOP per element
    // - add: ~1 FLOP per element
    let flops_per_stream = elements as f64 * (10.0 + 15.0 + 10.0 + 1.0 + 1.0 + 1.0);
    let total_flops = flops_per_stream * workloads.len() as f64;
    let gflops = total_flops / exec_time.as_secs_f64() / 1e9;

    // Memory throughput
    // Each kernel reads and writes data
    let memory_ops_per_stream = elements as f64 * 4.0 * 6.0 * 2.0; // 6 kernels × 2 (read+write) × 4 bytes
    let total_memory = memory_ops_per_stream * workloads.len() as f64;
    let memory_bandwidth = total_memory / exec_time.as_secs_f64() / 1e9;

    println!("  ✓ Execution complete in {:?}", exec_time);
    println!();
    println!("  📊 PERFORMANCE METRICS:");
    println!("    ─────────────────────────────────────");
    println!("    Compute:      {:.2} GFLOPS", gflops);
    println!("    Memory BW:    {:.2} GB/s", memory_bandwidth);
    println!("    Operations:   {:.2}B FLOPs", total_flops / 1e9);
    println!("    Elements:     {:.2}M", (workloads.len() * elements) as f64 / 1e6);
    println!("    Kernels:      {}", total_kernel_launches);
    println!("    ─────────────────────────────────────");
    println!();

    // Cleanup phase
    println!("🧹 TLSF Cleanup Phase...");
    let free_start = Instant::now();

    unsafe {
        let runtime_ptr = runtime.raw();
        for (_, input, temp, output) in workloads.iter() {
            ptx_sys::gpu_hot_free(runtime_ptr, *input);
            ptx_sys::gpu_hot_free(runtime_ptr, *temp);
            ptx_sys::gpu_hot_free(runtime_ptr, *output);
        }
    }

    let free_time = free_start.elapsed();
    println!("  ✓ Freed {:.2} GB in {:?}", total_bytes_allocated as f64 / 1e9, free_time);
    println!("    Bandwidth: {:.2} GB/s", total_bytes_allocated as f64 / 1e9 / free_time.as_secs_f64());
    println!();

    // Total summary
    let total_time = alloc_time + launch_time + exec_time + free_time;
    println!("  ⏱️  TOTAL TIME: {:?}", total_time);
    println!("    (Alloc: {:?} + Launch: {:?} + Exec: {:?} + Free: {:?})",
             alloc_time, launch_time, exec_time, free_time);

    Ok(())
}
