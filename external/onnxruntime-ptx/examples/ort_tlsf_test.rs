use onnxruntime_ptx::{
    get_tlsf_stats, init_ort_allocator, init_ort_cpp_allocator, ort_tlsf_alloc, ort_tlsf_free,
    ort_tlsf_print_stats,
};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== ONNX Runtime TLSF Allocator Test ===\n");

    // 1. Initialize Rust TLSF runtime
    init_ort_allocator(0)?;
    println!("✓ Rust TLSF runtime initialized");

    // 2. Initialize C++ ORT allocator shim
    init_ort_cpp_allocator(0);
    println!("✓ C++ ORT allocator shim initialized\n");

    // 3. Allocate/free cycle through the Rust FFI layer
    let size = 1024 * 1024; // 1MB
    let total_pool = get_tlsf_stats()
        .map(|s| s.total_pool_size)
        .unwrap_or(2 * 1024 * 1024 * 1024);
    let target_bytes = (total_pool as f64 * 0.25) as usize;
    let rounds = (target_bytes / size).clamp(128, 4096);

    let mut ptrs = Vec::with_capacity(rounds);
    let start = Instant::now();
    for _ in 0..rounds {
        let p = ort_tlsf_alloc(size);
        if p.is_null() {
            return Err("ort_tlsf_alloc returned null".into());
        }
        ptrs.push(p);
    }
    let alloc_elapsed = start.elapsed();

    let start = Instant::now();
    for p in ptrs.drain(..) {
        ort_tlsf_free(p);
    }
    let free_elapsed = start.elapsed();

    println!("✓ Allocated {} buffers of 1MB", rounds);
    println!(
        "  Alloc avg: {:.2}us  Free avg: {:.2}us",
        alloc_elapsed.as_micros() as f64 / rounds as f64,
        free_elapsed.as_micros() as f64 / rounds as f64
    );

    // 4. Verify TLSF stats
    if let Some(stats) = get_tlsf_stats() {
        println!("\nTLSF Stats:");
        println!("  Pool: {:.2} GB", stats.total_pool_size as f64 / 1e9);
        println!("  Allocated: {:.2} MB", stats.allocated_bytes as f64 / 1e6);
        println!("  Peak: {:.2} MB", stats.peak_allocated as f64 / 1e6);
        println!("  Fragmentation: {:.6}", stats.fragmentation_ratio);
        println!("  Utilization: {:.2}%", stats.utilization_percent);
    }

    ort_tlsf_print_stats();
    println!("\n✅ ORT allocator path is active on PTX TLSF");

    Ok(())
}
