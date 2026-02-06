use ferrite_xla::{
    get_tlsf_stats, init_xla_allocator, init_xla_cpp_allocator, xla_cpp_alloc, xla_cpp_free,
    xla_cpp_print_stats,
};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== XLA TLSF Allocator Test ===\n");

    init_xla_allocator(0)?;
    println!("✓ Rust TLSF runtime initialized");

    // Initialize and exercise the C++ shim that XLA/JAX would call.
    init_xla_cpp_allocator(0);
    println!("✓ C++ XLA allocator shim initialized\n");

    let size = 1024 * 1024; // 1MB
    let total_pool = get_tlsf_stats()
        .map(|s| s.total_pool_size)
        .unwrap_or(2 * 1024 * 1024 * 1024);
    let target_bytes = (total_pool as f64 * 0.25) as usize;
    let rounds = (target_bytes / size).clamp(128, 4096);

    let mut ptrs = Vec::with_capacity(rounds);
    let start = Instant::now();
    for _ in 0..rounds {
        let p = xla_cpp_alloc(size, 256);
        if p.is_null() {
            return Err("xla_cpp_alloc returned null".into());
        }
        ptrs.push(p);
    }
    let alloc_elapsed = start.elapsed();

    let start = Instant::now();
    for p in ptrs.drain(..) {
        xla_cpp_free(p, size);
    }
    let free_elapsed = start.elapsed();

    println!("✓ Allocated {} buffers of 1MB", rounds);
    println!(
        "  Alloc avg: {:.2}us  Free avg: {:.2}us",
        alloc_elapsed.as_micros() as f64 / rounds as f64,
        free_elapsed.as_micros() as f64 / rounds as f64
    );

    if let Some(stats) = get_tlsf_stats() {
        println!("\nTLSF Stats:");
        println!("  Pool: {:.2} GB", stats.total_pool_size as f64 / 1e9);
        println!("  Allocated: {:.2} MB", stats.allocated_bytes as f64 / 1e6);
        println!("  Peak: {:.2} MB", stats.peak_allocated as f64 / 1e6);
        println!("  Fragmentation: {:.6}", stats.fragmentation_ratio);
        println!("  Utilization: {:.2}%", stats.utilization_percent);
    }

    xla_cpp_print_stats();
    println!("\n✅ XLA allocator path is active on PTX TLSF");

    Ok(())
}
