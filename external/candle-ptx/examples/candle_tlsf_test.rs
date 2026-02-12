use candle_ptx::candle_core::{Device, Tensor};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Candle + PTX-OS TLSF Allocator Test ===\n");

    // 1. Initialize TLSF runtime before candle touches the GPU
    candle_ptx::init()?;
    println!("✓ TLSF runtime initialized");

    // 2. Create a candle CUDA device (cudarc's malloc_sync will use TLSF)
    let device = Device::new_cuda(0)?;
    println!("✓ Candle CUDA device created\n");

    // 3. Allocate tensors and do some compute
    let start = Instant::now();
    let a = Tensor::randn(0f32, 1.0, (1024, 1024), &device)?;
    let b = Tensor::randn(0f32, 1.0, (1024, 1024), &device)?;
    let alloc_elapsed = start.elapsed();

    let start = Instant::now();
    let c = a.matmul(&b)?;
    let matmul_elapsed = start.elapsed();

    let start = Instant::now();
    let d = (&c + &c)?;
    let _e = d.exp()?;
    let ops_elapsed = start.elapsed();

    println!("✓ Tensor operations complete:");
    println!("  Alloc 2x [1024,1024]: {:?}", alloc_elapsed);
    println!("  Matmul:               {:?}", matmul_elapsed);
    println!("  Add + Exp:            {:?}", ops_elapsed);

    // 4. Verify TLSF is active by checking stats
    if let Some(stats) = candle_ptx::get_tlsf_stats() {
        println!("\nTLSF Stats:");
        println!("  Pool:          {:.2} GB", stats.total_pool_size as f64 / 1e9);
        println!("  Allocated:     {:.2} MB", stats.allocated_bytes as f64 / 1e6);
        println!("  Peak:          {:.2} MB", stats.peak_allocated as f64 / 1e6);
        println!("  Fragmentation: {:.6}", stats.fragmentation_ratio);
        println!("  Utilization:   {:.2}%", stats.utilization_percent);

        assert!(
            stats.allocated_bytes > 0,
            "TLSF allocated_bytes should be > 0 (proves TLSF is routing allocations)"
        );
        println!("\n✅ Candle GPU allocations are routed through PTX-OS TLSF");
    } else {
        return Err("TLSF stats not available — runtime may not be initialized".into());
    }

    candle_ptx::print_tlsf_stats();

    Ok(())
}
