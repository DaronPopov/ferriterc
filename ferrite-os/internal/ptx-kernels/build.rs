use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=kernels/");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Detect CUDA compute capability from environment or default to sm_80
    let cuda_arch = env::var("CUDA_ARCH").unwrap_or_else(|_| "sm_80".to_string());

    println!("cargo:rerun-if-env-changed=CUDA_ARCH");

    // Build Candle kernels
    let mut build = cc::Build::new();
    build
        .cuda(true)
        .flag("-cudart=shared")
        .flag(&format!("-arch={}", cuda_arch))
        .flag("-std=c++17")
        .flag("--expt-relaxed-constexpr")
        .flag("-Xcompiler=-fPIC")
        .flag("--relocatable-device-code=true")  // Enable RDC for multi-file CUDA projects
        .opt_level(3)
        .include("kernels/candle");

    // Add test kernels (simple, always work)
    let kernel_files = [
        "test_kernels.cu",
        "sha256_kernel.cu",
    ];

    // Candle kernels - now enabled (F32 only for initial testing)
    let candle_files = [
        "candle/launcher_simple.cu",  // Simplified launcher with F32 only
        "candle/binary_f32.cu",       // F32-only binary ops
        "candle/unary_f32.cu",        // F32-only unary ops
    ];

    for file in &kernel_files {
        let path = format!("kernels/{}", file);
        build.file(&path);
    }

    for file in &candle_files {
        let path = format!("kernels/{}", file);
        build.file(&path);
    }

    build.compile("candle_kernels");

    // Link against CUDA runtime
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cuda");

    // Add CUDA library search paths
    if let Ok(cuda_path) = env::var("CUDA_PATH") {
        println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
    } else {
        // Common CUDA installation paths
        println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");

        // Architecture-specific system library paths
        #[cfg(target_arch = "x86_64")]
        println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu");

        #[cfg(target_arch = "aarch64")]
        println!("cargo:rustc-link-search=native=/usr/lib/aarch64-linux-gnu");
    }

    println!("cargo:rustc-link-search=native={}", out_dir.display());
}
