use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=kernels/");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Detect CUDA compute capability from environment or default to sm_80
    let cuda_arch = env::var("CUDA_ARCH").unwrap_or_else(|_| "sm_75".to_string());

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
        "candle/scan_f32.cu",         // F32-only prefix scan (cumsum)
        "candle/gather_f32.cu",       // F32-only gather/scatter
        "candle/topk_f32.cu",         // F32-only top-k selection
        "candle/indexing.cu",         // Candle index_select, gather, scatter, scatter_add
        "candle/indexing_launchers.cu", // Launchers for indexing kernels
        "candle/sort.cu",             // Candle bitonic argsort
        "candle/sort_launchers.cu",   // Launchers for argsort kernels
        "candle/ternary.cu",          // Candle where conditional
        "candle/where_launchers.cu",  // Launchers for where kernel
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
    if let Some(cuda_path) = env::var("CUDA_PATH")
        .ok()
        .or_else(|| env::var("CUDA_HOME").ok())
        .map(PathBuf::from)
        .or_else(find_cuda_kernels)
    {
        let cuda_lib = if cuda_path.join("lib64").exists() {
            cuda_path.join("lib64")
        } else {
            cuda_path.join("lib")
        };
        println!("cargo:rustc-link-search=native={}", cuda_lib.display());
    } else {
        println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
        #[cfg(target_arch = "x86_64")]
        println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu");
        #[cfg(target_arch = "aarch64")]
        println!("cargo:rustc-link-search=native=/usr/lib/aarch64-linux-gnu");
    }

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
}

fn find_cuda_kernels() -> Option<PathBuf> {
    let common = [
        "/usr/local/cuda", "/usr/local/cuda-12.9", "/usr/local/cuda-12.8",
        "/usr/local/cuda-12.6", "/usr/local/cuda-12",
        "/opt/cuda", "/usr/lib/cuda",
    ];
    for p in &common {
        let p = PathBuf::from(p);
        if p.join("include/cuda_runtime.h").exists() { return Some(p); }
    }
    if let Ok(out) = Command::new("which").arg("nvcc").output() {
        if out.status.success() {
            let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if let Some(cuda) = PathBuf::from(&s).parent().and_then(|p| p.parent()) {
                if cuda.join("include/cuda_runtime.h").exists() {
                    return Some(cuda.to_path_buf());
                }
            }
        }
    }
    None
}
