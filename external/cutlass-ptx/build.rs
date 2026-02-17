use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=cpp/");
    println!("cargo:rerun-if-env-changed=CUDA_ARCH");

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let cutlass_include = manifest_dir.join("cutlass/include");

    assert!(
        cutlass_include.join("cutlass/gemm/device/gemm.h").exists(),
        "CUTLASS headers not found at {}. Run: git clone --depth 1 --branch v3.7.0 \
         https://github.com/NVIDIA/cutlass.git cutlass",
        cutlass_include.display()
    );

    let cuda_arch = env::var("CUDA_ARCH").unwrap_or_else(|_| "sm_87".to_string());

    cc::Build::new()
        .cuda(true)
        .file("cpp/cutlass_gemm.cu")
        .flag("-std=c++17")
        .flag(&format!("-arch={}", cuda_arch))
        .flag("-O3")
        .flag("--expt-relaxed-constexpr")
        .flag("-Xcompiler=-fPIC")
        .flag("-Xcompiler=-Wno-unused-parameter")
        .include(&cutlass_include)
        .include(cutlass_include.join("cute"))
        .compile("cutlass_ptx_kernels");

    println!("cargo:rustc-link-lib=static=cutlass_ptx_kernels");
    println!("cargo:rustc-link-lib=stdc++");

    // CUDA runtime linkage
    println!("cargo:rustc-link-lib=cudart");
    if let Ok(cuda_path) = env::var("CUDA_HOME") {
        let lib_dir = if PathBuf::from(&cuda_path).join("lib64").exists() {
            format!("{}/lib64", cuda_path)
        } else {
            format!("{}/lib", cuda_path)
        };
        println!("cargo:rustc-link-search=native={}", lib_dir);
    } else {
        println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    }
}
