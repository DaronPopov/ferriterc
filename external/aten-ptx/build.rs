// Build script to compile C++ PyTorch allocator

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn looks_like_libtorch(root: &Path) -> bool {
    (root.join("include/torch/torch.h").exists()
        || root
            .join("include/torch/csrc/api/include/torch/torch.h")
            .exists())
        && (root.join("lib/libc10_cuda.so").exists() || root.join("lib/libtorch_cuda.so").exists())
}

fn detect_python_torch_root() -> Option<PathBuf> {
    let output = Command::new("python3")
        .args([
            "-c",
            "import os, torch; print(os.path.dirname(torch.__file__)) if getattr(torch.version, 'cuda', None) else exit(1)",
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if s.is_empty() {
        None
    } else {
        Some(PathBuf::from(s))
    }
}

fn resolve_libtorch(manifest_dir: &Path) -> PathBuf {
    if let Ok(p) = env::var("LIBTORCH") {
        let path = PathBuf::from(p);
        if looks_like_libtorch(&path) {
            return path;
        }
    }

    let mut candidates = Vec::new();
    if let Some(py_root) = detect_python_torch_root() {
        candidates.push(py_root);
    }
    if let Some(external_dir) = manifest_dir.parent() {
        candidates.push(external_dir.join("libtorch"));
        if let Some(repo_root) = external_dir.parent() {
            candidates.push(repo_root.join("external/libtorch"));
        }
    }

    for c in candidates {
        if looks_like_libtorch(&c) {
            return c;
        }
    }

    panic!(
        "Could not find libtorch. Set LIBTORCH=/path/to/libtorch (must contain include/torch/torch.h or include/torch/csrc/api/include/torch/torch.h)."
    );
}

fn main() {
    println!("cargo:rerun-if-changed=tlsf_allocator_full.cpp");
    println!("cargo:rerun-if-env-changed=LIBTORCH");
    println!("cargo:rerun-if-env-changed=CUDA_INCLUDE");

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let libtorch = resolve_libtorch(&manifest_dir);
    let required_hdr = libtorch.join("include/c10/cuda/CUDACachingAllocator.h");
    assert!(
        required_hdr.exists(),
        "libtorch at {} is missing c10/cuda/CUDACachingAllocator.h (need CUDA-enabled libtorch)",
        libtorch.display()
    );

    // Find CUDA
    let cuda_include = env::var("CUDA_INCLUDE").unwrap_or_else(|_| "/usr/local/cuda/include".to_string());

    // Compile C++ allocator
    // Use -isystem for libtorch/CUDA headers to suppress upstream warnings
    cc::Build::new()
        .cpp(true)
        .file("tlsf_allocator_full.cpp")
        .flag("-std=c++17")
        .flag("-O3")
        .flag("-Wno-unused-parameter")
        .flag(&format!("-isystem{}/include", libtorch.display()))
        .flag(&format!(
            "-isystem{}/include/torch/csrc/api/include",
            libtorch.display()
        ))
        .flag(&format!("-isystem{}", cuda_include))
        .compile("aten_tlsf_allocator");

    println!("cargo:rustc-link-lib=static=aten_tlsf_allocator");
    println!("cargo:rustc-link-lib=stdc++");

    // Link c10_cuda which contains the allocator symbol
    println!("cargo:rustc-link-search=native={}", libtorch.join("lib").display());
    println!("cargo:rustc-link-lib=dylib=c10_cuda");
}
