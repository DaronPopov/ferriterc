// Build script to compile C++ XLA allocator shim

fn main() {
    println!("cargo:rerun-if-changed=cpp/xla_tlsf_allocator.cpp");
    println!("cargo:rerun-if-changed=cpp/xla_tlsf_allocator.h");

    cc::Build::new()
        .cpp(true)
        .file("cpp/xla_tlsf_allocator.cpp")
        .flag("-std=c++17")
        .flag("-O3")
        .compile("xla_tlsf_allocator");

    println!("cargo:rustc-link-lib=static=xla_tlsf_allocator");
    println!("cargo:rustc-link-lib=stdc++");
}
