// Build script to compile C++ ORT allocator shim

fn main() {
    println!("cargo:rerun-if-changed=cpp/ort_tlsf_allocator.cpp");
    println!("cargo:rerun-if-changed=cpp/ort_tlsf_allocator.h");

    cc::Build::new()
        .cpp(true)
        .file("cpp/ort_tlsf_allocator.cpp")
        .flag("-std=c++17")
        .flag("-O3")
        .compile("ort_tlsf_allocator");

    println!("cargo:rustc-link-lib=static=ort_tlsf_allocator");
    println!("cargo:rustc-link-lib=stdc++");
}
