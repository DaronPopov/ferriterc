# Subsystem B Contract: CUDA OS Layer

## Purpose
Own native CUDA runtime, allocator, kernels, and OS hooks with stable ABI behavior.

## Owned Paths
- `ferrite-os/core/runtime/**`
- `ferrite-os/core/kernels/**`
- `ferrite-os/core/memory/**`
- `ferrite-os/core/os/**`
- `ferrite-os/core/hooks/**`

## Public Interfaces
- C/CUDA ABI exported by `ferrite-os/lib/libptx_os.so`
- Hook ABI via `ferrite-os/lib/libptx_hook.so`
- Headers under `ferrite-os/core/include/**`

## Forbidden Cross-Dependencies
- No dependency on daemon TUI modules
- No dependency on installer CLI parsing/pinning logic

## No-Break Rules
- Preserve FFI symbols and semantics consumed by Rust crates
- Preserve stream/memory lifecycle behavior and fallback paths
