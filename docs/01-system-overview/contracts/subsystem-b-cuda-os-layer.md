# Subsystem B Contract: CUDA OS Layer

## Purpose
Own native CUDA runtime, allocator, kernels, and OS hooks with stable ABI behavior.

## Owned Paths
- `ferrite-os/native/core/runtime/**`
- `ferrite-os/native/core/kernels/**`
- `ferrite-os/native/core/memory/**`
- `ferrite-os/native/core/os/**`
- `ferrite-os/native/core/hooks/**`

## Public Interfaces
- C/CUDA ABI exported by `ferrite-os/lib/libptx_os.so`
- Hook ABI via `ferrite-os/lib/libptx_hook.so`
- Headers under `ferrite-os/native/core/include/**`
- IPC key constants in `gpu_hot_runtime.h`: `GPU_HOT_IPC_KEY_PREFIX` (`/ptx_os_`), `GPU_HOT_IPC_KEY_SUFFIX` (`_v1`), `GPU_HOT_IPC_KEY_MAX_LEN` (64)

## Forbidden Cross-Dependencies
- No dependency on daemon TUI modules
- No dependency on installer CLI parsing/pinning logic

## No-Break Rules
- Preserve FFI symbols and semantics consumed by Rust crates
- Preserve stream/memory lifecycle behavior and fallback paths
- IPC shared memory key is per-UID (`/ptx_os_{uid}_v1`) to prevent collisions on multi-user systems
