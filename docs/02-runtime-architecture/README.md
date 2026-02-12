# 02.00 Runtime Architecture

## Layers

1. CUDA/C++ layer
   - `ferrite-os/native/core/` (`.cu`, `.c`, headers)
   - Produces runtime shared objects in `ferrite-os/lib/`
2. Rust FFI/system layer
   - `ferrite-os/crates/internal/ptx-sys`
   - Raw bindings and link configuration
3. Runtime services layer
   - `ferrite-os/crates/public/ptx-runtime`
   - Memory/stream/device/runtime abstractions
4. Language/script layer
   - `ferrite-gpu-lang`
   - User-facing script execution and integrations

## Memory Model

- TLSF allocator runtime is built in the CUDA layer and linked by higher layers.
- Rust crates consume the allocator through FFI bindings and runtime APIs.
- libtorch integrations use this runtime via bridge crates in `external/`.

## IPC Model

- Shared memory key is per-UID (`/ptx_os_{uid}_v1`), preventing collisions between users on shared machines.
- Key is constructed at runtime using `GPU_HOT_IPC_KEY_PREFIX` + `getuid()` + `GPU_HOT_IPC_KEY_SUFFIX` (see `hot_runtime_init.inl`).
- Constants defined in `ferrite-os/native/core/include/gpu/gpu_hot_runtime.h`.

## Execution Model

- Scripts are run as Rust examples in `ferrite-gpu-lang/examples`.
- `ferrite-run` symlinks arbitrary `.rs` scripts into the examples directory and runs them with selected feature flags.
- Torch-dependent scripts run with `torch` + resolved CUDA feature.

## Daemon Model

- Daemon crate: `ferrite-os/crates/internal/ptx-daemon`
- Primary CLI entrypoint: `ferrite-daemon` (root wrapper / installed command)
- Binary artifacts: `ferrite-os/target/release/ferrite-daemon` and `ferrite-os/target/release/ferrite`
- Optional systemd installation via `install.sh --enable-service`

## Critical Runtime Boundaries

- Build boundary:
  `ferrite-os` native artifacts must exist before Rust runtime layers can link.
  `ptx-sys` build script panics with actionable message if `libptx_os.so` is not found — enforces that `make all` runs first.
- ABI boundary:
  `ptx-sys` binds Rust to native CUDA runtime artifacts.
- Feature boundary:
  CUDA-family features must match resolved compatibility selection for Torch integrations.

## Failure Surfaces

- Missing/incompatible CUDA toolkit or driver.
- Missing shared libs in `LD_LIBRARY_PATH`.
- Feature mismatch between selected CUDA family and linked crates.
- Daemon service path/environment misconfiguration.
- Build with wrong default SM (`ptx-kernels` defaults to `sm_75` if `CUDA_ARCH` is unset).

## Runbook

- `02.01`: Runtime operations: `runbooks/runtime-operations.md`
