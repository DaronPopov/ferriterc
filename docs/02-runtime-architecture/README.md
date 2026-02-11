# Runtime Architecture

## Layers

1. CUDA/C++ layer
   - `ferrite-os/core/` (`.cu`, `.c`, headers)
   - Produces runtime shared objects in `ferrite-os/lib/`
2. Rust FFI/system layer
   - `ferrite-os/internal/ptx-sys`
   - Raw bindings and link configuration
3. Runtime services layer
   - `ferrite-os/ptx-runtime`
   - Memory/stream/device/runtime abstractions
4. Language/script layer
   - `ferrite-gpu-lang`
   - User-facing script execution and integrations

## Memory Model

- TLSF allocator runtime is built in the CUDA layer and linked by higher layers.
- Rust crates consume the allocator through FFI bindings and runtime APIs.
- libtorch integrations use this runtime via bridge crates in `external/`.

## Execution Model

- Scripts are run as Rust examples in `ferrite-gpu-lang/examples`.
- `ferrite-run` symlinks arbitrary `.rs` scripts into the examples directory and runs them with selected feature flags.
- Torch-dependent scripts run with `torch` + resolved CUDA feature.

## Daemon Model

- Daemon crate: `ferrite-os/internal/ptx-daemon`
- Binaries: `ferrite-daemon` and `ferrite`
- Optional systemd installation via `install.sh --enable-service`

## Critical Runtime Boundaries

- Build boundary:
  `ferrite-os` native artifacts must exist before Rust runtime layers can link.
- ABI boundary:
  `ptx-sys` binds Rust to native CUDA runtime artifacts.
- Feature boundary:
  CUDA-family features must match resolved compatibility selection for Torch integrations.

## Failure Surfaces

- Missing/incompatible CUDA toolkit or driver.
- Missing shared libs in `LD_LIBRARY_PATH`.
- Feature mismatch between selected CUDA family and linked crates.
- Daemon service path/environment misconfiguration.

## Runbook

- Runtime operations: `runbooks/runtime-operations.md`
