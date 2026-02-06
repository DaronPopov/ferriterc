# Install Guide

## Quick Start

```bash
./install.sh
```

If SM auto-detect fails:

```bash
./install.sh --sm 86
```

## Requirements

- Linux (x86_64 recommended)
- NVIDIA GPU + driver
- CUDA toolkit (`nvcc` in `PATH`)
- Rust toolchain (`cargo`)
- `make`, `gcc`

## Behavior

`install.sh` is self-contained in this repo folder:

- builds runtime/libs locally
- resolves/provisions libtorch automatically
- builds Torch + XLA integrations
- validates runtime script execution

No global project files are required outside this directory.

## Optional Environment

- `SM` / `CUDA_SM` / `GPU_SM`: force compute capability
- `LIBTORCH`: explicit libtorch root
- `LIBTORCH_VERSION`: default `2.3.0`
- `LIBTORCH_CUDA_TAG`: default `cu121`
- `LIBTORCH_URL`: custom download URL

## Verify Manually

```bash
cd ferrite-gpu-lang
cargo run --release --example script_runtime
cargo run --release --features torch --example script_cv_detect
cd ../external/ferrite-xla
cargo run --release --example xla_allocator_test
```
