# Install Guide

## Quick Start

```bash
./install.sh
```

If SM auto-detect fails:

```bash
./install.sh --sm 86
```

Blackwell GPUs:

```bash
./install.sh --sm 100   # B100/B200
./install.sh --sm 120   # GB200
```

## Requirements

- Linux (x86_64 or aarch64)
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
- `LIBTORCH_VERSION`: default `2.9.0`
- `LIBTORCH_CUDA_TAG`: auto-detected from nvcc (e.g. `cu126`)
- `LIBTORCH_URL`: custom download URL
- `TORCH_CPYTHON_TAG`: default `cp311` (aarch64 wheel selection only)

## aarch64 (Grace Blackwell / Grace Hopper / Jetson)

On aarch64, the installer downloads the PyTorch wheel and extracts C++ libraries
from it (no Python runtime needed). This works automatically:

```bash
./install.sh --sm 100   # Grace Blackwell GB200
```

To override, set `LIBTORCH` to an existing libtorch root or use `LIBTORCH_URL`
to point at a custom archive.

## Verify Manually

```bash
cd ferrite-gpu-lang
cargo run --release --example script_runtime
cargo run --release --features torch --example script_cv_detect
cd ../external/ferrite-xla
cargo run --release --example xla_allocator_test
```
