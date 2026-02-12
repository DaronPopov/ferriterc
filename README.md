# ferriterc

`ferriterc` is a CUDA-focused runtime and tooling stack built in Rust/CUDA.
It builds from source in this repository and links to external libtorch binaries.

## Install

```bash
git clone https://github.com/DaronPopov/ferriterc.git
cd ferriterc
./install.sh
```

## Requirements

- Linux (`x86_64` or `aarch64`)
- NVIDIA GPU driver
- CUDA toolkit (`nvcc` available in `PATH`)

`install.sh` handles:
- host build tools (when missing, via distro package manager)
- Rust toolchain bootstrap (when missing)
- CUPTI installation attempt (Linux package managers)
- libtorch provisioning and extraction
- source builds only (no precompiled ferrite binaries)

No Python torch install is required.

## Install Options

Set SM explicitly:

```bash
./install.sh --sm 86
```

Enable boot-time daemon service:

```bash
./install.sh --enable-service
```

Pin exact external artifacts/features:

```bash
./install.sh --pins "sm=89,libtorch_url=https://download.pytorch.org/libtorch/cu126/libtorch-shared-with-deps-2.9.0%2Bcu126.zip,libtorch_tag=cu126,cudarc_feature=cuda-12060"
```

Equivalent explicit flags:

```bash
./install.sh --sm 89 \
  --libtorch-url "https://download.pytorch.org/libtorch/cu126/libtorch-shared-with-deps-2.9.0%2Bcu126.zip" \
  --libtorch-tag cu126 \
  --cudarc-feature cuda-12060
```

CUDA compatibility is selected from `compat.toml` based on detected toolkit version.

## What `install.sh` does

1. Builds `ferrite-os` (TLSF allocator, stream pool, IPC)
2. Builds `ferrite-gpu-lang` with Torch support
3. Builds `external/ferrite-torch` examples
4. Builds `external/ferrite-xla` backend example
5. Validates torch runtime scripts
6. Checks `ferrite-os/workloads/finetune_engine/` scripts compile
7. Checks `ferrite-os/workloads/mathematics_engine/` scripts compile

## Running Scripts

```bash
./ferrite-run ferrite-os/workloads/finetune_engine/scripting_finetune.rs
./ferrite-run ferrite-os/workloads/mathematics_engine/monte_carlo/path_pricer.rs -- --paths 10000000
```

`ferrite-run` auto-detects the `--torch` feature from import statements.

Legacy top-level paths (`finetune_engine/`, `mathematics_engine/`) are kept as compatibility symlinks.

## Running Daemon

```bash
./ferrite-daemon serve
./ferrite-daemon ping
./ferrite-daemon run-list
```

`ferrite-daemon` is a root-level wrapper that sets runtime/libtorch library paths and launches the daemon binary.

## Components

```
ferriterc/
  ferrite-daemon         Root daemon launcher wrapper
  ferrite-os/            GPU runtime core (TLSF allocator, stream/runtime plumbing, IPC)
  ferrite-gpu-lang/      Rust GPU scripting layer
  external/
    aten-ptx/            PyTorch ATen TLSF allocator bridge
    cudarc-ptx/          CUDA driver abstraction
    ferrite-torch/       Torch integration examples
    ferrite-xla/         XLA backend integration
  ferrite-os/workloads/finetune_engine/  ML fine-tuning control plane
    scripting_finetune   LoRA fine-tuning script entrypoint
    checkpoint/          Adapter checkpoint save/load
    loader/              Safetensors shard loader
    eval/                Validation loop
    scheduler/           LR schedules
    merge/               LoRA adapter merge
    quantize/            Streaming quantization (f16/bf16/int8/nf4)
    dataset/             Packed sequence batching
    telemetry/           Training metrics + divergence detection
    distributed/         Multi-GPU wave scheduling
    architectures/       Model architecture experiments
  ferrite-os/workloads/mathematics_engine/  Quantitative finance compute modules
    monte_carlo/         Monte Carlo pricing
    portfolio/           Covariance and portfolio analytics
    risk/                VaR/CVaR workflows
    pde/                 Finite difference PDE solver (Black-Scholes)
    matrix/              Matrix decomposition/linear algebra routines
    greeks/              Greeks computation workflows
```

## LibTorch Provisioning

Installer resolves libtorch in this order:

1. `LIBTORCH` env path
2. `external/libtorch`
3. Auto-download (x86_64: libtorch zip, aarch64: torch wheel → C++ lib extraction)

Auto-download controls:
- `LIBTORCH_VERSION` (default `2.9.0`)
- `LIBTORCH_CUDA_TAG` (auto-selected from `compat.toml`, or override manually)
- `CUDARC_CUDA_FEATURE` (auto-selected from `compat.toml`, or override manually)
- `LIBTORCH_URL` (full override)
