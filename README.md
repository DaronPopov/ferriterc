# ferriterc

Self-contained GPU compute runtime with O(1) TLSF memory management,
shard-streamed execution, and wave-scheduled CUDA streams.

## One-Line Install

```bash
./install.sh
```

Optional explicit SM:

```bash
./install.sh --sm 86
```

## What `install.sh` does

1. Builds `ferrite-os` (TLSF allocator, stream pool, IPC)
2. Builds `ferrite-gpu-lang` with Torch support
3. Builds `external/ferrite-torch` examples
4. Builds `external/ferrite-xla` backend example
5. Validates torch runtime scripts
6. Checks `finetune_engine/` scripts compile
7. Checks `mathematics_engine/` scripts compile

## Running Scripts

```bash
./ferrite-run finetune_engine/scripting_finetune.rs
./ferrite-run mathematics_engine/monte_carlo/path_pricer.rs -- --paths 10000000
```

`ferrite-run` auto-detects the `--torch` feature from import statements.

## Components

```
ferriterc/
  ferrite-os/            GPU OS runtime (TLSF allocator, 100K streams, IPC)
  ferrite-gpu-lang/      Rust GPU scripting layer
  external/
    aten-ptx/            PyTorch ATen TLSF allocator bridge
    cudarc-ptx/          CUDA driver abstraction
    ferrite-torch/       Torch integration examples
    ferrite-xla/         XLA backend integration
  finetune_engine/       ML fine-tuning control plane
    scripting_finetune   Shard-streamed LoRA fine-tuning
    checkpoint/          Adapter checkpoint save/load
    loader/              Safetensors shard loader
    eval/                Validation loop
    scheduler/           LR schedules
    merge/               LoRA adapter merge
    quantize/            Streaming quantization (f16/bf16/int8/nf4)
    dataset/             Packed sequence batching
    telemetry/           Training metrics + divergence detection
    distributed/         Multi-GPU wave scheduling
    architectures/       Novel architectures (streaming MoE)
  mathematics_engine/    Quantitative finance / large-scale computation
    monte_carlo/         50M+ path Monte Carlo option pricing
    portfolio/           Streaming covariance for 10K+ asset universes
    risk/                VaR/CVaR via millions of streamed scenarios
    pde/                 Finite difference PDE solver (Black-Scholes)
    matrix/              Block-Cholesky, LU, eigenvalue decomposition
    greeks/              Full Greeks surface for 100K+ instrument books
```

## LibTorch Provisioning

Installer resolves libtorch in this order:

1. `LIBTORCH` env path
2. `external/libtorch`
3. Auto-download libtorch (CUDA tag auto-detected from local `nvcc`)
4. Optional Python torch fallback (`LIBTORCH_ALLOW_PYTORCH=1`)

Auto-download controls:
- `LIBTORCH_VERSION` (default `2.3.0`)
- `LIBTORCH_CUDA_TAG` (default `cu121`)
- `LIBTORCH_URL` (full override)
- `LIBTORCH_ALLOW_PYTORCH` (default `0`)
