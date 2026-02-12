# Ferrite Install (Simple)

Repository:
`https://github.com/DaronPopov/ferriterc`

## Prerequisites

- Linux (`x86_64` / `aarch64`)
- NVIDIA driver installed
- CUDA toolkit pre-installed (`nvcc` available in `PATH`, or set `CUDA_PATH`)
- Installer auto-provisions the rest: host build tools, Rust toolchain, CUPTI, libtorch, and onnxruntime

## 1) Clone

```bash
git clone https://github.com/DaronPopov/ferriterc.git
cd ferriterc
```

## 2) Install (Auto)

```bash
./install.sh
```

If auto SM detection fails:

```bash
./install.sh --sm 86
```

That is enough for most machines.

## 3) Core-Only Install

Build only the core OS (daemon, TUI, ptx-runtime) without downloading libtorch
or building torch-dependent crates. Useful for headless GPU servers or CI:

```bash
./install.sh --core-only
```

```bash
./install.sh --core-only --sm 86
```

## 4) Install (Pinned Exact Versions/Links)

Use this when you want full manual control of external binaries.

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

## 5) Optional: Start on Boot

```bash
./install.sh --enable-service
```

## 6) Run

```bash
./ferrite-run finetune_engine/scripting_finetune.rs
```

## Notes

- CUDA toolkit is user-managed by default (installer expects `nvcc` to exist)
- If you want legacy behavior, use `./install.sh --auto-install-cuda`
- Installer auto-handles host build tooling, Rust toolchain, CUPTI, libtorch, onnxruntime, and external integration dependency provisioning (aten/candle/xla)
- No precompiled Ferrite binaries are fetched; installer builds this repository from source
- No Python torch install is required (aarch64 path extracts C++ libtorch artifacts only)
- Use `--core-only` to skip the ~2GB libtorch download when you only need the core runtime
