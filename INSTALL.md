# Ferrite Install (Simple)

Repository:
`https://github.com/DaronPopov/ferriterc`

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

## 3) Install (Pinned Exact Versions/Links)

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

## 4) Optional: Start on Boot

```bash
./install.sh --enable-service
```

## 5) Run

```bash
./ferrite-run finetune_engine/scripting_finetune.rs
```

## Notes

- Linux only (`x86_64` / `aarch64`)
- Requires NVIDIA driver + CUDA toolkit (`nvcc` present)
- Installer auto-handles host build tooling, Rust toolchain, CUPTI, and libtorch provisioning
- No precompiled Ferrite binaries are fetched; installer builds this repository from source
- No Python torch install is required (aarch64 path extracts C++ libtorch artifacts only)
