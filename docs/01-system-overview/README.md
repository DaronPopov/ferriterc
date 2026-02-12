# 01.00 System Overview

## Purpose

`ferriterc` is a Rust/CUDA runtime stack that builds from source and links to external CUDA/libtorch binaries resolved at install time.

## Repository Map

- `ferrite-os/`
  Core runtime workspace with low-level CUDA integration and daemon binaries.
- `ferrite-gpu-lang/`
  Script/runtime layer on top of `ferrite-os`.
- `external/`
  Integrations and forks used by the runtime (`aten-ptx`, `cudarc-ptx`, `ferrite-torch`, `ferrite-xla`).
- `ferrite-os/workloads/finetune_engine/`
  Training/fine-tuning scripts and orchestration examples.
- `ferrite-os/workloads/mathematics_engine/`
  Quantitative compute scripts and examples.
- `docs/`
  This documentation set.

## Primary Runtime Path

1. Install flow resolves machine-specific compatibility and dependencies.
2. CUDA toolkit is auto-installed if `nvcc` is absent (only NVIDIA driver required pre-installed).
3. CUDA runtime libraries are set up. External libtorch is provisioned unless `--core-only` mode is used.
4. `ferrite-os` and dependent Rust crates are built (3 steps core-only, 9 steps full).
5. Scripts run via `ferrite-run` or daemon APIs.

## Key Interfaces

- Install/build entrypoint: `install.sh`
- Script execution entrypoint: `ferrite-run`
- Daemon execution entrypoint: `ferrite-daemon` (installed command / root wrapper)
- Compatibility resolver: `scripts/resolve_cuda_compat.sh`
- Daemon binary artifact: `ferrite-os/target/release/ferrite-daemon`

## Invariants

- The repo ships source; libtorch fetch is optional (skipped with `--core-only`).
- Only the NVIDIA driver is required pre-installed; CUDA toolkit is auto-installed by the installer if absent.
- Compatibility selection is CUDA-version driven and controlled by `compat.toml`.
- Python torch installation is not required.

## Subsystem Contracts

- `01.01`: Contract index: `contracts/README.md`
- `01.02`: Subsystem A: `contracts/subsystem-a-tui.md`
- `01.03`: Subsystem B: `contracts/subsystem-b-cuda-os-layer.md`
- `01.04`: Subsystem C: `contracts/subsystem-c-rust-runtime-core.md`
- `01.05`: Subsystem D: `contracts/subsystem-d-installer-provisioner.md`
- `01.06`: Subsystem E: `contracts/subsystem-e-external-integrations.md`
- `01.07`: Subsystem F: `contracts/subsystem-f-daemon-service-lifecycle.md`
- `01.08`: Subsystem G: `contracts/subsystem-g-diagnostics-error-contracts.md`
