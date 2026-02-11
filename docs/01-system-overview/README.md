# System Overview

## Purpose

`ferriterc` is a Rust/CUDA runtime stack that builds from source and links to external CUDA/libtorch binaries resolved at install time.

## Repository Map

- `ferrite-os/`
  Core runtime workspace with low-level CUDA integration and daemon binaries.
- `ferrite-gpu-lang/`
  Script/runtime layer on top of `ferrite-os`.
- `external/`
  Integrations and forks used by the runtime (`aten-ptx`, `cudarc-ptx`, `ferrite-torch`, `ferrite-xla`).
- `finetune_engine/`
  Training/fine-tuning scripts and orchestration examples.
- `mathematics_engine/`
  Quantitative compute scripts and examples.
- `docs/`
  This documentation set.

## Primary Runtime Path

1. Install flow resolves machine-specific compatibility and dependencies.
2. CUDA runtime libraries and external libtorch are provisioned.
3. `ferrite-os` and dependent Rust crates are built.
4. Scripts run via `ferrite-run` or daemon APIs.

## Key Interfaces

- Install/build entrypoint: `install.sh`
- Script execution entrypoint: `ferrite-run`
- Compatibility resolver: `scripts/resolve_cuda_compat.sh`
- Daemon binary: `ferrite-os/target/release/ferrite-daemon`

## Invariants

- The repo ships source; external binaries are fetched during install.
- Compatibility selection is CUDA-version driven and controlled by `compat.toml`.
- Python torch installation is not required.

## Subsystem Contracts

- Contract index: `contracts/README.md`
- Subsystem A: `contracts/subsystem-a-tui.md`
- Subsystem B: `contracts/subsystem-b-cuda-os-layer.md`
- Subsystem C: `contracts/subsystem-c-rust-runtime-core.md`
- Subsystem D: `contracts/subsystem-d-installer-provisioner.md`
- Subsystem E: `contracts/subsystem-e-external-integrations.md`
- Subsystem F: `contracts/subsystem-f-daemon-service-lifecycle.md`
- Subsystem G: `contracts/subsystem-g-diagnostics-error-contracts.md`
