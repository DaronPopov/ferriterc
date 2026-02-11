# Subsystem E Contract: External Integrations

## Purpose
Bridge external ecosystems (ATen/cudarc/torch/xla) to ferrite runtime contracts.

## Owned Paths
- `external/aten-ptx/**`
- `external/cudarc-ptx/**`
- `external/ferrite-torch/**`
- `external/ferrite-xla/**`

## Public Interfaces
- Integration crate public APIs and build feature flags
- Runtime linkage contracts to `ptx-sys` / `ptx-runtime`

## Forbidden Cross-Dependencies
- No dependency on installer argument parsing internals
- No dependency on daemon TUI modules

## No-Break Rules
- Keep integration runtime compatibility behavior stable
- Keep feature-flag and link contracts aligned with resolver outputs
