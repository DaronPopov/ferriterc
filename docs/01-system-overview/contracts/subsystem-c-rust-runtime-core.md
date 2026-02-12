# Subsystem C Contract: Rust Runtime Core

## Purpose
Provide safe runtime/compiler/tensor/autograd boundaries over native ABI.

## Owned Paths
- `ferrite-os/crates/public/ptx-runtime/src/**`
- `ferrite-os/crates/internal/ptx-compiler/src/**`
- `ferrite-os/crates/internal/ptx-tensor/src/**`
- `ferrite-os/crates/internal/ptx-autograd/src/**`
- `ferrite-gpu-lang/src/runtime/**`

## Public Interfaces
- `ptx-runtime` public Rust API (`PtxRuntime`, stream/memory/stats APIs)
- `ptx-compiler` graph compile APIs
- `ptx-tensor` and `ptx-autograd` public crate APIs

## Forbidden Cross-Dependencies
- No dependency on installer internals (`scripts/install/lib/**`)
- No dependency on daemon TUI modules

## No-Break Rules
- Keep crate public APIs and execution ordering behavior stable
- Keep tensor/autograd numerical behavior stable
