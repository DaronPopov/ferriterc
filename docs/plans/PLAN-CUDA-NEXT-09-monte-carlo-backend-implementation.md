# PLAN-CUDA-NEXT-09: Monte Carlo Backend Implementation

## Mission
Replace ptx-compute Monte Carlo stubs with real GPU-backed implementation.

## Win Condition
- Build/tests pass.
- APIs execute real sampling/count kernels.
- Correctness validated against CPU reference tolerance.

## Primary Targets
- `ferrite-os/internal/ptx-compute/src/monte_carlo.rs`
- `ferrite-os/internal/ptx-kernels/kernels/` (new kernels if needed)
- `mathematics_engine/monte_carlo/path_pricer.rs`

## Steps
1. Add kernel interfaces for random sampling + in-circle reduction.
2. Implement single-stream and multi-stream paths.
3. Add deterministic seed tests and tolerance checks.
4. Validate integration through math engine script.

## Gates
```bash
cd ferrite-os
cargo build --workspace --release
cargo test -p ptx-compute
cargo test -p ptx-kernels
cd ..
./ferrite-run mathematics_engine/monte_carlo/path_pricer.rs -- --paths 1000000
```

## Done
- [ ] Monte Carlo APIs are non-placeholder.
- [ ] Correctness tests pass.
- [ ] Build/test/integration pass.
