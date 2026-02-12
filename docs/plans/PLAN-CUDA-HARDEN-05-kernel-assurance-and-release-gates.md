# PLAN-CUDA-HARDEN-05: Kernel Assurance and Release Gates

## Mission
Require guard-enforced kernel launches and make hardening tests a mandatory pre-merge gate.

## Win Condition
- Build/tests pass.
- Safe API path uses guard checks consistently.
- Regression suite catches launch/memory-safety regressions.

## Primary Targets
- `ferrite-os/internal/ptx-kernels/src/guards.rs`
- `ferrite-os/internal/ptx-kernels/src/safe_api/launch.rs`
- `ferrite-gpu-lang/examples/tests/test_harden.rs`

## Steps
1. Route safe launches through guard wrappers.
2. Add negative tests for invalid pointer/size/stream context.
3. Define required release gate command bundle.

## Gates
```bash
cd ferrite-os
make
cargo build --workspace --release
cargo test -p ptx-kernels
cd ..
cargo run --release -p ferrite-gpu-lang --example test_harden
```

## Done
- [ ] Guard path enforced.
- [ ] Hardening tests expanded.
- [ ] Build/test/integration pass.
