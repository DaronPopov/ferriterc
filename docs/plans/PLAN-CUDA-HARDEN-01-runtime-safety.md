# PLAN-CUDA-HARDEN-01: Runtime Safety Hardening

## Mission
Eliminate panic-driven failure behavior in runtime hot paths and enforce strict pointer/stream validation with structured errors.

## Win Condition
- Build passes.
- Tests pass.
- Runtime, FFI, and daemon integration stay ABI/API compatible.

## Primary Targets
- `ferrite-os/ptx-runtime/src/runtime/scheduling.rs`
- `ferrite-os/ptx-runtime/src/runtime/lifecycle.rs`
- `ferrite-os/ptx-runtime/src/error.rs`
- `ferrite-os/internal/ptx-kernels/src/guards.rs`

## Steps
1. Convert panic/assert runtime guard paths to `Result` errors.
2. Harden invalid stream ID and pointer ownership checks.
3. Add tests for invalid free/invalid stream/invalid launch context.
4. Verify error remediation text is actionable.

## Gates
```bash
cd ferrite-os
make
cargo build --workspace --release
cargo test -p ptx-runtime
cargo test -p ptx-kernels
```

## Done
- [ ] No user-triggerable panic paths in runtime hot APIs.
- [ ] Tests cover new failure branches.
- [ ] Build/test/integration all pass.
