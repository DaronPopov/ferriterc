# PLAN-CUDA-HARDEN-04: Recovery and Determinism

## Mission
Make failure handling and recovery behavior deterministic enough for production incident response.

## Win Condition
- Build/tests pass.
- Retry policy follows failure classification predictably.
- Watchdog/recovery flow is test-covered.

## Primary Targets
- `ferrite-os/ptx-runtime/src/job/failure.rs`
- `ferrite-os/ptx-runtime/src/runtime/resilience.rs`
- `ferrite-os/internal/ptx-daemon/src/supervisor.rs`

## Steps
1. Tighten error classification (transient/permanent/unknown).
2. Align restart policy decisions with classification.
3. Add deterministic scheduling/recovery knobs where feasible.
4. Add targeted recovery tests + runbook updates.

## Gates
```bash
cd ferrite-os
make
cargo build --workspace --release
cargo test -p ptx-runtime
cargo test -p ptx-daemon
```

## Done
- [ ] Recovery path deterministic and observable.
- [ ] Retry semantics tested.
- [ ] Build/test/integration pass.
