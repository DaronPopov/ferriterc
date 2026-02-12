# PLAN-CUDA-NEXT-06: Scheduler State Integration (Remove Placeholder Control Plane)

## Mission
Replace placeholder scheduler responses with live runtime scheduler state.

## Win Condition
- Build/tests pass.
- Scheduler command JSON reports real queue/job/tenant state.
- Integration validated through daemon command checks.

## Primary Targets
- `ferrite-os/internal/ptx-daemon/src/scheduler_commands.rs`
- `ferrite-os/internal/ptx-daemon/src/commands.rs`
- `ferrite-os/ptx-runtime/src/scheduler/mod.rs`

## Steps
1. Add runtime scheduler query APIs for queue depth, active jobs, tenant snapshots.
2. Remove static values (`queue_depth:0`, `active_jobs:0`, empty arrays) from daemon handlers.
3. Add tests for populated/empty scheduler state responses.

## Gates
```bash
cd ferrite-os
cargo build --workspace --release
cargo test -p ptx-runtime scheduler::
cargo test -p ptx-daemon
```

## Done
- [ ] Placeholder scheduler payloads removed.
- [ ] Runtime-backed responses verified.
- [ ] Build/test/integration pass.
