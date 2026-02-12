# PLAN-CUDA-HARDEN-03: Multi-Tenant Isolation and Quotas

## Mission
Enforce quotas consistently across scheduler admission, stream assignment, and VRAM accounting.

## Win Condition
- Build/tests pass.
- Tenant limits are enforced end-to-end.
- No tenant can exceed configured quotas through scheduler/runtime paths.

## Primary Targets
- `ferrite-os/ptx-runtime/src/scheduler/mod.rs`
- `ferrite-os/ptx-runtime/src/scheduler/dispatcher.rs`
- `ferrite-os/ptx-runtime/src/scheduler/policy.rs`
- `ferrite-os/internal/ptx-daemon/src/config.rs`

## Steps
1. Audit quota paths for gaps.
2. Enforce quotas at admission and allocation boundaries.
3. Verify usage counters update on success/failure/cancel.
4. Add fairness/starvation and quota-denial tests.

## Gates
```bash
cd ferrite-os
cargo build --workspace --release
cargo test -p ptx-runtime scheduler::
cargo test -p ptx-daemon
```

## Done
- [ ] Quotas enforced.
- [ ] Counter symmetry verified.
- [ ] Build/test/integration pass.
