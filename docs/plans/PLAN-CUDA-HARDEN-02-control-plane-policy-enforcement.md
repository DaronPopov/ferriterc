# PLAN-CUDA-HARDEN-02: Control Plane Policy Enforcement

## Mission
Upgrade policy from passive logging to active enforcement for scheduler/control-plane operations.

## Win Condition
- Build/test passes.
- Policy decisions enforce allow/deny on mutating commands.
- Audit and event stream reflect real decisions.

## Primary Targets
- `ferrite-os/internal/ptx-daemon/src/policy/engine.rs`
- `ferrite-os/internal/ptx-daemon/src/policy/decision.rs`
- `ferrite-os/internal/ptx-daemon/src/scheduler_commands.rs`
- `ferrite-os/internal/ptx-daemon/src/commands.rs`

## Steps
1. Define strict-mode rule behavior and denial defaults.
2. Ensure all mutating control-plane paths evaluate policy before execution.
3. Standardize denial response payload: reason code + remediation.
4. Add tests for permissive vs strict behavior.

## Gates
```bash
cd ferrite-os
cargo build --workspace --release
cargo test -p ptx-daemon
```

## Done
- [ ] Mutating commands are policy-gated.
- [ ] Denials are machine-readable.
- [ ] Build/test/integration pass.
