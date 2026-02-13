# LLM Definition Of Done

Scope: changes to `ferrite-os`, `ferrite-gpu-lang`, daemon/runtime control plane, and runtime architecture docs.

## 1. Build Gate

From repo root:

```bash
cd ferrite-os
make all
cargo check -p ferrite-daemon
```

## 2. Runtime Smoke Gate

Run sequentially (`--test-threads=1` and one command at a time):

```bash
cargo test -p ferrite-daemon --test daemon_integration task_isa_v0_roundtrip -- --nocapture --test-threads=1
cargo test -p ferrite-daemon --test daemon_integration task_completion_v1_roundtrip -- --test-threads=1
cargo test -p ferrite-daemon --test daemon_integration task_dag_continuation_v1_roundtrip -- --test-threads=1
cargo test -p ferrite-daemon --test daemon_integration task_timeslice_fairness_v1_roundtrip -- --test-threads=1
cargo test -p ferrite-daemon --test daemon_integration task_tenant_budget_starvation_guard_v1_roundtrip -- --test-threads=1
```

## 3. Documentation Gate

If behavior, ABI, opcodes, commands, or test expectations change, update all affected docs in the same change:

1. Root overview: `README.md`
2. Runtime architecture index: `docs/02-runtime-architecture/README.md`
3. ISA spec/status: `docs/02-runtime-architecture/gpu-isa-v0-design.md`
4. Daemon command docs: `ferrite-os/crates/internal/ptx-daemon/README.md`
5. Agent runbook: `AGENT.md`

## 4. Commit Gate

Before push:

1. Ensure no unrelated or generated directories are accidentally staged (for example `external/onnxruntime/` if untracked locally).
2. Summarize what was validated and what was intentionally not run.
3. Use a commit message that names the subsystem changed (`isa`, `scheduler`, `daemon`, `docs`, etc.).

## 5. Known Caveat

`permissive_mode_full_suite` and `strict_mode_full_suite` can fail due to local environment/example-path issues unrelated to core ISA/scheduler correctness. Unless a task specifically targets those scenarios, treat the smoke gate above as required for done.
