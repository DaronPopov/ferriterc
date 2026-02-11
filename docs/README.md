# Ferrite Documentation

This folder contains architecture and programming guides for `ferriterc`.
It is written for engineers and LLM coding agents.

## Reading Order

1. `01-system-overview/README.md`
2. `02-runtime-architecture/README.md`
3. `03-build-and-portability/README.md`
4. `04-llm-programming-guides/README.md`
5. `plans/PLAN-INDEX.txt`

## Structure

- `01-system-overview/`
  What this repository is, major components, and high-level flow.
- `02-runtime-architecture/`
  Runtime internals: memory model, execution path, daemon, and interfaces.
- `03-build-and-portability/`
  Installer flow, compatibility resolution, artifact provisioning, and pinning.
- `04-llm-programming-guides/`
  Practical implementation and debugging playbooks for LLM-driven changes.
- `plans/`
  Refactor plans and drift reports used for behavior-preserving migrations.

## Scope

- Factual behavior only.
- No benchmarks or performance claims in this doc set.
- Use source files as the authority when docs and code diverge.

## Source-of-Truth Files

- Installer: `install.sh`
- Runtime launcher: `ferrite-run`
- Compatibility mapping: `compat.toml`
- Compatibility resolver: `scripts/resolve_cuda_compat.sh`
- Runtime workspace: `ferrite-os/`
- Language/runtime APIs: `ferrite-gpu-lang/`

## Architecture Contracts and Runbooks

- Subsystem contracts index: `01-system-overview/contracts/README.md`
- Runtime operations runbook: `02-runtime-architecture/runbooks/runtime-operations.md`
- Install/provision runbook: `03-build-and-portability/runbooks/install-and-provision.md`
- Debug/remediation runbook: `03-build-and-portability/runbooks/debugging-and-remediation.md`
- Agent routing and validation gates: `04-llm-programming-guides/contracts/task-routing-and-gates.md`
