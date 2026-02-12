# 00.00 Ferrite Documentation

This folder contains architecture and programming guides for `ferriterc`.
It is written for engineers and LLM coding agents.

## Reading Order

1. `01-system-overview/README.md` (`01.00`)
2. `02-runtime-architecture/README.md` (`02.00`)
3. `03-build-and-portability/README.md` (`03.00`)
4. `04-llm-programming-guides/README.md` (`04.00`)
5. `../INSTALL.md` (`05.00`)
6. `DOCS-REWRITE-PLAN.md` (`99.00`, maintenance)

## Structure

- `01-system-overview/`
  What this repository is, major components, and high-level flow.
- `02-runtime-architecture/`
  Runtime internals: memory model, execution path, daemon, and interfaces.
- `03-build-and-portability/`
  Installer flow, compatibility resolution, artifact provisioning, and pinning.
- `04-llm-programming-guides/`
  Practical implementation and debugging playbooks for LLM-driven changes.
- `DOCS-REWRITE-PLAN.md`
  Maintenance checklist for docs normalization and drift cleanup.

## Scope

- Factual behavior only.
- No benchmarks or performance claims in this doc set.
- Use source files as the authority when docs and code diverge.

## Source-of-Truth Files

- Installer: `install.sh`
- Runtime launcher: `ferrite-run`
- Daemon launcher: `ferrite-daemon`
- Compatibility mapping: `compat.toml`
- Compatibility resolver: `scripts/resolve_cuda_compat.sh`
- Runtime workspace: `ferrite-os/`
- Language/runtime APIs: `ferrite-gpu-lang/`
- Installer libraries: `scripts/install/lib/*.sh`
- Daemon config: `ferrite-os/internal/ptx-daemon/ferrite-daemon.toml`

## Architecture Contracts and Runbooks

- `01.01`: Subsystem contracts index: `01-system-overview/contracts/README.md`
- `02.01`: Runtime operations runbook: `02-runtime-architecture/runbooks/runtime-operations.md`
- `03.01`: Install/provision runbook: `03-build-and-portability/runbooks/install-and-provision.md`
- `03.02`: Debug/remediation runbook: `03-build-and-portability/runbooks/debugging-and-remediation.md`
- `04.06`: Agent routing and validation gates: `04-llm-programming-guides/contracts/task-routing-and-gates.md`
- `04.07`: Loose app manifest contract: `04-llm-programming-guides/contracts/ferrite-app-manifest-v0-loose.md`
