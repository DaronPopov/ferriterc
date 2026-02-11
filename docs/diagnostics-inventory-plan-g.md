# Plan-G Diagnostics Inventory

This inventory captures baseline diagnostic/error surfaces before normalization.

## Subsystems

- Installer: `scripts/install/install.sh` + `scripts/install/lib/*.sh`
- Compatibility resolver: `scripts/resolve_cuda_compat.sh`
- Doctor: `ferrite-os/scripts/ptx_doctor.sh`
- Daemon: `ferrite-os/internal/ptx-daemon/src/{events.rs,state.rs,server.rs}`
- Runtime: `ferrite-os/ptx-runtime/src/{error.rs,telemetry.rs,stats.rs,resilience.rs}`

## Recurring Failure Classes

- CUDA not found or version not parseable
- Compatibility mapping missing for detected CUDA version
- Missing host prerequisites (`make`, `gcc`, `git`, `curl/wget`, `unzip/bsdtar`)
- Libtorch invalid/mismatch or unavailable download artifacts
- Service startup prerequisites missing (`systemctl`, daemon binary, config template)
- Runtime init/memory/stream/cublas/stable API errors
- Daemon request failures and lifecycle service issues

## Exit Code Patterns (Preserved)

- Resolver:
  - `1`: runtime/preflight/resolution failures
  - `2`: argument/format errors
- Installer:
  - `1`: invalid options, unsupported platform, missing prerequisites, provisioning/build/service failures
  - `0`: success
- Doctor:
  - `1`: missing env script/CUDA or blocking health checks
  - `0`: successful health run (warnings allowed)

## Diagnostics Contract Introduced by Plan-G

- Status vocabulary: `PASS`, `WARN`, `FAIL`
- Field schema:
  - `component`
  - `status`
  - `code`
  - `summary`
  - `remediation`
- Optional machine-readable mode:
  - Shell paths use `FERRITE_DIAG_FORMAT=json` (and per-script flags where supported)
