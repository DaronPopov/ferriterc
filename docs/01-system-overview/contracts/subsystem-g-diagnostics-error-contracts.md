# Subsystem G Contract: Diagnostics and Error Contracts

## Purpose
Define deterministic diagnostics and actionable error categories across install/runtime.

## Owned Paths
- `ferrite-os/tooling/scripts/ptx_doctor.sh`
- Error/diagnostic paths in daemon/runtime/install docs and scripts

## Public Interfaces
- `ptx_doctor` command contract
- Structured daemon command responses (JSON response/error payloads)
- Installer error messages and exit-code behavior
- `scripts/install/lib/diag.sh` defines the `diag_emit` function (structured diagnostic reporting with plain text and JSON output)
- CUDA toolkit auto-install diagnostic codes:
  - `INS-CUDA-0010`: cannot auto-install CUDA toolkit on this distro
  - `INS-CUDA-0011`: nvcc not on PATH after install
  - `INS-CUDA-0012`: CUDA toolkit auto-installed successfully

## Forbidden Cross-Dependencies
- No behavior-changing runtime edits from diagnostics-only work
- No speculative capability claims in docs

## No-Break Rules
- Keep diagnostics commands script-valid and machine-runnable
- Keep remediation text aligned with actual scripts and paths
