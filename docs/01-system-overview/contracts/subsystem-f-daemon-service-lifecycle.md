# Subsystem F Contract: Daemon and Service Lifecycle

## Purpose
Provide stable daemon lifecycle for interactive and service-managed operation.

## Owned Paths
- `ferrite-os/internal/ptx-daemon/src/**`
- `ferrite-os/internal/ptx-daemon/*.toml`
- `ferrite-os/internal/ptx-daemon/*.service`
- `ferrite-os/scripts/ptx_daemon.sh`
- `ferrite-os/scripts/ptx_daemon.service`

## Public Interfaces
- Daemon CLI commands and options (`ferrite-daemon`)
- Daemon socket command protocol (ping/status/stats/metrics/...)
- Service entrypoint scripts and unit files

## Forbidden Cross-Dependencies
- No direct dependency on CUDA core internals outside runtime API surface
- TUI modules must not define daemon lifecycle policy

## No-Break Rules
- Keep CLI/config/env precedence semantics stable
- Keep startup/shutdown/restart behavior stable across manual and systemd modes
