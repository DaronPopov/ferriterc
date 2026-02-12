# Subsystem F Contract: Daemon and Service Lifecycle

## Purpose
Provide stable daemon lifecycle for interactive and service-managed operation.

## Owned Paths
- `ferrite-os/crates/internal/ptx-daemon/src/**`
- `ferrite-os/crates/internal/ptx-daemon/*.toml`
- `ferrite-daemon`
- `ferrite-os/ferrite-daemon.sh` (local wrapper, non-canonical)
- `scripts/install/lib/service.sh`

## Public Interfaces
- Daemon CLI commands and options (`ferrite-daemon`)
- Daemon socket command protocol (ping/status/stats/metrics/...)
- Service entrypoint scripts and installer-generated systemd unit
- Daemon config defaults (`ferrite-daemon.toml`): `pool_fraction=0.25`, `max_streams=128`, `max_clients=32`, `keepalive_ms=5000`
- Config sections: `[scheduler]`, `[control_plane]`, `[jobs]`
- Systemd unit is generated dynamically by `scripts/install/lib/service.sh` at install time (not from a static file)

## Forbidden Cross-Dependencies
- No direct dependency on CUDA core internals outside runtime API surface
- TUI modules must not define daemon lifecycle policy

## No-Break Rules
- Keep CLI/config/env precedence semantics stable
- Keep startup/shutdown/restart behavior stable across manual and systemd modes
- Systemd unit must conditionally include LIBTORCH environment only when not in core-only mode
