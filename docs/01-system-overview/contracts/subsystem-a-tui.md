# Subsystem A Contract: TUI System

## Purpose
Provide interactive daemon UX without changing daemon command semantics.

## Owned Paths
- `ferrite-os/internal/ptx-daemon/src/tui/**`

## Public Interfaces
- TUI entrypoint through daemon process (`ferrite-daemon` default non-headless mode)
- Event ingestion from daemon event channel (`DaemonEvent`)

## Forbidden Cross-Dependencies
- No direct dependency on `ferrite-os/core/**` CUDA internals
- No install/provision policy logic (`install.sh`, `scripts/install/**`)

## No-Break Rules
- Keep command keybindings/command routing behavior stable unless explicitly requested
- Keep TUI isolated from core daemon lifecycle state transitions
