# AGENT.md

Operational guide for LLM agents working in `ferriterc`.

## Scope

- This file is for execution workflow: build, test, daemon simulation, and validation.
- Prefer behavior-preserving edits unless explicitly asked to change behavior.

## Repo Root

```bash
cd /home/daron/fdfd/ferriterc
```

## Core Entrypoints

- Scripts/runtime wrapper: `./ferrite-run`
- Daemon wrapper (root, canonical): `./ferrite-daemon`
- Installer: `./scripts/install.sh`
- Systemic test tooling: `testkit/`

## Build and Check Commands

Primary workspace checks:

```bash
cd ferrite-os
cargo check -p ptx-runner -p ferrite-daemon
```

If network/index is unavailable, prefer offline mode:

```bash
cargo check --offline -p ptx-runner -p ferrite-daemon
```

## Daemon Integration Tests (Canonical Harness)

File: `ferrite-os/crates/internal/ptx-daemon/tests/daemon_integration.rs`

List tests:

```bash
cd ferrite-os
cargo test -p ferrite-daemon --test daemon_integration -- --list
```

Run one suite (recommended first):

```bash
cargo test -p ferrite-daemon --test daemon_integration permissive_mode_full_suite -- --nocapture --test-threads=1
```

Run all daemon integration tests:

```bash
cargo test -p ferrite-daemon --test daemon_integration -- --test-threads=1 --nocapture
```

## Root Systemic Testkit

Tooling:

- Library: `testkit/rust/ferrite-testkit`
- CLI: `testkit/rust/ferrite-testkit-cli`
- Scenarios: `testkit/scenarios/*.toml`

Compile testkit crates:

```bash
cargo check --offline --manifest-path testkit/rust/ferrite-testkit/Cargo.toml
cargo check --offline --manifest-path testkit/rust/ferrite-testkit-cli/Cargo.toml
```

Run a scenario:

```bash
cargo run --manifest-path testkit/rust/ferrite-testkit-cli/Cargo.toml -- run --scenario testkit/scenarios/boot-health.toml
```

Run matrix + report:

```bash
cargo run --manifest-path testkit/rust/ferrite-testkit-cli/Cargo.toml -- matrix --dir testkit/scenarios --report testkit/reports/latest.json
```

## Demo App for `run-entry`

Daemon-discoverable entry:

`ferrite-os/crates/public/ferrite-apps/src/bin/ascii_tensor_orbit.rs#main`

Run directly:

```bash
cd ferrite-os
DURATION=1 cargo run -p ptx-runner -- run-entry crates/public/ferrite-apps/src/bin/ascii_tensor_orbit.rs#main
```

Run through daemon wrapper:

```bash
./ferrite-daemon run-entry crates/public/ferrite-apps/src/bin/ascii_tensor_orbit.rs#main
```

## Environment / Runtime Notes

- GPU-dependent tests require host CUDA device access.
- If sandboxed execution reports `cudaSetDevice(0)` OS/unsupported errors, run the same command with host permissions.
- Keep `LD_LIBRARY_PATH` and libtorch resolution consistent (root wrapper already handles common cases).

## Editing Hygiene

- Do not revert unrelated user changes.
- Avoid destructive git commands.
- Update docs when entrypoints/paths change.
- For docs path updates, treat these as canonical:
  - `ferrite-daemon` (root wrapper)
  - `ferrite-os/workloads/finetune_engine/`
  - `ferrite-os/workloads/mathematics_engine/`
