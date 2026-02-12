# Ferrite Testkit

System-level headless testing toolkit for Ferrite daemon workflows.

This lives at repo root so it can evolve as a first-class cross-subsystem test surface,
not only as a daemon-internal test file.

## Layout

- `scenarios/`: declarative scenario specs (`.toml`)
- `fixtures/`: reusable sample scripts/apps referenced by scenarios
- `reports/`: JSON test reports (machine-readable)
- `rust/ferrite-testkit/`: reusable harness library
- `rust/ferrite-testkit-cli/`: CLI runner

## Quickstart

From repo root:

```bash
cargo run --manifest-path testkit/rust/ferrite-testkit-cli/Cargo.toml -- \
  run --scenario testkit/scenarios/boot-health.toml
```

Run matrix:

```bash
cargo run --manifest-path testkit/rust/ferrite-testkit-cli/Cargo.toml -- \
  matrix --dir testkit/scenarios --report testkit/reports/latest.json
```

## Notes

- Requires a built daemon binary and runtime libs available in environment.
- Intended to complement (not replace) `ferrite-os/internal/ptx-daemon/tests/daemon_integration.rs`.
- Use `DURATION=1` style env in scenario daemon profile for short bounded runs.
