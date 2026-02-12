# 04.07 Ferrite App Manifest (Loose v0.1)

## Status

Exploratory contract for shaping Ferrite app scripting without restricting current Rust-first workflows.
This document is guidance, not an enforced runtime policy yet.

## Intent

- Keep full Rust capabilities for scripts and apps.
- Add a lightweight manifest shape for app identity, runtime hints, and operations.
- Prefer warning-first validation and compatibility fallbacks over hard failures.

## Validation Posture

- Minimal hard-fail fields: `app.id`, `app.entry`.
- All other fields are optional and defaulted.
- Unknown keys are accepted and ignored (with warning) for forward compatibility.
- Default mode remains permissive; strict mode can be layered later through daemon policy.

## Suggested Manifest Shape

```toml
manifest_version = 0

[app]
id = "com.acme.pricer"
version = "0.1.0"
entry = "ferrite-os/workloads/mathematics_engine/monte_carlo/path_pricer.rs"
description = "Monte Carlo pricing app"

[runtime]
profile = "batch"
features = ["torch", "cuda-12060"]
min_ferrite = "0.1.0"
enforce = false

[resources]
max_streams = 64
pool_fraction = 0.30

[capabilities]
fs_read = ["./data/**"]
fs_write = ["./out/**"]
network = false
ipc = ["daemon.default"]

[observability]
log_level = "info"
metrics_ns = "apps.pricer"

[lifecycle]
run = "main"
health = "health_check"
shutdown = "graceful_shutdown"
```

## Field Defaults and Warning Conditions

| Field | Default | Warning Condition |
|---|---|---|
| `manifest_version` | `0` | Unknown higher version; run in compatibility mode |
| `app.id` | none (required) | Missing or empty |
| `app.version` | `"0.0.0-dev"` | Not semver-like |
| `app.entry` | none (required) | Missing, non-`.rs`, or path not found |
| `app.description` | `""` | Too long (truncate in UI/logs) |
| `runtime.profile` | `"dev"` | Unknown profile; fallback to `dev` |
| `runtime.features` | auto-detect from imports | Unknown feature token ignored |
| `runtime.min_ferrite` | none | Current runtime appears older |
| `runtime.enforce` | `false` | `true` with unmet runtime/feature constraints |
| `resources.max_streams` | daemon default | Non-positive or very high value |
| `resources.pool_fraction` | daemon default | Out of `(0,1]`; clamp and warn |
| `resources.gpu_affinity` | none | Requested GPU unavailable |
| `capabilities.fs_read` | permissive in default mode | Path globs invalid/unresolvable |
| `capabilities.fs_write` | permissive in default mode | Writes outside workspace/policy bounds |
| `capabilities.network` | permissive in default mode | Requested in strict policy-disabled environment |
| `capabilities.ipc` | `["daemon.default"]` | Unknown channel name |
| `observability.log_level` | `"info"` | Unknown level; fallback to `info` |
| `observability.metrics_ns` | `app.id` | Invalid chars; sanitize and warn |
| `lifecycle.init` | none | Symbol not found at runtime |
| `lifecycle.run` | `"main"` | Symbol missing or wrong signature |
| `lifecycle.health` | none | Symbol missing; health marked unsupported |
| `lifecycle.shutdown` | none | Symbol missing; fallback to process signal handling |

## Warning Code Set (Proposed)

- `APP-MAN-0001`: missing optional field, default applied
- `APP-MAN-0002`: invalid value, fallback or clamp applied
- `APP-MAN-0003`: unknown key ignored
- `APP-MAN-0004`: requested capability not enforceable in current mode
- `APP-MAN-0005`: runtime constraint unmet but continued (`enforce=false`)

## Runtime Modes

- Default mode: permissive, warning-first, preserves current behavior.
- Strict mode: optional policy layer for capability enforcement and hard denials.

## Compatibility Rules

- If `manifest_version` is missing, treat as `0`.
- New fields should be additive and optional.
- Unknown fields must not block execution in default mode.
