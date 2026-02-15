# 03.03 Runbook: Jetson Embedded Bring-Up

This runbook is for deploying Ferrite on NVIDIA Jetson devices without changing core desktop/server behavior.

## Scope

- Jetson Orin, Xavier, TX2, Nano/TX1
- aarch64 Linux hosts
- CUDA runtime + Ferrite runtime bring-up

## Target SM Map

- Orin: `sm_87`
- Xavier: `sm_72`
- TX2: `sm_62`
- Nano/TX1: `sm_53`

Current kernel profile requires `sm_75+`, so Jetson Orin (`sm_87`) is the
embedded target supported by default in this profile. Legacy Jetson SKUs
need a separate legacy kernel profile.

## Install

From repo root:

```bash
./scripts/install.sh --sm 87
```

Replace `87` with your Jetson SKU SM if not Orin.

## Runtime Mode Controls

These settings are optional and do not change default desktop behavior.

- `PTX_ORIN_UM=1`: force Orin unified-memory kernel path
- `PTX_MANAGED_POOL=1`: force managed-memory TLSF backing pool
- `PTX_DISABLE_EMBEDDED_MANAGED_POOL=1`: disable automatic managed-pool on embedded integrated GPUs

Daemon config equivalents (`ferrite-os/crates/internal/ptx-daemon/ferrite-daemon.toml`):

- `prefer_orin_unified_memory = true|false`
- `use_managed_pool = true|false`

## Bring-Up Checklist

1. Confirm architecture:

```bash
uname -m
```

Expected: `aarch64`.

2. Run generic CUDA/runtime doctor:

```bash
cd ferrite-os
tooling/scripts/ptx_doctor.sh
```

3. Run Jetson-specific doctor:

```bash
tooling/scripts/jetson_doctor.sh
```

4. Start daemon and capture logs:

```bash
RUST_LOG=info ferrite-daemon serve --config ferrite-os/crates/internal/ptx-daemon/ferrite-daemon.toml 2>&1 | tee /tmp/ferrite_jetson.log
```

5. Validate signatures:

```bash
ferrite-os/tooling/scripts/jetson_doctor.sh --log-file /tmp/ferrite_jetson.log
```

## Expected Log Signatures

At runtime init (one of):

- `[Ferrite-OS] Orin unified-memory mode active (integrated=1, sm=87)`
- `[Ferrite-OS] Embedded managed-pool mode active (...)`

Managed pool:

- `[Ferrite-OS] Managed pool allocated:`

If persistent kernel boot is enabled:

- `[PTX-OS-ORIN-UM] Launching persistent kernel`
- `[GPU-OS-ORIN-UM] Kernel initialized. Unified-memory scheduler active.`
- `[PTX-OS-ORIN-UM] Persistent kernel resident.`

## Pass Criteria

- Installer/build succeeds with explicit Jetson SM.
- `jetson_doctor.sh` exits `0`.
- Runtime mode and managed-pool signatures appear in logs.
- No fatal runtime init errors.

## Common Remediation

- SM mismatch:
  Reinstall/build with explicit `--sm <expected>` and verify `PTX_GPU_SM`.
- Missing aarch64 CUDA libs:
  Verify `CUDA_PATH` points to Jetson CUDA toolkit with `targets/aarch64-linux/lib`.
- Missing runtime signatures:
  Enable explicit toggles (`PTX_ORIN_UM=1`, `PTX_MANAGED_POOL=1`) and re-test.
- Persistent kernel signatures missing:
  Enable daemon boot kernel mode and repeat log check.
