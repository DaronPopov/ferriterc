# 03.00 Build and Portability

## Install Pipeline

Installer entrypoint: `scripts/install.sh`

Core stages (from `scripts/install/install.sh`):

1. Platform/arch detection (`detect_platform_arch`).
2. Defaults + CLI parsing (`init_install_defaults`, `parse_install_args`).
3. Preflight checks: host tools, fetch tools, CUDA toolkit auto-install if `nvcc` absent, Rust toolchain (`run_preflight_checks`).
4. Compatibility resolution from `compat.toml` (skipped if `--core-only`).
5. CUDA env setup + SM detection (`setup_cuda_env`, `auto_detect_sm`, `export_build_env`).
6. Libtorch provisioning (skipped if `--core-only`).
7. Build (`run_build` — 3 steps core-only, 9 steps full).
8. Optional systemd service install (`--enable-service`).

## Compatibility Resolution

- Manifest: `compat.toml`
- Resolver: `scripts/resolve_cuda_compat.sh`
- Outputs:
  - `CUDARC_CUDA_FEATURE`
  - `LIBTORCH_CUDA_TAG_RESOLVED`

Resolution strategy:

1. Exact CUDA major.minor mapping.
2. Major-only mapping fallback.
3. Defaults fallback.

## Libtorch Provisioning

Order:

1. Existing `LIBTORCH` env path if valid.
2. Existing `external/libtorch` if valid.
3. Download/extract based on architecture and tag.

Notes:

- `x86_64`: libtorch archive URL path.
- `aarch64`: wheel archive extraction for C++ artifacts only.

## Pinning Modes

Automatic mode:

```bash
./scripts/install.sh
```

Pinned mode:

```bash
./scripts/install.sh --pins "sm=89,libtorch_url=<URL>,libtorch_tag=cu126,cudarc_feature=cuda-12060"
```

Equivalent explicit flags:

```bash
./scripts/install.sh --sm 89 --libtorch-url "<URL>" --libtorch-tag cu126 --cudarc-feature cuda-12060
```

Core-only mode (skips libtorch and torch-dependent crates):

```bash
./scripts/install.sh --core-only
./scripts/install.sh --core-only --sm 86
```

## Boot Service

Optional systemd setup:

```bash
./scripts/install.sh --enable-service
```

Installs and enables:

- `/etc/systemd/system/ferrite-daemon.service`
- `/etc/ferrite-os/daemon.toml`

## Portability Checklist

Before calling a machine "supported" (full mode):

1. Installer runs cleanly from fresh clone.
2. Compatibility resolver selects expected CUDA family.
3. libtorch fetch/extract succeeds.
4. Torch-enabled build succeeds.
5. At least one runtime script executes.
6. Optional daemon service starts and stays healthy.

Core-only variant:

1. Installer runs cleanly with `--core-only` from fresh clone.
2. CUDA toolkit auto-installs if absent (only NVIDIA driver required).
3. Core build (3 steps) completes.
4. `ferrite` and `ferrite-daemon` binaries are functional.

Installed command surface:

- Installer symlinks `ferrite` and `ferrite-daemon` into `~/.local/bin/`.
- With `~/.local/bin` on `PATH`, daemon is invokable from any directory as `ferrite-daemon`.

## Runbooks

- `03.01`: Install and provisioning: `runbooks/install-and-provision.md`
- `03.02`: Debugging and remediation: `runbooks/debugging-and-remediation.md`
