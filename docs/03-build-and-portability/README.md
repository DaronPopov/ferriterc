# Build and Portability

## Install Pipeline

Installer entrypoint: `install.sh`

Core stages:

1. Host checks/tool bootstrap.
2. CUDA discovery.
3. Compatibility resolution from `compat.toml`.
4. CUPTI verification/installation attempt.
5. libtorch provisioning (or pinned URL usage).
6. Build `ferrite-os`.
7. Build Rust/Torch integration layers.
8. Run compile/runtime validation steps.
9. Optional systemd daemon setup.

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
./install.sh
```

Pinned mode:

```bash
./install.sh --pins "sm=89,libtorch_url=<URL>,libtorch_tag=cu126,cudarc_feature=cuda-12060"
```

Equivalent explicit flags:

```bash
./install.sh --sm 89 --libtorch-url "<URL>" --libtorch-tag cu126 --cudarc-feature cuda-12060
```

## Boot Service

Optional systemd setup:

```bash
./install.sh --enable-service
```

Installs and enables:

- `/etc/systemd/system/ferrite-daemon.service`
- `/etc/ferrite-os/daemon.toml`

## Portability Checklist

Before calling a machine "supported":

1. Installer runs cleanly from fresh clone.
2. Compatibility resolver selects expected CUDA family.
3. libtorch fetch/extract succeeds.
4. Torch-enabled build succeeds.
5. At least one runtime script executes.
6. Optional daemon service starts and stays healthy.

## Runbooks

- Install and provisioning: `runbooks/install-and-provision.md`
- Debugging and remediation: `runbooks/debugging-and-remediation.md`
