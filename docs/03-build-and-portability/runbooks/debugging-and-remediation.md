# 03.02 Runbook: Debugging and Remediation

## Doctor Check
```bash
cd ferrite-os
scripts/ptx_doctor.sh
```

## If `libptx_os.so` Is Missing

The `ptx-sys` build script (`ferrite-os/internal/ptx-sys/build.rs`) panics with an actionable error message when `libptx_os.so` is not found. This enforces build order: the C/CUDA runtime must be built before Rust crates.

```bash
cd ferrite-os
make all
ls -l lib/libptx_os.so
```

## If CUDA Toolkit Auto-Install Fails

The installer auto-installs the CUDA toolkit via `ensure_cuda_toolkit()` in `scripts/install/lib/cuda.sh`. If it fails, check these diagnostic codes:

- `INS-CUDA-0010`: cannot auto-install on this distro — install manually from https://developer.nvidia.com/cuda-downloads
- `INS-CUDA-0011`: nvcc not on PATH after install — add `/usr/local/cuda/bin` to PATH or set `CUDA_PATH`

Manual workaround:
```bash
# Install CUDA toolkit manually, then re-run:
export PATH="/usr/local/cuda/bin:$PATH"
./install.sh
```

## If Shared Memory Collides Between Users

IPC shared memory keys are per-UID (`/ptx_os_{uid}_v1`). On multi-user systems, each user gets an isolated shared memory segment.

To inspect:
```bash
ls /dev/shm/ptx_os_*
```

If collisions occur, verify that each user is running their own daemon instance and that no stale segments remain from a crashed process. Remove stale segments with:
```bash
rm /dev/shm/ptx_os_<uid>_v1
```

## If Compatibility Selection Looks Wrong
```bash
./scripts/resolve_cuda_compat.sh --format env
cat compat.toml
```

## If Daemon Service Fails
```bash
systemctl status ferrite-daemon
journalctl -u ferrite-daemon -n 200
```

## If Script Execution Fails
```bash
./ferrite-run --help
cd ferrite-gpu-lang && cargo check
```
