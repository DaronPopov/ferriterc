# Debugging Playbook

## Installer Fails to Download Libtorch

Checks:

1. Confirm URL is complete and quoted.
2. Confirm `LIBTORCH_URL` is not set to a truncated value.
3. Re-run with explicit flags:
   - `--libtorch-url`
   - `--libtorch-tag`
   - `--cudarc-feature`

## CUDA Feature Mismatch

Checks:

1. Run `scripts/resolve_cuda_compat.sh --format env`.
2. Verify selected feature exists in crate `Cargo.toml` features.
3. Ensure build command includes `--no-default-features --features "torch,<feature>"`.

## Linker/Shared Library Errors

Checks:

1. `ferrite-os/lib/libptx_os.so` exists. If missing, `cargo build` will panic with an actionable message from `ptx-sys/build.rs`. Run `cd ferrite-os && make all` to fix.
2. `LIBTORCH/lib` exists and is valid (not applicable in `--core-only` mode).
3. `LD_LIBRARY_PATH` includes both runtime and libtorch paths.
4. `ptx-sys` build script link-search outputs are valid.

## Service Starts but Fails

Checks:

1. `systemctl status ferrite-daemon`
2. `journalctl -u ferrite-daemon -n 200`
3. Verify generated unit paths:
   - binary path
   - config path
   - `LD_LIBRARY_PATH`
4. Re-run install without `--enable-service` to isolate build vs service issues.

## CUDA Toolkit Auto-Install Failed

The installer auto-installs CUDA toolkit via `ensure_cuda_toolkit()` in `scripts/install/lib/cuda.sh`.

Checks:

1. Look for diagnostic codes in output:
   - `INS-CUDA-0010`: unsupported distro for auto-install — install manually from https://developer.nvidia.com/cuda-downloads
   - `INS-CUDA-0011`: nvcc not on PATH after install — add `/usr/local/cuda/bin` to PATH or set `CUDA_PATH`
   - `INS-CUDA-0012`: auto-install succeeded (informational)
2. Verify `nvcc` is accessible: `which nvcc`
3. Check CUDA path resolution: `ls /usr/local/cuda/bin/nvcc`

## Shared Memory IPC Collision

IPC shared memory keys are per-UID (`/ptx_os_{uid}_v1`), constructed from constants in `gpu_hot_runtime.h`.

Checks:

1. List active segments: `ls /dev/shm/ptx_os_*`
2. Verify the UID in the segment name matches the expected user.
3. Remove stale segments from crashed processes: `rm /dev/shm/ptx_os_<uid>_v1`
