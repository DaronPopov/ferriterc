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

1. `ferrite-os/lib/libptx_os.so` exists.
2. `LIBTORCH/lib` exists and is valid.
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

