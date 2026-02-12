# Change Playbooks

## Playbook A: Installer/Portability Update

1. Edit `compat.toml` for mapping changes.
2. Edit `scripts/resolve_cuda_compat.sh` if resolution logic changes.
3. Edit `install.sh` for CLI/workflow changes.
4. Update `INSTALL.md` and install section in `README.md`.
5. Validate:
   - `bash -n install.sh`
   - `bash -n scripts/resolve_cuda_compat.sh`
   - `scripts/resolve_cuda_compat.sh --format env`

## Playbook B: Runtime Build/Link Failure

1. Confirm native artifacts in `ferrite-os/lib/`.
2. Inspect `ferrite-os/internal/ptx-sys/build.rs`.
3. Confirm `LD_LIBRARY_PATH` construction in `install.sh` and `ferrite-run`.
4. Validate with a minimal example build.

## Playbook C: Torch Integration Change

1. Check feature wiring in:
   - `ferrite-gpu-lang/Cargo.toml`
   - `external/ferrite-torch/Cargo.toml`
2. Check bridge build scripts in `external/aten-ptx/`.
3. Ensure selected CUDA feature is applied to all torch-enabled cargo invocations.

## Playbook D: Daemon Startup Change

1. Update daemon crate under `ferrite-os/internal/ptx-daemon/`.
2. Update service installation logic in `install.sh` if needed.
3. Re-check generated unit environment and paths.
4. Validate with `systemctl status ferrite-daemon`.

## Playbook E: Rust Entry Runner Command Change

1. Update runner logic in `ferrite-os/internal/ptx-runner/src/main.rs`.
2. Wire daemon handlers/routing in:
   - `ferrite-os/internal/ptx-daemon/src/commands.rs`
   - `ferrite-os/internal/ptx-daemon/src/server/command_pipeline.rs`
   - `ferrite-os/internal/ptx-daemon/src/lifecycle.rs`
3. Keep policy classification current in `ferrite-os/internal/ptx-daemon/src/policy/engine.rs`.
4. Validate:
   - `cd ferrite-os`
   - `cargo check -p ptx-runner -p ferrite-daemon`
   - `cargo test -p ferrite-daemon --lib`
   - `cargo run -p ptx-runner -- run-list`
