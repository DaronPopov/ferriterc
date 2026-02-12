# Change Playbooks

## Playbook A: Installer/Portability Update

1. Edit `compat.toml` for mapping changes.
2. Edit `scripts/resolve_cuda_compat.sh` if resolution logic changes.
3. Edit `install.sh` or `scripts/install/lib/*.sh` for CLI/workflow changes (especially `args.sh`, `build.sh`, `cuda.sh`).
4. Update `INSTALL.md` and install section in `README.md`.
5. Validate:
   - `bash -n install.sh`
   - `bash -n scripts/resolve_cuda_compat.sh`
   - `bash -n scripts/install/lib/*.sh`
   - `scripts/resolve_cuda_compat.sh --format env`

## Playbook B: Runtime Build/Link Failure

1. Confirm native artifacts in `ferrite-os/lib/`.
2. Inspect `ferrite-os/crates/internal/ptx-sys/build.rs`.
3. Confirm `LD_LIBRARY_PATH` construction in `install.sh` and `ferrite-run`.
4. Validate with a minimal example build.

## Playbook C: Torch Integration Change

1. Check feature wiring in:
   - `ferrite-gpu-lang/Cargo.toml`
   - `external/ferrite-torch/Cargo.toml`
2. Check bridge build scripts in `external/aten-ptx/`.
3. Ensure selected CUDA feature is applied to all torch-enabled cargo invocations.

## Playbook D: Daemon Startup Change

1. Update daemon crate under `ferrite-os/crates/internal/ptx-daemon/`.
2. Update service installation logic in `install.sh` if needed.
3. Re-check generated unit environment and paths.
4. Validate with `systemctl status ferrite-daemon`.

## Playbook E: Rust Entry Runner Command Change

1. Update runner logic in `ferrite-os/crates/internal/ptx-runner/src/main.rs`.
2. Wire daemon handlers/routing in:
   - `ferrite-os/crates/internal/ptx-daemon/src/commands.rs`
   - `ferrite-os/crates/internal/ptx-daemon/src/server/command_pipeline.rs`
   - `ferrite-os/crates/internal/ptx-daemon/src/lifecycle.rs`
3. Keep policy classification current in `ferrite-os/crates/internal/ptx-daemon/src/policy/engine.rs`.
4. Validate:
   - `cd ferrite-os`
   - `cargo check -p ptx-runner -p ferrite-daemon`
   - `cargo test -p ferrite-daemon --lib`
   - `cargo run -p ptx-runner -- run-list`

## Playbook F: Core-Only Mode Change

When modifying what gets built in core-only vs full mode:

1. Update `scripts/install/lib/args.sh` for flag parsing.
2. Update `scripts/install/lib/build.sh` for `run_build()` conditionals and `print_success()`.
3. Update `scripts/install/install.sh` for orchestration flow.
4. Update `scripts/install/lib/service.sh` for systemd unit LIBTORCH guard.
5. Validate:
   - `bash -n scripts/install/lib/*.sh`
   - `bash -n install.sh`
