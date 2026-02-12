# Architecture Walkthrough (For LLMs)

## Goal

Given a requested change, identify the correct layer first, then edit only necessary files.

## Layer Map

1. Installer and compatibility
   - `install.sh`
   - `scripts/install/install.sh`
   - `scripts/install/lib/*.sh` (args, build, cuda, diag, env, libtorch, policy, preflight, rust, service)
   - `compat.toml`
   - `scripts/resolve_cuda_compat.sh`
2. Native runtime
   - `ferrite-os/core/`
   - `ferrite-os/Makefile`
3. Rust runtime bridge
   - `ferrite-os/internal/ptx-sys/`
   - `ferrite-os/ptx-runtime/`
4. Script/language layer
   - `ferrite-gpu-lang/`
   - `ferrite-run`
5. Integrations
   - `external/aten-ptx/`
   - `external/cudarc-ptx/`
   - `external/ferrite-torch/`
   - `external/ferrite-xla/`
6. Daemon configuration
   - `ferrite-os/internal/ptx-daemon/ferrite-daemon.toml`
7. App-manifest contract (exploratory)
   - `docs/04-llm-programming-guides/contracts/ferrite-app-manifest-v0-loose.md`

## Change Routing Rules

- If issue mentions install portability, start in installer/compat files.
- If issue mentions linking/FFI/shared libs, start in `ptx-sys` and `ptx-runtime`.
- If issue mentions script execution or examples, start in `ferrite-run` + `ferrite-gpu-lang`.
- If issue mentions torch/xla behavior, inspect `external/*` bridges and feature flags.
- If issue mentions `--core-only` or installer flags, start in `scripts/install/lib/args.sh` and `build.sh`.
- If issue mentions CUDA auto-install or toolkit, start in `scripts/install/lib/cuda.sh`.
- If issue mentions app lifecycle/capabilities/manifest behavior, start in `04.07` contract doc before code changes.

## Output Expectations

For each change, produce:

1. Affected layer.
2. Files changed.
3. Why those files are sufficient.
4. Verification commands.
