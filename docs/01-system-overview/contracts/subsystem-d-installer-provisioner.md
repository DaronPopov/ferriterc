# Subsystem D Contract: Installer and Provisioner

## Purpose
Provide deterministic one-line source install with auto and pinned compatibility flows.

## Owned Paths
- `install.sh`
- `scripts/install/install.sh`
- `scripts/install/lib/*.sh`
- `scripts/resolve_cuda_compat.sh`
- install sections in `README.md` and `INSTALL.md`

## Public Interfaces
- CLI flags for `./install.sh` (including `--core-only` which skips libtorch and torch-dependent crates)
- Compatibility resolver output contract (`--format env|json`)
- Modular installer libraries in `scripts/install/lib/*.sh`: args, build, cuda, diag, env, libtorch, policy, preflight, rust, service

## Forbidden Cross-Dependencies
- No daemon runtime command logic in installer parser/policy modules
- No CUDA runtime feature changes through doc-only edits

## No-Break Rules
- Keep CLI flag semantics and pin behavior stable
- Keep no-precompiled-ferrite-binaries policy explicit
- CUDA toolkit auto-install via `ensure_cuda_toolkit()` when `nvcc` is absent — only NVIDIA driver required pre-installed
