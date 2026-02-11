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
- CLI flags for `./install.sh`
- Compatibility resolver output contract (`--format env|json`)

## Forbidden Cross-Dependencies
- No daemon runtime command logic in installer parser/policy modules
- No CUDA runtime feature changes through doc-only edits

## No-Break Rules
- Keep CLI flag semantics and pin behavior stable
- Keep no-precompiled-ferrite-binaries policy explicit
