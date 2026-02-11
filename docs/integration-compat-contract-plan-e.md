# Plan-E Compatibility Contract

This document captures the current, behavior-preserving compatibility rules for external GPU integrations.

## Resolution Sources

- `compat.toml` defines CUDA-to-feature/tag mappings.
- `scripts/resolve_cuda_compat.sh` resolves local CUDA toolkit version via `nvcc --version`.
- `external/aten-ptx/build.rs` resolves libtorch location for build-time include/link.

## CUDA Compatibility Resolution Precedence

`scripts/resolve_cuda_compat.sh` resolves in this strict order:

1. Exact CUDA major.minor section: `cuda."<major>.<minor>"`
2. Major-only section: `cuda."<major>"`
3. `[defaults]`

If none resolve both `cudarc_feature` and `libtorch_cuda_tag`, the script exits non-zero.

## Libtorch Resolution Precedence

`external/aten-ptx/build.rs` resolves libtorch in this strict order:

1. `LIBTORCH` environment variable (must look like a CUDA-enabled libtorch layout)
2. `../libtorch` relative to `external/aten-ptx`
3. `../../external/libtorch` relative to `external/aten-ptx`

No Python fallback is used.

## Runtime Override Points

- CUDA/toolkit compatibility:
  - `compat.toml`
  - `scripts/resolve_cuda_compat.sh`
- Libtorch build path:
  - `LIBTORCH`
  - `CUDA_INCLUDE` (header path override in `external/aten-ptx/build.rs`)
- Runtime allocator init:
  - `init_pytorch_tlsf(device_id, pool_fraction)`
  - `init_pytorch_tlsf_ex(device_id, pool_fraction, num_streams)`

## Torch/PTX Bridge Invariants

- `torch_cuda_available()` forces CUDA backend load before checking availability.
- TLSF init path preserves:
  - explicit device id,
  - pool fraction validation,
  - stream count validation,
  - warmup allocation/free fallback check.

## Feature/Fallback Invariants

- Existing `cudarc` CUDA feature flags remain source-of-truth for selected toolkit ABI.
- Missing/failed TLSF runtime init continues to fall back to framework allocator behavior.
- No Python torch dependency is introduced by this integration path.
