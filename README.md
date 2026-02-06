# ferrite-runtime-src

Self-contained source runtime for Ferrite GPU compute.

Everything installs/builds inside this folder so users can treat it like an isolated runtime workspace.

## One-Line Install

```bash
./install.sh
```

Optional explicit SM:

```bash
./install.sh --sm 86
```

## What `install.sh` does

1. Builds `ferrite-os` (`libptx_os.so`, `libptx_hook.so`)
2. Builds `ferrite-gpu-lang` with Torch support
3. Builds `external/ferrite-torch` examples
4. Builds `external/ferrite-xla` backend example
5. Runs a Torch runtime validation script

## Included Components

- `ferrite-os` (custom allocator runtime + CUDA interception)
- `ferrite-gpu-lang` (Rust scripting/runtime layer)
- `external/aten-ptx`
- `external/cudarc-ptx`
- `external/ferrite-torch`
- `external/ferrite-xla`

## LibTorch Provisioning

Installer resolves libtorch in this order:

1. `LIBTORCH` env path
2. `external/libtorch`
3. Auto-download libtorch
4. Optional Python torch fallback (`LIBTORCH_ALLOW_PYTORCH=1`)

Auto-download controls:
- `LIBTORCH_VERSION` (default `2.3.0`)
- `LIBTORCH_CUDA_TAG` (default `cu121`)
- `LIBTORCH_URL` (full override)
- `LIBTORCH_ALLOW_PYTORCH` (default `0`)
