# Runbook: Install and Provision

## Preconditions
- Linux (`x86_64` or `aarch64`)
- NVIDIA driver
- CUDA toolkit (`nvcc` available)

## Standard Install
```bash
git clone https://github.com/DaronPopov/ferriterc.git
cd ferriterc
./install.sh
```

## Pinned Install
```bash
./install.sh --pins "sm=89,libtorch_url=https://download.pytorch.org/libtorch/cu126/libtorch-shared-with-deps-2.9.0%2Bcu126.zip,libtorch_tag=cu126,cudarc_feature=cuda-12060"
```

## Verify Compatibility Resolution
```bash
./scripts/resolve_cuda_compat.sh --format env
```

## Install Surface Validation
```bash
./install.sh --help
bash -n install.sh scripts/resolve_cuda_compat.sh scripts/install/install.sh scripts/install/lib/*.sh
```
