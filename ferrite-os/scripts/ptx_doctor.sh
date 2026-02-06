#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_SH="${ROOT_DIR}/scripts/ptx_env.sh"

if [ ! -x "$ENV_SH" ]; then
  echo "[ptx_doctor] ERROR: scripts/ptx_env.sh not found or not executable" >&2
  exit 1
fi

if ! "$ENV_SH" --format env --quiet >/dev/null 2>&1; then
  echo "[ptx_doctor] ERROR: CUDA toolkit not found. Set CUDA_PATH/CUDA_HOME." >&2
  exit 1
fi

while IFS='=' read -r key val; do
  if [ -n "$key" ]; then
    export "$key=$val"
  fi
done < <("$ENV_SH" --format env --quiet)

echo "PTX-OS Doctor"
echo "CUDA_PATH=${CUDA_PATH}"
echo "CUDA_LIB=${CUDA_LIB}"
echo "NVCC=${NVCC}"
echo "CUDA_VERSION=${CUDA_VERSION}"
echo "GPU_SM=${GPU_SM}"
echo "PTX_GPU_SM=${PTX_GPU_SM}"
echo

fail=0

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[ptx_doctor] WARN: nvidia-smi not found. GPU driver may be missing."
else
  nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null | head -1 | \
    sed 's/^/[ptx_doctor] GPU: /'
fi

if [ ! -x "$NVCC" ]; then
  echo "[ptx_doctor] ERROR: nvcc not found at ${NVCC}" >&2
  fail=1
fi

if [ ! -f "${ROOT_DIR}/lib/libptx_os.so" ]; then
  echo "[ptx_doctor] WARN: lib/libptx_os.so not built (run ./scripts/ptx_build.sh all)"
fi
if [ ! -f "${ROOT_DIR}/lib/libptx_hook.so" ]; then
  echo "[ptx_doctor] WARN: lib/libptx_hook.so not built (run ./scripts/ptx_build.sh all)"
fi

if [ "$fail" -ne 0 ]; then
  exit 1
fi

echo "[ptx_doctor] OK"
