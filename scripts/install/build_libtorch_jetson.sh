#!/usr/bin/env bash
# build_libtorch_jetson.sh — Build PyTorch from source for Jetson (sm_87)
#
# The standard PyTorch aarch64 wheel only ships sm_80+sm_90 cubins (ARM server).
# Jetson Orin requires sm_87.  This script builds PyTorch from source with the
# correct CUDA arch and copies the resulting libtorch into external/libtorch.
#
# Usage:  ./scripts/install/build_libtorch_jetson.sh [--jobs N]
#
# Typical build time: 60-90 min on Orin AGX, 90-120 min on Orin NX/Nano.
set -euo pipefail

# Ensure pip-installed tools (cmake, ninja) are on PATH ahead of system ones
export PATH="$HOME/.local/bin:$PATH"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTORCH_VERSION="${LIBTORCH_VERSION:-2.9.0}"
BUILD_DIR="${ROOT}/external/.cache/pytorch-build"
# PyTorch's generated ATen files consume ~2-3GB RAM each during compilation.
# Default to 2 parallel jobs to stay within typical Jetson memory limits.
# Override with --jobs N if you have more RAM or swap.
TOTAL_MEM_GB="$(awk '/MemTotal/ {printf "%d", $2/1024/1024}' /proc/meminfo 2>/dev/null || echo 8)"
DEFAULT_JOBS=$(( TOTAL_MEM_GB / 4 ))
[[ "$DEFAULT_JOBS" -lt 1 ]] && DEFAULT_JOBS=1
JOBS="${1:-$DEFAULT_JOBS}"

if [[ "${1:-}" == "--jobs" ]]; then
  JOBS="${2:-$DEFAULT_JOBS}"
fi

echo "============================================="
echo " Building PyTorch ${PYTORCH_VERSION} for Jetson (sm_87)"
echo " Build dir:  ${BUILD_DIR}"
echo " Parallel:   ${JOBS} jobs"
echo "============================================="

# Prerequisites check
for cmd in python3 pip3 git cmake ninja; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "[error] missing required tool: $cmd"
    echo "[hint]  sudo apt-get install -y python3 python3-pip git cmake ninja-build"
    exit 1
  fi
done

# Install Python build deps
echo "[1/5] Installing Python build dependencies..."
pip3 install --user --quiet \
  "cmake>=3.18" \
  ninja \
  pyyaml \
  typing_extensions \
  numpy \
  setuptools \
  wheel

# Clone PyTorch source
echo "[2/5] Cloning PyTorch v${PYTORCH_VERSION}..."
mkdir -p "${BUILD_DIR}"
if [[ -d "${BUILD_DIR}/pytorch" ]]; then
  cd "${BUILD_DIR}/pytorch"
  # Ignore tag conflicts from PyTorch CI — they are harmless
  git fetch --tags 2>/dev/null || true
  git checkout "v${PYTORCH_VERSION}" 2>/dev/null || git checkout "tags/v${PYTORCH_VERSION}"
  git submodule sync --recursive
  git submodule update --init --recursive
else
  git clone --recursive --depth 1 --branch "v${PYTORCH_VERSION}" \
    https://github.com/pytorch/pytorch.git "${BUILD_DIR}/pytorch"
  cd "${BUILD_DIR}/pytorch"
fi

# Configure for Jetson Orin
echo "[3/5] Configuring build for Jetson Orin (sm_87)..."
export TORCH_CUDA_ARCH_LIST="8.7"
export USE_CUDA=1
export USE_CUDNN=1
export USE_NCCL=0
export USE_DISTRIBUTED=0
export USE_MKLDNN=0
export USE_XNNPACK=0
export USE_QNNPACK=0
export USE_FBGEMM=0
export USE_KINETO=0
export USE_LITE_INTERPRETER_PROFILER=OFF
export USE_NUMPY=0
export USE_FLASH_ATTENTION=0
export USE_MEM_EFF_ATTENTION=0
export BUILD_TEST=0
export BUILD_CAFFE2=0
export MAX_JOBS="${JOBS}"
export CMAKE_BUILD_TYPE=Release
export USE_PRIORITIZED_TEXT_FOR_LD=1

# Use ccache if available — drastically speeds up rebuilds.
# Use the symlink/PATH approach so cmake's assembler invocation also works.
if command -v ccache >/dev/null 2>&1; then
  echo "[info] ccache detected — enabling compiler cache"
  CCACHE_DIR="${BUILD_DIR}/ccache-bin"
  mkdir -p "${CCACHE_DIR}"
  ln -sf "$(which ccache)" "${CCACHE_DIR}/gcc"
  ln -sf "$(which ccache)" "${CCACHE_DIR}/g++"
  ln -sf "$(which ccache)" "${CCACHE_DIR}/cc"
  ln -sf "$(which ccache)" "${CCACHE_DIR}/c++"
  export PATH="${CCACHE_DIR}:${PATH}"
fi

# Build
echo "[4/5] Building PyTorch (this will take a while)..."
python3 setup.py build 2>&1 | tee "${BUILD_DIR}/build.log"

# Extract libtorch
echo "[5/5] Extracting libtorch to external/libtorch..."
TORCH_DIR="$(python3 -c "import os; d='${BUILD_DIR}/pytorch/build/lib'; print(d if os.path.isdir(d+'/torch') else '${BUILD_DIR}/pytorch/torch')")"

DST="${ROOT}/external/libtorch"
rm -rf "${DST}"
mkdir -p "${DST}/lib" "${DST}/include"

# Copy shared libraries
cp -a "${BUILD_DIR}/pytorch/build/lib/"*.so* "${DST}/lib/" 2>/dev/null || true
cp -a "${BUILD_DIR}/pytorch/build/lib/torch/lib/"*.so* "${DST}/lib/" 2>/dev/null || true
# Also check the torch package directory
find "${BUILD_DIR}/pytorch" -path "*/torch/lib/lib*.so*" -exec cp -a {} "${DST}/lib/" \; 2>/dev/null || true

# Copy headers
if [[ -d "${BUILD_DIR}/pytorch/torch/include" ]]; then
  cp -a "${BUILD_DIR}/pytorch/torch/include/"* "${DST}/include/"
elif [[ -d "${BUILD_DIR}/pytorch/build/include" ]]; then
  cp -a "${BUILD_DIR}/pytorch/build/include/"* "${DST}/include/"
fi

# Also copy torch API headers
if [[ -d "${BUILD_DIR}/pytorch/torch/csrc/api/include" ]]; then
  cp -a "${BUILD_DIR}/pytorch/torch/csrc/api/include/"* "${DST}/include/"
fi

# Write version file
echo "${PYTORCH_VERSION}" > "${DST}/build-version"

# Verify
if [[ -f "${DST}/lib/libtorch_cuda.so" && -f "${DST}/lib/libc10_cuda.so" ]]; then
  ARCHS="$(cuobjdump "${DST}/lib/libtorch_cuda.so" 2>/dev/null | grep "^arch" | sort -u || echo "unknown")"
  echo ""
  echo "============================================="
  echo " libtorch built successfully!"
  echo " Location: ${DST}"
  echo " CUDA archs: ${ARCHS}"
  echo "============================================="
  echo ""
  echo "Now re-run the installer:"
  echo "  ./scripts/install/install.sh"
else
  echo "[error] Build did not produce expected libraries."
  echo "[hint]  Check ${BUILD_DIR}/build.log for errors."
  exit 1
fi
