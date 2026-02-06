#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SM="${SM:-${CUDA_SM:-${GPU_SM:-}}}"
VERBOSE=false
LIBTORCH_VERSION="${LIBTORCH_VERSION:-2.3.0}"
LIBTORCH_CUDA_TAG="${LIBTORCH_CUDA_TAG:-}"
LIBTORCH_URL="${LIBTORCH_URL:-}"
LIBTORCH_ALLOW_PYTORCH="${LIBTORCH_ALLOW_PYTORCH:-0}"

usage() {
  cat <<'EOF'
Ferrite Runtime Source Installer

Usage:
  ./install.sh [--sm <SM>] [--verbose]

Options:
  --sm <SM>     GPU compute capability (e.g. 75, 80, 86, 89, 90)
  --verbose     Enable verbose build output
  -h, --help    Show this help

Environment variable fallback:
  SM / CUDA_SM / GPU_SM

Torch provisioning env (optional):
  LIBTORCH                   Existing libtorch root
  LIBTORCH_VERSION           Default: 2.3.0
  LIBTORCH_CUDA_TAG          Auto-detected from nvcc (or set explicitly)
  LIBTORCH_URL               Override download URL entirely
  LIBTORCH_ALLOW_PYTORCH     Default 0. Set 1 to allow Python torch fallback
EOF
}

need_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "[error] missing required tool: $cmd"
    exit 1
  fi
}

ensure_rust_toolchain() {
  if command -v cargo >/dev/null 2>&1; then
    return 0
  fi
  echo "[info] cargo not found; installing rust toolchain via rustup"
  need_cmd curl
  curl https://sh.rustup.rs -sSf | sh -s -- -y
  # shellcheck disable=SC1090
  source "$HOME/.cargo/env"
  need_cmd cargo
}

detect_cuda_tag() {
  if [[ -n "${LIBTORCH_CUDA_TAG}" ]]; then
    return 0
  fi
  if ! command -v nvcc >/dev/null 2>&1; then
    LIBTORCH_CUDA_TAG="cu121"
    echo "[warn] nvcc not found while selecting LIBTORCH_CUDA_TAG, defaulting to ${LIBTORCH_CUDA_TAG}"
    return 0
  fi
  local rel
  rel="$(nvcc --version | sed -n 's/.*release \([0-9]\+\)\.\([0-9]\+\).*/\1.\2/p' | head -n1)"
  case "$rel" in
    11.8) LIBTORCH_CUDA_TAG="cu118" ;;
    12.*) LIBTORCH_CUDA_TAG="cu121" ;;
    *)
      LIBTORCH_CUDA_TAG="cu121"
      echo "[warn] unsupported/unknown CUDA release '${rel}', defaulting to ${LIBTORCH_CUDA_TAG}"
      ;;
  esac
}

is_valid_libtorch() {
  local d="$1"
  [[ -d "$d" ]] || return 1
  [[ -f "$d/include/c10/cuda/CUDACachingAllocator.h" ]] || return 1
  [[ -f "$d/lib/libc10_cuda.so" || -f "$d/lib/libtorch_cuda.so" ]] || return 1
  return 0
}

detect_python_torch_root() {
  if ! command -v python3 >/dev/null 2>&1; then
    return 1
  fi
  python3 - <<'PY' 2>/dev/null
import os
try:
    import torch
except Exception:
    raise SystemExit(1)
cuda = getattr(torch.version, "cuda", None)
if not cuda:
    raise SystemExit(2)
print(os.path.dirname(torch.__file__))
PY
}

download_libtorch() {
  local dst="$ROOT/external/libtorch"
  local tmp="$ROOT/external/.cache"
  local arch
  arch="$(uname -m)"
  if [[ "$arch" != "x86_64" ]]; then
    echo "[error] automatic libtorch download currently supports x86_64 only. Set LIBTORCH manually."
    exit 1
  fi
  mkdir -p "$tmp"

  local url="$LIBTORCH_URL"
  if [[ -z "$url" ]]; then
    url="https://download.pytorch.org/libtorch/${LIBTORCH_CUDA_TAG}/libtorch-cxx11-abi-shared-with-deps-${LIBTORCH_VERSION}%2B${LIBTORCH_CUDA_TAG}.zip"
  fi
  local zip="$tmp/libtorch-${LIBTORCH_VERSION}-${LIBTORCH_CUDA_TAG}.zip"

  echo "[info] downloading libtorch from: $url"
  if command -v curl >/dev/null 2>&1; then
    curl --retry 5 --retry-delay 2 --retry-all-errors -fL "$url" -o "$zip"
  elif command -v wget >/dev/null 2>&1; then
    wget --tries=5 --waitretry=2 -O "$zip" "$url"
  else
    echo "[error] need curl or wget to auto-download libtorch"
    exit 1
  fi

  echo "[info] extracting libtorch to external/"
  cd "$ROOT/external"
  if command -v unzip >/dev/null 2>&1; then
    unzip -o "$zip" >/dev/null
  elif command -v bsdtar >/dev/null 2>&1; then
    bsdtar -xf "$zip"
  elif command -v python3 >/dev/null 2>&1; then
    python3 - "$zip" "$ROOT/external" <<'PY'
import sys, zipfile, pathlib
z = pathlib.Path(sys.argv[1])
out = pathlib.Path(sys.argv[2])
with zipfile.ZipFile(z) as f:
    f.extractall(out)
PY
  else
    echo "[error] need unzip, bsdtar, or python3 to extract libtorch archive"
    exit 1
  fi

  if ! is_valid_libtorch "$dst"; then
    echo "[error] downloaded libtorch is invalid or missing CUDA libraries: $dst"
    exit 1
  fi
  echo "[info] libtorch ready at: $dst"
}

ensure_libtorch() {
  if [[ -n "${LIBTORCH:-}" ]] && is_valid_libtorch "${LIBTORCH}"; then
    echo "[info] using LIBTORCH from env: ${LIBTORCH}"
    return 0
  fi

  if is_valid_libtorch "$ROOT/external/libtorch"; then
    export LIBTORCH="$ROOT/external/libtorch"
    echo "[info] using bundled libtorch: ${LIBTORCH}"
    return 0
  fi

  if [[ "${LIBTORCH_ALLOW_PYTORCH}" == "1" ]]; then
    local py_root
    if py_root="$(detect_python_torch_root)"; then
      if is_valid_libtorch "$py_root"; then
        export LIBTORCH="$py_root"
        echo "[info] using Python torch libraries: ${LIBTORCH}"
        return 0
      fi
    fi
  fi

  download_libtorch
  export LIBTORCH="$ROOT/external/libtorch"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sm)
      SM="${2:-}"
      shift 2
      ;;
    --verbose)
      VERBOSE=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[error] unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

need_cmd make
need_cmd gcc
need_cmd g++
need_cmd nvidia-smi
need_cmd nvcc
ensure_rust_toolchain
detect_cuda_tag
echo "[info] selected LIBTORCH_CUDA_TAG: ${LIBTORCH_CUDA_TAG}"

if [[ -z "${SM}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    DETECTED="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d '.[:space:]')"
    if [[ -n "${DETECTED}" ]]; then
      SM="${DETECTED}"
      echo "[info] auto-detected GPU SM: ${SM}"
    fi
  fi
fi

if [[ -z "${SM}" ]]; then
  echo "[error] GPU SM not resolved. Pass --sm <SM> (example: --sm 86)."
  exit 1
fi

if ! [[ "${SM}" =~ ^[0-9]{2}$ ]]; then
  echo "[error] invalid --sm value '${SM}'. Expected two digits like 75, 80, 86, 89, 90."
  exit 1
fi

export SM
export GPU_SM="${SM}"
export CUDA_SM="${SM}"
export PTX_GPU_SM="sm_${SM}"
export LIBTORCH_BYPASS_VERSION_CHECK="${LIBTORCH_BYPASS_VERSION_CHECK:-1}"

ensure_libtorch
export LD_LIBRARY_PATH="${ROOT}/ferrite-os/lib:${LIBTORCH}/lib:${LD_LIBRARY_PATH:-}"

echo "[info] using GPU SM: ${SM}"
echo "[1/5] Building ferrite-os"
cd "$ROOT/ferrite-os"
if [[ "${VERBOSE}" == "true" ]]; then
  make all
else
  make all
fi

echo "[2/5] Building ferrite-gpu-lang (torch feature)"
cd "$ROOT/ferrite-gpu-lang"
cargo build --release --features torch

echo "[3/5] Building external/ferrite-torch examples"
cd "$ROOT/external/ferrite-torch"
cargo build --release --examples

echo "[4/5] Building external/ferrite-xla example"
cd "$ROOT/external/ferrite-xla"
cargo build --release --example xla_allocator_test

echo "[5/5] Validating ferrite-gpu-lang torch+xla scripts"
cd "$ROOT/ferrite-gpu-lang"
cargo run --release --features torch --example script_cv_detect >/dev/null

echo "[ok] runtime source build complete (sm_${SM})"
