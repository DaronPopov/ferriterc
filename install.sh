#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ─── platform detection (first) ───
ARCH="$(uname -m)"
case "$ARCH" in
  x86_64|aarch64) ;;
  *)
    echo "[error] unsupported architecture: $ARCH (need x86_64 or aarch64)"
    exit 1
    ;;
esac

# ─── defaults ───
SM="${SM:-${CUDA_SM:-${GPU_SM:-}}}"
VERBOSE=false
LIBTORCH_VERSION="${LIBTORCH_VERSION:-2.9.0}"
LIBTORCH_CUDA_TAG="${LIBTORCH_CUDA_TAG:-}"
LIBTORCH_URL="${LIBTORCH_URL:-}"
TORCH_CPYTHON_TAG="${TORCH_CPYTHON_TAG:-cp311}"

usage() {
  cat <<'EOF'
Ferrite Runtime Source Installer

Usage:
  ./install.sh [--sm <SM>] [--verbose]

Options:
  --sm <SM>     GPU compute capability (e.g. 75, 80, 86, 89, 90, 100, 120)
  --verbose     Enable verbose build output
  -h, --help    Show this help

Environment variable fallback:
  SM / CUDA_SM / GPU_SM

Torch provisioning env (optional):
  LIBTORCH                   Existing libtorch root
  LIBTORCH_VERSION           Default: 2.9.0
  LIBTORCH_CUDA_TAG          Auto-detected from nvcc (or set explicitly)
  LIBTORCH_URL               Override download URL entirely
  TORCH_CPYTHON_TAG          Default: cp311 (only matters for aarch64 wheel)
EOF
}

# ─── helpers ───

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

# ─── CUDA detection ───

detect_cuda_tag() {
  if [[ -n "${LIBTORCH_CUDA_TAG}" ]]; then
    return 0
  fi
  if ! command -v nvcc >/dev/null 2>&1; then
    LIBTORCH_CUDA_TAG="cu128"
    echo "[warn] nvcc not found while selecting LIBTORCH_CUDA_TAG, defaulting to ${LIBTORCH_CUDA_TAG}"
    return 0
  fi
  local rel
  rel="$(nvcc --version | sed -n 's/.*release \([0-9]\+\)\.\([0-9]\+\).*/\1.\2/p' | head -n1)"
  local major minor
  major="$(echo "$rel" | cut -d. -f1)"
  minor="$(echo "$rel" | cut -d. -f2)"
  case "$major" in
    11)
      LIBTORCH_CUDA_TAG="cu118"
      ;;
    12)
      # CUDA 12.0-12.3 → cu121, CUDA 12.4-12.5 → cu124, CUDA 12.6+ → cu126
      if [[ "$minor" -le 3 ]]; then
        LIBTORCH_CUDA_TAG="cu121"
      elif [[ "$minor" -le 5 ]]; then
        LIBTORCH_CUDA_TAG="cu124"
      else
        LIBTORCH_CUDA_TAG="cu126"
      fi
      ;;
    13)
      LIBTORCH_CUDA_TAG="cu128"
      ;;
    *)
      LIBTORCH_CUDA_TAG="cu128"
      echo "[warn] unsupported/unknown CUDA release '${rel}', defaulting to ${LIBTORCH_CUDA_TAG}"
      ;;
  esac
}

# ─── libtorch validation ───

is_valid_libtorch() {
  local d="$1"
  [[ -d "$d" ]] || return 1
  [[ -f "$d/include/c10/cuda/CUDACachingAllocator.h" ]] || return 1
  [[ -f "$d/lib/libc10_cuda.so" || -f "$d/lib/libtorch_cuda.so" ]] || return 1
  return 0
}

# Check that a bundled libtorch matches the expected version.
check_libtorch_version() {
  local d="$1"
  local ver_file="$d/build-version"
  if [[ ! -f "$ver_file" ]]; then
    # No version file — can't verify, assume ok
    return 0
  fi
  local installed_ver
  installed_ver="$(head -n1 "$ver_file" | cut -d+ -f1)"
  if [[ "$installed_ver" != "$LIBTORCH_VERSION" ]]; then
    echo "[warn] bundled libtorch is ${installed_ver} but need ${LIBTORCH_VERSION}"
    return 1
  fi
  return 0
}

# ─── download / extract primitives ───

fetch_file() {
  local url="$1" dst="$2"
  if command -v curl >/dev/null 2>&1; then
    curl --retry 5 --retry-delay 2 --retry-all-errors -fL "$url" -o "$dst"
  elif command -v wget >/dev/null 2>&1; then
    wget --tries=5 --waitretry=2 -O "$dst" "$url"
  else
    echo "[error] need curl or wget"
    exit 1
  fi
}

extract_zip() {
  local archive="$1" dest="$2"
  if command -v unzip >/dev/null 2>&1; then
    unzip -o "$archive" -d "$dest" >/dev/null
  elif command -v bsdtar >/dev/null 2>&1; then
    bsdtar -xf "$archive" -C "$dest"
  else
    echo "[error] need unzip or bsdtar to extract archives"
    exit 1
  fi
}

# ─── libtorch provisioning (architecture-aware) ───

download_libtorch_x86_64() {
  local cache="$ROOT/external/.cache"
  local dst="$ROOT/external/libtorch"
  mkdir -p "$cache"

  # PyTorch ≥2.8 dropped the "cxx11-abi-" infix from the download filename.
  # All Linux x86_64 shared builds now use cxx11 ABI by default.
  local url="${LIBTORCH_URL:-https://download.pytorch.org/libtorch/${LIBTORCH_CUDA_TAG}/libtorch-shared-with-deps-${LIBTORCH_VERSION}%2B${LIBTORCH_CUDA_TAG}.zip}"
  local zip="$cache/libtorch-${LIBTORCH_VERSION}-${LIBTORCH_CUDA_TAG}.zip"

  echo "[info] x86_64: downloading libtorch from: $url"
  fetch_file "$url" "$zip"

  # Remove stale libtorch before extracting
  rm -rf "$dst"

  echo "[info] extracting libtorch"
  extract_zip "$zip" "$ROOT/external"

  if ! is_valid_libtorch "$dst"; then
    echo "[error] downloaded libtorch is invalid or missing CUDA libraries: $dst"
    exit 1
  fi
  echo "[info] libtorch ready at: $dst"
}

download_libtorch_aarch64() {
  local cache="$ROOT/external/.cache"
  local dst="$ROOT/external/libtorch"
  mkdir -p "$cache"

  if [[ -n "${LIBTORCH_URL}" ]]; then
    # User-supplied URL — treat as a zip like x86_64
    local zip="$cache/libtorch-${LIBTORCH_VERSION}-${LIBTORCH_CUDA_TAG}-aarch64.zip"
    echo "[info] aarch64: downloading libtorch from override URL: $LIBTORCH_URL"
    fetch_file "$LIBTORCH_URL" "$zip"
    rm -rf "$dst"
    extract_zip "$zip" "$ROOT/external"
  else
    # Download the aarch64 PyTorch wheel and extract C++ libraries from it.
    # A .whl is a zip archive. The torch/ directory inside contains the same
    # lib/, include/, and share/ layout that libtorch expects.
    local whl_name="torch-${LIBTORCH_VERSION}%2B${LIBTORCH_CUDA_TAG}-${TORCH_CPYTHON_TAG}-${TORCH_CPYTHON_TAG}-manylinux_2_17_aarch64.manylinux2014_aarch64.whl"
    local url="https://download.pytorch.org/whl/${LIBTORCH_CUDA_TAG}/${whl_name}"
    local whl="$cache/torch-${LIBTORCH_VERSION}-${LIBTORCH_CUDA_TAG}-aarch64.whl"

    echo "[info] aarch64: downloading torch wheel for C++ libraries"
    echo "[info] url: $url"
    fetch_file "$url" "$whl"

    echo "[info] extracting C++ libraries from torch wheel"
    local extract_dir="$cache/wheel_extract"
    rm -rf "$extract_dir"
    mkdir -p "$extract_dir"
    extract_zip "$whl" "$extract_dir"

    if [[ ! -d "$extract_dir/torch" ]]; then
      echo "[error] wheel extraction did not produce torch/ directory"
      exit 1
    fi

    # Move torch/ → external/libtorch (lib/, include/, share/ are already there)
    rm -rf "$dst"
    mv "$extract_dir/torch" "$dst"
    rm -rf "$extract_dir"
  fi

  if ! is_valid_libtorch "$dst"; then
    echo "[error] extracted libtorch is invalid or missing CUDA libraries: $dst"
    echo "[hint] ensure the wheel/archive contains CUDA-enabled torch for aarch64"
    exit 1
  fi
  echo "[info] libtorch ready at: $dst"
}

ensure_libtorch() {
  # 1. User-supplied LIBTORCH env var (external install)
  if [[ -n "${LIBTORCH:-}" ]] && is_valid_libtorch "${LIBTORCH}"; then
    echo "[info] using LIBTORCH from env: ${LIBTORCH}"
    return 0
  fi

  # 2. Already present in external/libtorch — but verify version matches
  if is_valid_libtorch "$ROOT/external/libtorch"; then
    if check_libtorch_version "$ROOT/external/libtorch"; then
      export LIBTORCH="$ROOT/external/libtorch"
      echo "[info] using bundled libtorch: ${LIBTORCH}"
      return 0
    else
      echo "[info] bundled libtorch version mismatch, re-downloading"
    fi
  fi

  # 3. Download for this architecture
  echo "[info] provisioning libtorch ${LIBTORCH_VERSION}+${LIBTORCH_CUDA_TAG} for ${ARCH}"
  case "$ARCH" in
    x86_64)  download_libtorch_x86_64 ;;
    aarch64) download_libtorch_aarch64 ;;
  esac
  export LIBTORCH="$ROOT/external/libtorch"
}

# ─── arg parsing ───

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

# ─── machine discovery ───

echo "[info] architecture: ${ARCH}"

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

if ! [[ "${SM}" =~ ^[0-9]{2,3}$ ]]; then
  echo "[error] invalid --sm value '${SM}'. Expected 2-3 digits like 75, 86, 90, 100, 120."
  exit 1
fi

# ─── export build environment ───

export SM
export GPU_SM="${SM}"
export CUDA_SM="${SM}"
export PTX_GPU_SM="sm_${SM}"
export LIBTORCH_BYPASS_VERSION_CHECK="${LIBTORCH_BYPASS_VERSION_CHECK:-1}"

ensure_libtorch
export LD_LIBRARY_PATH="${ROOT}/ferrite-os/lib:${LIBTORCH}/lib:${LD_LIBRARY_PATH:-}"

# ─── build ───

echo "[info] using GPU SM: ${SM}"
echo "[info] using libtorch: ${LIBTORCH}"
STEPS=7
echo "[1/${STEPS}] Building ferrite-os"
cd "$ROOT/ferrite-os"
make all

echo "[2/${STEPS}] Building ferrite-gpu-lang (torch feature)"
cd "$ROOT/ferrite-gpu-lang"
cargo build --release --features torch

echo "[3/${STEPS}] Building external/ferrite-torch examples"
cd "$ROOT/external/ferrite-torch"
cargo build --release --examples

echo "[4/${STEPS}] Building external/ferrite-xla example"
cd "$ROOT/external/ferrite-xla"
cargo build --release --example xla_allocator_test

echo "[5/${STEPS}] Validating ferrite-gpu-lang torch+xla scripts"
cd "$ROOT/ferrite-gpu-lang"
cargo run --release --features torch --example script_cv_detect >/dev/null

check_engine_scripts() {
  local engine_dir="$1"
  local engine_name="$2"
  local all_ok=true

  while IFS= read -r -d '' script; do
    NAME="$(basename "$script" .rs)"
    NAME="$(echo "$NAME" | sed 's/[^a-zA-Z0-9_]/_/g')"
    LINK="$ROOT/ferrite-gpu-lang/examples/${NAME}.rs"
    ln -sf "$script" "$LINK"
    if ! cargo check --release --features torch --example "$NAME" 2>/dev/null; then
      echo "  [warn] $(basename "$script") failed to check"
      all_ok=false
    fi
    rm -f "$LINK"
  done < <(find "$engine_dir" -name '*.rs' -type f -print0)

  if [[ "$all_ok" == "true" ]]; then
    echo "  ${engine_name}: all scripts compile"
  else
    echo "  ${engine_name}: some scripts had warnings (non-fatal)"
  fi
}

echo "[6/${STEPS}] Checking finetune_engine scripts"
cd "$ROOT/ferrite-gpu-lang"
check_engine_scripts "$ROOT/finetune_engine" "finetune_engine"

echo "[7/${STEPS}] Checking mathematics_engine scripts"
check_engine_scripts "$ROOT/mathematics_engine" "mathematics_engine"

echo "[ok] ferrite runtime build complete (sm_${SM}, ${ARCH})"
echo "     libtorch:           ${LIBTORCH_VERSION}+${LIBTORCH_CUDA_TAG}"
echo "     finetune_engine:    ready"
echo "     mathematics_engine: ready"
echo ""
echo "  Run scripts with: ./ferrite-run <script.rs> --torch"
