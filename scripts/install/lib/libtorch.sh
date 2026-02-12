# libtorch.sh — download, verify, and provision libtorch
# Sourced by scripts/install/install.sh

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
    diag_emit "installer.libtorch" "WARN" "INS-LIBTORCH-0001" "bundled libtorch version mismatch (${installed_ver} != ${LIBTORCH_VERSION})" "installer will re-download compatible libtorch"
    return 1
  fi
  return 0
}

fetch_file() {
  local url="$1" dst="$2"
  if command -v curl >/dev/null 2>&1; then
    curl --retry 5 --retry-delay 2 --retry-all-errors -fL "$url" -o "$dst"
  elif command -v wget >/dev/null 2>&1; then
    wget --tries=5 --waitretry=2 -O "$dst" "$url"
  else
    echo "[error] need curl or wget"
    diag_emit "installer.libtorch" "FAIL" "INS-LIBTORCH-0002" "missing downloader (curl/wget)" "install curl or wget"
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
    diag_emit "installer.libtorch" "FAIL" "INS-LIBTORCH-0003" "missing archive extractor (unzip/bsdtar)" "install unzip or bsdtar"
    exit 1
  fi
}

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
    diag_emit "installer.libtorch" "FAIL" "INS-LIBTORCH-0004" "downloaded libtorch is invalid: $dst" "verify LIBTORCH_URL/LIBTORCH_CUDA_TAG and retry"
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
      diag_emit "installer.libtorch" "FAIL" "INS-LIBTORCH-0005" "torch/ directory missing in extracted wheel" "use a compatible aarch64 torch wheel"
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
    diag_emit "installer.libtorch" "FAIL" "INS-LIBTORCH-0006" "extracted libtorch is invalid: $dst" "use CUDA-enabled torch/libtorch archive for aarch64"
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
