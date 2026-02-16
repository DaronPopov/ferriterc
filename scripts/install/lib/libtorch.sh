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
    # On Jetson, we may intentionally use a different libtorch version (NVIDIA
    # does not always publish the latest).  Accept any 2.x within the same
    # major series to avoid pointless re-downloads.
    local inst_major="${installed_ver%%.*}"
    local want_major="${LIBTORCH_VERSION%%.*}"
    if is_jetson 2>/dev/null && [[ "$inst_major" == "$want_major" ]]; then
      echo "[info] Jetson: accepting libtorch ${installed_ver} (requested ${LIBTORCH_VERSION})"
      return 0
    fi
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

# Detect whether we are running on an NVIDIA Jetson (Tegra) board.
# Jetson requires NVIDIA-built wheels with sm_72/sm_87 kernels — the standard
# PyTorch aarch64 wheel only ships sm_80+sm_90 (ARM server targets).
is_jetson() {
  # Check kernel version for "tegra"
  if uname -r 2>/dev/null | grep -qi tegra; then
    return 0
  fi
  # Check device-tree model string
  local model=""
  if [[ -r /proc/device-tree/model ]]; then
    model="$(tr -d '\0' < /proc/device-tree/model 2>/dev/null | tr '[:upper:]' '[:lower:]')"
  fi
  case "$model" in
    *jetson*|*orin*|*xavier*|*tx2*|*nano*) return 0 ;;
  esac
  return 1
}

# Download a torch wheel, extract it, and move torch/ → external/libtorch.
extract_wheel_to_libtorch() {
  local url="$1"
  local cache="$ROOT/external/.cache"
  local dst="$ROOT/external/libtorch"
  local whl="$cache/torch-aarch64.whl"

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

  rm -rf "$dst"
  mv "$extract_dir/torch" "$dst"
  rm -rf "$extract_dir"
}

# Try to download a Jetson-specific wheel from NVIDIA's Jetson AI Lab index.
# These wheels are compiled with sm_72 and sm_87 (Jetson Xavier/Orin).
# Falls back to the standard PyTorch wheel if the Jetson index is unreachable.
download_libtorch_jetson() {
  local cache="$ROOT/external/.cache"
  mkdir -p "$cache"

  # NVIDIA Jetson AI Lab index — best source for Jetson-native wheels.
  # Override with JETSON_TORCH_INDEX if the default changes.
  local jetson_index="${JETSON_TORCH_INDEX:-https://pypi.jetson-ai-lab.io}"
  local jetson_ver="${JETSON_TORCH_VERSION:-2.9.1}"
  local jetson_tag="${TORCH_CPYTHON_TAG:-cp310}"

  # Jetson wheels use the linux_aarch64 platform tag (not manylinux).
  local whl_name="torch-${jetson_ver}-${jetson_tag}-${jetson_tag}-linux_aarch64.whl"

  echo "[info] Jetson detected — trying NVIDIA Jetson AI Lab wheel"

  # The Jetson AI Lab index is a devpi server — scrape the actual download URL
  # from the simple index page rather than guessing the hash-based path.
  local simple_url="${jetson_index}/jp6/${LIBTORCH_CUDA_TAG}/+simple/torch/"
  local url
  url="$(curl -sL --max-time 15 "$simple_url" 2>/dev/null \
    | grep -oP 'href="\K[^"]+'"${jetson_tag}"'-'"${jetson_tag}"'-linux_aarch64\.whl[^"]*' \
    | grep "torch-${jetson_ver}" \
    | head -1)"

  # If simple index didn't work, try direct path
  if [[ -z "$url" ]]; then
    url="${jetson_index}/jp6/${LIBTORCH_CUDA_TAG}/${whl_name}"
  fi

  echo "[info] url: $url"

  if curl --head --silent --fail --max-time 10 "$url" >/dev/null 2>&1; then
    extract_wheel_to_libtorch "$url"
    return 0
  fi

  echo "[warn] Jetson wheel not available at: $url"

  # Fallback: NVIDIA developer redistribution index (older builds)
  local nv_url="https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/"
  echo "[info] trying NVIDIA developer index: $nv_url"
  local nv_whl
  nv_whl="$(curl -sL "$nv_url" 2>/dev/null | grep -oP 'href="\Ktorch-[^"]+linux_aarch64\.whl' | tail -1)"
  if [[ -n "$nv_whl" ]]; then
    extract_wheel_to_libtorch "${nv_url}${nv_whl}"
    return 0
  fi

  echo "[warn] no Jetson wheel found on NVIDIA indexes"
  return 1
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
  elif is_jetson; then
    # Jetson boards need NVIDIA-built wheels with sm_72/sm_87 kernels.
    # The standard PyTorch aarch64 wheel only contains sm_80+sm_90 (ARM server)
    # which causes "no kernel image available" at runtime on Orin/Xavier.
    if ! download_libtorch_jetson; then
      echo "[error] could not obtain a Jetson-compatible libtorch"
      echo "[hint] provide a Jetson torch wheel manually via --libtorch-url or LIBTORCH_URL"
      diag_emit "installer.libtorch" "FAIL" "INS-LIBTORCH-0007" "no Jetson-compatible libtorch available" "provide a Jetson torch wheel via --libtorch-url"
      exit 1
    fi
  else
    # Non-Jetson aarch64 (e.g. Grace Hopper, Ampere Altra) — standard wheel.
    local plat_tag="${TORCH_AARCH64_PLAT_TAG:-manylinux_2_28_aarch64}"
    local whl_name="torch-${LIBTORCH_VERSION}%2B${LIBTORCH_CUDA_TAG}-${TORCH_CPYTHON_TAG}-${TORCH_CPYTHON_TAG}-${plat_tag}.whl"
    local url="https://download.pytorch.org/whl/${LIBTORCH_CUDA_TAG}/${whl_name}"
    extract_wheel_to_libtorch "$url"
  fi

  if ! is_valid_libtorch "$dst"; then
    echo "[error] extracted libtorch is invalid or missing CUDA libraries: $dst"
    echo "[hint] ensure the wheel/archive contains CUDA-enabled torch for aarch64"
    diag_emit "installer.libtorch" "FAIL" "INS-LIBTORCH-0006" "extracted libtorch is invalid: $dst" "use CUDA-enabled torch/libtorch archive for aarch64"
    exit 1
  fi
  echo "[info] libtorch ready at: $dst"
}

# The PyTorch wheel's libtorch_cuda.so links NCCL, cuSPARSELt, and NVSHMEM
# for multi-GPU / sparse ops.  On single-GPU Jetson boards these libraries
# are absent and the code paths are never hit.  Create empty stub .so files
# so the dynamic loader is satisfied at runtime.
stub_missing_libtorch_deps() {
  local libdir="$1/lib"
  local libs="libcusparseLt.so.0 libnccl.so.2 libnvshmem_host.so.3 libcudss.so.0"
  for lib in $libs; do
    if [[ ! -f "$libdir/$lib" ]]; then
      echo "[info] creating stub for missing optional dep: $lib"
      echo "" | gcc -shared -x c - -o "$libdir/$lib" -Wl,-soname,"$lib" -nostdlib 2>/dev/null \
        || echo "[warn] failed to create stub $lib (gcc not available?)"
    fi
  done
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

  # On aarch64, PyTorch wheel bundles deps not present on single-GPU Jetson
  if [[ "$ARCH" == "aarch64" ]]; then
    stub_missing_libtorch_deps "$LIBTORCH"
  fi
}
