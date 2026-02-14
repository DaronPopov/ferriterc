# cuda.sh — CUDA toolkit, CUPTI, SM resolution
# Sourced by scripts/install/install.sh

resolve_cuda_path() {
  local cand
  for cand in "${CUDA_PATH:-}" "${CUDA_HOME:-}" "/usr/local/cuda"; do
    if [[ -n "$cand" && -f "$cand/include/cuda_runtime.h" ]]; then
      echo "$cand"
      return 0
    fi
  done
  if command -v nvcc >/dev/null 2>&1; then
    local nvcc_path
    nvcc_path="$(command -v nvcc)"
    local guess
    guess="$(cd "$(dirname "$nvcc_path")/.." && pwd)"
    if [[ -f "$guess/include/cuda_runtime.h" ]]; then
      echo "$guess"
      return 0
    fi
  fi
  return 1
}

ensure_cuda_toolkit() {
  if cmd_exists nvcc; then
    return 0
  fi

  if [[ "${AUTO_INSTALL_CUDA:-false}" != "true" ]]; then
    echo "[error] nvcc not found (CUDA toolkit is required and auto-install is disabled)"
    echo "[hint] install CUDA manually, then re-run installer:"
    echo "       https://developer.nvidia.com/cuda-downloads"
    echo "[hint] or opt in to legacy behavior: ./scripts/install.sh --auto-install-cuda"
    diag_emit "installer.cuda" "FAIL" "INS-CUDA-0013" "nvcc missing and CUDA auto-install disabled" "install CUDA toolkit manually or pass --auto-install-cuda"
    exit 1
  fi

  echo "[info] nvcc not found; attempting CUDA toolkit install"

  if cmd_exists apt-get; then
    # Try NVIDIA repo package first (provides latest toolkit)
    if sudo_cmd apt-get update -y && sudo_cmd apt-get install -y cuda-toolkit 2>/dev/null; then
      true
    else
      # Fallback to distro package
      sudo_cmd apt-get install -y nvidia-cuda-toolkit
    fi
  elif cmd_exists dnf; then
    sudo_cmd dnf install -y cuda-toolkit || sudo_cmd dnf install -y nvidia-cuda-toolkit
  elif cmd_exists yum; then
    sudo_cmd yum install -y cuda-toolkit
  elif cmd_exists pacman; then
    sudo_cmd pacman -Sy --noconfirm cuda
  elif cmd_exists zypper; then
    sudo_cmd zypper --non-interactive install cuda-toolkit
  else
    echo "[error] cannot auto-install CUDA toolkit on this distro"
    echo "[hint] install the CUDA toolkit manually: https://developer.nvidia.com/cuda-downloads"
    diag_emit "installer.cuda" "FAIL" "INS-CUDA-0010" "cannot auto-install CUDA toolkit" "install CUDA toolkit manually and re-run"
    exit 1
  fi

  # Update PATH to find newly installed nvcc
  for d in /usr/local/cuda/bin /usr/local/cuda-*/bin /opt/cuda/bin; do
    if [[ -x "$d/nvcc" ]]; then
      export PATH="$d:$PATH"
      break
    fi
  done

  if ! cmd_exists nvcc; then
    echo "[error] CUDA toolkit install completed but nvcc still not found"
    echo "[hint] ensure /usr/local/cuda/bin is on your PATH, or set CUDA_PATH"
    diag_emit "installer.cuda" "FAIL" "INS-CUDA-0011" "nvcc not on PATH after install" "add /usr/local/cuda/bin to PATH or set CUDA_PATH"
    exit 1
  fi

  echo "[info] CUDA toolkit installed, nvcc: $(command -v nvcc)"
  diag_emit "installer.cuda" "PASS" "INS-CUDA-0012" "CUDA toolkit auto-installed" "none"
}

has_cupti() {
  if ldconfig -p 2>/dev/null | grep -q 'libcupti\.so'; then
    return 0
  fi
  local cuda_root="${1:-}"
  if [[ -n "$cuda_root" ]]; then
    if [[ -f "$cuda_root/targets/x86_64-linux/lib/libcupti.so" || -f "$cuda_root/targets/aarch64-linux/lib/libcupti.so" || -f "$cuda_root/extras/CUPTI/lib64/libcupti.so" ]]; then
      return 0
    fi
  fi
  return 1
}

cuda_mm_dash() {
  local rel major minor
  rel="$(nvcc --version | sed -n 's/.*release \([0-9]\+\)\.\([0-9]\+\).*/\1.\2/p' | head -n1)"
  major="$(echo "$rel" | cut -d. -f1)"
  minor="$(echo "$rel" | cut -d. -f2)"
  echo "${major}-${minor}"
}

install_cupti_linux() {
  local mm_dash="$1"
  if command -v apt-get >/dev/null 2>&1; then
    sudo_cmd apt-get update -y
    local pkgs=(
      "cuda-cupti-${mm_dash}"
      "cuda-cupti-dev-${mm_dash}"
      "cuda-command-line-tools-${mm_dash}"
      "nvidia-cuda-toolkit"
    )
    local p
    for p in "${pkgs[@]}"; do
      if sudo_cmd apt-get install -y "$p"; then
        return 0
      fi
    done
    return 1
  elif command -v dnf >/dev/null 2>&1; then
    sudo_cmd dnf install -y "cuda-cupti-${mm_dash}" "cuda-cupti-devel-${mm_dash}" || \
      sudo_cmd dnf install -y cuda-toolkit || return 1
    return 0
  elif command -v yum >/dev/null 2>&1; then
    sudo_cmd yum install -y "cuda-cupti-${mm_dash}" "cuda-cupti-devel-${mm_dash}" || \
      sudo_cmd yum install -y cuda-toolkit || return 1
    return 0
  elif command -v pacman >/dev/null 2>&1; then
    sudo_cmd pacman -Sy --noconfirm cuda cuda-tools || return 1
    return 0
  elif command -v zypper >/dev/null 2>&1; then
    sudo_cmd zypper --non-interactive install "cuda-cupti-${mm_dash}" || \
      sudo_cmd zypper --non-interactive install cuda-toolkit || return 1
    return 0
  fi
  return 1
}

ensure_cupti() {
  local cuda_root="$1"
  if has_cupti "$cuda_root"; then
    echo "[info] CUPTI detected"
    return 0
  fi
  if [[ "$(uname -s)" != "Linux" ]]; then
    echo "[warn] CUPTI auto-install currently supports Linux package managers"
    diag_emit "installer.cuda" "WARN" "INS-CUDA-0001" "CUPTI auto-install unsupported on this OS" "install CUPTI manually if required"
    return 1
  fi

  local mm_dash
  mm_dash="$(cuda_mm_dash)"
  echo "[info] CUPTI not found; attempting install for CUDA ${mm_dash}"
  if install_cupti_linux "$mm_dash" && has_cupti "$cuda_root"; then
    echo "[info] CUPTI installed successfully"
    diag_emit "installer.cuda" "PASS" "INS-CUDA-0002" "CUPTI installed successfully" "none"
    return 0
  fi
  echo "[error] failed to install CUPTI automatically"
  echo "[hint] install package cuda-cupti-${mm_dash} (or cuda-cupti-dev-${mm_dash})"
  diag_emit "installer.cuda" "FAIL" "INS-CUDA-0003" "failed to install CUPTI automatically" "install cuda-cupti-${mm_dash} or cuda-cupti-dev-${mm_dash}"
  return 1
}

resolve_cupti_lib_dir() {
  local cuda_root="$1"
  local arch="$(uname -m)"
  local target
  case "$arch" in
    x86_64) target="x86_64-linux" ;;
    aarch64) target="aarch64-linux" ;;
    *) target="" ;;
  esac
  if [[ -n "$target" && -d "$cuda_root/targets/$target/lib" ]]; then
    echo "$cuda_root/targets/$target/lib"
    return 0
  fi
  if [[ -d "$cuda_root/extras/CUPTI/lib64" ]]; then
    echo "$cuda_root/extras/CUPTI/lib64"
    return 0
  fi
  return 1
}

detect_sm_with_nvcc() {
  local cuda_root="$1"
  local src="$ROOT/ferrite-os/tooling/scripts/detect_sm.cu"
  local tmp
  tmp="$(mktemp -t ferrite_detect_sm.XXXXXX)"
  if "$cuda_root/bin/nvcc" -O2 -std=c++11 "$src" -o "$tmp" >/dev/null 2>&1; then
    local sm
    sm="$("$tmp" 2>/dev/null | tr -cd '0-9')"
    rm -f "$tmp"
    if [[ -n "$sm" ]]; then
      echo "$sm"
      return 0
    fi
  else
    rm -f "$tmp"
  fi
  return 1
}

resolve_cuda_compat() {
  local compat_script="$ROOT/scripts/resolve_cuda_compat.sh"
  if [[ ! -x "$compat_script" ]]; then
    echo "[error] missing compat resolver: $compat_script"
    diag_emit "installer.compat" "FAIL" "INS-COMPAT-0001" "missing compat resolver script" "restore scripts/resolve_cuda_compat.sh"
    exit 1
  fi

  local compat_out
  compat_out="$("$compat_script" --format env --quiet)"

  local detected_cudarc_feature=""
  local detected_libtorch_tag=""
  while IFS='=' read -r k v; do
    case "$k" in
      CUDARC_CUDA_FEATURE) detected_cudarc_feature="$v" ;;
      LIBTORCH_CUDA_TAG_RESOLVED) detected_libtorch_tag="$v" ;;
    esac
  done <<< "$compat_out"

  if [[ -z "${CUDARC_CUDA_FEATURE}" ]]; then
    CUDARC_CUDA_FEATURE="$detected_cudarc_feature"
  fi
  if [[ -z "${LIBTORCH_CUDA_TAG}" ]]; then
    LIBTORCH_CUDA_TAG="$detected_libtorch_tag"
  fi

  if [[ -z "${CUDARC_CUDA_FEATURE}" ]]; then
    echo "[error] failed to resolve CUDARC_CUDA_FEATURE from compat.toml"
    diag_emit "installer.compat" "FAIL" "INS-COMPAT-0002" "failed to resolve CUDARC_CUDA_FEATURE" "add mapping to compat.toml or pass --cudarc-feature"
    exit 1
  fi
  if [[ -z "${LIBTORCH_CUDA_TAG}" ]]; then
    echo "[error] failed to resolve LIBTORCH_CUDA_TAG from compat.toml"
    diag_emit "installer.compat" "FAIL" "INS-COMPAT-0003" "failed to resolve LIBTORCH_CUDA_TAG" "add mapping to compat.toml or pass --libtorch-tag"
    exit 1
  fi
}

# ─── orchestration helpers called from runner ───

setup_cuda_env() {
  CUDA_PATH_RESOLVED="$(resolve_cuda_path || true)"
  if [[ -z "$CUDA_PATH_RESOLVED" ]]; then
    echo "[error] failed to resolve CUDA toolkit path"
    diag_emit "installer.cuda" "FAIL" "INS-CUDA-0004" "failed to resolve CUDA toolkit path" "set CUDA_PATH/CUDA_HOME or install CUDA toolkit"
    exit 1
  fi
  export CUDA_PATH="$CUDA_PATH_RESOLVED"
  export CUDA_HOME="$CUDA_PATH_RESOLVED"
  ensure_cupti "$CUDA_PATH_RESOLVED" || true
  CUPTI_LIB_DIR="$(resolve_cupti_lib_dir "$CUDA_PATH_RESOLVED" || true)"
  if [[ -n "${CUPTI_LIB_DIR:-}" ]]; then
    export CUPTI_LIB_DIR
    export LD_LIBRARY_PATH="${CUPTI_LIB_DIR}:${LD_LIBRARY_PATH:-}"
  fi
}

auto_detect_sm() {
  if [[ -z "${SM}" ]]; then
    if cmd_exists nvidia-smi; then
      DETECTED="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d '.[:space:]')"
      if [[ -n "${DETECTED}" ]]; then
        SM="${DETECTED}"
        echo "[info] auto-detected GPU SM: ${SM}"
      fi
    fi
  fi

  if [[ -z "${SM}" ]]; then
    DETECTED_NVCC="$(detect_sm_with_nvcc "$CUDA_PATH_RESOLVED" || true)"
    if [[ -n "${DETECTED_NVCC:-}" ]]; then
      SM="${DETECTED_NVCC}"
      echo "[info] auto-detected GPU SM via nvcc probe: ${SM}"
    fi
  fi

  if [[ -z "${SM}" ]]; then
    echo "[error] GPU SM not resolved. Pass --sm <SM> (example: --sm 86)."
    diag_emit "installer.cuda" "FAIL" "INS-CUDA-0005" "GPU SM not resolved" "pass --sm <SM> (example: --sm 86)"
    exit 1
  fi

  if ! [[ "${SM}" =~ ^[0-9]{2,3}$ ]]; then
    echo "[error] invalid --sm value '${SM}'. Expected 2-3 digits like 75, 86, 90, 100, 120."
    diag_emit "installer.cuda" "FAIL" "INS-CUDA-0006" "invalid --sm value '${SM}'" "use two or three digits such as 75, 86, 90, 100, 120"
    exit 1
  fi
}

export_build_env() {
  export SM
  export GPU_SM="${SM}"
  export CUDA_SM="${SM}"
  export PTX_GPU_SM="sm_${SM}"
  export CUDA_ARCH="sm_${SM}"
  export CUDARC_CUDA_FEATURE
  export LIBTORCH_BYPASS_VERSION_CHECK="${LIBTORCH_BYPASS_VERSION_CHECK:-1}"
}
