# env.sh — platform helpers, package management, CLI utilities
# Sourced by scripts/install/install.sh

if ! declare -F diag_emit >/dev/null 2>&1; then
  diag_emit() { :; }
fi

usage() {
  cat <<'EOF'
Ferrite Runtime Source Installer

Usage:
  ./install.sh [--sm <SM>] [--verbose] [--enable-service]
               [--libtorch-url <URL>] [--libtorch-tag <TAG>] [--cudarc-feature <FEATURE>]
               [--pins "<k=v,k=v,...>"]

Options:
  --sm <SM>     GPU compute capability (e.g. 75, 80, 86, 89, 90, 100, 120)
  --verbose     Enable verbose build output
  --enable-service
                Install and enable ferrite-daemon systemd service for boot startup
  --core-only     Build only the core OS (daemon, TUI, ptx-runtime).
                  Skips libtorch download and torch-dependent crates.
  --libtorch-url <URL>
                Pin an exact external libtorch archive/wheel URL
  --libtorch-tag <TAG>
                Pin libtorch CUDA tag directly (e.g. cu126, cu128)
  --cudarc-feature <FEATURE>
                Pin cudarc CUDA feature directly (e.g. cuda-12060)
  --pins "<k=v,...>"
                Pin multiple values in one quoted string. Keys:
                sm, libtorch_url, libtorch_tag, cudarc_feature
  -h, --help    Show this help

Environment variable fallback:
  SM / CUDA_SM / GPU_SM

Torch provisioning env (optional):
  LIBTORCH                   Existing libtorch root
  LIBTORCH_VERSION           Default: 2.9.0
  LIBTORCH_CUDA_TAG          Auto-selected from compat.toml (or set explicitly)
  CUDARC_CUDA_FEATURE        Auto-selected from compat.toml (or set explicitly)
  LIBTORCH_URL               Override download URL entirely
  TORCH_CPYTHON_TAG          Default: cp311 (only matters for aarch64 wheel)
EOF
}

need_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "[error] missing required tool: $cmd"
    diag_emit "installer.preflight" "FAIL" "INS-PREF-0001" "missing required tool: $cmd" "install '$cmd' and re-run installer"
    exit 1
  fi
}

cmd_exists() {
  command -v "$1" >/dev/null 2>&1
}

sudo_cmd() {
  if [[ "${EUID:-$(id -u)}" -eq 0 ]]; then
    "$@"
  elif command -v sudo >/dev/null 2>&1; then
    sudo "$@"
  else
    echo "[error] need root privileges (or sudo) to install CUPTI"
    diag_emit "installer.env" "FAIL" "INS-ENV-0001" "root privileges required for CUPTI install" "run as root or install sudo"
    exit 1
  fi
}

pm_install_pkgs() {
  local pkgs=("$@")
  if [[ ${#pkgs[@]} -eq 0 ]]; then
    return 0
  fi

  if cmd_exists apt-get; then
    sudo_cmd apt-get update -y
    sudo_cmd apt-get install -y "${pkgs[@]}"
    return 0
  fi
  if cmd_exists dnf; then
    sudo_cmd dnf install -y "${pkgs[@]}"
    return 0
  fi
  if cmd_exists yum; then
    sudo_cmd yum install -y "${pkgs[@]}"
    return 0
  fi
  if cmd_exists pacman; then
    sudo_cmd pacman -Sy --noconfirm "${pkgs[@]}"
    return 0
  fi
  if cmd_exists zypper; then
    sudo_cmd zypper --non-interactive install "${pkgs[@]}"
    return 0
  fi
  return 1
}

ensure_host_build_tools() {
  local missing=()
  local c
  for c in make gcc g++ git; do
    if ! cmd_exists "$c"; then
      missing+=("$c")
    fi
  done

  if [[ ${#missing[@]} -eq 0 ]]; then
    return 0
  fi

  echo "[info] missing host tools: ${missing[*]}"
  echo "[info] attempting automatic installation"

  if cmd_exists apt-get; then
    pm_install_pkgs build-essential git ca-certificates pkg-config
  elif cmd_exists dnf || cmd_exists yum; then
    pm_install_pkgs gcc gcc-c++ make git pkgconf-pkg-config ca-certificates
  elif cmd_exists pacman; then
    pm_install_pkgs base-devel git pkgconf ca-certificates
  elif cmd_exists zypper; then
    pm_install_pkgs gcc gcc-c++ make git pkg-config ca-certificates
  else
    echo "[error] unsupported package manager for auto-installing build tools"
    echo "[hint] install manually: make gcc g++ git"
    diag_emit "installer.preflight" "FAIL" "INS-PREF-0002" "unsupported package manager for auto-installing host build tools" "install make gcc g++ git manually"
    exit 1
  fi

  for c in make gcc g++ git; do
    need_cmd "$c"
  done
}

ensure_fetch_tools() {
  if ! cmd_exists curl && ! cmd_exists wget; then
    echo "[info] missing download tool (curl/wget), attempting install"
    if cmd_exists apt-get; then
      pm_install_pkgs curl wget ca-certificates
    elif cmd_exists dnf || cmd_exists yum; then
      pm_install_pkgs curl wget ca-certificates
    elif cmd_exists pacman; then
      pm_install_pkgs curl wget ca-certificates
    elif cmd_exists zypper; then
      pm_install_pkgs curl wget ca-certificates
    else
      echo "[error] cannot auto-install curl/wget on this distro"
      diag_emit "installer.preflight" "FAIL" "INS-PREF-0003" "cannot auto-install curl/wget" "install curl or wget manually"
      exit 1
    fi
  fi

  if ! cmd_exists unzip && ! cmd_exists bsdtar; then
    echo "[info] missing archive extractor (unzip/bsdtar), attempting install"
    if cmd_exists apt-get; then
      pm_install_pkgs unzip libarchive-tools
    elif cmd_exists dnf || cmd_exists yum; then
      pm_install_pkgs unzip bsdtar
    elif cmd_exists pacman; then
      pm_install_pkgs unzip libarchive
    elif cmd_exists zypper; then
      pm_install_pkgs unzip bsdtar
    else
      echo "[error] cannot auto-install unzip/bsdtar on this distro"
      diag_emit "installer.preflight" "FAIL" "INS-PREF-0004" "cannot auto-install unzip/bsdtar" "install unzip or bsdtar manually"
      exit 1
    fi
  fi

  if ! cmd_exists curl && ! cmd_exists wget; then
    echo "[error] still missing curl/wget after auto-install attempt"
    diag_emit "installer.preflight" "FAIL" "INS-PREF-0005" "download tool still missing after auto-install attempt" "install curl or wget manually"
    exit 1
  fi
  if ! cmd_exists unzip && ! cmd_exists bsdtar; then
    echo "[error] still missing unzip/bsdtar after auto-install attempt"
    diag_emit "installer.preflight" "FAIL" "INS-PREF-0006" "archive extractor still missing after auto-install attempt" "install unzip or bsdtar manually"
    exit 1
  fi
}

apply_pins() {
  local spec="$1"
  # Support comma or semicolon separators.
  spec="${spec//;/,}"
  local IFS=','
  local parts=()
  read -r -a parts <<< "$spec"

  local part key val
  for part in "${parts[@]}"; do
    # Trim surrounding whitespace
    part="$(echo "$part" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
    [[ -z "$part" ]] && continue
    if [[ "$part" != *=* ]]; then
      echo "[error] invalid --pins entry '$part' (expected key=value)"
      diag_emit "installer.args" "FAIL" "INS-ARGS-0001" "invalid --pins entry '$part'" "use --pins key=value[,key=value]"
      exit 1
    fi
    key="${part%%=*}"
    val="${part#*=}"
    key="$(echo "$key" | tr '[:upper:]' '[:lower:]' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
    val="$(echo "$val" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
    # Strip optional wrapping quotes.
    val="${val%\"}"
    val="${val#\"}"
    val="${val%\'}"
    val="${val#\'}"

    case "$key" in
      sm) SM="$val" ;;
      libtorch_url) LIBTORCH_URL="$val" ;;
      libtorch_tag) LIBTORCH_CUDA_TAG="$val" ;;
      cudarc_feature) CUDARC_CUDA_FEATURE="$val" ;;
      *)
        echo "[error] unknown --pins key '$key'"
        echo "[hint] supported keys: sm, libtorch_url, libtorch_tag, cudarc_feature"
        diag_emit "installer.args" "FAIL" "INS-ARGS-0002" "unknown --pins key '$key'" "use one of: sm, libtorch_url, libtorch_tag, cudarc_feature"
        exit 1
        ;;
    esac
  done
}
