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
               [--auto-install-cuda]
               [--pins "<k=v,k=v,...>"]

Options:
  --sm <SM>     GPU compute capability (e.g. 75, 80, 86, 89, 90, 100, 120)
  --verbose     Enable verbose build output
  --enable-service
                Install and enable ferrite-daemon systemd service for boot startup
  --core-only     Build only the core OS (daemon, TUI, ptx-runtime).
                  Skips libtorch download and torch-dependent crates.
  --with-capture  Enable the capture feature (OpenCV camera + vision ops).
                  Auto-installs OpenCV + clang/C++ header deps if needed.
  --auto-install-cuda
                Opt-in: auto-install CUDA toolkit if `nvcc` is missing.
                Default is off (CUDA is user-managed prerequisite).
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
  AUTO_INSTALL_CUDA          true/false (default: false)
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

has_opencv() {
  # Check if OpenCV dev headers/pkgconfig are available.
  if pkg-config --exists opencv4 2>/dev/null; then
    return 0
  fi
  if pkg-config --exists opencv 2>/dev/null; then
    return 0
  fi
  # Fallback: check for the header directly.
  if [[ -f /usr/include/opencv4/opencv2/core.hpp ]] || \
     [[ -f /usr/include/opencv2/core.hpp ]] || \
     [[ -f /usr/local/include/opencv4/opencv2/core.hpp ]]; then
    return 0
  fi
  return 1
}

has_clang_cpp_stdlib() {
  # opencv crate bindings are generated with clang parsing C++ headers.
  # Probe for a usable clang + C++ stdlib header stack by compiling <memory>.
  if ! cmd_exists clang; then
    return 1
  fi

  local probe_src
  probe_src="$(mktemp /tmp/ferrite-cxx-probe-XXXXXX.cpp)"
  cat > "$probe_src" <<'EOF'
#include <memory>
int main() {
  auto p = std::make_unique<int>(1);
  return *p - 1;
}
EOF

  if clang -x c++ -std=c++14 -fsyntax-only "$probe_src" >/dev/null 2>&1; then
    rm -f "$probe_src"
    return 0
  fi

  rm -f "$probe_src"
  return 1
}

has_clang_cpp_stdlib_with_args() {
  local extra_args="$1"
  if ! cmd_exists clang; then
    return 1
  fi

  local probe_src
  probe_src="$(mktemp /tmp/ferrite-cxx-probe-XXXXXX.cpp)"
  cat > "$probe_src" <<'EOF'
#include <memory>
int main() {
  auto p = std::make_unique<int>(1);
  return *p - 1;
}
EOF

  if clang -x c++ -std=c++14 -fsyntax-only $extra_args "$probe_src" >/dev/null 2>&1; then
    rm -f "$probe_src"
    return 0
  fi

  rm -f "$probe_src"
  return 1
}

clang_selected_gcc_major() {
  if ! cmd_exists clang; then
    return 1
  fi

  local selected
  selected="$(clang -x c++ -E -v - </dev/null 2>&1 | sed -n 's#.*Selected GCC installation: .*/\([0-9][0-9]*\)$#\1#p' | head -n 1)"
  if [[ -n "$selected" ]]; then
    echo "$selected"
    return 0
  fi
  return 1
}

build_clang_cpp_fallback_args() {
  local cxx_ver
  cxx_ver="$(ls -1 /usr/include/c++ 2>/dev/null | rg '^[0-9]+$' | sort -n | tail -n 1 || true)"
  if [[ -z "$cxx_ver" ]]; then
    return 1
  fi

  local triplet
  triplet="$(gcc -dumpmachine 2>/dev/null || true)"
  local args="-isystem/usr/include/c++/${cxx_ver}"
  if [[ -n "$triplet" && -d "/usr/include/${triplet}/c++/${cxx_ver}" ]]; then
    args="${args} -isystem/usr/include/${triplet}/c++/${cxx_ver}"
  fi
  if [[ -d "/usr/include/c++/${cxx_ver}/backward" ]]; then
    args="${args} -isystem/usr/include/c++/${cxx_ver}/backward"
  fi

  echo "$args"
  return 0
}

configure_opencv_clang_args() {
  # If clang can already parse standard C++ headers, no extra args needed.
  if has_clang_cpp_stdlib; then
    return 0
  fi

  # Ubuntu can have gcc-12 installed without matching libstdc++-12 headers.
  # If clang selects that GCC, try installing matching C++ dev headers.
  if cmd_exists apt-get; then
    local gcc_major
    gcc_major="$(clang_selected_gcc_major || true)"
    if [[ -n "$gcc_major" && ! -f "/usr/include/c++/${gcc_major}/memory" ]]; then
      echo "[warn] clang selected GCC ${gcc_major}, but /usr/include/c++/${gcc_major} is missing; attempting repair"
      if ! sudo_cmd apt-get install -y "g++-${gcc_major}" "libstdc++-${gcc_major}-dev"; then
        echo "[warn] unable to install g++-${gcc_major}/libstdc++-${gcc_major}-dev automatically"
      fi
      if has_clang_cpp_stdlib; then
        return 0
      fi
    fi
  fi

  local fallback_args
  fallback_args="$(build_clang_cpp_fallback_args || true)"
  if [[ -z "$fallback_args" ]]; then
    return 1
  fi

  if has_clang_cpp_stdlib_with_args "$fallback_args"; then
    if [[ -n "${OPENCV_CLANG_ARGS:-}" ]]; then
      if [[ "${OPENCV_CLANG_ARGS}" != *"${fallback_args}"* ]]; then
        export OPENCV_CLANG_ARGS="${OPENCV_CLANG_ARGS} ${fallback_args}"
      fi
    else
      export OPENCV_CLANG_ARGS="${fallback_args}"
    fi
    echo "[warn] exporting OPENCV_CLANG_ARGS to provide C++ stdlib include paths"
    return 0
  fi

  return 1
}

has_capture_build_stack() {
  if ! has_opencv; then
    return 1
  fi
  if has_clang_cpp_stdlib; then
    return 0
  fi
  if [[ -n "${OPENCV_CLANG_ARGS:-}" ]] && has_clang_cpp_stdlib_with_args "${OPENCV_CLANG_ARGS}"; then
    return 0
  fi
  return 1
}

ensure_opencv() {
  if has_capture_build_stack; then
    echo "[info] OpenCV capture build stack detected"
    return 0
  fi

  if has_opencv; then
    echo "[warn] OpenCV detected but clang/C++ header stack is incomplete; attempting repair"
  else
    echo "[info] OpenCV dev libraries not found; attempting auto-install"
  fi

  if cmd_exists apt-get; then
    pm_install_pkgs libopencv-dev clang libclang-dev g++
  elif cmd_exists dnf; then
    pm_install_pkgs opencv-devel clang clang-devel gcc-c++ libstdc++-devel
  elif cmd_exists yum; then
    pm_install_pkgs opencv-devel clang clang-devel gcc-c++ libstdc++-devel
  elif cmd_exists pacman; then
    pm_install_pkgs opencv clang gcc
  elif cmd_exists zypper; then
    pm_install_pkgs opencv-devel clang libclang-devel gcc-c++
  else
    echo "[error] cannot auto-install capture dependencies on this distro"
    echo "[hint] install OpenCV + clang + C++ dev headers manually:"
    echo "       apt: sudo apt install libopencv-dev clang libclang-dev g++"
    echo "       dnf: sudo dnf install opencv-devel clang clang-devel gcc-c++ libstdc++-devel"
    diag_emit "installer.preflight" "FAIL" "INS-PREF-0010" "cannot auto-install capture dependencies" "install OpenCV, clang/libclang, and C++ headers manually"
    exit 1
  fi

  if ! has_opencv; then
    echo "[error] OpenCV still not found after auto-install attempt"
    echo "[hint] install OpenCV development libraries manually (libopencv-dev / opencv-devel)"
    diag_emit "installer.preflight" "FAIL" "INS-PREF-0011" "OpenCV still missing after auto-install" "install manually"
    exit 1
  fi

  if ! configure_opencv_clang_args; then
    echo "[error] clang/C++ headers still unavailable after auto-install attempt"
    echo "[hint] ensure clang and matching C++ stdlib headers are installed (g++, libstdc++-dev)"
    diag_emit "installer.preflight" "FAIL" "INS-PREF-0012" "clang/C++ headers missing after auto-install" "install clang and C++ stdlib headers manually"
    exit 1
  fi

  echo "[info] OpenCV capture build stack installed successfully"
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
