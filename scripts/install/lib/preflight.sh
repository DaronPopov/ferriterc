# preflight.sh — platform detection and preflight checks
# Sourced by scripts/install/install.sh

detect_platform_arch() {
  ARCH="$(uname -m)"
  case "$ARCH" in
    x86_64|aarch64) ;;
    *)
      echo "[error] unsupported architecture: $ARCH (need x86_64 or aarch64)"
      diag_emit "installer.preflight" "FAIL" "INS-PREF-0007" "unsupported architecture: $ARCH" "use x86_64 or aarch64 host"
      exit 1
      ;;
  esac
}

ensure_submodules() {
  if [[ -f "$ROOT/.gitmodules" ]]; then
    # Check if any submodule is missing its checkout
    if git -C "$ROOT" submodule status 2>/dev/null | grep -q '^-'; then
      echo "[info] initializing git submodules..."
      git -C "$ROOT" submodule update --init --recursive
    else
      echo "[info] submodules up to date"
    fi
  fi
}

run_preflight_checks() {
  echo "[info] architecture: ${ARCH}"
  ensure_submodules
  ensure_host_build_tools
  ensure_fetch_tools
  ensure_cuda_toolkit
  ensure_rust_toolchain
}
