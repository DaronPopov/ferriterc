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

run_preflight_checks() {
  echo "[info] architecture: ${ARCH}"
  ensure_host_build_tools
  ensure_fetch_tools
  ensure_cuda_toolkit
  ensure_rust_toolchain
}
