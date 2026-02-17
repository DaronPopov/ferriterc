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

  # Detect Jetson (Tegra) platforms.
  # torch-sys pre-generated C++ bindings are built against x86_64 libtorch
  # and reference functions (cuSPARSELt, flash-attention dispatch) absent in
  # the Jetson aarch64 libtorch wheel.  orin-infer uses candle (not tch), so
  # core-only mode provides the full LLM inference stack without torch-sys.
  IS_JETSON=false
  if [[ "$ARCH" == "aarch64" ]] && uname -r 2>/dev/null | grep -qi tegra; then
    IS_JETSON=true
    if [[ "${CORE_ONLY}" != "true" ]]; then
      echo "[info] Jetson (Tegra) platform detected — enabling core-only mode"
      echo "[info] torch-sys x86_64 bindings are incompatible with Jetson libtorch"
      echo "[info] orin-infer LLM engine uses candle and does not require libtorch"
      CORE_ONLY=true
    fi
  fi
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
