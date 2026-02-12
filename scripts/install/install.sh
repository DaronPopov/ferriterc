#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "$SCRIPT_DIR/lib/env.sh"
source "$SCRIPT_DIR/lib/diag.sh"
source "$SCRIPT_DIR/lib/preflight.sh"
source "$SCRIPT_DIR/lib/args.sh"
source "$SCRIPT_DIR/lib/policy.sh"
source "$SCRIPT_DIR/lib/cuda.sh"
source "$SCRIPT_DIR/lib/rust.sh"
source "$SCRIPT_DIR/lib/libtorch.sh"
source "$SCRIPT_DIR/lib/onnxruntime.sh"
source "$SCRIPT_DIR/lib/build.sh"
source "$SCRIPT_DIR/lib/service.sh"

main() {
  detect_platform_arch

  # defaults + CLI parsing (surface unchanged)
  init_install_defaults
  parse_install_args "$@"

  # preflight boundary
  run_preflight_checks

  # deterministic compat/pin policy boundary
  if [[ "${CORE_ONLY}" != "true" ]]; then
    resolve_install_policy
  fi

  # CUDA environment + SM resolution
  setup_cuda_env
  auto_detect_sm
  export_build_env

  # provisioning boundary
  if [[ "${CORE_ONLY}" != "true" ]]; then
    ensure_libtorch
    ensure_onnxruntime
    local ort_lib_path=""
    if [[ -n "${ONNXRUNTIME_ROOT:-}" && -d "${ONNXRUNTIME_ROOT}/lib" ]]; then
      ort_lib_path="${ONNXRUNTIME_ROOT}/lib:"
    fi
    # Include ferrite-os runtime, libtorch, and external integration libs
    export LD_LIBRARY_PATH="${ROOT}/ferrite-os/lib:${LIBTORCH}/lib:${ort_lib_path}${ROOT}/external/aten-ptx/target/release:${ROOT}/external/candle-ptx/target/release:${ROOT}/external/onnxruntime-ptx/target/release:${LD_LIBRARY_PATH:-}"
  else
    export LD_LIBRARY_PATH="${ROOT}/ferrite-os/lib:${LD_LIBRARY_PATH:-}"
  fi

  # Optional: provision OpenCV for capture feature
  if [[ "${WITH_CAPTURE}" == "true" ]]; then
    ensure_opencv
  fi

  # build/install boundary
  run_build

  # optional service boundary
  [[ "$ENABLE_SERVICE" == "true" ]] && install_systemd_service

  print_success
}

main "$@"
