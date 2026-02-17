# args.sh — installer defaults and CLI parsing
# Sourced by scripts/install/install.sh

init_install_defaults() {
  # keep existing semantics/default precedence
  SM="${SM:-${CUDA_SM:-${GPU_SM:-}}}"
  VERBOSE=false
  ENABLE_SERVICE=false
  LIBTORCH_VERSION="${LIBTORCH_VERSION:-2.9.0}"
  LIBTORCH_CUDA_TAG="${LIBTORCH_CUDA_TAG:-}"
  CUDARC_CUDA_FEATURE="${CUDARC_CUDA_FEATURE:-}"
  LIBTORCH_URL="${LIBTORCH_URL:-}"
  ONNXRUNTIME_VERSION="${ONNXRUNTIME_VERSION:-1.20.1}"
  ONNXRUNTIME_URL="${ONNXRUNTIME_URL:-}"
  TORCH_CPYTHON_TAG="${TORCH_CPYTHON_TAG:-cp311}"
  CORE_ONLY="${CORE_ONLY:-false}"
  WITH_CAPTURE=false
  AUTO_INSTALL_CUDA="${AUTO_INSTALL_CUDA:-false}"
}

parse_install_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --sm)             SM="${2:-}";                  shift 2 ;;
      --verbose)        VERBOSE=true;                 shift ;;
      --enable-service) ENABLE_SERVICE=true;          shift ;;
      --libtorch-url)   LIBTORCH_URL="${2:-}";        shift 2 ;;
      --libtorch-tag)   LIBTORCH_CUDA_TAG="${2:-}";   shift 2 ;;
      --cudarc-feature) CUDARC_CUDA_FEATURE="${2:-}"; shift 2 ;;
      --onnxruntime-version) ONNXRUNTIME_VERSION="${2:-}"; shift 2 ;;
      --onnxruntime-url) ONNXRUNTIME_URL="${2:-}";    shift 2 ;;
      --core-only)      CORE_ONLY=true;                 shift ;;
      --with-capture)   WITH_CAPTURE=true;             shift ;;
      --auto-install-cuda) AUTO_INSTALL_CUDA=true;     shift ;;
      --pins)           apply_pins "${2:-}";          shift 2 ;;
      -h|--help)        usage; exit 0 ;;
      *)
        echo "[error] unknown option: $1"
        diag_emit "installer.args" "FAIL" "INS-ARGS-0003" "unknown option: $1" "use --help to view supported options"
        usage
        exit 1
        ;;
    esac
  done
}
