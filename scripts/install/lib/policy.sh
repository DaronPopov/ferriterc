# policy.sh — install policy and compatibility resolution
# Sourced by scripts/install/install.sh

resolve_install_policy() {
  # Preserve resolver semantics from resolve_cuda_compat.sh + explicit overrides.
  resolve_cuda_compat

  echo "[info] selected LIBTORCH_CUDA_TAG: ${LIBTORCH_CUDA_TAG}"
  echo "[info] selected CUDARC_CUDA_FEATURE: ${CUDARC_CUDA_FEATURE}"
  [[ -n "${LIBTORCH_URL}" ]] && echo "[info] pinned LIBTORCH_URL: ${LIBTORCH_URL}"
  [[ -n "${ONNXRUNTIME_URL:-}" ]] && echo "[info] pinned ONNXRUNTIME_URL: ${ONNXRUNTIME_URL}"
  [[ -n "${ONNXRUNTIME_VERSION:-}" ]] && echo "[info] selected ONNXRUNTIME_VERSION: ${ONNXRUNTIME_VERSION}"
  diag_emit "installer.compat" "PASS" "INS-COMPAT-0004" "compat policy resolved" "none"
}
