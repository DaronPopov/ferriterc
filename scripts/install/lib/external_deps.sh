# external_deps.sh — verify and prefetch external integration crates
# Sourced by scripts/install/install.sh

CUTLASS_VERSION="v3.7.0"
CUTLASS_DIR="${ROOT}/external/cutlass-ptx/cutlass"
CUTLASS_SENTINEL="${CUTLASS_DIR}/include/cutlass/gemm/device/gemm.h"

ensure_cutlass() {
  if [[ -f "${CUTLASS_SENTINEL}" ]]; then
    echo "[info] CUTLASS headers found at ${CUTLASS_DIR}"
    return 0
  fi

  echo "[info] CUTLASS headers not found — cloning NVIDIA/cutlass ${CUTLASS_VERSION}"
  if ! git clone --depth 1 --branch "${CUTLASS_VERSION}" \
       https://github.com/NVIDIA/cutlass.git "${CUTLASS_DIR}"; then
    echo "[error] failed to clone CUTLASS ${CUTLASS_VERSION}"
    diag_emit "installer.external" "FAIL" "INS-EXT-0010" \
      "failed to clone CUTLASS ${CUTLASS_VERSION} into ${CUTLASS_DIR}" \
      "check network connectivity and retry"
    exit 1
  fi

  if [[ ! -f "${CUTLASS_SENTINEL}" ]]; then
    echo "[error] CUTLASS clone succeeded but sentinel header not found"
    diag_emit "installer.external" "FAIL" "INS-EXT-0011" \
      "CUTLASS sentinel header missing after clone" \
      "verify CUTLASS version tag and repository structure"
    exit 1
  fi

  echo "[info] CUTLASS ${CUTLASS_VERSION} ready"
  diag_emit "installer.external" "PASS" "INS-EXT-0012" \
    "CUTLASS ${CUTLASS_VERSION} cloned to ${CUTLASS_DIR}" "none"
}

external_manifests=(
  "external/aten-ptx/Cargo.toml"
  "external/candle-ptx/Cargo.toml"
  "external/onnxruntime-ptx/Cargo.toml"
  "external/ferrite-torch/Cargo.toml"
  "external/ferrite-xla/Cargo.toml"
)

external_target_lib_paths() {
  local dirs=(
    "external/aten-ptx/target/release"
    "external/candle-ptx/target/release"
    "external/onnxruntime-ptx/target/release"
    "external/ferrite-xla/target/release"
  )
  local out=""
  local d
  for d in "${dirs[@]}"; do
    out+="${ROOT}/${d}:"
  done
  echo "$out"
}

ensure_external_integrations() {
  local m
  for m in "${external_manifests[@]}"; do
    if [[ ! -f "${ROOT}/${m}" ]]; then
      echo "[error] missing external integration manifest: ${ROOT}/${m}"
      diag_emit "installer.external" "FAIL" "INS-EXT-0001" "missing external integration manifest: ${m}" "restore repository external integrations and re-run installer"
      exit 1
    fi
  done

  echo "[info] prefetching cargo dependencies for external integrations"
  for m in "${external_manifests[@]}"; do
    if ! cargo fetch --manifest-path "${ROOT}/${m}"; then
      echo "[error] failed to prefetch dependencies for ${m}"
      diag_emit "installer.external" "FAIL" "INS-EXT-0002" "failed to prefetch external integration dependencies: ${m}" "check network/cargo configuration and retry"
      exit 1
    fi
  done

  diag_emit "installer.external" "PASS" "INS-EXT-0003" "external integration manifests and dependencies are ready" "none"
}
