# external_deps.sh — verify and prefetch external integration crates
# Sourced by scripts/install/install.sh

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
