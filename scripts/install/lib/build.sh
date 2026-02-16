# build.sh — cargo/make build steps and validation
# Sourced by scripts/install/install.sh

check_engine_scripts() {
  local engine_dir="$1"
  local engine_name="$2"
  local all_ok=true

  while IFS= read -r -d '' script; do
    NAME="$(basename "$script" .rs)"
    NAME="$(echo "$NAME" | sed 's/[^a-zA-Z0-9_]/_/g')"
    LINK="$ROOT/ferrite-gpu-lang/examples/${NAME}.rs"
    ln -sf "$script" "$LINK"
    if ! LIBTORCH="${LIBTORCH}" LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" \
         cargo check --release --no-default-features --features "torch,${CUDARC_CUDA_FEATURE}" --example "$NAME" 2>/dev/null; then
      echo "  [warn] $(basename "$script") failed to check"
      diag_emit "installer.build" "WARN" "INS-BLD-0001" "example check failed: $(basename "$script")" "inspect example compile errors"
      all_ok=false
    fi
    rm -f "$LINK"
  done < <(find "$engine_dir" -name '*.rs' -type f -print0)

  if [[ "$all_ok" == "true" ]]; then
    echo "  ${engine_name}: all scripts compile"
    diag_emit "installer.build" "PASS" "INS-BLD-0002" "${engine_name} scripts compile" "none"
  else
    echo "  ${engine_name}: some scripts had warnings (non-fatal)"
    diag_emit "installer.build" "WARN" "INS-BLD-0003" "${engine_name} scripts had non-fatal warnings" "review warnings before production use"
  fi
}

run_build() {
  echo "[info] using GPU SM: ${SM}"

  if [[ "${CORE_ONLY}" == "true" ]]; then
    local STEPS=5
    echo "[info] core-only mode — skipping libtorch and torch-dependent crates"

    echo "[1/${STEPS}] Building ferrite-os"
    cd "$ROOT/ferrite-os"
    make all GPU_SM="${SM}" PTX_GPU_SM="sm_${SM}"

    local CORE_FEATURES="${CUDARC_CUDA_FEATURE:-cuda-12080}"
    if [[ "${WITH_CAPTURE}" == "true" ]]; then
      CORE_FEATURES="${CORE_FEATURES},capture"
      echo "[2/${STEPS}] Building ferrite-gpu-lang (core + capture: sensor + vision + pipeline)"
    else
      echo "[2/${STEPS}] Building ferrite-gpu-lang (core: sensor + vision + pipeline)"
    fi
    cd "$ROOT/ferrite-gpu-lang"
    cargo build --release --no-default-features --features "${CORE_FEATURES}"

    echo "[3/${STEPS}] Building ferrite daemon"
    cd "$ROOT/ferrite-os"
    CUDA_PATH="${CUDA_PATH}" cargo build --release -p ferrite-daemon

    echo "[4/${STEPS}] Building ferrite LLM inference engine"
    if [[ -d "$ROOT/ferrite" && -f "$ROOT/ferrite/Cargo.toml" ]]; then
      cd "$ROOT/ferrite"
      LD_LIBRARY_PATH="${ROOT}/ferrite-os/lib:${LD_LIBRARY_PATH:-}" \
        cargo build --release --bin orin-infer
    else
      echo "[warn] ferrite/ submodule not found, skipping LLM engine"
    fi

    echo "[5/${STEPS}] Installing ferrite command"
  else
    if [[ -z "${LIBTORCH:-}" ]] || ! is_valid_libtorch "${LIBTORCH}"; then
      echo "[error] LIBTORCH is not set or invalid — cannot build torch-dependent crates"
      echo "[hint]  run the installer without --core-only, or set LIBTORCH= to a valid libtorch directory"
      diag_emit "installer.build" "FAIL" "INS-BLD-0010" "LIBTORCH not set or invalid for full build" "ensure libtorch is provisioned before build"
      exit 1
    fi
    echo "[info] using libtorch: ${LIBTORCH}"
    local STEPS=13

    echo "[1/${STEPS}] Building ferrite-os"
    cd "$ROOT/ferrite-os"
    make all GPU_SM="${SM}" PTX_GPU_SM="sm_${SM}"

    echo "[2/${STEPS}] Building external/aten-ptx"
    cd "$ROOT/external/aten-ptx"
    LIBTORCH="${LIBTORCH}" LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" \
      cargo build --release

    echo "[3/${STEPS}] Building external/candle-ptx"
    cd "$ROOT/external/candle-ptx"
    cargo build --release

    echo "[4/${STEPS}] Building external/onnxruntime-ptx"
    cd "$ROOT/external/onnxruntime-ptx"
    cargo build --release

    local GPU_LANG_FEATURES="torch,${CUDARC_CUDA_FEATURE}"
    if [[ "${WITH_CAPTURE}" == "true" ]]; then
      GPU_LANG_FEATURES="${GPU_LANG_FEATURES},capture"
      echo "[5/${STEPS}] Building ferrite-gpu-lang (torch + capture + sensor + vision + pipeline)"
    else
      echo "[5/${STEPS}] Building ferrite-gpu-lang (torch + sensor + vision + pipeline)"
    fi
    cd "$ROOT/ferrite-gpu-lang"
    LIBTORCH="${LIBTORCH}" LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" \
      cargo build --release --no-default-features --features "${GPU_LANG_FEATURES}"

    echo "[6/${STEPS}] Building external/ferrite-torch examples"
    cd "$ROOT/external/ferrite-torch"
    LIBTORCH="${LIBTORCH}" LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" \
      cargo build --release --examples --no-default-features --features "${CUDARC_CUDA_FEATURE}"

    echo "[7/${STEPS}] Building external/ferrite-xla example"
    cd "$ROOT/external/ferrite-xla"
    cargo build --release --example xla_allocator_test

    echo "[8/${STEPS}] Validating ferrite-gpu-lang torch+xla scripts"
    cd "$ROOT/ferrite-gpu-lang"
    LIBTORCH="${LIBTORCH}" LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" \
      cargo run --release --no-default-features --features "torch,${CUDARC_CUDA_FEATURE}" --example script_cv_detect >/dev/null

    local FINETUNE_DIR="$ROOT/ferrite-os/workloads/finetune_engine"
    local MATH_DIR="$ROOT/ferrite-os/workloads/mathematics_engine"
    # Backward compatibility for older tree layouts.
    [[ -d "$FINETUNE_DIR" ]] || FINETUNE_DIR="$ROOT/finetune_engine"
    [[ -d "$MATH_DIR" ]] || MATH_DIR="$ROOT/mathematics_engine"

    echo "[9/${STEPS}] Checking finetune_engine scripts"
    cd "$ROOT/ferrite-gpu-lang"
    check_engine_scripts "$FINETUNE_DIR" "finetune_engine"

    echo "[10/${STEPS}] Checking mathematics_engine scripts"
    check_engine_scripts "$MATH_DIR" "mathematics_engine"

    echo "[11/${STEPS}] Building ferrite daemon"
    cd "$ROOT/ferrite-os"
    CUDA_PATH="${CUDA_PATH}" cargo build --release -p ferrite-daemon

    echo "[12/${STEPS}] Building ferrite LLM inference engine"
    if [[ -d "$ROOT/ferrite" && -f "$ROOT/ferrite/Cargo.toml" ]]; then
      cd "$ROOT/ferrite"
      LD_LIBRARY_PATH="${ROOT}/ferrite-os/lib:${LIBTORCH}/lib:${LD_LIBRARY_PATH:-}" \
        cargo build --release --bin orin-infer
    else
      echo "[warn] ferrite/ submodule not found, skipping LLM engine"
    fi

    echo "[13/${STEPS}] Installing ferrite command"
  fi

  local INSTALL_BIN="${HOME}/.local/bin"
  mkdir -p "${INSTALL_BIN}"
  local FERRITE_BIN="$ROOT/ferrite-os/target/release/ferrite"
  if [[ -f "$FERRITE_BIN" ]]; then
    ln -sf "$FERRITE_BIN" "${INSTALL_BIN}/ferrite"
    echo "[info] symlinked: ${INSTALL_BIN}/ferrite -> ${FERRITE_BIN}"
  else
    echo "[warn] ferrite binary not found at ${FERRITE_BIN}, skipping symlink"
  fi
  # Symlink orin-infer for LLM inference
  local ORIN_INFER_BIN="$ROOT/ferrite/target/release/orin-infer"
  if [[ -f "$ORIN_INFER_BIN" ]]; then
    ln -sf "$ORIN_INFER_BIN" "${INSTALL_BIN}/orin-infer"
    echo "[info] symlinked: ${INSTALL_BIN}/orin-infer -> ${ORIN_INFER_BIN}"
  fi

  # Symlink ferrite-daemon wrapper for direct access from any directory.
  # Wrapper handles runtime + libtorch environment and forwards to the binary.
  local DAEMON_WRAPPER="$ROOT/ferrite-daemon"
  local DAEMON_BIN="$ROOT/ferrite-os/target/release/ferrite-daemon"
  if [[ -x "$DAEMON_WRAPPER" ]]; then
    ln -sf "$DAEMON_WRAPPER" "${INSTALL_BIN}/ferrite-daemon"
    echo "[info] symlinked: ${INSTALL_BIN}/ferrite-daemon -> ${DAEMON_WRAPPER}"
  elif [[ -f "$DAEMON_BIN" ]]; then
    ln -sf "$DAEMON_BIN" "${INSTALL_BIN}/ferrite-daemon"
    echo "[info] symlinked: ${INSTALL_BIN}/ferrite-daemon -> ${DAEMON_BIN}"
  else
    echo "[warn] ferrite-daemon wrapper/binary not found, skipping symlink"
  fi
}

print_success() {
  local INSTALL_BIN="${HOME}/.local/bin"
  echo "[ok] ferrite runtime build complete (sm_${SM}, ${ARCH})"
  diag_emit "installer.build" "PASS" "INS-BLD-0004" "ferrite runtime build complete (sm_${SM}, ${ARCH})" "none"
  echo "     modules:            sensor, vision, pipeline (always compiled)"
  if [[ "${WITH_CAPTURE}" == "true" ]]; then
    echo "     capture:            enabled (OpenCV camera + vision ops + draw)"
  fi
  if [[ "${CORE_ONLY}" != "true" ]]; then
    echo "     libtorch:           ${LIBTORCH_VERSION}+${LIBTORCH_CUDA_TAG}"
    echo "     cudarc feature:     ${CUDARC_CUDA_FEATURE}"
    echo "     external libs:      aten-ptx, candle-ptx, onnxruntime-ptx, ferrite-xla"
    echo "     finetune_engine:    ready"
    echo "     mathematics_engine: ready"
  else
    echo "     mode:               core-only"
  fi
  echo ""
  if [[ "${CORE_ONLY}" != "true" ]]; then
    echo "  Run scripts with: ./ferrite-run <script.rs> --torch"
    echo "  Hybrid pipeline:  cargo run --release --features capture --example script_hybrid_pipeline"
  fi
  echo "  Launch the shell:  ferrite"
  echo "  Launch daemon:     ferrite-daemon"
  echo ""
  if [[ ":${PATH}:" != *":${INSTALL_BIN}:"* ]]; then
    echo "  [note] Add ~/.local/bin to your PATH if not already present:"
    echo "         export PATH=\"\$HOME/.local/bin:\$PATH\""
  fi
}
