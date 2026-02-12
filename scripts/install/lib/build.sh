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
    if ! cargo check --release --no-default-features --features "torch,${CUDARC_CUDA_FEATURE}" --example "$NAME" 2>/dev/null; then
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
    local STEPS=3
    echo "[info] core-only mode — skipping libtorch and torch-dependent crates"

    echo "[1/${STEPS}] Building ferrite-os"
    cd "$ROOT/ferrite-os"
    make all GPU_SM="${SM}" PTX_GPU_SM="sm_${SM}"

    echo "[2/${STEPS}] Building ferrite daemon"
    cd "$ROOT/ferrite-os"
    CUDA_PATH="${CUDA_PATH}" cargo build --release -p ferrite-daemon

    echo "[3/${STEPS}] Installing ferrite command"
  else
    echo "[info] using libtorch: ${LIBTORCH}"
    local STEPS=9

    echo "[1/${STEPS}] Building ferrite-os"
    cd "$ROOT/ferrite-os"
    make all GPU_SM="${SM}" PTX_GPU_SM="sm_${SM}"

    echo "[2/${STEPS}] Building ferrite-gpu-lang (torch feature)"
    cd "$ROOT/ferrite-gpu-lang"
    cargo build --release --no-default-features --features "torch,${CUDARC_CUDA_FEATURE}"

    echo "[3/${STEPS}] Building external/ferrite-torch examples"
    cd "$ROOT/external/ferrite-torch"
    cargo build --release --examples --no-default-features --features "${CUDARC_CUDA_FEATURE}"

    echo "[4/${STEPS}] Building external/ferrite-xla example"
    cd "$ROOT/external/ferrite-xla"
    cargo build --release --example xla_allocator_test

    echo "[5/${STEPS}] Validating ferrite-gpu-lang torch+xla scripts"
    cd "$ROOT/ferrite-gpu-lang"
    cargo run --release --no-default-features --features "torch,${CUDARC_CUDA_FEATURE}" --example script_cv_detect >/dev/null

    echo "[6/${STEPS}] Checking finetune_engine scripts"
    cd "$ROOT/ferrite-gpu-lang"
    check_engine_scripts "$ROOT/finetune_engine" "finetune_engine"

    echo "[7/${STEPS}] Checking mathematics_engine scripts"
    check_engine_scripts "$ROOT/mathematics_engine" "mathematics_engine"

    echo "[8/${STEPS}] Building ferrite daemon"
    cd "$ROOT/ferrite-os"
    CUDA_PATH="${CUDA_PATH}" cargo build --release -p ferrite-daemon

    echo "[9/${STEPS}] Installing ferrite command"
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
  # Also symlink ferrite-daemon for direct access
  local DAEMON_BIN="$ROOT/ferrite-os/target/release/ferrite-daemon"
  if [[ -f "$DAEMON_BIN" ]]; then
    ln -sf "$DAEMON_BIN" "${INSTALL_BIN}/ferrite-daemon"
  fi
}

print_success() {
  local INSTALL_BIN="${HOME}/.local/bin"
  echo "[ok] ferrite runtime build complete (sm_${SM}, ${ARCH})"
  diag_emit "installer.build" "PASS" "INS-BLD-0004" "ferrite runtime build complete (sm_${SM}, ${ARCH})" "none"
  if [[ "${CORE_ONLY}" != "true" ]]; then
    echo "     libtorch:           ${LIBTORCH_VERSION}+${LIBTORCH_CUDA_TAG}"
    echo "     cudarc feature:     ${CUDARC_CUDA_FEATURE}"
    echo "     finetune_engine:    ready"
    echo "     mathematics_engine: ready"
  else
    echo "     mode:               core-only"
  fi
  echo ""
  if [[ "${CORE_ONLY}" != "true" ]]; then
    echo "  Run scripts with: ./ferrite-run <script.rs> --torch"
  fi
  echo "  Launch the shell:  ferrite"
  echo ""
  if [[ ":${PATH}:" != *":${INSTALL_BIN}:"* ]]; then
    echo "  [note] Add ~/.local/bin to your PATH if not already present:"
    echo "         export PATH=\"\$HOME/.local/bin:\$PATH\""
  fi
}
