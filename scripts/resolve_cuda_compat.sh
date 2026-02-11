#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPAT_FILE="${ROOT}/compat.toml"
FORMAT="env"
QUIET=0
DIAG_FORMAT="${FERRITE_DIAG_FORMAT:-plain}"
DIAG_LIB="${ROOT}/scripts/install/lib/diag.sh"

if [[ -f "$DIAG_LIB" ]]; then
  # shellcheck disable=SC1090
  source "$DIAG_LIB"
fi

emit_diag() {
  local status="$1"
  local code="$2"
  local summary="$3"
  local remediation="$4"
  if declare -F diag_emit >/dev/null 2>&1; then
    FERRITE_DIAG_FORMAT="$DIAG_FORMAT" diag_emit "compat.resolver" "$status" "$code" "$summary" "$remediation" "stderr"
  fi
}

log() {
  if [[ "$QUIET" -eq 0 ]]; then
    echo "$@" >&2
  fi
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --format)
        FORMAT="${2:-env}"
        shift 2
        ;;
      --format=*)
        FORMAT="${1#*=}"
        shift
        ;;
      --quiet)
        QUIET=1
        shift
        ;;
      --diag-format)
        DIAG_FORMAT="${2:-plain}"
        shift 2
        ;;
      --diag-format=*)
        DIAG_FORMAT="${1#*=}"
        shift
        ;;
      *)
        echo "[compat] unknown argument: $1" >&2
        emit_diag "FAIL" "COMPAT-ARGS-0001" "unknown argument: $1" "use --format <env|json> [--quiet] [--diag-format <plain|json>]"
        exit 2
        ;;
    esac
  done
}

preflight() {
  if [[ ! -f "$COMPAT_FILE" ]]; then
    echo "[compat] missing compatibility manifest: $COMPAT_FILE" >&2
    emit_diag "FAIL" "COMPAT-PREF-0001" "missing compatibility manifest: $COMPAT_FILE" "restore compat.toml at repository root"
    exit 1
  fi

  if ! command -v nvcc >/dev/null 2>&1; then
    echo "[compat] nvcc not found in PATH" >&2
    emit_diag "FAIL" "COMPAT-PREF-0002" "nvcc not found in PATH" "install CUDA toolkit and ensure nvcc is on PATH"
    exit 1
  fi
}

read_cuda_version() {
  NVCC_REL="$(nvcc --version | sed -n 's/.*release \([0-9]\+\)\.\([0-9]\+\).*/\1.\2/p' | head -n1)"
  if [[ -z "${NVCC_REL:-}" ]]; then
    echo "[compat] unable to parse CUDA version from nvcc --version" >&2
    emit_diag "FAIL" "COMPAT-DET-0001" "unable to parse CUDA version from nvcc --version" "verify nvcc output and CUDA installation"
    exit 1
  fi
  CUDA_MAJOR="${NVCC_REL%%.*}"
  CUDA_MINOR="${NVCC_REL#*.}"
}

toml_get() {
  local section="$1"
  local key="$2"
  awk -v sec="$section" -v key="$key" '
    {
      line=$0
      sub(/[[:space:]]*#.*/, "", line)
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", line)
      if (line == "[" sec "]") {
        in_sec=1
        next
      }
      if (in_sec && line ~ /^\[/) {
        in_sec=0
      }
      if (in_sec && line ~ ("^" key "[[:space:]]*=")) {
        sub("^[^=]*=[[:space:]]*", "", line)
        gsub(/^"/, "", line)
        gsub(/"$/, "", line)
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", line)
        print line
        exit
      }
    }
  ' "$COMPAT_FILE"
}

resolve_compat() {
  local section
  section="cuda.\"${CUDA_MAJOR}.${CUDA_MINOR}\""
  CUDARC_FEATURE="$(toml_get "$section" "cudarc_feature" || true)"
  LIBTORCH_TAG="$(toml_get "$section" "libtorch_cuda_tag" || true)"

  if [[ -z "${CUDARC_FEATURE:-}" || -z "${LIBTORCH_TAG:-}" ]]; then
    section="cuda.\"${CUDA_MAJOR}\""
    CUDARC_FEATURE="$(toml_get "$section" "cudarc_feature" || true)"
    LIBTORCH_TAG="$(toml_get "$section" "libtorch_cuda_tag" || true)"
  fi

  if [[ -z "${CUDARC_FEATURE:-}" || -z "${LIBTORCH_TAG:-}" ]]; then
    section="defaults"
    CUDARC_FEATURE="$(toml_get "$section" "cudarc_feature" || true)"
    LIBTORCH_TAG="$(toml_get "$section" "libtorch_cuda_tag" || true)"
  fi

  if [[ -z "${CUDARC_FEATURE:-}" || -z "${LIBTORCH_TAG:-}" ]]; then
    echo "[compat] compatibility resolution failed for CUDA ${NVCC_REL}" >&2
    emit_diag "FAIL" "COMPAT-RES-0001" "compatibility resolution failed for CUDA ${NVCC_REL}" "add matching entry to compat.toml or update defaults"
    exit 1
  fi

  log "[compat] cuda=${NVCC_REL} cudarc=${CUDARC_FEATURE} libtorch=${LIBTORCH_TAG}"
  emit_diag "PASS" "COMPAT-RES-0002" "resolved CUDA ${NVCC_REL} to cudarc=${CUDARC_FEATURE}, libtorch=${LIBTORCH_TAG}" "none"
}

emit_result() {
  case "$FORMAT" in
    env)
      echo "CUDA_VERSION=${NVCC_REL}"
      echo "CUDA_MAJOR=${CUDA_MAJOR}"
      echo "CUDA_MINOR=${CUDA_MINOR}"
      echo "CUDARC_CUDA_FEATURE=${CUDARC_FEATURE}"
      echo "LIBTORCH_CUDA_TAG_RESOLVED=${LIBTORCH_TAG}"
      ;;
    json)
      printf '{"cuda_version":"%s","cuda_major":"%s","cuda_minor":"%s","cudarc_feature":"%s","libtorch_cuda_tag":"%s"}\n' \
        "$NVCC_REL" "$CUDA_MAJOR" "$CUDA_MINOR" "$CUDARC_FEATURE" "$LIBTORCH_TAG"
      ;;
    *)
      echo "[compat] unknown format: $FORMAT" >&2
      emit_diag "FAIL" "COMPAT-ARGS-0002" "unknown format: $FORMAT" "use --format env or --format json"
      exit 2
      ;;
  esac
}

main() {
  parse_args "$@"
  preflight
  read_cuda_version
  resolve_compat
  emit_result
}

main "$@"
