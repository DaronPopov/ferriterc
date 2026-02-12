# onnxruntime.sh — download, verify, and provision ONNX Runtime
# Sourced by scripts/install/install.sh

is_valid_onnxruntime() {
  local d="$1"
  [[ -d "$d" ]] || return 1
  if [[ ! -f "$d/lib/libonnxruntime.so" && ! -f "$d/lib/libonnxruntime.so.1" ]]; then
    return 1
  fi
  if [[ -f "$d/include/onnxruntime/core/session/onnxruntime_c_api.h" ]] || \
     [[ -f "$d/include/onnxruntime_c_api.h" ]]; then
    return 0
  fi
  return 1
}

check_onnxruntime_version() {
  local d="$1"
  local ver_file="$d/VERSION_NUMBER"
  if [[ -z "${ONNXRUNTIME_VERSION:-}" || ! -f "$ver_file" ]]; then
    return 0
  fi
  local installed_ver
  installed_ver="$(head -n1 "$ver_file" | tr -d '[:space:]')"
  if [[ "$installed_ver" != "$ONNXRUNTIME_VERSION" ]]; then
    echo "[warn] bundled onnxruntime is ${installed_ver} but need ${ONNXRUNTIME_VERSION}"
    diag_emit "installer.onnxruntime" "WARN" "INS-ORT-0001" "bundled onnxruntime version mismatch (${installed_ver} != ${ONNXRUNTIME_VERSION})" "installer will re-download compatible onnxruntime"
    return 1
  fi
  return 0
}

extract_tgz() {
  local archive="$1" dest="$2"
  if command -v tar >/dev/null 2>&1; then
    tar -xzf "$archive" -C "$dest"
    return 0
  fi
  if command -v bsdtar >/dev/null 2>&1; then
    bsdtar -xzf "$archive" -C "$dest"
    return 0
  fi
  echo "[error] need tar or bsdtar to extract onnxruntime archive"
  diag_emit "installer.onnxruntime" "FAIL" "INS-ORT-0002" "missing archive extractor (tar/bsdtar)" "install tar or bsdtar"
  exit 1
}

default_onnxruntime_urls() {
  local version="$1"
  case "$ARCH" in
    x86_64)
      echo "https://github.com/microsoft/onnxruntime/releases/download/v${version}/onnxruntime-linux-x64-gpu-${version}.tgz"
      echo "https://github.com/microsoft/onnxruntime/releases/download/v${version}/onnxruntime-linux-x64-${version}.tgz"
      ;;
    aarch64)
      echo "https://github.com/microsoft/onnxruntime/releases/download/v${version}/onnxruntime-linux-aarch64-${version}.tgz"
      echo "https://github.com/microsoft/onnxruntime/releases/download/v${version}/onnxruntime-linux-arm64-${version}.tgz"
      ;;
    *)
      return 1
      ;;
  esac
}

download_onnxruntime() {
  local cache="$ROOT/external/.cache"
  local dst="$ROOT/external/onnxruntime"
  mkdir -p "$cache"

  local version="${ONNXRUNTIME_VERSION:-1.20.1}"
  local url="${ONNXRUNTIME_URL:-}"

  local archive="$cache/onnxruntime-${version}-${ARCH}.tgz"
  local extract_dir="$cache/onnxruntime_extract"

  if [[ -n "$url" ]]; then
    echo "[info] downloading onnxruntime from pinned URL: $url"
    fetch_file "$url" "$archive"
  else
    local candidate_url
    local downloaded=false
    while IFS= read -r candidate_url; do
      [[ -z "$candidate_url" ]] && continue
      echo "[info] trying onnxruntime URL: $candidate_url"
      if fetch_file "$candidate_url" "$archive"; then
        url="$candidate_url"
        downloaded=true
        break
      fi
    done < <(default_onnxruntime_urls "$version")

    if [[ "$downloaded" != "true" ]]; then
      echo "[error] failed to download onnxruntime ${version} for ${ARCH}"
      diag_emit "installer.onnxruntime" "FAIL" "INS-ORT-0005" "failed to download onnxruntime archive" "set ONNXRUNTIME_URL to a valid release archive and retry"
      exit 1
    fi
  fi

  rm -rf "$extract_dir"
  mkdir -p "$extract_dir"
  extract_tgz "$archive" "$extract_dir"

  local lib
  lib="$(find "$extract_dir" -type f -name 'libonnxruntime.so*' | head -n1 || true)"
  if [[ -z "$lib" ]]; then
    echo "[error] extracted onnxruntime archive does not contain libonnxruntime.so"
    diag_emit "installer.onnxruntime" "FAIL" "INS-ORT-0003" "onnxruntime shared library missing after extraction" "verify ONNXRUNTIME_URL or ONNXRUNTIME_VERSION"
    exit 1
  fi

  local root_dir
  root_dir="$(dirname "$(dirname "$lib")")"
  rm -rf "$dst"
  mv "$root_dir" "$dst"
  rm -rf "$extract_dir"

  if ! is_valid_onnxruntime "$dst"; then
    echo "[error] downloaded onnxruntime is invalid: $dst"
    diag_emit "installer.onnxruntime" "FAIL" "INS-ORT-0004" "downloaded onnxruntime is invalid" "verify ONNXRUNTIME_URL or ONNXRUNTIME_VERSION"
    exit 1
  fi
  echo "[info] onnxruntime ready at: $dst"
}

ensure_onnxruntime() {
  if [[ -n "${ONNXRUNTIME_ROOT:-}" ]] && is_valid_onnxruntime "${ONNXRUNTIME_ROOT}"; then
    echo "[info] using ONNXRUNTIME_ROOT from env: ${ONNXRUNTIME_ROOT}"
    return 0
  fi

  if is_valid_onnxruntime "$ROOT/external/onnxruntime"; then
    if check_onnxruntime_version "$ROOT/external/onnxruntime"; then
      export ONNXRUNTIME_ROOT="$ROOT/external/onnxruntime"
      echo "[info] using bundled onnxruntime: ${ONNXRUNTIME_ROOT}"
      return 0
    else
      echo "[info] bundled onnxruntime version mismatch, re-downloading"
    fi
  fi

  echo "[info] provisioning onnxruntime for ${ARCH}"
  download_onnxruntime
  export ONNXRUNTIME_ROOT="$ROOT/external/onnxruntime"
}
