#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export LD_LIBRARY_PATH="${ROOT_DIR}/lib:${LD_LIBRARY_PATH:-}"

exec "${ROOT_DIR}/rust/target/release/ptx-daemon" "$@"
