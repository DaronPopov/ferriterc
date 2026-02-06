#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_SH="$ROOT_DIR/scripts/ptx_env.sh"
BUILD_SH="$ROOT_DIR/scripts/ptx_build.sh"

if [ ! -x "$ENV_SH" ]; then
  echo "[ptx_bench] ERROR: $ENV_SH not found or not executable" >&2
  exit 1
fi

if [ ! -x "$BUILD_SH" ]; then
  echo "[ptx_bench] ERROR: $BUILD_SH not found or not executable" >&2
  exit 1
fi

OUT_FILE=""
DO_BUILD=1
BASELINE=0

usage() {
  cat <<USAGE
Usage: $0 [--no-build] [--baseline] [--out <file>]

  --no-build   Skip build step
  --baseline   Enable cudaMalloc/cudaFree baseline in alloc benchmark
  --out FILE   Write results to FILE (default: benchmarks/ptx_bench_YYYYMMDD_HHMMSS.txt)
USAGE
}

while [ $# -gt 0 ]; do
  case "$1" in
    --no-build)
      DO_BUILD=0
      shift
      ;;
    --baseline)
      BASELINE=1
      shift
      ;;
    --out)
      OUT_FILE="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

# Load centralized env
while IFS='=' read -r key val; do
  if [ -n "$key" ]; then
    export "$key=$val"
  fi
done < <("$ENV_SH" --format env --quiet)

export LD_LIBRARY_PATH="$ROOT_DIR/lib:${LD_LIBRARY_PATH:-}"

if [ -z "$OUT_FILE" ]; then
  ts="$(date +%Y%m%d_%H%M%S)"
  OUT_FILE="$ROOT_DIR/benchmarks/ptx_bench_${ts}.txt"
fi

mkdir -p "$(dirname "$OUT_FILE")"

run_cmd() {
  local title="$1"
  shift
  echo "=== $title ===" | tee -a "$OUT_FILE"
  echo "cmd: $*" | tee -a "$OUT_FILE"
  "$@" 2>&1 | tee -a "$OUT_FILE"
  echo "" | tee -a "$OUT_FILE"
}

{
  echo "PTX-OS Benchmark Run"
  echo "Timestamp: $(date)"
  echo ""
  echo "Environment:"
  "$ENV_SH" --format env --quiet
  echo ""
} | tee "$OUT_FILE"

if [ "$DO_BUILD" -eq 1 ]; then
  run_cmd "Build (core + rust)" "$BUILD_SH" all
fi

if [ "$BASELINE" -eq 1 ]; then
  export PTX_BENCH_BASELINE=1
  export PTX_BENCH_ASYNC=1
fi

run_cmd "Allocator Benchmark" bash -lc "cd '$ROOT_DIR/rust' && cargo run --release -p ptx-runtime --example bench_alloc"
run_cmd "Async Free Benchmark" bash -lc "cd '$ROOT_DIR/rust' && cargo run --release -p ptx-runtime --example bench_async_free"
run_cmd "Pipeline Benchmark" bash -lc "cd '$ROOT_DIR/rust' && cargo run --release -p ptx-runtime --example bench_pipeline"
run_cmd "Dynamic-Shape Benchmark" bash -lc "cd '$ROOT_DIR/rust' && cargo run --release -p ptx-tensor --example bench_dynamic_shapes"
run_cmd "Elementwise Ops Benchmark" bash -lc "cd '$ROOT_DIR/rust' && cargo run --release -p ptx-tensor --example bench_ops"
run_cmd "Matmul Benchmark" bash -lc "cd '$ROOT_DIR/rust' && cargo run --release -p ptx-tensor --example bench_matmul"

echo "Results saved to: $OUT_FILE" | tee -a "$OUT_FILE"
