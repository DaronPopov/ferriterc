# Reproducibility Guide

This document defines how Ferrite performance and capability claims must be produced and reported.

## Claim Tiers

- `Validated`: Measured on real workloads with reproducible commands and attached raw logs.
- `Demo`: Uses synthetic or simplified workloads; useful for architecture demos but not production-equivalent.
- `Aspirational`: Not yet measured in this repository; must not be presented as achieved.

## Ground Rules

- Every published metric must include:
  - Commit SHA
  - Run date/time
  - GPU model, VRAM size, driver version, CUDA version
  - Exact command used
  - Raw output log path
- If a workload is synthetic, label it `Demo` in the same sentence as the metric.
- If no artifact exists for a metric, treat it as `Aspirational`.

## Canonical Benchmark Command

Run from repo root:

```bash
cd ferriterc/ferrite-os
./scripts/ptx_bench_all.sh --out benchmarks/ptx_bench_$(date +%Y%m%d_%H%M%S).txt
```

This script records host, commit, and GPU metadata, then runs available benchmark examples.

## Suggested Validation Matrix

Run each of the following at least 3 times on an idle machine and report mean/stddev:

- `ptx-runtime`:
  - `jitter_benchmark`
  - `fused_kernel_benchmark`
  - `candle_performance_benchmark`
- `ptx-tensor`:
  - `bench_dynamic_shapes`
  - `bench_ops`
  - `bench_matmul`

## Workload Status Notes

- `finetune_engine/architectures/streaming_moe.rs` is currently a `Demo` workload for expert base weights.
  - The file explicitly notes synthetic shard generation.
  - Do not present its numbers as production MoE training parity until real checkpoint-backed experts are used.

## Reporting Template

Use this template when adding numbers to docs:

```text
Metric: <name>
Tier: Validated | Demo | Aspirational
Value: <number + unit>
Hardware: <GPU, VRAM, driver, CUDA>
Commit: <sha>
Command: <full command>
Raw log: <repo-relative path>
Notes: <baseline method, error bars, caveats>
```

## Documentation Policy

Before adding or updating benchmark values in `architecture.txt` or README files:

1. Run benchmark(s) and save raw logs in `ferrite-os/benchmarks/`.
2. Add a short summary table that links to raw log files.
3. Mark each row with `Validated`, `Demo`, or `Aspirational`.
4. Keep claim wording scoped to tested conditions (hardware + config).
