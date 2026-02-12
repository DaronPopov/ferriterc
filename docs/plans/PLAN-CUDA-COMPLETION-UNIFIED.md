# PLAN-CUDA-COMPLETION-UNIFIED

## Objective
Complete the remaining CUDA/runtime roadmap to production-ready status with one execution plan.

## Global Win Condition
All remaining work is complete when:
1. Code builds cleanly.
2. Tests pass (unit + integration + targeted GPU checks).
3. Runtime/daemon/script integrations function without placeholder behavior.
4. Remaining migration/TODO blockers are closed or explicitly deferred with tracked issues.

## Current Snapshot (from repo)
- Scheduler/control-plane integration appears largely implemented.
- Compiler pass realization is in progress.
- Safe API migration (`08`) is near completion but still has deprecated markers in gpu-lang runtime.
- Monte Carlo backend is still placeholder/stub.
- Non-contiguous tensor host export still TODO.
- Autograd has unsupported TODO gaps.
- Kernel coverage/perf TODOs remain.
- TLSF allocator compaction/expand/shrink is not implemented.

## Execution Order (single-track sequence)

### Phase 1: Close `08` Safe API Migration
Scope:
- Remove remaining deprecated call path usage in:
  - `ferrite-gpu-lang/src/lib.rs`
  - `ferrite-gpu-lang/src/runtime/tensor.rs`
- Ensure runtime kernel launches use `ptx_kernels::safe_api` + guard layer.

Deliverables:
- No `#[allow(deprecated)]` left for migration TODOs.
- Tests proving safe launch success/failure behavior.

Gate:
```bash
cargo build --release -p ferrite-gpu-lang
cargo test -p ferrite-gpu-lang
cd ferrite-os && cargo test -p ptx-kernels
```

### Phase 2: Finalize `07` Compiler Passes
Scope:
- Complete real constant fold + dead code elimination behavior.
- Ensure output/metadata invariants and deterministic pass results.

Deliverables:
- Non-noop graph rewrites for supported patterns.
- Strong pass tests for mixed constant/non-constant graphs.

Gate:
```bash
cd ferrite-os
cargo build --workspace --release
cargo test -p ptx-compiler
```

### Phase 3: Implement `09` Monte Carlo Backend
Scope:
- Replace Monte Carlo stubs in `ptx-compute` with real GPU-backed kernels.
- Add deterministic seed and CPU-reference correctness checks.

Deliverables:
- Working single-stream and multi-stream Monte Carlo APIs.
- Math-engine integration validated.

Gate:
```bash
cd ferrite-os
cargo test -p ptx-compute
cargo test -p ptx-kernels
cd ..
./ferrite-run mathematics_engine/monte_carlo/path_pricer.rs -- --paths 1000000
```

### Phase 4: Implement `10` Tensor Non-Contiguous Host Export
Scope:
- Add contiguous materialization fallback for `to_vec`/typed exports.
- Preserve contiguous fast path.

Deliverables:
- `to_vec` works for transposed/permuted views.
- Extended tests in tensor + gpu-lang harden suite.

Gate:
```bash
cd ferrite-os
cargo test -p ptx-tensor
cargo test -p ptx-runtime
cd ..
cargo run --release -p ferrite-gpu-lang --example test_harden
```

### Phase 5: Autograd Completeness Pass
Scope:
- Resolve high-impact `NotSupported` backward gaps first:
  - matmul backward
  - partial reduction backward broadcast behavior

Deliverables:
- Backward support coverage expanded for core training paths.
- Targeted gradient correctness tests.

Gate:
```bash
cd ferrite-os
cargo test -p ptx-autograd
cargo test -p ptx-tensor
```

### Phase 6: Kernel Coverage + Correctness Hardening
Scope:
- Close obvious kernel TODOs with correctness priority:
  - indexing bound/error behavior
  - missing min/max wrappers
  - key conv/reduce TODO hotspots

Deliverables:
- Added regression tests for edge cases and unsupported dtype behavior.
- Improved error messages/diagnostics for kernel failures.

Gate:
```bash
cd ferrite-os
cargo test -p ptx-kernels
cargo run --release -p ptx-kernels --example test_candle
cargo run --release -p ptx-kernels --example test_candle_tlsf
```

### Phase 7: TLSF Allocator Maturity (Post-feature stabilization)
Scope:
- Implement or formally defer with tracking for:
  - compaction
  - pool expansion
  - pool shrinking

Deliverables:
- Either production implementation + tests, or explicit defer docs with acceptance criteria and issue IDs.

Gate:
```bash
cd ferrite-os
make
cargo build --workspace --release
./scripts/ptx_doctor.sh
```

## Final Release Gate (must pass)
Run after all phases:
```bash
cd ferrite-os
make
cargo build --workspace --release
cargo test --workspace
./scripts/release-gate.sh
cd ..
cargo build --release -p ferrite-gpu-lang
cargo test -p ferrite-gpu-lang
```
If GPU environment is available:
```bash
cd ferrite-os
./test.sh
cargo run --release -p ptx-runtime --example parallel_batch_processing
cargo run --release -p ptx-runtime --example transformer_attention_layer
cd ..
./ferrite-run mathematics_engine/monte_carlo/path_pricer.rs -- --paths 1000000
```

## Required Artifacts Per Phase
- Changed file list + rationale.
- Behavior delta summary (before/after).
- Command outputs for phase gate.
- Known risks and follow-ups.

## Exit Criteria
Plan is complete only when:
- All 7 phases are closed.
- Final release gate passes.
- No unresolved high-priority TODO/stub paths remain in runtime-critical code.
