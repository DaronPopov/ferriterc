# GPU Manycore Execution Roadmap

Status: Draft  
Scope: `ferrite-os` + `ferrite-gpu-lang`  
Goal: Make Ferrite's runtime model behave more like a GPU-native manycore OS, where GPU SMs execute queued work as long-lived workers (not just per-op host launches).

## Why This Document

The current stack already has strong host-side scheduling, stream parallelism, memory management, and JIT graph compilation.  
The next step is to make device-side execution semantics first-class so "GPU as parallel CPU cores" is operationally true, not just conceptual.

## Current State (Readthrough Assessment)

What exists now:

1. Persistent-kernel scaffolding exists.
   - Host boot hook: `ptx_os_boot_persistent_kernel(...)` in `ferrite-os/native/core/runtime/modules/hot_runtime_shared_context.inl`.
   - Device kernel loop + task polling exists in `ferrite-os/native/core/kernels/os_kernel.cu`.
   - Host enqueue API exists: `ptx_os_submit_task(...)` in `ferrite-os/native/core/runtime/modules/hot_runtime_tasks.inl`.
   - Runtime exposure exists via `PtxRuntime::submit_task(...)` in `ferrite-os/crates/public/ptx-runtime/src/runtime/resilience.rs`.
   - Daemon boot is opt-in (`boot_kernel`), and dev config currently defaults to `false` in `ferrite-os/crates/internal/ptx-daemon/dev-config.toml`.
2. Main production execution path is still host-driven graph/kernels.
   - `ptx-compiler` compiles to CUDA graph backend by default (`ferrite-os/crates/internal/ptx-compiler/src/ir/graph/pipeline.rs`).
   - Dispatch backend exists but is not the default execution path (`ferrite-os/crates/internal/ptx-compiler/src/backend/ptx_dispatch.rs`).
3. Runtime scheduler is host-side stream assignment and quota admission.
   - Multi-tenant scheduler/dispatcher in `ferrite-os/crates/public/ptx-runtime/src/scheduler/*`.
4. JIT language is graph-oriented and compiles to runtime ops.
   - Full JIT pipeline in `ferrite-gpu-lang/src/jit/execute.rs`.
   - Tile annotations and shape contracts are currently compile-time semantics, not direct device micro-scheduler directives (`ferrite-gpu-lang/src/jit/validate.rs`, `ferrite-gpu-lang/src/jit/lower.rs`).

Practical conclusion:

- Ferrite today is a strong GPU runtime + graph compiler stack.
- It is not yet a full device-resident work-queue executor with robust kernel-level fairness, preemption model, and tenant isolation on GPU workers.

## Target Model

### Control/Data Plane Split

```text
Host Control Plane
  - admission, quotas, policy, DAG submission, observability
  - fills device-visible work queues
        |
        v
Device Data Plane
  - persistent worker kernel(s) per GPU
  - fetch/decode/execute work descriptors
  - manage dependencies, retries, and completion signals
```

### Worker-Style Execution Loop

```text
for each resident worker group on GPU:
  pop next ready task from priority queue
  resolve deps + resource claims
  run operator/kernels
  publish completion + metrics
  continue (no host relaunch required)
```

## Architecture Boundaries (Keep Explicit)

1. Subsystem B: CUDA OS Layer (`ferrite-os/native/core/**`)
   - Owns persistent kernel runtime, queue ABI, device-side execution.
2. Subsystem C: Rust Runtime Core (`ptx-runtime`, `ptx-compiler`, `ptx-tensor`, `ptx-autograd`)
   - Owns safe APIs, graph lowering, host scheduler integration, telemetry.
3. Language layer (`ferrite-gpu-lang`)
   - Owns orchestration semantics, contracts, and lowering to runtime descriptors.

Rule: Language semantics should not bypass runtime invariants; they lower into stable runtime descriptors/ABI.

## Design Principles

1. Stable descriptor ABI before feature growth.
2. Queue correctness and observability before peak performance tuning.
3. Host and device schedulers must share one canonical job/task state model.
4. Deterministic fallback path (host launch path) must remain available for debugging.
5. Tenant isolation and memory accounting are mandatory, not optional.

## Roadmap

## Phase 0: Baseline and Contracts

Deliverables:

1. Define `PTXTaskDescV1` and `PTXTaskResultV1` ABI (versioned, fixed-size, no raw host pointers except explicit shared handles).
2. Define task lifecycle state machine shared by host scheduler and device executor.
3. Add metrics contract:
   - enqueue latency
   - queue depth by priority
   - dispatch-to-start latency
   - completion latency
   - failure codes

Exit criteria:

- ABI documented and validated by compile-time size/alignment tests in `ptx-sys`.

## Phase 1: Real Queue Substrate

Deliverables:

1. Replace/extend single ring queue with:
   - per-priority lock-free ring buffers
   - completion queue
   - dead-letter/error queue
2. Add host/device memory fences and sequence counters for correctness.
3. Add backpressure semantics (bounded queue, explicit denial reasons).

Exit criteria:

- Queue survives stress at high submission rates without corruption or silent drops.

## Phase 2: Device Executor MVP

Deliverables:

1. Persistent worker kernel v1:
   - one resident worker group per SM class (configurable)
   - pop/execute/commit loop
2. Operator dispatch table on device:
   - start with existing tensor op subset (relu/add/mul/reduce/matmul).
3. Completion signaling back to host scheduler.

Exit criteria:

- End-to-end graph segments run through device queue path with correctness parity vs host launch path.

## Phase 3: Dependency-Aware Task Graphs

Deliverables:

1. Device-visible DAG metadata:
   - dependency counters
   - ready-list promotion
2. Task chaining without host round-trips for short operator chains.
3. Failure propagation semantics (fail-fast per graph/job).

Exit criteria:

- Multi-op chains execute with reduced host wakeups and measurable launch-overhead reduction.

## Phase 4: Fairness and Isolation on Device

Deliverables:

1. Tenant-aware queues on device.
2. Weighted/fair-share dispatch strategy mirroring host policy.
3. VRAM/token budgets enforced at task-claim time.

Exit criteria:

- No tenant starvation under mixed-load stress; quotas enforced consistently.

## Phase 5: Language-Orchestration Integration

Deliverables:

1. Add orchestration IR in `ferrite-gpu-lang` for stage/task semantics:
   - `stage`, `depends_on`, `retry`, `timeout`, `checkpoint`.
2. Lower orchestration IR to `PTXTaskDescV1` batches.
3. Keep existing JIT graph mode as a backend target.

Exit criteria:

- Same language can emit either host-graph execution or device-queue execution.

## Phase 6: Multi-GPU and Distributed Work Stealing

Deliverables:

1. Device-group task queues and peer routing.
2. Cross-GPU completion propagation.
3. Optional NCCL-aware collective task kinds.

Exit criteria:

- One logical job graph can execute across multiple GPUs with bounded skew.

## Immediate MVP Slice (Recommended Next Build)

Implement first:

1. `PTXTaskDescV1` + `PTXTaskResultV1` structs in `ptx-sys` and native headers.
2. Dedicated completion queue in shared state.
3. `submit_task_v1()` and `poll_completion_v1()` in `ptx-runtime`.
4. Device executor handles 5 opcodes:
   - `NOP`
   - `UNARY_F32`
   - `BINARY_F32`
   - `REDUCE_SUM_F32`
   - `SHUTDOWN`
5. Integration benchmark:
   - compare host-launched micro-ops vs queued persistent execution for small ops.

This slice gives the fastest proof that Ferrite is shifting from "launch-driven runtime" to "resident worker runtime."

## Validation Plan

Correctness:

1. Differential tests: same inputs through host path and device-queue path.
2. Fault injection: invalid opcode, queue overflow, worker timeout.
3. Queue invariants: monotonic sequence, no duplicate completions.

Performance:

1. P50/P99 enqueue-to-start latency.
2. Small-op throughput improvement vs host launch baseline.
3. CPU utilization drop during high task-rate workloads.

Reliability:

1. Daemon restart with in-flight queue recovery behavior defined.
2. Watchdog + emergency flush interactions validated.

## Risks and Mitigations

1. Risk: queue races and memory ordering bugs.
   - Mitigation: sequence-number protocol, aggressive stress/fuzz tests.
2. Risk: persistent kernel reduces available SMs for user compute.
   - Mitigation: configurable residency and dynamic worker throttling.
3. Risk: divergence between host scheduler policy and device behavior.
   - Mitigation: shared policy metadata + conformance tests.

## Success Criteria

This roadmap is successful when all are true:

1. A meaningful subset of workloads executes via device-resident workers by default.
2. Host launch overhead is no longer dominant for fine-grained task chains.
3. Tenant fairness/quotas hold under stress in both host and device paths.
4. `ferrite-gpu-lang` can act as orchestration frontend, not only graph DSL.
