# GPU ISA v0 Design

Status: Draft  
Scope: `ferrite-os` + `ferrite-gpu-lang`  
Objective: Add a minimal device ISA execution layer so Ferrite can run bounded instruction streams on resident GPU workers, not only fixed opcodes.

## Implementation Status (as of February 13, 2026)

Implemented now:

1. `ISA_RUN` task opcode + ABI structs in native header and `ptx-sys`.
2. Persistent-kernel decode/execute loop with:
   - Phase A: `NOP`, `HALT`, `TRAP`, `YIELD`
   - Phase B subset: `JMP`, `MOVI`, `ADD`, `SUB`, `BR_EQ`, `ASSERT_EQI`
   - Phase C subset: `LD_U32`, `ST_U32`, `LD_CONST`, `SYSCALL` (`SYS_YIELD`, `SYS_SIGNAL`)
3. Deterministic trap classes including invalid PC/opcode/assert/memory/syscall cases.
4. Daemon submit surface:
   - `task-submit-isa-v0 <tenant> [mode] [slice_steps] [priority]`
   - current inline conformance modes include:
     `halt`, `trap`, `yield`, `movi`, `arith`, `branch`, `jmp`, `pc_oob`, `mem_ld`, `mem_oob`, `sys_yield`, `sys_bad`
5. Integration coverage in `daemon_integration` for these inline modes and scheduler compatibility.

Not implemented yet:

1. Dedicated `task-poll-isa-v0` API/command (current flow uses `task-poll-v1` completions).
2. Full `ferrite-gpu-lang` ISA backend lowering and backend selector rollout.
3. Full opcode families listed in this design (`BR_NE`, `BR_LT`, `BR_GE`, vector/fp families, richer syscall table).

## Why This Is The Next Piece

Ferrite already has:

1. A persistent-kernel scheduler (`os_kernel.cu`).
2. Versioned task ABI (`PTXTaskDescV1` / completion ring).
3. Cooperative yielding, fairness (`vruntime`), and tenant budgets.
4. DAG dependency and continuation semantics.

What is missing is a stable "program format" between language lowering and device execution.  
GPU ISA v0 provides that format.

## Boundaries

1. Subsystem B (CUDA OS layer) owns decode/execute, VM state, and task lifecycle on device.
2. Subsystem C (Rust runtime core) owns host-side program packaging, submission, and polling APIs.
3. Language layer (`ferrite-gpu-lang`) owns lowering from language IR into ISA bytecode.

Rule: language emits ISA packets through runtime APIs; it does not bypass scheduler or queue ABI.

## Execution Model

```text
Ferrite Language IR
  -> ISA Lowering (bytecode + metadata)
  -> task-submit-v1 (opcode = ISA_RUN)
  -> persistent kernel scheduler
  -> fetch/decode/execute N instructions
  -> yield or complete
  -> completion ring (+ optional continuation)
```

## Task ABI Mapping

`ISA_RUN` uses task descriptor slots:

1. `args[0]`: pointer/handle to `PTXISAProgramV0`
2. `args[1]`: pointer/handle to `PTXISAContextV0`
3. `args[2]`: max instructions per slice (budget hint)
4. `args[3]`: reserved for future (debug mask / trace mode)
5. `args[4..7]`: remain scheduler-reserved metadata (`work`, `quantum`, `depends_on`, `continuation`)

Design constraint: ISA execution must remain compatible with existing flags (`WAIT_ON_TASK`, `CONTINUATION`, `COOPERATIVE`).

## Core Data Structures

```c
typedef struct PTXISAProgramV0 {
    uint32_t abi_version;      // must match ISA v0
    uint32_t flags;            // readonly/debug bits
    uint32_t code_words;       // number of 64-bit words
    uint32_t const_bytes;      // const segment size
    uint64_t code_ptr;         // device-visible pointer/handle
    uint64_t const_ptr;        // device-visible pointer/handle
    uint32_t entry_pc;         // instruction index
    uint32_t reserved;
} PTXISAProgramV0;

typedef struct PTXISAContextV0 {
    uint32_t abi_version;
    uint32_t state_flags;      // running, halted, trapped
    uint32_t trap_code;        // nonzero on error
    uint32_t last_opcode;
    uint32_t regs[32];         // GPR file
    uint32_t pc;               // instruction index
    uint32_t pred;             // predicate register
    uint32_t mem_size;         // bytes in linear memory window
    uint64_t mem_ptr;          // base pointer/handle
    uint32_t steps_total;      // lifetime executed instructions
    uint32_t steps_last_slice; // last dispatch count
} PTXISAContextV0;
```

## Instruction Encoding v0

Fixed 64-bit word, little-endian:

1. `opcode` (8)
2. `fmt` (4)
3. `rd` (5)
4. `rs0` (5)
5. `rs1` (5)
6. `rs2` (5)
7. `imm32` (32)

Rationale:

1. Fixed width gives cheap decode in persistent kernel loop.
2. 32-bit immediates avoid extension words for common control/address math.
3. 32 registers are enough for v0 while keeping context compact.

## Opcode Families v0

Control:

1. `NOP`
2. `HALT`
3. `TRAP imm32`
4. `JMP imm32`
5. `BR_EQ`, `BR_NE`, `BR_LT`, `BR_GE` (PC-relative)

Integer/bit ops:

1. `MOV`, `MOVI`
2. `ADD`, `SUB`, `MUL`, `MAD`
3. `AND`, `OR`, `XOR`, `SHL`, `SHR`
4. `MIN`, `MAX`

Memory (bounds-checked against `mem_size`):

1. `LD_U32 rd, [rs0 + imm32]`
2. `ST_U32 [rs0 + imm32], rs1`
3. `LD_CONST rd, [imm32]`

Runtime/sys:

1. `YIELD` (cooperative yield, no completion)
2. `SYSCALL imm32` (small, fixed syscall table)

## Syscall Surface v0

Small and explicit:

1. `SYS_YIELD` -> mark yielded and return to scheduler.
2. `SYS_SIGNAL` -> OR signal bits into runtime signal mask.
3. `SYS_TASK_SPAWN` -> enqueue predefined opcode task (bounded, policy checked).
4. `SYS_METRIC_ADD` -> bump named counter bucket.

Non-goal in v0: arbitrary host callbacks from device bytecode.

## Scheduler Integration

`ISA_RUN` dispatch algorithm:

1. Scheduler selects task with existing priority + `vruntime` + tenant budget rules.
2. Kernel executes up to `slice_steps` instructions (`min(task budget, args[2], global cap)`).
3. Outcomes:
   - `HALT` -> complete status OK.
   - `YIELD`/step budget exhausted -> status yielded, task remains active.
   - `TRAP`/decode fault/oob memory -> complete with runtime error status.
4. Charge tenant budget by executed instruction count.
5. Update `vruntime` by executed instruction count.

This keeps one accounting unit for cooperative tasks and ISA tasks.

## Safety and Determinism

Required invariants:

1. PC must stay within `code_words`.
2. All memory accesses must be in-bounds (`addr + width <= mem_size`).
3. `SYSCALL` ids outside table trap with deterministic error.
4. Max instructions per slice enforced regardless of bytecode content.
5. No raw host pointers embedded in bytecode; use runtime-managed handles/pointers only.

## Host API Additions (Runtime Layer)

Additions in `ptx-runtime`:

1. `submit_isa_v0(program, context, opts) -> task_id`
2. `poll_isa_v0(task_id) -> {running|yielded|completed|trapped}`
3. `alloc_isa_memory(bytes) -> handle/pointer`
4. `encode_isa_v0(words, const_segment) -> PTXISAProgramV0`

Daemon command additions:

1. `task-submit-isa-v0 <tenant> [mode] [slice_steps] [priority]`
2. `task-poll-v1` (current completion polling path for ISA tasks)

Current inline conformance mode set:

1. `halt`, `trap`, `yield`
2. `movi`, `arith`, `branch`, `jmp`, `pc_oob`
3. `mem_ld`, `mem_oob`, `sys_yield`, `sys_bad`

## Language Integration

`ferrite-gpu-lang` adds a backend mode:

1. Existing graph backend remains default and fallback.
2. ISA backend lowers a verified subset first:
   - integer loops
   - control flow
   - scalar reductions
   - runtime syscalls (`yield`, metrics, signal)
3. Unsupported constructs fallback to existing graph/operator path.

This keeps rollout incremental and avoids freezing language evolution.

## Rollout Plan

Phase A: Skeleton

1. Reserve `ISA_RUN` opcode and structures in native header + `ptx-sys`.
2. Add decode loop with `NOP/HALT/TRAP/YIELD`.
3. Add one integration test that executes a 3-instruction program.

Phase B: Arithmetic + Branch

1. Add integer ALU + branch opcodes.
2. Add deterministic trap codes.
3. Add conformance tests for branch behavior and PC rules.

Phase C: Memory + Syscalls

1. Add bounds-checked loads/stores.
2. Add minimal syscall table.
3. Add multi-tenant stress test with budget fairness for ISA tasks.

Phase D: Language Backend

1. Add lowering in `ferrite-gpu-lang`.
2. Add backend selection flag (`graph` vs `isa-v0`).
3. Add differential tests vs graph path for supported subset.

## Validation Matrix

Correctness:

1. ISA decode/execute golden tests.
2. Trap coverage tests (bad opcode, bad PC, oob memory).
3. Differential tests against host reference interpreter.

Scheduler behavior:

1. No starvation under mixed tenants.
2. Cooperative yield/resume order stability.
3. DAG + continuation compatibility with ISA tasks.

Operational:

1. Daemon restart should reset stale ISA scheduler state.
2. Completion ring integrity under high instruction-rate workloads.

## Non-Goals For v0

1. Floating-point vector ISA.
2. Full Rust semantics in-device.
3. General-purpose dynamic linking or dynamic code generation on GPU.
4. Cross-GPU bytecode migration.

## Success Criteria

GPU ISA v0 is successful when:

1. Ferrite can submit, run, yield, and complete bytecode tasks through the existing scheduler path.
2. Tenant budget/fairness rules apply uniformly to ISA and non-ISA cooperative work.
3. `ferrite-gpu-lang` can target ISA v0 for a useful subset while preserving graph fallback.
