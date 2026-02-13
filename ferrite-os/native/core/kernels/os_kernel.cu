#include <cuda_runtime.h>
#include <stdio.h>
#include "gpu/gpu_hot_runtime.h"

// ============================================================================
// PTX-OS Persistent Kernel
// This kernel stays resident in the GPU, polling for tasks and managing 
// system state without CPU intervention.
// ============================================================================

__device__ inline void ptx_os_push_completion(
    PTXSystemState* state,
    const PTXOSTask* task,
    uint32_t status
) {
    PTXTaskResultQueue* cq = &state->completion_queue;
    uint32_t head = cq->head;
    uint32_t next_head = (head + 1) % PTX_MAX_COMPLETION_QUEUE_SIZE;

    // Drop with overrun accounting when completion queue is full.
    if (next_head == cq->tail) {
        atomicAdd((unsigned int*)&cq->overruns, 1u);
        return;
    }

    PTXTaskResultV1* out = &cq->results[head];
    out->abi_version = PTX_TASK_ABI_V1;
    out->task_id = task->task_id;
    out->opcode = task->opcode;
    out->priority = task->priority;
    out->tenant_id = task->tenant_id;
    out->status = status;
    out->submitted_at = task->submitted_at;
    out->started_at = task->started_at;
    out->completed_at = task->completed_at;

    __threadfence_system();
    cq->head = next_head;
}

__device__ inline bool ptx_os_task_id_completed(const PTXSystemState* state, uint32_t task_id) {
    if (task_id == 0) {
        return true;
    }

    // Fast path: check completion ring slots (including entries already polled by host).
    const PTXTaskResultQueue* cq = &state->completion_queue;
    for (uint32_t i = 0; i < PTX_MAX_COMPLETION_QUEUE_SIZE; ++i) {
        const PTXTaskResultV1* done = &cq->results[i];
        if (done->abi_version == PTX_TASK_ABI_V1 && done->task_id == task_id) {
            return true;
        }
    }

    // Fallback: check in-queue task records that have already completed.
    const PTXTaskQueue* q = &state->queue;
    for (uint32_t i = 0; i < PTX_MAX_QUEUE_SIZE; ++i) {
        const PTXOSTask* t = &q->tasks[i];
        if (t->task_id == task_id && t->completed) {
            return true;
        }
    }
    return false;
}

__device__ inline void ptx_os_enqueue_continuation(PTXSystemState* state, const PTXOSTask* parent) {
    if ((parent->flags & PTX_TASK_FLAG_CONTINUATION) == 0) {
        return;
    }

    uint32_t continuation_opcode = (uint32_t)(uintptr_t)parent->args[PTX_TASK_META_ARG_CONTINUATION];
    PTXTaskQueue* q = &state->queue;
    uint32_t head = q->head;
    uint32_t next_head = (head + 1) % PTX_MAX_QUEUE_SIZE;
    if (next_head == q->tail) {
        return; // queue full: drop continuation
    }

    PTXOSTask* task = &q->tasks[head];
    uint32_t task_id = atomicAdd((unsigned int*)&state->next_task_id, 1u) + 1u;
    task->task_id = task_id;
    task->opcode = continuation_opcode;
    task->priority = parent->priority;
    task->tenant_id = parent->tenant_id;
    task->flags = 0;
    task->arg_count = 0;
    task->yield_count = 0;
    task->active = true;
    task->completed = false;
    for (int i = 0; i < PTX_MAX_TASK_ARGS; ++i) {
        task->args[i] = NULL;
    }
    task->submitted_at = clock64();
    task->started_at = 0;
    task->completed_at = 0;
    task->vruntime = state->tenant_vruntime[parent->tenant_id % PTX_MAX_QUEUE_SIZE];

    __threadfence_system();
    q->head = next_head;
    atomicAdd((int*)&state->active_tasks, 1);
}

__device__ inline uint32_t ptx_os_task_slice_hint(const PTXOSTask* task) {
    if (task->opcode == PTX_TASK_OPCODE_COOPERATIVE_WORK) {
        uint32_t remaining = (uint32_t)(uintptr_t)task->args[PTX_TASK_META_ARG_WORK_REMAINING];
        uint32_t quantum = (uint32_t)(uintptr_t)task->args[PTX_TASK_META_ARG_QUANTUM];
        if (remaining == 0) {
            remaining = 1024;
        }
        if (quantum == 0) {
            quantum = 64;
        }
        uint32_t slice = remaining < quantum ? remaining : quantum;
        return slice == 0 ? 1 : slice;
    }
    if (task->opcode == PTX_TASK_OPCODE_ISA_RUN) {
        uint32_t slice_steps = (uint32_t)(uintptr_t)task->args[2];
        if (slice_steps == 0) {
            slice_steps = 64;
        }
        return slice_steps;
    }
    return 1;
}

__device__ inline uint32_t ptx_os_refresh_tenant_budget(
    PTXSystemState* state,
    uint32_t tenant_bucket,
    uint32_t epoch
) {
    uint32_t last_epoch = state->tenant_budget_epoch[tenant_bucket];
    uint32_t budget = state->tenant_budget[tenant_bucket];

    if (last_epoch == 0) {
        last_epoch = epoch;
        if (budget == 0) {
            budget = PTX_TENANT_BUDGET_INITIAL;
        }
    }

    if (epoch > last_epoch) {
        uint32_t delta = epoch - last_epoch;
        uint64_t refill = (uint64_t)budget + ((uint64_t)delta * (uint64_t)PTX_TENANT_BUDGET_REFILL_PER_TICK);
        budget = refill > PTX_TENANT_BUDGET_MAX ? PTX_TENANT_BUDGET_MAX : (uint32_t)refill;
        last_epoch = epoch;
    }

    state->tenant_budget[tenant_bucket] = budget;
    state->tenant_budget_epoch[tenant_bucket] = last_epoch;
    return budget;
}

__device__ inline void ptx_os_charge_tenant_budget(
    PTXSystemState* state,
    uint32_t tenant_bucket,
    uint32_t epoch,
    uint32_t charge
) {
    if (charge == 0) {
        return;
    }
    uint32_t budget = ptx_os_refresh_tenant_budget(state, tenant_bucket, epoch);
    state->tenant_budget[tenant_bucket] = budget > charge ? (budget - charge) : 0;
    state->tenant_budget_epoch[tenant_bucket] = epoch;
}

__device__ inline uint8_t ptx_isa_v0_decode_opcode(uint64_t word) {
    return (uint8_t)(word & 0xFFu);
}

__device__ inline uint8_t ptx_isa_v0_decode_fmt(uint64_t word) {
    return (uint8_t)((word >> 8) & 0x0Fu);
}

__device__ inline uint8_t ptx_isa_v0_decode_rd(uint64_t word) {
    return (uint8_t)((word >> 12) & 0x1Fu);
}

__device__ inline uint8_t ptx_isa_v0_decode_rs0(uint64_t word) {
    return (uint8_t)((word >> 17) & 0x1Fu);
}

__device__ inline uint8_t ptx_isa_v0_decode_rs1(uint64_t word) {
    return (uint8_t)((word >> 22) & 0x1Fu);
}

__device__ inline uint8_t ptx_isa_v0_decode_rs2(uint64_t word) {
    return (uint8_t)((word >> 27) & 0x1Fu);
}

__device__ inline uint32_t ptx_isa_v0_decode_imm32(uint64_t word) {
    return (uint32_t)((word >> 32) & 0xFFFFFFFFu);
}

__device__ inline int32_t ptx_isa_v0_decode_imm32_s(uint64_t word) {
    return (int32_t)ptx_isa_v0_decode_imm32(word);
}

__device__ inline bool ptx_isa_v0_calc_target_pc(
    uint32_t pc,
    int32_t rel,
    uint32_t code_words,
    uint32_t* out_pc
) {
    int64_t target = (int64_t)(int32_t)pc + (int64_t)rel;
    if (target < 0 || (uint64_t)target >= (uint64_t)code_words) {
        return false;
    }
    *out_pc = (uint32_t)target;
    return true;
}

__device__ inline bool ptx_isa_v0_resolve_u32_addr(
    uint64_t base_ptr,
    uint32_t mem_size,
    uint32_t base_off,
    uint32_t imm,
    uint64_t* out_addr
) {
    if (base_ptr == 0) {
        return false;
    }
    uint64_t offset64 = (uint64_t)base_off + (uint64_t)imm;
    if (offset64 > 0xFFFFFFFFULL) {
        return false;
    }
    uint64_t end = offset64 + sizeof(uint32_t);
    if (end > (uint64_t)mem_size) {
        return false;
    }
    *out_addr = base_ptr + offset64;
    return true;
}

__global__ void ptx_os_kernel(PTXSystemState* state) {
    uint32_t tid = threadIdx.x; // We assume 1D block for OS management
    
    // Boot sequence
    if (tid == 0) {
        state->kernel_running = true;
        state->shutdown_requested = false;
        state->scheduler_epoch = 1;
        for (uint32_t i = 0; i < PTX_MAX_QUEUE_SIZE; ++i) {
            state->tenant_budget[i] = PTX_TENANT_BUDGET_INITIAL;
            state->tenant_budget_epoch[i] = state->scheduler_epoch;
        }
        printf("[GPU-OS] Kernel Life-Cycle Initialized. VRAM OS is now ACTIVE.\n");
    }
    
    __syncthreads();
    
    uint64_t iterations = 0;
    
    // The "infinite" loop of the OS
    while (!state->shutdown_requested) {
        // Standard block sync
        __syncthreads();
        
        // Thread 0: Primary Scheduler / Task Dispatcher
        if (tid == 0) {
            // Check Task Queue
            if (state->queue.head != state->queue.tail) {
                int best_idx = -1;
                int min_priority = 256;
                uint64_t min_vruntime = ~0ULL;
                uint32_t best_slice_hint = 1;
                uint32_t sched_epoch = state->scheduler_epoch + 1;
                state->scheduler_epoch = sched_epoch;
                
                uint32_t current_tail = state->queue.tail;
                uint32_t current_head = state->queue.head;
                
                for (uint32_t i = current_tail; i != current_head; i = (i + 1) % PTX_MAX_QUEUE_SIZE) {
                    PTXOSTask* candidate = &state->queue.tasks[i];
                    if (candidate->active && !candidate->completed) {
                        if (candidate->flags & PTX_TASK_FLAG_WAIT_ON_TASK) {
                            uint32_t dep_task_id =
                                (uint32_t)(uintptr_t)candidate->args[PTX_TASK_META_ARG_DEPENDENCY];
                            if (!ptx_os_task_id_completed(state, dep_task_id)) {
                                continue;
                            }
                        }

                        uint32_t tenant_bucket = candidate->tenant_id % PTX_MAX_QUEUE_SIZE;
                        uint32_t budget = ptx_os_refresh_tenant_budget(state, tenant_bucket, sched_epoch);
                        uint32_t slice_hint = ptx_os_task_slice_hint(candidate);
                        if (budget < slice_hint) {
                            continue;
                        }
                        if (candidate->priority < min_priority ||
                            (candidate->priority == min_priority && candidate->vruntime < min_vruntime)) {
                            min_priority = candidate->priority;
                            min_vruntime = candidate->vruntime;
                            best_slice_hint = slice_hint;
                            best_idx = i;
                            if (min_priority == 0) break; // Realtime optimization
                        }
                    }
                }

                if (best_idx != -1) {
                    PTXOSTask* task = &state->queue.tasks[best_idx];
                    uint32_t tenant_bucket = task->tenant_id % PTX_MAX_QUEUE_SIZE;
                    uint32_t tenant_budget = ptx_os_refresh_tenant_budget(state, tenant_bucket, sched_epoch);
                    if (tenant_budget < best_slice_hint) {
                        continue;
                    }
                    state->active_priority_level = task->priority;
                    if (task->started_at == 0) {
                        task->started_at = clock64();
                    }
                    uint32_t task_status = PTX_TASK_STATUS_OK;
                    bool task_finished = true;
                    uint32_t consumed_ticks = 0;
                    uint32_t budget_charge = 1;
                    
                    switch (task->opcode) {
                        case PTX_TASK_OPCODE_NOP: // NOP
                            break;
                            
                        case PTX_TASK_OPCODE_COMPUTE: // COMPUTE - dispatch kernel via function pointer in args[0]
                            {
                                typedef void (*compute_fn_t)(void**, int);
                                compute_fn_t fn = (compute_fn_t)task->args[0];
                                if (fn) {
                                    fn<<<1, 256>>>(&task->args[1], task->priority);
                                } else {
                                    task_status = PTX_TASK_STATUS_RUNTIME_ERROR;
                                }
                            }
                            break;
                            
                        case PTX_TASK_OPCODE_SHUTDOWN: // SHUTDOWN
                            state->shutdown_requested = true;
                            break;

                        case PTX_TASK_OPCODE_SWAP_IN: // SWAP_IN (Virtual Memory Manager)
                            printf("[GPU-VMM] Swapping task_id %d back to Resident VRAM\n", task->task_id);
                            // Signal host to perform VMM swap via signal_mask bit 2
                            atomicOr((unsigned long long*)&state->signal_mask, 0x4ULL);
                            break;

                        case PTX_TASK_OPCODE_VFS_MOUNT: // VFS_MOUNT (Tensor Filesystem)
                            state->fs_node_count++;
                            printf("[PTX-FS] Mounted Segment Index: %d | Nodes Active: %d\n", (int)(size_t)task->args[0], state->fs_node_count);
                            break;

                        case PTX_TASK_OPCODE_INTERRUPT: // INTERRUPT (Simulated hardware interrupt)
                            printf("[PTX-INT] Software Interrupt Generated: %p\n", task->args[0]);
                            atomicOr((unsigned long long*)&state->signal_mask, (unsigned long long)task->args[0]);
                            break;

                        case PTX_TASK_OPCODE_LAUNCH_KERNEL: // LAUNCH_KERNEL (Recursive Task Dispatch / CDP)
                            {
                                 // Cast the task's argument buffer directly to a PTXKernelLaunch descriptor
                                 PTXKernelLaunch* launch = (PTXKernelLaunch*)&task->args[0];
                                 if (launch && launch->kernel_func) {
                                     printf("[PTX-CDP] Recursive Launch: Func=%p | Grid=(%d,%d) | Block=%d\n", 
                                            launch->kernel_func, launch->grid.x, launch->grid.y, launch->block.x);
                                     
                                     typedef void (*kernel_ptr_t)(void**);
                                     kernel_ptr_t func = (kernel_ptr_t)launch->kernel_func;
                                     
                                     // Pass the address of the inline argument array
                                     func<<<launch->grid, launch->block, launch->shared_mem, launch->stream>>>(launch->arg_values);
                                 }
                            }
                            break;

                        case PTX_TASK_OPCODE_COOPERATIVE_WORK: // COOPERATIVE_WORK (time-sliced synthetic workload)
                            {
                                uint32_t remaining =
                                    (uint32_t)(uintptr_t)task->args[PTX_TASK_META_ARG_WORK_REMAINING];
                                uint32_t quantum =
                                    (uint32_t)(uintptr_t)task->args[PTX_TASK_META_ARG_QUANTUM];

                                if (remaining == 0) {
                                    remaining = 1024;
                                }
                                if (quantum == 0) {
                                    quantum = 64;
                                }

                                uint32_t slice = remaining < quantum ? remaining : quantum;
                                if (slice > tenant_budget) {
                                    slice = tenant_budget;
                                }
                                if (slice == 0) {
                                    task_status = PTX_TASK_STATUS_YIELDED;
                                    task_finished = false;
                                    budget_charge = 0;
                                    break;
                                }
                                volatile uint32_t sink = 0;
                                uint32_t iters = slice * 4096;
                                for (uint32_t i = 0; i < iters; ++i) {
                                    sink += (i ^ task->task_id);
                                }
                                (void)sink;

                                remaining -= slice;
                                consumed_ticks = slice;
                                budget_charge = slice;
                                task->vruntime += (uint64_t)slice;
                                atomicAdd((unsigned long long*)&state->tenant_vruntime[tenant_bucket], (unsigned long long)slice);

                                if (remaining > 0) {
                                    task->args[PTX_TASK_META_ARG_WORK_REMAINING] =
                                        (void*)(uintptr_t)remaining;
                                    task->yield_count += 1;
                                    task_status = PTX_TASK_STATUS_YIELDED;
                                    task_finished = false;
                                }
                            }
                            break;

                        case PTX_TASK_OPCODE_ISA_RUN: // ISA_RUN (v0 decode/execute skeleton)
                            {
                                PTXISAProgramV0* program = (PTXISAProgramV0*)task->args[0];
                                PTXISAContextV0* context = (PTXISAContextV0*)task->args[1];
                                uint32_t step_cap = (uint32_t)(uintptr_t)task->args[2];
                                if (step_cap == 0) {
                                    step_cap = 64;
                                }
                                if (step_cap > tenant_budget) {
                                    step_cap = tenant_budget;
                                }
                                if (step_cap == 0) {
                                    task_status = PTX_TASK_STATUS_YIELDED;
                                    task_finished = false;
                                    budget_charge = 0;
                                    break;
                                }

                                uint32_t executed = 0;
                                bool halted = false;
                                bool trapped = false;
                                bool yielded = false;

                                // Inline mode: program words are embedded directly in task args.
                                // args[1] == NULL is the selector to avoid dereferencing invalid pointers.
                                if (context == NULL) {
                                    uint32_t word_count = (uint32_t)(uintptr_t)task->args[3];
                                    if (word_count == 0) {
                                        word_count = 1;
                                    }
                                    if (word_count > 3) {
                                        word_count = 3;
                                    }
                                    uint32_t pc = (uint32_t)(uintptr_t)task->args[PTX_TASK_META_ARG_WORK_REMAINING];
                                    uint32_t steps_total = (uint32_t)(uintptr_t)task->args[PTX_TASK_META_ARG_QUANTUM];
                                    uint32_t regs[32] = {0};
                                    uint32_t inline_mem[4] = {0, 0, 0, 0};
                                    uint64_t inline_mem_ptr = (uint64_t)(uintptr_t)&inline_mem[0];
                                    uint32_t inline_mem_size = (uint32_t)sizeof(inline_mem);

                                    while (executed < step_cap) {
                                        if (pc >= word_count) {
                                            trapped = true;
                                            break;
                                        }
                                        uint64_t insn = 0;
                                        if (pc == 0) {
                                            insn = (uint64_t)(uintptr_t)task->args[0];
                                        } else if (pc == 1) {
                                            insn = (uint64_t)(uintptr_t)task->args[PTX_TASK_META_ARG_DEPENDENCY];
                                        } else {
                                            insn = (uint64_t)(uintptr_t)task->args[PTX_TASK_META_ARG_CONTINUATION];
                                        }
                                        uint8_t opcode = ptx_isa_v0_decode_opcode(insn);
                                        uint8_t rd = ptx_isa_v0_decode_rd(insn);
                                        uint8_t rs0 = ptx_isa_v0_decode_rs0(insn);
                                        uint8_t rs1 = ptx_isa_v0_decode_rs1(insn);
                                        uint32_t imm = ptx_isa_v0_decode_imm32(insn);
                                        int32_t rel = ptx_isa_v0_decode_imm32_s(insn);
                                        uint32_t target_pc = 0;
                                        uint64_t mem_addr = 0;
                                        (void)ptx_isa_v0_decode_fmt(insn);
                                        (void)ptx_isa_v0_decode_rs2(insn);
                                        executed += 1;
                                        steps_total += 1;

                                        switch (opcode) {
                                            case PTX_ISA_V0_OP_NOP:
                                                pc += 1;
                                                break;
                                            case PTX_ISA_V0_OP_HALT:
                                                pc += 1;
                                                halted = true;
                                                break;
                                            case PTX_ISA_V0_OP_TRAP:
                                                trapped = true;
                                                break;
                                            case PTX_ISA_V0_OP_YIELD:
                                                pc += 1;
                                                yielded = true;
                                                break;
                                            case PTX_ISA_V0_OP_JMP:
                                                if (!ptx_isa_v0_calc_target_pc(pc, rel, word_count, &target_pc)) {
                                                    trapped = true;
                                                } else {
                                                    pc = target_pc;
                                                }
                                                break;
                                            case PTX_ISA_V0_OP_MOVI:
                                                regs[rd] = imm;
                                                pc += 1;
                                                break;
                                            case PTX_ISA_V0_OP_ADD:
                                                regs[rd] = regs[rs0] + regs[rs1];
                                                pc += 1;
                                                break;
                                            case PTX_ISA_V0_OP_SUB:
                                                regs[rd] = regs[rs0] - regs[rs1];
                                                pc += 1;
                                                break;
                                            case PTX_ISA_V0_OP_BR_EQ:
                                                if (regs[rs0] == regs[rs1]) {
                                                    if (!ptx_isa_v0_calc_target_pc(pc, rel, word_count, &target_pc)) {
                                                        trapped = true;
                                                    } else {
                                                        pc = target_pc;
                                                    }
                                                } else {
                                                    pc += 1;
                                                }
                                                break;
                                            case PTX_ISA_V0_OP_ASSERT_EQI:
                                                if (regs[rd] != imm) {
                                                    trapped = true;
                                                } else {
                                                    pc += 1;
                                                }
                                                break;
                                            case PTX_ISA_V0_OP_LD_U32:
                                                if (!ptx_isa_v0_resolve_u32_addr(
                                                        inline_mem_ptr,
                                                        inline_mem_size,
                                                        regs[rs0],
                                                        imm,
                                                        &mem_addr)) {
                                                    trapped = true;
                                                } else {
                                                    regs[rd] = *((uint32_t*)(uintptr_t)mem_addr);
                                                    pc += 1;
                                                }
                                                break;
                                            case PTX_ISA_V0_OP_ST_U32:
                                                if (!ptx_isa_v0_resolve_u32_addr(
                                                        inline_mem_ptr,
                                                        inline_mem_size,
                                                        regs[rs0],
                                                        imm,
                                                        &mem_addr)) {
                                                    trapped = true;
                                                } else {
                                                    *((uint32_t*)(uintptr_t)mem_addr) = regs[rs1];
                                                    pc += 1;
                                                }
                                                break;
                                            case PTX_ISA_V0_OP_LD_CONST:
                                                trapped = true;
                                                break;
                                            case PTX_ISA_V0_OP_SYSCALL:
                                                if (imm == PTX_ISA_V0_SYS_YIELD) {
                                                    pc += 1;
                                                    yielded = true;
                                                } else if (imm == PTX_ISA_V0_SYS_SIGNAL) {
                                                    atomicOr((unsigned long long*)&state->signal_mask, (unsigned long long)regs[rd]);
                                                    pc += 1;
                                                } else {
                                                    trapped = true;
                                                }
                                                break;
                                            default:
                                                trapped = true;
                                                break;
                                        }

                                        if (halted || trapped || yielded) {
                                            break;
                                        }
                                    }

                                    task->args[PTX_TASK_META_ARG_WORK_REMAINING] = (void*)(uintptr_t)pc;
                                    task->args[PTX_TASK_META_ARG_QUANTUM] = (void*)(uintptr_t)steps_total;
                                } else {
                                    if (!program) {
                                        context->state_flags = PTX_ISA_V0_STATE_TRAPPED;
                                        context->trap_code = PTX_ISA_V0_TRAP_NULL_PROGRAM;
                                        task_status = PTX_TASK_STATUS_RUNTIME_ERROR;
                                        budget_charge = 0;
                                        break;
                                    }
                                    if (program->abi_version != PTX_ISA_ABI_V0 || context->abi_version != PTX_ISA_ABI_V0) {
                                        context->state_flags = PTX_ISA_V0_STATE_TRAPPED;
                                        context->trap_code = PTX_ISA_V0_TRAP_INVALID_OPCODE;
                                        task_status = PTX_TASK_STATUS_RUNTIME_ERROR;
                                        budget_charge = 0;
                                        break;
                                    }

                                    uint64_t* code = (uint64_t*)(uintptr_t)program->code_ptr;
                                    if (!code || program->code_words == 0) {
                                        context->state_flags = PTX_ISA_V0_STATE_TRAPPED;
                                        context->trap_code = PTX_ISA_V0_TRAP_NULL_PROGRAM;
                                        task_status = PTX_TASK_STATUS_RUNTIME_ERROR;
                                        budget_charge = 0;
                                        break;
                                    }

                                    context->state_flags = PTX_ISA_V0_STATE_RUNNING;
                                    if (context->steps_total == 0) {
                                        context->pc = program->entry_pc;
                                    }
                                    context->steps_last_slice = 0;
                                    while (executed < step_cap) {
                                        if (context->pc >= program->code_words) {
                                            context->state_flags = PTX_ISA_V0_STATE_TRAPPED;
                                            context->trap_code = PTX_ISA_V0_TRAP_INVALID_PC;
                                            trapped = true;
                                            break;
                                        }

                                        uint64_t insn = code[context->pc];
                                        uint8_t opcode = ptx_isa_v0_decode_opcode(insn);
                                        uint8_t rd = ptx_isa_v0_decode_rd(insn);
                                        uint8_t rs0 = ptx_isa_v0_decode_rs0(insn);
                                        uint8_t rs1 = ptx_isa_v0_decode_rs1(insn);
                                        uint32_t imm = ptx_isa_v0_decode_imm32(insn);
                                        int32_t rel = ptx_isa_v0_decode_imm32_s(insn);
                                        uint32_t target_pc = 0;
                                        uint64_t mem_addr = 0;
                                        (void)ptx_isa_v0_decode_fmt(insn);
                                        (void)ptx_isa_v0_decode_rs2(insn);
                                        context->last_opcode = (uint32_t)opcode;
                                        context->steps_total += 1;
                                        executed += 1;

                                        switch (opcode) {
                                            case PTX_ISA_V0_OP_NOP:
                                                context->pc += 1;
                                                break;
                                            case PTX_ISA_V0_OP_HALT:
                                                context->pc += 1;
                                                context->state_flags = PTX_ISA_V0_STATE_HALTED;
                                                context->trap_code = PTX_ISA_V0_TRAP_NONE;
                                                halted = true;
                                                break;
                                            case PTX_ISA_V0_OP_TRAP:
                                                context->state_flags = PTX_ISA_V0_STATE_TRAPPED;
                                                context->trap_code = imm;
                                                if (context->trap_code == 0) {
                                                    context->trap_code = PTX_ISA_V0_TRAP_INVALID_OPCODE;
                                                }
                                                trapped = true;
                                                break;
                                            case PTX_ISA_V0_OP_YIELD:
                                                context->pc += 1;
                                                context->state_flags = PTX_ISA_V0_STATE_YIELDED;
                                                context->trap_code = PTX_ISA_V0_TRAP_NONE;
                                                yielded = true;
                                                break;
                                            case PTX_ISA_V0_OP_JMP:
                                                if (!ptx_isa_v0_calc_target_pc(context->pc, rel, program->code_words, &target_pc)) {
                                                    context->state_flags = PTX_ISA_V0_STATE_TRAPPED;
                                                    context->trap_code = PTX_ISA_V0_TRAP_INVALID_PC;
                                                    trapped = true;
                                                } else {
                                                    context->pc = target_pc;
                                                }
                                                break;
                                            case PTX_ISA_V0_OP_MOVI:
                                                context->regs[rd] = imm;
                                                context->pc += 1;
                                                break;
                                            case PTX_ISA_V0_OP_ADD:
                                                context->regs[rd] = context->regs[rs0] + context->regs[rs1];
                                                context->pc += 1;
                                                break;
                                            case PTX_ISA_V0_OP_SUB:
                                                context->regs[rd] = context->regs[rs0] - context->regs[rs1];
                                                context->pc += 1;
                                                break;
                                            case PTX_ISA_V0_OP_BR_EQ:
                                                if (context->regs[rs0] == context->regs[rs1]) {
                                                    if (!ptx_isa_v0_calc_target_pc(context->pc, rel, program->code_words, &target_pc)) {
                                                        context->state_flags = PTX_ISA_V0_STATE_TRAPPED;
                                                        context->trap_code = PTX_ISA_V0_TRAP_INVALID_PC;
                                                        trapped = true;
                                                    } else {
                                                        context->pc = target_pc;
                                                    }
                                                } else {
                                                    context->pc += 1;
                                                }
                                                break;
                                            case PTX_ISA_V0_OP_ASSERT_EQI:
                                                if (context->regs[rd] != imm) {
                                                    context->state_flags = PTX_ISA_V0_STATE_TRAPPED;
                                                    context->trap_code = PTX_ISA_V0_TRAP_ASSERT_FAILED;
                                                    trapped = true;
                                                } else {
                                                    context->pc += 1;
                                                }
                                                break;
                                            case PTX_ISA_V0_OP_LD_U32:
                                                if (!ptx_isa_v0_resolve_u32_addr(
                                                        context->mem_ptr,
                                                        context->mem_size,
                                                        context->regs[rs0],
                                                        imm,
                                                        &mem_addr)) {
                                                    context->state_flags = PTX_ISA_V0_STATE_TRAPPED;
                                                    context->trap_code = PTX_ISA_V0_TRAP_INVALID_MEM;
                                                    trapped = true;
                                                } else {
                                                    context->regs[rd] = *((uint32_t*)(uintptr_t)mem_addr);
                                                    context->pc += 1;
                                                }
                                                break;
                                            case PTX_ISA_V0_OP_ST_U32:
                                                if (!ptx_isa_v0_resolve_u32_addr(
                                                        context->mem_ptr,
                                                        context->mem_size,
                                                        context->regs[rs0],
                                                        imm,
                                                        &mem_addr)) {
                                                    context->state_flags = PTX_ISA_V0_STATE_TRAPPED;
                                                    context->trap_code = PTX_ISA_V0_TRAP_INVALID_MEM;
                                                    trapped = true;
                                                } else {
                                                    *((uint32_t*)(uintptr_t)mem_addr) = context->regs[rs1];
                                                    context->pc += 1;
                                                }
                                                break;
                                            case PTX_ISA_V0_OP_LD_CONST:
                                                if (program->const_ptr == 0 ||
                                                    ((uint64_t)imm + sizeof(uint32_t)) > (uint64_t)program->const_bytes) {
                                                    context->state_flags = PTX_ISA_V0_STATE_TRAPPED;
                                                    context->trap_code = PTX_ISA_V0_TRAP_INVALID_MEM;
                                                    trapped = true;
                                                } else {
                                                    uint64_t const_addr = program->const_ptr + (uint64_t)imm;
                                                    context->regs[rd] = *((uint32_t*)(uintptr_t)const_addr);
                                                    context->pc += 1;
                                                }
                                                break;
                                            case PTX_ISA_V0_OP_SYSCALL:
                                                if (imm == PTX_ISA_V0_SYS_YIELD) {
                                                    context->pc += 1;
                                                    context->state_flags = PTX_ISA_V0_STATE_YIELDED;
                                                    context->trap_code = PTX_ISA_V0_TRAP_NONE;
                                                    yielded = true;
                                                } else if (imm == PTX_ISA_V0_SYS_SIGNAL) {
                                                    atomicOr((unsigned long long*)&state->signal_mask, (unsigned long long)context->regs[rd]);
                                                    context->pc += 1;
                                                } else {
                                                    context->state_flags = PTX_ISA_V0_STATE_TRAPPED;
                                                    context->trap_code = PTX_ISA_V0_TRAP_INVALID_SYSCALL;
                                                    trapped = true;
                                                }
                                                break;
                                            default:
                                                context->state_flags = PTX_ISA_V0_STATE_TRAPPED;
                                                context->trap_code = PTX_ISA_V0_TRAP_INVALID_OPCODE;
                                                trapped = true;
                                                break;
                                        }

                                        if (halted || trapped || yielded) {
                                            break;
                                        }
                                    }
                                    context->steps_last_slice = executed;
                                }

                                consumed_ticks = executed;
                                budget_charge = executed;
                                task->vruntime += (uint64_t)executed;
                                atomicAdd((unsigned long long*)&state->tenant_vruntime[tenant_bucket], (unsigned long long)executed);

                                if (trapped) {
                                    task_status = PTX_TASK_STATUS_RUNTIME_ERROR;
                                } else if (halted) {
                                    task_status = PTX_TASK_STATUS_OK;
                                } else {
                                    task_status = PTX_TASK_STATUS_YIELDED;
                                    task_finished = false;
                                    task->yield_count += 1;
                                }
                            }
                            break;
                            
                        default:
                            printf("[GPU-OS] Unknown Opcode: %d\n", task->opcode);
                            task_status = PTX_TASK_STATUS_UNSUPPORTED_OPCODE;
                            break;
                    }

                    if (task_finished) {
                        task->completed_at = clock64();
                        task->completed = true;
                        task->active = false;
                        state->total_ops++;
                        if (state->active_tasks > 0) {
                            state->active_tasks--;
                        }
                        ptx_os_push_completion(state, task, task_status);
                        ptx_os_enqueue_continuation(state, task);
                    } else {
                        // Cooperative yield: task remains active and re-enters scheduling scan.
                        if (consumed_ticks > 0) {
                            state->total_ops++;
                        }
                    }
                    ptx_os_charge_tenant_budget(state, tenant_bucket, sched_epoch, budget_charge);
                    
                    // Compact queue tail past all non-active entries.
                    while (state->queue.tail != state->queue.head) {
                        PTXOSTask* tail_task = &state->queue.tasks[state->queue.tail];
                        if (tail_task->active) {
                            break;
                        }
                        state->queue.tail = (state->queue.tail + 1) % PTX_MAX_QUEUE_SIZE;
                    }
                }
            }
            
            // --- Feature 4: Async Signal/Interrupt Processing ---
            if (state->signal_mask != 0) {
                uint64_t signals = atomicExch((unsigned long long*)&state->signal_mask, 0);
                if (signals & 0x1) {
                    printf("[PTX-OS] Heartbeat Pulse Received from Host/App\n");
                }
                if (signals & 0x2) {
                    printf("[PTX-OS] EMERGENCY FLUSH REQUESTED. Clearing non-critical pools.\n");
                }
                state->interrupt_cnt++;
            }
        }

        // Feature: OS Health Debugger
        if (tid == 0 && iterations % 10000000 == 0) {
            printf("[GPU-OS-DEBUG] Head: %d | Tail: %d | Total Ops: %lld\n", 
                   state->queue.head, state->queue.tail, state->total_ops);
        }

        // Small delay to prevent saturation of the command bridge
        for (int i = 0; i < 500; i++) {
             __threadfence_system(); // Ensure visibility across GPU and CPU
        }
        iterations++;
    }
    
    if (tid == 0) {
        state->kernel_running = false;
        printf("[GPU-OS] Kernel Shutdown. Returning control to hardware.\n");
    }
}

// ============================================================================
// Internal Test Kernels for CDP
// ============================================================================

__global__ void k_test_cdp(void** args) {
    int tid = threadIdx.x;
    if (tid == 0) {
        float* data = (float*)args[0];
        float value = *(float*)&args[1];
        printf("[PTX-APP] CDP Kernel Executing! Writing %f to %p\n", value, data);
        if (data) *data = value;
    }
}

// Device-side pointer to the test kernel (for CDP retrieval)
__device__ void (*d_test_kernel_ptr)(void**) = k_test_cdp;

extern "C" void* ptx_get_test_kernel_ptr_internal() {
    void* h_ptr;
    cudaMemcpyFromSymbol(&h_ptr, d_test_kernel_ptr, sizeof(void*));
    return h_ptr;
}

extern "C" void ptx_os_launch_persistent_kernel(GPUHotRuntime* runtime, PTXSystemState* d_state) {
    int blockSize = 256;
    int osBlocks = 1;
    
    printf("[PTX-OS] Launching Persistent Kernel...\n");
    printf("[PTX-OS] Configuration: %d Blocks x %d Threads/Block\n", osBlocks, blockSize);
    
    // Launch the Persistent Kernel
    ptx_os_kernel<<<osBlocks, blockSize, 0, 0>>>(d_state);
    cudaError_t err = cudaGetLastError();
    
    if (err != cudaSuccess) {
        printf("[PTX-OS] [ERROR] Kernel Launch failed: %s\n", cudaGetErrorString(err));
        return;
    }
    printf("[PTX-OS] OS Kernel is now resident in VRAM.\n");
}
