// ============================================================================
// Task Submission API
// ============================================================================

static int ptx_os_enqueue_task_v1_locked(PTXSystemState* state, const PTXTaskDescV1* desc) {
    PTXTaskQueue* queue = &state->queue;

    uint32_t next_head = (queue->head + 1) % PTX_MAX_QUEUE_SIZE;
    if (next_head == queue->tail) {
        return -1;
    }

    uint32_t task_id = __sync_add_and_fetch(&state->next_task_id, 1);
    PTXOSTask* task = &queue->tasks[queue->head];
    task->task_id = task_id;
    task->opcode = desc->opcode;
    task->priority = desc->priority;
    task->tenant_id = desc->tenant_id;
    task->flags = desc->flags;
    task->arg_count = desc->arg_count > PTX_MAX_TASK_ARGS ? PTX_MAX_TASK_ARGS : desc->arg_count;
    task->yield_count = 0;
    task->active = true;
    task->completed = false;
    task->submitted_at = GetTickCount64();
    task->started_at = 0;
    task->completed_at = 0;
    task->vruntime = state->tenant_vruntime[desc->tenant_id % PTX_MAX_QUEUE_SIZE];

    if (desc->args) {
        memcpy(task->args, desc->args, sizeof(void*) * PTX_MAX_TASK_ARGS);
    } else {
        memset(task->args, 0, sizeof(void*) * PTX_MAX_TASK_ARGS);
    }

    __sync_synchronize();
    queue->head = next_head;
    __sync_add_and_fetch(&state->active_tasks, 1);

    return (int)task_id;
}

int ptx_os_submit_task_v1(GPUHotRuntime* runtime, const PTXTaskDescV1* desc) {
    if (!runtime || !runtime->system_state || !desc) return -1;
    if (desc->abi_version != PTX_TASK_ABI_V1) return -1;
    if (desc->arg_count > PTX_MAX_TASK_ARGS) return -1;

    PTXSystemState* state = runtime->system_state;
    PTXTaskQueue* queue = &state->queue;

    while (__sync_lock_test_and_set(&queue->lock, 1)) {
        // spin
    }

    int task_id = ptx_os_enqueue_task_v1_locked(state, desc);
    __sync_lock_release(&queue->lock);
    return task_id;
}

int ptx_os_submit_task(GPUHotRuntime* runtime, uint32_t opcode, uint32_t priority, void* args[PTX_MAX_TASK_ARGS]) {
    PTXTaskDescV1 desc;
    memset(&desc, 0, sizeof(desc));
    desc.abi_version = PTX_TASK_ABI_V1;
    desc.opcode = opcode;
    desc.priority = priority;
    desc.flags = 0;
    desc.tenant_id = 0;
    desc.arg_count = PTX_MAX_TASK_ARGS;
    if (args) {
        memcpy(desc.args, args, sizeof(void*) * PTX_MAX_TASK_ARGS);
    }
    return ptx_os_submit_task_v1(runtime, &desc);
}

int ptx_os_poll_completion_v1(GPUHotRuntime* runtime, PTXTaskResultV1* out_result) {
    if (!runtime || !runtime->system_state || !out_result) return -1;

    PTXTaskResultQueue* cq = &runtime->system_state->completion_queue;
    uint32_t tail = cq->tail;
    if (tail == cq->head) {
        return 0;  // empty
    }

    *out_result = cq->results[tail];
    __sync_synchronize();
    cq->tail = (tail + 1) % PTX_MAX_COMPLETION_QUEUE_SIZE;
    return 1;
}
