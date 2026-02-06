/*
 * PTX-OS Stable Runtime ABI implementation.
 */

#include "ptx_stable_runtime.h"

#include "gpu/gpu_hot_runtime.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Weak hook linkage: libptx_os is loadable without libptx_hook present.
extern "C" void ptx_context_hook_capture(int, void*) __attribute__((weak));
extern "C" void ptx_context_hook_set_ptx_active(bool) __attribute__((weak));

struct PTXStableRuntime {
    GPUHotRuntime* runtime;
    int device_id;
    uint32_t magic;
};

static pthread_mutex_t g_stable_lock = PTHREAD_MUTEX_INITIALIZER;
static PTXStableRuntime g_singleton = {0};
static uint32_t g_refcount = 0;
static const uint32_t PTX_STABLE_MAGIC = 0x50545853u; // "PTXS"

static bool stable_is_live(const PTXStableRuntime* rt) {
    return rt && rt->magic == PTX_STABLE_MAGIC && rt->runtime != NULL;
}

static void stable_export_runtime_ptr(GPUHotRuntime* runtime) {
    char buf[32];
    snprintf(buf, sizeof(buf), "%llx", (unsigned long long)(uintptr_t)runtime);
    setenv("PTX_RUNTIME_PTR", buf, 1);
}

static void stable_clear_runtime_ptr(void) {
    unsetenv("PTX_RUNTIME_PTR");
}

const char* ptx_stable_runtime_version(void) {
    return "ptx-stable-runtime/1";
}

uint32_t ptx_stable_runtime_abi_version(void) {
    return PTX_STABLE_ABI_VERSION;
}

const char* ptx_stable_strerror(PTXStableStatus status) {
    switch (status) {
        case PTX_STABLE_OK: return "ok";
        case PTX_STABLE_ERR_INVALID_ARG: return "invalid argument";
        case PTX_STABLE_ERR_ABI_MISMATCH: return "abi mismatch";
        case PTX_STABLE_ERR_INIT_FAILED: return "runtime init failed";
        case PTX_STABLE_ERR_NOT_INITIALIZED: return "runtime not initialized";
        case PTX_STABLE_ERR_NOT_OWNED: return "pointer not owned by PTX runtime";
        case PTX_STABLE_ERR_ALLOC_FAILED: return "allocation failed";
        case PTX_STABLE_ERR_INTERNAL: return "internal error";
        default: return "unknown status";
    }
}

static PTXStableStatus stable_fill_runtime(GPUHotRuntime* runtime, int device_id, PTXStableRuntime** out_runtime) {
    if (!runtime || !out_runtime) return PTX_STABLE_ERR_INVALID_ARG;

    g_singleton.runtime = runtime;
    g_singleton.device_id = device_id;
    g_singleton.magic = PTX_STABLE_MAGIC;
    *out_runtime = &g_singleton;
    return PTX_STABLE_OK;
}

PTXStableStatus ptx_stable_init(const PTXStableConfig* config, PTXStableRuntime** out_runtime) {
    if (!out_runtime) return PTX_STABLE_ERR_INVALID_ARG;

    pthread_mutex_lock(&g_stable_lock);

    if (stable_is_live(&g_singleton)) {
        g_refcount++;
        *out_runtime = &g_singleton;
        pthread_mutex_unlock(&g_stable_lock);
        return PTX_STABLE_OK;
    }

    GPUHotConfig hot_cfg = gpu_hot_default_config();
    int device_id = 0;

    if (config) {
        if (config->struct_size != sizeof(PTXStableConfig) ||
            config->abi_version != PTX_STABLE_ABI_VERSION) {
            pthread_mutex_unlock(&g_stable_lock);
            return PTX_STABLE_ERR_ABI_MISMATCH;
        }
        if (config->device_id != PTX_STABLE_INVALID_DEVICE) {
            device_id = config->device_id;
        }
        if (config->pool_fraction > 0.0f) {
            hot_cfg.pool_fraction = config->pool_fraction;
            hot_cfg.fixed_pool_size = 0;
        }
        if (config->fixed_pool_size > 0) {
            hot_cfg.fixed_pool_size = (size_t)config->fixed_pool_size;
            hot_cfg.pool_fraction = 0.0f;
        }
        if (config->reserve_vram > 0) {
            hot_cfg.reserve_vram = (size_t)config->reserve_vram;
        }
        if (config->max_streams > 0) {
            hot_cfg.max_streams = config->max_streams;
        }
        hot_cfg.quiet_init = config->quiet_init != 0;
        hot_cfg.enable_leak_detection = config->enable_leak_detection != 0;
        hot_cfg.enable_pool_health = config->enable_pool_health != 0;
    }

    GPUHotRuntime* runtime = gpu_hot_init_with_config(device_id, NULL, &hot_cfg);
    if (!runtime) {
        pthread_mutex_unlock(&g_stable_lock);
        return PTX_STABLE_ERR_INIT_FAILED;
    }

    stable_export_runtime_ptr(runtime);

    void* ctx = gpu_hot_get_context(runtime);
    if (ctx && ptx_context_hook_capture) {
        ptx_context_hook_capture(device_id, ctx);
    }
    if (ptx_context_hook_set_ptx_active) {
        ptx_context_hook_set_ptx_active(true);
    }

    g_refcount = 1;
    PTXStableStatus st = stable_fill_runtime(runtime, device_id, out_runtime);
    pthread_mutex_unlock(&g_stable_lock);
    return st;
}

PTXStableStatus ptx_stable_retain(PTXStableRuntime** out_runtime) {
    return ptx_stable_init(NULL, out_runtime);
}

PTXStableStatus ptx_stable_get(PTXStableRuntime** out_runtime) {
    if (!out_runtime) return PTX_STABLE_ERR_INVALID_ARG;

    pthread_mutex_lock(&g_stable_lock);
    if (!stable_is_live(&g_singleton)) {
        pthread_mutex_unlock(&g_stable_lock);
        return PTX_STABLE_ERR_NOT_INITIALIZED;
    }
    *out_runtime = &g_singleton;
    pthread_mutex_unlock(&g_stable_lock);
    return PTX_STABLE_OK;
}

PTXStableStatus ptx_stable_release(PTXStableRuntime* runtime) {
    pthread_mutex_lock(&g_stable_lock);

    if (!stable_is_live(&g_singleton) || runtime != &g_singleton) {
        pthread_mutex_unlock(&g_stable_lock);
        return PTX_STABLE_ERR_NOT_INITIALIZED;
    }

    if (g_refcount > 1) {
        g_refcount--;
        pthread_mutex_unlock(&g_stable_lock);
        return PTX_STABLE_OK;
    }

    g_refcount = 0;

    if (ptx_context_hook_set_ptx_active) {
        ptx_context_hook_set_ptx_active(false);
    }

    gpu_hot_shutdown(g_singleton.runtime);
    stable_clear_runtime_ptr();

    memset(&g_singleton, 0, sizeof(g_singleton));
    pthread_mutex_unlock(&g_stable_lock);
    return PTX_STABLE_OK;
}

PTXStableStatus ptx_stable_alloc(PTXStableRuntime* runtime, size_t size, void** out_ptr) {
    if (!stable_is_live(runtime) || !out_ptr || size == 0) return PTX_STABLE_ERR_INVALID_ARG;
    void* p = gpu_hot_alloc(runtime->runtime, size);
    if (!p) return PTX_STABLE_ERR_ALLOC_FAILED;
    *out_ptr = p;
    return PTX_STABLE_OK;
}

PTXStableStatus ptx_stable_free(PTXStableRuntime* runtime, void* ptr) {
    if (!stable_is_live(runtime)) return PTX_STABLE_ERR_INVALID_ARG;
    if (!ptr) return PTX_STABLE_OK;
    if (!gpu_hot_owns_ptr(runtime->runtime, ptr)) return PTX_STABLE_ERR_NOT_OWNED;
    gpu_hot_free(runtime->runtime, ptr);
    return PTX_STABLE_OK;
}

PTXStableStatus ptx_stable_owns_ptr(PTXStableRuntime* runtime, const void* ptr, bool* out_owned) {
    if (!stable_is_live(runtime) || !out_owned) return PTX_STABLE_ERR_INVALID_ARG;
    *out_owned = ptr ? gpu_hot_owns_ptr(runtime->runtime, (void*)ptr) : false;
    return PTX_STABLE_OK;
}

PTXStableStatus ptx_stable_get_context(PTXStableRuntime* runtime, void** out_cu_context) {
    if (!stable_is_live(runtime) || !out_cu_context) return PTX_STABLE_ERR_INVALID_ARG;
    *out_cu_context = gpu_hot_get_context(runtime->runtime);
    return *out_cu_context ? PTX_STABLE_OK : PTX_STABLE_ERR_INTERNAL;
}

PTXStableStatus ptx_stable_get_stats(PTXStableRuntime* runtime, PTXStableStats* out_stats) {
    if (!stable_is_live(runtime) || !out_stats) return PTX_STABLE_ERR_INVALID_ARG;

    GPUHotStats hot_stats;
    TLSFPoolStats tlsf_stats;
    memset(&hot_stats, 0, sizeof(hot_stats));
    memset(&tlsf_stats, 0, sizeof(tlsf_stats));
    gpu_hot_get_stats(runtime->runtime, &hot_stats);
    gpu_hot_get_tlsf_stats(runtime->runtime, &tlsf_stats);

    memset(out_stats, 0, sizeof(*out_stats));
    out_stats->vram_allocated = hot_stats.vram_allocated;
    out_stats->vram_used = hot_stats.vram_used;
    out_stats->vram_free = hot_stats.vram_free;
    out_stats->gpu_utilization = hot_stats.gpu_utilization;
    out_stats->pool_total = tlsf_stats.total_pool_size;
    out_stats->pool_used = tlsf_stats.allocated_bytes;
    out_stats->pool_free = tlsf_stats.free_bytes;
    out_stats->pool_peak = tlsf_stats.peak_allocated;
    out_stats->pool_fallbacks = tlsf_stats.fallback_count;
    out_stats->pool_fragmentation = tlsf_stats.fragmentation_ratio;
    out_stats->total_ops = hot_stats.total_ops;
    out_stats->active_streams = (uint32_t)hot_stats.active_streams;
    out_stats->watchdog_tripped = hot_stats.watchdog_tripped ? 1 : 0;

    pthread_mutex_lock(&g_stable_lock);
    out_stats->refcount = g_refcount;
    pthread_mutex_unlock(&g_stable_lock);
    return PTX_STABLE_OK;
}

PTXStableStatus ptx_stable_get_hot_runtime(PTXStableRuntime* runtime, void** out_gpu_hot_runtime) {
    if (!stable_is_live(runtime) || !out_gpu_hot_runtime) return PTX_STABLE_ERR_INVALID_ARG;
    *out_gpu_hot_runtime = (void*)runtime->runtime;
    return PTX_STABLE_OK;
}
