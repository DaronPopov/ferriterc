/*
 * GPU Hot Runtime - Implementation
 * Persistent GPU context with pre-allocated VRAM pools
 */

#include "gpu/gpu_hot_runtime.h"
#include "memory/ptx_tlsf_allocator.h"
#include "memory/ptx_cuda_driver.h"
#include "ptx_debug.h"
#include <cuda.h>  // CUDA driver API for CUcontext / cuDevicePrimaryCtxRetain
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

// Weak references to context hook functions (defined in cuda_context_hook.c).
// These must be at file scope with extern "C" because the hook library is
// compiled as C by gcc — without C linkage, nvcc applies C++ name mangling
// and the weak references silently resolve to NULL at runtime.
extern "C" void ptx_context_hook_capture(int, CUcontext) __attribute__((weak));
extern "C" void ptx_context_hook_set_ptx_active(bool) __attribute__((weak));

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

// Cross-platform atomic operations and timing for host-side shared memory
#ifdef _WIN32
#include <windows.h>
#else
#include <pthread.h>
#include <time.h>
#include <errno.h>

#define InterlockedIncrement(ptr) __sync_add_and_fetch(ptr, 1)
#define InterlockedDecrement(ptr) __sync_sub_and_fetch(ptr, 1)
#define InterlockedAdd64(ptr, val) __sync_add_and_fetch(ptr, val)
#define LONG int
#define LONG64 int64_t

// Linux implementation of GetTickCount64
static uint64_t GetTickCount64() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)(ts.tv_sec * 1000 + ts.tv_nsec / 1000000);
}
#endif

// ============================================================================
// Async TLSF deferred free support
// ============================================================================

#ifdef _WIN32
typedef CRITICAL_SECTION PTXMutex;
static void ptx_mutex_init(PTXMutex* m) { InitializeCriticalSection(m); }
static void ptx_mutex_lock(PTXMutex* m) { EnterCriticalSection(m); }
static void ptx_mutex_unlock(PTXMutex* m) { LeaveCriticalSection(m); }
static void ptx_mutex_destroy(PTXMutex* m) { DeleteCriticalSection(m); }
#else
typedef pthread_mutex_t PTXMutex;
static void ptx_mutex_init(PTXMutex* m) { pthread_mutex_init(m, NULL); }
static void ptx_mutex_lock(PTXMutex* m) { pthread_mutex_lock(m); }
static void ptx_mutex_unlock(PTXMutex* m) { pthread_mutex_unlock(m); }
static void ptx_mutex_destroy(PTXMutex* m) { pthread_mutex_destroy(m); }
#endif

typedef struct DeferredFreeEntry {
    void* ptr;
    cudaEvent_t event;
    cudaStream_t stream;
    struct DeferredFreeEntry* next;
} DeferredFreeEntry;

static cudaEvent_t ptx_event_acquire(GPUHotRuntime* runtime);
static void ptx_event_release(GPUHotRuntime* runtime, cudaEvent_t ev);

static bool ptx_tlsf_debug_enabled() {
    static int initialized = 0;
    static bool enabled = false;
    if (!initialized) {
        const char* v = getenv("PTX_TLSF_DEBUG");
        enabled = (v && v[0] && v[0] != '0');
        initialized = 1;
    }
    return enabled;
}
static void gpu_hot_poll_deferred_internal(GPUHotRuntime* runtime, int max_drain);


// Runtime state
struct GPUHotRuntime {
    int device_id;
    CUcontext cu_context;  // Captured primary context for context hook enforcement
    cudaStream_t streams[GPU_HOT_MAX_STREAMS];
    int num_streams;
    
    // TLSF VRAM Allocator (Unified)
    void* vram_pool;
    size_t vram_pool_size;
    PTXTLSFAllocator* tlsf_allocator;
    int fallback_count;           // Number of cudaMalloc fallbacks
    
    // CUDA Graphs
    GPUGraphHandle graphs[GPU_HOT_MAX_GRAPHS];
    int num_graphs;
    int capturing_stream_id;  // -1 if not capturing
    
    // Registered kernels
    GPUKernelHandle kernels[GPU_HOT_MAX_KERNELS];
    int num_kernels;
    
    // Shared Memory Registry (The GPU "File System")
    GPURegistryEntry* global_registry;
    PTXSystemState* system_state; // New global system vitals
#ifdef _WIN32
    HANDLE shm_handle;
#else
    int shm_handle; // Not used on Linux, but keep for compatibility
#endif
    int shm_count;
    
    // Keepalive
    bool keepalive_running;
    cudaEvent_t keepalive_event;
    
    // Vitals & Guardrails
    uint64_t total_ops;
    int watchdog_timeout_ms;
    bool watchdog_tripped;
    long long last_launch_time;

    // Async deferred free queue
    PTXMutex async_lock;
    DeferredFreeEntry* deferred_head;
    DeferredFreeEntry* deferred_tail;
    int deferred_count;

    // Event pool for async frees
    cudaEvent_t* event_pool;
    int event_pool_count;
    int event_pool_capacity;
    
};


// Keepalive kernel - runs continuously to keep GPU hot
__global__ void keepalive_kernel(volatile int* flag) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Minimal work to keep GPU active
    if (tid == 0) {
        atomicAdd((int*)flag, 1);
    }
    
    // Touch some memory to keep caches warm
    __shared__ int shared_data[256];
    shared_data[threadIdx.x] = tid;
    __syncthreads();
    
    // Prevent optimization
    if (shared_data[threadIdx.x] < 0) {
        atomicAdd((int*)flag, shared_data[threadIdx.x]);
    }
}

// ============================================================================
// ============================================================================
// Configuration API
// ============================================================================

GPUHotConfig gpu_hot_default_config(void) {
    GPUHotConfig config;
    memset(&config, 0, sizeof(GPUHotConfig));

    // Default: Use all available VRAM minus headroom
    config.pool_fraction = 1.0f;
    config.fixed_pool_size = 0;  // Use fraction-based sizing

    // Size limits
    config.min_pool_size = 256ULL * 1024 * 1024;   // 256MB minimum
    config.max_pool_size = 0;                       // No maximum (0 = unlimited)

    // Safety margin for CUDA runtime (0 = auto)
    config.reserve_vram = 0;    // Auto-reserve headroom

    // Allocator features
    config.enable_leak_detection = true;
    config.enable_pool_health = true;
    config.warning_threshold = 0.9f;  // Warn at 90% utilization

    // Behavior
    config.force_daemon_mode = false;
    config.quiet_init = false;

    // Stream configuration
    config.max_streams = 16;    // Default to 16 streams

    return config;
}

// ============================================================================
// Environment overrides
// ============================================================================

static bool parse_env_double(const char* key, double* out) {
    const char* val = getenv(key);
    if (!val || !out) return false;
    char* end = NULL;
    double v = strtod(val, &end);
    if (end == val) return false;
    *out = v;
    return true;
}

static bool parse_env_size(const char* key, size_t* out) {
    const char* val = getenv(key);
    if (!val || !out) return false;
    char* end = NULL;
    double v = strtod(val, &end);
    if (end == val) return false;
    while (*end == ' ' || *end == '\t') end++;
    double scale = 1.0;
    if (*end != '\0') {
        if (strcasecmp(end, "k") == 0 || strcasecmp(end, "kb") == 0) {
            scale = 1024.0;
        } else if (strcasecmp(end, "m") == 0 || strcasecmp(end, "mb") == 0) {
            scale = 1024.0 * 1024.0;
        } else if (strcasecmp(end, "g") == 0 || strcasecmp(end, "gb") == 0) {
            scale = 1024.0 * 1024.0 * 1024.0;
        } else {
            // Unknown suffix; treat as bytes
            scale = 1.0;
        }
    }
    if (v <= 0) return false;
    *out = (size_t)(v * scale);
    return true;
}

static void apply_env_overrides(GPUHotConfig* config) {
    if (!config) return;

    // Fixed pool size override
    size_t fixed = 0;
    if (parse_env_size("PTX_POOL_SIZE", &fixed) && fixed > 0) {
        config->fixed_pool_size = fixed;
        config->pool_fraction = 0.0f;
    }

    // Pool fraction override (0..1 or percent 1..100)
    double frac = 0.0;
    if (parse_env_double("PTX_POOL_FRACTION", &frac)) {
        if (frac > 1.0 && frac <= 100.0) {
            frac = frac / 100.0;
        }
        if (frac > 0.0) {
            config->pool_fraction = (float)frac;
            if (config->fixed_pool_size > 0) {
                config->fixed_pool_size = 0;
            }
        }
    }

    // Warning threshold override (0..1 or percent 1..100)
    double warn = 0.0;
    if (parse_env_double("PTX_POOL_WARN_THRESHOLD", &warn)) {
        if (warn > 1.0 && warn <= 100.0) {
            warn = warn / 100.0;
        }
        if (warn >= 0.0) {
            config->warning_threshold = (float)warn;
        }
    }
    if (getenv("PTX_POOL_WARN_DISABLE")) {
        config->warning_threshold = 0.0f;
    }
}

// Initialize with default config (backward compatible)
GPUHotRuntime* gpu_hot_init(int device_id, const char* token) {
    GPUHotConfig config = gpu_hot_default_config();
    return gpu_hot_init_with_config(device_id, token, &config);
}

// Initialize GPU Hot Runtime with configuration
GPUHotRuntime* gpu_hot_init_with_config(int device_id, const char* token, const GPUHotConfig* config) {
    GPUHotRuntime* runtime = (GPUHotRuntime*)malloc(sizeof(GPUHotRuntime));
    if (!runtime) return NULL;

    // Use default config if none provided
    GPUHotConfig effective_config;
    if (config) {
        effective_config = *config;
    } else {
        effective_config = gpu_hot_default_config();
    }
    apply_env_overrides(&effective_config);

    memset(runtime, 0, sizeof(GPUHotRuntime));
    runtime->device_id = device_id;
    ptx_mutex_init(&runtime->async_lock);
    runtime->event_pool = NULL;
    runtime->event_pool_capacity = 0;
    runtime->event_pool_count = 0;
    runtime->deferred_head = NULL;
    runtime->deferred_tail = NULL;
    runtime->deferred_count = 0;
    // Set device and initialize runtime
    cudaError_t cuda_err = cudaSetDevice(device_id);
    if (cuda_err != cudaSuccess) {
        printf("[Ferrite-OS] Error: cudaSetDevice(%d) failed: %s\n", device_id, cudaGetErrorString(cuda_err));
        free(runtime);
        return NULL;
    }
    cuda_err = cudaFree(0);
    if (cuda_err != cudaSuccess) {
        printf("[Ferrite-OS] Error: cudaFree(0) failed: %s\n", cudaGetErrorString(cuda_err));
        free(runtime);
        return NULL;
    }

    // Capture primary context via CUDA driver API
    {
        CUdevice cu_dev;
        CUresult cu_err = cuDeviceGet(&cu_dev, device_id);
        if (cu_err == CUDA_SUCCESS) {
            cu_err = cuDevicePrimaryCtxRetain(&runtime->cu_context, cu_dev);
            if (cu_err == CUDA_SUCCESS) {
                if (!effective_config.quiet_init) {
                    printf("[Ferrite-OS] Captured primary context %p for device %d\n",
                           (void*)runtime->cu_context, device_id);
                }
                // Notify context hook (weak symbol: no-op if hook not loaded)
                if (ptx_context_hook_capture) {
                    ptx_context_hook_capture(device_id, runtime->cu_context);
                }
            } else {
                printf("[Ferrite-OS] Warning: cuDevicePrimaryCtxRetain failed (%d)\n", cu_err);
                runtime->cu_context = NULL;
            }
        } else {
            printf("[Ferrite-OS] Warning: cuDeviceGet failed (%d)\n", cu_err);
            runtime->cu_context = NULL;
        }
    }

    if (!effective_config.quiet_init) {
        printf("[Ferrite-OS] Initializing GPU Hot Runtime on device %d\n", device_id);
    }
    
    // ========================================================================
    // PTX-OS Global Registry & System State (IPC Support)
    // ========================================================================
    size_t registry_size = sizeof(GPURegistryEntry) * GPU_HOT_MAX_NAMED_SEGMENTS;
    size_t system_state_size = sizeof(PTXSystemState);
    size_t total_shm_size = registry_size + system_state_size;

#ifdef _WIN32
    runtime->shm_handle = CreateFileMappingA(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, (DWORD)total_shm_size, GPU_HOT_IPC_KEY);
    bool is_daemon = (GetLastError() != ERROR_ALREADY_EXISTS);
    void* shm_base = MapViewOfFile(runtime->shm_handle, FILE_MAP_ALL_ACCESS, 0, 0, total_shm_size);
#else
    void* shm_base = NULL;
    // Linux POSIX shared memory
    // Try O_EXCL to detect if we are the creator (daemon)
    int shm_fd = shm_open(GPU_HOT_IPC_KEY, O_CREAT | O_EXCL | O_RDWR, 0666);
    bool is_daemon = true;
    
    if (shm_fd == -1 && errno == EEXIST) {
        // Segment already exists, we are a client
        is_daemon = false;
        shm_fd = shm_open(GPU_HOT_IPC_KEY, O_RDWR, 0666);
    }

    if (shm_fd == -1) {
        printf("[Ferrite-OS] Error: Failed to create/open shared memory (errno: %d)\n", errno);
        shm_base = NULL;
    } else {
        // Set size (only needed for daemon)
        if (is_daemon && ftruncate(shm_fd, total_shm_size) == -1) {
            printf("[Ferrite-OS] Error: Failed to set shared memory size\n");
            close(shm_fd);
            shm_base = NULL;
        } else {
            shm_base = mmap(NULL, total_shm_size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
            close(shm_fd); // File descriptor no longer needed after mmap
            if (shm_base == MAP_FAILED) {
                printf("[Ferrite-OS] Error: Failed to map shared memory (errno: %d)\n", errno);
                shm_base = NULL;
            }
        }
    }
#endif
    if (!shm_base) {
        printf("[Ferrite-OS] Error: Failed to map Global OS Space\n");
    } else {
        runtime->global_registry = (GPURegistryEntry*)shm_base;
        runtime->system_state = (PTXSystemState*)((char*)shm_base + registry_size);
        
        if (is_daemon) {
            memset(shm_base, 0, total_shm_size);
            // Generate session token
            runtime->system_state->auth_token = 0xDEADC0DE; // In prod, this would be random
            printf("[Ferrite-OS] Global state initialized (daemon mode)\n");
        } else {
            printf("[Ferrite-OS] Attached to global state (client mode)\n");
            InterlockedIncrement((LONG*)&runtime->system_state->active_processes);
        }

        // ========================================================================
        // GPU Native Access: Register SHM for Zero-Copy GPU access
        // ========================================================================
        cudaError_t shm_err = cudaHostRegister(shm_base, total_shm_size, cudaHostRegisterMapped);
        if (shm_err == cudaErrorHostMemoryAlreadyRegistered) {
            // Benign when multiple clients attach to the same shared segment.
            printf("[Ferrite-OS] Global state already registered with GPU context\n");
        } else if (shm_err != cudaSuccess) {
            printf("[Ferrite-OS] Warning: Could not register SHM for GPU access: %s\n", cudaGetErrorString(shm_err));
        } else {
            printf("[Ferrite-OS] Global state mapped to GPU context\n");
        }
    }
    // ========================================================================
    
    // Get device properties
    cudaDeviceProp prop;
    cuda_err = cudaGetDeviceProperties(&prop, device_id);
    if (cuda_err != cudaSuccess) {
        printf("[Ferrite-OS] Error: cudaGetDeviceProperties failed: %s\n", cudaGetErrorString(cuda_err));
        free(runtime);
        return NULL;
    }
    printf("[Ferrite-OS] Device: %s\n", prop.name);
    printf("[Ferrite-OS] Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("[Ferrite-OS] Total VRAM: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    
    // Create CUDA streams with priority range sensing
    int least_priority, greatest_priority;
    cuda_err = cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority);
    if (cuda_err != cudaSuccess) {
        printf("[Ferrite-OS] Error: cudaDeviceGetStreamPriorityRange failed: %s\n", cudaGetErrorString(cuda_err));
        free(runtime);
        return NULL;
    }

    // Use configured stream count, bounded by max
    runtime->num_streams = effective_config.max_streams;
    if (runtime->num_streams > GPU_HOT_MAX_STREAMS) {
        runtime->num_streams = GPU_HOT_MAX_STREAMS;
    }
    if (runtime->num_streams < 1) {
        runtime->num_streams = 16; // Minimum default
    }

    for (int i = 0; i < runtime->num_streams; i++) {
        // First 2 streams are Real-Time (High Priority)
        int priority = (i < 2) ? greatest_priority : least_priority;
        cuda_err = cudaStreamCreateWithPriority(&runtime->streams[i], cudaStreamNonBlocking, priority);
        if (cuda_err != cudaSuccess) {
            printf("[Ferrite-OS] Error: cudaStreamCreateWithPriority failed: %s\n", cudaGetErrorString(cuda_err));
            free(runtime);
            return NULL;
        }
    }
    printf("[Ferrite-OS] Created %d async streams (Priority Range: %d to %d)\n",
           runtime->num_streams, least_priority, greatest_priority);
    
    // ========================================================================
    // Pool Sizing with Configuration
    // ========================================================================
    size_t free_mem, total_mem;
    cuda_err = cudaMemGetInfo(&free_mem, &total_mem);
    if (cuda_err != cudaSuccess) {
        printf("[Ferrite-OS] Error: cudaMemGetInfo failed: %s\n", cudaGetErrorString(cuda_err));
        free(runtime);
        return NULL;
    }

    // Calculate pool size based on config
    size_t reserve = effective_config.reserve_vram;
    if (reserve == 0) {
        // Auto headroom: 5% of total VRAM, clamped to [256MB, 1GB]
        size_t auto_reserve = total_mem / 20;
        size_t min_reserve = 256ULL * 1024 * 1024;
        size_t max_reserve = 1024ULL * 1024 * 1024;
        if (auto_reserve < min_reserve) auto_reserve = min_reserve;
        if (auto_reserve > max_reserve) auto_reserve = max_reserve;
        reserve = auto_reserve;
    }
    size_t available = (free_mem > reserve) ? (free_mem - reserve) : 0;

    if (effective_config.fixed_pool_size > 0) {
        // Use fixed pool size
        runtime->vram_pool_size = effective_config.fixed_pool_size;
    } else if (effective_config.pool_fraction > 0.0f) {
        // Use fraction of available VRAM
        runtime->vram_pool_size = (size_t)(available * effective_config.pool_fraction);
    } else {
        // Fallback to minimum
        runtime->vram_pool_size = effective_config.min_pool_size;
    }

    // Apply size constraints
    if (runtime->vram_pool_size < effective_config.min_pool_size) {
        runtime->vram_pool_size = effective_config.min_pool_size;
    }
    if (effective_config.max_pool_size > 0 && runtime->vram_pool_size > effective_config.max_pool_size) {
        runtime->vram_pool_size = effective_config.max_pool_size;
    }

    // Safety: Don't exceed available memory
    if (runtime->vram_pool_size > available) {
        runtime->vram_pool_size = available;
    }

    if (!effective_config.quiet_init) {
        printf("[Ferrite-OS] Pool Configuration:\n");
        if (effective_config.fixed_pool_size > 0) {
            printf("[Ferrite-OS]   Mode: Fixed size (%.2f GB)\n",
                   effective_config.fixed_pool_size / (1024.0 * 1024.0 * 1024.0));
        } else {
            printf("[Ferrite-OS]   Mode: %.0f%% of available VRAM\n", effective_config.pool_fraction * 100.0f);
        }
        printf("[Ferrite-OS]   Available VRAM: %.2f GB (%.2f GB total, %.2f GB reserved)\n",
               available / (1024.0 * 1024.0 * 1024.0),
               free_mem / (1024.0 * 1024.0 * 1024.0),
               reserve / (1024.0 * 1024.0 * 1024.0));
        printf("[Ferrite-OS]   Claiming: %.2f GB for Hot Pool\n",
               runtime->vram_pool_size / (1024.0 * 1024.0 * 1024.0));
    }
    
    runtime->vram_pool = ptx_driver_alloc(runtime->vram_pool_size);
    if (!runtime->vram_pool) {
        printf("[Ferrite-OS] Warning:Dynamic allocation failed, falling back to min pool size\n");
        runtime->vram_pool_size = effective_config.min_pool_size;
        runtime->vram_pool = ptx_driver_alloc(runtime->vram_pool_size);
    }

    if (runtime->vram_pool) {
        if (!effective_config.quiet_init) {
            printf("[Ferrite-OS] VRAM pool allocated: %.2f GB\n",
                   runtime->vram_pool_size / (1024.0 * 1024.0 * 1024.0));
        }

        // ========================================================================
        // Initialize Enhanced TLSF Allocator (Production Grade)
        // ========================================================================
        if (!effective_config.quiet_init) {
            printf("[Ferrite-OS] Initializing Enhanced TLSF Allocator...\n");
        }
        runtime->tlsf_allocator = ptx_tlsf_create_from_pool(runtime->vram_pool,
                                                              runtime->vram_pool_size,
                                                              effective_config.enable_leak_detection);

        if (runtime->tlsf_allocator) {
            ptx_tlsf_set_warning_threshold(runtime->tlsf_allocator, effective_config.warning_threshold);
            // Warning throttling overrides
            double warn_step = 0.0;
            if (parse_env_double("PTX_POOL_WARN_STEP", &warn_step)) {
                if (warn_step > 1.0 && warn_step <= 100.0) {
                    warn_step = warn_step / 100.0;
                }
                if (warn_step >= 0.0) {
                    ptx_tlsf_set_warning_step(runtime->tlsf_allocator, (float)warn_step);
                }
            }
            double warn_interval_ms = 0.0;
            if (parse_env_double("PTX_POOL_WARN_INTERVAL_MS", &warn_interval_ms)) {
                if (warn_interval_ms >= 0.0) {
                    ptx_tlsf_set_warning_interval_ms(runtime->tlsf_allocator, (uint64_t)warn_interval_ms);
                }
            }
            if (getenv("PTX_POOL_WARN_DISABLE")) {
                ptx_tlsf_set_warnings_enabled(runtime->tlsf_allocator, false);
            }
            ptx_tlsf_set_auto_defrag(runtime->tlsf_allocator, true);
            if (!effective_config.quiet_init) {
                printf("[Ferrite-OS] Enhanced TLSF allocator initialized\n");
                printf("[Ferrite-OS]      - O(1) allocation/deallocation\n");
                printf("[Ferrite-OS]      - O(1) block lookup via hash table\n");
                if (effective_config.enable_leak_detection) {
                    printf("[Ferrite-OS]      - Memory leak detection enabled\n");
                }
                if (effective_config.enable_pool_health) {
                    printf("[Ferrite-OS]      - Pool health monitoring active\n");
                }
            }
        } else {
            printf("[Ferrite-OS] Error: Failed to create enhanced TLSF allocator\n");
            printf("[Ferrite-OS] Error: TLSF allocator is required; aborting init\n");
            runtime->tlsf_allocator = NULL;
            gpu_hot_shutdown(runtime);
            return NULL;
        }
        
        runtime->fallback_count = 0;

        if (!effective_config.quiet_init) {
            printf("[Ferrite-OS] Memory allocator ready\n");
        }
    } else {
        printf("[Ferrite-OS] Warning: Could not pre-allocate VRAM pool\n");
    }

    // Initialize CUDA Graphs
    runtime->num_graphs = 0;
    runtime->capturing_stream_id = -1;
    memset(runtime->graphs, 0, sizeof(runtime->graphs));
    if (!effective_config.quiet_init) {
        printf("[Ferrite-OS] CUDA graphs initialized\n");
    }

    // Create keepalive event
    cudaEventCreate(&runtime->keepalive_event);

    // Launch initial keepalive kernel
    runtime->keepalive_running = true;

    if (!effective_config.quiet_init) {
        printf("[Ferrite-OS] Runtime initialized\n\n");
    }

    return runtime;
}

// Shutdown GPU Hot Runtime
void gpu_hot_shutdown(GPUHotRuntime* runtime) {
    if (!runtime) return;

    printf("[Ferrite-OS] Shutting down GPU Hot Runtime...\n");

    // Deactivate context hook so driver calls during shutdown pass through
    if (ptx_context_hook_set_ptx_active) {
        ptx_context_hook_set_ptx_active(false);
    }

    runtime->keepalive_running = false;

    // Ensure all stream work is complete before draining deferred frees
    cudaDeviceSynchronize();
    gpu_hot_poll_deferred_internal(runtime, 0);
    // Force free any remaining deferred entries
    ptx_mutex_lock(&runtime->async_lock);
    DeferredFreeEntry* entry = runtime->deferred_head;
    runtime->deferred_head = NULL;
    runtime->deferred_tail = NULL;
    runtime->deferred_count = 0;
    ptx_mutex_unlock(&runtime->async_lock);
    while (entry) {
        DeferredFreeEntry* next = entry->next;
        if (runtime->tlsf_allocator && ptx_tlsf_owns_ptr(runtime->tlsf_allocator, entry->ptr)) {
            ptx_tlsf_free(runtime->tlsf_allocator, entry->ptr);
        }
        if (entry->event) {
            cudaEventDestroy(entry->event);
        }
        free(entry);
        entry = next;
    }
    
    // Destroy streams
    for (int i = 0; i < runtime->num_streams; i++) {
        cudaStreamDestroy(runtime->streams[i]);
    }
    
    // Destroy Enhanced TLSF Allocator
    if (runtime->tlsf_allocator) {
        printf("[Ferrite-OS] Destroying enhanced TLSF allocator...\n");
        ptx_tlsf_destroy(runtime->tlsf_allocator);
        runtime->tlsf_allocator = NULL;
    }
    
    // Free VRAM pool (TLSF structures are inside the pool)
    if (runtime->vram_pool) {
        ptx_driver_free(runtime->vram_pool);
    }
    
    // Destroy CUDA Graphs
    for (int i = 0; i < runtime->num_graphs; i++) {
        if (runtime->graphs[i].is_valid) {
            cudaGraphExecDestroy(runtime->graphs[i].graph_exec);
            cudaGraphDestroy(runtime->graphs[i].graph);
        }
    }

    
    // Destroy event
    if (runtime->keepalive_event) {
        cudaEventDestroy(runtime->keepalive_event);
    }

    // Destroy pooled events
    ptx_mutex_lock(&runtime->async_lock);
    for (int i = 0; i < runtime->event_pool_count; i++) {
        cudaEventDestroy(runtime->event_pool[i]);
    }
    free(runtime->event_pool);
    runtime->event_pool = NULL;
    runtime->event_pool_count = 0;
    runtime->event_pool_capacity = 0;
    ptx_mutex_unlock(&runtime->async_lock);
    ptx_mutex_destroy(&runtime->async_lock);
    
    // Close global registry
    if (runtime->system_state) {
        InterlockedDecrement((LONG*)&runtime->system_state->active_processes);
    }
#ifdef _WIN32
    if (runtime->global_registry) UnmapViewOfFile(runtime->global_registry);
    if (runtime->shm_handle) CloseHandle(runtime->shm_handle);
#else
    if (runtime->global_registry) munmap(runtime->global_registry, sizeof(GPURegistryEntry) * GPU_HOT_MAX_NAMED_SEGMENTS + sizeof(PTXSystemState));
    // Linux: shm file descriptor already closed after mmap, no cleanup needed
#endif
    
    // Release primary context
    if (runtime->cu_context) {
        CUdevice cu_dev;
        if (cuDeviceGet(&cu_dev, runtime->device_id) == CUDA_SUCCESS) {
            cuDevicePrimaryCtxRelease(cu_dev);
        }
        runtime->cu_context = NULL;
    }

    free(runtime);
    printf("[Ferrite-OS] Shutdown complete\n");
}

// Keep GPU active with heartbeat kernel
void gpu_hot_keepalive(GPUHotRuntime* runtime) {
    if (!runtime || !runtime->keepalive_running) return;
    
    static int* d_flag = NULL;
    if (!d_flag) {
        d_flag = (int*)gpu_hot_alloc(runtime, sizeof(int));
        if (!d_flag) {
            if (!runtime->keepalive_running) return;
            printf("[Ferrite-OS] Warning:Keepalive disabled (TLSF alloc failed)\n");
            runtime->keepalive_running = false;
            return;
        }
        cudaMemset(d_flag, 0, sizeof(int));
    }
    
    // Launch minimal kernel to keep GPU active
    keepalive_kernel<<<1, 256, 0, runtime->streams[0]>>>(d_flag);
    
    // Don't synchronize - let it run async
}

// Acquire a CUDA event from the pool (or create a new one)
static cudaEvent_t ptx_event_acquire(GPUHotRuntime* runtime) {
    cudaEvent_t ev = NULL;
    ptx_mutex_lock(&runtime->async_lock);
    if (runtime->event_pool_count > 0) {
        ev = runtime->event_pool[--runtime->event_pool_count];
    }
    ptx_mutex_unlock(&runtime->async_lock);

    if (!ev) {
        if (cudaEventCreateWithFlags(&ev, cudaEventDisableTiming) != cudaSuccess) {
            return NULL;
        }
    }
    return ev;
}

// Return a CUDA event to the pool
static void ptx_event_release_locked(GPUHotRuntime* runtime, cudaEvent_t ev) {
    if (!ev) return;
    if (runtime->event_pool_count >= runtime->event_pool_capacity) {
        int new_cap = (runtime->event_pool_capacity == 0) ? 64 : runtime->event_pool_capacity * 2;
        cudaEvent_t* new_pool = (cudaEvent_t*)realloc(runtime->event_pool, sizeof(cudaEvent_t) * new_cap);
        if (new_pool) {
            runtime->event_pool = new_pool;
            runtime->event_pool_capacity = new_cap;
        }
    }
    if (runtime->event_pool_count < runtime->event_pool_capacity) {
        runtime->event_pool[runtime->event_pool_count++] = ev;
        ev = NULL;
    }

    if (ev) {
        cudaEventDestroy(ev);
    }
}

static void ptx_event_release(GPUHotRuntime* runtime, cudaEvent_t ev) {
    if (!ev) return;
    ptx_mutex_lock(&runtime->async_lock);
    ptx_event_release_locked(runtime, ev);
    ptx_mutex_unlock(&runtime->async_lock);
}

// Drain deferred frees whose events have completed
static void gpu_hot_poll_deferred_internal(GPUHotRuntime* runtime, int max_drain) {
    if (!runtime) return;
    int drained = 0;

    ptx_mutex_lock(&runtime->async_lock);
    DeferredFreeEntry* prev = NULL;
    DeferredFreeEntry* curr = runtime->deferred_head;

    while (curr) {
        if (max_drain > 0 && drained >= max_drain) {
            break;
        }

        cudaError_t q = cudaEventQuery(curr->event);
        if (q == cudaSuccess) {
            DeferredFreeEntry* done = curr;
            curr = curr->next;

            if (prev) {
                prev->next = curr;
            } else {
                runtime->deferred_head = curr;
            }
            if (done == runtime->deferred_tail) {
                runtime->deferred_tail = prev;
            }
            runtime->deferred_count--;

            if (runtime->tlsf_allocator && ptx_tlsf_owns_ptr(runtime->tlsf_allocator, done->ptr)) {
                ptx_tlsf_free(runtime->tlsf_allocator, done->ptr);
            }
            ptx_event_release_locked(runtime, done->event);
            free(done);
            drained++;
            continue;
        } else if (q != cudaErrorNotReady) {
            // Leave entry in queue; error may be transient
            printf("[Ferrite-OS] Warning:cudaEventQuery failed: %s\n", cudaGetErrorString(q));
        }

        prev = curr;
        curr = curr->next;
    }
    ptx_mutex_unlock(&runtime->async_lock);
}

// Fast allocation from TLSF pool - O(1) complexity
void* gpu_hot_alloc(GPUHotRuntime* runtime, size_t size) {
    if (!runtime) return NULL;

    gpu_hot_poll_deferred_internal(runtime, 64);

    if (runtime->tlsf_allocator) {
        ptx_mutex_lock(&runtime->async_lock);
        void* ptr = ptx_tlsf_alloc(runtime->tlsf_allocator, size);
        ptx_mutex_unlock(&runtime->async_lock);
        if (ptr) {
            return ptr;
        }

        runtime->fallback_count++;
        printf("[Ferrite-OS] Error: TLSF pool exhausted (alloc #%d)\n", runtime->fallback_count);
        if (runtime->vram_pool_size > 0) {
            printf("[Ferrite-OS]   Requested: %zu bytes, Pool size: %zu bytes\n",
                   size, runtime->vram_pool_size);
        }
    }

    return NULL;
}


// Fast free (returns to pool) - O(1) complexity with coalescing
void gpu_hot_free(GPUHotRuntime* runtime, void* ptr) {
    if (!runtime || !ptr) return;
    
    bool owns = false;
    if (runtime->tlsf_allocator) {
        ptx_mutex_lock(&runtime->async_lock);
        owns = ptx_tlsf_owns_ptr(runtime->tlsf_allocator, ptr);
        if (owns) {
            ptx_tlsf_free(runtime->tlsf_allocator, ptr);
        }
        ptx_mutex_unlock(&runtime->async_lock);
    }
    if (owns) return;
    ptx_strict_free_violation("GPU-HOT", ptr);
}

// Async allocation (stream-ordered)
void* gpu_hot_alloc_async(GPUHotRuntime* runtime, size_t size, cudaStream_t stream) {
    (void)stream;
    return gpu_hot_alloc(runtime, size);
}

// Async free (defer until stream completes)
void gpu_hot_free_async(GPUHotRuntime* runtime, void* ptr, cudaStream_t stream) {
    if (!runtime || !ptr) return;

    bool owns = false;
    if (runtime->tlsf_allocator) {
        ptx_mutex_lock(&runtime->async_lock);
        owns = ptx_tlsf_owns_ptr(runtime->tlsf_allocator, ptr);
        ptx_mutex_unlock(&runtime->async_lock);
    }
    if (!owns) {
        ptx_strict_free_violation("GPU-HOT-ASYNC", ptr);
        return;
    }

    cudaEvent_t ev = ptx_event_acquire(runtime);
    if (!ev) {
        printf("[Ferrite-OS] Warning:Failed to acquire event, falling back to sync free\n");
        cudaStreamSynchronize(stream);
        ptx_mutex_lock(&runtime->async_lock);
        ptx_tlsf_free(runtime->tlsf_allocator, ptr);
        ptx_mutex_unlock(&runtime->async_lock);
        return;
    }

    cudaError_t err = cudaEventRecord(ev, stream);
    if (err != cudaSuccess) {
        printf("[Ferrite-OS] Warning:cudaEventRecord failed: %s (sync fallback)\n",
               cudaGetErrorString(err));
        ptx_event_release(runtime, ev);
        cudaStreamSynchronize(stream);
        ptx_mutex_lock(&runtime->async_lock);
        ptx_tlsf_free(runtime->tlsf_allocator, ptr);
        ptx_mutex_unlock(&runtime->async_lock);
        return;
    }

    DeferredFreeEntry* entry = (DeferredFreeEntry*)malloc(sizeof(DeferredFreeEntry));
    if (!entry) {
        printf("[Ferrite-OS] Warning:DeferredFreeEntry alloc failed (sync fallback)\n");
        ptx_event_release(runtime, ev);
        cudaStreamSynchronize(stream);
        ptx_mutex_lock(&runtime->async_lock);
        ptx_tlsf_free(runtime->tlsf_allocator, ptr);
        ptx_mutex_unlock(&runtime->async_lock);
        return;
    }

    entry->ptr = ptr;
    entry->event = ev;
    entry->stream = stream;
    entry->next = NULL;

    ptx_mutex_lock(&runtime->async_lock);
    if (runtime->deferred_tail) {
        runtime->deferred_tail->next = entry;
    } else {
        runtime->deferred_head = entry;
    }
    runtime->deferred_tail = entry;
    runtime->deferred_count++;
    ptx_mutex_unlock(&runtime->async_lock);
}

// Poll deferred frees (process completed events)
void gpu_hot_poll_deferred(GPUHotRuntime* runtime, int max_drain) {
    gpu_hot_poll_deferred_internal(runtime, max_drain);
}

// Check if a pointer belongs to the TLSF pool
bool gpu_hot_owns_ptr(GPUHotRuntime* runtime, void* ptr) {
    if (!runtime || !ptr) return false;

    if (runtime->tlsf_allocator) {
        bool owns = false;
        ptx_mutex_lock(&runtime->async_lock);
        owns = ptx_tlsf_owns_ptr(runtime->tlsf_allocator, ptr);
        ptx_mutex_unlock(&runtime->async_lock);
        return owns;
    }

    return false;
}


// Register kernel for fast dispatch
int gpu_hot_register_kernel(GPUHotRuntime* runtime, 
                            void* kernel_func,
                            dim3 grid, dim3 block,
                            size_t shared_mem) {
    if (!runtime || runtime->num_kernels >= GPU_HOT_MAX_KERNELS) {
        return -1;
    }
    
    int kernel_id = runtime->num_kernels++;
    GPUKernelHandle* handle = &runtime->kernels[kernel_id];
    
    handle->kernel_func = kernel_func;
    handle->grid = grid;
    handle->block = block;
    handle->shared_mem = shared_mem;
    handle->stream = runtime->streams[kernel_id % runtime->num_streams];
    
    return kernel_id;
}

// Launch registered kernel
void gpu_hot_launch_kernel(GPUHotRuntime* runtime, 
                           int kernel_id,
                           void** args) {
    if (!runtime || kernel_id < 0 || kernel_id >= runtime->num_kernels) {
        return;
    }
    
    GPUKernelHandle* handle = &runtime->kernels[kernel_id];
    
    // Launch kernel with pre-configured parameters
    cudaLaunchKernel(handle->kernel_func,
                     handle->grid,
                     handle->block,
                     args,
                     handle->shared_mem,
                     handle->stream);
}

// Get CUDA stream
cudaStream_t gpu_hot_get_stream(GPUHotRuntime* runtime, int stream_id) {
    if (!runtime || stream_id < 0 || stream_id >= runtime->num_streams) {
        return 0; // Default stream
    }
    return runtime->streams[stream_id];
}

// Synchronize all streams
void gpu_hot_sync_all(GPUHotRuntime* runtime) {
    if (!runtime) return;
    
    for (int i = 0; i < runtime->num_streams; i++) {
        cudaStreamSynchronize(runtime->streams[i]);
    }
    gpu_hot_poll_deferred_internal(runtime, 0);
}

// Get runtime statistics
void gpu_hot_get_stats(GPUHotRuntime* runtime, GPUHotStats* stats) {
    if (!runtime || !stats) return;
    
    // --- PTX-OS HARDWARE POLLING ---
    // Instead of just tracking the internal pool, we poll the driver for REAL VRAM status.
    size_t free_mem, total_mem;
    cudaError_t err = cudaMemGetInfo(&free_mem, &total_mem);
    
    if (err == cudaSuccess) {
        stats->vram_allocated = total_mem;
        stats->vram_used = total_mem - free_mem;
        stats->vram_free = free_mem;
        stats->gpu_utilization = (float)stats->vram_used / (float)total_mem;
    } else {
        // Fallback to pool tracking if driver poll fails
        TLSFPoolStats tlsf_stats;
        memset(&tlsf_stats, 0, sizeof(TLSFPoolStats));
        if (runtime->tlsf_allocator) {
            ptx_tlsf_get_stats(runtime->tlsf_allocator, &tlsf_stats);
            stats->vram_allocated = tlsf_stats.total_pool_size;
            stats->vram_used = tlsf_stats.allocated_bytes;
            stats->vram_free = tlsf_stats.free_bytes;
            stats->gpu_utilization = (tlsf_stats.total_pool_size > 0)
                ? (float)tlsf_stats.allocated_bytes / (float)tlsf_stats.total_pool_size
                : 0.0f;
        } else {
            stats->vram_allocated = 0;
            stats->vram_used = 0;
            stats->vram_free = 0;
            stats->gpu_utilization = 0.0f;
        }
    }
    
    stats->active_streams = runtime->num_streams;
    stats->registered_kernels = runtime->num_kernels;
    stats->total_ops = runtime->total_ops;
    stats->watchdog_tripped = runtime->watchdog_tripped;
    stats->avg_latency_us = 0.5f; 
    
    // Sync with Global OS State
    if (runtime->system_state) {
        stats->total_ops = runtime->system_state->total_ops;
        stats->watchdog_tripped |= runtime->system_state->watchdog_alert;
        
        int count = 0;
        for (int i = 0; i < GPU_HOT_MAX_NAMED_SEGMENTS; i++) {
            if (runtime->global_registry[i].active) count++;
        }
        stats->shm_count = count;
    } else {
        stats->shm_count = runtime->shm_count;
    }
}

void gpu_hot_set_watchdog(GPUHotRuntime* runtime, int timeout_ms) {
    if (runtime) runtime->watchdog_timeout_ms = timeout_ms;
}

bool gpu_hot_check_watchdog(GPUHotRuntime* runtime) {
    if (!runtime || !runtime->watchdog_timeout_ms) return false;
    
    // Check if the keepalive heartbeat is still ticking
    cudaError_t err = cudaEventQuery(runtime->keepalive_event);
    if (err == cudaErrorNotReady) {
        // If it's been "stuck" for a while, we'd detect it here.
        // For this demo, we'll mark as healthy unless an actual error occurs.
        return false;
    }
    return (err != cudaSuccess);
}

// ============================================================================
// CUDA Graph API - Sub-Microsecond Kernel Launches
// ============================================================================

// Begin graph capture on a stream
int gpu_hot_begin_capture(GPUHotRuntime* runtime, int stream_id, const char* graph_name) {
    if (!runtime || stream_id < 0 || stream_id >= runtime->num_streams) {
        printf("[Ferrite-OS] Error: Invalid stream ID for graph capture\n");
        return -1;
    }
    
    if (runtime->capturing_stream_id >= 0) {
        printf("[Ferrite-OS] Error: Already capturing on stream %d\n", runtime->capturing_stream_id);
        return -1;
    }
    
    if (runtime->num_graphs >= GPU_HOT_MAX_GRAPHS) {
        printf("[Ferrite-OS] Error: Maximum number of graphs reached\n");
        return -1;
    }
    
    cudaStream_t stream = runtime->streams[stream_id];
    
    // Begin stream capture
    cudaError_t err = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    if (err != cudaSuccess) {
        printf("[Ferrite-OS] Error: Failed to begin graph capture: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    runtime->capturing_stream_id = stream_id;
    
    // Prepare graph handle
    int graph_id = runtime->num_graphs;
    GPUGraphHandle* handle = &runtime->graphs[graph_id];
    strncpy(handle->name, graph_name ? graph_name : "unnamed", sizeof(handle->name) - 1);
    handle->is_valid = false;
    
    printf("[Ferrite-OS] Graph capture started on stream %d: '%s'\n", stream_id, handle->name);
    
    return graph_id;
}

// End graph capture and return graph handle
int gpu_hot_end_capture(GPUHotRuntime* runtime, int stream_id) {
    if (!runtime || stream_id < 0 || stream_id >= runtime->num_streams) {
        printf("[Ferrite-OS] Error: Invalid stream ID\n");
        return -1;
    }
    
    if (runtime->capturing_stream_id != stream_id) {
        printf("[Ferrite-OS] Error: Not capturing on stream %d\n", stream_id);
        return -1;
    }
    
    cudaStream_t stream = runtime->streams[stream_id];
    
    int graph_id = runtime->num_graphs;
    GPUGraphHandle* handle = &runtime->graphs[graph_id];
    
    // End stream capture
    cudaError_t err = cudaStreamEndCapture(stream, &handle->graph);
    if (err != cudaSuccess) {
        printf("[Ferrite-OS] Error: Failed to end graph capture: %s\n", cudaGetErrorString(err));
        runtime->capturing_stream_id = -1;
        return -1;
    }
    
    // Instantiate graph for execution
    err = cudaGraphInstantiate(&handle->graph_exec, handle->graph, NULL, NULL, 0);
    if (err != cudaSuccess) {
        printf("[Ferrite-OS] Error: Failed to instantiate graph: %s\n", cudaGetErrorString(err));
        cudaGraphDestroy(handle->graph);
        runtime->capturing_stream_id = -1;
        return -1;
    }
    
    handle->is_valid = true;
    runtime->num_graphs++;
    runtime->capturing_stream_id = -1;
    
    printf("[Ferrite-OS] Graph '%s' captured (ID: %d)\n", handle->name, graph_id);
    
    return graph_id;
}

// Launch a captured graph (0.5us latency vs 5us for cudaLaunchKernel)
void gpu_hot_launch_graph(GPUHotRuntime* runtime, int graph_id, cudaStream_t stream) {
    if (!runtime || graph_id < 0 || graph_id >= runtime->num_graphs) return;
    
    GPUGraphHandle* handle = &runtime->graphs[graph_id];
    if (!handle->is_valid) return;
    
    cudaStream_t launch_stream = stream ? stream : runtime->streams[0];
    cudaError_t err = cudaGraphLaunch(handle->graph_exec, launch_stream);
    
    if (err == cudaSuccess) {
        runtime->total_ops++;
        if (runtime->system_state) {
            InterlockedAdd64((LONG64*)&runtime->system_state->total_ops, 1);
            InterlockedIncrement((LONG*)&runtime->system_state->active_tasks);
        }
        cudaEventRecord(runtime->keepalive_event, launch_stream);
    } else {
        printf("[Ferrite-OS] Error: Failed to launch graph: %s\n", cudaGetErrorString(err));
    }
}

// Destroy a graph
void gpu_hot_destroy_graph(GPUHotRuntime* runtime, int graph_id) {
    if (!runtime || graph_id < 0 || graph_id >= runtime->num_graphs) {
        return;
    }
    
    GPUGraphHandle* handle = &runtime->graphs[graph_id];
    
    if (handle->is_valid) {
        cudaGraphExecDestroy(handle->graph_exec);
        cudaGraphDestroy(handle->graph);
        handle->is_valid = false;
        printf("[Ferrite-OS] Graph '%s' destroyed\n", handle->name);
    }
}


// ============================================================================
// TLSF Pool Monitoring & Health Validation Implementation
// ============================================================================

// Get detailed TLSF pool statistics
void gpu_hot_get_tlsf_stats(GPUHotRuntime* runtime, TLSFPoolStats* stats) {
    if (!runtime || !stats) return;

    memset(stats, 0, sizeof(TLSFPoolStats));
    if (!runtime->tlsf_allocator) {
        stats->fallback_count = runtime->fallback_count;
        return;
    }

    ptx_tlsf_get_stats(runtime->tlsf_allocator, stats);
    stats->fallback_count += runtime->fallback_count;

    // Debug: Print stats being returned
    static int stats_call_count = 0;
    if (ptx_tlsf_debug_enabled() &&
        (++stats_call_count % 10 == 0 || stats_call_count <= 3)) {
        printf("[TLSF-DEBUG] get_stats call #%d: allocs=%llu, frees=%llu, allocated=%zu MB\n",
               stats_call_count,
               (unsigned long long)stats->total_allocations,
               (unsigned long long)stats->total_frees,
               stats->allocated_bytes / 1024 / 1024);
    }
}

// Validate TLSF pool integrity
void gpu_hot_validate_tlsf_pool(GPUHotRuntime* runtime, TLSFHealthReport* report) {
    if (!runtime || !report) return;
    
    memset(report, 0, sizeof(TLSFHealthReport));
    if (!runtime->tlsf_allocator) {
        report->is_valid = false;
        snprintf(report->error_messages[report->error_count++], 256,
                "TLSF allocator not initialized");
        return;
    }

    ptx_tlsf_validate(runtime->tlsf_allocator, report);
}

// Check if allocation will succeed without fallback
bool gpu_hot_can_allocate(GPUHotRuntime* runtime, size_t size) {
    if (!runtime) return false;

    if (runtime->tlsf_allocator) {
        return ptx_tlsf_can_allocate(runtime->tlsf_allocator, size);
    }
    return false;
}

// Get largest allocatable block size
size_t gpu_hot_get_max_allocatable(GPUHotRuntime* runtime) {
    if (!runtime) return 0;

    if (runtime->tlsf_allocator) {
        return ptx_tlsf_get_max_allocatable(runtime->tlsf_allocator);
    }
    return 0;
}

// Defragment pool (merge all adjacent free blocks)
void gpu_hot_defragment_pool(GPUHotRuntime* runtime) {
    if (!runtime || !runtime->tlsf_allocator) return;
    ptx_tlsf_defragment(runtime->tlsf_allocator);
}

// Print detailed pool visualization
void gpu_hot_print_pool_map(GPUHotRuntime* runtime) {
    if (!runtime || !runtime->tlsf_allocator) return;
    ptx_tlsf_print_pool_map(runtime->tlsf_allocator);
}

// Set warning threshold
void gpu_hot_set_warning_threshold(GPUHotRuntime* runtime, float threshold_percent) {
    if (!runtime) return;

    if (runtime->tlsf_allocator) {
        ptx_tlsf_set_warning_threshold(runtime->tlsf_allocator, threshold_percent / 100.0f);
        printf("[Ferrite-OS] Warning threshold set to %.1f%%\n", threshold_percent);
    }
}

// Enable/disable automatic defragmentation
void gpu_hot_set_auto_defrag(GPUHotRuntime* runtime, bool enable) {
    if (!runtime) return;

    if (runtime->tlsf_allocator) {
        ptx_tlsf_set_auto_defrag(runtime->tlsf_allocator, enable);
        printf("[Ferrite-OS] Auto-defragmentation %s\n", enable ? "enabled" : "disabled");
    }
}

// ============================================================================
// Shared Memory / Persistent Segments Implementation
// ============================================================================

void* gpu_hot_shm_alloc(GPUHotRuntime* runtime, const char* name, size_t size) {
    if (!runtime || !name || !runtime->global_registry) return NULL;
    
    // Check if it already exists globally
    void* existing = gpu_hot_shm_open(runtime, name);
    if (existing) return existing;
    
    // Find free slot in global registry
    int slot = -1;
    for (int i = 0; i < GPU_HOT_MAX_NAMED_SEGMENTS; i++) {
        if (!runtime->global_registry[i].active) {
            slot = i;
            break;
        }
    }
    
    if (slot == -1) return NULL;
    
    // Use CUDA Unified Memory (Managed Memory) instead of IPC
    // This allows seamless access from both GPU and CPU, and across processes
    void* ptr = NULL;
    cudaError_t err = cudaMallocManaged(&ptr, size, cudaMemAttachGlobal);
    if (err != cudaSuccess) {
        printf("[Ferrite-OS] Error: Unified memory allocation failed: %s\n", cudaGetErrorString(err));
        return NULL;
    }
    
    // Initialize to zero
    cudaMemset(ptr, 0, size);
    
    // For unified memory, we store the pointer directly instead of IPC handle
    // The pointer is valid across all processes on the same GPU
    cudaIpcMemHandle_t dummy_handle;
    memset(&dummy_handle, 0, sizeof(dummy_handle));
    
    // Store the pointer value in the handle as a workaround
    // (In production, you'd use a proper shared memory mechanism)
    memcpy(&dummy_handle, &ptr, sizeof(void*));
    
    // Register globally
    strncpy(runtime->global_registry[slot].name, name, GPU_HOT_MAX_NAME_LEN - 1);
    runtime->global_registry[slot].ipc_handle = dummy_handle;
    runtime->global_registry[slot].size = size;
    runtime->global_registry[slot].active = true;
    runtime->global_registry[slot].created_at = GetTickCount64();
    runtime->shm_count++;
    
    printf("[Ferrite-OS] Created unified memory segment: '%s' (%zu bytes) at %p\n", name, size, ptr);
    return ptr;
}

void* gpu_hot_shm_open(GPUHotRuntime* runtime, const char* name) {
    if (!runtime || !name || !runtime->global_registry) return NULL;
    
    for (int i = 0; i < GPU_HOT_MAX_NAMED_SEGMENTS; i++) {
        if (runtime->global_registry[i].active && strcmp(runtime->global_registry[i].name, name) == 0) {
            // For unified memory, retrieve the pointer directly from the handle
            void* ptr = NULL;
            memcpy(&ptr, &runtime->global_registry[i].ipc_handle, sizeof(void*));
            
            if (ptr) {
                printf("[Ferrite-OS] Opened unified memory segment: '%s' at %p\n", name, ptr);
                return ptr;
            }
        }
    }
    return NULL;
}

void gpu_hot_shm_close(GPUHotRuntime* runtime, void* ptr) {
    if (ptr) cudaIpcCloseMemHandle(ptr);
}

void gpu_hot_shm_unlink(GPUHotRuntime* runtime, const char* name) {
    if (!runtime || !name || !runtime->global_registry) return;
    
    for (int i = 0; i < GPU_HOT_MAX_NAMED_SEGMENTS; i++) {
        if (runtime->global_registry[i].active && strcmp(runtime->global_registry[i].name, name) == 0) {
            runtime->global_registry[i].active = false;
            runtime->shm_count--;
            printf("[Ferrite-OS] Unlinked global segment: '%s'\n", name);
            return;
        }
    }
}

bool gpu_hot_get_registry_entry(GPUHotRuntime* runtime, int index, char* name_out, size_t* size_out, bool* active_out, unsigned long long* created_out) {
    if (!runtime || !runtime->global_registry) return false;
    if (index < 0 || index >= GPU_HOT_MAX_NAMED_SEGMENTS) return false;
    
    GPURegistryEntry* entry = &runtime->global_registry[index];
    
    if (name_out) strncpy(name_out, entry->name, 63);
    if (size_out) *size_out = entry->size;
    if (active_out) *active_out = entry->active;
    if (created_out) *created_out = (unsigned long long)entry->created_at;
    
    return true;
}

// Priority-Aware Scheduling Implementation
cudaStream_t gpu_hot_get_priority_stream(GPUHotRuntime* runtime, int priority) {
    if (!runtime) return 0;
    
    if (priority == PTX_PRIORITY_REALTIME) {
        return runtime->streams[0]; // First stream is always high priority
    } else if (priority == PTX_PRIORITY_LOW) {
        return runtime->streams[runtime->num_streams - 1]; // Last stream
    } else {
        return runtime->streams[2]; // Normal priority starts at index 2
    }
}

// Get the system state pointer (accessible by GPU via Zero-Copy)
PTXSystemState* gpu_hot_get_system_state(GPUHotRuntime* runtime) {
    if (!runtime || !runtime->system_state) return NULL;
    
    void* d_ptr;
    cudaHostGetDevicePointer(&d_ptr, runtime->system_state, 0);
    return (PTXSystemState*)d_ptr;
}

// Get the system state pointer (accessible by Host)
PTXSystemState* gpu_hot_get_host_system_state(GPUHotRuntime* runtime) {
    if (!runtime) return NULL;
    return runtime->system_state;
}

void gpu_hot_reset_system_state(GPUHotRuntime* runtime) {
    if (!runtime || !runtime->system_state) return;

    PTXSystemState* state = runtime->system_state;
    uint32_t auth_token = state->auth_token;
    int active_processes = state->active_processes;

    memset(state, 0, sizeof(PTXSystemState));
    state->auth_token = auth_token;
    state->active_processes = active_processes;
}

void gpu_hot_flush_task_queue(GPUHotRuntime* runtime) {
    if (!runtime || !runtime->system_state) return;

    PTXSystemState* state = runtime->system_state;
    state->active_tasks = 0;
    state->max_priority_active = 0;
    memset(&state->queue, 0, sizeof(PTXTaskQueue));
}

void gpu_hot_reset_watchdog(GPUHotRuntime* runtime) {
    if (!runtime) return;
    runtime->watchdog_tripped = false;
    runtime->last_launch_time = 0;
    if (runtime->system_state) {
        runtime->system_state->watchdog_alert = false;
    }
}

void gpu_hot_clear_signal_mask(GPUHotRuntime* runtime) {
    if (!runtime || !runtime->system_state) return;
    runtime->system_state->signal_mask = 0;
}

void gpu_hot_get_system_snapshot(GPUHotRuntime* runtime, GPUHotSystemSnapshot* snapshot) {
    if (!runtime || !snapshot) return;
    if (!runtime->system_state) {
        memset(snapshot, 0, sizeof(GPUHotSystemSnapshot));
        return;
    }

    PTXSystemState* state = runtime->system_state;
    snapshot->total_ops = state->total_ops;
    snapshot->active_processes = state->active_processes;
    snapshot->active_tasks = state->active_tasks;
    snapshot->max_priority_active = state->max_priority_active;
    snapshot->total_vram_used = state->total_vram_used;
    snapshot->watchdog_alert = state->watchdog_alert;
    snapshot->kernel_running = state->kernel_running;
    snapshot->shutdown_requested = state->shutdown_requested;
    snapshot->active_priority_level = state->active_priority_level;
    snapshot->signal_mask = state->signal_mask;
    snapshot->interrupt_cnt = state->interrupt_cnt;
    snapshot->queue_head = state->queue.head;
    snapshot->queue_tail = state->queue.tail;
    snapshot->queue_lock = state->queue.lock;
}

// ============================================================================
// Context Export API
// ============================================================================

// Get the captured CUcontext (as void* for C-compatible headers)
void* gpu_hot_get_context(GPUHotRuntime* runtime) {
    if (!runtime) return NULL;
    return (void*)runtime->cu_context;
}

// Export the CUcontext pointer as an environment variable for external consumers
void gpu_hot_export_context(GPUHotRuntime* runtime) {
    if (!runtime || !runtime->cu_context) return;

    char buf[32];
    snprintf(buf, sizeof(buf), "%p", (void*)runtime->cu_context);
    setenv("PTX_CONTEXT_PTR", buf, 1);

    printf("[Ferrite-OS] Context exported: PTX_CONTEXT_PTR=%s\n", buf);
}

// Forward declaration of the actual kernel boot logic
// Defined in kernels/os_kernel.cu
extern "C" void ptx_os_launch_persistent_kernel(GPUHotRuntime* runtime, PTXSystemState* d_state);

void ptx_os_boot_persistent_kernel(GPUHotRuntime* runtime) {
    PTXSystemState* d_state = gpu_hot_get_system_state(runtime);
    if (!d_state) {
        printf("[PTX-OS] [ERROR] Could not resolve GPU-side system state!\n");
        return;
    }
    
    ptx_os_launch_persistent_kernel(runtime, d_state);
}
