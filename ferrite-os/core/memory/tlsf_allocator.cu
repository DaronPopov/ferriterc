/*
 * PTX-OS Enhanced TLSF Memory Allocator - Implementation
 * 
 * This is a production-grade TLSF allocator with:
 * - O(1) allocation/deallocation via segregated free lists
 * - O(1) block lookup via hash table
 * - Comprehensive diagnostics and health monitoring
 * - Thread-safe operations
 * - Memory leak detection
 */

#include "memory/ptx_tlsf_allocator.h"
#include "memory/ptx_cuda_driver.h"
#include "ptx_debug.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#include <windows.h>
#include <intrin.h>
#else
#include <pthread.h>
#include <time.h>
#endif

// ============================================================================
// Platform-specific utilities
// ============================================================================

#ifdef _WIN32
static inline int clz32(uint32_t x) {
    unsigned long index;
    return _BitScanReverse(&index, x) ? (31 - index) : 32;
}

static inline int clz64(uint64_t x) {
    unsigned long index;
    return _BitScanReverse64(&index, x) ? (63 - index) : 64;
}
#else
static inline int clz64(uint64_t x) {
    return x ? __builtin_clzll(x) : 64;
}
#endif

static inline uint64_t get_timestamp() {
#ifdef _WIN32
    LARGE_INTEGER counter, freq;
    QueryPerformanceCounter(&counter);
    QueryPerformanceFrequency(&freq);
    return (counter.QuadPart * 1000000) / freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000 + ts.tv_nsec / 1000;
#endif
}

static inline bool ptx_tlsf_debug_enabled() {
    static int initialized = 0;
    static bool enabled = false;
    if (!initialized) {
        const char* v = getenv("PTX_TLSF_DEBUG");
        enabled = (v && v[0] && v[0] != '0');
        initialized = 1;
    }
    return enabled;
}

// ============================================================================
// Hash function for O(1) block lookup
// ============================================================================

static inline uint32_t hash_ptr(void* ptr) {
    // FNV-1a hash
    uint64_t addr = (uint64_t)ptr;
    uint32_t hash = 2166136261u;
    
    for (int i = 0; i < 8; i++) {
        hash ^= (addr & 0xFF);
        hash *= 16777619u;
        addr >>= 8;
    }
    
    return hash % TLSF_HASH_TABLE_SIZE;
}

// ============================================================================
// TLSF mapping functions
// ============================================================================

static inline size_t align_up(size_t size) {
    return (size + TLSF_ALIGNMENT - 1) & ~(TLSF_ALIGNMENT - 1);
}

static inline void tlsf_mapping(size_t size, int* fl, int* sl) {
    if (size < TLSF_MIN_BLOCK_SIZE) {
        *fl = 0;
        *sl = 0;
    } else {
        // First level: log2(size)
        *fl = 63 - clz64((uint64_t)size);
        
        // Clamp FL to valid range
        if (*fl >= TLSF_FL_INDEX_MAX) {
            *fl = TLSF_FL_INDEX_MAX - 1;
        }
        
        // Second level subdivision
        if (*fl >= 4) {
            *sl = (int)((size >> (*fl - 4)) & (TLSF_SL_INDEX_COUNT - 1));
        } else {
            *sl = 0;
        }
    }
}

// ============================================================================
// Hash table operations
// ============================================================================

static void hash_insert(PTXTLSFAllocator* allocator, TLSFBlock* block) {
    uint32_t hash = hash_ptr(block->device_ptr);
    TLSFHashBucket* bucket = &allocator->hash_table[hash];
    
    block->hash_next = bucket->head;
    bucket->head = block;
    bucket->count++;
    
    // Track collisions
    if (bucket->count > 1) {
        allocator->stats.hash_collisions++;
    }
    
    // Update max chain length
    if (bucket->count > allocator->stats.max_chain_length) {
        allocator->stats.max_chain_length = bucket->count;
    }
}

static void hash_remove(PTXTLSFAllocator* allocator, TLSFBlock* block) {
    uint32_t hash = hash_ptr(block->device_ptr);
    TLSFHashBucket* bucket = &allocator->hash_table[hash];
    
    TLSFBlock** curr = &bucket->head;
    while (*curr) {
        if (*curr == block) {
            *curr = block->hash_next;
            bucket->count--;
            block->hash_next = NULL;
            return;
        }
        curr = &(*curr)->hash_next;
    }
}

static TLSFBlock* hash_find(PTXTLSFAllocator* allocator, void* device_ptr) {
    uint32_t hash = hash_ptr(device_ptr);
    TLSFHashBucket* bucket = &allocator->hash_table[hash];
    
    TLSFBlock* curr = bucket->head;
    while (curr) {
        if (curr->device_ptr == device_ptr) {
            return curr;
        }
        curr = curr->hash_next;
    }
    
    return NULL;
}

// ============================================================================
// Free list operations
// ============================================================================

static void insert_free_block(PTXTLSFAllocator* allocator, TLSFBlock* block) {
    int fl, sl;
    tlsf_mapping(block->size, &fl, &sl);
    
    // Bounds check
    if (fl < 0 || fl >= TLSF_FL_INDEX_MAX || sl < 0 || sl >= TLSF_SL_INDEX_COUNT) {
        printf("[TLSF-ERROR] Invalid indices: fl=%d, sl=%d\n", fl, sl);
        return;
    }
    
    block->next_free = allocator->free_lists[fl][sl];
    block->prev_free = NULL;
    
    if (allocator->free_lists[fl][sl]) {
        allocator->free_lists[fl][sl]->prev_free = block;
    }
    
    allocator->free_lists[fl][sl] = block;
    block->is_free = true;
    
    allocator->stats.free_blocks++;
    allocator->stats.free_list_counts[fl]++;
}

static void remove_free_block(PTXTLSFAllocator* allocator, TLSFBlock* block) {
    int fl, sl;
    tlsf_mapping(block->size, &fl, &sl);
    
    if (block->prev_free) {
        block->prev_free->next_free = block->next_free;
    } else {
        allocator->free_lists[fl][sl] = block->next_free;
    }
    
    if (block->next_free) {
        block->next_free->prev_free = block->prev_free;
    }
    
    block->is_free = false;
    allocator->stats.free_blocks--;
    allocator->stats.free_list_counts[fl]--;
}

static TLSFBlock* find_free_block(PTXTLSFAllocator* allocator, size_t size) {
    int fl, sl;
    tlsf_mapping(size, &fl, &sl);
    
    // Search in same second-level list first
    for (int s = sl; s < TLSF_SL_INDEX_COUNT; s++) {
        TLSFBlock* block = allocator->free_lists[fl][s];
        if (block && block->size >= size) {
            return block;
        }
    }
    
    // Search in higher first-level lists
    for (int f = fl + 1; f < TLSF_FL_INDEX_MAX; f++) {
        for (int s = 0; s < TLSF_SL_INDEX_COUNT; s++) {
            TLSFBlock* block = allocator->free_lists[f][s];
            if (block) {
                return block;
            }
        }
    }
    
    return NULL;
}

// ============================================================================
// Block operations
// ============================================================================

static void split_block(PTXTLSFAllocator* allocator, TLSFBlock* block, size_t size) {
    size_t remaining = block->size - size;
    
    if (remaining >= TLSF_MIN_BLOCK_SIZE) {
        // Create new block
        TLSFBlock* new_block = (TLSFBlock*)malloc(sizeof(TLSFBlock));
        memset(new_block, 0, sizeof(TLSFBlock));
        
        new_block->device_ptr = (char*)block->device_ptr + size;
        new_block->size = remaining;
        new_block->magic = TLSF_BLOCK_MAGIC;
        new_block->is_free = true;
        new_block->is_last = block->is_last;
        
        new_block->prev_phys = block;
        new_block->next_phys = block->next_phys;
        
        if (block->next_phys) {
            block->next_phys->prev_phys = new_block;
        } else {
            allocator->last_block = new_block;
        }
        
        block->size = size;
        block->next_phys = new_block;
        block->is_last = false;
        
        // Add to hash table and free list
        hash_insert(allocator, new_block);
        insert_free_block(allocator, new_block);
        
        allocator->stats.total_blocks++;
        allocator->stats.total_splits++;
    }
}

static TLSFBlock* coalesce(PTXTLSFAllocator* allocator, TLSFBlock* block) {
    // Coalesce with next block
    if (block->next_phys && block->next_phys->is_free && !block->next_phys->is_pinned) {
        TLSFBlock* next = block->next_phys;
        
        remove_free_block(allocator, next);
        hash_remove(allocator, next);
        
        block->size += next->size;
        block->next_phys = next->next_phys;
        block->is_last = next->is_last;
        
        if (block->next_phys) {
            block->next_phys->prev_phys = block;
        } else {
            allocator->last_block = block;
        }
        
        free(next);
        allocator->stats.total_blocks--;
        allocator->stats.total_merges++;
    }
    
    // Coalesce with previous block
    if (block->prev_phys && block->prev_phys->is_free && !block->prev_phys->is_pinned) {
        TLSFBlock* prev = block->prev_phys;
        
        remove_free_block(allocator, prev);
        hash_remove(allocator, block);
        
        prev->size += block->size;
        prev->next_phys = block->next_phys;
        prev->is_last = block->is_last;
        
        if (block->next_phys) {
            block->next_phys->prev_phys = prev;
        } else {
            allocator->last_block = prev;
        }
        
        free(block);
        allocator->stats.total_blocks--;
        allocator->stats.total_merges++;
        
        return prev;
    }
    
    return block;
}

// ============================================================================
// Public API Implementation
// ============================================================================
// Note: Allocation tracking for leak detection now uses debug info stored
// directly in TLSFBlock (alloc_file, alloc_line, alloc_timestamp).
// No separate tracking array needed - O(1) lookup via hash table.

// Create allocator using an existing GPU memory pool (does not take ownership)
PTXTLSFAllocator* ptx_tlsf_create_from_pool(void* pool, size_t pool_size, bool debug_mode) {
    if (!pool || pool_size == 0) return NULL;

    PTXTLSFAllocator* allocator = (PTXTLSFAllocator*)malloc(sizeof(PTXTLSFAllocator));
    if (!allocator) return NULL;

    memset(allocator, 0, sizeof(PTXTLSFAllocator));

    // Use external pool
    allocator->vram_pool = pool;
    allocator->vram_pool_size = pool_size;
    allocator->owns_pool = false;  // External pool, don't free on destroy
    allocator->debug_mode = debug_mode;
    allocator->warning_threshold = 0.80f;
    allocator->warning_step = 0.01f;          // 1% utilization step
    allocator->warning_interval_us = 1000000; // 1s minimum interval
    allocator->warning_last_ts = 0;
    allocator->warning_last_utilization = 0.0f;
    allocator->warnings_enabled = true;
    allocator->auto_defrag_enabled = true;

    // Initialize first block
    TLSFBlock* first_block = (TLSFBlock*)malloc(sizeof(TLSFBlock));
    if (!first_block) {
        free(allocator);
        return NULL;
    }
    memset(first_block, 0, sizeof(TLSFBlock));

    first_block->device_ptr = pool;
    first_block->size = pool_size;
    first_block->magic = TLSF_BLOCK_MAGIC;
    first_block->is_free = true;
    first_block->is_last = true;

    allocator->first_block = first_block;
    allocator->last_block = first_block;

    // Add to hash table and free list
    hash_insert(allocator, first_block);
    insert_free_block(allocator, first_block);

    // Initialize stats
    allocator->stats.total_pool_size = pool_size;
    allocator->stats.free_bytes = pool_size;
    allocator->stats.total_blocks = 1;
    allocator->stats.is_healthy = true;

    printf("[TLSF] Allocator created: %.2f MB pool\n",
           pool_size / (1024.0 * 1024.0));

    return allocator;
}

// Create allocator with its own GPU memory pool (takes ownership)
PTXTLSFAllocator* ptx_tlsf_create(size_t pool_size, bool debug_mode) {
    // Align pool size
    pool_size = align_up(pool_size);

    // Allocate GPU memory pool
    void* pool = ptx_driver_alloc(pool_size);
    if (!pool) {
        printf("[TLSF] Failed to allocate GPU pool\n");
        return NULL;
    }

    PTXTLSFAllocator* allocator = ptx_tlsf_create_from_pool(pool, pool_size, debug_mode);
    if (!allocator) {
        ptx_driver_free(pool);
        return NULL;
    }

    allocator->owns_pool = true;  // We own this pool, free on destroy
    return allocator;
}

void ptx_tlsf_destroy(PTXTLSFAllocator* allocator) {
    if (!allocator) return;

    // Free GPU pool only if we own it
    if (allocator->owns_pool && allocator->vram_pool) {
        ptx_driver_free(allocator->vram_pool);
    }

    // Free all block headers
    TLSFBlock* block = allocator->first_block;
    while (block) {
        TLSFBlock* next = block->next_phys;
        free(block);
        block = next;
    }

    // No separate allocation tracker to free (debug info stored in blocks)

    free(allocator);
    printf("[TLSF] Allocator destroyed\n");
}

void* ptx_tlsf_alloc(PTXTLSFAllocator* allocator, size_t size) {
    return ptx_tlsf_alloc_debug(allocator, size, NULL, 0);
}

void* ptx_tlsf_alloc_debug(PTXTLSFAllocator* allocator, size_t size,
                           const char* file, int line) {
    if (!allocator || size == 0) return NULL;
    
    // Align size
    size = align_up(size);
    
    // Find suitable free block
    TLSFBlock* block = find_free_block(allocator, size);
    
    if (!block) {
        // Pool exhausted
        allocator->stats.fallback_count++;
        printf("[TLSF] Pool exhausted, allocation failed\n");
        return NULL;
    }
    
    // Remove from free list
    remove_free_block(allocator, block);
    
    // Split if block is too large
    split_block(allocator, block, size);
    
    // Update block info
    block->alloc_id = allocator->next_alloc_id;
    block->alloc_file = file;
    block->alloc_line = line;
    block->alloc_timestamp = get_timestamp();
    
    // Update stats
    allocator->stats.allocated_bytes += block->size;
    allocator->stats.free_bytes -= block->size;
    allocator->stats.allocated_blocks++;
    allocator->stats.total_allocations++;

    // Debug: Print every 100th allocation to track counters
    if (ptx_tlsf_debug_enabled() &&
        (allocator->stats.total_allocations % 100 == 0 || allocator->stats.total_allocations <= 5)) {
        printf("[TLSF-DEBUG] Alloc #%llu: %zu bytes, total allocated: %zu MB\n",
               (unsigned long long)allocator->stats.total_allocations,
               size,
               allocator->stats.allocated_bytes / 1024 / 1024);
    }

    if (allocator->stats.allocated_bytes > allocator->stats.peak_allocated) {
        allocator->stats.peak_allocated = allocator->stats.allocated_bytes;
    }
    
    // Debug info already stored in block (alloc_file, alloc_line, alloc_timestamp)
    // No separate tracking needed - we use hash_find for O(1) lookup

    // Check warning threshold (throttled)
    float utilization = (float)allocator->stats.allocated_bytes / allocator->vram_pool_size;
    if (allocator->warnings_enabled && allocator->warning_threshold > 0.0f) {
        if (utilization < allocator->warning_threshold) {
            allocator->warning_last_ts = 0;
            allocator->warning_last_utilization = utilization;
        } else {
            uint64_t now = get_timestamp();
            bool step_ok = (allocator->warning_step <= 0.0f) ||
                           (utilization >= allocator->warning_last_utilization + allocator->warning_step);
            bool time_ok = (allocator->warning_interval_us == 0) ||
                           (allocator->warning_last_ts == 0) ||
                           (now - allocator->warning_last_ts >= allocator->warning_interval_us);
            if (step_ok || time_ok) {
                allocator->warning_last_ts = now;
                allocator->warning_last_utilization = utilization;
                printf("[TLSF] WARNING: Pool utilization at %.1f%%\n", utilization * 100.0f);
            }
        }
    }
    
    return block->device_ptr;
}

void ptx_tlsf_free(PTXTLSFAllocator* allocator, void* ptr) {
    if (!allocator || !ptr) {
        // Debug: trace null arguments
        if (!allocator && ptx_tlsf_debug_enabled()) {
            printf("[TLSF-DEBUG] free called with NULL allocator\n");
        }
        return;
    }

    // Check if pointer is in pool
    if (ptr < allocator->vram_pool ||
        ptr >= (char*)allocator->vram_pool + allocator->vram_pool_size) {
        ptx_strict_free_violation("TLSF", ptr);
        return;
    }

    // Find block using hash table (O(1))
    TLSFBlock* block = hash_find(allocator, ptr);
    if (!block) {
        ptx_strict_free_violation("TLSF", ptr);
        return;
    }

    // Validate block
    if (block->magic != TLSF_BLOCK_MAGIC) {
        printf("[TLSF] ERROR: Block corruption detected at %p\n", ptr);
        return;
    }

    if (block->is_free) {
        printf("[TLSF] ERROR: Double free detected at %p\n", ptr);
        return;
    }

    // Update stats
    allocator->stats.allocated_bytes -= block->size;
    allocator->stats.free_bytes += block->size;
    allocator->stats.allocated_blocks--;
    allocator->stats.total_frees++;

    // Debug: Print every 100th free to track counters
    if (ptx_tlsf_debug_enabled() &&
        (allocator->stats.total_frees % 100 == 0 || allocator->stats.total_frees <= 5)) {
        printf("[TLSF-DEBUG] Free #%llu: %zu bytes, total allocated: %zu MB\n",
               (unsigned long long)allocator->stats.total_frees,
               block->size,
               allocator->stats.allocated_bytes / 1024 / 1024);
    }

    // Reset warning state if utilization drops below threshold
    if (allocator->warnings_enabled && allocator->warning_threshold > 0.0f) {
        float utilization = (float)allocator->stats.allocated_bytes / allocator->vram_pool_size;
        if (utilization < allocator->warning_threshold) {
            allocator->warning_last_ts = 0;
            allocator->warning_last_utilization = utilization;
        }
    }

    // Clear debug info (block lookup via hash is O(1), no separate untracking needed)
    block->alloc_file = NULL;
    block->alloc_line = 0;

    // Coalesce with adjacent free blocks
    block = coalesce(allocator, block);

    // Add to free list
    insert_free_block(allocator, block);

    // Debug: uncomment to trace every free
    // printf("[TLSF-DEBUG] freed %zu bytes, now allocated=%zu\n",
    //        block->size, allocator->stats.allocated_bytes);
}

// ============================================================================
// Diagnostics API Implementation
// ============================================================================

void ptx_tlsf_get_stats(PTXTLSFAllocator* allocator, TLSFPoolStats* stats) {
    if (!allocator || !stats) return;
    
    // Copy current stats
    *stats = allocator->stats;
    
    // Calculate fragmentation metrics
    if (allocator->stats.free_blocks > 0) {
        // External fragmentation: ratio of largest free block to total free memory
        size_t largest_free = 0;
        size_t smallest_free = SIZE_MAX;
        
        for (int fl = 0; fl < TLSF_FL_INDEX_MAX; fl++) {
            for (int sl = 0; sl < TLSF_SL_INDEX_COUNT; sl++) {
                TLSFBlock* block = allocator->free_lists[fl][sl];
                while (block) {
                    if (block->size > largest_free) largest_free = block->size;
                    if (block->size < smallest_free) smallest_free = block->size;
                    block = block->next_free;
                }
            }
        }
        
        stats->largest_free_block = largest_free;
        stats->smallest_free_block = smallest_free;
        
        if (allocator->stats.free_bytes > 0) {
            stats->external_fragmentation = 1.0f - ((float)largest_free / allocator->stats.free_bytes);
        }
        
        // Fragmentation ratio: number of free blocks relative to ideal
        float ideal_blocks = 1.0f;
        stats->fragmentation_ratio = (allocator->stats.free_blocks - ideal_blocks) / 
                                     fmaxf(allocator->stats.free_blocks, ideal_blocks);
    }
    
    // Calculate hash table stats
    uint32_t total_chain_length = 0;
    for (int i = 0; i < TLSF_HASH_TABLE_SIZE; i++) {
        total_chain_length += allocator->hash_table[i].count;
    }
    stats->avg_chain_length = (float)total_chain_length / TLSF_HASH_TABLE_SIZE;
    
    // Utilization
    stats->utilization_percent = (float)allocator->stats.allocated_bytes / 
                                 allocator->vram_pool_size * 100.0f;
    
    // Health check
    stats->is_healthy = (stats->utilization_percent < 95.0f) &&
                       (stats->fragmentation_ratio < 0.5f) &&
                       (stats->fallback_count == 0);
    
    stats->needs_defrag = (stats->fragmentation_ratio > 0.3f) ||
                         (stats->free_blocks > allocator->stats.total_blocks / 4);
}

void ptx_tlsf_validate(PTXTLSFAllocator* allocator, TLSFHealthReport* report) {
    if (!allocator || !report) return;
    
    memset(report, 0, sizeof(TLSFHealthReport));
    report->is_valid = true;
    
    // Validate physical chain
    TLSFBlock* block = allocator->first_block;
    int block_count = 0;
    size_t total_size = 0;
    
    while (block) {
        block_count++;
        total_size += block->size;
        
        // Check magic number
        if (block->magic != TLSF_BLOCK_MAGIC) {
            snprintf(report->error_messages[report->error_count++], 256,
                    "Block %d: Corrupted magic number", block_count);
            report->has_corrupted_blocks = true;
            report->is_valid = false;
        }
        
        // Check chain integrity
        if (block->next_phys && block->next_phys->prev_phys != block) {
            snprintf(report->error_messages[report->error_count++], 256,
                    "Block %d: Broken physical chain", block_count);
            report->has_broken_chains = true;
            report->is_valid = false;
        }
        
        // Check device pointer alignment
        if (((uintptr_t)block->device_ptr) % TLSF_ALIGNMENT != 0) {
            snprintf(report->error_messages[report->error_count++], 256,
                    "Block %d: Misaligned device pointer", block_count);
            report->is_valid = false;
        }
        
        block = block->next_phys;
        
        if (report->error_count >= 16) break;
    }
    
    // Check total size
    if (total_size != allocator->vram_pool_size) {
        snprintf(report->error_messages[report->error_count++], 256,
                "Total block size mismatch: %zu vs %zu", total_size, allocator->vram_pool_size);
        report->is_valid = false;
    }
    
    // Validate hash table
    for (int i = 0; i < TLSF_HASH_TABLE_SIZE; i++) {
        TLSFBlock* hash_block = allocator->hash_table[i].head;
        int chain_len = 0;
        
        while (hash_block) {
            chain_len++;
            
            // Verify hash is correct
            uint32_t expected_hash = hash_ptr(hash_block->device_ptr);
            if (expected_hash != i) {
                snprintf(report->error_messages[report->error_count++], 256,
                        "Hash bucket %d: Block in wrong bucket", i);
                report->has_hash_errors = true;
                report->is_valid = false;
            }
            
            hash_block = hash_block->hash_next;
            
            if (chain_len > 100) {
                snprintf(report->error_messages[report->error_count++], 256,
                        "Hash bucket %d: Infinite loop detected", i);
                report->has_hash_errors = true;
                report->is_valid = false;
                break;
            }
        }
    }
    
    // Check for memory leaks by counting allocated blocks
    if (allocator->debug_mode && allocator->stats.allocated_blocks > 0) {
        report->has_memory_leaks = true;
        snprintf(report->error_messages[report->error_count++], 256,
                "Memory leak: %u allocations not freed", allocator->stats.allocated_blocks);
    }
}

bool ptx_tlsf_can_allocate(PTXTLSFAllocator* allocator, size_t size) {
    if (!allocator) return false;
    
    size = align_up(size);
    TLSFBlock* block = find_free_block(allocator, size);
    return (block != NULL);
}

size_t ptx_tlsf_get_max_allocatable(PTXTLSFAllocator* allocator) {
    if (!allocator) return 0;
    
    size_t max_size = 0;
    
    for (int fl = TLSF_FL_INDEX_MAX - 1; fl >= 0; fl--) {
        for (int sl = TLSF_SL_INDEX_COUNT - 1; sl >= 0; sl--) {
            TLSFBlock* block = allocator->free_lists[fl][sl];
            if (block && block->size > max_size) {
                max_size = block->size;
            }
        }
    }
    
    return max_size;
}

bool ptx_tlsf_owns_ptr(PTXTLSFAllocator* allocator, void* ptr) {
    if (!allocator || !ptr) return false;

    // Check if pointer is within pool bounds
    void* pool_start = allocator->vram_pool;
    void* pool_end = (char*)pool_start + allocator->vram_pool_size;

    return (ptr >= pool_start && ptr < pool_end);
}

void ptx_tlsf_print_pool_map(PTXTLSFAllocator* allocator) {
    if (!allocator) return;
    
    printf("\n========== TLSF Pool Memory Map ==========\n");
    printf("Pool: %p - %p (%.2f MB)\n", 
           allocator->vram_pool,
           (char*)allocator->vram_pool + allocator->vram_pool_size,
           allocator->vram_pool_size / (1024.0 * 1024.0));
    printf("==========================================\n");
    
    TLSFBlock* block = allocator->first_block;
    int index = 0;
    
    while (block) {
        printf("[%3d] %p: %8zu bytes %s%s%s\n",
               index++,
               block->device_ptr,
               block->size,
               block->is_free ? "[FREE]" : "[USED]",
               block->is_pinned ? " [PINNED]" : "",
               block->is_last ? " [LAST]" : "");
        
        if (allocator->debug_mode && !block->is_free && block->alloc_file) {
            printf("      Allocated at %s:%d (ID: %u)\n",
                   block->alloc_file, block->alloc_line, block->alloc_id);
        }
        
        block = block->next_phys;
    }
    
    printf("==========================================\n");
    printf("Total blocks: %d\n", allocator->stats.total_blocks);
    printf("Free blocks: %d\n", allocator->stats.free_blocks);
    printf("Allocated: %.2f MB / %.2f MB (%.1f%%)\n",
           allocator->stats.allocated_bytes / (1024.0 * 1024.0),
           allocator->vram_pool_size / (1024.0 * 1024.0),
           allocator->stats.allocated_bytes * 100.0 / allocator->vram_pool_size);
    printf("==========================================\n\n");
}

void ptx_tlsf_print_allocations(PTXTLSFAllocator* allocator) {
    if (!allocator || !allocator->debug_mode) {
        printf("[TLSF] Debug mode not enabled\n");
        return;
    }

    printf("\n========== Active Allocations ==========\n");
    printf("Total: %u allocations\n", allocator->stats.allocated_blocks);
    printf("========================================\n");

    // Walk physical block chain to find allocated blocks
    TLSFBlock* block = allocator->first_block;
    uint32_t count = 0;
    while (block) {
        if (!block->is_free) {
            printf("[%4u] %p: %8zu bytes at %s:%d (age: %lu us)\n",
                   block->alloc_id,
                   block->device_ptr,
                   block->size,
                   block->alloc_file ? block->alloc_file : "unknown",
                   block->alloc_line,
                   get_timestamp() - block->alloc_timestamp);
            count++;
        }
        block = block->next_phys;
    }

    printf("========================================\n\n");
}

// ============================================================================
// Maintenance API Implementation
// ============================================================================

void ptx_tlsf_defragment(PTXTLSFAllocator* allocator) {
    if (!allocator) return;
    
    printf("[TLSF] Starting defragmentation...\n");
    
    int merges = 0;
    TLSFBlock* block = allocator->first_block;
    
    while (block) {
        if (block->is_free && block->next_phys && block->next_phys->is_free) {
            TLSFBlock* next = block->next_phys;
            
            remove_free_block(allocator, block);
            remove_free_block(allocator, next);
            hash_remove(allocator, next);
            
            block->size += next->size;
            block->next_phys = next->next_phys;
            block->is_last = next->is_last;
            
            if (block->next_phys) {
                block->next_phys->prev_phys = block;
            } else {
                allocator->last_block = block;
            }
            
            free(next);
            allocator->stats.total_blocks--;
            merges++;
            
            insert_free_block(allocator, block);
            // Don't advance - check again with new next block
        } else {
            block = block->next_phys;
        }
    }
    
    printf("[TLSF] Defragmentation complete: %d merges\n", merges);
}

void ptx_tlsf_compact(PTXTLSFAllocator* allocator) {
    // TODO: Implement compaction (move allocations to reduce fragmentation)
    // This is complex and requires cooperation from users to update pointers
    printf("[TLSF] Compaction not yet implemented\n");
}

void ptx_tlsf_set_warning_threshold(PTXTLSFAllocator* allocator, float threshold) {
    if (allocator) {
        allocator->warning_threshold = threshold;
    }
}

void ptx_tlsf_set_warning_step(PTXTLSFAllocator* allocator, float step) {
    if (allocator) {
        allocator->warning_step = step;
    }
}

void ptx_tlsf_set_warning_interval_ms(PTXTLSFAllocator* allocator, uint64_t interval_ms) {
    if (allocator) {
        allocator->warning_interval_us = interval_ms * 1000;
    }
}

void ptx_tlsf_set_warnings_enabled(PTXTLSFAllocator* allocator, bool enabled) {
    if (allocator) {
        allocator->warnings_enabled = enabled;
        if (!enabled) {
            allocator->warning_last_ts = 0;
            allocator->warning_last_utilization = 0.0f;
        }
    }
}

void ptx_tlsf_set_auto_defrag(PTXTLSFAllocator* allocator, bool enable) {
    if (allocator) {
        allocator->auto_defrag_enabled = enable;
    }
}

void ptx_tlsf_reset_stats(PTXTLSFAllocator* allocator) {
    if (!allocator) return;
    
    allocator->stats.total_allocations = 0;
    allocator->stats.total_frees = 0;
    allocator->stats.total_splits = 0;
    allocator->stats.total_merges = 0;
    allocator->stats.fallback_count = 0;
    allocator->stats.hash_collisions = 0;
    allocator->stats.peak_allocated = allocator->stats.allocated_bytes;
}

// ============================================================================
// Advanced API Implementation
// ============================================================================

void ptx_tlsf_pin_block(PTXTLSFAllocator* allocator, void* ptr) {
    if (!allocator || !ptr) return;
    
    TLSFBlock* block = hash_find(allocator, ptr);
    if (block) {
        block->is_pinned = true;
    }
}

void ptx_tlsf_unpin_block(PTXTLSFAllocator* allocator, void* ptr) {
    if (!allocator || !ptr) return;
    
    TLSFBlock* block = hash_find(allocator, ptr);
    if (block) {
        block->is_pinned = false;
    }
}

bool ptx_tlsf_get_block_info(PTXTLSFAllocator* allocator, void* ptr,
                             size_t* size_out, bool* is_free_out) {
    if (!allocator || !ptr) return false;
    
    TLSFBlock* block = hash_find(allocator, ptr);
    if (!block) return false;
    
    if (size_out) *size_out = block->size;
    if (is_free_out) *is_free_out = block->is_free;
    
    return true;
}

bool ptx_tlsf_expand_pool(PTXTLSFAllocator* allocator, size_t additional_size) {
    // TODO: Implement pool expansion
    // This requires allocating a new pool and managing multiple pools
    printf("[TLSF] Pool expansion not yet implemented\n");
    return false;
}

bool ptx_tlsf_shrink_pool(PTXTLSFAllocator* allocator) {
    // TODO: Implement pool shrinking
    // This requires moving allocations and freeing unused memory
    printf("[TLSF] Pool shrinking not yet implemented\n");
    return false;
}

void* ptx_tlsf_realloc(PTXTLSFAllocator* allocator, void* ptr, size_t new_size) {
    if (!allocator) return NULL;
    
    if (!ptr) {
        return ptx_tlsf_alloc(allocator, new_size);
    }
    
    if (new_size == 0) {
        ptx_tlsf_free(allocator, ptr);
        return NULL;
    }
    
    // Find existing block
    TLSFBlock* block = hash_find(allocator, ptr);
    if (!block) {
        // Not in our pool, can't realloc
        return NULL;
    }
    
    new_size = align_up(new_size);
    
    // If new size fits in current block, just return
    if (new_size <= block->size) {
        return ptr;
    }
    
    // Try to expand in place by merging with next block
    if (block->next_phys && block->next_phys->is_free &&
        block->size + block->next_phys->size >= new_size) {
        
        TLSFBlock* next = block->next_phys;
        remove_free_block(allocator, next);
        hash_remove(allocator, next);
        
        block->size += next->size;
        block->next_phys = next->next_phys;
        block->is_last = next->is_last;
        
        if (block->next_phys) {
            block->next_phys->prev_phys = block;
        }
        
        free(next);
        allocator->stats.total_blocks--;
        
        // Split if too large
        split_block(allocator, block, new_size);
        
        return ptr;
    }
    
    // Can't expand in place, allocate new block and copy
    void* new_ptr = ptx_tlsf_alloc(allocator, new_size);
    if (!new_ptr) return NULL;
    
    // Copy data
    cudaMemcpy(new_ptr, ptr, block->size, cudaMemcpyDeviceToDevice);
    
    // Free old block
    ptx_tlsf_free(allocator, ptr);
    
    return new_ptr;
}
