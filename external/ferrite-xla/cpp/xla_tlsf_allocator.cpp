// XLA TLSF Allocator Implementation

#include "xla_tlsf_allocator.h"
#include <iostream>
#include <cstring>

XLATLSFAllocator::XLATLSFAllocator(int device_id)
    : device_id_(device_id), initialized_(false) {
    tlsf_init_ffi(device_id);
    initialized_ = true;
    std::cout << "[XLA-TLSF] Allocator initialized on device " << device_id << std::endl;
}

XLATLSFAllocator::~XLATLSFAllocator() {
    if (initialized_) {
        std::cout << "[XLA-TLSF] Allocator destroyed" << std::endl;
    }
}

void* XLATLSFAllocator::Allocate(size_t size, size_t alignment) {
    if (!initialized_) {
        std::cerr << "[XLA-TLSF] ERROR: Allocator not initialized!" << std::endl;
        return nullptr;
    }

    // For now, ignore alignment (TLSF handles alignment internally)
    // In production, we'd need to handle alignment properly
    (void)alignment;  // Suppress unused parameter warning
    void* ptr = tlsf_alloc_ffi(size);

    if (!ptr) {
        std::cerr << "[XLA-TLSF] Failed to allocate " << size << " bytes" << std::endl;
        return nullptr;
    }

    return ptr;
}

void XLATLSFAllocator::Deallocate(void* ptr, size_t size) {
    if (!initialized_) {
        std::cerr << "[XLA-TLSF] ERROR: Allocator not initialized!" << std::endl;
        return;
    }

    if (!ptr) {
        return; // Null pointer, nothing to free
    }

    (void)size;  // TLSF doesn't need size to free, suppress warning
    tlsf_free_ffi(ptr);
}

void XLATLSFAllocator::PrintStats() {
    tlsf_print_stats_ffi();
}

// C interface implementations
static XLATLSFAllocator* global_allocator = nullptr;

void xla_tlsf_init(int device_id) {
    if (!global_allocator) {
        global_allocator = new XLATLSFAllocator(device_id);
    }
}

void* xla_tlsf_alloc(size_t size, size_t alignment) {
    if (!global_allocator) {
        std::cerr << "[XLA-TLSF] ERROR: Must call xla_tlsf_init first!" << std::endl;
        return nullptr;
    }
    return global_allocator->Allocate(size, alignment);
}

void xla_tlsf_free(void* ptr, size_t size) {
    if (!global_allocator) {
        std::cerr << "[XLA-TLSF] ERROR: Allocator not initialized!" << std::endl;
        return;
    }
    global_allocator->Deallocate(ptr, size);
}

void xla_tlsf_stats() {
    if (global_allocator) {
        global_allocator->PrintStats();
    }
}
