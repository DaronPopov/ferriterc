// ORT TLSF Allocator Implementation

#include "ort_tlsf_allocator.h"
#include <iostream>

OrtTLSFAllocator::OrtTLSFAllocator(int device_id)
    : device_id_(device_id), initialized_(false) {
    ort_tlsf_init(device_id);
    initialized_ = true;
    std::cout << "[ORT-TLSF] Allocator initialized on device " << device_id << std::endl;
}

OrtTLSFAllocator::~OrtTLSFAllocator() {
    if (initialized_) {
        std::cout << "[ORT-TLSF] Allocator destroyed" << std::endl;
    }
}

void* OrtTLSFAllocator::Alloc(size_t size) {
    if (!initialized_) {
        std::cerr << "[ORT-TLSF] ERROR: Allocator not initialized!" << std::endl;
        return nullptr;
    }

    void* ptr = ort_tlsf_alloc(size);

    if (!ptr) {
        std::cerr << "[ORT-TLSF] Failed to allocate " << size << " bytes" << std::endl;
        return nullptr;
    }

    return ptr;
}

void OrtTLSFAllocator::Free(void* ptr) {
    if (!initialized_) {
        std::cerr << "[ORT-TLSF] ERROR: Allocator not initialized!" << std::endl;
        return;
    }

    if (!ptr) {
        return;
    }

    ort_tlsf_free(ptr);
}

void OrtTLSFAllocator::PrintStats() {
    ort_tlsf_print_stats();
}

// Global allocator instance
static OrtTLSFAllocator* global_allocator = nullptr;

void ort_tlsf_create_allocator(int device_id) {
    if (!global_allocator) {
        global_allocator = new OrtTLSFAllocator(device_id);
    }
}

void ort_tlsf_destroy_allocator() {
    if (global_allocator) {
        delete global_allocator;
        global_allocator = nullptr;
    }
}

void* ort_tlsf_allocator_alloc(size_t size) {
    if (!global_allocator) {
        std::cerr << "[ORT-TLSF] ERROR: Must call ort_tlsf_create_allocator first!" << std::endl;
        return nullptr;
    }
    return global_allocator->Alloc(size);
}

void ort_tlsf_allocator_free(void* ptr) {
    if (!global_allocator) {
        std::cerr << "[ORT-TLSF] ERROR: Allocator not initialized!" << std::endl;
        return;
    }
    global_allocator->Free(ptr);
}
