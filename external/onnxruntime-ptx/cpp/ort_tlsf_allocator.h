// ORT TLSF Allocator Interface
//
// This provides a custom ONNX Runtime GPU allocator that uses PTX-OS TLSF
// instead of the default CUDA allocator.
//
// ORT Allocator Interface:
// - Alloc(size) -> void*
// - Free(ptr)
// - Info() -> const OrtMemoryInfo*

#ifndef ORT_TLSF_ALLOCATOR_H
#define ORT_TLSF_ALLOCATOR_H

#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// Rust FFI functions (implemented in Rust)
// These call into PTX-OS TLSF allocator
void ort_tlsf_init(int device_id);
void* ort_tlsf_alloc(size_t size);
void ort_tlsf_free(void* ptr);
void ort_tlsf_print_stats();

// C++ ORT Allocator wrapper
class OrtTLSFAllocator {
public:
    OrtTLSFAllocator(int device_id);
    ~OrtTLSFAllocator();

    // ORT allocator interface
    void* Alloc(size_t size);
    void Free(void* ptr);
    const char* Name() const { return "ORT-TLSF"; }

    // Stats
    void PrintStats();

private:
    int device_id_;
    bool initialized_;
};

// C interface for creating/destroying the global allocator instance
void ort_tlsf_create_allocator(int device_id);
void ort_tlsf_destroy_allocator();

// C interface for direct alloc/free through the global instance
void* ort_tlsf_allocator_alloc(size_t size);
void ort_tlsf_allocator_free(void* ptr);

#ifdef __cplusplus
}
#endif

#endif // ORT_TLSF_ALLOCATOR_H
