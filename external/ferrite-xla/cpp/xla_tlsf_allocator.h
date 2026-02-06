// XLA TLSF Allocator Interface
//
// This provides a custom XLA GPU allocator that uses PTX-OS TLSF
// instead of the default BFC (Best-Fit with Coalescing) allocator.
//
// XLA Allocator Interface:
// - Allocate(size, alignment) -> void*
// - Deallocate(ptr, size)
// - Name() -> string

#ifndef XLA_TLSF_ALLOCATOR_H
#define XLA_TLSF_ALLOCATOR_H

#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// Rust FFI functions (implemented in Rust)
// These call into PTX-OS TLSF allocator
void* tlsf_alloc_ffi(size_t size);
void tlsf_free_ffi(void* ptr);
void tlsf_init_ffi(int device_id);
void tlsf_print_stats_ffi();

// C++ XLA Allocator wrapper
class XLATLSFAllocator {
public:
    XLATLSFAllocator(int device_id);
    ~XLATLSFAllocator();

    // XLA allocator interface
    void* Allocate(size_t size, size_t alignment = 256);
    void Deallocate(void* ptr, size_t size);
    const char* Name() const { return "TLSF"; }

    // Stats
    void PrintStats();

private:
    int device_id_;
    bool initialized_;
};

// C interface for easier FFI from Rust
void* xla_tlsf_alloc(size_t size, size_t alignment);
void xla_tlsf_free(void* ptr, size_t size);
void xla_tlsf_init(int device_id);
void xla_tlsf_stats();

#ifdef __cplusplus
}
#endif

#endif // XLA_TLSF_ALLOCATOR_H
