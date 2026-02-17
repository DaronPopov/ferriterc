// Complete PyTorch TLSF Allocator Implementation
// Implements ALL required methods from c10::cuda::CUDACachingAllocator::CUDAAllocator
// Compatible with PyTorch 2.9.x (Jetson AI Lab wheel)

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Exception.h>
#include <cuda_runtime.h>
#include <iostream>
#include <unordered_map>
#include <mutex>
#include <memory>

// Rust FFI functions
extern "C" {
    void tlsf_init_ffi(int device_id, size_t block_size, double pool_fraction);
    void* tlsf_alloc_ffi(size_t size);
    void tlsf_free_ffi(void* ptr);
    void tlsf_print_stats_ffi();
}

namespace c10 {
namespace cuda {
namespace CUDACachingAllocator {

class TLSFAllocator : public CUDAAllocator {
private:
    int device_id_;
    bool initialized_;
    bool enabled_;
    std::mutex mutex_;
    std::unordered_map<void*, size_t> allocations_;
    size_t peak_allocated_;
    size_t current_allocated_;

public:
    TLSFAllocator() : device_id_(0), initialized_(false), enabled_(true),
                       peak_allocated_(0), current_allocated_(0) {}

    void* raw_alloc(size_t nbytes) override {
        void* ptr = tlsf_alloc_ffi(nbytes);
        if (ptr) {
            std::lock_guard<std::mutex> lock(mutex_);
            allocations_[ptr] = nbytes;
            current_allocated_ += nbytes;
            if (current_allocated_ > peak_allocated_) {
                peak_allocated_ = current_allocated_;
            }
        }
        return ptr;
    }

    void* raw_alloc_with_stream(size_t nbytes, cudaStream_t stream) override {
        (void)stream;
        return raw_alloc(nbytes);
    }

    void raw_delete(void* ptr) override {
        if (!ptr) return;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = allocations_.find(ptr);
            if (it != allocations_.end()) {
                current_allocated_ -= it->second;
                allocations_.erase(it);
            }
        }
        tlsf_free_ffi(ptr);
    }

    void init(int /*device_count*/) override {
        if (!initialized_) {
            device_id_ = 0;
            tlsf_init_ffi(device_id_, 2 * 1024 * 1024, 0.70);
            initialized_ = true;
            std::cout << "[aten-ptx] TLSF allocator initialized" << std::endl;
        }
    }

    bool initialized() override {
        return initialized_;
    }

    void setMemoryFraction(double fraction, c10::DeviceIndex device) override {
        (void)fraction;
        (void)device;
    }

    void emptyCache(
        c10::DeviceIndex device = -1,
        MempoolId_t mempool_id = {0, 0}) override {
        (void)device;
        (void)mempool_id;
    }

    void cacheInfo(c10::DeviceIndex device, size_t* largestBlock) override {
        (void)device;
        *largestBlock = 0;
    }

    void* getBaseAllocation(void* ptr, size_t* size) override {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = allocations_.find(ptr);
        if (it != allocations_.end()) {
            *size = it->second;
            return ptr;
        }
        *size = 0;
        return nullptr;
    }

    void recordStream(const DataPtr& ptr, CUDAStream stream) override {
        (void)ptr;
        (void)stream;
    }

    DeviceStats getDeviceStats(c10::DeviceIndex device) override {
        (void)device;
        DeviceStats stats;

        std::lock_guard<std::mutex> lock(mutex_);

        Stat allocated_stat;
        allocated_stat.current = current_allocated_;
        allocated_stat.peak = peak_allocated_;

        Stat zero_stat;
        zero_stat.current = 0;
        zero_stat.peak = 0;

        stats.allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)] = allocated_stat;
        stats.allocated_bytes[static_cast<size_t>(StatType::SMALL_POOL)] = zero_stat;
        stats.allocated_bytes[static_cast<size_t>(StatType::LARGE_POOL)] = zero_stat;

        stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)] = allocated_stat;
        stats.reserved_bytes[static_cast<size_t>(StatType::SMALL_POOL)] = zero_stat;
        stats.reserved_bytes[static_cast<size_t>(StatType::LARGE_POOL)] = zero_stat;

        stats.active_bytes[static_cast<size_t>(StatType::AGGREGATE)] = allocated_stat;
        stats.active_bytes[static_cast<size_t>(StatType::SMALL_POOL)] = zero_stat;
        stats.active_bytes[static_cast<size_t>(StatType::LARGE_POOL)] = zero_stat;

        stats.inactive_split_bytes[static_cast<size_t>(StatType::AGGREGATE)] = zero_stat;
        stats.inactive_split_bytes[static_cast<size_t>(StatType::SMALL_POOL)] = zero_stat;
        stats.inactive_split_bytes[static_cast<size_t>(StatType::LARGE_POOL)] = zero_stat;

        stats.num_alloc_retries = 0;
        stats.num_ooms = 0;

        return stats;
    }

    void resetAccumulatedStats(c10::DeviceIndex device) override {
        (void)device;
    }

    void resetPeakStats(c10::DeviceIndex device) override {
        (void)device;
        std::lock_guard<std::mutex> lock(mutex_);
        peak_allocated_ = current_allocated_;
    }

    SnapshotInfo snapshot(
        c10::DeviceIndex device = -1,
        MempoolId_t mempool_id = {0, 0}) override {
        (void)device;
        (void)mempool_id;
        return SnapshotInfo();
    }

    void beginAllocateToPool(
        c10::DeviceIndex device,
        MempoolId_t mempool_id,
        std::function<bool(cudaStream_t)> filter) override {
        (void)device;
        (void)mempool_id;
        (void)filter;
    }

    void endAllocateToPool(c10::DeviceIndex device, MempoolId_t mempool_id) override {
        (void)device;
        (void)mempool_id;
    }

    void releasePool(c10::DeviceIndex device, MempoolId_t mempool_id) override {
        (void)device;
        (void)mempool_id;
    }

    int getPoolUseCount(
        c10::DeviceIndex device,
        MempoolId_t mempool_id = {0, 0}) override {
        (void)device;
        (void)mempool_id;
        return 0;
    }

    std::shared_ptr<void> getIpcDevPtr(std::string handle) override {
        (void)handle;
        return nullptr;
    }

    void recordHistory(
        bool enabled,
        CreateContextFn context_recorder,
        size_t alloc_trace_max_entries,
        RecordContext when) override {
        (void)enabled;
        (void)context_recorder;
        (void)alloc_trace_max_entries;
        (void)when;
    }

    void attachOutOfMemoryObserver(OutOfMemoryObserver observer) override {
        (void)observer;
    }

    void attachAllocatorTraceTracker(AllocatorTraceTracker tracker) override {
        (void)tracker;
    }

    void enablePeerAccess(c10::DeviceIndex dev, c10::DeviceIndex dev_to_access) override {
        (void)dev;
        (void)dev_to_access;
    }

    cudaError_t memcpyAsync(
        void* dst,
        int dstDevice,
        const void* src,
        int srcDevice,
        size_t count,
        cudaStream_t stream,
        bool p2p_enabled) override {
        (void)dstDevice;
        (void)srcDevice;
        (void)p2p_enabled;
        return cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToDevice, stream);
    }

    std::shared_ptr<AllocatorState> getCheckpointState(
        c10::DeviceIndex device,
        MempoolId_t id) override {
        (void)device;
        (void)id;
        return nullptr;
    }

    CheckpointDelta setCheckpointPoolState(
        c10::DeviceIndex device,
        std::shared_ptr<AllocatorState> pps) override {
        (void)device;
        (void)pps;
        return CheckpointDelta();
    }

    std::string name() override {
        return "TLSF";
    }

    DeleterFnPtr raw_deleter() const override {
        return &tlsf_free_ffi;
    }

    DataPtr allocate(size_t nbytes) override {
        void* ptr = raw_alloc(nbytes);
        if (!ptr) {
            return DataPtr(nullptr, Device(DeviceType::CUDA, device_id_));
        }
        return DataPtr(
            ptr,
            ptr,
            raw_deleter(),
            Device(DeviceType::CUDA, device_id_)
        );
    }

    void copy_data(void* dest, const void* src, std::size_t count) const override {
        cudaError_t err = cudaMemcpy(dest, src, count, cudaMemcpyDeviceToDevice);
        if (err != cudaSuccess) {
            std::cerr << "[aten-ptx] cudaMemcpy failed: " << cudaGetErrorString(err) << std::endl;
        }
    }
};

// Global instance
static TLSFAllocator* g_tlsf_allocator = nullptr;

} // namespace CUDACachingAllocator
} // namespace cuda
} // namespace c10

// C interface
extern "C" {
    void aten_tlsf_init(int /*device_id*/) {
        using namespace c10::cuda::CUDACachingAllocator;

        if (!g_tlsf_allocator) {
            g_tlsf_allocator = new TLSFAllocator();
            g_tlsf_allocator->init(1);

            allocator.store(g_tlsf_allocator, std::memory_order_release);

            std::cout << "[aten-ptx] TLSF allocator registered!" << std::endl;
            std::cout << "[aten-ptx] All PyTorch CUDA ops now use TLSF!" << std::endl;
        }
    }

    void aten_tlsf_stats() {
        tlsf_print_stats_ffi();
    }

    void aten_set_cuda_stream(void* raw_stream, int device_id) {
        auto stream = c10::cuda::getStreamFromExternal(
            static_cast<cudaStream_t>(raw_stream),
            static_cast<c10::DeviceIndex>(device_id)
        );
        c10::cuda::setCurrentCUDAStream(stream);
    }

    void aten_reset_default_stream(int device_id) {
        auto stream = c10::cuda::getDefaultCUDAStream(
            static_cast<c10::DeviceIndex>(device_id)
        );
        c10::cuda::setCurrentCUDAStream(stream);
    }
}
