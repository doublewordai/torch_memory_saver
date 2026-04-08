#pragma once
#include <sys/types.h>
#include <stdio.h>
#include <unordered_map>
#include <atomic>
#include <mutex>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include <condition_variable>
#include "utils.h"
#include "macro.h"

#if TMS_ROCM_LEGACY_CHUNKED
#include "hardware_amd_support.h"
#endif

enum class AllocationState {
    // Memory is mapped and accessible
    ACTIVE,
    // Memory is unmapped and inaccessible
    PAUSED
};

enum class ArtifactBackend {
    NONE,
    RAM,
    DISK
};

struct AllocationMetadata {
    size_t size;
    CUdevice device;
    std::string tag;
    AllocationState state;
    bool enable_cpu_backup;
    void* cpu_backup;
    ArtifactBackend artifact_backend;
    std::string artifact_path;
    uint64_t artifact_offset;

#if TMS_ROCM_LEGACY_CHUNKED
    // ROCm 6.x: Chunked allocation workaround
    size_t aligned_size;
    std::vector<CUmemGenericAllocationHandle> allocHandles;
    std::vector<size_t> chunk_sizes;
#else
    // CUDA and ROCm 7.0+: Single allocation handle
    CUmemGenericAllocationHandle allocHandle;
#endif
};

struct DiskArtifactHostCache {
    void* host_buffer = nullptr;
    size_t size = 0;
};

enum class DiskPrefetchSlotState {
    FREE,
    READING,
    READY
};

struct DiskPrefetchSlot {
    void* host_buffer = nullptr;
    uint64_t file_offset = 0;
    size_t aligned_bytes = 0;
    size_t valid_bytes = 0;
    DiskPrefetchSlotState state = DiskPrefetchSlotState::FREE;
};

struct DiskPrefetchState {
    std::string path;
    uint64_t total_size = 0;
    uint64_t padded_total_size = 0;
    size_t ring_slot_bytes = 0;
    std::vector<DiskPrefetchSlot> slots;
    std::mutex mutex;
    std::condition_variable cv;
    std::thread producer;
    uint64_t next_read_offset = 0;
    bool producer_done = false;
    bool producer_started = false;
    bool stop_requested = false;
};

class TorchMemorySaver {
public:
    static TorchMemorySaver& instance();

    cudaError_t malloc(
        void** ptr,
        CUdevice device,
        size_t size,
        const std::string& tag,
        bool enable_cpu_backup,
        ArtifactBackend artifact_backend,
        const std::string& artifact_path
    );
    cudaError_t free(void *ptr);

    void pause(const std::string& tag);
    void resume(const std::string& tag);
    void preload(const std::string& tag);
    void set_memory_margin_bytes(uint64_t value) {
        memory_margin_bytes_.store(value);
    }
    uint8_t* get_cpu_backup_pointer(const uint8_t* query_gpu_ptr, uint64_t query_size);

private:
    TorchMemorySaver();
    ~TorchMemorySaver() = default;
    TorchMemorySaver(const TorchMemorySaver&) = delete;
    TorchMemorySaver& operator=(const TorchMemorySaver&) = delete;

    std::mutex allocator_metadata_mutex_;
    std::unordered_map<void*, AllocationMetadata> allocation_metadata_;
    std::unordered_map<std::string, DiskArtifactHostCache> disk_artifact_host_cache_;
    std::unordered_map<std::string, std::shared_ptr<DiskPrefetchState>> disk_prefetch_states_;
    std::atomic<uint64_t> memory_margin_bytes_ = 0;
};
