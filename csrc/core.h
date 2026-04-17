#pragma once
#include <sys/types.h>
#include <cstdint>
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

struct DiskPrefetchState;

struct AllocationMetadata {
    size_t size;
    CUdevice device;
    std::string tag;
    AllocationState state;
    bool enable_cpu_backup;
    void* cpu_backup;
    std::string disk_backup_path;
    uint64_t disk_backup_offset;

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

struct SharedArtifactHostMapping {
    std::string shm_name;
    std::string completion_token;
    void* mapping_base = nullptr;
    void* payload_base = nullptr;
    size_t artifact_size = 0;
    size_t mapped_size = 0;
    size_t payload_offset = 0;
    size_t block_payload_bytes = 0;
    size_t block_count = 0;
    int shm_fd = -1;
    bool cuda_registered = false;
    std::vector<uint8_t> registered_blocks;
};

struct SharedRingGlobalHeader {
    char magic[8];
    uint32_t version;
    uint32_t header_bytes;
    uint32_t block_header_bytes;
    uint32_t block_count;
    uint64_t block_payload_bytes;
    uint64_t payload_offset;
    uint64_t total_buffer_bytes;
    uint64_t artifact_size;
    uint64_t completion_generation;
    uint32_t producer_done;
    uint32_t consumer_state;
    uint32_t error_code;
    uint32_t reserved;
};

struct SharedRingBlockHeader {
    uint32_t state;
    uint32_t reserved0;
    uint64_t file_offset;
    uint64_t valid_bytes;
    uint64_t sequence;
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
        const std::string& disk_backup_path
    );
    cudaError_t free(void *ptr);

    void pause(const std::string& tag);
    void resume(const std::string& tag);
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
    std::unordered_map<std::string, SharedArtifactHostMapping> shared_artifact_mappings_;
    std::unordered_map<std::string, std::shared_ptr<DiskPrefetchState>> disk_prefetch_states_;
    std::atomic<uint64_t> memory_margin_bytes_ = 0;
};
