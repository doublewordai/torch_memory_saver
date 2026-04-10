#pragma once

#include "core.h"

#include <condition_variable>
#include <memory>
#include <thread>
#include <unordered_map>
#include <vector>

enum class DiskPrefetchSlotState {
    FREE,
    READING,
    READY,
    DISPATCHING,
    IN_FLIGHT
};

struct DiskPrefetchSlot {
    void* host_buffer = nullptr;
    uint64_t file_offset = 0;
    size_t aligned_bytes = 0;
    size_t valid_bytes = 0;
    std::vector<std::pair<int, cudaEvent_t>> completion_events;
    DiskPrefetchSlotState state = DiskPrefetchSlotState::FREE;
};

struct DiskPrefetchState {
    std::string path;
    uint64_t total_size = 0;
    uint64_t padded_total_size = 0;
    size_t ring_slot_bytes = 0;
    size_t read_thread_count = 1;
    bool use_direct_io = true;
    void* mapped_artifact = nullptr;
    int buffered_fd = -1;
    std::vector<DiskPrefetchSlot> slots;
    std::mutex mutex;
    std::condition_variable cv;
    std::vector<std::thread> producers;
    std::thread recycler;
    uint64_t next_read_offset = 0;
    size_t in_flight_reads = 0;
    bool producer_done = false;
    bool producer_started = false;
    bool recycler_started = false;
    bool stop_requested = false;
};

struct AllocationRef {
    void* ptr;
    AllocationMetadata* metadata;
};

namespace DiskOffload {

uint64_t invalid_disk_backup_offset();
bool has_disk_backup(const AllocationMetadata& metadata);
uint64_t total_size(const std::vector<AllocationRef>& items);

void materialize_artifact(const std::vector<AllocationRef>& items);
std::shared_ptr<DiskPrefetchState> create_prefetch_state(const std::string& path, uint64_t total_size);
void start_prefetch_if_needed(const std::shared_ptr<DiskPrefetchState>& state);
void destroy_prefetch_state(const std::shared_ptr<DiskPrefetchState>& state);
std::shared_ptr<DiskPrefetchState> pop_prefetch_state(
    std::unordered_map<std::string, std::shared_ptr<DiskPrefetchState>>& disk_prefetch_states,
    const std::string& path
);

}  // namespace DiskOffload
