#include "core.h"
#include "utils.h"
#include "macro.h"
#include "api_forwarder.h"

#include <algorithm>
#include <cerrno>
#include <condition_variable>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <thread>
#include <unistd.h>
#include <unordered_map>

namespace {

constexpr uint64_t kInvalidArtifactOffset = std::numeric_limits<uint64_t>::max();
constexpr uint64_t kDirectIoAlignmentBytes = 4096;

uint64_t align_up(uint64_t value, uint64_t alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
}

bool matches_tag(const std::string& filter_tag, const AllocationMetadata& metadata) {
    return filter_tag.empty() || metadata.tag == filter_tag;
}

bool uses_artifact_backend(ArtifactBackend artifact_backend) {
    return artifact_backend != ArtifactBackend::NONE;
}

bool keep_host_backup_after_resume(const AllocationMetadata& metadata) {
    return metadata.artifact_backend == ArtifactBackend::RAM;
}

bool should_return_host_backup(const AllocationMetadata& metadata) {
    return metadata.enable_cpu_backup || metadata.artifact_backend == ArtifactBackend::RAM || metadata.cpu_backup != nullptr;
}

struct BatchMemcpyGroup {
    std::vector<void*> dsts;
    std::vector<void*> srcs;
    std::vector<size_t> sizes;
};

struct AllocationRef {
    void* ptr;
    AllocationMetadata* metadata;
};

uint64_t artifact_total_size(const std::vector<AllocationRef>& items) {
    uint64_t total_size = 0;
    for (const AllocationRef& item : items) {
        total_size = std::max(total_size, item.metadata->artifact_offset + static_cast<uint64_t>(item.metadata->size));
    }
    return total_size;
}

void add_batch_copy(
    std::unordered_map<int, BatchMemcpyGroup>& groups,
    CUdevice device,
    void* dst,
    void* src,
    size_t size
) {
    BatchMemcpyGroup& group = groups[static_cast<int>(device)];
    group.dsts.push_back(dst);
    group.srcs.push_back(src);
    group.sizes.push_back(size);
}

void run_batch_memcpy(
    std::unordered_map<int, BatchMemcpyGroup>& groups,
    cudaMemcpyKind fallback_kind
) {
    if (groups.empty()) {
        return;
    }

    int original_device = 0;
    CUDA_ERROR_CHECK(cudaGetDevice(&original_device));

    std::unordered_map<int, cudaStream_t> streams_by_device;
    for (const auto& entry : groups) {
        const int device = entry.first;
        CUDA_ERROR_CHECK(cudaSetDevice(device));
        cudaStream_t stream;
        CUDA_ERROR_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        streams_by_device.emplace(device, stream);
    }

    for (auto& entry : groups) {
        const int device = entry.first;
        BatchMemcpyGroup& group = entry.second;
        if (group.dsts.empty()) {
            continue;
        }

        CUDA_ERROR_CHECK(cudaSetDevice(device));
        cudaStream_t stream = streams_by_device.at(device);

#if defined(USE_CUDA)
        cudaMemcpyAttributes attrs{};
        attrs.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
        size_t attrs_idx = 0;
#if CUDART_VERSION >= 13000
        CUDA_ERROR_CHECK(cudaMemcpyBatchAsync(
            group.dsts.data(),
            group.srcs.data(),
            group.sizes.data(),
            group.dsts.size(),
            &attrs,
            &attrs_idx,
            1,
            stream
        ));
#else
        size_t fail_idx = SIZE_MAX;
        CUDA_ERROR_CHECK(cudaMemcpyBatchAsync(
            group.dsts.data(),
            group.srcs.data(),
            group.sizes.data(),
            group.dsts.size(),
            &attrs,
            &attrs_idx,
            1,
            &fail_idx,
            stream
        ));
#endif
#else
        for (size_t i = 0; i < group.dsts.size(); ++i) {
            CUDA_ERROR_CHECK(cudaMemcpyAsync(
                group.dsts[i],
                group.srcs[i],
                group.sizes[i],
                fallback_kind,
                stream
            ));
        }
#endif
    }

    for (const auto& entry : streams_by_device) {
        CUDA_ERROR_CHECK(cudaSetDevice(entry.first));
        CUDA_ERROR_CHECK(cudaStreamSynchronize(entry.second));
        CUDA_ERROR_CHECK(cudaStreamDestroy(entry.second));
    }

    CUDA_ERROR_CHECK(cudaSetDevice(original_device));
}

void pwrite_all(int fd, const void* buf, size_t size, uint64_t offset) {
    const uint8_t* cursor = static_cast<const uint8_t*>(buf);
    size_t remaining = size;
    uint64_t current_offset = offset;

    while (remaining > 0) {
        const ssize_t written = pwrite(fd, cursor, remaining, static_cast<off_t>(current_offset));
        if (written < 0) {
            if (errno == EINTR) {
                continue;
            }
            std::cerr << "[torch_memory_saver.cpp] pwrite failed errno=" << errno
                      << " (" << std::strerror(errno) << ")"
                      << " file=" << __FILE__ << " func=" << __func__ << " line=" << __LINE__
                      << std::endl;
            exit(1);
        }
        cursor += written;
        remaining -= static_cast<size_t>(written);
        current_offset += static_cast<uint64_t>(written);
    }
}

void pread_all(int fd, void* buf, size_t size, uint64_t offset) {
    uint8_t* cursor = static_cast<uint8_t*>(buf);
    size_t remaining = size;
    uint64_t current_offset = offset;

    while (remaining > 0) {
        const ssize_t read_bytes = pread(fd, cursor, remaining, static_cast<off_t>(current_offset));
        if (read_bytes < 0) {
            if (errno == EINTR) {
                continue;
            }
            std::cerr << "[torch_memory_saver.cpp] pread failed errno=" << errno
                      << " (" << std::strerror(errno) << ")"
                      << " file=" << __FILE__ << " func=" << __func__ << " line=" << __LINE__
                      << std::endl;
            exit(1);
        }
        SIMPLE_CHECK(read_bytes > 0, "Unexpected EOF while reading artifact");
        cursor += read_bytes;
        remaining -= static_cast<size_t>(read_bytes);
        current_offset += static_cast<uint64_t>(read_bytes);
    }
}

void materialize_disk_artifact(const std::vector<AllocationRef>& items) {
    SIMPLE_CHECK(!items.empty(), "materialize_disk_artifact expects non-empty items");

    std::vector<AllocationRef> ordered = items;
    std::sort(ordered.begin(), ordered.end(), [](const AllocationRef& lhs, const AllocationRef& rhs) {
        return lhs.ptr < rhs.ptr;
    });

    const std::string& path = ordered.front().metadata->artifact_path;
    SIMPLE_CHECK(!path.empty(), "artifact_backend=disk requires artifact_path");

    const int fd = open(path.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0644);
    if (fd < 0) {
        std::cerr << "[torch_memory_saver.cpp] open for write failed path=" << path
                  << " errno=" << errno << " (" << std::strerror(errno) << ")"
                  << " file=" << __FILE__ << " func=" << __func__ << " line=" << __LINE__
                  << std::endl;
        exit(1);
    }

    uint64_t offset = 0;
    (void) posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);
    for (const AllocationRef& item : ordered) {
        AllocationMetadata& metadata = *item.metadata;
        SIMPLE_CHECK(metadata.cpu_backup != nullptr, "Disk artifact write requires a host backup");
        metadata.artifact_offset = offset;
        pwrite_all(fd, metadata.cpu_backup, metadata.size, offset);
        offset += metadata.size;
    }

    const uint64_t padded_offset = align_up(offset, kDirectIoAlignmentBytes);
    if (padded_offset != offset) {
        uint8_t padding[kDirectIoAlignmentBytes] = {};
        pwrite_all(fd, padding, padded_offset - offset, offset);
    }

    (void) fdatasync(fd);
    (void) posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);

    if (close(fd) != 0) {
        std::cerr << "[torch_memory_saver.cpp] close after write failed path=" << path
                  << " errno=" << errno << " (" << std::strerror(errno) << ")"
                  << " file=" << __FILE__ << " func=" << __func__ << " line=" << __LINE__
                  << std::endl;
        exit(1);
    }
}

std::shared_ptr<DiskPrefetchState> create_disk_prefetch_state(const std::string& path, uint64_t total_size) {
    SIMPLE_CHECK(!path.empty(), "artifact_backend=disk requires artifact_path");

    const size_t configured_chunk_bytes = static_cast<size_t>(get_uint64_env_var(
        "TMS_DISK_RING_BUFFER_BYTES",
        128ull * 1024ull * 1024ull
    ));
    const size_t ring_slot_bytes = static_cast<size_t>(align_up(
        std::max<uint64_t>(configured_chunk_bytes, kDirectIoAlignmentBytes),
        kDirectIoAlignmentBytes
    ));
    const size_t ring_slot_count = std::max<size_t>(2, static_cast<size_t>(get_uint64_env_var("TMS_DISK_RING_SLOTS", 4)));

    auto state = std::make_shared<DiskPrefetchState>();
    state->path = path;
    state->total_size = total_size;
    state->padded_total_size = align_up(total_size, kDirectIoAlignmentBytes);
    state->ring_slot_bytes = ring_slot_bytes;
    state->slots.resize(ring_slot_count);

    for (DiskPrefetchSlot& slot : state->slots) {
        CUDA_ERROR_CHECK(cudaMallocHost(&slot.host_buffer, ring_slot_bytes));
    }

    return state;
}

void start_disk_prefetch_if_needed(const std::shared_ptr<DiskPrefetchState>& state) {
    {
        std::lock_guard<std::mutex> lock(state->mutex);
        if (state->producer_started) {
            return;
        }
        state->producer_started = true;
    }

    std::cout << "[torch_memory_saver.cpp] disk prefetch started"
              << " path=" << state->path
              << " bytes=" << state->total_size
              << " ring_slot_bytes=" << state->ring_slot_bytes
              << " ring_slots=" << state->slots.size()
              << std::endl;

    state->producer = std::thread([state]() {
        const int fd = open(state->path.c_str(), O_RDONLY | O_DIRECT);
        if (fd < 0) {
            std::cerr << "[torch_memory_saver.cpp] open for read failed path=" << state->path
                      << " errno=" << errno << " (" << std::strerror(errno) << ")"
                      << " file=" << __FILE__ << " func=" << __func__ << " line=" << __LINE__
                      << std::endl;
            exit(1);
        }

        (void) posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);
        (void) posix_fadvise(fd, 0, 0, POSIX_FADV_WILLNEED);

        while (true) {
            size_t slot_index = 0;
            uint64_t file_offset = 0;
            size_t aligned_bytes = 0;
            size_t valid_bytes = 0;
            {
                std::unique_lock<std::mutex> lock(state->mutex);
                state->cv.wait(lock, [&]() {
                    if (state->stop_requested || state->next_read_offset >= state->padded_total_size) {
                        return true;
                    }
                    for (const DiskPrefetchSlot& slot : state->slots) {
                        if (slot.state == DiskPrefetchSlotState::FREE) {
                            return true;
                        }
                    }
                    return false;
                });

                if (state->stop_requested) {
                    break;
                }

                if (state->next_read_offset >= state->padded_total_size) {
                    state->producer_done = true;
                    state->cv.notify_all();
                    break;
                }

                while (slot_index < state->slots.size() && state->slots[slot_index].state != DiskPrefetchSlotState::FREE) {
                    ++slot_index;
                }
                SIMPLE_CHECK(slot_index < state->slots.size(), "Expected a free disk prefetch slot");

                DiskPrefetchSlot& slot = state->slots[slot_index];
                file_offset = state->next_read_offset;
                aligned_bytes = static_cast<size_t>(std::min<uint64_t>(
                    state->ring_slot_bytes,
                    state->padded_total_size - file_offset
                ));
                valid_bytes = static_cast<size_t>(std::min<uint64_t>(
                    aligned_bytes,
                    state->total_size - file_offset
                ));
                slot.file_offset = file_offset;
                slot.aligned_bytes = aligned_bytes;
                slot.valid_bytes = valid_bytes;
                slot.state = DiskPrefetchSlotState::READING;
                state->next_read_offset += aligned_bytes;
            }

            pread_all(fd, state->slots[slot_index].host_buffer, aligned_bytes, file_offset);

            {
                std::lock_guard<std::mutex> lock(state->mutex);
                state->slots[slot_index].state = DiskPrefetchSlotState::READY;
            }
            state->cv.notify_all();
        }

        (void) posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
        if (close(fd) != 0) {
            std::cerr << "[torch_memory_saver.cpp] close after read failed path=" << state->path
                      << " errno=" << errno << " (" << std::strerror(errno) << ")"
                      << " file=" << __FILE__ << " func=" << __func__ << " line=" << __LINE__
                      << std::endl;
            exit(1);
        }

        if (!state->stop_requested && state->producer_done) {
            std::cout << "[torch_memory_saver.cpp] disk prefetch complete"
                      << " path=" << state->path
                      << " bytes=" << state->total_size
                      << std::endl;
        }
    });
}

void destroy_disk_prefetch_state(const std::shared_ptr<DiskPrefetchState>& state) {
    if (state == nullptr) {
        return;
    }

    {
        std::lock_guard<std::mutex> lock(state->mutex);
        state->stop_requested = true;
    }
    state->cv.notify_all();

    if (state->producer.joinable()) {
        state->producer.join();
    }

    for (DiskPrefetchSlot& slot : state->slots) {
        if (slot.host_buffer != nullptr) {
            CUDA_ERROR_CHECK(cudaFreeHost(slot.host_buffer));
            slot.host_buffer = nullptr;
        }
    }
}

std::shared_ptr<DiskPrefetchState> pop_disk_prefetch_state(
    std::unordered_map<std::string, std::shared_ptr<DiskPrefetchState>>& disk_prefetch_states,
    const std::string& path
) {
    auto it = disk_prefetch_states.find(path);
    if (it == disk_prefetch_states.end()) {
        return nullptr;
    }
    auto state = it->second;
    disk_prefetch_states.erase(it);
    return state;
}

// Collect H2D copy entries from a single slot into an existing batch group.
// Advances allocation_index/allocation_offset as it maps allocations to the slot.
void collect_slot_copies(
    DiskPrefetchSlot& slot,
    std::vector<AllocationRef>& path_items,
    size_t& allocation_index,
    size_t& allocation_offset,
    std::unordered_map<int, BatchMemcpyGroup>& h2d_groups
) {
    const uint64_t slot_data_begin = slot.file_offset;
    const uint64_t slot_data_end = slot_data_begin + slot.valid_bytes;

    while (allocation_index < path_items.size()) {
        AllocationMetadata& metadata = *path_items[allocation_index].metadata;
        const uint64_t allocation_file_offset = metadata.artifact_offset + allocation_offset;
        if (allocation_file_offset >= slot_data_end) {
            break;
        }

        const size_t src_offset = static_cast<size_t>(allocation_file_offset - slot_data_begin);
        const size_t copy_bytes = static_cast<size_t>(std::min<uint64_t>(
            slot_data_end - allocation_file_offset,
            metadata.size - allocation_offset
        ));
        add_batch_copy(
            h2d_groups,
            metadata.device,
            static_cast<uint8_t*>(path_items[allocation_index].ptr) + allocation_offset,
            static_cast<uint8_t*>(slot.host_buffer) + src_offset,
            copy_bytes
        );

        allocation_offset += copy_bytes;
        if (allocation_offset == metadata.size) {
            allocation_offset = 0;
            ++allocation_index;
        }
    }
}

// Consume prefetched disk data to GPU. Takes an external h2d_groups map so that
// ready slots can be coalesced with other H2D copies (e.g. RAM-backed allocations)
// into a single batched dispatch by the caller.
//
// Returns the file offset up to which ready slots were drained into h2d_groups.
// The caller is responsible for calling run_batch_memcpy on h2d_groups, then
// calling consume_disk_prefetch_remaining() to handle any slots the producer
// hasn't filled yet.
uint64_t consume_disk_prefetch_ready(
    const std::shared_ptr<DiskPrefetchState>& state,
    std::vector<AllocationRef>& path_items,
    size_t& allocation_index,
    size_t& allocation_offset,
    std::unordered_map<int, BatchMemcpyGroup>& h2d_groups,
    std::vector<size_t>& consumed_slot_indices
) {
    uint64_t expected_file_offset = 0;

    std::lock_guard<std::mutex> lock(state->mutex);
    while (expected_file_offset < state->padded_total_size) {
        size_t slot_index = SIZE_MAX;
        for (size_t i = 0; i < state->slots.size(); ++i) {
            if (state->slots[i].state == DiskPrefetchSlotState::READY &&
                state->slots[i].file_offset == expected_file_offset) {
                slot_index = i;
                break;
            }
        }
        if (slot_index == SIZE_MAX) {
            break;  // Next slot not ready — stop collecting.
        }

        DiskPrefetchSlot& slot = state->slots[slot_index];
        collect_slot_copies(slot, path_items, allocation_index, allocation_offset, h2d_groups);
        consumed_slot_indices.push_back(slot_index);
        expected_file_offset += slot.aligned_bytes;
    }

    return expected_file_offset;
}

// Handle the remaining slots that weren't ready during the initial drain.
// Called after the caller has dispatched and synced the coalesced batch.
void consume_disk_prefetch_remaining(
    const std::shared_ptr<DiskPrefetchState>& state,
    std::vector<AllocationRef>& path_items,
    size_t& allocation_index,
    size_t& allocation_offset,
    uint64_t expected_file_offset,
    std::vector<size_t>& consumed_slot_indices
) {
    while (expected_file_offset < state->padded_total_size) {
        size_t slot_index = 0;
        {
            std::unique_lock<std::mutex> lock(state->mutex);
            state->cv.wait(lock, [&]() {
                for (size_t i = 0; i < state->slots.size(); ++i) {
                    if (state->slots[i].state == DiskPrefetchSlotState::READY &&
                        state->slots[i].file_offset == expected_file_offset) {
                        return true;
                    }
                }
                return state->producer_done || state->stop_requested;
            });

            while (slot_index < state->slots.size()) {
                if (state->slots[slot_index].state == DiskPrefetchSlotState::READY &&
                    state->slots[slot_index].file_offset == expected_file_offset) {
                    break;
                }
                ++slot_index;
            }
            SIMPLE_CHECK(slot_index < state->slots.size(), "Expected a ready disk prefetch slot");
        }

        DiskPrefetchSlot& slot = state->slots[slot_index];
        std::unordered_map<int, BatchMemcpyGroup> h2d_groups;
        collect_slot_copies(slot, path_items, allocation_index, allocation_offset, h2d_groups);
        run_batch_memcpy(h2d_groups, cudaMemcpyHostToDevice);

        {
            std::lock_guard<std::mutex> lock(state->mutex);
            slot.state = DiskPrefetchSlotState::FREE;
        }
        state->cv.notify_all();
        expected_file_offset += slot.aligned_bytes;
    }

}

}  // namespace

TorchMemorySaver::TorchMemorySaver() {}

TorchMemorySaver &TorchMemorySaver::instance() {
    static TorchMemorySaver instance;
    return instance;
}

cudaError_t TorchMemorySaver::malloc(
    void **ptr,
    CUdevice device,
    size_t size,
    const std::string& tag,
    const bool enable_cpu_backup,
    const ArtifactBackend artifact_backend,
    const std::string& artifact_path
) {
#if TMS_ROCM_LEGACY_CHUNKED
    return ROCmHIPImplementation::rocm_malloc(ptr, device, size, tag, enable_cpu_backup, allocation_metadata_, allocator_metadata_mutex_);

#else
    const uint64_t memory_margin_bytes = memory_margin_bytes_.load();
    if (memory_margin_bytes > 0) {
        size_t free_bytes, total_bytes;
        CUDA_ERROR_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
        if (memory_margin_bytes + size > free_bytes) {
            std::cout << "[torch_memory_saver.cpp] TorchMemorySaver::malloc return OOM since"
                << " memory_margin_bytes=" << memory_margin_bytes
                << " (alloc)size=" << size
                << " free_bytes=" << free_bytes
                << std::endl;
            return cudaErrorMemoryAllocation;
        }
    }

    CUmemGenericAllocationHandle allocHandle;

    cudaError_t ret = CUDAUtils::cu_mem_create(&allocHandle, size, device);
    if (ret != cudaSuccess) {
        return ret;
    }

    CURESULT_CHECK(cuMemAddressReserve((CUdeviceptr *) ptr, size, 0, 0, 0));
    CURESULT_CHECK(cuMemMap((CUdeviceptr) * ptr, size, 0, allocHandle, 0));
    CUDAUtils::cu_mem_set_access(*ptr, size, device);

    {
        const std::lock_guard<std::mutex> lock(allocator_metadata_mutex_);
        allocation_metadata_.emplace(
            *ptr,
            AllocationMetadata{
                size,
                device,
                tag,
                AllocationState::ACTIVE,
                enable_cpu_backup,
                nullptr,
                artifact_backend,
                artifact_path,
                kInvalidArtifactOffset,
                allocHandle
            }
        );
    }

#ifdef TMS_DEBUG_LOG
    std::cout << "[torch_memory_saver.cpp] TorchMemorySaver.malloc "
              << " ptr=" << ptr << " *ptr=" << *ptr << " size=" << size
              << " allocHandle=" << allocHandle << " tag=" << tag
              << std::endl;
#endif

#endif
    return cudaSuccess;
}

cudaError_t TorchMemorySaver::free(void *ptr) {
#if TMS_ROCM_LEGACY_CHUNKED
    return ROCmHIPImplementation::rocm_free(ptr, allocation_metadata_, allocator_metadata_mutex_);

#else
    AllocationMetadata metadata;
    std::shared_ptr<DiskPrefetchState> prefetch_state_to_destroy = nullptr;
    {
        const std::lock_guard<std::mutex> lock(allocator_metadata_mutex_);
        if (allocation_metadata_.count(ptr) == 0) {
            return APIForwarder::call_real_cuda_free(ptr);
        }

        metadata = allocation_metadata_[ptr];
        allocation_metadata_.erase(ptr);
        if (metadata.artifact_backend == ArtifactBackend::DISK) {
            bool release_disk_state = true;
            for (const auto& entry : allocation_metadata_) {
                if (entry.second.artifact_backend == ArtifactBackend::DISK &&
                    entry.second.artifact_path == metadata.artifact_path) {
                    release_disk_state = false;
                    break;
                }
            }
            if (release_disk_state) {
                prefetch_state_to_destroy = pop_disk_prefetch_state(disk_prefetch_states_, metadata.artifact_path);
            }
        }
    }

    destroy_disk_prefetch_state(prefetch_state_to_destroy);

    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    if (metadata.state == AllocationState::ACTIVE) {
        CURESULT_CHECK(cuMemUnmap((CUdeviceptr) ptr, metadata.size));
        CURESULT_CHECK(cuMemRelease(metadata.allocHandle));
    }
    CURESULT_CHECK(cuMemAddressFree((CUdeviceptr) ptr, metadata.size));

    if (nullptr != metadata.cpu_backup) {
        CUDA_ERROR_CHECK(cudaFreeHost(metadata.cpu_backup));
        metadata.cpu_backup = nullptr;
    }

#ifdef TMS_DEBUG_LOG
    std::cout << "[torch_memory_saver.cpp] TorchMemorySaver.free "
              << " ptr=" << ptr << " metadata.size=" << metadata.size
              << " metadata.allocHandle=" << metadata.allocHandle << " tag=" << metadata.tag
              << std::endl;
#endif

#endif
    return cudaSuccess;
}

void TorchMemorySaver::pause(const std::string& tag) {
#if TMS_ROCM_LEGACY_CHUNKED
    ROCmHIPImplementation::rocm_pause(tag, allocation_metadata_, allocator_metadata_mutex_);

#else
    std::vector<AllocationRef> matching_items;
    std::unordered_map<std::string, std::vector<AllocationRef>> disk_items_by_path;
    std::unordered_map<std::string, bool> disk_path_needs_materialize;
    std::vector<std::shared_ptr<DiskPrefetchState>> prefetch_states_to_destroy;

    {
        const std::lock_guard<std::mutex> lock(allocator_metadata_mutex_);
        matching_items.reserve(allocation_metadata_.size());

        for (auto it = allocation_metadata_.begin(); it != allocation_metadata_.end(); ++it) {
            AllocationMetadata& metadata = it->second;
            if (!matches_tag(tag, metadata)) {
                continue;
            }

            if (metadata.state != AllocationState::ACTIVE) {
                std::cerr << "[torch_memory_saver.cpp] Cannot pause allocation that is not active."
                          << " tag=" << metadata.tag << " ptr=" << std::to_string((uintptr_t)it->first)
                          << " file=" << __FILE__ << " func=" << __func__ << " line=" << __LINE__
                          << std::endl;
                exit(1);
            }

            matching_items.push_back(AllocationRef{it->first, &metadata});
            if (metadata.artifact_backend == ArtifactBackend::DISK) {
                disk_items_by_path[metadata.artifact_path].push_back(AllocationRef{it->first, &metadata});
                if (metadata.artifact_offset == kInvalidArtifactOffset) {
                    disk_path_needs_materialize[metadata.artifact_path] = true;
                }
            }
        }

        std::unordered_map<int, BatchMemcpyGroup> d2h_groups;
        for (const AllocationRef& item : matching_items) {
            void* ptr = item.ptr;
            AllocationMetadata& metadata = *item.metadata;

            const bool needs_legacy_copy = metadata.enable_cpu_backup;
            const bool needs_ram_artifact_copy = metadata.artifact_backend == ArtifactBackend::RAM && metadata.cpu_backup == nullptr;
            const bool needs_disk_artifact_copy = metadata.artifact_backend == ArtifactBackend::DISK &&
                disk_path_needs_materialize[metadata.artifact_path];

            if (needs_legacy_copy || needs_ram_artifact_copy || needs_disk_artifact_copy) {
                if (metadata.cpu_backup == nullptr) {
                    CUDA_ERROR_CHECK(cudaMallocHost(&metadata.cpu_backup, metadata.size));
                }
                add_batch_copy(d2h_groups, metadata.device, metadata.cpu_backup, ptr, metadata.size);
            }
        }

        run_batch_memcpy(d2h_groups, cudaMemcpyDeviceToHost);

        for (auto& entry : disk_items_by_path) {
            if (!disk_path_needs_materialize[entry.first]) {
                continue;
            }
            materialize_disk_artifact(entry.second);
            for (const AllocationRef& item : entry.second) {
                AllocationMetadata& metadata = *item.metadata;
                if (metadata.cpu_backup != nullptr) {
                    CUDA_ERROR_CHECK(cudaFreeHost(metadata.cpu_backup));
                    metadata.cpu_backup = nullptr;
                }
            }
        }

        for (const AllocationRef& item : matching_items) {
            void *ptr = item.ptr;
            AllocationMetadata& metadata = *item.metadata;

            CURESULT_CHECK(cuMemUnmap((CUdeviceptr) ptr, metadata.size));
            CURESULT_CHECK(cuMemRelease(metadata.allocHandle));
            metadata.state = AllocationState::PAUSED;

#ifdef TMS_DEBUG_LOG
            std::cout << "[torch_memory_saver.cpp] TorchMemorySaver.pause"
                      << " ptr=" << ptr << " metadata.size=" << metadata.size << " metadata.allocHandle="
                      << metadata.allocHandle << " tag=" << metadata.tag << " filter_tag=" << tag
                      << " metadata.enable_cpu_backup=" << metadata.enable_cpu_backup
                      << std::endl;
#endif
        }

        for (const auto& entry : disk_items_by_path) {
            auto state = pop_disk_prefetch_state(disk_prefetch_states_, entry.first);
            if (state != nullptr) {
                prefetch_states_to_destroy.push_back(state);
            }
        }
    }

    for (const auto& state : prefetch_states_to_destroy) {
        destroy_disk_prefetch_state(state);
    }
#endif
}

void TorchMemorySaver::preload(const std::string& tag) {
#if TMS_ROCM_LEGACY_CHUNKED
    (void) tag;

#else
    std::vector<std::shared_ptr<DiskPrefetchState>> states_to_start;

    {
        const std::lock_guard<std::mutex> lock(allocator_metadata_mutex_);
        std::unordered_map<std::string, std::vector<AllocationRef>> disk_items_by_path;

        for (auto it = allocation_metadata_.begin(); it != allocation_metadata_.end(); ++it) {
            AllocationMetadata& metadata = it->second;
            if (!matches_tag(tag, metadata) || metadata.artifact_backend != ArtifactBackend::DISK) {
                continue;
            }

            if (metadata.state != AllocationState::PAUSED) {
                std::cerr << "[torch_memory_saver.cpp] Cannot preload disk artifact for allocation that is not paused."
                          << " tag=" << metadata.tag << " ptr=" << std::to_string((uintptr_t)it->first)
                          << " file=" << __FILE__ << " func=" << __func__ << " line=" << __LINE__
                          << std::endl;
                exit(1);
            }

            disk_items_by_path[metadata.artifact_path].push_back(AllocationRef{it->first, &metadata});
        }

        for (const auto& entry : disk_items_by_path) {
            const std::string& path = entry.first;
            auto state_it = disk_prefetch_states_.find(path);
            if (state_it == disk_prefetch_states_.end()) {
                state_it = disk_prefetch_states_.emplace(path, create_disk_prefetch_state(path, artifact_total_size(entry.second))).first;
            }
            states_to_start.push_back(state_it->second);
        }
    }

    for (const auto& state : states_to_start) {
        start_disk_prefetch_if_needed(state);
    }
#endif
}

void TorchMemorySaver::resume(const std::string& tag) {
#if TMS_ROCM_LEGACY_CHUNKED
    ROCmHIPImplementation::rocm_resume(tag, allocation_metadata_, allocator_metadata_mutex_);

#else
    std::vector<AllocationRef> matching_items;
    std::unordered_map<std::string, std::vector<AllocationRef>> disk_items_by_path;
    std::unordered_map<std::string, std::shared_ptr<DiskPrefetchState>> disk_prefetch_states_for_resume;
    std::unordered_map<int, BatchMemcpyGroup> h2d_groups;
    std::vector<std::shared_ptr<DiskPrefetchState>> states_to_destroy;

    {
        const std::lock_guard<std::mutex> lock(allocator_metadata_mutex_);
        matching_items.reserve(allocation_metadata_.size());

        for (auto it = allocation_metadata_.begin(); it != allocation_metadata_.end(); ++it) {
            AllocationMetadata &metadata = it->second;
            if (!matches_tag(tag, metadata)) {
                continue;
            }

            if (metadata.state != AllocationState::PAUSED) {
                std::cerr << "[torch_memory_saver.cpp] Cannot resume allocation that is not paused. "
                          << " tag=" << metadata.tag << " ptr=" << std::to_string((uintptr_t)it->first)
                          << " file=" << __FILE__ << " func=" << __func__ << " line=" << __LINE__
                          << std::endl;
                exit(1);
            }

            matching_items.push_back(AllocationRef{it->first, &metadata});
            if (metadata.artifact_backend == ArtifactBackend::DISK) {
                disk_items_by_path[metadata.artifact_path].push_back(AllocationRef{it->first, &metadata});
            }
        }

        for (auto& entry : disk_items_by_path) {
            const std::string& path = entry.first;
            auto state_it = disk_prefetch_states_.find(path);
            if (state_it == disk_prefetch_states_.end()) {
                state_it = disk_prefetch_states_.emplace(path, create_disk_prefetch_state(path, artifact_total_size(entry.second))).first;
            }
            disk_prefetch_states_for_resume.emplace(path, state_it->second);
        }
    }

    for (const auto& entry : disk_prefetch_states_for_resume) {
        start_disk_prefetch_if_needed(entry.second);
    }

    for (const AllocationRef& item : matching_items) {
        void *ptr = item.ptr;
        AllocationMetadata &metadata = *item.metadata;

        CUmemGenericAllocationHandle newAllocHandle;
        CUDA_ERROR_CHECK(CUDAUtils::cu_mem_create(&newAllocHandle, metadata.size, metadata.device));

        CURESULT_CHECK(cuMemMap((CUdeviceptr) ptr, metadata.size, 0, newAllocHandle, 0));
        CUDAUtils::cu_mem_set_access(ptr, metadata.size, metadata.device);

        if (metadata.enable_cpu_backup || metadata.artifact_backend == ArtifactBackend::RAM) {
            SIMPLE_CHECK(metadata.cpu_backup != nullptr, "Expected a host backup for RAM-backed resume");
            add_batch_copy(h2d_groups, metadata.device, ptr, metadata.cpu_backup, metadata.size);
        }

        metadata.state = AllocationState::ACTIVE;
        metadata.allocHandle = newAllocHandle;
    }

    // Phase 1: Drain any READY disk prefetch slots into the same h2d_groups batch.
    // This coalesces prefetched disk data with RAM-backed data into a single dispatch.
    struct DiskResumeState {
        std::shared_ptr<DiskPrefetchState> prefetch_state;
        std::vector<AllocationRef>* items;
        size_t allocation_index = 0;
        size_t allocation_offset = 0;
        uint64_t resume_offset = 0;
        std::vector<size_t> consumed_slot_indices;
    };
    std::vector<DiskResumeState> disk_resume_states;

    for (auto& entry : disk_items_by_path) {
        DiskResumeState drs;
        drs.prefetch_state = disk_prefetch_states_for_resume.at(entry.first);
        drs.items = &entry.second;

        std::sort(drs.items->begin(), drs.items->end(), [](const AllocationRef& lhs, const AllocationRef& rhs) {
            return lhs.metadata->artifact_offset < rhs.metadata->artifact_offset;
        });

        drs.resume_offset = consume_disk_prefetch_ready(
            drs.prefetch_state, *drs.items,
            drs.allocation_index, drs.allocation_offset,
            h2d_groups, drs.consumed_slot_indices
        );
        disk_resume_states.push_back(std::move(drs));
    }

    // Single batched dispatch for all RAM + ready disk data.
    run_batch_memcpy(h2d_groups, cudaMemcpyHostToDevice);

    // Free the ready slots now that run_batch_memcpy has finished reading from them.
    for (auto& drs : disk_resume_states) {
        if (!drs.consumed_slot_indices.empty()) {
            {
                std::lock_guard<std::mutex> lock(drs.prefetch_state->mutex);
                for (size_t idx : drs.consumed_slot_indices) {
                    drs.prefetch_state->slots[idx].state = DiskPrefetchSlotState::FREE;
                }
            }
            drs.prefetch_state->cv.notify_all();
            drs.consumed_slot_indices.clear();
        }
    }

    // Phase 2: Handle any remaining disk slots the producer hasn't filled yet.
    for (auto& drs : disk_resume_states) {
        consume_disk_prefetch_remaining(
            drs.prefetch_state, *drs.items,
            drs.allocation_index, drs.allocation_offset,
            drs.resume_offset, drs.consumed_slot_indices
        );
    }

    {
        const std::lock_guard<std::mutex> lock(allocator_metadata_mutex_);
        for (const AllocationRef& item : matching_items) {
            AllocationMetadata &metadata = *item.metadata;
            if (metadata.cpu_backup == nullptr) {
                continue;
            }

            if (metadata.enable_cpu_backup || !keep_host_backup_after_resume(metadata)) {
                CUDA_ERROR_CHECK(cudaFreeHost(metadata.cpu_backup));
                metadata.cpu_backup = nullptr;
            }
        }

        for (const auto& entry : disk_prefetch_states_for_resume) {
            auto state = pop_disk_prefetch_state(disk_prefetch_states_, entry.first);
            if (state != nullptr) {
                states_to_destroy.push_back(state);
            }
        }
    }

    // Destroy prefetch states in a detached thread to avoid blocking resume
    // on cudaFreeHost of pinned ring buffer memory.
    if (!states_to_destroy.empty()) {
        std::thread([states = std::move(states_to_destroy)]() {
            for (const auto& state : states) {
                destroy_disk_prefetch_state(state);
            }
        }).detach();
    }

#endif
}

uint8_t* TorchMemorySaver::get_cpu_backup_pointer(const uint8_t* query_gpu_ptr, uint64_t query_size) {
    const std::lock_guard<std::mutex> lock(allocator_metadata_mutex_);

    for (auto it = allocation_metadata_.begin(); it != allocation_metadata_.end(); ++it) {
        uint8_t *ptr = (uint8_t*) it->first;
        AllocationMetadata &metadata = it->second;

#if TMS_ROCM_LEGACY_CHUNKED
        size_t total_size = metadata.aligned_size;
#else
        size_t total_size = metadata.size;
#endif

        if ((ptr <= query_gpu_ptr) && (query_gpu_ptr + query_size <= ptr + total_size)) {
            const size_t offset = query_gpu_ptr - ptr;
            if (metadata.state == AllocationState::ACTIVE) {
                return nullptr;
            }
            if (metadata.artifact_backend == ArtifactBackend::DISK) {
                return nullptr;
            }
            if (metadata.cpu_backup == nullptr) {
                SIMPLE_CHECK(should_return_host_backup(metadata),
                    "get_cpu_backup_pointer: found paused allocation but cpu_backup does not exist");
            }
            return (metadata.cpu_backup == nullptr) ? nullptr : (uint8_t*) metadata.cpu_backup + offset;
        }
    }

    std::cerr << "[torch_memory_saver.cpp] get_cpu_backup_pointer fail to find backup "
              << " query_gpu_ptr=" << query_gpu_ptr << " query_size=" << query_size
              << std::endl;
    exit(1);
}
