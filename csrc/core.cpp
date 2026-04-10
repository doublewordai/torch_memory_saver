#include "core.h"
#include "disk_offload.h"
#include "utils.h"
#include "macro.h"
#include "api_forwarder.h"

#include <algorithm>
#include <thread>
#include <unordered_map>

namespace {

bool matches_tag(const std::string& filter_tag, const AllocationMetadata& metadata) {
    return filter_tag.empty() || metadata.tag == filter_tag;
}

bool should_return_host_backup(const AllocationMetadata& metadata) {
    return metadata.enable_cpu_backup || metadata.cpu_backup != nullptr;
}

struct BatchMemcpyGroup {
    std::vector<void*> dsts;
    std::vector<void*> srcs;
    std::vector<size_t> sizes;
};

struct AsyncMemcpyContext {
    int original_device = 0;
    std::unordered_map<int, cudaStream_t> streams_by_device;
};

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

AsyncMemcpyContext create_async_memcpy_context() {
    AsyncMemcpyContext context;
    CUDA_ERROR_CHECK(cudaGetDevice(&context.original_device));
    return context;
}

cudaStream_t get_async_stream_for_device(AsyncMemcpyContext& context, int device) {
    auto it = context.streams_by_device.find(device);
    if (it != context.streams_by_device.end()) {
        return it->second;
    }

    CUDA_ERROR_CHECK(cudaSetDevice(device));
    cudaStream_t stream;
    CUDA_ERROR_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    context.streams_by_device.emplace(device, stream);
    return stream;
}

std::vector<int> get_used_devices(const std::unordered_map<int, BatchMemcpyGroup>& groups) {
    std::vector<int> devices;
    devices.reserve(groups.size());
    for (const auto& entry : groups) {
        if (!entry.second.dsts.empty()) {
            devices.push_back(entry.first);
        }
    }
    return devices;
}

void enqueue_batch_memcpy_async(
    AsyncMemcpyContext& context,
    std::unordered_map<int, BatchMemcpyGroup>& groups,
    cudaMemcpyKind fallback_kind
) {
    if (groups.empty()) {
        return;
    }

    for (auto& entry : groups) {
        const int device = entry.first;
        BatchMemcpyGroup& group = entry.second;
        if (group.dsts.empty()) {
            continue;
        }

        CUDA_ERROR_CHECK(cudaSetDevice(device));
        cudaStream_t stream = get_async_stream_for_device(context, device);

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

    CUDA_ERROR_CHECK(cudaSetDevice(context.original_device));
}

void synchronize_and_destroy_async_memcpy_context(AsyncMemcpyContext& context) {
    for (const auto& entry : context.streams_by_device) {
        CUDA_ERROR_CHECK(cudaSetDevice(entry.first));
        CUDA_ERROR_CHECK(cudaStreamSynchronize(entry.second));
        CUDA_ERROR_CHECK(cudaStreamDestroy(entry.second));
    }
    context.streams_by_device.clear();
    CUDA_ERROR_CHECK(cudaSetDevice(context.original_device));
}

void record_disk_prefetch_slot_events(
    AsyncMemcpyContext& context,
    const std::shared_ptr<DiskPrefetchState>& state,
    size_t slot_index,
    const std::vector<int>& used_devices
) {
    SIMPLE_CHECK(!used_devices.empty(), "Expected at least one device when recording slot events");

    std::vector<std::pair<int, cudaEvent_t>> completion_events;
    completion_events.reserve(used_devices.size());
    for (const int device : used_devices) {
        CUDA_ERROR_CHECK(cudaSetDevice(device));
        cudaEvent_t event;
        CUDA_ERROR_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
        CUDA_ERROR_CHECK(cudaEventRecord(event, get_async_stream_for_device(context, device)));
        completion_events.emplace_back(device, event);
    }

    {
        std::lock_guard<std::mutex> lock(state->mutex);
        DiskPrefetchSlot& slot = state->slots[slot_index];
        SIMPLE_CHECK(slot.state == DiskPrefetchSlotState::DISPATCHING,
            "Expected disk prefetch slot to be dispatching when recording events");
        SIMPLE_CHECK(slot.completion_events.empty(),
            "Expected disk prefetch slot to not already have completion events");
        slot.completion_events = std::move(completion_events);
        slot.state = DiskPrefetchSlotState::IN_FLIGHT;
    }

    CUDA_ERROR_CHECK(cudaSetDevice(context.original_device));
    state->cv.notify_all();
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
        const uint64_t allocation_file_offset = metadata.disk_backup_offset + allocation_offset;
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

size_t find_ready_slot_locked(
    const std::shared_ptr<DiskPrefetchState>& state,
    uint64_t expected_file_offset
) {
    for (size_t i = 0; i < state->slots.size(); ++i) {
        if (state->slots[i].state == DiskPrefetchSlotState::READY &&
            state->slots[i].file_offset == expected_file_offset) {
            return i;
        }
    }
    return SIZE_MAX;
}

bool dispatch_ready_disk_slot_async(
    AsyncMemcpyContext& async_memcpy_context,
    const std::shared_ptr<DiskPrefetchState>& state,
    std::vector<AllocationRef>& path_items,
    size_t& allocation_index,
    size_t& allocation_offset,
    uint64_t& expected_file_offset
) {
    std::unordered_map<int, BatchMemcpyGroup> h2d_groups;
    std::vector<int> used_devices;
    size_t slot_index = SIZE_MAX;
    bool free_empty_slot = false;

    {
        std::lock_guard<std::mutex> lock(state->mutex);
        slot_index = find_ready_slot_locked(state, expected_file_offset);
        if (slot_index == SIZE_MAX) {
            return false;
        }

        DiskPrefetchSlot& slot = state->slots[slot_index];
        SIMPLE_CHECK(slot.state == DiskPrefetchSlotState::READY,
            "Expected disk prefetch slot to be ready before dispatch");
        slot.state = DiskPrefetchSlotState::DISPATCHING;
        SIMPLE_CHECK(slot.completion_events.empty(),
            "Expected disk prefetch slot to not already have completion events before dispatch");

        collect_slot_copies(slot, path_items, allocation_index, allocation_offset, h2d_groups);
        expected_file_offset += slot.aligned_bytes;
        used_devices = get_used_devices(h2d_groups);
        free_empty_slot = used_devices.empty();
        if (free_empty_slot) {
            slot.state = DiskPrefetchSlotState::FREE;
        }
    }

    if (free_empty_slot) {
        state->cv.notify_all();
        return true;
    }

    enqueue_batch_memcpy_async(async_memcpy_context, h2d_groups, cudaMemcpyHostToDevice);
    record_disk_prefetch_slot_events(async_memcpy_context, state, slot_index, used_devices);
    return true;
}

void drain_ready_disk_slots_async(
    AsyncMemcpyContext& async_memcpy_context,
    const std::shared_ptr<DiskPrefetchState>& state,
    std::vector<AllocationRef>& path_items,
    size_t& allocation_index,
    size_t& allocation_offset,
    uint64_t& expected_file_offset
) {
    while (dispatch_ready_disk_slot_async(
        async_memcpy_context,
        state,
        path_items,
        allocation_index,
        allocation_offset,
        expected_file_offset
    )) {
    }
}

void consume_disk_prefetch_remaining_async(
    AsyncMemcpyContext& async_memcpy_context,
    const std::shared_ptr<DiskPrefetchState>& state,
    std::vector<AllocationRef>& path_items,
    size_t& allocation_index,
    size_t& allocation_offset,
    uint64_t& expected_file_offset
) {
    while (expected_file_offset < state->padded_total_size) {
        if (dispatch_ready_disk_slot_async(
            async_memcpy_context,
            state,
            path_items,
            allocation_index,
            allocation_offset,
            expected_file_offset
        )) {
            continue;
        }

        std::unique_lock<std::mutex> lock(state->mutex);
        state->cv.wait(lock, [&]() {
            return find_ready_slot_locked(state, expected_file_offset) != SIZE_MAX ||
                state->producer_done || state->stop_requested;
        });
        SIMPLE_CHECK(!state->stop_requested, "Disk prefetch stopped before resume completed");
        SIMPLE_CHECK(
            find_ready_slot_locked(state, expected_file_offset) != SIZE_MAX || !state->producer_done,
            "Disk prefetch finished without producing the expected resume slot"
        );
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
    const std::string& disk_backup_path
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
                disk_backup_path,
                DiskOffload::invalid_disk_backup_offset(),
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
        if (DiskOffload::has_disk_backup(metadata)) {
            bool release_disk_state = true;
            for (const auto& entry : allocation_metadata_) {
                if (DiskOffload::has_disk_backup(entry.second) &&
                    entry.second.disk_backup_path == metadata.disk_backup_path) {
                    release_disk_state = false;
                    break;
                }
            }
            if (release_disk_state) {
                prefetch_state_to_destroy = DiskOffload::pop_prefetch_state(disk_prefetch_states_, metadata.disk_backup_path);
            }
        }
    }

    DiskOffload::destroy_prefetch_state(prefetch_state_to_destroy);

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
            if (DiskOffload::has_disk_backup(metadata)) {
                disk_items_by_path[metadata.disk_backup_path].push_back(AllocationRef{it->first, &metadata});
                if (metadata.disk_backup_offset == DiskOffload::invalid_disk_backup_offset()) {
                    disk_path_needs_materialize[metadata.disk_backup_path] = true;
                }
            }
        }

        std::unordered_map<int, BatchMemcpyGroup> d2h_groups;
        for (const AllocationRef& item : matching_items) {
            void* ptr = item.ptr;
            AllocationMetadata& metadata = *item.metadata;

            const bool needs_legacy_copy = metadata.enable_cpu_backup;
            const bool needs_disk_artifact_copy = DiskOffload::has_disk_backup(metadata) &&
                disk_path_needs_materialize[metadata.disk_backup_path];

            if (needs_legacy_copy || needs_disk_artifact_copy) {
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
            DiskOffload::materialize_artifact(entry.second);
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
            auto state = DiskOffload::pop_prefetch_state(disk_prefetch_states_, entry.first);
            if (state != nullptr) {
                prefetch_states_to_destroy.push_back(state);
            }
        }
    }

    for (const auto& state : prefetch_states_to_destroy) {
        DiskOffload::destroy_prefetch_state(state);
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
    AsyncMemcpyContext async_memcpy_context = create_async_memcpy_context();

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
            if (DiskOffload::has_disk_backup(metadata)) {
                disk_items_by_path[metadata.disk_backup_path].push_back(AllocationRef{it->first, &metadata});
            }
        }

        for (auto& entry : disk_items_by_path) {
            const std::string& path = entry.first;
            auto state_it = disk_prefetch_states_.find(path);
            if (state_it == disk_prefetch_states_.end()) {
                state_it = disk_prefetch_states_.emplace(path, DiskOffload::create_prefetch_state(path, DiskOffload::total_size(entry.second))).first;
            }
            disk_prefetch_states_for_resume.emplace(path, state_it->second);
        }
    }

    for (const auto& entry : disk_prefetch_states_for_resume) {
        DiskOffload::start_prefetch_if_needed(entry.second);
    }

    for (const AllocationRef& item : matching_items) {
        void *ptr = item.ptr;
        AllocationMetadata &metadata = *item.metadata;

        CUmemGenericAllocationHandle newAllocHandle;
        CUDA_ERROR_CHECK(CUDAUtils::cu_mem_create(&newAllocHandle, metadata.size, metadata.device));

        CURESULT_CHECK(cuMemMap((CUdeviceptr) ptr, metadata.size, 0, newAllocHandle, 0));
        CUDAUtils::cu_mem_set_access(ptr, metadata.size, metadata.device);

        if (metadata.enable_cpu_backup) {
            SIMPLE_CHECK(metadata.cpu_backup != nullptr, "Expected a host backup for CPU-backed resume");
            add_batch_copy(h2d_groups, metadata.device, ptr, metadata.cpu_backup, metadata.size);
        }

        metadata.state = AllocationState::ACTIVE;
        metadata.allocHandle = newAllocHandle;
    }

    // Phase 1: enqueue RAM-backed restores and drain any READY disk prefetch slots
    // without blocking on completion. Ring slots are recycled by event queries.
    struct DiskResumeState {
        std::shared_ptr<DiskPrefetchState> prefetch_state;
        std::vector<AllocationRef>* items;
        size_t allocation_index = 0;
        size_t allocation_offset = 0;
        uint64_t resume_offset = 0;
    };
    std::vector<DiskResumeState> disk_resume_states;

    enqueue_batch_memcpy_async(async_memcpy_context, h2d_groups, cudaMemcpyHostToDevice);

    for (auto& entry : disk_items_by_path) {
        DiskResumeState drs;
        drs.prefetch_state = disk_prefetch_states_for_resume.at(entry.first);
        drs.items = &entry.second;

        std::sort(drs.items->begin(), drs.items->end(), [](const AllocationRef& lhs, const AllocationRef& rhs) {
            return lhs.metadata->disk_backup_offset < rhs.metadata->disk_backup_offset;
        });

        drain_ready_disk_slots_async(
            async_memcpy_context,
            drs.prefetch_state,
            *drs.items,
            drs.allocation_index,
            drs.allocation_offset,
            drs.resume_offset
        );
        disk_resume_states.push_back(std::move(drs));
    }

    // Phase 2: enqueue remaining disk slots as they become ready.
    for (auto& drs : disk_resume_states) {
        consume_disk_prefetch_remaining_async(
            async_memcpy_context,
            drs.prefetch_state,
            *drs.items,
            drs.allocation_index,
            drs.allocation_offset,
            drs.resume_offset
        );
    }

    synchronize_and_destroy_async_memcpy_context(async_memcpy_context);

    {
        const std::lock_guard<std::mutex> lock(allocator_metadata_mutex_);
        for (const AllocationRef& item : matching_items) {
            AllocationMetadata &metadata = *item.metadata;
            if (metadata.cpu_backup == nullptr) {
                continue;
            }

            if (metadata.enable_cpu_backup) {
                CUDA_ERROR_CHECK(cudaFreeHost(metadata.cpu_backup));
                metadata.cpu_backup = nullptr;
            }
        }

        for (const auto& entry : disk_prefetch_states_for_resume) {
            auto state = DiskOffload::pop_prefetch_state(disk_prefetch_states_, entry.first);
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
                DiskOffload::destroy_prefetch_state(state);
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
            if (DiskOffload::has_disk_backup(metadata)) {
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
