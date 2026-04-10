#include "disk_offload.h"

#include "macro.h"
#include "utils.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <limits>
#include <sys/mman.h>
#include <unistd.h>

namespace {

constexpr uint64_t kInvalidDiskBackupOffset = std::numeric_limits<uint64_t>::max();
constexpr uint64_t kDirectIoAlignmentBytes = 4096;

enum class DiskReadMode {
    DIRECT,
    BUFFERED
};

uint64_t align_up(uint64_t value, uint64_t alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
}

size_t default_disk_read_threads(bool use_direct_io) {
    if (use_direct_io) {
        return 1;
    }

    const unsigned int concurrency = std::thread::hardware_concurrency();
    if (concurrency == 0) {
        return 3;
    }
    return std::max<size_t>(1, std::min<size_t>(3, concurrency));
}

DiskReadMode get_disk_read_mode() {
    const std::string mode = get_string_env_var("TMS_DISK_READ_MODE");
    if (mode.empty() || mode == "direct") {
        return DiskReadMode::DIRECT;
    }
    if (mode == "buffered") {
        return DiskReadMode::BUFFERED;
    }

    std::cerr << "[torch_memory_saver.cpp] Unsupported disk read mode "
              << " name=TMS_DISK_READ_MODE value=" << mode
              << " (expected one of: direct, buffered)"
              << std::endl;
    exit(1);
}

void destroy_slot_completion_events(DiskPrefetchSlot& slot) {
    for (const auto& completion_event : slot.completion_events) {
        CUDA_ERROR_CHECK(cudaSetDevice(completion_event.first));
        CUDA_ERROR_CHECK(cudaEventDestroy(completion_event.second));
    }
    slot.completion_events.clear();
}

bool recycle_completed_disk_prefetch_slots(const std::shared_ptr<DiskPrefetchState>& state) {
    bool freed_any = false;

    int original_device = 0;
    CUDA_ERROR_CHECK(cudaGetDevice(&original_device));

    {
        std::lock_guard<std::mutex> lock(state->mutex);
        for (DiskPrefetchSlot& slot : state->slots) {
            if (slot.state != DiskPrefetchSlotState::IN_FLIGHT) {
                continue;
            }

            bool all_complete = true;
            for (const auto& completion_event : slot.completion_events) {
                CUDA_ERROR_CHECK(cudaSetDevice(completion_event.first));
                const cudaError_t query_result = cudaEventQuery(completion_event.second);
                if (query_result == cudaErrorNotReady) {
                    all_complete = false;
                    break;
                }
                CUDA_ERROR_CHECK(query_result);
            }

            if (!all_complete) {
                continue;
            }

            destroy_slot_completion_events(slot);
            slot.state = DiskPrefetchSlotState::FREE;
            freed_any = true;
        }
    }

    CUDA_ERROR_CHECK(cudaSetDevice(original_device));
    if (freed_any) {
        state->cv.notify_all();
    }
    return freed_any;
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

}  // namespace

namespace DiskOffload {

uint64_t invalid_disk_backup_offset() {
    return kInvalidDiskBackupOffset;
}

bool has_disk_backup(const AllocationMetadata& metadata) {
    return !metadata.disk_backup_path.empty();
}

uint64_t total_size(const std::vector<AllocationRef>& items) {
    uint64_t total_size = 0;
    for (const AllocationRef& item : items) {
        total_size = std::max(total_size, item.metadata->disk_backup_offset + static_cast<uint64_t>(item.metadata->size));
    }
    return total_size;
}

void materialize_artifact(const std::vector<AllocationRef>& items) {
    SIMPLE_CHECK(!items.empty(), "materialize_artifact expects non-empty items");
    const DiskReadMode disk_read_mode = get_disk_read_mode();

    std::vector<AllocationRef> ordered = items;
    std::sort(ordered.begin(), ordered.end(), [](const AllocationRef& lhs, const AllocationRef& rhs) {
        return lhs.ptr < rhs.ptr;
    });

    const std::string& path = ordered.front().metadata->disk_backup_path;
    SIMPLE_CHECK(!path.empty(), "disk_backup_loc requires a non-empty path");

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
        metadata.disk_backup_offset = offset;
        pwrite_all(fd, metadata.cpu_backup, metadata.size, offset);
        offset += metadata.size;
    }

    const uint64_t padded_offset = align_up(offset, kDirectIoAlignmentBytes);
    if (padded_offset != offset) {
        uint8_t padding[kDirectIoAlignmentBytes] = {};
        pwrite_all(fd, padding, padded_offset - offset, offset);
    }

    (void) fdatasync(fd);
    if (disk_read_mode == DiskReadMode::DIRECT) {
        (void) posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
    }

    if (close(fd) != 0) {
        std::cerr << "[torch_memory_saver.cpp] close after write failed path=" << path
                  << " errno=" << errno << " (" << std::strerror(errno) << ")"
                  << " file=" << __FILE__ << " func=" << __func__ << " line=" << __LINE__
                  << std::endl;
        exit(1);
    }
}

std::shared_ptr<DiskPrefetchState> create_prefetch_state(const std::string& path, uint64_t total_size) {
    SIMPLE_CHECK(!path.empty(), "disk_backup_loc requires a non-empty path");
    const DiskReadMode disk_read_mode = get_disk_read_mode();
    const bool use_direct_io = (disk_read_mode == DiskReadMode::DIRECT);
    const size_t read_thread_count = std::max<size_t>(1, static_cast<size_t>(get_uint64_env_var(
        "TMS_DISK_READ_THREADS",
        default_disk_read_threads(use_direct_io)
    )));

    const uint64_t default_chunk_bytes = use_direct_io
        ? 128ull * 1024ull * 1024ull
        : 8ull * 1024ull * 1024ull;
    const size_t configured_chunk_bytes = static_cast<size_t>(get_uint64_env_var(
        "TMS_DISK_RING_BUFFER_BYTES",
        default_chunk_bytes
    ));
    const size_t ring_slot_bytes = static_cast<size_t>(align_up(
        std::max<uint64_t>(configured_chunk_bytes, kDirectIoAlignmentBytes),
        kDirectIoAlignmentBytes
    ));
    const size_t ring_slot_count = std::max<size_t>(
        read_thread_count + 1,
        std::max<size_t>(2, static_cast<size_t>(get_uint64_env_var("TMS_DISK_RING_SLOTS", 4)))
    );

    auto state = std::make_shared<DiskPrefetchState>();
    state->path = path;
    state->total_size = total_size;
    state->padded_total_size = align_up(total_size, kDirectIoAlignmentBytes);
    state->ring_slot_bytes = ring_slot_bytes;
    state->read_thread_count = read_thread_count;
    state->use_direct_io = use_direct_io;
    state->slots.resize(ring_slot_count);

    for (DiskPrefetchSlot& slot : state->slots) {
        CUDA_ERROR_CHECK(cudaMallocHost(&slot.host_buffer, ring_slot_bytes));
    }

    return state;
}

void start_prefetch_if_needed(const std::shared_ptr<DiskPrefetchState>& state) {
    {
        std::lock_guard<std::mutex> lock(state->mutex);
        if (state->producer_started) {
            return;
        }
        state->producer_started = true;
        state->recycler_started = true;
    }

    std::cout << "[torch_memory_saver.cpp] disk prefetch started"
              << " path=" << state->path
              << " bytes=" << state->total_size
              << " io_mode=" << (state->use_direct_io ? "direct" : "buffered")
              << " read_threads=" << state->read_thread_count
              << " ring_slot_bytes=" << state->ring_slot_bytes
              << " ring_slots=" << state->slots.size()
              << std::endl;

    if (!state->use_direct_io) {
        state->buffered_fd = open(state->path.c_str(), O_RDONLY);
        if (state->buffered_fd < 0) {
            std::cerr << "[torch_memory_saver.cpp] open for buffered mmap failed path=" << state->path
                      << " errno=" << errno << " (" << std::strerror(errno) << ")"
                      << " file=" << __FILE__ << " func=" << __func__ << " line=" << __LINE__
                      << std::endl;
            exit(1);
        }

        (void) posix_fadvise(state->buffered_fd, 0, 0, POSIX_FADV_SEQUENTIAL);
        (void) posix_fadvise(state->buffered_fd, 0, 0, POSIX_FADV_WILLNEED);

        state->mapped_artifact = mmap(
            nullptr,
            static_cast<size_t>(state->padded_total_size),
            PROT_READ,
            MAP_SHARED,
            state->buffered_fd,
            0
        );
        if (state->mapped_artifact == MAP_FAILED) {
            std::cerr << "[torch_memory_saver.cpp] mmap failed path=" << state->path
                      << " errno=" << errno << " (" << std::strerror(errno) << ")"
                      << " file=" << __FILE__ << " func=" << __func__ << " line=" << __LINE__
                      << std::endl;
            exit(1);
        }

        (void) madvise(state->mapped_artifact, static_cast<size_t>(state->padded_total_size), MADV_SEQUENTIAL);
        (void) madvise(state->mapped_artifact, static_cast<size_t>(state->padded_total_size), MADV_WILLNEED);
    }

    state->producers.reserve(state->read_thread_count);
    for (size_t thread_idx = 0; thread_idx < state->read_thread_count; ++thread_idx) {
        state->producers.emplace_back([state]() {
            int fd = -1;
            if (state->use_direct_io) {
                fd = open(state->path.c_str(), O_RDONLY | O_DIRECT);
                if (fd < 0) {
                    std::cerr << "[torch_memory_saver.cpp] open for read failed path=" << state->path
                              << " errno=" << errno << " (" << std::strerror(errno) << ")"
                              << " file=" << __FILE__ << " func=" << __func__ << " line=" << __LINE__
                              << std::endl;
                    exit(1);
                }

                (void) posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);
            }

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
                        if (state->in_flight_reads == 0) {
                            state->producer_done = true;
                            state->cv.notify_all();
                        }
                        break;
                    }

                    while (slot_index < state->slots.size() &&
                           state->slots[slot_index].state != DiskPrefetchSlotState::FREE) {
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
                    ++state->in_flight_reads;
                }

                if (state->use_direct_io) {
                    pread_all(fd, state->slots[slot_index].host_buffer, aligned_bytes, file_offset);
                } else {
                    std::memcpy(
                        state->slots[slot_index].host_buffer,
                        static_cast<const uint8_t*>(state->mapped_artifact) + file_offset,
                        valid_bytes
                    );
                    if (aligned_bytes > valid_bytes) {
                        std::memset(
                            static_cast<uint8_t*>(state->slots[slot_index].host_buffer) + valid_bytes,
                            0,
                            aligned_bytes - valid_bytes
                        );
                    }
                }

                {
                    std::lock_guard<std::mutex> lock(state->mutex);
                    state->slots[slot_index].state = DiskPrefetchSlotState::READY;
                    SIMPLE_CHECK(state->in_flight_reads > 0, "Disk prefetch read accounting underflow");
                    --state->in_flight_reads;
                    if (!state->stop_requested &&
                        state->next_read_offset >= state->padded_total_size &&
                        state->in_flight_reads == 0) {
                        state->producer_done = true;
                    }
                }
                state->cv.notify_all();
            }

            if (state->use_direct_io) {
                (void) posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
                if (close(fd) != 0) {
                    std::cerr << "[torch_memory_saver.cpp] close after read failed path=" << state->path
                              << " errno=" << errno << " (" << std::strerror(errno) << ")"
                              << " file=" << __FILE__ << " func=" << __func__ << " line=" << __LINE__
                              << std::endl;
                    exit(1);
                }
            }
        });
    }

    state->recycler = std::thread([state]() {
        while (true) {
            {
                std::unique_lock<std::mutex> lock(state->mutex);
                state->cv.wait(lock, [&]() {
                    if (state->stop_requested) {
                        return true;
                    }
                    for (const DiskPrefetchSlot& slot : state->slots) {
                        if (slot.state == DiskPrefetchSlotState::IN_FLIGHT) {
                            return true;
                        }
                    }
                    return false;
                });

                if (state->stop_requested) {
                    break;
                }
            }

            if (!recycle_completed_disk_prefetch_slots(state)) {
                std::this_thread::sleep_for(std::chrono::microseconds(25));
            }
        }
    });
}

void destroy_prefetch_state(const std::shared_ptr<DiskPrefetchState>& state) {
    if (state == nullptr) {
        return;
    }

    {
        std::lock_guard<std::mutex> lock(state->mutex);
        state->stop_requested = true;
    }
    state->cv.notify_all();

    for (std::thread& producer : state->producers) {
        if (producer.joinable()) {
            producer.join();
        }
    }

    if (state->recycler.joinable()) {
        state->recycler.join();
    }

    if (state->mapped_artifact != nullptr) {
        if (munmap(state->mapped_artifact, static_cast<size_t>(state->padded_total_size)) != 0) {
            std::cerr << "[torch_memory_saver.cpp] munmap failed path=" << state->path
                      << " errno=" << errno << " (" << std::strerror(errno) << ")"
                      << " file=" << __FILE__ << " func=" << __func__ << " line=" << __LINE__
                      << std::endl;
            exit(1);
        }
        state->mapped_artifact = nullptr;
    }

    if (state->buffered_fd >= 0) {
        if (close(state->buffered_fd) != 0) {
            std::cerr << "[torch_memory_saver.cpp] close after buffered mmap failed path=" << state->path
                      << " errno=" << errno << " (" << std::strerror(errno) << ")"
                      << " file=" << __FILE__ << " func=" << __func__ << " line=" << __LINE__
                      << std::endl;
            exit(1);
        }
        state->buffered_fd = -1;
    }

    for (DiskPrefetchSlot& slot : state->slots) {
        SIMPLE_CHECK(slot.completion_events.empty(),
            "Destroying disk prefetch state with pending slot completion events");
        if (slot.host_buffer != nullptr) {
            CUDA_ERROR_CHECK(cudaFreeHost(slot.host_buffer));
            slot.host_buffer = nullptr;
        }
    }

    if (!state->stop_requested && state->producer_done) {
        std::cout << "[torch_memory_saver.cpp] disk prefetch complete"
                  << " path=" << state->path
                  << " bytes=" << state->total_size
                  << std::endl;
    }
}

std::shared_ptr<DiskPrefetchState> pop_prefetch_state(
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

}  // namespace DiskOffload
