#include "core.h"
#include "disk_offload.h"
#include "utils.h"
#include "macro.h"
#include "api_forwarder.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <numeric>
#include <optional>
#include <sstream>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>

namespace {

constexpr const char* kShmDaemonSocketEnv = "TMS_SHM_DAEMON_SOCKET";
constexpr const char* kArtifactCompleteSuffix = ".complete";
constexpr size_t kCudaHostRegisterChunkBytes = 1ull << 30;
constexpr size_t kSharedArtifactCopyChunkBytes = 128ull * 1024ull * 1024ull;
constexpr const char kSharedRingMagic[8] = {'T', 'M', 'S', 'R', 'I', 'N', 'G', '\0'};
constexpr uint32_t kSharedRingVersion = 1;
constexpr uint32_t kSharedRingErrorGeneric = 1;

enum class SharedRingBlockState : uint32_t {
    FREE = 0,
    WRITING = 1,
    READY = 2,
    READING = 3,
    ERROR = 4,
};

enum class SharedRingConsumerState : uint32_t {
    IDLE = 0,
    ATTACHED = 1,
    DONE = 2,
};

struct StagedArtifactInfo {
    std::string shm_name;
    std::string completion_token;
    uint64_t artifact_size = 0;
    uint64_t mapped_size = 0;
    uint64_t block_count = 0;
    uint64_t block_payload_bytes = 0;
    uint64_t payload_offset = 0;
    int shm_fd = -1;
};

struct DaemonResponse {
    std::string line;
    int received_fd = -1;
};

bool matches_tag(const std::string& filter_tag, const AllocationMetadata& metadata) {
    return filter_tag.empty() || metadata.tag == filter_tag;
}

bool should_return_host_backup(const AllocationMetadata& metadata) {
    return metadata.enable_cpu_backup || metadata.cpu_backup != nullptr;
}

std::string get_shm_daemon_socket_path() {
    return get_string_env_var(kShmDaemonSocketEnv);
}

std::string artifact_completion_marker_path(const std::string& artifact_path) {
    return artifact_path + kArtifactCompleteSuffix;
}

double duration_ms(
    const std::chrono::steady_clock::time_point& start,
    const std::chrono::steady_clock::time_point& end
) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

std::vector<std::string> split_tab_fields(const std::string& line) {
    std::vector<std::string> fields;
    std::stringstream stream(line);
    std::string field;
    while (std::getline(stream, field, '\t')) {
        fields.push_back(field);
    }
    return fields;
}

std::string read_artifact_completion_token(const std::string& artifact_path) {
    std::ifstream marker_file(artifact_completion_marker_path(artifact_path));
    if (!marker_file.is_open()) {
        return "";
    }
    std::string line;
    std::getline(marker_file, line);
    if (line.empty()) {
        return "";
    }
    const std::vector<std::string> fields = split_tab_fields(line);
    if (fields.size() >= 2) {
        return fields[1];
    }
    return line;
}

uint32_t atomic_load_u32(const uint32_t* ptr) {
    return __atomic_load_n(ptr, __ATOMIC_ACQUIRE);
}

void atomic_store_u32(uint32_t* ptr, uint32_t value) {
    __atomic_store_n(ptr, value, __ATOMIC_RELEASE);
}

uint64_t atomic_load_u64(const uint64_t* ptr) {
    return __atomic_load_n(ptr, __ATOMIC_ACQUIRE);
}

void atomic_store_u64(uint64_t* ptr, uint64_t value) {
    __atomic_store_n(ptr, value, __ATOMIC_RELEASE);
}

SharedRingGlobalHeader* shared_ring_global_header(const SharedArtifactHostMapping& mapping) {
    return static_cast<SharedRingGlobalHeader*>(mapping.mapping_base);
}

SharedRingBlockHeader* shared_ring_block_headers(const SharedArtifactHostMapping& mapping) {
    return reinterpret_cast<SharedRingBlockHeader*>(static_cast<uint8_t*>(mapping.mapping_base) + sizeof(SharedRingGlobalHeader));
}

void* shared_ring_block_payload(const SharedArtifactHostMapping& mapping, size_t block_index) {
    return static_cast<uint8_t*>(mapping.payload_base) + block_index * mapping.block_payload_bytes;
}

bool shared_ring_header_valid(const SharedRingGlobalHeader& header) {
    return std::memcmp(header.magic, kSharedRingMagic, sizeof(header.magic)) == 0 &&
           header.version == kSharedRingVersion &&
           header.header_bytes == sizeof(SharedRingGlobalHeader) &&
           header.block_header_bytes == sizeof(SharedRingBlockHeader);
}

bool cuda_host_register_chunked(void* host_buffer, size_t size, unsigned int flags) {
    uint8_t* cursor = static_cast<uint8_t*>(host_buffer);
    size_t remaining = size;
    std::vector<void*> registered_ptrs;
    while (remaining > 0) {
        const size_t chunk = std::min(remaining, kCudaHostRegisterChunkBytes);
        cudaError_t register_result = cudaHostRegister(cursor, chunk, flags);
        if (register_result != cudaSuccess) {
            for (void* registered_ptr : registered_ptrs) {
                (void) cudaHostUnregister(registered_ptr);
            }
            return false;
        }
        registered_ptrs.push_back(cursor);
        cursor += chunk;
        remaining -= chunk;
    }
    return true;
}

void cuda_host_unregister_chunked(void* host_buffer, size_t size) {
    uint8_t* cursor = static_cast<uint8_t*>(host_buffer);
    size_t remaining = size;
    while (remaining > 0) {
        const size_t chunk = std::min(remaining, kCudaHostRegisterChunkBytes);
        cudaError_t unregister_result = cudaHostUnregister(cursor);
        SIMPLE_CHECK(unregister_result == cudaSuccess, "cudaHostUnregister failed for shared artifact mapping");
        cursor += chunk;
        remaining -= chunk;
    }
}

std::optional<DaemonResponse> shm_daemon_request(
    const std::string& command,
    const std::string& artifact_path,
    uint64_t artifact_size
) {
    const std::string socket_path = get_shm_daemon_socket_path();
    if (socket_path.empty()) {
        return std::nullopt;
    }

    const int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) {
        return std::nullopt;
    }

    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    SIMPLE_CHECK(socket_path.size() < sizeof(addr.sun_path), "Daemon socket path too long");
    std::snprintf(addr.sun_path, sizeof(addr.sun_path), "%s", socket_path.c_str());

    if (connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        (void) close(fd);
        return std::nullopt;
    }

    const std::string request = command + "\t" + artifact_path + "\t" + std::to_string(artifact_size) + "\n";
    size_t sent = 0;
    while (sent < request.size()) {
        const ssize_t written = write(fd, request.data() + sent, request.size() - sent);
        if (written < 0) {
            if (errno == EINTR) {
                continue;
            }
            (void) close(fd);
            return std::nullopt;
        }
        sent += static_cast<size_t>(written);
    }

    DaemonResponse response;
    char buffer[1024];
    char control[CMSG_SPACE(sizeof(int))];
    iovec iov{};
    iov.iov_base = buffer;
    iov.iov_len = sizeof(buffer);
    msghdr msg{};
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;
    msg.msg_control = control;
    msg.msg_controllen = sizeof(control);

    while (true) {
        const ssize_t n = recvmsg(fd, &msg, 0);
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            if (response.received_fd >= 0) {
                (void) close(response.received_fd);
            }
            (void) close(fd);
            return std::nullopt;
        }
        if (n == 0) {
            break;
        }

        response.line.append(buffer, buffer + n);
        for (cmsghdr* cmsg = CMSG_FIRSTHDR(&msg); cmsg != nullptr; cmsg = CMSG_NXTHDR(&msg, cmsg)) {
            if (cmsg->cmsg_level == SOL_SOCKET && cmsg->cmsg_type == SCM_RIGHTS && cmsg->cmsg_len >= CMSG_LEN(sizeof(int))) {
                int received_fd = -1;
                std::memcpy(&received_fd, CMSG_DATA(cmsg), sizeof(int));
                if (response.received_fd >= 0) {
                    (void) close(response.received_fd);
                }
                response.received_fd = received_fd;
            }
        }

        if (response.line.find('\n') != std::string::npos) {
            break;
        }

        msg.msg_control = control;
        msg.msg_controllen = sizeof(control);
    }

    (void) close(fd);
    const size_t newline_pos = response.line.find('\n');
    if (newline_pos != std::string::npos) {
        response.line.resize(newline_pos);
    }
    if (response.line.empty()) {
        if (response.received_fd >= 0) {
            (void) close(response.received_fd);
        }
        return std::nullopt;
    }
    return response;
}

std::optional<StagedArtifactInfo> lookup_staged_artifact_once(
    const std::string& artifact_path,
    uint64_t artifact_size
) {
    std::optional<DaemonResponse> response = shm_daemon_request("lookup", artifact_path, artifact_size);
    if (!response.has_value()) {
        return std::nullopt;
    }

    const std::vector<std::string> fields = split_tab_fields(response->line);
    if (fields.empty()) {
        if (response->received_fd >= 0) {
            (void) close(response->received_fd);
        }
        return std::nullopt;
    }

    if ((fields[0] != "streaming" && fields[0] != "ready") || fields.size() != 8 || response->received_fd < 0) {
        if (response->received_fd >= 0) {
            (void) close(response->received_fd);
        }
        return std::nullopt;
    }

    StagedArtifactInfo info;
    info.shm_name = fields[1];
    info.completion_token = fields[2];
    info.artifact_size = static_cast<uint64_t>(std::stoull(fields[3]));
    info.mapped_size = static_cast<uint64_t>(std::stoull(fields[4]));
    info.block_count = static_cast<uint64_t>(std::stoull(fields[5]));
    info.block_payload_bytes = static_cast<uint64_t>(std::stoull(fields[6]));
    info.payload_offset = static_cast<uint64_t>(std::stoull(fields[7]));
    info.shm_fd = response->received_fd;
    return info;
}

std::optional<StagedArtifactInfo> lookup_staged_artifact(
    const std::string& artifact_path,
    uint64_t artifact_size
) {
    std::optional<StagedArtifactInfo> staged = lookup_staged_artifact_once(artifact_path, artifact_size);
    if (staged.has_value()) {
        std::cout << "[torch_memory_saver.cpp] daemon lookup accepted staged mapping"
                  << " path=" << artifact_path
                  << " artifact_bytes=" << artifact_size
                  << " shm_name=" << staged->shm_name
                  << " staged_file_bytes=" << staged->artifact_size
                  << " mapped_bytes=" << staged->mapped_size
                  << " block_bytes=" << staged->block_payload_bytes
                  << " block_count=" << staged->block_count
                  << std::endl;
        return staged;
    }
    if (staged.has_value() && staged->shm_fd >= 0) {
        (void) close(staged->shm_fd);
    }
    return std::nullopt;
}

void destroy_shared_artifact_mapping(SharedArtifactHostMapping& mapping) {
    if (mapping.mapping_base != nullptr) {
        for (size_t block_index = 0; block_index < mapping.registered_blocks.size(); ++block_index) {
            if (mapping.registered_blocks[block_index] == 0) {
                continue;
            }
            void* block_payload = shared_ring_block_payload(mapping, block_index);
            const cudaError_t unregister_result = cudaHostUnregister(block_payload);
            SIMPLE_CHECK(unregister_result == cudaSuccess, "cudaHostUnregister failed for shared artifact block");
        }
    }
    mapping.cuda_registered = false;
    mapping.registered_blocks.clear();

    if (mapping.mapping_base != nullptr) {
        SharedRingGlobalHeader* header = shared_ring_global_header(mapping);
        atomic_store_u32(&header->consumer_state, static_cast<uint32_t>(SharedRingConsumerState::DONE));
        SIMPLE_CHECK(munmap(mapping.mapping_base, mapping.mapped_size) == 0, "munmap failed for shared artifact mapping");
        mapping.mapping_base = nullptr;
        mapping.payload_base = nullptr;
    }

    if (mapping.shm_fd >= 0) {
        SIMPLE_CHECK(close(mapping.shm_fd) == 0, "close failed for shared artifact mapping");
        mapping.shm_fd = -1;
    }

    mapping.artifact_size = 0;
    mapping.mapped_size = 0;
    mapping.payload_offset = 0;
    mapping.block_payload_bytes = 0;
    mapping.block_count = 0;
    mapping.shm_name.clear();
    mapping.completion_token.clear();
}

bool ensure_shared_artifact_mapping(
    std::unordered_map<std::string, SharedArtifactHostMapping>& shared_artifact_mappings,
    const std::string& artifact_path,
    uint64_t artifact_size
) {
    const auto lookup_start = std::chrono::steady_clock::now();
    std::cout << "[torch_memory_saver.cpp] ensure shared artifact mapping"
              << " path=" << artifact_path
              << " artifact_bytes=" << artifact_size
              << std::endl;
    const std::optional<StagedArtifactInfo> staged = lookup_staged_artifact(artifact_path, artifact_size);
    const auto lookup_end = std::chrono::steady_clock::now();
    if (!staged.has_value()) {
        return false;
    }

    SharedArtifactHostMapping& mapping = shared_artifact_mappings[artifact_path];
    if (mapping.mapping_base != nullptr &&
        mapping.shm_name == staged->shm_name &&
        mapping.artifact_size == artifact_size &&
        mapping.mapped_size == staged->mapped_size &&
        mapping.block_count == staged->block_count &&
        mapping.block_payload_bytes == staged->block_payload_bytes &&
        mapping.payload_offset == staged->payload_offset) {
        if (staged->shm_fd >= 0) {
            (void) close(staged->shm_fd);
        }
        return true;
    }

    destroy_shared_artifact_mapping(mapping);

    const int shm_fd = staged->shm_fd;
    if (shm_fd < 0) {
        shared_artifact_mappings.erase(artifact_path);
        return false;
    }

    const auto mmap_start = std::chrono::steady_clock::now();
    void* mapping_base = mmap(nullptr, staged->mapped_size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
    const auto mmap_end = std::chrono::steady_clock::now();
    if (mapping_base == MAP_FAILED) {
        (void) close(shm_fd);
        shared_artifact_mappings.erase(artifact_path);
        return false;
    }

    SharedRingGlobalHeader* header = static_cast<SharedRingGlobalHeader*>(mapping_base);
    if (!shared_ring_header_valid(*header) ||
        header->total_buffer_bytes != staged->mapped_size ||
        header->block_count != staged->block_count ||
        header->block_payload_bytes != staged->block_payload_bytes ||
        header->payload_offset != staged->payload_offset) {
        (void) munmap(mapping_base, staged->mapped_size);
        (void) close(shm_fd);
        shared_artifact_mappings.erase(artifact_path);
        return false;
    }

    mapping.shm_name = staged->shm_name;
    mapping.completion_token = staged->completion_token;
    mapping.mapping_base = mapping_base;
    mapping.payload_base = static_cast<uint8_t*>(mapping_base) + staged->payload_offset;
    mapping.artifact_size = static_cast<size_t>(artifact_size);
    mapping.mapped_size = static_cast<size_t>(staged->mapped_size);
    mapping.payload_offset = static_cast<size_t>(staged->payload_offset);
    mapping.block_payload_bytes = static_cast<size_t>(staged->block_payload_bytes);
    mapping.block_count = static_cast<size_t>(staged->block_count);
    mapping.shm_fd = shm_fd;
    mapping.cuda_registered = false;
    mapping.registered_blocks.assign(mapping.block_count, 0);
    atomic_store_u32(&header->consumer_state, static_cast<uint32_t>(SharedRingConsumerState::ATTACHED));
    std::cout << "[torch_memory_saver.cpp] shared artifact mapping ready"
              << " path=" << artifact_path
              << " shm_name=" << mapping.shm_name
              << " artifact_bytes=" << mapping.artifact_size
              << " staged_file_bytes=" << staged->artifact_size
              << " mapped_bytes=" << mapping.mapped_size
              << " block_bytes=" << mapping.block_payload_bytes
              << " block_count=" << mapping.block_count
              << " lookup_ms=" << duration_ms(lookup_start, lookup_end)
              << " mmap_ms=" << duration_ms(mmap_start, mmap_end)
              << " cuda_host_register_ms=0"
              << std::endl;
    return true;
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

struct SharedInFlightBlock {
    size_t block_index = 0;
    std::vector<std::pair<int, cudaEvent_t>> completion_events;
};

size_t total_group_bytes(const std::unordered_map<int, BatchMemcpyGroup>& groups) {
    size_t total_bytes = 0;
    for (const auto& entry : groups) {
        total_bytes += std::accumulate(entry.second.sizes.begin(), entry.second.sizes.end(), static_cast<size_t>(0));
    }
    return total_bytes;
}

size_t total_group_copies(const std::unordered_map<int, BatchMemcpyGroup>& groups) {
    size_t total_copies = 0;
    for (const auto& entry : groups) {
        total_copies += entry.second.sizes.size();
    }
    return total_copies;
}

bool ensure_shared_ring_block_registered(SharedArtifactHostMapping& mapping, size_t block_index) {
    SIMPLE_CHECK(block_index < mapping.block_count, "Shared ring block index out of range");
    if (mapping.registered_blocks.empty()) {
        mapping.registered_blocks.assign(mapping.block_count, 0);
    }
    if (mapping.registered_blocks[block_index] != 0) {
        return false;
    }
    const auto start = std::chrono::steady_clock::now();
    void* block_payload = shared_ring_block_payload(mapping, block_index);
    const cudaError_t register_result = cudaHostRegister(block_payload, mapping.block_payload_bytes, cudaHostRegisterPortable);
    SIMPLE_CHECK(register_result == cudaSuccess, "cudaHostRegister failed for shared artifact block");
    const auto end = std::chrono::steady_clock::now();
    mapping.registered_blocks[block_index] = 1;
    mapping.cuda_registered = true;
    std::cout << "[torch_memory_saver.cpp] shared block registered"
              << " block_index=" << block_index
              << " block_bytes=" << mapping.block_payload_bytes
              << " elapsed_ms=" << duration_ms(start, end)
              << std::endl;
    return true;
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

void add_registered_mapping_batch_copy(
    std::unordered_map<int, BatchMemcpyGroup>& groups,
    const SharedArtifactHostMapping& mapping,
    CUdevice device,
    void* dst,
    void* src,
    size_t size
) {
    uint8_t* dst_cursor = static_cast<uint8_t*>(dst);
    uint8_t* src_cursor = static_cast<uint8_t*>(src);
    size_t remaining = size;
    uint8_t* mapping_base = static_cast<uint8_t*>(mapping.mapping_base);
    const size_t mapping_bytes = mapping.mapped_size;

    while (remaining > 0) {
        SIMPLE_CHECK(src_cursor >= mapping_base, "shared mapping source pointer underflow");
        const size_t src_offset = static_cast<size_t>(src_cursor - mapping_base);
        SIMPLE_CHECK(src_offset < mapping_bytes, "shared mapping source pointer overflow");
        const size_t chunk_offset = src_offset % kCudaHostRegisterChunkBytes;
        const size_t chunk_remaining = kCudaHostRegisterChunkBytes - chunk_offset;
        const size_t copy_bytes = std::min(remaining, std::min(chunk_remaining, kSharedArtifactCopyChunkBytes));
        add_batch_copy(groups, device, dst_cursor, src_cursor, copy_bytes);
        dst_cursor += copy_bytes;
        src_cursor += copy_bytes;
        remaining -= copy_bytes;
    }
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

void record_shared_block_events(
    AsyncMemcpyContext& context,
    size_t block_index,
    const std::vector<int>& used_devices,
    std::vector<SharedInFlightBlock>& pending_blocks
) {
    if (used_devices.empty()) {
        SharedInFlightBlock pending;
        pending.block_index = block_index;
        pending_blocks.push_back(std::move(pending));
        return;
    }

    SharedInFlightBlock pending;
    pending.block_index = block_index;
    pending.completion_events.reserve(used_devices.size());
    for (const int device : used_devices) {
        CUDA_ERROR_CHECK(cudaSetDevice(device));
        cudaEvent_t event;
        CUDA_ERROR_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
        CUDA_ERROR_CHECK(cudaEventRecord(event, get_async_stream_for_device(context, device)));
        pending.completion_events.emplace_back(device, event);
    }

    CUDA_ERROR_CHECK(cudaSetDevice(context.original_device));
    pending_blocks.push_back(std::move(pending));
}

bool shared_block_complete(const SharedInFlightBlock& pending, bool wait) {
    for (const auto& entry : pending.completion_events) {
        const int device = entry.first;
        const cudaEvent_t event = entry.second;
        CUDA_ERROR_CHECK(cudaSetDevice(device));
        if (wait) {
            CUDA_ERROR_CHECK(cudaEventSynchronize(event));
            continue;
        }
        const cudaError_t query_result = cudaEventQuery(event);
        if (query_result == cudaSuccess) {
            continue;
        }
        if (query_result == cudaErrorNotReady) {
            return false;
        }
        CUDA_ERROR_CHECK(query_result);
    }
    return true;
}

void release_completed_shared_blocks(
    AsyncMemcpyContext& context,
    const SharedArtifactHostMapping& mapping,
    std::vector<SharedInFlightBlock>& pending_blocks,
    bool wait_for_all
) {
    size_t write_index = 0;
    for (size_t read_index = 0; read_index < pending_blocks.size(); ++read_index) {
        SharedInFlightBlock& pending = pending_blocks[read_index];
        if (!shared_block_complete(pending, wait_for_all)) {
            if (write_index != read_index) {
                pending_blocks[write_index] = std::move(pending);
            }
            write_index += 1;
            continue;
        }

        for (const auto& entry : pending.completion_events) {
            CUDA_ERROR_CHECK(cudaSetDevice(entry.first));
            CUDA_ERROR_CHECK(cudaEventDestroy(entry.second));
        }
        CUDA_ERROR_CHECK(cudaSetDevice(context.original_device));
        SharedRingBlockHeader* block_headers = shared_ring_block_headers(mapping);
        SharedRingBlockHeader& block = block_headers[pending.block_index];
        atomic_store_u64(&block.valid_bytes, 0);
        atomic_store_u32(&block.state, static_cast<uint32_t>(SharedRingBlockState::FREE));
    }
    pending_blocks.resize(write_index);
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
    cudaMemcpyKind fallback_kind,
    bool allow_batch = true,
    const char* phase_label = "unspecified"
) {
    if (groups.empty()) {
        return;
    }

    const size_t copy_count = total_group_copies(groups);
    const size_t total_bytes = total_group_bytes(groups);
    const auto start = std::chrono::steady_clock::now();

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

        if (allow_batch) {
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
        } else {
            for (size_t i = 0; i < group.dsts.size(); ++i) {
                CUDA_ERROR_CHECK(cudaMemcpyAsync(
                    group.dsts[i],
                    group.srcs[i],
                    group.sizes[i],
                    fallback_kind,
                    stream
                ));
            }
        }
    }

    for (const auto& entry : streams_by_device) {
        CUDA_ERROR_CHECK(cudaSetDevice(entry.first));
        CUDA_ERROR_CHECK(cudaStreamSynchronize(entry.second));
        CUDA_ERROR_CHECK(cudaStreamDestroy(entry.second));
    }

    CUDA_ERROR_CHECK(cudaSetDevice(original_device));
    const auto end = std::chrono::steady_clock::now();
    const double elapsed_ms = duration_ms(start, end);
    const double gib_per_s = elapsed_ms > 0.0
        ? (static_cast<double>(total_bytes) / (1024.0 * 1024.0 * 1024.0)) / (elapsed_ms / 1000.0)
        : 0.0;
    std::cout << "[torch_memory_saver.cpp] batch memcpy complete"
              << " phase=" << phase_label
              << " allow_batch=" << (allow_batch ? "true" : "false")
              << " devices=" << streams_by_device.size()
              << " copies=" << copy_count
              << " bytes=" << total_bytes
              << " elapsed_ms=" << elapsed_ms
              << " gib_per_s=" << gib_per_s
              << std::endl;
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

void collect_shared_ring_block_copies(
    const SharedArtifactHostMapping& mapping,
    size_t block_index,
    std::vector<AllocationRef>& path_items,
    size_t& allocation_index,
    size_t& allocation_offset,
    std::unordered_map<int, BatchMemcpyGroup>& h2d_groups
) {
    SharedRingBlockHeader* block_headers = shared_ring_block_headers(mapping);
    SharedRingBlockHeader& block = block_headers[block_index];
    const uint64_t block_file_offset = atomic_load_u64(&block.file_offset);
    const uint64_t block_valid_bytes = atomic_load_u64(&block.valid_bytes);
    const uint64_t block_end = block_file_offset + block_valid_bytes;

    while (allocation_index < path_items.size()) {
        AllocationMetadata& metadata = *path_items[allocation_index].metadata;
        const uint64_t allocation_file_offset = metadata.disk_backup_offset + allocation_offset;
        if (allocation_file_offset >= block_end) {
            break;
        }

        const size_t src_offset = static_cast<size_t>(allocation_file_offset - block_file_offset);
        const size_t copy_bytes = static_cast<size_t>(std::min<uint64_t>(
            block_end - allocation_file_offset,
            metadata.size - allocation_offset
        ));
        add_registered_mapping_batch_copy(
            h2d_groups,
            mapping,
            metadata.device,
            static_cast<uint8_t*>(path_items[allocation_index].ptr) + allocation_offset,
            static_cast<uint8_t*>(shared_ring_block_payload(mapping, block_index)) + src_offset,
            copy_bytes
        );

        allocation_offset += copy_bytes;
        if (allocation_offset == metadata.size) {
            allocation_offset = 0;
            ++allocation_index;
        }
    }
}

uint64_t consume_shared_ring_ready(
    SharedArtifactHostMapping& mapping,
    std::vector<AllocationRef>& path_items,
    size_t& allocation_index,
    size_t& allocation_offset,
    AsyncMemcpyContext& async_memcpy_context,
    std::vector<SharedInFlightBlock>& pending_blocks
) {
    const auto start = std::chrono::steady_clock::now();
    SharedRingBlockHeader* block_headers = shared_ring_block_headers(mapping);

    // Phase 1: Scan all READY blocks in file-offset order and collect their indices.
    std::vector<size_t> ready_indices;
    std::vector<uint64_t> ready_file_offsets;
    uint64_t scan_offset = 0;
    while (scan_offset < mapping.artifact_size) {
        size_t block_index = SIZE_MAX;
        for (size_t i = 0; i < mapping.block_count; ++i) {
            const SharedRingBlockHeader& block = block_headers[i];
            if (atomic_load_u32(&block.state) == static_cast<uint32_t>(SharedRingBlockState::READY) &&
                atomic_load_u64(&block.file_offset) == scan_offset) {
                block_index = i;
                break;
            }
        }
        if (block_index == SIZE_MAX) {
            break;
        }
        atomic_store_u32(&block_headers[block_index].state, static_cast<uint32_t>(SharedRingBlockState::READING));
        ready_indices.push_back(block_index);
        ready_file_offsets.push_back(scan_offset);
        scan_offset += atomic_load_u64(&block_headers[block_index].valid_bytes);
    }

    const size_t ready_block_count = ready_indices.size();
    if (ready_block_count == 0) {
        const auto end = std::chrono::steady_clock::now();
        std::cout << "[torch_memory_saver.cpp] shared ring ready drain"
                  << " artifact_bytes=" << mapping.artifact_size
                  << " ready_blocks=0"
                  << " ready_bytes=0"
                  << " elapsed_ms=" << duration_ms(start, end)
                  << std::endl;
        return 0;
    }

    // Phase 2: Overlap registration with H2D copies using a producer thread.
    // The registration thread registers blocks ahead of the main thread which
    // enqueues H2D copies.  This fully overlaps cudaHostRegister (CPU-side page
    // pinning) with GPU DMA transfers.
    std::atomic<size_t> registered_count{0};
    const auto register_start = std::chrono::steady_clock::now();

    std::thread register_thread([&]() {
        for (size_t i = 0; i < ready_block_count; ++i) {
            ensure_shared_ring_block_registered(mapping, ready_indices[i]);
            registered_count.store(i + 1, std::memory_order_release);
        }
    });

    uint64_t copy_total_us = 0;
    for (size_t i = 0; i < ready_block_count; ++i) {
        // Spin-wait until the registration thread has registered this block.
        while (registered_count.load(std::memory_order_acquire) <= i) {
            // Yield to avoid burning CPU while waiting for the first block.
            std::this_thread::yield();
        }

        const size_t block_index = ready_indices[i];
        std::unordered_map<int, BatchMemcpyGroup> h2d_groups;
        collect_shared_ring_block_copies(mapping, block_index, path_items, allocation_index, allocation_offset, h2d_groups);
        const std::vector<int> used_devices = get_used_devices(h2d_groups);
        const auto copy_start = std::chrono::steady_clock::now();
        enqueue_batch_memcpy_async(async_memcpy_context, h2d_groups, cudaMemcpyHostToDevice);
        const auto copy_end = std::chrono::steady_clock::now();
        copy_total_us += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(copy_end - copy_start).count());
        record_shared_block_events(async_memcpy_context, block_index, used_devices, pending_blocks);
    }

    register_thread.join();
    const auto register_end = std::chrono::steady_clock::now();

    const auto end = std::chrono::steady_clock::now();
    std::cout << "[torch_memory_saver.cpp] shared ring ready drain"
              << " artifact_bytes=" << mapping.artifact_size
              << " ready_blocks=" << ready_block_count
              << " ready_bytes=" << scan_offset
              << " register_thread_ms=" << duration_ms(register_start, register_end)
              << " copy_enqueue_us=" << copy_total_us
              << " elapsed_ms=" << duration_ms(start, end)
              << std::endl;

    return scan_offset;
}

void consume_shared_ring_remaining(
    SharedArtifactHostMapping& mapping,
    std::vector<AllocationRef>& path_items,
    size_t& allocation_index,
    size_t& allocation_offset,
    uint64_t expected_file_offset,
    AsyncMemcpyContext& async_memcpy_context,
    std::vector<SharedInFlightBlock>& pending_blocks
) {
    if (expected_file_offset >= mapping.artifact_size) {
        // All blocks were consumed in the ready phase.
        release_completed_shared_blocks(async_memcpy_context, mapping, pending_blocks, true);
        return;
    }

    const auto start = std::chrono::steady_clock::now();
    SharedRingGlobalHeader* header = shared_ring_global_header(mapping);
    SharedRingBlockHeader* block_headers = shared_ring_block_headers(mapping);

    // For the remaining path we use the same registration-thread pattern.
    // A producer thread finds READY blocks and registers them; the main thread
    // picks up registered blocks and enqueues H2D copies.

    // Shared state between registration thread and main thread.
    struct PendingRegistration {
        size_t block_index;
        uint64_t file_offset;
        uint64_t valid_bytes;
    };
    std::vector<PendingRegistration> registered_queue;
    std::mutex queue_mutex;
    std::atomic<size_t> registered_count{0};
    std::atomic<bool> registration_done{false};

    std::thread register_thread([&]() {
        uint64_t reg_offset = expected_file_offset;
        while (reg_offset < mapping.artifact_size) {
            // Wait for the next block to become READY.
            size_t block_index = SIZE_MAX;
            while (block_index == SIZE_MAX) {
                for (size_t i = 0; i < mapping.block_count; ++i) {
                    const SharedRingBlockHeader& block = block_headers[i];
                    if (atomic_load_u32(&block.state) == static_cast<uint32_t>(SharedRingBlockState::READY) &&
                        atomic_load_u64(&block.file_offset) == reg_offset) {
                        block_index = i;
                        break;
                    }
                }
                if (block_index != SIZE_MAX) {
                    break;
                }
                if (atomic_load_u32(&header->error_code) != 0) {
                    registration_done.store(true, std::memory_order_release);
                    return;
                }
                if (atomic_load_u32(&header->producer_done) == 1) {
                    // Scan once more before giving up.
                    for (size_t i = 0; i < mapping.block_count; ++i) {
                        const SharedRingBlockHeader& block = block_headers[i];
                        if (atomic_load_u32(&block.state) == static_cast<uint32_t>(SharedRingBlockState::READY) &&
                            atomic_load_u64(&block.file_offset) == reg_offset) {
                            block_index = i;
                            break;
                        }
                    }
                    if (block_index == SIZE_MAX) {
                        SIMPLE_CHECK(false, "Shared ring producer completed without providing the expected block");
                    }
                    break;
                }
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }

            atomic_store_u32(&block_headers[block_index].state, static_cast<uint32_t>(SharedRingBlockState::READING));
            ensure_shared_ring_block_registered(mapping, block_index);

            const uint64_t valid_bytes = atomic_load_u64(&block_headers[block_index].valid_bytes);
            {
                std::lock_guard<std::mutex> lock(queue_mutex);
                registered_queue.push_back(PendingRegistration{block_index, reg_offset, valid_bytes});
            }
            registered_count.store(registered_count.load(std::memory_order_relaxed) + 1, std::memory_order_release);
            reg_offset += valid_bytes;
        }
        registration_done.store(true, std::memory_order_release);
    });

    size_t consumed = 0;
    uint64_t resumed_bytes = 0;
    while (expected_file_offset < mapping.artifact_size) {
        // Wait for the registration thread to produce the next block.
        while (registered_count.load(std::memory_order_acquire) <= consumed) {
            release_completed_shared_blocks(async_memcpy_context, mapping, pending_blocks, false);
            std::this_thread::yield();
        }

        PendingRegistration reg;
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            reg = registered_queue[consumed];
        }
        consumed += 1;

        std::unordered_map<int, BatchMemcpyGroup> h2d_groups;
        collect_shared_ring_block_copies(mapping, reg.block_index, path_items, allocation_index, allocation_offset, h2d_groups);
        const std::vector<int> used_devices = get_used_devices(h2d_groups);
        enqueue_batch_memcpy_async(async_memcpy_context, h2d_groups, cudaMemcpyHostToDevice);
        record_shared_block_events(async_memcpy_context, reg.block_index, used_devices, pending_blocks);
        expected_file_offset += reg.valid_bytes;
        resumed_bytes += reg.valid_bytes;
    }

    register_thread.join();
    release_completed_shared_blocks(async_memcpy_context, mapping, pending_blocks, true);
    const auto end = std::chrono::steady_clock::now();
    std::cout << "[torch_memory_saver.cpp] shared ring remaining drain"
              << " artifact_bytes=" << mapping.artifact_size
              << " resumed_bytes=" << resumed_bytes
              << " blocks=" << consumed
              << " elapsed_ms=" << duration_ms(start, end)
              << std::endl;
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
    std::optional<SharedArtifactHostMapping> shared_mapping_to_destroy;
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
                auto shared_it = shared_artifact_mappings_.find(metadata.disk_backup_path);
                if (shared_it != shared_artifact_mappings_.end()) {
                    shared_mapping_to_destroy = std::move(shared_it->second);
                    shared_artifact_mappings_.erase(shared_it);
                }
            }
        }
    }

    DiskOffload::destroy_prefetch_state(prefetch_state_to_destroy);
    if (shared_mapping_to_destroy.has_value()) {
        destroy_shared_artifact_mapping(*shared_mapping_to_destroy);
    }

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
    std::vector<SharedArtifactHostMapping> shared_mappings_to_destroy;

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

        run_batch_memcpy(d2h_groups, cudaMemcpyDeviceToHost, true, "pause_disk_offload");

        for (const auto& entry : disk_items_by_path) {
            auto state = DiskOffload::pop_prefetch_state(disk_prefetch_states_, entry.first);
            if (state != nullptr) {
                prefetch_states_to_destroy.push_back(state);
            }
            auto shared_it = shared_artifact_mappings_.find(entry.first);
            if (shared_it != shared_artifact_mappings_.end()) {
                shared_mappings_to_destroy.push_back(std::move(shared_it->second));
                shared_artifact_mappings_.erase(shared_it);
            }
        }

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

    }

    for (const auto& state : prefetch_states_to_destroy) {
        DiskOffload::destroy_prefetch_state(state);
    }
    for (auto& mapping : shared_mappings_to_destroy) {
        destroy_shared_artifact_mapping(mapping);
    }
#endif
}

void TorchMemorySaver::resume(const std::string& tag) {
#if TMS_ROCM_LEGACY_CHUNKED
    ROCmHIPImplementation::rocm_resume(tag, allocation_metadata_, allocator_metadata_mutex_);

#else
    std::vector<AllocationRef> matching_items;
    std::unordered_map<std::string, std::vector<AllocationRef>> disk_items_by_path;
    std::unordered_map<std::string, std::vector<AllocationRef>> shared_items_by_path;
    std::unordered_map<std::string, std::shared_ptr<DiskPrefetchState>> disk_prefetch_states_for_resume;
    std::unordered_map<int, BatchMemcpyGroup> h2d_groups;
    std::vector<std::shared_ptr<DiskPrefetchState>> states_to_destroy;
    std::vector<std::string> shared_paths_to_release;
    std::vector<SharedArtifactHostMapping> shared_mappings_to_destroy;
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

        std::vector<std::string> shared_backed_paths;
        for (auto& entry : disk_items_by_path) {
            const std::string& path = entry.first;
            const uint64_t total_artifact_bytes = DiskOffload::total_size(entry.second);
            std::cout << "[torch_memory_saver.cpp] resume path candidate"
                      << " path=" << path
                      << " artifact_bytes=" << total_artifact_bytes
                      << " allocations=" << entry.second.size()
                      << std::endl;
            if (ensure_shared_artifact_mapping(shared_artifact_mappings_, path, total_artifact_bytes)) {
                std::cout << "[torch_memory_saver.cpp] resume using shared artifact path"
                          << " path=" << path
                          << " artifact_bytes=" << total_artifact_bytes
                          << std::endl;
                shared_items_by_path.emplace(path, entry.second);
                shared_backed_paths.push_back(path);
                continue;
            }

            std::cout << "[torch_memory_saver.cpp] resume falling back to disk prefetch"
                      << " path=" << path
                      << " artifact_bytes=" << total_artifact_bytes
                      << std::endl;
            auto state_it = disk_prefetch_states_.find(path);
            if (state_it == disk_prefetch_states_.end()) {
                state_it = disk_prefetch_states_.emplace(path, DiskOffload::create_prefetch_state(path, total_artifact_bytes)).first;
            }
            disk_prefetch_states_for_resume.emplace(path, state_it->second);
        }
        for (const std::string& path : shared_backed_paths) {
            disk_items_by_path.erase(path);
        }
    }

    for (const auto& entry : disk_prefetch_states_for_resume) {
        DiskOffload::start_prefetch_if_needed(entry.second);
    }

    const auto remap_start = std::chrono::steady_clock::now();
    size_t remap_allocation_count = 0;
    size_t remap_allocation_bytes = 0;
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
        remap_allocation_count += 1;
        remap_allocation_bytes += metadata.size;
    }
    const auto remap_end = std::chrono::steady_clock::now();
    std::cout << "[torch_memory_saver.cpp] resume remap complete"
              << " allocations=" << remap_allocation_count
              << " bytes=" << remap_allocation_bytes
              << " elapsed_ms=" << duration_ms(remap_start, remap_end)
              << std::endl;

    struct SharedResumeState {
        SharedArtifactHostMapping* mapping;
        std::vector<AllocationRef>* items;
        size_t allocation_index = 0;
        size_t allocation_offset = 0;
        uint64_t resume_offset = 0;
        std::vector<SharedInFlightBlock> pending_blocks;
    };
    std::vector<SharedResumeState> shared_resume_states;
    shared_resume_states.reserve(shared_items_by_path.size());
    for (auto& entry : shared_items_by_path) {
        auto mapping_it = shared_artifact_mappings_.find(entry.first);
        SIMPLE_CHECK(mapping_it != shared_artifact_mappings_.end(), "Expected shared artifact mapping for staged path");
        shared_paths_to_release.push_back(entry.first);
        std::sort(entry.second.begin(), entry.second.end(), [](const AllocationRef& lhs, const AllocationRef& rhs) {
            return lhs.metadata->disk_backup_offset < rhs.metadata->disk_backup_offset;
        });

        SharedResumeState srs;
        srs.mapping = &mapping_it->second;
        srs.items = &entry.second;
        srs.resume_offset = consume_shared_ring_ready(
            *srs.mapping,
            *srs.items,
            srs.allocation_index,
            srs.allocation_offset,
            async_memcpy_context,
            srs.pending_blocks
        );
        shared_resume_states.push_back(std::move(srs));
    }

    // Phase 1: drain any READY disk prefetch slots without blocking on completion.
    // Ring slots are recycled by event queries.
    struct DiskResumeState {
        std::shared_ptr<DiskPrefetchState> prefetch_state;
        std::vector<AllocationRef>* items;
        size_t allocation_index = 0;
        size_t allocation_offset = 0;
        uint64_t resume_offset = 0;
    };
    std::vector<DiskResumeState> disk_resume_states;

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

    const bool allow_batched_h2d = true;
    run_batch_memcpy(h2d_groups, cudaMemcpyHostToDevice, allow_batched_h2d, "resume_initial_h2d");

    // Phase 2: consume the shared ring and any remaining disk slots.
    for (auto& srs : shared_resume_states) {
        consume_shared_ring_remaining(
            *srs.mapping,
            *srs.items,
            srs.allocation_index,
            srs.allocation_offset,
            srs.resume_offset,
            async_memcpy_context,
            srs.pending_blocks
        );
    }

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

        for (const std::string& path : shared_paths_to_release) {
            auto mapping_it = shared_artifact_mappings_.find(path);
            if (mapping_it != shared_artifact_mappings_.end()) {
                shared_mappings_to_destroy.push_back(std::move(mapping_it->second));
                shared_artifact_mappings_.erase(mapping_it);
            }
        }

        for (const auto& entry : disk_prefetch_states_for_resume) {
            auto state = DiskOffload::pop_prefetch_state(disk_prefetch_states_, entry.first);
            if (state != nullptr) {
                states_to_destroy.push_back(state);
            }
        }
    }

    for (auto& mapping : shared_mappings_to_destroy) {
        destroy_shared_artifact_mapping(mapping);
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
