// Pipelined compressed restore benchmark — CUDA C++ with nvcomp batch API.
//
// Proves that byte-shuffle + nvcomp deflate, pipelined across blocks with
// CUDA streams, achieves higher effective H2D throughput than raw transfer.
//
// Build: see Makefile
// Run:   ./compressed_restore [--size-mib N] [--block-mib N] [--iters N]

#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <curand.h>
#include <nvcomp/gdeflate.h>

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,        \
              cudaGetErrorString(err));                                         \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#define CHECK_NVCOMP(call)                                                     \
  do {                                                                         \
    nvcompStatus_t s = (call);                                                 \
    if (s != nvcompSuccess) {                                                  \
      fprintf(stderr, "nvcomp error at %s:%d: code=%d\n", __FILE__, __LINE__, \
              (int)s);                                                         \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// ---------------------------------------------------------------------------
// Byte-shuffle kernels
// ---------------------------------------------------------------------------

// Split interleaved BF16 bytes: [h0,l0,h1,l1,...] -> [h0,h1,...,l0,l1,...]
__global__ void byte_shuffle_kernel(const uint8_t *in, uint8_t *out, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  size_t half = n / 2;
  if (i < half) {
    out[i] = in[2 * i];           // high bytes
    out[half + i] = in[2 * i + 1]; // low bytes
  }
}

// Inverse: [h0,h1,...,l0,l1,...] -> [h0,l0,h1,l1,...]
__global__ void byte_unshuffle_kernel(const uint8_t *in, uint8_t *out,
                                      size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  size_t half = n / 2;
  if (i < half) {
    out[2 * i] = in[i];
    out[2 * i + 1] = in[half + i];
  }
}

void byte_shuffle(const uint8_t *d_in, uint8_t *d_out, size_t n,
                  cudaStream_t stream = 0) {
  int threads = 256;
  int blocks = ((n / 2) + threads - 1) / threads;
  byte_shuffle_kernel<<<blocks, threads, 0, stream>>>(d_in, d_out, n);
}

void byte_unshuffle(const uint8_t *d_in, uint8_t *d_out, size_t n,
                    cudaStream_t stream = 0) {
  int threads = 256;
  int blocks = ((n / 2) + threads - 1) / threads;
  byte_unshuffle_kernel<<<blocks, threads, 0, stream>>>(d_in, d_out, n);
}

// ---------------------------------------------------------------------------
// Compressed block: holds compressed chunks for one pipeline block
// ---------------------------------------------------------------------------

static constexpr size_t CHUNK_SIZE = 65536; // 64KB — nvcomp deflate max

struct CompressedBlock {
  // Host-side (pinned): packed compressed chunks
  uint8_t *h_packed;
  size_t packed_bytes;

  // Per-chunk metadata
  std::vector<size_t> compressed_sizes; // actual compressed size per chunk
  std::vector<size_t> original_sizes;   // original (uncompressed) size per chunk
  size_t total_original;
  size_t num_chunks;
};

// ---------------------------------------------------------------------------
// Compress a GPU buffer into blocks of 64KB chunks
// ---------------------------------------------------------------------------

std::vector<CompressedBlock>
compress_data(const uint8_t *d_shuffled, size_t total_bytes,
              size_t block_size) {
  std::vector<CompressedBlock> blocks;
  nvcompBatchedGdeflateCompressOpts_t comp_opts = {0, {}};

  size_t max_compressed_chunk;
  CHECK_NVCOMP(nvcompBatchedGdeflateCompressGetMaxOutputChunkSize(
      CHUNK_SIZE, comp_opts, &max_compressed_chunk));

  size_t offset = 0;
  while (offset < total_bytes) {
    size_t block_bytes = std::min(block_size, total_bytes - offset);
    size_t num_chunks = (block_bytes + CHUNK_SIZE - 1) / CHUNK_SIZE;

    // Build pointer/size arrays on host, then copy to device
    std::vector<const void *> h_in_ptrs(num_chunks);
    std::vector<size_t> h_in_sizes(num_chunks);
    for (size_t c = 0; c < num_chunks; c++) {
      size_t chunk_off = c * CHUNK_SIZE;
      h_in_ptrs[c] = d_shuffled + offset + chunk_off;
      h_in_sizes[c] = std::min(CHUNK_SIZE, block_bytes - chunk_off);
    }

    // Allocate device arrays
    void **d_in_ptrs;
    size_t *d_in_sizes;
    CHECK_CUDA(cudaMalloc(&d_in_ptrs, num_chunks * sizeof(void *)));
    CHECK_CUDA(cudaMalloc(&d_in_sizes, num_chunks * sizeof(size_t)));
    CHECK_CUDA(cudaMemcpy(d_in_ptrs, h_in_ptrs.data(),
                           num_chunks * sizeof(void *),
                           cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_in_sizes, h_in_sizes.data(),
                           num_chunks * sizeof(size_t),
                           cudaMemcpyHostToDevice));

    // Allocate compressed output buffers (one per chunk, max size)
    std::vector<void *> h_out_ptrs(num_chunks);
    uint8_t *d_out_buf;
    CHECK_CUDA(cudaMalloc(&d_out_buf, num_chunks * max_compressed_chunk));
    for (size_t c = 0; c < num_chunks; c++)
      h_out_ptrs[c] = d_out_buf + c * max_compressed_chunk;

    void **d_out_ptrs;
    size_t *d_out_sizes;
    CHECK_CUDA(cudaMalloc(&d_out_ptrs, num_chunks * sizeof(void *)));
    CHECK_CUDA(cudaMalloc(&d_out_sizes, num_chunks * sizeof(size_t)));
    CHECK_CUDA(cudaMemcpy(d_out_ptrs, h_out_ptrs.data(),
                           num_chunks * sizeof(void *),
                           cudaMemcpyHostToDevice));

    // Temp buffer
    size_t temp_bytes;
    CHECK_NVCOMP(nvcompBatchedGdeflateCompressGetTempSizeAsync(
        num_chunks, CHUNK_SIZE, comp_opts, &temp_bytes, block_bytes));
    void *d_temp;
    CHECK_CUDA(cudaMalloc(&d_temp, temp_bytes));

    // Compress
    CHECK_NVCOMP(nvcompBatchedGdeflateCompressAsync(
        (const void *const *)d_in_ptrs, d_in_sizes, CHUNK_SIZE, num_chunks,
        d_temp, temp_bytes, (void *const *)d_out_ptrs, d_out_sizes, comp_opts,
        nullptr, 0));
    CHECK_CUDA(cudaDeviceSynchronize());

    // Read back compressed sizes
    std::vector<size_t> comp_sizes(num_chunks);
    CHECK_CUDA(cudaMemcpy(comp_sizes.data(), d_out_sizes,
                           num_chunks * sizeof(size_t),
                           cudaMemcpyDeviceToHost));

    // Pack compressed chunks contiguously into pinned host memory
    size_t packed_total = 0;
    for (size_t c = 0; c < num_chunks; c++)
      packed_total += comp_sizes[c];

    CompressedBlock blk;
    CHECK_CUDA(cudaMallocHost(&blk.h_packed, packed_total));
    size_t pack_off = 0;
    for (size_t c = 0; c < num_chunks; c++) {
      CHECK_CUDA(cudaMemcpy(blk.h_packed + pack_off,
                             (uint8_t *)h_out_ptrs[c], comp_sizes[c],
                             cudaMemcpyDeviceToHost));
      pack_off += comp_sizes[c];
    }
    blk.packed_bytes = packed_total;
    blk.compressed_sizes = std::move(comp_sizes);
    blk.original_sizes.resize(num_chunks);
    for (size_t c = 0; c < num_chunks; c++)
      blk.original_sizes[c] = h_in_sizes[c];
    blk.total_original = block_bytes;
    blk.num_chunks = num_chunks;
    blocks.push_back(std::move(blk));

    cudaFree(d_in_ptrs);
    cudaFree(d_in_sizes);
    cudaFree(d_out_buf);
    cudaFree(d_out_ptrs);
    cudaFree(d_out_sizes);
    cudaFree(d_temp);
    offset += block_bytes;
  }
  return blocks;
}

// ---------------------------------------------------------------------------
// Decompress + unshuffle a block from a GPU staging buffer
// ---------------------------------------------------------------------------

// Kernel to build pointer arrays on GPU from base address + offsets.
// Eliminates H2D metadata copies from the decompress stream.
__global__ void build_ptr_array(void **ptrs, const uint8_t *base,
                                const size_t *offsets, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    ptrs[i] = (void *)(base + offsets[i]);
}

struct BlockMeta {
  // Pre-uploaded to GPU (once, during preparation)
  size_t *d_comp_offsets;  // byte offset of each compressed chunk within packed block
  size_t *d_comp_sizes;
  size_t *d_decomp_offsets; // byte offset of each decompressed chunk within decomp buf
  size_t *d_decomp_sizes;
  size_t num_chunks;
};

struct DecompressCtx {
  void **d_comp_ptrs;
  void **d_decomp_ptrs;
  void *d_temp;
  uint8_t *d_decomp_buf;
  size_t temp_bytes;
  size_t max_chunks;
  size_t max_block_bytes;

  void init(size_t max_chunks_, size_t max_block_bytes_) {
    max_chunks = max_chunks_;
    max_block_bytes = max_block_bytes_;
    CHECK_CUDA(cudaMalloc(&d_comp_ptrs, max_chunks * sizeof(void *)));
    CHECK_CUDA(cudaMalloc(&d_decomp_ptrs, max_chunks * sizeof(void *)));
    CHECK_CUDA(cudaMalloc(&d_decomp_buf, max_block_bytes));

    nvcompBatchedGdeflateDecompressOpts_t opts = {
        NVCOMP_DECOMPRESS_BACKEND_DEFAULT, {}};
    CHECK_NVCOMP(nvcompBatchedGdeflateDecompressGetTempSizeAsync(
        max_chunks, CHUNK_SIZE, opts, &temp_bytes, max_block_bytes));
    CHECK_CUDA(cudaMalloc(&d_temp, temp_bytes));
  }

  void free() {
    cudaFree(d_comp_ptrs);
    cudaFree(d_decomp_ptrs);
    cudaFree(d_decomp_buf);
    cudaFree(d_temp);
  }
};

// Pre-upload block metadata to GPU (called once during preparation, not on hot path)
BlockMeta upload_block_meta(const CompressedBlock &blk) {
  BlockMeta meta;
  meta.num_chunks = blk.num_chunks;
  size_t nc = blk.num_chunks;

  std::vector<size_t> comp_offsets(nc), decomp_offsets(nc);
  size_t co = 0, do_ = 0;
  for (size_t c = 0; c < nc; c++) {
    comp_offsets[c] = co;
    decomp_offsets[c] = do_;
    co += blk.compressed_sizes[c];
    do_ += blk.original_sizes[c];
  }

  CHECK_CUDA(cudaMalloc(&meta.d_comp_offsets, nc * sizeof(size_t)));
  CHECK_CUDA(cudaMalloc(&meta.d_comp_sizes, nc * sizeof(size_t)));
  CHECK_CUDA(cudaMalloc(&meta.d_decomp_offsets, nc * sizeof(size_t)));
  CHECK_CUDA(cudaMalloc(&meta.d_decomp_sizes, nc * sizeof(size_t)));

  CHECK_CUDA(cudaMemcpy(meta.d_comp_offsets, comp_offsets.data(),
                         nc * sizeof(size_t), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(meta.d_comp_sizes, blk.compressed_sizes.data(),
                         nc * sizeof(size_t), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(meta.d_decomp_offsets, decomp_offsets.data(),
                         nc * sizeof(size_t), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(meta.d_decomp_sizes, blk.original_sizes.data(),
                         nc * sizeof(size_t), cudaMemcpyHostToDevice));
  return meta;
}

void decompress_and_unshuffle(const uint8_t *d_staging,
                              const BlockMeta &meta, DecompressCtx &ctx,
                              uint8_t *d_output, cudaStream_t stream) {
  size_t nc = meta.num_chunks;
  int threads = 256;
  int blocks = (nc + threads - 1) / threads;

  // Build pointer arrays on GPU — no H2D copies needed
  build_ptr_array<<<blocks, threads, 0, stream>>>(
      ctx.d_comp_ptrs, d_staging, meta.d_comp_offsets, nc);
  build_ptr_array<<<blocks, threads, 0, stream>>>(
      ctx.d_decomp_ptrs, ctx.d_decomp_buf, meta.d_decomp_offsets, nc);

  nvcompBatchedGdeflateDecompressOpts_t opts = {
      NVCOMP_DECOMPRESS_BACKEND_DEFAULT, {}};
  CHECK_NVCOMP(nvcompBatchedGdeflateDecompressAsync(
      (const void *const *)ctx.d_comp_ptrs, meta.d_comp_sizes,
      meta.d_decomp_sizes, nullptr, nc, ctx.d_temp, ctx.temp_bytes,
      (void *const *)ctx.d_decomp_ptrs, opts, nullptr, stream));

  // Last block may be smaller than nc * CHUNK_SIZE — caller passes actual size via d_output range
  // We use total decompressed bytes: sum of decomp sizes. For uniform chunks this equals
  // the block's total_original. Pass it explicitly.
}

void decompress_unshuffle_block(const uint8_t *d_staging,
                                const CompressedBlock &blk,
                                const BlockMeta &meta, DecompressCtx &ctx,
                                uint8_t *d_output, cudaStream_t stream) {
  decompress_and_unshuffle(d_staging, meta, ctx, d_output, stream);
  byte_unshuffle(ctx.d_decomp_buf, d_output, blk.total_original, stream);
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

double bench_raw_h2d(const uint8_t *h_data, size_t total_bytes,
                     uint8_t *d_dst, int warmup, int iters) {
  for (int i = 0; i < warmup; i++) {
    CHECK_CUDA(cudaMemcpy(d_dst, h_data, total_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaDeviceSynchronize());
  }
  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iters; i++) {
    CHECK_CUDA(cudaMemcpy(d_dst, h_data, total_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaDeviceSynchronize());
  }
  auto t1 = std::chrono::high_resolution_clock::now();
  double secs =
      std::chrono::duration<double>(t1 - t0).count();
  return ((double)total_bytes * iters / (1ULL << 30)) / secs;
}

double bench_sequential(const std::vector<CompressedBlock> &blocks,
                        const std::vector<BlockMeta> &metas,
                        size_t total_bytes, DecompressCtx &ctx,
                        uint8_t *d_staging, uint8_t *d_output, int warmup,
                        int iters) {
  for (int w = 0; w < warmup; w++) {
    size_t out_off = 0;
    for (size_t bi = 0; bi < blocks.size(); bi++) {
      CHECK_CUDA(cudaMemcpy(d_staging, blocks[bi].h_packed,
                             blocks[bi].packed_bytes, cudaMemcpyHostToDevice));
      decompress_unshuffle_block(d_staging, blocks[bi], metas[bi], ctx,
                                 d_output + out_off, 0);
      CHECK_CUDA(cudaDeviceSynchronize());
      out_off += blocks[bi].total_original;
    }
  }

  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iters; i++) {
    size_t out_off = 0;
    for (size_t bi = 0; bi < blocks.size(); bi++) {
      CHECK_CUDA(cudaMemcpy(d_staging, blocks[bi].h_packed,
                             blocks[bi].packed_bytes, cudaMemcpyHostToDevice));
      decompress_unshuffle_block(d_staging, blocks[bi], metas[bi], ctx,
                                 d_output + out_off, 0);
      CHECK_CUDA(cudaDeviceSynchronize());
      out_off += blocks[bi].total_original;
    }
  }
  auto t1 = std::chrono::high_resolution_clock::now();
  return ((double)total_bytes * iters / (1ULL << 30)) /
         std::chrono::duration<double>(t1 - t0).count();
}

double bench_pipelined(const std::vector<CompressedBlock> &blocks,
                       const std::vector<BlockMeta> &metas,
                       size_t total_bytes, DecompressCtx ctx[2],
                       uint8_t *d_staging0, uint8_t *d_staging1,
                       uint8_t *d_output, int warmup, int iters) {
  cudaStream_t stream_h2d, stream_gpu;
  CHECK_CUDA(cudaStreamCreateWithFlags(&stream_h2d, cudaStreamNonBlocking));
  CHECK_CUDA(cudaStreamCreateWithFlags(&stream_gpu, cudaStreamNonBlocking));

  uint8_t *staging[2] = {d_staging0, d_staging1};

  // Track when each staging buffer's decompress finishes, so H2D doesn't
  // overwrite a buffer still being read by the GPU.
  cudaEvent_t decomp_done[2];
  CHECK_CUDA(cudaEventCreate(&decomp_done[0]));
  CHECK_CUDA(cudaEventCreate(&decomp_done[1]));

  auto run_once = [&]() {
    size_t out_off = 0;
    cudaEvent_t prev_h2d_done = nullptr;

    for (size_t i = 0; i < blocks.size(); i++) {
      auto &blk = blocks[i];
      int buf = i % 2;

      // Wait for any prior decompress using this staging buffer to finish
      // before overwriting it with new H2D data.
      if (i >= 2)
        CHECK_CUDA(cudaStreamWaitEvent(stream_h2d, decomp_done[buf]));

      // H2D on stream_h2d
      CHECK_CUDA(cudaMemcpyAsync(staging[buf], blk.h_packed, blk.packed_bytes,
                                  cudaMemcpyHostToDevice, stream_h2d));
      cudaEvent_t h2d_done;
      CHECK_CUDA(cudaEventCreate(&h2d_done));
      CHECK_CUDA(cudaEventRecord(h2d_done, stream_h2d));

      // Decompress previous block on stream_gpu (overlaps with current H2D)
      if (i > 0) {
        int prev_buf = (i - 1) % 2;
        CHECK_CUDA(cudaStreamWaitEvent(stream_gpu, prev_h2d_done));
        decompress_unshuffle_block(staging[prev_buf], blocks[i - 1],
                                   metas[i - 1], ctx[prev_buf],
                                   d_output + out_off - blocks[i - 1].total_original,
                                   stream_gpu);
        CHECK_CUDA(cudaEventRecord(decomp_done[prev_buf], stream_gpu));
      }

      if (prev_h2d_done)
        CHECK_CUDA(cudaEventDestroy(prev_h2d_done));
      prev_h2d_done = h2d_done;
      out_off += blk.total_original;
    }

    // Last block
    int last_buf = (blocks.size() - 1) % 2;
    CHECK_CUDA(cudaStreamWaitEvent(stream_gpu, prev_h2d_done));
    decompress_unshuffle_block(staging[last_buf], blocks.back(),
                               metas.back(), ctx[last_buf],
                               d_output + out_off - blocks.back().total_original,
                               stream_gpu);
    CHECK_CUDA(cudaEventRecord(decomp_done[last_buf], stream_gpu));
    CHECK_CUDA(cudaEventDestroy(prev_h2d_done));
    CHECK_CUDA(cudaDeviceSynchronize());
  };

  for (int w = 0; w < warmup; w++)
    run_once();

  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iters; i++)
    run_once();
  auto t1 = std::chrono::high_resolution_clock::now();

  CHECK_CUDA(cudaEventDestroy(decomp_done[0]));
  CHECK_CUDA(cudaEventDestroy(decomp_done[1]));
  CHECK_CUDA(cudaStreamDestroy(stream_h2d));
  CHECK_CUDA(cudaStreamDestroy(stream_gpu));

  return ((double)total_bytes * iters / (1ULL << 30)) /
         std::chrono::duration<double>(t1 - t0).count();
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char **argv) {
  size_t size_mib = 512;
  size_t block_mib = 64;
  int iters = 10;

  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--size-mib") && i + 1 < argc)
      size_mib = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--block-mib") && i + 1 < argc)
      block_mib = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--iters") && i + 1 < argc)
      iters = atoi(argv[++i]);
  }

  size_t total_bytes = size_mib << 20;
  size_t block_size = block_mib << 20;

  cudaDeviceProp prop;
  CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
  printf("Device: %s\n", prop.name);
  printf("Data: %zu MiB BF16, block: %zu MiB, chunks: 64 KB\n", size_mib,
         block_mib);
  printf("\n");

  // Generate realistic BF16 weight data: normal(0, 0.02) converted to BF16.
  // BF16 = upper 16 bits of float32: sign(1) + exponent(8) + mantissa(7).
  // Small values cluster high bytes around 0x00/0x80 → low entropy → compresses well.
  printf("Generating synthetic BF16 weight data...\n");
  uint8_t *d_raw;
  CHECK_CUDA(cudaMalloc(&d_raw, total_bytes));
  {
    size_t n_bf16 = total_bytes / 2;
    // Generate float32 normals, then truncate to BF16
    size_t n_floats = n_bf16;
    // curandGenerateNormal requires even count
    if (n_floats % 2 != 0) n_floats++;
    float *d_float;
    CHECK_CUDA(cudaMalloc(&d_float, n_floats * sizeof(float)));
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 42);
    curandGenerateNormal(gen, d_float, n_floats, 0.0f, 0.02f);
    curandDestroyGenerator(gen);

    // Convert float32 → BF16 on CPU (upper 16 bits of each float)
    std::vector<float> h_float(n_bf16);
    CHECK_CUDA(cudaMemcpy(h_float.data(), d_float, n_bf16 * sizeof(float),
                           cudaMemcpyDeviceToHost));
    std::vector<uint16_t> h_bf16(n_bf16);
    for (size_t i = 0; i < n_bf16; i++) {
      union { float f; uint32_t u; } v;
      v.f = h_float[i];
      h_bf16[i] = (uint16_t)(v.u >> 16);
    }
    CHECK_CUDA(cudaMemcpy(d_raw, h_bf16.data(), total_bytes,
                           cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaFree(d_float));
  }

  // Byte-shuffle (per block)
  printf("Byte-shuffling per block...\n");
  uint8_t *d_shuffled;
  CHECK_CUDA(cudaMalloc(&d_shuffled, total_bytes));
  for (size_t off = 0; off < total_bytes; off += block_size) {
    size_t bsz = std::min(block_size, total_bytes - off);
    byte_shuffle(d_raw + off, d_shuffled + off, bsz);
  }
  CHECK_CUDA(cudaDeviceSynchronize());

  // Compress
  printf("Compressing with nvcomp deflate...\n");
  auto blocks = compress_data(d_shuffled, total_bytes, block_size);

  size_t total_compressed = 0;
  for (auto &b : blocks)
    total_compressed += b.packed_bytes;
  double ratio = (double)total_compressed / total_bytes;
  printf("  Blocks: %zu, ratio: %.3f (%.1f%% reduction)\n", blocks.size(),
         ratio, (1 - ratio) * 100);
  printf("\n");

  // Pre-upload block metadata to GPU (once, not on hot path)
  std::vector<BlockMeta> metas;
  for (auto &blk : blocks)
    metas.push_back(upload_block_meta(blk));

  // Pinned host copy of raw data for H2D benchmark
  uint8_t *h_raw;
  CHECK_CUDA(cudaMallocHost(&h_raw, total_bytes));
  CHECK_CUDA(
      cudaMemcpy(h_raw, d_raw, total_bytes, cudaMemcpyDeviceToHost));

  // Verify correctness
  printf("Verifying round-trip...\n");
  uint8_t *d_output;
  CHECK_CUDA(cudaMalloc(&d_output, total_bytes));

  size_t max_packed = 0, max_chunks = 0;
  for (auto &b : blocks) {
    max_packed = std::max(max_packed, b.packed_bytes);
    max_chunks = std::max(max_chunks, b.num_chunks);
  }

  uint8_t *d_staging;
  CHECK_CUDA(cudaMalloc(&d_staging, max_packed));
  DecompressCtx ctx;
  ctx.init(max_chunks, block_size);

  size_t out_off = 0;
  for (size_t bi = 0; bi < blocks.size(); bi++) {
    CHECK_CUDA(cudaMemcpy(d_staging, blocks[bi].h_packed,
                           blocks[bi].packed_bytes, cudaMemcpyHostToDevice));
    decompress_unshuffle_block(d_staging, blocks[bi], metas[bi], ctx,
                               d_output + out_off, 0);
    CHECK_CUDA(cudaDeviceSynchronize());
    out_off += blocks[bi].total_original;
  }

  // Compare
  std::vector<uint8_t> h_raw_check(total_bytes), h_out_check(total_bytes);
  CHECK_CUDA(cudaMemcpy(h_raw_check.data(), d_raw, total_bytes,
                         cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(h_out_check.data(), d_output, total_bytes,
                         cudaMemcpyDeviceToHost));
  bool match = (h_raw_check == h_out_check);
  printf("  Round-trip: %s\n\n", match ? "OK" : "FAILED");
  if (!match) {
    for (size_t i = 0; i < total_bytes; i++) {
      if (h_raw_check[i] != h_out_check[i]) {
        printf("  First mismatch at byte %zu: got=%u exp=%u\n", i,
               h_out_check[i], h_raw_check[i]);
        break;
      }
    }
    return 1;
  }

  // Benchmarks
  printf("Benchmarking (%d iterations)...\n\n", iters);

  double raw_gbps = bench_raw_h2d(h_raw, total_bytes, d_output, 3, iters);
  printf("  Raw H2D:              %.2f GB/s\n", raw_gbps);

  double seq_gbps = bench_sequential(blocks, metas, total_bytes, ctx,
                                     d_staging, d_output, 2, iters);
  printf("  Sequential compressed: %.2f GB/s effective\n", seq_gbps);

  // Pipelined needs 2 staging buffers and 2 decompress contexts
  uint8_t *d_staging2;
  CHECK_CUDA(cudaMalloc(&d_staging2, max_packed));
  DecompressCtx ctx2[2];
  ctx2[0].init(max_chunks, block_size);
  ctx2[1].init(max_chunks, block_size);

  double pipe_gbps = bench_pipelined(blocks, metas, total_bytes, ctx2,
                                     d_staging, d_staging2, d_output, 2, iters);
  printf("  Pipelined compressed:  %.2f GB/s effective\n", pipe_gbps);

  // Verify pipelined output matches original
  CHECK_CUDA(cudaMemcpy(h_out_check.data(), d_output, total_bytes,
                         cudaMemcpyDeviceToHost));
  bool pipe_match = (h_raw_check == h_out_check);
  printf("  Pipelined round-trip:  %s\n", pipe_match ? "OK" : "FAILED");
  if (!pipe_match) {
    size_t mismatches = 0;
    for (size_t i = 0; i < total_bytes; i++)
      if (h_raw_check[i] != h_out_check[i]) mismatches++;
    printf("  %zu/%zu bytes differ\n", mismatches, total_bytes);
    return 1;
  }

  // Summary
  printf("\n");
  printf("============================================================\n");
  printf("RESULTS\n");
  printf("============================================================\n");
  printf("  Data:                 %zu MiB BF16\n", size_mib);
  printf("  Compression ratio:    %.3f (%.1f%% PCIe reduction)\n", ratio,
         (1 - ratio) * 100);
  printf("  Raw H2D:              %.2f GB/s\n", raw_gbps);
  printf("  Sequential compressed: %.2f GB/s effective\n", seq_gbps);
  printf("  Pipelined compressed:  %.2f GB/s effective\n", pipe_gbps);
  printf("\n");
  printf("  Pipelined vs raw:     %.2fx\n", pipe_gbps / raw_gbps);
  printf("  Sequential vs raw:    %.2fx\n", seq_gbps / raw_gbps);
  if (pipe_gbps > raw_gbps)
    printf("  PIPELINED WINS: +%.1f%% effective throughput\n",
           (pipe_gbps / raw_gbps - 1) * 100);
  else
    printf("  PIPELINED LOSES: -%.1f%% effective throughput\n",
           (1 - pipe_gbps / raw_gbps) * 100);

  // Cleanup
  ctx.free();
  ctx2[0].free();
  ctx2[1].free();
  cudaFree(d_raw);
  cudaFree(d_shuffled);
  cudaFree(d_output);
  cudaFree(d_staging);
  cudaFree(d_staging2);
  cudaFreeHost(h_raw);
  for (auto &b : blocks)
    cudaFreeHost(b.h_packed);

  return 0;
}
