# Patch Notes

This fork is ahead of the upstream `README.md`. This file documents the behavior added by the current patch set.

## Summary

- `region()` and `cuda_graph()` accept `disk_backup_loc` in addition to `tag` and `enable_cpu_backup`.
- The native allocator tracks disk-backup metadata per allocation and can restore paused tensors from local disk.
- Disk-backed backups can optionally be accelerated by a daemon-owned shared-memory ring buffer.
- The daemon can watch checkpoint directories directly; TMS falls back to direct disk resume when the daemon path is unavailable.
- Thread-local init state can now be seeded from environment variables for disk-backed regions.
- The patch includes examples, tests, and Kubernetes deployment examples for weight offload workflows.

## Python API Delta

### `torch_memory_saver.region(...)`

New signature:

```python
with torch_memory_saver.region(
    tag="default",
    enable_cpu_backup=False,
    disk_backup_loc=None,
):
    ...
```

Rules:

- `enable_cpu_backup` and `disk_backup_loc` are mutually exclusive.
- `disk_backup_loc` points at the file used to persist paused contents for the region.
- when `disk_backup_loc` is omitted, TMS can fall back to a tag-scoped env default via `TMS_TAG_DISK_BACKUP_LOC_<TAG>`

### `torch_memory_saver.cuda_graph(...)`

`cuda_graph()` accepts the same `disk_backup_loc` argument as `region()`.

The existing restriction still applies:

- pauseable CUDA graph capture is only supported in `hook_mode="preload"`.

### `torch_memory_saver.get_cpu_backup(tensor, zero_copy=False)`

Behavior is now:

- returns a CPU tensor for paused allocations backed by legacy `enable_cpu_backup=True`
- returns `None` for active allocations
- returns `None` for paused allocations backed by `disk_backup_loc`

## Backup Semantics

### Legacy CPU Backup

`enable_cpu_backup=True` keeps the old behavior:

- copy tensor contents to pinned host memory on `pause()`
- restore them on `resume()`
- free the host copy after resume

This is the safest option when contents may change between pause/resume cycles.

### Disk Backup

`disk_backup_loc="..."` writes paused contents to a file and restores from that file later.

Pause behavior:

- on the first materialization, matching allocations are copied to host memory
- they are packed into a single backup file at `disk_backup_loc`
- each allocation records its byte offset inside that file
- the temporary host copy is dropped after the backup file is written

Resume behavior:

- data is read back from disk into a pinned host ring buffer
- ready chunks are copied back to GPU, with batched H2D dispatch where possible

Intended use:

- large, stable tensors such as model weights when GPU memory matters more than local disk bandwidth

Important constraint:

- the current implementation reuses an existing disk backup after it has been materialized
- if a tensor is modified after resume, a later pause will not automatically rewrite the backup file
- use this backend for data that should be treated as immutable across pause/resume cycles

## Disk Resume Path

The native layer has an explicit direct-disk resume pipeline:

- one resume state is maintained per disk-backup path
- a producer thread reads the backup file with `O_DIRECT`
- reads land in pinned host ring-buffer slots
- `resume()` drains ready slots into a batched H2D copy set
- remaining slots are consumed as they become ready

The ring buffer is controlled by environment variables:

- `TMS_DISK_RING_BUFFER_BYTES`
  default: `134217728` bytes (128 MiB) per slot before alignment
- `TMS_DISK_RING_SLOTS`
  default: `4`

## Shared-Memory Staging Daemon

There is now a daemonized fast path for disk backups:

- `pause()` only materializes the disk backup file
- the daemon watches checkpoint directories and stages backup files on its own
- the daemon owns a hugepage-backed shared ring buffer per staged backup
- the daemon reads backup data into that ring with `O_DIRECT`
- each ring block has in-band state and offset metadata
- on `resume()`, TMS first asks the daemon whether a staged ring exists for the backup file
- if it does, the daemon passes the backing FD over the Unix socket immediately, the current process `mmap()`s it, `cudaHostRegister()`s it in-process, and restores from the shared ring while the daemon continues staging later blocks
- if no staged ring is ready, `resume()` falls back to the existing direct disk prefetch path

The daemon itself stays out of CUDA entirely:

- it does not create a CUDA context
- it does not call `cudaHostRegister()`
- host registration happens only in the consumer process after attach

Daemon entry point:

- `torch-memory-saver-shm-daemon`

Daemon-related environment variables:

- `TMS_SHM_DAEMON_SOCKET`
  unix-domain socket used by `resume()` to talk to the daemon
- `TMS_SHM_DAEMON_WATCH_DIRS`
  colon-separated checkpoint directories the daemon scans and stages from
- `TMS_SHM_DAEMON_SCAN_INTERVAL_S`
  polling interval used when scanning watch directories
- `TMS_SHM_DAEMON_HUGETLB_DIR`
  if set, the daemon allocates staged shared buffers from this hugepage-backed directory instead of normal `memfd`
- `TMS_SHM_DAEMON_REQUIRE_HUGETLB`
  if set to `1`, fail staging rather than falling back to normal-page `memfd` when hugepage allocation fails
- `TMS_SHM_DAEMON_BLOCK_BYTES`
  payload bytes per shared ring block
- `TMS_SHM_DAEMON_BLOCK_COUNT`
  number of shared ring blocks per staged backup
- `TMS_SHM_DAEMON_MAX_STAGED_BYTES`
  optional daemon-wide cap for staged hugepage-backed buffers before LRU eviction
- `TMS_SHM_DAEMON_TRACE_FILE`
  if set, the daemon writes a Perfetto/Chrome trace JSON file covering ring allocation and disk-read chunks

Shared ring behavior:

- the ring buffer is a single-producer, single-consumer structure
- the daemon is the producer
- TMS is the consumer
- block headers and the global header are kept separate from payload bytes
- block state transitions are coordinated via shared atomic metadata

The current ring protocol uses:

- `FREE`
- `WRITING`
- `READY`
- `READING`
- `ERROR`

The current daemon implementation restages a backup after the previous consumer marks the ring `DONE`.

## Environment Variables

New thread-local init variable:

- `TMS_INIT_DISK_BACKUP_PATH`

New tag-scoped env default:

- `TMS_TAG_DISK_BACKUP_LOC_<TAG>`
  - used only when `disk_backup_loc` is omitted
  - `<TAG>` is uppercased and non-alphanumeric runs become `_`
  - example: `tag="weights"` maps to `TMS_TAG_DISK_BACKUP_LOC_WEIGHTS`
  - example: `tag="kv_cache"` maps to `TMS_TAG_DISK_BACKUP_LOC_KV_CACHE`

New consumer-side resume variable:

- `TMS_RESUME_TRANSFER_MODE`
  - `registered` (default): `cudaHostRegister` each shared ring block before DMA, pipelining registration with in-flight transfers. Best on hardware where the CUDA driver's pageable staging path is slow (observed on B200 / PCIe Gen5 with driver 580.x: ~8 GB/s staging vs ~39 GB/s pipelined).
  - `direct`: skip registration and let the CUDA runtime stage through its internal bounce buffer. Best on hardware where the staging path already saturates PCIe (observed on RTX 4090 / PCIe Gen4 with driver 590.x: ~24 GB/s staging vs ~25 GB/s PCIe ceiling).

Existing init variables still apply:

- `TMS_INIT_ENABLE`
- `TMS_INIT_ENABLE_CPU_BACKUP`

Nested region overrides restore the prior tag, interesting-region flag, CPU-backup flag, and disk-backup path on exit.

## Native Runtime Changes

The C++ layer now tracks additional metadata per allocation:

- disk-backup path
- byte offset inside the disk backup file

Operational changes:

- `pause()` and `resume()` group copies by device and use batched memcpy dispatch when available
- `free()` can clean up allocations that are already paused
- per-path disk prefetch state is torn down when no matching disk-backed allocations remain
- the daemon path tears down any attached shared mapping once resume completes

## Examples And Tests Added Or Updated

- `test/examples/disk_backups.py`
  exercises disk-backed pause/resume
- `test/test_examples.py`
  includes the new disk-backup example
- `test/examples/nested_region.py`
  validates nested region config restoration in preload mode
- `test/examples/training_engine.py`
  covers `disable()` with init-time region enablement

## Practical Guidance

Use these modes based on data lifecycle:

- `enable_cpu_backup=True`
  when correctness matters and tensor contents may change between pauses
- `disk_backup_loc="..."`
  when contents are stable and host RAM pressure matters more than disk IO cost

For the daemonized shared-memory path, run the daemon ahead of time and set `TMS_SHM_DAEMON_SOCKET`; `resume()` will consume the staged ring when it is ready and otherwise fall back to direct disk.

## Kubernetes Deployment Shape

The intended production model is:

- one `DaemonSet` pod per node running `torch-memory-saver-shm-daemon`
- the daemon pod owns the node-local hugepage budget
- consumer pods mount the daemon socket directory and the checkpoint directory via `hostPath`
- TMS in the consumer pod talks to the daemon only through the Unix socket

Important operational requirements:

- the daemon pod must request `hugepages-2Mi`
- the daemon pod must have access to the node's hugepage mount, typically `/dev/hugepages`
- consumer pods do not need hugepage resources, but they must run on the same node as the daemon instance serving their checkpoint path
- the checkpoint directory must be node-local and identically mounted into the daemon and consumer pods

Concrete examples:

- [`examples/k8s/tms-shm-daemonset.yaml`](examples/k8s/tms-shm-daemonset.yaml)
- [`examples/k8s/sglang-consumer-pod.yaml`](examples/k8s/sglang-consumer-pod.yaml)

Container packaging:

- [`Dockerfile.daemon`](Dockerfile.daemon) builds a minimal CUDA-runtime image for `torch-memory-saver-shm-daemon`

## Profiling

The daemon can optionally emit a trace file:

- set `TMS_SHM_DAEMON_TRACE_FILE=/tmp/tms-daemon.trace.json`
- open the resulting JSON in Perfetto or Chrome tracing
