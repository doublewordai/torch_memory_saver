# Patch Notes

This fork is ahead of the upstream `README.md`. This file documents the behavior added by the current patch set.

## Summary

- `region()` and `cuda_graph()` now accept artifact storage options in addition to `tag` and `enable_cpu_backup`.
- A new `preload()` API starts asynchronous disk prefetch before `resume()`.
- The native allocator tracks artifact metadata per allocation and can restore paused tensors from RAM or disk artifacts.
- Thread-local init state can now be seeded from environment variables for artifact-backed regions.
- The patch includes new examples, tests, and a benchmark harness for weight offload workflows.

## Python API Delta

### `torch_memory_saver.region(...)`

New signature:

```python
with torch_memory_saver.region(
    tag="default",
    enable_cpu_backup=False,
    artifact_backend=None,
    artifact_path=None,
):
    ...
```

Rules:

- `artifact_backend` may be `None`, `"ram"`, or `"disk"`.
- `enable_cpu_backup` and `artifact_backend` are mutually exclusive.
- `artifact_path` is only valid when `artifact_backend == "disk"`.
- `artifact_backend == "disk"` requires `artifact_path`.

### `torch_memory_saver.cuda_graph(...)`

`cuda_graph()` accepts the same artifact arguments as `region()`.

The existing restriction still applies:

- pauseable CUDA graph capture is only supported in `hook_mode="preload"`.

### `torch_memory_saver.preload(tag=None)`

New API:

```python
torch_memory_saver.preload("weights")
```

This starts asynchronous disk reads for paused allocations that use `artifact_backend="disk"`. It is optional; `resume()` will still work without it, but preloading lets the producer thread fill the ring buffer ahead of demand.

### `torch_memory_saver.get_cpu_backup(tensor, zero_copy=False)`

Behavior is now:

- returns a CPU tensor for paused allocations backed by legacy `enable_cpu_backup=True`
- returns a CPU tensor for paused allocations backed by `artifact_backend="ram"`
- returns `None` for active allocations
- returns `None` for paused allocations backed by `artifact_backend="disk"`

## Artifact Backend Semantics

### Legacy CPU Backup

`enable_cpu_backup=True` keeps the old behavior:

- copy tensor contents to pinned host memory on `pause()`
- restore them on `resume()`
- free the host copy after resume

This is the safest option when contents may change between pause/resume cycles.

### RAM Artifact Backend

`artifact_backend="ram"` keeps a host-side artifact in pinned memory.

What changes versus legacy CPU backup:

- the host artifact is retained after `resume()`
- `get_cpu_backup()` can expose that paused host copy
- the path is optimized for repeated offload/resume of stable data

Intended use:

- model weights or other data that is expected to stay unchanged while the artifact is reused

Important constraint:

- the current implementation reuses an existing RAM artifact instead of forcing a fresh device-to-host copy on every later `pause()`
- treat this backend as suitable for content that is effectively immutable between pauses

### Disk Artifact Backend

`artifact_backend="disk"` writes paused contents to a file and restores from that file later.

Pause behavior:

- on the first materialization, matching allocations are copied to host memory
- they are packed into a single artifact file at `artifact_path`
- each allocation records its byte offset inside that file
- the temporary host copy is dropped after the artifact is written

Resume behavior:

- data is read back from disk into a pinned host ring buffer
- ready chunks are copied back to GPU, with batched H2D dispatch where possible

Intended use:

- large, stable tensors such as model weights when GPU memory matters more than local disk bandwidth

Important constraint:

- the current implementation reuses an existing disk artifact after it has been materialized
- if a tensor is modified after resume, a later pause will not automatically rewrite the artifact file
- use this backend for data that should be treated as immutable across pause/resume cycles

## Disk Prefetch Path

The native layer now has an explicit disk prefetch pipeline:

- `preload()` discovers paused disk-backed allocations by tag
- one prefetch state is maintained per artifact path
- a producer thread reads the artifact file with `O_DIRECT`
- reads land in pinned host ring-buffer slots
- `resume()` first drains already-ready slots into a batched H2D copy set
- any remaining slots are consumed as they become ready

The ring buffer is controlled by environment variables:

- `TMS_DISK_RING_BUFFER_BYTES`
  default: `134217728` bytes (128 MiB) per slot before alignment
- `TMS_DISK_RING_SLOTS`
  default: `4`

## Environment Variables

New thread-local init variables:

- `TMS_INIT_ARTIFACT_BACKEND`
- `TMS_INIT_ARTIFACT_PATH`

Existing init variables still apply:

- `TMS_INIT_ENABLE`
- `TMS_INIT_ENABLE_CPU_BACKUP`

Nested region overrides restore the prior tag, interesting-region flag, CPU-backup flag, artifact backend, and artifact path on exit.

## Native Runtime Changes

The C++ layer now tracks additional metadata per allocation:

- artifact backend
- artifact path
- artifact byte offset inside a disk artifact

Operational changes:

- `pause()` and `resume()` group copies by device and use batched memcpy dispatch when available
- `free()` can clean up allocations that are already paused
- per-path disk prefetch state is torn down when no matching disk-backed allocations remain

## Examples And Tests Added Or Updated

- `test/examples/artifact_backends.py`
  exercises RAM artifacts, disk artifacts, and `preload()`
- `test/test_examples.py`
  includes the new artifact backend example
- `test/examples/nested_region.py`
  validates nested region config restoration in preload mode
- `test/examples/training_engine.py`
  covers `disable()` with init-time region enablement
- `.benchmarks/hf_weight_offload_bench.py`
  benchmark harness for Hugging Face model weight offload
- `scripts/dev_env.sh`
  helper to activate the repo-local environment with `PYTHONNOUSERSITE=1`

## Practical Guidance

Use these modes based on data lifecycle:

- `enable_cpu_backup=True`
  when correctness matters and tensor contents may change between pauses
- `artifact_backend="ram"`
  when contents are stable and you want faster repeated resume without keeping GPU memory resident
- `artifact_backend="disk"`
  when contents are stable and host RAM pressure matters more than disk IO cost

For disk artifacts, call `preload(tag)` shortly before `resume(tag)` when you want to overlap file IO with other work.
