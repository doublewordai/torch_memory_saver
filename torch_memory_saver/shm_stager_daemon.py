import argparse
import array
import contextlib
import ctypes
import errno
import json
import mmap
import os
from pathlib import Path
import socket
import socketserver
import struct
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from typing import Optional


K_DEFAULT_SOCKET = "/tmp/torch_memory_saver_shm_daemon.sock"
K_DEFAULT_TRACE_FILE = ""
K_DEFAULT_HUGETLB_DIR = ""
K_DEFAULT_SCAN_INTERVAL_S = 1.0
K_DEFAULT_BLOCK_BYTES = 512 * 1024 * 1024
K_DEFAULT_BLOCK_COUNT = 8
K_DEFAULT_MAX_STAGED_BYTES = 0
K_ARTIFACT_COMPLETE_SUFFIX = ".complete"
K_DIRECT_IO_ALIGNMENT = 4096
K_HUGEPAGE_ALIGNMENT = 2 * 1024 * 1024
K_RING_MAGIC = b"TMSRING\0"
K_RING_VERSION = 1
K_BLOCK_STATE_FREE = 0
K_BLOCK_STATE_WRITING = 1
K_BLOCK_STATE_READY = 2
K_BLOCK_STATE_READING = 3
K_BLOCK_STATE_ERROR = 4
K_CONSUMER_IDLE = 0
K_CONSUMER_ATTACHED = 1
K_CONSUMER_DONE = 2
K_GLOBAL_FORMAT = "<8sIIIIQQQQQIIII"
K_BLOCK_FORMAT = "<IIQQQ"
K_GLOBAL_BYTES = struct.calcsize(K_GLOBAL_FORMAT)
K_BLOCK_HEADER_BYTES = struct.calcsize(K_BLOCK_FORMAT)


def align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def artifact_signature(path: str, stat_result: os.stat_result) -> str:
    del stat_result
    real = os.path.realpath(path)
    sanitized = []
    for ch in real:
        if ch.isalnum():
            sanitized.append(ch.lower())
        else:
            sanitized.append("_")
    signature = "".join(sanitized).strip("_")
    if not signature:
        signature = "artifact"
    if len(signature) > 64:
        signature = signature[-64:]
    return signature


def shm_name_for_signature(signature: str) -> str:
    return f"/tms_{signature}"


def completion_generation(token: str) -> int:
    try:
        return int(token)
    except ValueError:
        return 0


@dataclass(frozen=True)
class CompletionMarker:
    expected_size: int
    token: str


def monotonic_ns() -> int:
    return time.monotonic_ns()


class TraceWriter:
    def __init__(self, path: str) -> None:
        self._path = path
        self._enabled = bool(path)
        self._lock = threading.Lock()
        self._first_event = True
        self._stream = None
        if self._enabled:
            self._stream = open(path, "w", encoding="utf-8")
            self._stream.write('{"traceEvents":[')

    def now_us(self) -> int:
        return time.time_ns() // 1000

    def write_complete(self, category: str, name: str, start_us: int, duration_us: int, args: dict[str, object]) -> None:
        if not self._enabled or self._stream is None:
            return
        event = {
            "name": name,
            "cat": category,
            "ph": "X",
            "ts": start_us,
            "dur": duration_us,
            "pid": os.getpid(),
            "tid": threading.get_ident(),
            "args": args,
        }
        with self._lock:
            if not self._first_event:
                self._stream.write(",")
            self._first_event = False
            self._stream.write(json.dumps(event, separators=(",", ":")))
            self._stream.flush()

    def close(self) -> None:
        if self._stream is None:
            return
        self._stream.write("]}")
        self._stream.flush()
        self._stream.close()
        self._stream = None


class TraceScope(contextlib.AbstractContextManager):
    def __init__(self, writer: TraceWriter, category: str, name: str, **args: object) -> None:
        self.writer = writer
        self.category = category
        self.name = name
        self.args = dict(args)
        self.start_us = writer.now_us()

    def add_arg(self, key: str, value: object) -> None:
        self.args[key] = value

    def __exit__(self, exc_type, exc, tb) -> None:
        self.writer.write_complete(self.category, self.name, self.start_us, self.writer.now_us() - self.start_us, self.args)
        return False


class Libc:
    def __init__(self) -> None:
        self.libc = ctypes.CDLL(None, use_errno=True)
        self.libc.pread.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_longlong]
        self.libc.pread.restype = ctypes.c_ssize_t

    def pread_into(self, fd: int, address: int, size: int, offset: int) -> None:
        cursor = address
        remaining = size
        current_offset = offset
        while remaining > 0:
            read_bytes = self.libc.pread(fd, ctypes.c_void_p(cursor), remaining, current_offset)
            if read_bytes < 0:
                err = ctypes.get_errno()
                if err == errno.EINTR:
                    continue
                raise OSError(err, os.strerror(err))
            if read_bytes == 0:
                raise RuntimeError("unexpected EOF while staging artifact")
            cursor += read_bytes
            remaining -= read_bytes
            current_offset += read_bytes


@dataclass
class Entry:
    artifact_path: str
    signature: str
    shm_name: str
    artifact_size: int
    block_payload_bytes: int
    block_count: int
    payload_offset: int
    mapped_size: int
    state: str = "loading"
    error: Optional[str] = None
    shm_fd: int = -1
    shm_map: Optional[mmap.mmap] = None
    host_address: int = 0
    backing_kind: str = "memfd"
    backing_path: Optional[str] = None
    next_read_offset: int = 0
    producer_done: bool = False
    last_access_ns: int = 0
    consumer_attached: bool = False
    completion_token: str = ""
    staging: bool = False

    def close(self) -> None:
        if self.shm_map is not None:
            self.shm_map.close()
            self.shm_map = None
        if self.shm_fd >= 0:
            os.close(self.shm_fd)
            self.shm_fd = -1
        if self.backing_path is not None:
            try:
                os.unlink(self.backing_path)
            except FileNotFoundError:
                pass
            self.backing_path = None


class SharedRingLayout:
    def __init__(self, block_payload_bytes: int, block_count: int) -> None:
        self.block_payload_bytes = align_up(max(block_payload_bytes, K_DIRECT_IO_ALIGNMENT), K_DIRECT_IO_ALIGNMENT)
        self.block_count = max(2, block_count)
        self.headers_bytes = align_up(K_GLOBAL_BYTES + self.block_count * K_BLOCK_HEADER_BYTES, K_DIRECT_IO_ALIGNMENT)
        self.mapped_size = align_up(self.headers_bytes + self.block_count * self.block_payload_bytes, K_HUGEPAGE_ALIGNMENT)


class StageManager:
    def __init__(
        self,
        block_bytes: int,
        block_count: int,
        trace_file: str,
        hugetlb_dir: str,
        require_hugetlb: bool,
        watch_dirs: list[str],
        scan_interval_s: float,
        max_staged_bytes: int,
    ) -> None:
        self.layout = SharedRingLayout(block_bytes, block_count)
        self.libc = Libc()
        self.lock = threading.Lock()
        self.entries: dict[str, Entry] = {}
        self.trace = TraceWriter(trace_file)
        self.hugetlb_dir = hugetlb_dir
        self.require_hugetlb = require_hugetlb
        self.watch_dirs = [str(Path(path)) for path in watch_dirs if path]
        self.scan_interval_s = scan_interval_s
        self.max_staged_bytes = max_staged_bytes
        self.stop_event = threading.Event()
        self.scan_thread: Optional[threading.Thread] = None
        if self.watch_dirs:
            self.scan_thread = threading.Thread(target=self._scan_loop, daemon=True)
            self.scan_thread.start()

    def close(self) -> None:
        self.stop_event.set()
        if self.scan_thread is not None:
            self.scan_thread.join(timeout=1.0)
        with self.lock:
            entries = list(self.entries.values())
            self.entries.clear()
        for entry in entries:
            entry.close()
        self.trace.close()

    def _entry_mapped_bytes(self, entry: Entry) -> int:
        if entry.shm_map is None or entry.shm_fd < 0:
            return 0
        return entry.mapped_size

    def _current_staged_bytes(self) -> int:
        return sum(self._entry_mapped_bytes(entry) for entry in self.entries.values())

    def _evict_if_needed(self, bytes_needed: int) -> None:
        if self.max_staged_bytes <= 0:
            return
        while self._current_staged_bytes() + bytes_needed > self.max_staged_bytes:
            candidates = [
                entry
                for entry in self.entries.values()
                if not entry.consumer_attached
            ]
            if not candidates:
                raise RuntimeError("hugepage staging budget exhausted")
            victim = min(candidates, key=lambda entry: entry.last_access_ns)
            self.entries.pop(victim.signature, None)
            victim.close()

    def _allocate_memfd(self, entry: Entry) -> tuple[int, mmap.mmap, int, str, Optional[str]]:
        shm_fd = os.memfd_create(entry.shm_name.lstrip("/"), os.MFD_CLOEXEC)
        os.ftruncate(shm_fd, entry.mapped_size)
        shm_map = mmap.mmap(shm_fd, entry.mapped_size, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE)
        host_address = ctypes.addressof(ctypes.c_char.from_buffer(shm_map))
        return shm_fd, shm_map, host_address, "memfd", None

    def _allocate_hugetlb(self, entry: Entry) -> tuple[int, mmap.mmap, int, str, Optional[str]]:
        if not self.hugetlb_dir:
            raise RuntimeError("hugetlb directory is not configured")
        hugetlb_path = Path(self.hugetlb_dir)
        hugetlb_path.mkdir(parents=True, exist_ok=True)
        backing_path = hugetlb_path / entry.shm_name.lstrip("/")
        shm_fd = os.open(backing_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            os.ftruncate(shm_fd, entry.mapped_size)
            shm_map = mmap.mmap(shm_fd, entry.mapped_size, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE)
        except Exception:
            os.close(shm_fd)
            try:
                os.unlink(backing_path)
            except FileNotFoundError:
                pass
            raise
        host_address = ctypes.addressof(ctypes.c_char.from_buffer(shm_map))
        return shm_fd, shm_map, host_address, "hugetlb", str(backing_path)

    def _allocate_backing(self, entry: Entry) -> tuple[int, mmap.mmap, int, str, Optional[str]]:
        if not self.hugetlb_dir:
            return self._allocate_memfd(entry)
        try:
            with TraceScope(self.trace, "daemon", "hugetlb_create_and_map", artifact_path=entry.artifact_path, mapped_size=entry.mapped_size):
                return self._allocate_hugetlb(entry)
        except Exception:
            if self.require_hugetlb:
                raise
            return self._allocate_memfd(entry)

    def _ensure_entry_backing_locked(self, entry: Entry) -> None:
        if entry.shm_map is not None and entry.shm_fd >= 0 and entry.host_address != 0:
            self._initialize_ring(entry)
            return
        shm_fd, shm_map, host_address, backing_kind, backing_path = self._allocate_backing(entry)
        entry.shm_fd = shm_fd
        entry.shm_map = shm_map
        entry.host_address = host_address
        entry.backing_kind = backing_kind
        entry.backing_path = backing_path
        self._initialize_ring(entry)

    def _global_pack(self, entry: Entry, producer_done: int, consumer_state: int, error_code: int) -> bytes:
        return struct.pack(
            K_GLOBAL_FORMAT,
            K_RING_MAGIC,
            K_RING_VERSION,
            K_GLOBAL_BYTES,
            K_BLOCK_HEADER_BYTES,
            entry.block_count,
            entry.block_payload_bytes,
            entry.payload_offset,
            entry.mapped_size,
            entry.artifact_size,
            completion_generation(entry.completion_token),
            producer_done,
            consumer_state,
            error_code,
            0,
        )

    def _write_global(self, entry: Entry, *, producer_done: Optional[int] = None, consumer_state: Optional[int] = None, error_code: Optional[int] = None) -> None:
        assert entry.shm_map is not None
        current = struct.unpack_from(K_GLOBAL_FORMAT, entry.shm_map, 0)
        packed = self._global_pack(
            entry,
            current[10] if producer_done is None else producer_done,
            current[11] if consumer_state is None else consumer_state,
            current[12] if error_code is None else error_code,
        )
        struct.pack_into(K_GLOBAL_FORMAT, entry.shm_map, 0, *struct.unpack(K_GLOBAL_FORMAT, packed))

    def _read_consumer_state(self, entry: Entry) -> int:
        assert entry.shm_map is not None
        return struct.unpack_from(K_GLOBAL_FORMAT, entry.shm_map, 0)[11]

    def _block_header_offset(self, block_index: int) -> int:
        return K_GLOBAL_BYTES + block_index * K_BLOCK_HEADER_BYTES

    def _payload_offset_for(self, entry: Entry, block_index: int) -> int:
        return entry.payload_offset + block_index * entry.block_payload_bytes

    def _read_block(self, entry: Entry, block_index: int) -> tuple[int, int, int, int]:
        assert entry.shm_map is not None
        state, _reserved0, file_offset, valid_bytes, sequence = struct.unpack_from(
            K_BLOCK_FORMAT, entry.shm_map, self._block_header_offset(block_index)
        )
        return state, file_offset, valid_bytes, sequence

    def _write_block(self, entry: Entry, block_index: int, state: int, file_offset: int, valid_bytes: int, sequence: int) -> None:
        assert entry.shm_map is not None
        struct.pack_into(
            K_BLOCK_FORMAT,
            entry.shm_map,
            self._block_header_offset(block_index),
            state,
            0,
            file_offset,
            valid_bytes,
            sequence,
        )

    def _initialize_ring(self, entry: Entry) -> None:
        assert entry.shm_map is not None
        struct.pack_into(K_GLOBAL_FORMAT, entry.shm_map, 0, *struct.unpack(K_GLOBAL_FORMAT, self._global_pack(entry, 0, K_CONSUMER_IDLE, 0)))
        for block_index in range(entry.block_count):
            self._write_block(entry, block_index, K_BLOCK_STATE_FREE, 0, 0, 0)

    def _cleanup_replaced_entries(self, artifact_path: str, signature: str) -> None:
        stale_keys = [
            key for key, existing in self.entries.items()
            if existing.artifact_path == artifact_path and existing.signature != signature
        ]
        for key in stale_keys:
            entry = self.entries.pop(key)
            entry.close()

    def _entry_needs_restage(self, entry: Entry) -> bool:
        if entry.shm_map is None:
            return False
        consumer_state = self._read_consumer_state(entry)
        return consumer_state == K_CONSUMER_DONE

    def _drop_entry_locked(self, entry: Entry) -> None:
        self.entries.pop(entry.signature, None)
        entry.close(self.cuda)

    def _replace_entry_for_marker_locked(
        self,
        existing: Optional[Entry],
        artifact_path: str,
        stat_result: os.stat_result,
        signature: str,
        marker: CompletionMarker,
    ) -> Optional[Entry]:
        if existing is not None and existing.staging:
            return existing
        if existing is not None and existing.completion_token == marker.token and existing.artifact_size == stat_result.st_size:
            existing.last_access_ns = monotonic_ns()
            if self._entry_needs_restage(existing):
                self._ensure_entry_backing_locked(existing)
                existing.state = "streaming"
                existing.error = None
                existing.producer_done = False
                existing.consumer_attached = False
                existing.staging = True
                thread = threading.Thread(target=self._stage_entry, args=(existing, True), daemon=True)
                thread.start()
            return existing
        if existing is not None:
            # A new completion token for the same artifact path means a new checkpoint
            # generation is authoritative, even if the old ring still claims ATTACHED.
            # The old consumer may have died without flipping the in-band state back, and
            # returning the stale generation causes the client to miss and fall back to disk.
            if existing.shm_map is not None and existing.shm_fd >= 0 and existing.host_address != 0:
                existing.artifact_size = stat_result.st_size
                existing.completion_token = marker.token
                self._ensure_entry_backing_locked(existing)
                existing.state = "streaming"
                existing.error = None
                existing.producer_done = False
                existing.consumer_attached = False
                existing.last_access_ns = monotonic_ns()
                existing.staging = True
                thread = threading.Thread(target=self._stage_entry, args=(existing, True), daemon=True)
                thread.start()
                return existing
            self._drop_entry_locked(existing)
        entry = Entry(
            artifact_path=artifact_path,
            signature=signature,
            shm_name=shm_name_for_signature(signature),
            artifact_size=stat_result.st_size,
            block_payload_bytes=self.layout.block_payload_bytes,
            block_count=self.layout.block_count,
            payload_offset=self.layout.headers_bytes,
            mapped_size=self.layout.mapped_size,
            last_access_ns=monotonic_ns(),
            completion_token=marker.token,
            staging=True,
        )
        self._evict_if_needed(entry.mapped_size)
        self._ensure_entry_backing_locked(entry)
        entry.state = "streaming"
        self.entries[signature] = entry
        thread = threading.Thread(target=self._stage_entry, args=(entry,), daemon=True)
        thread.start()
        return entry

    def ensure_staged(self, artifact_path: str, marker: CompletionMarker) -> Optional[Entry]:
        stat_result = os.stat(artifact_path)
        if marker.expected_size and stat_result.st_size < marker.expected_size:
            return None
        signature = artifact_signature(artifact_path, stat_result)
        with self.lock:
            self._cleanup_replaced_entries(artifact_path, signature)
            entry = self.entries.get(signature)
            if entry is not None and (
                entry.state == "error" or
                entry.artifact_size != stat_result.st_size
            ):
                self._drop_entry_locked(entry)
                entry = None
            return self._replace_entry_for_marker_locked(entry, artifact_path, stat_result, signature, marker)

    def lookup(self, artifact_path: str, expected_size: int) -> Optional[Entry]:
        try:
            stat_result = os.stat(artifact_path)
        except FileNotFoundError:
            return None
        if expected_size and stat_result.st_size < expected_size:
            return None
        marker = self._read_completion_marker_for_artifact(artifact_path)
        if marker is not None:
            entry = self.ensure_staged(artifact_path, marker)
            if entry is None:
                return None
            entry.last_access_ns = monotonic_ns()
            return entry
        signature = artifact_signature(artifact_path, stat_result)
        with self.lock:
            self._cleanup_replaced_entries(artifact_path, signature)
            entry = self.entries.get(signature)
            if entry is not None and entry.state == "error":
                self._drop_entry_locked(entry)
                entry = None
            if entry is not None and entry.artifact_size != stat_result.st_size:
                self._drop_entry_locked(entry)
                entry = None
        if entry is None:
            return None
        if expected_size and stat_result.st_size < expected_size:
            return None
        entry.last_access_ns = monotonic_ns()
        return entry

    def _scan_loop(self) -> None:
        while not self.stop_event.is_set():
            for watch_dir in self.watch_dirs:
                try:
                    for child in Path(watch_dir).iterdir():
                        if not child.is_file() or child.suffix != K_ARTIFACT_COMPLETE_SUFFIX:
                            continue
                        try:
                            marker = self._read_completion_marker(child)
                            artifact_path = str(child.with_suffix(""))
                            if not os.path.isfile(artifact_path):
                                continue
                            self.ensure_staged(artifact_path, marker)
                        except Exception:
                            continue
                except FileNotFoundError:
                    continue
            self.stop_event.wait(self.scan_interval_s)

    def _read_completion_marker(self, marker_path: Path) -> CompletionMarker:
        content = marker_path.read_text(encoding="utf-8").strip()
        if not content:
            return CompletionMarker(expected_size=0, token="")
        parts = content.split("\t")
        expected_size = int(parts[0])
        token = parts[1] if len(parts) > 1 else content
        return CompletionMarker(expected_size=expected_size, token=token)

    def _read_completion_marker_for_artifact(self, artifact_path: str) -> Optional[CompletionMarker]:
        marker_path = Path(f"{artifact_path}{K_ARTIFACT_COMPLETE_SUFFIX}")
        try:
            return self._read_completion_marker(marker_path)
        except FileNotFoundError:
            return None

    def _stage_entry(self, entry: Entry, reuse_existing: bool = False) -> None:
        phase = "open_artifact"
        next_read_offset = 0
        try:
            with TraceScope(self.trace, "daemon", "stage_entry", artifact_path=entry.artifact_path, artifact_size=entry.artifact_size, block_payload_bytes=entry.block_payload_bytes, block_count=entry.block_count):
                file_fd = os.open(entry.artifact_path, os.O_RDONLY | os.O_DIRECT)
                try:
                    if entry.shm_map is None or entry.shm_fd < 0 or entry.host_address == 0:
                        phase = "allocate_backing"
                        shm_fd, shm_map, host_address, backing_kind, backing_path = self._allocate_backing(entry)
                        entry.shm_fd = shm_fd
                        entry.shm_map = shm_map
                        entry.host_address = host_address
                        entry.backing_kind = backing_kind
                        entry.backing_path = backing_path
                    if entry.shm_map is None or entry.shm_fd < 0 or entry.host_address == 0:
                        raise RuntimeError("shared backing is unavailable for staging")
                    phase = "initialize_ring"
                    self._initialize_ring(entry)
                    entry.state = "streaming"

                    phase = "posix_fadvise"
                    os.posix_fadvise(file_fd, 0, 0, os.POSIX_FADV_SEQUENTIAL)
                    os.posix_fadvise(file_fd, 0, 0, os.POSIX_FADV_WILLNEED)
                    sequence = 1
                    while next_read_offset < entry.artifact_size:
                        chosen_block = None
                        while chosen_block is None:
                            for block_index in range(entry.block_count):
                                block_state, _file_offset, _valid_bytes, _sequence = self._read_block(entry, block_index)
                                if block_state == K_BLOCK_STATE_FREE:
                                    chosen_block = block_index
                                    break
                            if chosen_block is not None:
                                break
                            if self.stop_event.is_set():
                                raise RuntimeError("stager stopping")
                            consumer_state = self._read_consumer_state(entry)
                            entry.consumer_attached = consumer_state == K_CONSUMER_ATTACHED
                            time.sleep(0.001)

                        valid_bytes = min(entry.block_payload_bytes, entry.artifact_size - next_read_offset)
                        aligned_bytes = align_up(valid_bytes, K_DIRECT_IO_ALIGNMENT)
                        self._write_block(entry, chosen_block, K_BLOCK_STATE_WRITING, next_read_offset, valid_bytes, sequence)
                        payload_address = entry.host_address + self._payload_offset_for(entry, chosen_block)
                        phase = "pread_into"
                        with TraceScope(self.trace, "daemon", "disk_read_into_ring_block", artifact_path=entry.artifact_path, block_index=chosen_block, file_offset=next_read_offset, valid_bytes=valid_bytes, aligned_bytes=aligned_bytes):
                            self.libc.pread_into(file_fd, payload_address, aligned_bytes, next_read_offset)
                        self._write_block(entry, chosen_block, K_BLOCK_STATE_READY, next_read_offset, valid_bytes, sequence)
                        next_read_offset += valid_bytes
                        sequence += 1

                    entry.producer_done = True
                    phase = "producer_done"
                    self._write_global(entry, producer_done=1)
                    entry.state = "ready"
                    entry.staging = False
                finally:
                    os.close(file_fd)
        except Exception as exc:  # noqa: BLE001
            try:
                if entry.shm_map is not None:
                    self._write_global(entry, error_code=1)
            except Exception:
                pass
            try:
                entry.close()
            except Exception:
                pass
            entry.error = f"{phase}: {exc}"
            entry.state = "error"
            entry.staging = False
            with self.lock:
                current = self.entries.get(entry.signature)
                if current is entry:
                    self.entries.pop(entry.signature, None)
            print(
                f"[torch_memory_saver.shm_daemon] stage failed path={entry.artifact_path} phase={phase} "
                f"offset={next_read_offset} error={exc}\n{traceback.format_exc()}",
                file=sys.stderr,
                flush=True,
            )


class Handler(socketserver.StreamRequestHandler):
    def _send_response(self, payload: str, fd: Optional[int] = None) -> None:
        data = payload.encode("utf-8")
        if fd is None:
            self.wfile.write(data)
            return
        ancillary = [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", [fd]).tobytes())]
        self.request.sendmsg([data], ancillary)

    def handle(self) -> None:
        line = self.rfile.readline().decode("utf-8").rstrip("\n")
        if not line:
            return
        parts = line.split("\t")
        if len(parts) != 3:
            self.wfile.write(b"error\tbad_request\n")
            return

        command, artifact_path, artifact_size_text = parts
        artifact_size = int(artifact_size_text)
        manager: StageManager = self.server.manager  # type: ignore[attr-defined]

        try:
            if command != "lookup":
                self._send_response("error\tunknown_command\n")
                return

            entry = manager.lookup(artifact_path, artifact_size)
            if entry is None:
                self._send_response("missing\n")
                return
            if entry.state == "error":
                self._send_response(f"error\t{entry.error}\n")
                return
            if entry.shm_fd < 0:
                self._send_response("loading\n")
                return

            if entry.shm_map is not None:
                manager._write_global(entry, consumer_state=K_CONSUMER_ATTACHED)
                entry.consumer_attached = True

            self._send_response(
                f"{entry.state}\t{entry.shm_name}\t{entry.completion_token}\t{entry.artifact_size}\t{entry.mapped_size}\t{entry.block_count}\t{entry.block_payload_bytes}\t{entry.payload_offset}\n",
                entry.shm_fd,
            )
        except Exception as exc:  # noqa: BLE001
            self._send_response(f"error\t{exc}\n")


class Server(socketserver.ThreadingUnixStreamServer):
    allow_reuse_address = True

    def __init__(self, socket_path: str, manager: StageManager) -> None:
        self.manager = manager
        super().__init__(socket_path, Handler)
        os.chmod(socket_path, 0o666)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket-path", default=os.environ.get("TMS_SHM_DAEMON_SOCKET", K_DEFAULT_SOCKET))
    parser.add_argument("--trace-file", default=os.environ.get("TMS_SHM_DAEMON_TRACE_FILE", K_DEFAULT_TRACE_FILE))
    parser.add_argument("--hugetlb-dir", default=os.environ.get("TMS_SHM_DAEMON_HUGETLB_DIR", K_DEFAULT_HUGETLB_DIR))
    parser.add_argument("--require-hugetlb", action="store_true", default=os.environ.get("TMS_SHM_DAEMON_REQUIRE_HUGETLB", "0") == "1")
    parser.add_argument("--watch-dirs", default=os.environ.get("TMS_SHM_DAEMON_WATCH_DIRS", ""))
    parser.add_argument("--scan-interval-s", type=float, default=float(os.environ.get("TMS_SHM_DAEMON_SCAN_INTERVAL_S", K_DEFAULT_SCAN_INTERVAL_S)))
    parser.add_argument("--block-bytes", type=int, default=int(os.environ.get("TMS_SHM_DAEMON_BLOCK_BYTES", K_DEFAULT_BLOCK_BYTES)))
    parser.add_argument("--block-count", type=int, default=int(os.environ.get("TMS_SHM_DAEMON_BLOCK_COUNT", K_DEFAULT_BLOCK_COUNT)))
    parser.add_argument("--max-staged-bytes", type=int, default=int(os.environ.get("TMS_SHM_DAEMON_MAX_STAGED_BYTES", K_DEFAULT_MAX_STAGED_BYTES)))
    args = parser.parse_args()

    socket_path = Path(args.socket_path)
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    if socket_path.exists():
        socket_path.unlink()

    manager = StageManager(
        block_bytes=args.block_bytes,
        block_count=args.block_count,
        trace_file=args.trace_file,
        hugetlb_dir=args.hugetlb_dir,
        require_hugetlb=args.require_hugetlb,
        watch_dirs=args.watch_dirs.split(":") if args.watch_dirs else [],
        scan_interval_s=args.scan_interval_s,
        max_staged_bytes=args.max_staged_bytes,
    )
    try:
        with Server(str(socket_path), manager) as server:
            server.serve_forever()
    finally:
        manager.close()


if __name__ == "__main__":
    main()
