import contextlib
import ctypes
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import uuid

import torch

from torch_memory_saver import torch_memory_saver


@contextlib.contextmanager
def _capture_stdout_fd():
    read_fd, write_fd = os.pipe()
    saved_stdout_fd = os.dup(1)
    try:
        os.dup2(write_fd, 1)
        os.close(write_fd)
        yield read_fd
    finally:
        ctypes.CDLL(None).fflush(None)
        os.dup2(saved_stdout_fd, 1)
        os.close(saved_stdout_fd)


def _read_captured_fd(read_fd: int) -> str:
    chunks = []
    while True:
        chunk = os.read(read_fd, 4096)
        if not chunk:
            break
        chunks.append(chunk)
    os.close(read_fd)
    return b"".join(chunks).decode("utf-8", errors="replace")


def _wait_for_socket(socket_path: str, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if os.path.exists(socket_path):
            return
        time.sleep(0.05)
    raise TimeoutError(f"daemon socket did not appear: {socket_path}")


def _clear_mem_pools() -> None:
    impl = getattr(torch_memory_saver, "_impl", None)
    if impl is None:
        return
    mem_pools = getattr(impl, "_mem_pools", None)
    if mem_pools is not None:
        mem_pools.clear()


def run(hook_mode: str):
    assert hook_mode == "torch"
    torch_memory_saver.hook_mode = hook_mode

    with tempfile.TemporaryDirectory(prefix="tms-shm-daemon-") as tmp_dir:
        tmp_path = Path(tmp_dir)
        artifact_path = tmp_path / f"weights-{uuid.uuid4().hex}.bin"
        socket_path = tmp_path / "daemon.sock"
        trace_path = tmp_path / "daemon-trace.json"

        env = os.environ.copy()
        env["PYTHONPATH"] = os.getcwd() + os.pathsep + env.get("PYTHONPATH", "")
        env["PYTHONNOUSERSITE"] = "1"
        env["TMS_SHM_DAEMON_SOCKET"] = str(socket_path)
        env["TMS_SHM_DAEMON_WATCH_DIRS"] = str(tmp_path)
        env["TMS_SHM_DAEMON_SCAN_INTERVAL_S"] = "0.05"
        env["TMS_SHM_DAEMON_BLOCK_BYTES"] = str(4 * 1024 * 1024)
        env["TMS_SHM_DAEMON_BLOCK_COUNT"] = "4"
        env["TMS_SHM_DAEMON_TRACE_FILE"] = str(trace_path)

        daemon_proc = subprocess.Popen(
            [sys.executable, "-m", "torch_memory_saver.shm_stager_daemon"],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            _wait_for_socket(str(socket_path), timeout_s=10.0)

            old_socket = os.environ.get("TMS_SHM_DAEMON_SOCKET")
            os.environ["TMS_SHM_DAEMON_SOCKET"] = str(socket_path)
            try:
                with torch_memory_saver.region(tag="disk_weights", disk_backup_loc=str(artifact_path)):
                    disk_tensor = torch.arange(4_000_000, dtype=torch.float32, device="cuda")
                disk_expected = disk_tensor.cpu()

                torch_memory_saver.pause("disk_weights")
                assert artifact_path.exists()
                assert Path(f"{artifact_path}.complete").exists()

                with _capture_stdout_fd() as read_fd:
                    torch_memory_saver.resume("disk_weights")
                captured = _read_captured_fd(read_fd)

                assert "shared artifact mapping ready" in captured
                assert torch.equal(disk_tensor.cpu(), disk_expected)
                assert torch_memory_saver.get_cpu_backup(disk_tensor) is None
                _clear_mem_pools()
            finally:
                if old_socket is None:
                    os.environ.pop("TMS_SHM_DAEMON_SOCKET", None)
                else:
                    os.environ["TMS_SHM_DAEMON_SOCKET"] = old_socket
        finally:
            daemon_proc.terminate()
            try:
                daemon_proc.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                daemon_proc.kill()
                daemon_proc.wait(timeout=10.0)

            if daemon_proc.returncode not in (0, -15):
                raise RuntimeError(f"daemon exited unexpectedly with code {daemon_proc.returncode}")
