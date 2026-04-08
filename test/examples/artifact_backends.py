import os
import tempfile
import time
import uuid

import torch

from torch_memory_saver import torch_memory_saver


def run(hook_mode: str):
    assert hook_mode == "torch"
    torch_memory_saver.hook_mode = hook_mode

    with torch_memory_saver.region(tag="weights", artifact_backend="ram"):
        ram_tensor = torch.arange(4_000_000, dtype=torch.float32, device="cuda")
    ram_expected = ram_tensor.cpu()

    torch_memory_saver.pause("weights")
    assert torch.equal(torch_memory_saver.get_cpu_backup(ram_tensor), ram_expected)
    torch_memory_saver.resume("weights")
    assert torch.equal(ram_tensor.cpu(), ram_expected)

    artifact_path = os.path.join(tempfile.gettempdir(), f"tms_artifact_{uuid.uuid4().hex}.bin")
    try:
        with torch_memory_saver.region(tag="disk_weights", artifact_backend="disk", artifact_path=artifact_path):
            disk_tensor = torch.arange(4_000_000, dtype=torch.float32, device="cuda")
        disk_expected = disk_tensor.cpu()

        torch_memory_saver.pause("disk_weights")
        assert os.path.exists(artifact_path)
        torch_memory_saver.resume("disk_weights")
        assert torch.equal(disk_tensor.cpu(), disk_expected)

        torch_memory_saver.pause("disk_weights")
        torch_memory_saver.preload("disk_weights")
        time.sleep(0.1)
        assert torch_memory_saver.get_cpu_backup(disk_tensor) is None
        torch_memory_saver.resume("disk_weights")
        assert torch.equal(disk_tensor.cpu(), disk_expected)
    finally:
        if os.path.exists(artifact_path):
            os.remove(artifact_path)
