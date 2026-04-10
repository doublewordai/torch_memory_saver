import os
import tempfile
import uuid

import torch

from torch_memory_saver import torch_memory_saver


def run(hook_mode: str):
    assert hook_mode == "torch"
    torch_memory_saver.hook_mode = hook_mode

    disk_backup_loc = os.path.join(tempfile.gettempdir(), f"tms_artifact_{uuid.uuid4().hex}.bin")
    try:
        with torch_memory_saver.region(tag="disk_weights", disk_backup_loc=disk_backup_loc):
            disk_tensor = torch.arange(4_000_000, dtype=torch.float32, device="cuda")
        disk_expected = disk_tensor.cpu()

        torch_memory_saver.pause("disk_weights")
        assert os.path.exists(disk_backup_loc)
        torch_memory_saver.resume("disk_weights")
        assert torch.equal(disk_tensor.cpu(), disk_expected)
        assert torch_memory_saver.get_cpu_backup(disk_tensor) is None
    finally:
        if os.path.exists(disk_backup_loc):
            os.remove(disk_backup_loc)
