import os
from unittest.mock import patch

import pytest

from torch_memory_saver.entrypoint import (
    _get_env_disk_backup_loc_for_tag,
    _normalize_disk_backup_options,
)


def test_explicit_disk_backup_loc_wins_over_tag_env():
    with patch.dict(
        os.environ,
        {"TMS_TAG_DISK_BACKUP_LOC_WEIGHTS": "/env/weights.bin"},
        clear=False,
    ):
        result = _normalize_disk_backup_options(
            tag="weights",
            enable_cpu_backup=False,
            disk_backup_loc="/explicit/weights.bin",
        )
    assert result == "/explicit/weights.bin"


def test_tag_env_applies_when_explicit_arg_omitted():
    with patch.dict(
        os.environ,
        {"TMS_TAG_DISK_BACKUP_LOC_WEIGHTS": "/env/weights.bin"},
        clear=False,
    ):
        result = _normalize_disk_backup_options(
            tag="weights",
            enable_cpu_backup=False,
            disk_backup_loc=None,
        )
    assert result == "/env/weights.bin"


def test_enable_cpu_backup_overrides_tag_env_default():
    with patch.dict(
        os.environ,
        {"TMS_TAG_DISK_BACKUP_LOC_WEIGHTS": "/env/weights.bin"},
        clear=False,
    ):
        result = _normalize_disk_backup_options(
            tag="weights",
            enable_cpu_backup=True,
            disk_backup_loc=None,
        )
    assert result == ""


def test_explicit_disk_backup_loc_and_cpu_backup_still_conflict():
    with pytest.raises(ValueError, match="mutually exclusive"):
        _normalize_disk_backup_options(
            tag="weights",
            enable_cpu_backup=True,
            disk_backup_loc="/explicit/weights.bin",
        )


def test_tag_env_normalization_matches_sglang_tags():
    with patch.dict(
        os.environ,
        {"TMS_TAG_DISK_BACKUP_LOC_KV_CACHE": "/env/kv_cache.bin"},
        clear=False,
    ):
        result = _get_env_disk_backup_loc_for_tag("kv_cache")
    assert result == "/env/kv_cache.bin"
