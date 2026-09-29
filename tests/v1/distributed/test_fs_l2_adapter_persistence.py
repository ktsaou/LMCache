# SPDX-License-Identifier: Apache-2.0
"""Persistence tests for the file-system L2 adapter."""

# Standard
from pathlib import Path
from typing import cast
from unittest.mock import patch
import os
import time

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.internal_api import L2StoreResult
from lmcache.v1.distributed.l2_adapters.fs_l2_adapter import (
    FSL2Adapter,
    FSL2AdapterConfig,
    _bounded_relative_path_to_object_key,
    _object_key_to_relative_path,
)
from lmcache.v1.memory_management import MemoryObj


class _Buffer:
    """Minimal memory object used by FS adapter lifecycle tests."""

    def __init__(self, data: bytes) -> None:
        self._data = bytearray(data)

    @property
    def byte_array(self) -> memoryview:
        """Return the writable byte view consumed by the adapter."""
        return memoryview(self._data)


def _memory_obj(data: bytes) -> MemoryObj:
    return cast(MemoryObj, _Buffer(data))


def _long_key(*, model_suffix: str = "", salt_suffix: str = "") -> ObjectKey:
    return ObjectKey(
        chunk_hash=b"\xde\xad\xbe\xef" * 8,
        model_name="organization/" + "model-segment-" * 12 + model_suffix,
        kv_rank=42,
        object_group_id=7,
        cache_salt="tenant-" + "s" * 120 + salt_suffix,
    )


def _wait_for_store(adapter: FSL2Adapter, key: ObjectKey, data: bytes) -> None:
    task_id = adapter.submit_store_task([key], [_memory_obj(data)])
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        result = adapter.pop_completed_store_tasks().get(task_id)
        if result is not None:
            assert result == L2StoreResult(True, len(data))
            return
        time.sleep(0.01)
    pytest.fail("store task did not complete within 5s")


def _wait_for_lookup(adapter: FSL2Adapter, key: ObjectKey) -> bool:
    task_id = adapter.submit_lookup_and_lock_task(
        [key], {0: MemoryLayoutDesc(shapes=[], dtypes=[])}
    )
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        result = adapter.query_lookup_and_lock_result(task_id)
        if result is not None:
            return result.test(0)
        time.sleep(0.01)
    raise AssertionError("lookup task did not complete within 5s")


def _wait_for_load(adapter: FSL2Adapter, key: ObjectKey, size: int) -> bytes:
    destination = _memory_obj(bytes(size))
    task_id = adapter.submit_load_task([key], [destination])
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        result = adapter.query_load_result(task_id)
        if result is not None:
            assert result.test(0)
            return bytes(destination.byte_array)
        time.sleep(0.01)
    raise AssertionError("load task did not complete within 5s")


def test_long_key_survives_restart(tmp_path: Path) -> None:
    """An oversized legacy name remains usable across adapter restarts."""
    key = _long_key(salt_suffix="a")
    legacy_filename = (
        "organization-SEP-"
        + "model-segment-" * 12
        + "@0x0000002a@7@"
        + "deadbeef" * 8
        + "@tenant-"
        + "s" * 120
        + "a.data"
    )
    name_max = os.pathconf(tmp_path, "PC_NAME_MAX")
    assert len(os.fsencode(legacy_filename)) > name_max
    payload = b"persistent payload"

    first = FSL2Adapter(FSL2AdapterConfig(base_path=str(tmp_path)))
    try:
        _wait_for_store(first, key, payload)
        assert _wait_for_lookup(first, key)
        assert _wait_for_load(first, key, len(payload)) == payload
    finally:
        first.close()

    second = FSL2Adapter(FSL2AdapterConfig(base_path=str(tmp_path)))
    try:
        assert _wait_for_lookup(second, key)
        assert _wait_for_load(second, key, len(payload)) == payload
        second.delete([key])
        assert not _wait_for_lookup(second, key)
    finally:
        second.close()


def test_short_legacy_filename_remains_readable(tmp_path: Path) -> None:
    """Representable keys retain and discover their exact legacy filename."""
    key = ObjectKey(
        chunk_hash=b"\xde\xad\xbe\xef",
        model_name="meta-llama/Llama-3",
        kv_rank=42,
        object_group_id=7,
        cache_salt="tenant-a",
    )
    legacy_filename = "meta-llama-SEP-Llama-3@0x0000002a@7@deadbeef@tenant-a.data"
    payload = b"legacy payload"
    (tmp_path / legacy_filename).write_bytes(payload)

    adapter = FSL2Adapter(FSL2AdapterConfig(base_path=str(tmp_path)))
    try:
        assert _wait_for_lookup(adapter, key)
        assert _wait_for_load(adapter, key, len(payload)) == payload
    finally:
        adapter.close()


def test_distinct_oversized_keys_use_distinct_stable_files(tmp_path: Path) -> None:
    """Bounded paths preserve key fields that differ near their ends."""
    keys = [
        _long_key(model_suffix="a", salt_suffix="a"),
        _long_key(model_suffix="b", salt_suffix="a"),
        _long_key(model_suffix="a", salt_suffix="b"),
    ]
    adapter = FSL2Adapter(FSL2AdapterConfig(base_path=str(tmp_path)))
    try:
        payloads = (b"base", b"model differs", b"salt differs")
        for key, payload in zip(keys, payloads, strict=True):
            _wait_for_store(adapter, key, payload)
        relative_paths = {
            path.relative_to(tmp_path) for path in tmp_path.rglob("*.data")
        }
        assert relative_paths == {_object_key_to_relative_path(key) for key in keys}
        name_max = os.pathconf(tmp_path, "PC_NAME_MAX")
        assert all(
            len(os.fsencode(component)) <= name_max
            for path in relative_paths
            for component in path.parts
        )
        assert {
            _bounded_relative_path_to_object_key(path) for path in relative_paths
        } == set(keys)
    finally:
        adapter.close()

    restarted = FSL2Adapter(FSL2AdapterConfig(base_path=str(tmp_path)))
    try:
        assert [_wait_for_lookup(restarted, key) for key in keys] == [True] * 3
        assert {
            path.relative_to(tmp_path) for path in tmp_path.rglob("*.data")
        } == relative_paths
    finally:
        restarted.close()


def test_oversized_chunk_hash_uses_bounded_components(tmp_path: Path) -> None:
    """A valid long chunk hash never creates an oversized path component."""
    key = ObjectKey(
        chunk_hash=bytes(range(128)),
        model_name="org/model",
        kv_rank=42,
        object_group_id=7,
        cache_salt="tenant-a",
    )
    relative_path = _object_key_to_relative_path(key)
    name_max = os.pathconf(tmp_path, "PC_NAME_MAX")
    assert all(
        len(os.fsencode(component)) <= name_max for component in relative_path.parts
    )
    assert _bounded_relative_path_to_object_key(relative_path) == key

    payload = b"long-hash payload"
    adapter = FSL2Adapter(FSL2AdapterConfig(base_path=str(tmp_path)))
    try:
        _wait_for_store(adapter, key, payload)
        assert _wait_for_load(adapter, key, len(payload)) == payload
    finally:
        adapter.close()


def test_oversized_numeric_fields_use_bounded_components(tmp_path: Path) -> None:
    """Object rank and group integers cannot exceed the component limit."""
    key = ObjectKey(
        chunk_hash=b"hash",
        model_name="org/model",
        kv_rank=1 << 4096,
        object_group_id=1 << 4096,
        cache_salt="tenant-a",
    )
    relative_path = _object_key_to_relative_path(key)
    name_max = os.pathconf(tmp_path, "PC_NAME_MAX")
    assert all(
        len(os.fsencode(component)) <= name_max for component in relative_path.parts
    )
    assert _bounded_relative_path_to_object_key(relative_path) == key


def test_empty_chunk_hash_round_trips_through_split_layout(tmp_path: Path) -> None:
    """A split bounded path preserves an empty chunk hash across restart."""
    key = ObjectKey(
        chunk_hash=b"",
        model_name="org/model",
        kv_rank=1 << 4096,
        object_group_id=7,
        cache_salt="tenant-a",
    )
    relative_path = _object_key_to_relative_path(key)

    assert "h0" in relative_path.parts
    assert _bounded_relative_path_to_object_key(relative_path) == key

    payload = b"empty-hash payload"
    adapter = FSL2Adapter(FSL2AdapterConfig(base_path=str(tmp_path)))
    try:
        _wait_for_store(adapter, key, payload)
        assert _wait_for_load(adapter, key, len(payload)) == payload
    finally:
        adapter.close()


def test_store_rejects_object_path_at_path_max(tmp_path: Path) -> None:
    """An unrepresentable complete path fails before creating directories."""
    path_max = os.pathconf(tmp_path, "PC_PATH_MAX")
    if path_max < 0:
        pytest.skip("filesystem reports no fixed PC_PATH_MAX")
    key = ObjectKey(
        chunk_hash=b"hash",
        model_name="m" * (path_max // 2 + 128),
        kv_rank=0,
    )
    full_path = tmp_path / _object_key_to_relative_path(key)
    assert len(os.fsencode(full_path)) >= path_max

    adapter = FSL2Adapter(FSL2AdapterConfig(base_path=str(tmp_path)))
    try:
        task_id = adapter.submit_store_task([key], [_memory_obj(b"payload")])
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            result = adapter.pop_completed_store_tasks().get(task_id)
            if result is not None:
                assert result == L2StoreResult(False, 0)
                break
            time.sleep(0.01)
        else:
            pytest.fail("rejected store did not complete within 5s")
        assert list(tmp_path.rglob("*.data")) == []
        assert not (tmp_path / ".lmcache-objects-v1").exists()
    finally:
        adapter.close()


def test_python_adapter_reads_and_deletes_oversized_legacy_file(
    tmp_path: Path,
) -> None:
    """Canonical bounded paths retain the representable flat-file fallback."""
    key = ObjectKey(
        chunk_hash=b"legacy-hash",
        model_name="org/model-with-a-long-name",
        kv_rank=42,
        object_group_id=7,
        cache_salt="tenant-a",
    )
    legacy_path = tmp_path / (
        "org-SEP-model-with-a-long-name@0x0000002a@7@"
        "6c65676163792d68617368@tenant-a.data"
    )
    payload = b"legacy payload"
    legacy_path.write_bytes(payload)

    module = "lmcache.v1.distributed.l2_adapters.fs_l2_adapter"
    with patch(f"{module}._LEGACY_FILENAME_MAX_BYTES", 32):
        assert _object_key_to_relative_path(key).parent != Path(".")
        adapter = FSL2Adapter(FSL2AdapterConfig(base_path=str(tmp_path)))
        try:
            assert _wait_for_lookup(adapter, key)
            assert _wait_for_load(adapter, key, len(payload)) == payload

            task_id = adapter.submit_store_task(
                [key], [_memory_obj(b"replacement must not be stored")]
            )
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                result = adapter.pop_completed_store_tasks().get(task_id)
                if result is not None:
                    assert result == L2StoreResult(True, 0)
                    break
                time.sleep(0.01)
            else:
                pytest.fail("duplicate store did not complete within 5s")

            assert legacy_path.read_bytes() == payload
            adapter.delete([key])
            deadline = time.monotonic() + 5.0
            while legacy_path.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert not legacy_path.exists()
        finally:
            adapter.close()


def test_path_does_not_depend_on_local_name_max(tmp_path: Path) -> None:
    """The same oversized key maps to one path on all supported hosts."""
    key = _long_key(salt_suffix="a")
    relative_paths = []

    real_pathconf = os.pathconf
    for dirname, name_max in (("standard", 255), ("large", 512)):
        base_path = tmp_path / dirname
        base_path.mkdir()
        with patch(
            "os.pathconf",
            side_effect=lambda path, name, name_max=name_max: (
                name_max if name == "PC_NAME_MAX" else real_pathconf(path, name)
            ),
        ):
            adapter = FSL2Adapter(FSL2AdapterConfig(base_path=str(base_path)))
        try:
            _wait_for_store(adapter, key, b"payload")
            relative_paths.append(
                next(base_path.rglob("*.data")).relative_to(base_path)
            )
        finally:
            adapter.close()

    assert relative_paths[0] == relative_paths[1]
    assert _bounded_relative_path_to_object_key(relative_paths[0]) == key


def test_oversized_surrogateescaped_key_can_be_stored(tmp_path: Path) -> None:
    """Bounded addressing accepts model strings restored from FS bytes."""
    escaped_byte = b"\x80".decode(errors="surrogateescape")
    key = _long_key(model_suffix=escaped_byte, salt_suffix="a")
    adapter = FSL2Adapter(FSL2AdapterConfig(base_path=str(tmp_path)))
    try:
        _wait_for_store(adapter, key, b"payload")
        assert _wait_for_lookup(adapter, key)
    finally:
        adapter.close()


def test_relative_tmp_dir_stores_oversized_key(tmp_path: Path) -> None:
    """Oversized keys also work when temporary files use a subdirectory."""
    key = _long_key(salt_suffix="a")
    adapter = FSL2Adapter(
        FSL2AdapterConfig(base_path=str(tmp_path), relative_tmp_dir="tmp")
    )
    try:
        _wait_for_store(adapter, key, b"payload")
        assert _wait_for_lookup(adapter, key)
        assert list((tmp_path / "tmp").iterdir()) == []
        data_files = list(tmp_path.rglob("*.data"))
        assert len(data_files) == 1
        assert all(
            len(os.fsencode(component)) <= os.pathconf(tmp_path, "PC_NAME_MAX")
            for component in data_files[0].relative_to(tmp_path).parts
        )
    finally:
        adapter.close()


def test_rejects_filesystem_below_protocol_limit(tmp_path: Path) -> None:
    """Initialization rejects limits below the filename protocol limit."""
    with (
        patch("os.pathconf", return_value=254),
        pytest.raises(ValueError, match=r"PC_NAME_MAX >= 255, got 254"),
    ):
        FSL2Adapter(FSL2AdapterConfig(base_path=str(tmp_path)))


def test_absent_keys_sees_objects_other_adapters_stored(tmp_path: Path) -> None:
    """Absence is checked on disk: an object another adapter instance stored
    in the shared directory is present, a deleted one is absent."""
    stored, missing = _long_key(salt_suffix="a"), _long_key(salt_suffix="b")
    reader = FSL2Adapter(FSL2AdapterConfig(base_path=str(tmp_path)))
    writer = FSL2Adapter(FSL2AdapterConfig(base_path=str(tmp_path)))
    try:
        assert reader.absent_keys([stored, missing]) == [stored, missing]
        _wait_for_store(writer, stored, b"payload")
        assert reader.absent_keys([stored, missing]) == [missing]
        writer.delete([stored])
        assert reader.absent_keys([stored, missing]) == [stored, missing]
    finally:
        writer.close()
        reader.close()
