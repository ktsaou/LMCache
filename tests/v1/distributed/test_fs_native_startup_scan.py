# SPDX-License-Identifier: Apache-2.0
"""Restart-inventory tests for the native filesystem L2 adapter."""

# Standard
from types import SimpleNamespace
import os

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.l2_adapters.fs_l2_adapter import (
    _object_key_to_filename,
    _object_key_to_relative_path,
)
from lmcache.v1.distributed.l2_adapters.fs_native_l2_adapter import (
    _absent_native_fs_keys,
    _scan_existing_key_sizes,
)


def _key(chunk_id: int, *, salt: str = "") -> ObjectKey:
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name="org/model",
        kv_rank=0x01020304,
        object_group_id=3,
        cache_salt=salt,
    )


def test_scan_returns_positive_objects_oldest_to_newest(tmp_path) -> None:
    oldest = _key(1)
    newest = _key(2, salt="tenant-a")
    oldest_path = tmp_path / _object_key_to_filename(oldest)
    newest_path = tmp_path / _object_key_to_filename(newest)
    oldest_path.write_bytes(b"old")
    newest_path.write_bytes(b"newest")
    os.utime(oldest_path, ns=(1_000, 1_000))
    os.utime(newest_path, ns=(2_000, 2_000))

    (tmp_path / "foreign.data").write_bytes(b"ignore")
    (tmp_path / "partial.tmp").write_bytes(b"ignore")
    (tmp_path / _object_key_to_filename(_key(3))).touch()
    (tmp_path / "subdir").mkdir()
    try:
        (tmp_path / "link.data").symlink_to(oldest_path)
    except OSError:
        pass

    inventory = _scan_existing_key_sizes(str(tmp_path))

    assert list(inventory.items()) == [(oldest, 3), (newest, 6)]


def test_scan_tiebreaks_equal_mtime_by_filename(tmp_path) -> None:
    keys = [_key(20), _key(10)]
    paths = [tmp_path / _object_key_to_filename(key) for key in keys]
    for path in paths:
        path.write_bytes(b"x")
        os.utime(path, ns=(1_000, 1_000))

    inventory = _scan_existing_key_sizes(str(tmp_path))

    expected = sorted(keys, key=_object_key_to_filename)
    assert list(inventory) == expected


def test_scan_inventories_bounded_long_key(tmp_path) -> None:
    """Restart accounting decodes a long model identity and 128-byte salt."""
    key = ObjectKey(
        chunk_hash=bytes.fromhex("70f8501b00e17eb724cd5eb68e21c012" * 2),
        model_name=(
            "/model/snapshots/378ca54585c46542bad1f3cb3ed0d73ae51cdb62"
            "##lmcache-dcp-layout-v1-d4-interleave4"
        ),
        kv_rank=0x04010401,
        object_group_id=7,
        cache_salt="tenant-" + "s" * 121,
    )
    path = tmp_path / _object_key_to_relative_path(key)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"bounded")

    assert _scan_existing_key_sizes(str(tmp_path)) == {key: 7}


def test_scan_inventories_split_path_with_empty_hash(tmp_path) -> None:
    """Restart inventory decodes the explicit zero-component hash field."""
    key = ObjectKey(
        chunk_hash=b"",
        model_name="org/model",
        kv_rank=1 << 4096,
        object_group_id=7,
        cache_salt="tenant-a",
    )
    path = tmp_path / _object_key_to_relative_path(key)
    assert "h0" in path.parts
    path.parent.mkdir(parents=True)
    path.write_bytes(b"bounded")

    assert _scan_existing_key_sizes(str(tmp_path)) == {key: 7}


def test_scan_missing_directory_is_empty(tmp_path) -> None:
    assert _scan_existing_key_sizes(str(tmp_path / "not-created")) == {}


def test_scan_fails_closed_on_directory_error(monkeypatch) -> None:
    def fail_scandir(_path):
        raise PermissionError("denied")

    monkeypatch.setattr(os, "scandir", fail_scandir)
    with pytest.raises(RuntimeError, match="Failed to scan native FS cache"):
        _scan_existing_key_sizes("/cache")


class _FakeScandir:
    def __init__(self, entries) -> None:
        self._entries = entries

    def __enter__(self):
        return self

    def __iter__(self):
        return iter(self._entries)

    def __exit__(self, *_args) -> None:
        return None


def test_scan_fails_closed_on_stat_error(monkeypatch) -> None:
    key = _key(1)
    entry = SimpleNamespace(
        name=_object_key_to_filename(key),
        path="/cache/object.data",
        is_file=lambda **_kwargs: True,
        stat=lambda **_kwargs: (_ for _ in ()).throw(PermissionError("denied")),
    )
    monkeypatch.setattr(os, "scandir", lambda _path: _FakeScandir([entry]))

    with pytest.raises(RuntimeError, match="Failed to stat native FS cache"):
        _scan_existing_key_sizes("/cache")


def test_scan_skips_entry_removed_before_stat(monkeypatch) -> None:
    key = _key(1)
    entry = SimpleNamespace(
        name=_object_key_to_filename(key),
        path="/cache/object.data",
        is_file=lambda **_kwargs: True,
        stat=lambda **_kwargs: (_ for _ in ()).throw(FileNotFoundError()),
    )
    monkeypatch.setattr(os, "scandir", lambda _path: _FakeScandir([entry]))

    assert _scan_existing_key_sizes("/cache") == {}


def test_scan_fails_closed_when_bounded_root_cannot_be_inspected(
    monkeypatch,
) -> None:
    """A bounded-tree stat error cannot silently reduce restart capacity."""
    monkeypatch.setattr(os, "scandir", lambda _path: _FakeScandir([]))
    real_stat = os.stat

    def fail_bounded_root(path, *, follow_symlinks=True):
        if str(path).endswith(".lmcache-objects-v1"):
            raise PermissionError("denied")
        return real_stat(path, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(os, "stat", fail_bounded_root)
    with pytest.raises(RuntimeError, match="bounded cache root"):
        _scan_existing_key_sizes("/cache")


def test_scan_rejects_non_directory_bounded_root(tmp_path) -> None:
    """A conflicting bounded-root file fails before native client startup."""
    (tmp_path / ".lmcache-objects-v1").write_bytes(b"conflict")
    with pytest.raises(RuntimeError, match="not a directory"):
        _scan_existing_key_sizes(str(tmp_path))


def test_scan_rejects_duplicate_decoded_keys(tmp_path) -> None:
    key = _key(10)
    canonical = _object_key_to_filename(key)
    # Uppercasing the whole filename would alter the model name. Only vary the
    # hex hash, which remains the same decoded ObjectKey.
    prefix, chunk_hash = canonical[:-5].rsplit("@", 1)
    alternate = f"{prefix}@{chunk_hash.upper()}.data"
    assert alternate != canonical
    (tmp_path / canonical).write_bytes(b"first")
    (tmp_path / alternate).write_bytes(b"second")

    with pytest.raises(RuntimeError, match="multiple filenames"):
        _scan_existing_key_sizes(str(tmp_path))


def test_absence_probe_sees_objects_any_process_published(tmp_path) -> None:
    """An object file under its canonical or legacy name counts as present,
    whoever wrote it; only keys with neither file are confirmed absent."""
    canonical, legacy, missing = _key(1), _key(2), _key(3)
    oversized = ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(4),
        model_name="m" * 240,
        kv_rank=0,
        object_group_id=1,
    )
    for key in (canonical, oversized):
        path = tmp_path / _object_key_to_relative_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"object")
    (tmp_path / _object_key_to_filename(legacy)).write_bytes(b"object")
    keys = [canonical, legacy, missing, oversized]
    assert _absent_native_fs_keys(str(tmp_path), keys) == [missing]
