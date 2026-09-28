# SPDX-License-Identifier: Apache-2.0
"""Counterexamples from the independent cache-integrity review."""

# Standard
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch
from unittest.mock import Mock
import asyncio
import hashlib
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import (
    EncodedObjectKey,
    MemoryLayoutDesc,
    ObjectKey,
    Tier,
)
from lmcache.v1.distributed.admission import AdmissionFailure
from lmcache.v1.multiprocess.cache_control.object_service import ObjectService
from lmcache.integration.vllm.checkpoint_scheduler import (
    CheckpointSchedulerBridge,
    _Lookup,
    _PendingTask,
)
from lmcache.v1.multiprocess.checkpoint_identity import CheckpointTokenRoots
from lmcache.v1.distributed.l2_adapters.fs_l2_adapter import (
    FSL2AdapterConfig,
    _object_key_to_relative_path,
)
from lmcache.v1.distributed.l2_adapters.native_connector_l2_adapter import (
    NativeConnectorL2Adapter,
)
from tests.v1.distributed.test_native_connector_l2_adapter import (
    MockNativeConnector,
    create_object_key,
)
from tests.v1.multiprocess.test_checkpoint_storage import (
    CheckpointSlots,
    drain_l2_stores,
    entry_keys,
    large_manifest,
    make_content_manifest,
    make_manifest,
    open_store,
    poll,
    publish_all,
    publish_before_restart,
    restores,
    wait_for,
)


@pytest.mark.parametrize("native", [False, True])
def test_capacity_return_after_failed_load_preserves_checkpoint(
    tmp_path: Path, native: bool
) -> None:
    entry = large_manifest(550, page_bytes=128 * 1024)
    publish_before_restart(tmp_path, native, [entry])
    with open_store(tmp_path, native, admission_timeout_seconds=0.15) as (
        service,
        index,
        storage,
        mapping,
    ):
        filler = ObjectKey(hashlib.sha256(b"active-writer").digest(), "filler", 0, 0)
        _, total = storage.get_l1_usage()
        lookup = storage.submit_prefetch_task
        query = storage.query_prefetch_status_detailed
        attempts = 0
        released = set()

        def reserve_then_lookup(*args: Any, **kwargs: Any) -> Any:
            nonlocal attempts
            attempts += 1
            result = storage.reserve_write_detailed(
                [filler],
                MemoryLayoutDesc([torch.Size([total])], [torch.uint8]),
                "new",
                internal=True,
            )
            assert result[filler][1] is not None
            return lookup(*args, **kwargs)

        def release_after_lookup(handle: Any) -> Any:
            result = query(handle)
            if result is not None and attempts not in released:
                assert result.reservation_failed
                storage.abort_write([filler])
                released.add(attempts)
            return result

        with (
            patch.object(
                storage, "submit_prefetch_task", side_effect=reserve_then_lookup
            ),
            patch.object(
                storage,
                "query_prefetch_status_detailed",
                side_effect=release_after_lookup,
            ),
        ):
            assert poll(service, service.begin_retrieve(entry, 0)) is False
        assert attempts >= 2
        assert index.get(entry.generation) == entry
        assert restores(service, mapping, entry)


@pytest.mark.parametrize("native", [False, True])
def test_public_l2_delete_retires_last_copy(tmp_path: Path, native: bool) -> None:
    with open_store(tmp_path, native) as (service, index, storage, mapping):
        entry = replace(make_manifest(), world_size=1)
        publish_all(service, index, mapping, entry)
        drain_l2_stores(storage)
        storage.clear()
        keys = entry_keys(entry)
        encoded = [
            EncodedObjectKey(
                k.chunk_hash.hex(),
                k.model_name,
                k.kv_rank,
                k.object_group_id,
                k.cache_salt,
            )
            for k in keys
        ]
        result = asyncio.run(
            ObjectService(SimpleNamespace(storage_manager=storage)).delete_objects(
                Tier.L2, None, encoded
            )
        )
        assert result == {"deleted": len(keys), "skipped": 0, "ok": True}
        assert index.find((entry.prefix,)) is None


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("policy", ["default", "retain"])
def test_temporary_restore_is_not_a_retained_replica(
    tmp_path: Path, native: bool, policy: str
) -> None:
    with open_store(
        tmp_path,
        native,
        store_policy="checkpoint_on_evict",
        prefetch_policy=policy,
        shutdown_flush_seconds=0,
    ) as (service, index, storage, mapping):
        entry = make_content_manifest(
            "temporary-source",
            prefix_token=1,
            attention_keys=tuple(
                hashlib.sha256(bytes([i])).hexdigest() for i in range(3)
            ),
        )
        keys = entry_keys(entry)
        publish_all(service, index, mapping, entry)
        retention = storage.checkpoint_retention
        retention.request_persist(keys)
        wait_for(lambda: all(retention.is_l2_resident(k) for k in keys), "persist")
        drain_l2_stores(storage)
        storage.clear()
        lease = poll(service, service.begin_retrieve(entry, 0))
        assert isinstance(lease, CheckpointSlots)
        descriptor, adapter = storage.l2_adapters()[0]
        retention.evict(keys, descriptor.index, adapter.delete)
        assert not any(retention.is_l2_resident(k) for k in keys)
        if policy == "default":
            assert index.find((entry.prefix,)) is None
            # A new publication may not inherit pages freed with this reader.
            replacement = replace(entry, generation=entry.generation + "-new")
            assert index.begin(replacement)
            assert service.prepare_store(replacement, 0) is AdmissionFailure.BUSY
            index.abort(replacement.generation)
        else:
            assert index.find((entry.prefix,)) == entry
        service.finish_retrieve(lease.lease_id)
        assert index.find((entry.prefix,)) is None or len(
            storage.get_readable_keys(keys)
        ) == len(keys)


@pytest.mark.parametrize("via_api", [False, True])
def test_store_delete_notifications_follow_physical_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, via_api: bool
) -> None:
    with open_store(
        tmp_path, store_policy="checkpoint_on_evict", shutdown_flush_seconds=0
    ) as (service, index, storage, mapping):
        entry = replace(make_manifest(), world_size=1)
        publish_all(service, index, mapping, entry)
        descriptor, adapter = storage.l2_adapters()[0]
        keys = entry_keys(entry)
        first, second = keys[:2]
        blocked, resume = threading.Event(), threading.Event()
        original = adapter._existing_key_path

        async def barrier(key: ObjectKey) -> Any:
            if key == second and not blocked.is_set():
                blocked.set()
                await asyncio.to_thread(resume.wait, 10)
            return await original(key)

        monkeypatch.setattr(adapter, "_existing_key_path", barrier)
        storage.checkpoint_retention.request_persist(keys)
        try:
            assert blocked.wait(5)
            if via_api:
                encoded = EncodedObjectKey(
                    first.chunk_hash.hex(),
                    first.model_name,
                    first.kv_rank,
                    first.object_group_id,
                    first.cache_salt,
                )
                result = asyncio.run(
                    ObjectService(
                        SimpleNamespace(storage_manager=storage)
                    ).delete_objects(Tier.L2, None, [encoded])
                )
                assert result["ok"]
            else:
                storage.checkpoint_retention.evict(
                    [first], descriptor.index, adapter.delete
                )
            assert first not in adapter.get_existing_key_sizes()
        finally:
            resume.set()
        drain_l2_stores(storage)
        assert storage.checkpoint_retention.is_l2_resident(first) == (
            first in adapter.get_existing_key_sizes()
        )
        storage.clear()
        assert index.find((entry.prefix,)) is None or restores(service, mapping, entry)


@pytest.mark.parametrize("native", [False, True])
def test_aligned_capacity_shortage_preserves_disk_checkpoint(
    tmp_path: Path, native: bool
) -> None:
    entry = replace(make_manifest(), world_size=1)
    publish_before_restart(tmp_path, native, [entry])
    with open_store(tmp_path, native, admission_timeout_seconds=0.15) as (
        service,
        index,
        storage,
        mapping,
    ):
        filler = ObjectKey(hashlib.sha256(b"alignment-filler").digest(), "filler", 0, 0)
        _, total = storage.get_l1_usage()
        reserved = storage.reserve_write_detailed(
            [filler],
            MemoryLayoutDesc([torch.Size([total - 4096])], [torch.uint8]),
            "new",
            internal=True,
        )
        assert reserved[filler][1] is not None
        try:
            assert poll(service, service.begin_retrieve(entry, 0)) is False
            assert index.get(entry.generation) == entry
        finally:
            storage.abort_write([filler])
        assert restores(service, mapping, entry)


def test_expired_prefetch_ownership_does_not_invalidate_disk_checkpoint(
    tmp_path: Path,
) -> None:
    entry = replace(make_manifest(), world_size=1)
    publish_before_restart(tmp_path, False, [entry])
    with open_store(tmp_path) as (service, index, storage, mapping):
        pin = storage.pin_readable_keys
        first = True

        def expired_once(keys: list[ObjectKey], **kwargs: Any) -> list[ObjectKey]:
            nonlocal first
            if first:
                first = False
                return []
            return pin(keys, **kwargs)

        with patch.object(storage, "pin_readable_keys", side_effect=expired_once):
            assert restores(service, mapping, entry)
        assert index.get(entry.generation) == entry


class _DelayedDelete(MockNativeConnector):
    """Model native close draining a syscall and discarding its completion."""

    def __init__(self) -> None:
        super().__init__()
        self.submitted = threading.Event()
        self.draining = threading.Event()
        self.release = threading.Event()

    def submit_batch_delete(self, keys: list[str]) -> int:
        self.submitted.set()
        return 1

    def close(self) -> None:
        self.draining.set()
        assert self.release.wait(10)
        super().close()


@pytest.mark.parametrize("remove_adapter", [False, True])
def test_native_close_settles_delete_only_after_backend_drain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, remove_adapter: bool
) -> None:
    client = _DelayedDelete()
    key = create_object_key(987)
    adapter = NativeConnectorL2Adapter(client, initial_key_sizes={key: 4096})
    monkeypatch.setattr(
        "lmcache.v1.distributed.storage_manager.create_l2_adapter", lambda *_: adapter
    )
    errors: list[Exception] = []
    deleted = threading.Event()
    closed = threading.Event()

    with open_store(tmp_path, shutdown_flush_seconds=0) as (_, _, storage, _):
        tier = storage.l2_adapters()[0][0].index

        def delete() -> None:
            try:
                storage.checkpoint_retention.evict([key], tier, adapter.delete)
            except Exception as exc:
                errors.append(exc)
            finally:
                deleted.set()

        def close() -> None:
            if remove_adapter:
                storage.delete_l2_adapter(tier)
            else:
                adapter.close()
            closed.set()

        deletion = threading.Thread(target=delete)
        closer = threading.Thread(target=close)
        deletion.start()
        try:
            assert client.submitted.wait(5)
            closer.start()
            assert client.draining.wait(5)
            assert not deleted.wait(0.1)
            assert not closed.is_set()
            with pytest.raises(RuntimeError, match="closing"):
                adapter.delete([key])
        finally:
            client.release.set()
            deletion.join(5)
            if closer.ident is not None:
                closer.join(5)
        assert deleted.is_set() and closed.is_set()
        assert len(errors) == 1
        assert "outcome unavailable" in str(errors[0])
        adapter.close()


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("via_api", [False, True])
def test_duplicate_delete_releases_every_fence(
    tmp_path: Path, native: bool, via_api: bool
) -> None:
    with open_store(tmp_path, native, shutdown_flush_seconds=0) as (
        service,
        index,
        storage,
        mapping,
    ):
        entry = make_content_manifest(
            "duplicate-source",
            prefix_token=1,
            attention_keys=tuple(
                hashlib.sha256(bytes([i])).hexdigest() for i in range(3)
            ),
        )
        publish_all(service, index, mapping, entry)
        drain_l2_stores(storage)
        keys = entry_keys(entry)
        selected = [keys[0], keys[0], keys[1]]
        if via_api:
            encoded = [
                EncodedObjectKey(
                    k.chunk_hash.hex(),
                    k.model_name,
                    k.kv_rank,
                    k.object_group_id,
                    k.cache_salt,
                )
                for k in selected
            ]
            result = asyncio.run(
                ObjectService(SimpleNamespace(storage_manager=storage)).delete_objects(
                    Tier.L2, None, encoded
                )
            )
            assert result == {"deleted": 2, "skipped": 0, "ok": True}
        else:
            descriptor, adapter = storage.l2_adapters()[0]
            storage.checkpoint_retention.evict(
                selected, descriptor.index, adapter.delete
            )
        new = replace(entry, generation="duplicate-next")
        assert index.begin(new)
        lease = service.prepare_store(new, 0)
        assert isinstance(lease, CheckpointSlots)
        service.finish_store(lease.lease_id, False)


def _retry_roots(
    roots: CheckpointTokenRoots, manifest: Any, *, copy_succeeded: bool = False
) -> tuple[Any, ...]:
    """Exercise the actual all-rank failure transition with an in-memory RPC sink."""
    client, manager = Mock(), Mock()
    manager.acknowledge_external_boundary_checkpoint.return_value = False
    bridge = CheckpointSchedulerBridge(manager, client, {}, 1)
    # Seed completed transfer state without constructing a GPU allocator.
    bridge._lookups["consumer"] = _Lookup(roots, Mock())
    bridge._tasks["failed"] = _PendingTask(
        SimpleNamespace(direction="RETRIEVE", manifest=manifest),
        SimpleNamespace(checkpoint_id=1, num_tokens=manifest.prefix.num_tokens),
        "consumer",
    )
    bridge.complete({"failed": {0: copy_succeeded}})
    assert bridge._lookups["consumer"].roots is roots
    return client.submit_request.call_args.args[1][0]


@pytest.mark.parametrize("length", [1, 4095, 4096, 4097, 8193])
def test_failed_restore_bounds_query_without_changing_identity(length: int) -> None:
    roots = CheckpointTokenRoots.build("retry", list(range(length + 3)))
    manifest = replace(make_manifest(), prefix=roots.prefix(length))
    query = _retry_roots(roots, manifest)
    expected = CheckpointTokenRoots.build("retry", list(range(length - 1))).roots
    assert query == expected


def test_publication_race_retries_same_checkpoint_without_longer_candidates() -> None:
    roots = CheckpointTokenRoots.build("retry", list(range(12)))
    manifest = replace(make_manifest(), prefix=roots.prefix(11))
    expected = CheckpointTokenRoots.build("retry", list(range(11))).roots
    assert _retry_roots(roots, manifest, copy_succeeded=True) == expected


@pytest.mark.parametrize("native", [False, True])
def test_failed_restore_uses_intact_shorter_without_forgetting_longest(
    tmp_path: Path, native: bool
) -> None:
    roots = CheckpointTokenRoots.build("retry", list(range(12)))
    longer = replace(make_manifest(), world_size=1, prefix=roots.prefix(11))
    shorter = replace(make_manifest(), world_size=1, prefix=roots.prefix(8))
    with open_store(tmp_path, native, admission_timeout_seconds=0.1) as (
        service,
        index,
        storage,
        mapping,
    ):
        publish_all(service, index, mapping, shorter)
        publish_all(service, index, mapping, longer)
        drain_l2_stores(storage)
        storage.clear()
        # Corrupt only this test's payload after the startup inventory was read.
        path = (
            tmp_path / "payloads" / _object_key_to_relative_path(entry_keys(longer)[-1])
        )
        path.write_bytes(b"")
        assert index.find(roots.roots) == longer
        assert not restores(service, mapping, longer)
        assert index.get(longer.generation) == longer
        candidate = index.find(_retry_roots(roots, longer))
        assert candidate == shorter
        assert restores(service, mapping, candidate)


@pytest.mark.parametrize("native", [False, True])
def test_recovery_retires_known_truncated_payload(tmp_path: Path, native: bool) -> None:
    entry = replace(make_manifest(), world_size=1)
    publish_before_restart(tmp_path, native, [entry])
    key = entry_keys(entry)[-1]
    path = tmp_path / "payloads" / _object_key_to_relative_path(key)
    path.write_bytes(b"x")
    with open_store(tmp_path, native) as (_, index, _, _):
        assert index.get(entry.generation) is None


@pytest.mark.parametrize("another_complete_copy", [False, True])
def test_malformed_replica_cannot_certify_last_copy(
    tmp_path: Path, another_complete_copy: bool
) -> None:
    with open_store(
        tmp_path, store_policy="checkpoint_on_evict", shutdown_flush_seconds=0
    ) as (service, index, storage, mapping):
        entry = replace(make_manifest(), world_size=1)
        publish_all(service, index, mapping, entry)
        key = entry_keys(entry)[-1]
        retention = storage.checkpoint_retention
        retention.record_l2_present(100, [key], [1])
        if another_complete_copy:
            retention.record_l2_present(101, [key], [96])
        assert storage.delete_l1_keys([key]) == (1, 0)
        assert (index.get(entry.generation) is not None) == another_complete_copy


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("first_copy", ["known_short", "unknown_size"])
def test_prefetch_selects_usable_replica(
    tmp_path: Path, native: bool, first_copy: str
) -> None:
    entry = replace(make_manifest(), world_size=1)
    with open_store(tmp_path, native, admission_timeout_seconds=0.1) as (
        service,
        index,
        storage,
        mapping,
    ):
        storage.add_l2_adapter(FSL2AdapterConfig(str(tmp_path / "backup")))
        publish_all(service, index, mapping, entry)
        drain_l2_stores(storage)
        key = entry_keys(entry)[-1]
        relative = _object_key_to_relative_path(key)
        assert (tmp_path / "backup" / relative).stat().st_size == 96
        if first_copy == "known_short":
            (tmp_path / "payloads" / relative).write_bytes(b"x")
            storage.checkpoint_retention.record_l2_present(0, [key], [1])
        else:
            storage.checkpoint_retention.record_l2_absent(0, [key])
        storage.clear()
        assert index.get(entry.generation) == entry
        assert restores(service, mapping, entry)
