# SPDX-License-Identifier: Apache-2.0
"""Loss under storage pressure must preserve coherent checkpoint lookup."""

# Standard
from dataclasses import replace
import hashlib
import time

# Third Party
import pytest

# First Party
from tests.v1.multiprocess.test_checkpoint_storage import (
    CheckpointSlots,
    drain_l2_stores,
    entry_keys,
    large_manifest,
    make_content_manifest,
    make_manifest,
    open_store,
    publish_all,
    restores,
    wait_for,
)
from lmcache.v1.distributed.l2_adapters.fs_l2_adapter import (
    FSL2Adapter,
    FSL2AdapterConfig,
)


def test_last_copy_deletion_retires_only_affected_branch() -> None:
    key = lambda value: hashlib.sha256(value.encode()).hexdigest()
    older = make_content_manifest(
        "older", prefix_token=3, attention_keys=(key("a"), key("b"), key("old"))
    )
    newer = make_content_manifest(
        "newer", prefix_token=4, attention_keys=(key("a"), key("b"), key("new"))
    )
    with open_store() as (service, index, storage, mapping):
        publish_all(service, index, mapping, older)
        publish_all(service, index, mapping, newer)
        unique = list(set(entry_keys(older)) - set(entry_keys(newer)))
        assert storage.delete_l1_keys(unique)[0] == len(unique)
        assert index.find((older.prefix,)) is None
        assert index.find((newer.prefix,)) == newer
        assert restores(service, mapping, newer)


def test_clear_retires_ram_only_manifests() -> None:
    with open_store() as (service, index, storage, mapping):
        entry = make_manifest()
        publish_all(service, index, mapping, entry)
        storage.clear()
        assert index.find((entry.prefix,)) is None


@pytest.mark.parametrize("native", [False, True])
def test_disk_failure_retires_cold_checkpoints_and_admits_new_work(
    tmp_path, monkeypatch, native
) -> None:
    with open_store(
        tmp_path,
        native,
        store_policy="checkpoint_on_evict",
        shutdown_flush_seconds=0,
        admission_timeout_seconds=2,
    ) as resources:
        service, index, storage, mapping = resources
        adapter = storage.l2_adapters()[0][1]

        def unavailable(*_args, **_kwargs):
            raise OSError("injected disk unavailable")

        monkeypatch.setattr(adapter, "submit_store_task", unavailable)
        entries = [large_manifest(500 + i) for i in range(7)]
        started = time.monotonic()
        for entry in entries:
            publish_all(service, index, mapping, entry)
        assert time.monotonic() - started < 15
        assert storage.get_admission_stats()["exhausted_timeouts"] == 0
        assert storage.checkpoint_retention.report_status()["retired_checkpoints"] > 0
        assert index.find((entries[-1].prefix,)) == entries[-1]
        assert restores(service, mapping, entries[-1])
        for entry in entries:
            if index.find((entry.prefix,)) is not None:
                assert set(storage.get_readable_keys(entry_keys(entry))) == set(
                    entry_keys(entry)
                )
        status = storage.report_status()
        # All work is bounded independently of how long failures persist.
        store_status = status["store_controller"]
        assert store_status["owned_source_bytes"] <= store_status["max_pending_bytes"]


@pytest.mark.parametrize("native", [False, True])
def test_transient_submission_failure_retries_and_restores(
    tmp_path, monkeypatch, native
) -> None:
    with open_store(
        tmp_path, native, store_policy="checkpoint_on_evict", shutdown_flush_seconds=0
    ) as (service, index, storage, mapping):
        entry = replace(make_manifest(), world_size=1)
        publish_all(service, index, mapping, entry)
        adapter = storage.l2_adapters()[0][1]
        submit = adapter.submit_store_task
        calls = []

        def fail_once(keys, objects):
            calls.append(list(keys))
            if len(calls) == 1:
                raise OSError("injected transient submission failure")
            return submit(keys, objects)

        monkeypatch.setattr(adapter, "submit_store_task", fail_once)
        storage.checkpoint_retention.request_persist(entry_keys(entry))
        wait_for(
            lambda: all(
                storage.checkpoint_retention.is_l2_resident(key)
                for key in entry_keys(entry)
            ),
            "retry did not persist checkpoint",
        )
        drain_l2_stores(storage)
        assert len(calls) > 1
        storage.clear()
        assert index.find((entry.prefix,)) == entry
        assert restores(service, mapping, entry)


def test_zero_shutdown_budget_removes_ram_only_directory_entries(tmp_path) -> None:
    entry = make_manifest()
    with open_store(
        tmp_path, True, store_policy="checkpoint_on_evict", shutdown_flush_seconds=0
    ) as (service, index, storage, mapping):
        publish_all(service, index, mapping, entry)
        assert index.find((entry.prefix,)) == entry
    with open_store(tmp_path, True) as (_, index, _, _):
        assert index.find((entry.prefix,)) is None


def test_duplicate_completed_rank_cannot_accumulate_pins() -> None:
    entry = make_manifest()
    with open_store() as (service, index, _, mapping):
        assert index.begin(entry)
        lease = service.prepare_store(entry, 0)
        assert isinstance(lease, CheckpointSlots)
        for group in lease.groups:
            for slot in group:
                assert slot is not None
                mapping[slot.offset : slot.offset + slot.length] = b"\0" * slot.length
        assert not service.finish_store(lease.lease_id, True)
        with pytest.raises(ValueError, match="already has a store lease"):
            service.prepare_store(entry, 0)
        index.abort(entry.generation)


def test_unknown_inventory_preserves_lookup_validation(tmp_path, monkeypatch) -> None:
    entry = replace(make_manifest(), world_size=1)
    with open_store(tmp_path) as (service, index, storage, mapping):
        publish_all(service, index, mapping, entry)
        drain_l2_stores(storage)
    monkeypatch.setattr(FSL2Adapter, "has_complete_inventory", lambda _self: False)
    monkeypatch.setattr(FSL2Adapter, "get_existing_key_sizes", lambda _self: {})
    with open_store(tmp_path) as (service, index, _, mapping):
        assert index.find((entry.prefix,)) == entry
        assert restores(service, mapping, entry)


def test_terminal_shutdown_flushes_complete_state_and_keeps_orphan_buffers(
    tmp_path,
) -> None:
    complete = replace(make_manifest(), world_size=1)
    orphan = replace(make_manifest(), world_size=1)
    with open_store(tmp_path, True, store_policy="checkpoint_on_evict") as resources:
        service, index, storage, mapping = resources
        publish_all(service, index, mapping, complete)
        assert index.begin(orphan)
        lease = service.prepare_store(orphan, 0)
        assert isinstance(lease, CheckpointSlots)
        service.prepare_terminal_shutdown()
        keys = entry_keys(orphan)
        assert storage.delete_l1_keys(keys) == (0, len(keys))
        assert service.prepare_store(orphan, 0) is None
        assert not service.finish_store(lease.lease_id, False)
    with open_store(tmp_path, True) as (service, index, _, mapping):
        assert index.find((complete.prefix,)) == complete
        assert restores(service, mapping, complete)


def test_persistence_retries_after_last_adapter_is_replaced(tmp_path) -> None:
    entry = replace(make_manifest(), world_size=1)
    with open_store(tmp_path, store_policy="checkpoint_on_evict") as resources:
        service, index, storage, mapping = resources
        publish_all(service, index, mapping, entry)
        storage.delete_l2_adapter(storage.l2_adapters()[0][0].index)
        retention = storage.checkpoint_retention
        assert retention.request_persist(entry_keys(entry)) > 0
        wait_for(
            lambda: retention.pending_count() == 0,
            "unroutable persistence stayed pending",
        )
        storage.add_l2_adapter(FSL2AdapterConfig(str(tmp_path / "replacement")))
        assert retention.request_persist(entry_keys(entry)) > 0
        wait_for(
            lambda: all(retention.is_l2_resident(key) for key in entry_keys(entry)),
            "replacement adapter never received retained pages",
        )


def test_terminal_shutdown_keeps_exposed_retrieve_ownership(tmp_path) -> None:
    from tests.v1.multiprocess.test_checkpoint_storage import poll

    entry = replace(make_manifest(), world_size=1)
    with open_store(tmp_path, True, store_policy="checkpoint_on_evict") as resources:
        service, index, storage, mapping = resources
        publish_all(service, index, mapping, entry)
        lease = service.begin_retrieve(entry, 0)
        assert lease is not None and isinstance(poll(service, lease), CheckpointSlots)
        service.prepare_terminal_shutdown()
        assert storage.delete_l1_keys(entry_keys(entry))[0] == 0
        service.finish_retrieve(lease)
    with open_store(tmp_path, True) as (service, index, _, mapping):
        assert index.find((entry.prefix,)) == entry
        assert restores(service, mapping, entry)
