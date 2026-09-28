# SPDX-License-Identifier: Apache-2.0
"""Stale metadata may expire; exposed copy buffers require drained completion."""

# Standard
from dataclasses import replace
import time

# First Party
from lmcache.v1.multiprocess.checkpoint_storage import (
    CheckpointPayloadStore,
    CheckpointSlots,
    checkpoint_object_keys,
)
from tests.v1.multiprocess.test_checkpoint_storage import (
    fill,
    make_manifest,
    open_store,
    poll,
)

ABANDONED_AFTER = 0.2


def test_abandoned_store_metadata_expires_but_copy_buffers_stay_owned() -> None:
    with open_store() as (_service, index, _storage, _mapping):
        service = CheckpointPayloadStore(
            _storage, index, abandoned_after_seconds=ABANDONED_AFTER
        )
        entry = replace(make_manifest(), world_size=1)
        assert index.begin(entry)
        lease = service.prepare_store(entry, 0)
        assert isinstance(lease, CheckpointSlots)
        # The worker dies here: no FINISH_STORE ever arrives.
        assert service.reclaim_abandoned() == 0  # too young to be abandoned
        time.sleep(ABANDONED_AFTER + 0.1)
        assert service.reclaim_abandoned() == 1
        status = service.report_status()
        assert status["store_leases"] == 1
        keys = [key for group in checkpoint_object_keys(entry, 0) for key in group]
        assert _storage.delete_l1_keys(keys)[0] == 0
        assert index.report_status()["pending_generations"] == 0
        assert not service.finish_store(lease.lease_id, True)
        # The generation can be staged and stored again.
        assert index.begin(entry)
        again = service.prepare_store(entry, 0)
        assert isinstance(again, CheckpointSlots)
        assert service.finish_store(again.lease_id, True)
        assert index.find((entry.prefix,)) == entry


def test_pending_generation_without_a_lease_is_released() -> None:
    with open_store() as (_service, index, storage, _mapping):
        service = CheckpointPayloadStore(
            storage, index, abandoned_after_seconds=ABANDONED_AFTER
        )
        entry = make_manifest()
        assert index.begin(entry)  # the producer died before any PREPARE_STORE
        time.sleep(ABANDONED_AFTER + 0.1)
        assert service.reclaim_abandoned() == 1
        assert index.begin(entry)


def test_abandoned_ready_retrieve_keeps_copy_ownership() -> None:
    with open_store() as (_service, index, storage, mapping):
        service = CheckpointPayloadStore(
            storage, index, abandoned_after_seconds=ABANDONED_AFTER
        )
        entry = replace(make_manifest(), world_size=1)
        assert index.begin(entry)
        lease = service.prepare_store(entry, 0)
        assert isinstance(lease, CheckpointSlots)
        fill(mapping, lease, 0)
        assert service.finish_store(lease.lease_id, True)
        lease_id = service.begin_retrieve(entry, 0)
        assert lease_id is not None
        assert isinstance(poll(service, lease_id), CheckpointSlots)
        keys = [key for group in checkpoint_object_keys(entry, 0) for key in group]
        # The reader holds the pages until it releases them.
        assert storage.delete_l1_keys(keys)[0] == 0
        time.sleep(ABANDONED_AFTER + 0.1)
        assert service.reclaim_abandoned() == 0
        assert service.report_status()["retrieve_leases"] == 1
        assert storage.delete_l1_keys(keys)[0] == 0
        service.finish_retrieve(lease_id)
        assert storage.delete_l1_keys(keys)[0] == len(keys)


def test_zero_age_disables_automatic_reclaim() -> None:
    with open_store() as (_service, index, _storage, _mapping):
        service = CheckpointPayloadStore(_storage, index, abandoned_after_seconds=0)
        first = replace(make_manifest(), world_size=1)
        assert index.begin(first)
        assert isinstance(service.prepare_store(first, 0), CheckpointSlots)
        service._last_reclaim -= 3600  # as if the reclaim interval had passed
        second = replace(make_manifest(), world_size=1)
        assert index.begin(second)
        assert isinstance(service.prepare_store(second, 0), CheckpointSlots)
        assert service.report_status()["store_leases"] == 2
