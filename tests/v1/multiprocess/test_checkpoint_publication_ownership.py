# SPDX-License-Identifier: Apache-2.0
"""Publication must own reused pages and every completed rank until commit."""

# Standard
from dataclasses import replace
import hashlib
import time

# First Party
from tests.v1.multiprocess.test_checkpoint_storage import (
    CheckpointSlots,
    checkpoint_object_keys,
    make_content_manifest,
    make_manifest,
    open_store,
    poll,
    publish_all,
)


def fill(mapping, lease: CheckpointSlots) -> None:
    for group in lease.groups:
        for slot in group:
            if slot is not None:
                mapping[slot.offset : slot.offset + slot.length] = b"\x33" * slot.length


def test_reused_pages_remain_owned_until_publication() -> None:
    key = lambda value: hashlib.sha256(value.encode()).hexdigest()
    older = make_content_manifest(
        "older", prefix_token=3, attention_keys=(key("a"), key("b"), key("old"))
    )
    newer = make_content_manifest(
        "newer", prefix_token=4, attention_keys=(key("a"), key("b"), key("new"))
    )
    with open_store() as (service, index, storage, mapping):
        publish_all(service, index, mapping, older)
        assert index.begin(newer)
        lease = service.prepare_store(newer, 0)
        assert isinstance(lease, CheckpointSlots)
        shared = list(checkpoint_object_keys(newer, 0)[0][:2])
        assert lease.groups[0][:2] == (None, None)
        assert storage.delete_l1_keys(shared, force=False) == (0, 2)
        fill(mapping, lease)
        assert service.finish_store(lease.lease_id, True)
        retrieve = service.begin_retrieve(newer, 0)
        assert retrieve is not None
        restored = poll(service, retrieve)
        assert isinstance(restored, CheckpointSlots)
        service.finish_retrieve(restored.lease_id)


def test_completed_rank_stays_owned_until_all_ranks_publish() -> None:
    manifest = replace(make_manifest(), world_size=2)
    with open_store() as (service, index, storage, mapping):
        assert index.begin(manifest)
        first = service.prepare_store(manifest, 0)
        assert isinstance(first, CheckpointSlots)
        fill(mapping, first)
        assert not service.finish_store(first.lease_id, True)
        keys = [key for group in checkpoint_object_keys(manifest, 0) for key in group]
        assert storage.delete_l1_keys(keys, force=False) == (0, len(keys))
        second = service.prepare_store(manifest, 1)
        assert isinstance(second, CheckpointSlots)
        fill(mapping, second)
        assert service.finish_store(second.lease_id, True)
        assert storage.delete_l1_keys(keys, force=False) == (len(keys), 0)


def test_abort_releases_completed_rank_but_not_live_writer() -> None:
    manifest = replace(make_manifest(), world_size=2)
    with open_store() as (service, index, storage, mapping):
        assert index.begin(manifest)
        first = service.prepare_store(manifest, 0)
        second = service.prepare_store(manifest, 1)
        assert isinstance(first, CheckpointSlots)
        assert isinstance(second, CheckpointSlots)
        fill(mapping, first)
        assert not service.finish_store(first.lease_id, True)
        index.abort(manifest.generation)
        keys = [key for group in checkpoint_object_keys(manifest, 0) for key in group]
        live = [key for group in checkpoint_object_keys(manifest, 1) for key in group]
        assert storage.delete_l1_keys(keys, force=False) == (len(keys), 0)
        assert storage.delete_l1_keys(live, force=False) == (0, len(live))
        assert not service.finish_store(second.lease_id, False)
        assert not service.finish_store(second.lease_id, False)


def test_copy_ownership_survives_client_lock_expiry() -> None:
    manifest = replace(make_manifest(), world_size=1)
    with open_store(lock_ttl_seconds=1) as (service, index, storage, mapping):
        assert index.begin(manifest)
        lease = service.prepare_store(manifest, 0)
        assert isinstance(lease, CheckpointSlots)
        keys = [key for group in checkpoint_object_keys(manifest, 0) for key in group]
        time.sleep(1.1)
        assert storage.delete_l1_keys(keys) == (0, len(keys))
        fill(mapping, lease)
        assert service.finish_store(lease.lease_id, True)
        retrieve = service.begin_retrieve(manifest, 0)
        assert retrieve is not None
        assert isinstance(poll(service, retrieve), CheckpointSlots)
        time.sleep(1.1)
        assert storage.delete_l1_keys(keys) == (0, len(keys))
        service.finish_retrieve(retrieve)
        assert storage.delete_l1_keys(keys) == (len(keys), 0)


def test_quiesce_prevents_admission_after_idle_check() -> None:
    with open_store() as (service, index, _, _):
        manifest = make_manifest()
        assert index.begin(manifest)
        service.quiesce()
        assert service.prepare_store(manifest, 0) is None
        assert service.begin_retrieve(manifest, 0) is None


def test_failed_copy_releases_internal_writer_after_client_ttl() -> None:
    manifest = replace(make_manifest(), world_size=1)
    with open_store(lock_ttl_seconds=1) as (service, index, storage, _):
        assert index.begin(manifest)
        lease = service.prepare_store(manifest, 0)
        assert isinstance(lease, CheckpointSlots)
        time.sleep(1.1)
        assert not service.finish_store(lease.lease_id, False)
        assert storage.get_l1_usage()[0] == 0
