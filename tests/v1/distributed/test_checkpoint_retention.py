# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for checkpoint supersession and write-on-evict bookkeeping.

Written against the CheckpointRetention contract: superseded pages are never
held back from eviction, current pages are held until an L2 copy exists or
their write times out, and L2 residency follows adapter listener events.
"""

# Standard
from unittest.mock import patch

# First Party
from lmcache.v1.distributed.api import RECURRENT_CHECKPOINT_MODEL_PREFIX, ObjectKey
from lmcache.v1.distributed.checkpoint_retention import CheckpointRetention
from lmcache.v1.distributed.storage_controllers.checkpoint_evict_store_policy import (
    CheckpointEvictStorePolicy,
)
from lmcache.v1.distributed.storage_controllers.store_policy import (
    create_store_policy,
    get_registered_store_policies,
)


def make_key(chunk_id: int, checkpoint: bool = True) -> ObjectKey:
    model = (
        f"{RECURRENT_CHECKPOINT_MODEL_PREFIX}v3-{'a' * 64}"
        if checkpoint
        else "test_model"
    )
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id), model_name=model, kv_rank=0
    )


def test_evict_policy_is_registered_and_requests_write_on_evict():
    assert "checkpoint_on_evict" in get_registered_store_policies()
    policy = create_store_policy("checkpoint_on_evict")
    assert isinstance(policy, CheckpointEvictStorePolicy)
    assert policy.writes_checkpoints_on_evict()
    assert not create_store_policy("checkpoint_on_reuse").writes_checkpoints_on_evict()
    assert not create_store_policy("default").writes_checkpoints_on_evict()


def test_without_write_on_evict_nothing_is_held_back():
    retention = CheckpointRetention()
    assert not retention.needs_persist_before_evict(make_key(1))


def test_current_checkpoint_page_is_held_until_l2_copy_exists():
    persisted: list[list[ObjectKey]] = []
    retention = CheckpointRetention(write_on_evict=True, persist=persisted.append)
    retention.listener_for(0)
    key = make_key(1)
    assert retention.needs_persist_before_evict(key)
    assert retention.request_persist([key]) == 1
    # A second eviction pass must not queue the same write again.
    assert retention.request_persist([key]) == 0
    assert persisted == [[key]]
    assert retention.pending_count() == 1
    assert retention.needs_persist_before_evict(key)

    retention.record_l2_present(0, [key], [4096])
    assert retention.is_l2_resident(key)
    assert not retention.needs_persist_before_evict(key)
    status = retention.report_status()
    assert status["write_on_evict_requests"] == 1
    assert status["write_on_evict_persisted"] == 1
    assert status["write_on_evict_pending"] == 0

    # Deleted from L2: the page needs a new write before L1 may drop it.
    retention.record_l2_absent(0, [key])
    assert retention.needs_persist_before_evict(key)


def test_ordinary_kv_chunks_are_never_held_or_tracked():
    retention = CheckpointRetention(write_on_evict=True)
    key = make_key(1, checkpoint=False)
    assert not retention.needs_persist_before_evict(key)
    retention.record_l2_present(0, [key], [4096])
    assert not retention.is_l2_resident(key)


def test_superseded_page_still_needs_persistence():
    retention = CheckpointRetention(write_on_evict=True)
    old, shared = make_key(1), make_key(2)
    assert retention.mark_superseded("gen-old", [old, shared]) == 2
    assert retention.is_superseded_generation("gen-old")
    assert retention.needs_persist_before_evict(old)
    # A newer checkpoint references the shared page again.
    retention.mark_current([shared])
    assert not retention.is_superseded(shared)
    assert retention.needs_persist_before_evict(shared)
    assert retention.superseded_keys() == [old]
    assert retention.mark_superseded("gen-old-2", [old]) == 0


def test_write_age_alone_does_not_allow_last_copy_eviction():
    retention = CheckpointRetention(write_on_evict=True, persist_timeout=5.0)
    key = make_key(1)
    with patch("time.monotonic", return_value=100.0):
        retention.request_persist([key])
    with patch("time.monotonic", return_value=104.0):
        assert retention.needs_persist_before_evict(key)
    with patch("time.monotonic", return_value=106.0):
        assert retention.needs_persist_before_evict(key)
    assert retention.report_status()["write_on_evict_timeouts"] == 0
    assert retention.pending_count() == 1


def test_superseded_in_adapter_respects_residency_and_budget():
    retention = CheckpointRetention()
    keys = [make_key(i) for i in range(5)]
    retention.mark_superseded("gen", keys)
    retention.record_l2_present(0, keys[:4], [100] * 4)
    retention.record_l2_present(1, keys[4:], [100])
    victims, size = retention.superseded_in_adapter(0, 250)
    assert victims == keys[:3]
    assert size == 300
    victims, size = retention.superseded_in_adapter(1, 10_000)
    assert victims == keys[4:]
    retention.forget_adapter(1)
    assert retention.superseded_in_adapter(1, 10_000) == ([], 0)


def test_listener_records_store_and_delete_events():
    retention = CheckpointRetention()
    listener = retention.listener_for(3)
    key = make_key(1)
    listener.on_l2_keys_stored([key], [64])
    assert retention.is_l2_resident(key)
    listener.on_l2_keys_deleted([key])
    assert not retention.is_l2_resident(key)


def test_bounds_forget_oldest_entries():
    retention = CheckpointRetention(max_superseded=2, max_tracked=2)
    keys = [make_key(i) for i in range(3)]
    retention.mark_superseded("gen", keys)
    assert retention.superseded_keys() == keys[1:]
    retention.record_l2_present(0, keys, [1, 1, 1])
    # Supersession hints are bounded; replica presence cannot silently expire.
    assert retention.is_l2_resident(keys[0])
    assert retention.is_l2_resident(keys[2])


def test_observations_report_every_stat_and_l2_checkpoint_bytes():
    retention = CheckpointRetention(write_on_evict=True)
    retention.mark_superseded("gen", [make_key(1)])
    retention.record_l2_present(0, [make_key(2), make_key(3)], [100, 50])
    stats = {attrs["stat"]: value for value, attrs in retention.observations()}
    assert stats["superseded_pages"] == 1
    assert stats["l2_checkpoint_bytes"] == 150
    assert stats["l2_resident_pages_tracked"] == 2
    assert "write_on_evict" not in stats
