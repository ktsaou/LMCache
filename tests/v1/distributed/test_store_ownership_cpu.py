# SPDX-License-Identifier: Apache-2.0
"""Persistence ownership, failure and byte bounds without a GPU runtime."""

# Standard
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
import threading
import time
import uuid

# Third Party
import torch

# First Party
from lmcache.v1.distributed.api import (
    RECURRENT_CHECKPOINT_MODEL_PREFIX,
    MemoryLayoutDesc,
    ObjectKey,
    PrefetchRequestSpec,
)
from lmcache.v1.distributed.checkpoint_retention import CheckpointRetention
from lmcache.v1.distributed.config import L1ManagerConfig, L1MemoryManagerConfig
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import MockL2AdapterConfig
from lmcache.v1.distributed.l2_adapters.fs_l2_adapter import (
    FSL2Adapter,
    FSL2AdapterConfig,
)
from lmcache.v1.distributed.l2_adapters.native_connector_l2_adapter import (
    NativeConnectorL2Adapter,
    _object_key_to_string,
)
from lmcache.v1.distributed.storage_controllers.store_controller import StoreController
from lmcache.v1.distributed.storage_controllers.prefetch_controller import (
    PrefetchController,
)
from lmcache.v1.distributed.storage_controllers.prefetch_policy import (
    DefaultPrefetchPolicy,
)
from lmcache.v1.distributed.storage_controllers.store_policy import (
    AdapterDescriptor,
    DefaultStorePolicy,
)
from tests.v1.distributed.test_native_connector_l2_adapter import MockNativeConnector
from tests.v1.distributed.test_fs_l2_adapter_persistence import _memory_obj
from tests.v1.multiprocess.test_checkpoint_storage import wait_for


class RecordingConnector(MockNativeConnector):
    def __init__(self) -> None:
        super().__init__()
        self.submissions: list[list[str]] = []

    def submit_batch_set(self, keys: list[str], memoryviews: list) -> int:
        self.submissions.append(list(keys))
        return super().submit_batch_set(keys, memoryviews)


@contextmanager
def pipeline(
    replicas: int = 1,
) -> Iterator[tuple[L1Manager, StoreController, list[RecordingConnector]]]:
    l1 = L1Manager(
        L1ManagerConfig(
            L1MemoryManagerConfig(
                65536, False, shm_name=f"lmcache_test_{uuid.uuid4().hex}"
            ),
            read_ttl_seconds=1,
        )
    )
    clients = [RecordingConnector() for _ in range(replicas)]
    adapters = [
        NativeConnectorL2Adapter(client, initial_key_sizes={}) for client in clients
    ]
    retention = CheckpointRetention()
    for i, adapter in enumerate(adapters):
        adapter.register_listener(retention.listener_for(i))
    controller = StoreController(
        l1,
        adapters,
        [
            AdapterDescriptor(
                i, MockL2AdapterConfig(max_size_gb=1, mock_bandwidth_gb=1)
            )
            for i in range(replicas)
        ],
        DefaultStorePolicy(),
        retention=retention,
        max_pending_bytes=8192,
    )
    try:
        yield l1, controller, clients
    finally:
        # Tests start the controller after inspecting the queue handoff.
        controller.stop()
        for adapter in adapters:
            adapter.close()
        controller.release_stopped_ownership()
        l1.close()


def write(
    l1: L1Manager, count: int = 2, *, checkpoint: bool = False
) -> list[ObjectKey]:
    model = (
        RECURRENT_CHECKPOINT_MODEL_PREFIX + "test" if checkpoint else "ordinary-test"
    )
    keys = [ObjectKey(ObjectKey.IntHash2Bytes(i), model, 0) for i in range(count)]
    layout = MemoryLayoutDesc([torch.Size([4096])], [torch.uint8])
    results = l1.reserve_write(keys, [False] * count, layout)
    for error, obj in results.values():
        assert error == L1Error.SUCCESS and obj is not None
        obj.tensor.fill_(33)
    l1.finish_write(keys)
    return keys


def test_queue_owns_pages_before_store_loop_and_bounds_bytes() -> None:
    with pipeline() as (l1, controller, _):
        keys = write(l1, 3)
        status = controller.report_status()
        assert status["owned_source_bytes"] == status["max_pending_bytes"] == 8192
        assert status["rejected_queue_pages"] == 1
        result = l1.delete(keys)
        assert result[keys[0]] == result[keys[1]] == L1Error.KEY_IS_LOCKED
        assert result[keys[2]] == L1Error.SUCCESS
        controller.start()
        wait_for(lambda: not controller.has_pending_work(), "store did not drain")
        assert all(value == L1Error.SUCCESS for value in l1.delete(keys[:2]).values())


def test_active_store_ownership_survives_ttl_and_stop() -> None:
    with pipeline() as (l1, controller, clients):
        clients[0].suppress_set_completion = True
        keys = write(l1)
        controller.start()
        wait_for(
            lambda: controller.report_status()["in_flight_task_count"] == 1,
            "no submission",
        )
        time.sleep(1.1)
        controller.cancel_queued(keys)
        assert all(value == L1Error.KEY_IS_LOCKED for value in l1.delete(keys).values())
        clients[0].complete_suppressed_sets()
        wait_for(lambda: not controller.has_pending_work(), "completion did not drain")
        assert all(value == L1Error.SUCCESS for value in l1.delete(keys).values())


def test_cancelled_queue_never_submits_recycled_buffers() -> None:
    with pipeline() as (l1, controller, clients):
        keys = write(l1)
        controller.cancel_queued(keys)
        assert all(value == L1Error.SUCCESS for value in l1.delete(keys).values())
        controller.start()
        assert not controller.has_pending_work()
        assert clients[0].submissions == []


def test_partial_failure_retries_only_missing_pages() -> None:
    with pipeline() as (l1, controller, clients):
        keys = write(l1, checkpoint=True)
        failed = _object_key_to_string(keys[1])
        clients[0].fail_set_keys.add(failed)
        clients[0].report_set_per_key_results = True
        controller.start()
        wait_for(lambda: len(clients[0].submissions) >= 2, "partial store not retried")
        assert clients[0].submissions[1] == [failed]
        clients[0].fail_set_keys.clear()
        wait_for(lambda: not controller.has_pending_work(), "retry did not drain")
        assert all(value == L1Error.SUCCESS for value in l1.delete(keys).values())


def test_cancel_and_rewrite_same_keys_does_not_duplicate_io() -> None:
    with pipeline() as (l1, controller, clients):
        keys = write(l1)
        controller.cancel_queued(keys)
        assert all(value == L1Error.SUCCESS for value in l1.delete(keys).values())
        assert write(l1) == keys
        controller.start()
        wait_for(
            lambda: not controller.has_pending_work(), "rewritten work did not drain"
        )
        assert sum(len(batch) for batch in clients[0].submissions) == len(keys)


def test_replication_serializes_destinations_within_byte_budget() -> None:
    with pipeline(2) as (l1, controller, clients):
        clients[0].suppress_set_completion = True
        write(l1)
        controller.start()
        wait_for(lambda: len(clients[0].submissions) == 1, "first replica missing")
        assert not clients[1].submissions
        assert controller.report_status()["owned_source_bytes"] == 8192
        clients[0].complete_suppressed_sets()
        wait_for(lambda: not controller.has_pending_work(), "replication did not drain")
        assert len(clients[1].submissions) == 1


def test_failed_first_destination_does_not_starve_healthy_replica() -> None:
    with pipeline(2) as (l1, controller, clients):
        clients[0].raise_set_submit = True
        write(l1, checkpoint=True)
        controller.start()
        wait_for(lambda: bool(clients[1].submissions), "healthy replica was starved")
        clients[0].raise_set_submit = False
        wait_for(
            lambda: not controller.has_pending_work(), "recovered replica did not drain"
        )


def test_prefetch_owns_destination_through_ttl_and_backend_shutdown(
    tmp_path: Path, monkeypatch
) -> None:
    l1 = L1Manager(
        L1ManagerConfig(
            L1MemoryManagerConfig(
                65536, False, shm_name=f"lmcache_test_{uuid.uuid4().hex}"
            ),
            write_ttl_seconds=1,
        )
    )
    config = FSL2AdapterConfig(str(tmp_path))
    adapter = FSL2Adapter(config)
    key = ObjectKey(ObjectKey.IntHash2Bytes(1), "prefetch-ownership", 0)
    task = adapter.submit_store_task([key], [_memory_obj(bytes(4096))])
    wait_for(
        lambda: task in adapter.pop_completed_store_tasks(),
        "seed store did not complete",
    )
    entered, release, closed = threading.Event(), threading.Event(), threading.Event()

    def blocked_read(path: Path, destination: memoryview) -> int:
        entered.set()
        assert release.wait(5)
        data = path.read_bytes()
        destination[:] = data
        return len(data)

    monkeypatch.setattr(adapter, "_use_odirect", True)
    monkeypatch.setattr(adapter, "_os_disk_bs", 4096)
    monkeypatch.setattr(adapter, "_read_with_odirect", blocked_read)
    controller = PrefetchController(
        l1, [adapter], [AdapterDescriptor(0, config)], DefaultPrefetchPolicy()
    )
    controller.start()
    controller.submit_prefetch_request(
        PrefetchRequestSpec(
            [key], {0: MemoryLayoutDesc([torch.Size([4096])], [torch.uint8])}
        )
    )
    stopped = False
    try:
        assert entered.wait(5)
        time.sleep(1.1)
        assert l1.reclaim_abandoned_writes() == 0
        assert l1.delete([key])[key] == L1Error.KEY_IS_LOCKED
        controller.stop()
        stopped = True
        assert l1.get_memory_usage()[0] == 4096

        def close_backend() -> None:
            adapter.close()
            closed.set()

        closer = threading.Thread(target=close_backend)
        closer.start()
        try:
            assert not closed.wait(0.2)
        finally:
            release.set()
            closer.join(5)
        assert closed.is_set()
        controller.release_stopped_ownership()
        assert l1.get_memory_usage()[0] == 0
    finally:
        release.set()
        if not stopped:
            controller.stop()
        if not closed.is_set():
            adapter.close()
        controller.release_stopped_ownership()
        l1.close()
