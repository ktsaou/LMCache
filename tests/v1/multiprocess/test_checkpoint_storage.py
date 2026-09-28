# SPDX-License-Identifier: Apache-2.0
"""Checkpoint payload publication and SHM lease lifetime with real L1 storage."""

# Standard
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from mmap import mmap
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, cast
from unittest.mock import ANY, patch
import hashlib
import json
import threading
import time
import uuid

# Third Party
from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest
import zmq

# First Party
from lmcache.v1.distributed.admission import AdmissionFailure
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.distributed.eviction_policy import (
    IsolatedLRUEvictionPolicy,
    LRUEvictionPolicy,
)
from lmcache.v1.distributed.l2_adapters.config import L2AdaptersConfig
from lmcache.v1.distributed.l2_adapters.fs_l2_adapter import FSL2AdapterConfig
from lmcache.v1.distributed.l2_adapters.fs_native_l2_adapter import (
    FSNativeL2AdapterConfig,
)
from lmcache.v1.distributed.storage_manager import StorageManager
from lmcache.v1.mp_observability.errors import LMCacheTimeoutError
from lmcache.v1.multiprocess import checkpoint_storage, checkpoint_transfer
from lmcache.v1.multiprocess.checkpoint_index import (
    CheckpointIndex,
    CheckpointManifest,
    CheckpointPrefix,
)
from lmcache.v1.multiprocess.checkpoint_storage import (
    CheckpointPayloadStore,
    CheckpointSlots,
    checkpoint_object_keys,
)
from lmcache.v1.multiprocess.checkpoint_transfer import (
    CheckpointTransferJob,
    CheckpointTransferWorker,
    UnsafeCheckpointCopyError,
)
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
from lmcache.v1.multiprocess.http_apis.cache_api import router as cache_router
from lmcache.v1.multiprocess.http_apis.dependencies import build_context
from lmcache.v1.multiprocess.modules.checkpoint import CheckpointModule
from lmcache.v1.multiprocess.modules.management import ManagementModule
from lmcache.v1.multiprocess.mq import MessageQueueClient, MessageQueueServer
from lmcache.v1.multiprocess.posix_shm import shm_open_pool_as_mmap
from lmcache.v1.multiprocess.protocol import (
    RequestType,
    get_handler_type,
    get_payload_classes,
)
from lmcache.v1.multiprocess.protocols.checkpoint import CheckpointLeaseResponse
from lmcache.v1.multiprocess.server import MPCacheServer


@contextmanager
def open_store(
    path: Path | None = None,
    native: bool = False,
    *,
    shm_name: str | None = None,
    store_policy: str = "default",
    shutdown_flush_seconds: float = 30.0,
    admission_timeout_seconds: float = 8.0,
    prefetch_policy: str = "default",
) -> Iterator[tuple[CheckpointPayloadStore, CheckpointIndex, StorageManager, mmap]]:
    name = shm_name or f"lmcache_l1_pool_checkpoint_test_{uuid.uuid4().hex}"
    size = 4 * 1024 * 1024
    storage = StorageManager(
        StorageManagerConfig(
            L1ManagerConfig(L1MemoryManagerConfig(size, False, shm_name=name)),
            EvictionConfig(eviction_policy="LRU"),
            l2_adapter_config=L2AdaptersConfig(
                [
                    (FSNativeL2AdapterConfig if native else FSL2AdapterConfig)(
                        str(path / "payloads")
                    )
                ]
                if path is not None
                else []
            ),
            store_policy=store_policy,
            prefetch_policy=prefetch_policy,
            checkpoint_shutdown_flush_seconds=shutdown_flush_seconds,
            store_admission_timeout_seconds=admission_timeout_seconds,
        )
    )
    with ExitStack() as cleanup:
        cleanup.callback(storage.close)
        index = CheckpointIndex(path / "checkpoint-index.sqlite3" if path else None)
        cleanup.callback(index.close)
        mapping = cleanup.enter_context(shm_open_pool_as_mmap(name, size))
        yield CheckpointPayloadStore(storage, index), index, storage, mapping


@pytest.fixture
def store() -> Iterator[
    tuple[CheckpointPayloadStore, CheckpointIndex, StorageManager, mmap]
]:
    with open_store() as resources:
        yield resources


def make_manifest() -> CheckpointManifest:
    return CheckpointManifest(
        uuid.uuid4().hex,
        CheckpointPrefix("weights-and-layout-and-salt", 4096, b"a" * 32, (1, 2, 3)),
        4,
        json.dumps(
            {
                "schema_version": 1,
                "page_groups": [
                    {
                        "name": "target.attention.0",
                        "page_bytes": 128,
                        "positions": [0, 1, 2],
                    },
                    {
                        "name": "target.recurrent.0",
                        "page_bytes": 256,
                        "positions": [16],
                    },
                    {"name": "draft.context.0", "page_bytes": 64, "positions": [7, 8]},
                    {
                        "name": "target-draft-auxiliary",
                        "page_bytes": 96,
                        "positions": [0],
                    },
                ],
            }
        ).encode(),
    )


def make_content_manifest(
    generation: str,
    *,
    prefix_token: int,
    attention_keys: tuple[str, ...],
) -> CheckpointManifest:
    """Build a v2 manifest whose attention pages can span generations."""
    unique = lambda label: hashlib.sha256(label.encode()).hexdigest()
    return CheckpointManifest(
        generation,
        CheckpointPrefix(
            "weights-and-layout-and-salt",
            4096,
            b"a" * 32,
            (1, 2, prefix_token),
        ),
        1,
        json.dumps(
            {
                "schema_version": 2,
                "page_groups": [
                    {
                        "name": "target.attention.0",
                        "page_bytes": 128,
                        "positions": [0, 1, 2],
                        "content_keys": list(attention_keys),
                    },
                    {
                        "name": "target.recurrent.0",
                        "page_bytes": 256,
                        "positions": [16],
                        "content_keys": [unique(f"recurrent-{prefix_token}")],
                    },
                    {
                        "name": "target-draft-auxiliary",
                        "page_bytes": 96,
                        "positions": [0],
                        "content_keys": [unique(f"auxiliary-{prefix_token}")],
                    },
                ],
            }
        ).encode(),
    )


@contextmanager
def open_checkpoint_rpc() -> Iterator[
    tuple[MessageQueueClient, CheckpointModule, mmap, str]
]:
    """Expose real SHM and a typed message queue to storage/worker contract tests."""
    name = f"lmcache_l1_pool_checkpoint_rpc_{uuid.uuid4().hex}"
    with open_store(shm_name=name) as (_, _, storage, mapping):
        module = CheckpointModule(
            cast(
                MPCacheServerContext,
                SimpleNamespace(
                    storage_manager=storage,
                    shm_pool_info={"shm_name": name, "pool_size": 4 * 1024 * 1024},
                ),
            )
        )
        url = f"inproc://checkpoint-{uuid.uuid4().hex}"
        context = zmq.Context.instance()
        server = MessageQueueServer(url, context)
        blocking: list[RequestType] = []
        for spec in module.get_handlers():
            server.add_handler(
                spec.request_type,
                get_payload_classes(spec.request_type),
                get_handler_type(spec.request_type),
                spec.handler,
            )
            if spec.request_type != RequestType.CHECKPOINT_CAPABILITIES:
                blocking.append(spec.request_type)
        server.add_normal_thread_pool(blocking, max_workers=4)
        server.start()
        client = MessageQueueClient(url, context)

        try:
            yield client, module, mapping, name
        finally:
            client.close()
            server.close()
            module.close()


def test_checkpoint_rpc_roundtrip_uses_metadata_and_shared_bytes() -> None:
    """All-rank publication and SHM leases survive the actual MQ codec/dispatch."""
    with open_checkpoint_rpc() as (client, module, mapping, name):

        def call(kind: RequestType, *args: object) -> Any:
            return client.submit_request(kind, list(args)).result(timeout=5)

        # Appending checkpoint operations must not renumber existing wire IDs.
        assert RequestType.COMMIT_RETRIEVE.value == 19
        assert RequestType.GET_EXPERIMENTAL.value == 32
        capability = call(RequestType.CHECKPOINT_CAPABILITIES)
        assert capability.format_version == 1
        assert capability.shm_name == name
        assert not capability.durable_index
        entry = make_manifest()
        assert call(RequestType.CHECKPOINT_BEGIN, entry)
        for rank in range(4):
            assert call(RequestType.CHECKPOINT_FIND, (entry.prefix,)) is None
            lease = call(RequestType.CHECKPOINT_PREPARE_STORE, entry, rank)
            assert lease.status == "ready"
            for group_id, group in enumerate(lease.slots):
                for page_id, (offset, size) in enumerate(group):
                    mapping[offset : offset + size] = (
                        bytes([rank * 16 + group_id * 4 + page_id]) * size
                    )
            assert call(RequestType.CHECKPOINT_FINISH_STORE, lease.lease_id, True) == (
                rank == 3
            )
        assert call(RequestType.CHECKPOINT_FIND, (entry.prefix,)) == entry
        for rank in range(4):
            lookup = call(RequestType.CHECKPOINT_BEGIN_RETRIEVE, entry, rank)
            assert lookup.status == "pending"
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                lease = call(RequestType.CHECKPOINT_POLL_RETRIEVE, lookup.lease_id)
                if lease.status != "pending":
                    break
            assert lease.status == "ready"
            with pytest.raises(RuntimeError, match="must drain"):
                module.close()
            for group_id, group in enumerate(lease.slots):
                for page_id, (offset, size) in enumerate(group):
                    assert mapping[offset : offset + size] == (
                        bytes([rank * 16 + group_id * 4 + page_id]) * size
                    )
            assert call(RequestType.CHECKPOINT_FINISH_RETRIEVE, lease.lease_id)
        status = module.report_status()["recurrent_checkpoints"]
        assert status["store_leases"] == status["retrieve_leases"] == 0


def test_content_addressed_pages_skip_resident_duplicate_copies(store) -> None:
    service, index, _storage, mapping = store
    shared = tuple(
        hashlib.sha256(f"attention-{i}".encode()).hexdigest() for i in range(3)
    )
    first = make_content_manifest(
        "recurrent-content-v1:" + "1" * 64,
        prefix_token=3,
        attention_keys=shared,
    )
    second = make_content_manifest(
        "recurrent-content-v1:" + "2" * 64,
        prefix_token=4,
        attention_keys=shared,
    )

    assert index.begin(first)
    first_lease = service.prepare_store(first, 0)
    assert first_lease is not None
    assert all(slot is not None for group in first_lease.groups for slot in group)
    for group_id, group in enumerate(first_lease.groups):
        for slot in group:
            assert slot is not None
            mapping[slot.offset : slot.offset + slot.length] = (
                bytes([group_id + 1]) * slot.length
            )
    assert service.finish_store(first_lease.lease_id, True)

    assert index.begin(second)
    second_lease = service.prepare_store(second, 0)
    assert second_lease is not None
    assert second_lease.groups[0] == (None, None, None)
    assert all(slot is not None for group in second_lease.groups[1:] for slot in group)
    for group_id, group in enumerate(second_lease.groups[1:], start=1):
        for slot in group:
            assert slot is not None
            mapping[slot.offset : slot.offset + slot.length] = (
                bytes([group_id + 1]) * slot.length
            )
    assert service.finish_store(second_lease.lease_id, True)
    assert checkpoint_object_keys(first, 0)[0] == checkpoint_object_keys(second, 0)[0]
    assert checkpoint_object_keys(first, 0)[1:] != checkpoint_object_keys(second, 0)[1:]


@pytest.mark.parametrize("first_succeeds", [True, False])
def test_content_addressed_pages_coalesce_with_inflight_writer(
    store, first_succeeds: bool
) -> None:
    """Overlapping boundary generations wait for shared immutable pages.

    A successful owner makes the shared attention pages no-copy objects for the
    waiter. If the owner aborts, the waiter reserves and writes those pages.
    Neither outcome may discard the waiter's complete generation.
    """
    service, index, _storage, mapping = store
    shared = tuple(
        hashlib.sha256(f"inflight-attention-{i}".encode()).hexdigest() for i in range(3)
    )
    first = make_content_manifest(
        "recurrent-content-v1:" + "3" * 64,
        prefix_token=31,
        attention_keys=shared,
    )
    second = make_content_manifest(
        "recurrent-content-v1:" + "4" * 64,
        prefix_token=32,
        attention_keys=shared,
    )

    assert index.begin(first)
    assert index.begin(second)
    first_lease = service.prepare_store(first, 0)
    assert first_lease is not None
    fill(mapping, first_lease, 1)

    assert service.prepare_store(second, 0) is AdmissionFailure.BUSY
    assert service.finish_store(first_lease.lease_id, first_succeeds) is first_succeeds
    second_lease = service.prepare_store(second, 0)

    assert second_lease is not None
    if first_succeeds:
        assert second_lease.groups[0] == (None, None, None)
    else:
        assert all(slot is not None for slot in second_lease.groups[0])
    for group_id, group in enumerate(second_lease.groups):
        for slot in group:
            if slot is not None:
                mapping[slot.offset : slot.offset + slot.length] = (
                    bytes([group_id + 2]) * slot.length
                )
    assert service.finish_store(second_lease.lease_id, True)
    assert index.find((second.prefix,)) == second


def test_transfer_worker_retries_inflight_content_without_blocking_rpc() -> None:
    """A producer retries busy content while the owning GPU copy completes."""
    with open_checkpoint_rpc() as (client, module, mapping, _name):
        shared = tuple(
            hashlib.sha256(f"worker-inflight-{i}".encode()).hexdigest()
            for i in range(3)
        )
        first = make_content_manifest(
            "recurrent-content-v1:" + "5" * 64,
            prefix_token=41,
            attention_keys=shared,
        )
        second = make_content_manifest(
            "recurrent-content-v1:" + "6" * 64,
            prefix_token=42,
            attention_keys=shared,
        )
        first_started = threading.Event()
        release_first = threading.Event()

        def copy_pages(
            job: CheckpointTransferJob, lease: CheckpointLeaseResponse
        ) -> None:
            if job.manifest == first:
                first_started.set()
                assert release_first.wait(timeout=5)
            for group_id, group in enumerate(lease.slots):
                for offset, size in group:
                    if offset >= 0:
                        mapping[offset : offset + size] = bytes([group_id + 3]) * size

        worker = CheckpointTransferWorker(client, copy_pages, workers=2)
        assert module.begin(first)
        assert module.begin(second)
        first_result = worker.submit(CheckpointTransferJob(first, 0, "STORE", ()))
        assert first_result is not None and first_started.wait(timeout=5)
        second_result = worker.submit(CheckpointTransferJob(second, 0, "STORE", ()))
        assert second_result is not None
        time.sleep(0.05)
        assert not second_result.done()
        release_first.set()
        assert first_result.result(timeout=5)
        assert second_result.result(timeout=5)
        assert module.find((first.prefix,)) == first
        assert module.find((second.prefix,)) == second
        worker.close()


def test_background_checkpoint_copy_retains_lease_until_callback_drains() -> None:
    """A blocked copy is not published and cannot surrender its buffer to close."""
    with open_checkpoint_rpc() as (client, module, mapping, _name):
        started, drained = threading.Event(), threading.Event()
        expected = b"\x25" * 128

        def copy_pages(job, lease) -> None:
            started.set()
            assert drained.wait(timeout=5)
            for group in lease.slots:
                for offset, size in group:
                    if job.direction == "STORE":
                        mapping[offset : offset + size] = expected[:1] * size
                    else:
                        assert mapping[offset : offset + size] == expected[:1] * size

        worker = CheckpointTransferWorker(client, copy_pages, workers=1, max_pending=1)
        entry = replace(make_manifest(), world_size=1)
        assert module.begin(entry)
        job = CheckpointTransferJob(entry, 0, "STORE", ())
        completion = worker.submit(job)
        try:
            assert completion is not None and started.wait(timeout=5)
            assert not completion.done()
            assert worker.submit(job) is None
            assert module.find((entry.prefix,)) is None
            with pytest.raises(RuntimeError, match="must drain"):
                module.close()
            drained.set()
            assert completion.result(timeout=5)
            assert module.find((entry.prefix,)) == entry
            restore = worker.submit(replace(job, direction="RETRIEVE"))
            assert restore is not None and restore.result(timeout=5)
        finally:
            drained.set()
            worker.close()
        assert module.report_status()["recurrent_checkpoints"]["retrieve_leases"] == 0


@pytest.mark.parametrize("unsafe", [False, True])
def test_checkpoint_copy_failure_releases_only_proven_drained_leases(
    unsafe: bool,
) -> None:
    with open_checkpoint_rpc() as (client, module, _mapping, _name):
        lease_ids = []

        def copy_pages(_job, lease) -> None:
            lease_ids.append(lease.lease_id)
            if unsafe:
                raise UnsafeCheckpointCopyError("injected undrained DMA")
            raise ValueError("injected failure after draining DMA")

        worker = CheckpointTransferWorker(client, copy_pages)
        entry = replace(make_manifest(), world_size=1)
        assert module.begin(entry)
        completion = worker.submit(CheckpointTransferJob(entry, 0, "STORE", ()))
        assert completion is not None
        with pytest.raises(UnsafeCheckpointCopyError if unsafe else ValueError):
            completion.result(timeout=5)
        assert module.find((entry.prefix,)) is None
        if unsafe:
            with pytest.raises(UnsafeCheckpointCopyError):
                worker.close()
            assert module.report_status()["recurrent_checkpoints"]["store_leases"] == 1
            # The test callback submitted no real DMA; simulate worker teardown.
            module.finish_store(lease_ids[0], False)
        else:
            worker.close()
        assert module.report_status()["recurrent_checkpoints"]["store_leases"] == 0


def test_checkpoint_lease_budget_never_recycles_live_copy_buffers(store) -> None:
    _, index, storage, _ = store
    service = CheckpointPayloadStore(storage, index, max_leases=1)
    entry = make_manifest()
    assert index.begin(entry)
    lease = service.prepare_store(entry, 0)
    assert isinstance(lease, CheckpointSlots)
    assert service.begin_retrieve(entry, 0) is None
    assert service.prepare_store(entry, 1) is None
    assert service.report_status()["store_leases"] == 1
    assert not service.finish_store(lease.lease_id, False)
    assert service.report_status()["store_leases"] == 0
    lookup = service.begin_retrieve(entry, 0)
    assert lookup is not None
    assert poll(service, lookup) is False
    assert service.report_status()["retrieve_leases"] == 0


@pytest.mark.parametrize(
    "request_type",
    [
        RequestType.CHECKPOINT_PREPARE_STORE,
        RequestType.CHECKPOINT_BEGIN_RETRIEVE,
        RequestType.CHECKPOINT_POLL_RETRIEVE,
    ],
)
def test_delayed_lease_reply_is_drained_without_starting_gpu_copy(
    request_type: RequestType,
) -> None:
    """An RPC deadline must not abandon a server-owned SHM reservation."""
    with open_checkpoint_rpc() as (client, module, _mapping, _name):
        entry = replace(make_manifest(), world_size=1)
        assert module.begin(entry)
        direction: Literal["STORE", "RETRIEVE"] = (
            "STORE"
            if request_type == RequestType.CHECKPOINT_PREPARE_STORE
            else "RETRIEVE"
        )
        if direction == "RETRIEVE":
            stored = module.prepare_store(entry, 0)
            assert stored.status == "ready"
            assert module.finish_store(stored.lease_id, True)
        delayed = []

        class DelayedReply:
            def __init__(self, future: Any) -> None:
                self.future = future
                self.injected = False
                self.reply = None

            def result(self, timeout: float | None = None) -> Any:
                reply = self.future.result(timeout=timeout)
                if not self.injected and (
                    request_type != RequestType.CHECKPOINT_POLL_RETRIEVE
                    or reply.status == "ready"
                ):
                    self.injected = True
                    self.reply = reply
                    delayed.append(reply)
                    raise LMCacheTimeoutError("injected late lease reply")
                return reply

        class DelayedClient:
            def submit_request(self, kind: RequestType, payload: list[Any]) -> Any:
                future: Any = client.submit_request(kind, payload)
                if kind == request_type and not delayed:
                    return DelayedReply(future)
                return future

        copied = []
        worker = CheckpointTransferWorker(
            cast(MessageQueueClient, DelayedClient()),
            lambda job, lease: copied.append((job, lease)),
        )
        try:
            completion = worker.submit(CheckpointTransferJob(entry, 0, direction, ()))
            assert completion is not None
            with pytest.raises(TimeoutError):
                completion.result(timeout=5)
            assert delayed and not copied
            status = module.report_status()["recurrent_checkpoints"]
            assert status["store_leases"] == status["retrieve_leases"] == 0
            if direction == "RETRIEVE":
                assert module.find((entry.prefix,)) == entry
        finally:
            worker.close()
            # The fault injector submits no GPU work. Reclaim a failed
            # implementation's lease so the reproducer can shut down safely.
            status = module.report_status()["recurrent_checkpoints"]
            if delayed and status["store_leases"]:
                module.finish_store(delayed[-1].lease_id, False)
            if delayed and status["retrieve_leases"]:
                reply = delayed[-1]
                if reply.status == "ready":
                    module.finish_retrieve(reply.lease_id)
                else:
                    module.cancel_retrieve(reply.lease_id)
                    for _ in range(1000):
                        if module.poll_retrieve(reply.lease_id).status == "miss":
                            break
                        time.sleep(0.001)


@pytest.mark.parametrize("direction", ["STORE", "RETRIEVE"])
@pytest.mark.parametrize("lost", [False, True])
def test_checkpoint_finish_acknowledgement_retains_ownership(
    direction: Literal["STORE", "RETRIEVE"],
    lost: bool,
) -> None:
    """A finish reply is reconciled once; unknown completion stops admission."""
    with open_checkpoint_rpc() as (client, module, _mapping, _name):
        entry = replace(make_manifest(), world_size=1)
        assert module.begin(entry)
        if direction == "RETRIEVE":
            stored = module.prepare_store(entry, 0)
            assert module.finish_store(stored.lease_id, True)
        finish = (
            RequestType.CHECKPOINT_FINISH_STORE
            if direction == "STORE"
            else RequestType.CHECKPOINT_FINISH_RETRIEVE
        )
        submissions = []

        class DelayedFinish:
            def __init__(self, future: Any) -> None:
                self.future = future
                self.injected = False

            def result(self, timeout: float | None = None) -> Any:
                reply = self.future.result(timeout=timeout)
                if lost or not self.injected:
                    self.injected = True
                    raise LMCacheTimeoutError("injected late finish acknowledgement")
                return reply

        class DelayedFinishClient:
            def submit_request(self, kind: RequestType, payload: list[Any]) -> Any:
                future: Any = client.submit_request(kind, payload)
                if kind == finish:
                    submissions.append(kind)
                    return DelayedFinish(future)
                return future

        copies = []
        worker = CheckpointTransferWorker(
            cast(MessageQueueClient, DelayedFinishClient()),
            lambda job, lease: copies.append((job, lease)),
        )
        job = CheckpointTransferJob(entry, 0, direction, ())
        completion = worker.submit(job)
        assert completion is not None
        try:
            if lost:
                with pytest.raises(UnsafeCheckpointCopyError, match="ownership"):
                    completion.result(timeout=5)
                assert worker.submit(job) is None
                with pytest.raises(UnsafeCheckpointCopyError):
                    worker.close()
            else:
                assert completion.result(timeout=5)
                worker.close()
            assert submissions == [finish]
            assert len(copies) == 1
            status = module.report_status()["recurrent_checkpoints"]
            assert status["store_leases"] == status["retrieve_leases"] == 0
            assert module.find((entry.prefix,)) == entry
        finally:
            # The injector processes the server operation before delaying its
            # acknowledgement; no real DMA or server lease remains at teardown.
            try:
                worker.close()
            except UnsafeCheckpointCopyError:
                pass


def test_failed_storage_commit_releases_uncommitted_write_reservations(
    store: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A drained copy with a failed commit must not strand exclusively locked pages."""
    service, index, storage, mapping = store
    entry = make_manifest()
    assert index.begin(entry)
    lease = service.prepare_store(entry, 0)
    assert lease is not None
    fill(mapping, lease, 0)

    def fail_commit(_keys: list[Any]) -> None:
        raise RuntimeError("injected storage commit failure")

    with monkeypatch.context() as patch:
        patch.setattr(storage, "finish_write", fail_commit)
        with pytest.raises(RuntimeError, match="injected storage commit failure"):
            service.finish_store(lease.lease_id, True)
    assert index.find((entry.prefix,)) is None
    assert service.report_status()["store_leases"] == 0
    assert index.begin(entry)
    replacement = service.prepare_store(entry, 0)
    assert replacement is not None
    service.finish_store(replacement.lease_id, False)


def test_unresolved_lease_timeout_stops_transfer_admission() -> None:
    """Unknown remote ownership is fatal, never a reusable completed transfer."""
    with open_checkpoint_rpc() as (client, module, _mapping, _name):
        entry = replace(make_manifest(), world_size=1)
        assert module.begin(entry)
        leases = []

        class LostReply:
            def __init__(self, future: Any) -> None:
                self.future = future

            def result(self, timeout: float | None = None) -> Any:
                leases.append(self.future.result(timeout=timeout))
                raise LMCacheTimeoutError("injected unreconciled lease ownership")

        class LostReplyClient:
            def submit_request(self, kind: RequestType, payload: list[Any]) -> Any:
                future: Any = client.submit_request(kind, payload)
                if kind == RequestType.CHECKPOINT_PREPARE_STORE:
                    return LostReply(future)
                return future

        copies = []
        worker = CheckpointTransferWorker(
            cast(MessageQueueClient, LostReplyClient()),
            lambda job, lease: copies.append((job, lease)),
        )
        job = CheckpointTransferJob(entry, 0, "STORE", ())
        completion = worker.submit(job)
        assert completion is not None
        try:
            with pytest.raises(UnsafeCheckpointCopyError, match="ownership"):
                completion.result(timeout=5)
            assert worker.submit(job) is None
            assert not copies
            assert module.report_status()["recurrent_checkpoints"]["store_leases"] == 1
            with pytest.raises(UnsafeCheckpointCopyError):
                worker.close()
        finally:
            # No DMA is submitted by this injector. Process teardown is the
            # required reclamation boundary for an actual lost reply.
            if leases:
                module.finish_store(leases[0].lease_id, False)


def test_payload_capacity_miss_preserves_a_pinned_checkpoint(store: Any) -> None:
    """Oversized admission neither publishes partial data nor evicts a live read."""
    service, index, _storage, mapping = store
    entry = replace(make_manifest(), world_size=1)
    publish_all(service, index, mapping, entry)
    read = poll(service, service.begin_retrieve(entry, 0))
    assert isinstance(read, CheckpointSlots)
    payload = json.loads(entry.payload)
    payload["page_groups"][0]["page_bytes"] = 2 * 1024 * 1024
    oversized = replace(
        entry,
        generation=uuid.uuid4().hex,
        prefix=replace(entry.prefix, tail_tokens=(11, 12)),
        payload=json.dumps(payload).encode(),
    )
    assert index.begin(oversized)
    try:
        assert service.prepare_store(oversized, 0) is None
        assert index.find((oversized.prefix,)) is None
        assert index.find((entry.prefix,)) == entry
        for group_id, group in enumerate(read.groups):
            for page_id, slot in enumerate(group):
                assert slot is not None
                assert mapping[slot.offset : slot.offset + slot.length] == (
                    bytes([group_id * 4 + page_id]) * slot.length
                )
    finally:
        service.finish_retrieve(read.lease_id)
    status = service.report_status()
    assert status["store_leases"] == status["retrieve_leases"] == 0
    replacement = replace(entry, generation=uuid.uuid4().hex)
    publish_all(service, index, mapping, replacement)


def fill(mapping: mmap, lease: CheckpointSlots, rank: int) -> None:
    for group_id, group in enumerate(lease.groups):
        for page_id, slot in enumerate(group):
            assert slot is not None
            mapping[slot.offset : slot.offset + slot.length] = (
                bytes([rank * 16 + group_id * 4 + page_id]) * slot.length
            )


def poll(service: CheckpointPayloadStore, lease_id: str) -> CheckpointSlots | bool:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        result = service.poll_retrieve(lease_id)
        if result is not None:
            return result
        time.sleep(0.001)
    raise AssertionError("checkpoint prefetch did not complete within ten seconds")


def publish_all(
    service: CheckpointPayloadStore,
    index: CheckpointIndex,
    mapping: mmap,
    entry: CheckpointManifest,
) -> None:
    assert index.begin(entry)
    for rank in range(entry.world_size):
        lease = service.prepare_store(entry, rank)
        assert isinstance(lease, CheckpointSlots)
        fill(mapping, lease, rank)
        assert service.finish_store(lease.lease_id, True) == (
            rank == entry.world_size - 1
        )


def test_shm_roundtrip_all_ranks_groups_and_read_lease_lifetime(store: Any) -> None:
    service, index, storage, mapping = store
    entry = make_manifest()
    publish_all(service, index, mapping, entry)
    assert index.find((entry.prefix,)) == entry
    for rank in range(entry.world_size):
        lease = poll(service, service.begin_retrieve(entry, rank))
        assert isinstance(lease, CheckpointSlots)
        for group_id, group in enumerate(lease.groups):
            for page_id, slot in enumerate(group):
                assert slot is not None
                assert (
                    mapping[slot.offset : slot.offset + slot.length]
                    == bytes([rank * 16 + group_id * 4 + page_id]) * slot.length
                )
        keys = [key for group in checkpoint_object_keys(entry, rank) for key in group]
        assert storage.delete_l1_keys(keys)[0] == 0
        service.finish_retrieve(lease.lease_id)
        assert storage.delete_l1_keys(keys)[0] == len(keys)


@pytest.mark.parametrize("expose_slots", [False, True])
def test_http_nonforced_clear_preserves_checkpoint_read_and_write_leases(
    store: Any, expose_slots: bool
) -> None:
    """HTTP eviction retains actual SHM owners before and after slot exposure."""
    service, index, storage, mapping = store
    entry = make_manifest()
    publish_all(service, index, mapping, entry)
    pending = make_manifest()
    assert index.begin(pending)
    writable = service.prepare_store(pending, 0)
    assert writable is not None
    fill(mapping, writable, 0)
    lease_id = service.begin_retrieve(entry, 0)
    assert lease_id is not None
    if expose_slots:
        assert isinstance(poll(service, lease_id), CheckpointSlots)
    context = cast(MPCacheServerContext, SimpleNamespace(storage_manager=storage))
    engine = MPCacheServer(context, [ManagementModule(context)])
    app = FastAPI()
    app.include_router(cache_router)
    app.state.context = build_context(engine)
    with TestClient(app) as client:
        response = client.post("/cache/clear", json={"tier": "l1", "force": False})
        assert response.status_code == 200, response.text
        retained = [key for group in checkpoint_object_keys(entry, 0) for key in group]
        evicted = [key for group in checkpoint_object_keys(entry, 1) for key in group]
        assert storage.get_readable_keys(retained) == retained
        assert storage.get_readable_keys(evicted) == []
        lease = poll(service, lease_id)
        assert isinstance(lease, CheckpointSlots)
        for group_id, group in enumerate(lease.groups):
            for page_id, slot in enumerate(group):
                assert slot is not None
                assert mapping[slot.offset : slot.offset + slot.length] == (
                    bytes([group_id * 4 + page_id]) * slot.length
                )
        service.finish_retrieve(lease_id)
        assert not service.finish_store(writable.lease_id, True)
        committed = [
            key for group in checkpoint_object_keys(pending, 0) for key in group
        ]
        assert storage.get_readable_keys(committed) == committed
        response = client.post("/cache/clear", json={"force": False})
        assert response.status_code == 200
        assert storage.get_readable_keys(retained + committed) == []


@pytest.mark.parametrize("reservation", ["short", "raises"])
@pytest.mark.parametrize("rollback", ["storage", "index"])
def test_store_reservation_rollback_failure_releases_admission(
    store: Any,
    monkeypatch: pytest.MonkeyPatch,
    reservation: str,
    rollback: str,
) -> None:
    """Cleanup failures cannot leave an unissued copy counted as a live lease."""
    service, index, storage, _mapping = store
    entry = make_manifest()
    assert index.begin(entry)

    def fail(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("injected reservation rollback failure")

    with monkeypatch.context() as patch:
        patch.setattr(
            storage,
            "reserve_write_detailed",
            fail if reservation == "raises" else lambda *_args, **_kwargs: {},
        )
        patch.setattr(
            storage if rollback == "storage" else index,
            "abort_write" if rollback == "storage" else "abort",
            fail,
        )
        with pytest.raises(RuntimeError, match="injected"):
            service.prepare_store(entry, 0)
    assert service.report_status()["store_leases"] == 0
    if rollback == "storage":
        assert not index.is_pending(entry)


def test_failed_rank_prevents_publication_after_other_ranks_complete(
    store: Any,
) -> None:
    service, index, storage, mapping = store
    entry = make_manifest()
    assert index.begin(entry)
    leases = []
    for rank in range(entry.world_size):
        lease = service.prepare_store(entry, rank)
        assert lease is not None
        fill(mapping, lease, rank)
        leases.append(lease)
    for rank, lease in enumerate(leases):
        assert not service.finish_store(lease.lease_id, rank != 1)
    assert index.find((entry.prefix,)) is None
    failed_keys = [key for group in checkpoint_object_keys(entry, 1) for key in group]
    assert storage.get_readable_keys(failed_keys) == []
    assert service.prepare_store(entry, 1) is None


def test_rank_cannot_replace_the_staged_payload_layout(store: Any) -> None:
    service, index, _storage, _mapping = store
    entry = make_manifest()
    assert index.begin(entry)
    payload = json.loads(entry.payload)
    payload["page_groups"][0]["page_bytes"] *= 2
    changed = replace(entry, payload=json.dumps(payload).encode())
    with pytest.raises(ValueError, match="changed"):
        service.prepare_store(changed, 0)


def test_missing_page_invalidates_bundle_and_releases_other_read_locks(
    store: Any,
) -> None:
    service, index, storage, mapping = store
    entry = make_manifest()
    publish_all(service, index, mapping, entry)
    keys = [key for group in checkpoint_object_keys(entry, 2) for key in group]
    assert storage.delete_l1_keys(keys[:1])[0] == 1
    assert poll(service, service.begin_retrieve(entry, 2)) is False
    assert index.find((entry.prefix,)) is None
    assert storage.delete_l1_keys(keys[1:])[0] == len(keys) - 1


def test_retrieve_miss_log_tells_missing_pages_from_cancellation(
    store: Any,
) -> None:
    """A failed retrieve says how many pages were readable, and whether the
    engine cancelled it, so eviction and slow storage can be told apart."""
    service, index, storage, mapping = store
    entry = make_manifest()
    publish_all(service, index, mapping, entry)
    missing = [key for group in checkpoint_object_keys(entry, 2) for key in group]
    intact = [key for group in checkpoint_object_keys(entry, 0) for key in group]
    assert storage.delete_l1_keys(missing[:1])[0] == 1
    with patch.object(checkpoint_storage.logger, "info") as info:
        assert poll(service, service.begin_retrieve(entry, 2)) is False
        lease_id = service.begin_retrieve(entry, 0)
        service.cancel_retrieve(lease_id)
        assert poll(service, lease_id) is False
    tokens = entry.prefix.num_tokens
    assert [call.args[1:] for call in info.call_args_list] == [
        (
            tokens,
            2,
            "missed; its checkpoint is no longer listed",
            ANY,
            len(missing) - 1,
            len(missing),
        ),
        (tokens, 0, "was cancelled by the engine", ANY, len(intact), len(intact)),
    ]


def test_retrieve_pending_past_the_deadline_is_cancelled_with_a_warning() -> None:
    """A restore stuck behind storage is logged before the engine recomputes."""

    class Reply:
        def __init__(self, value: Any) -> None:
            self.value = value

        def result(self, timeout: float | None = None) -> Any:
            return self.value

    class SlowStorageClient:
        def __init__(self) -> None:
            self.requests: list[RequestType] = []

        def submit_request(self, kind: RequestType, payload: list[Any]) -> Reply:
            self.requests.append(kind)
            if kind == RequestType.CHECKPOINT_CANCEL_RETRIEVE:
                return Reply(None)
            cancelled = RequestType.CHECKPOINT_CANCEL_RETRIEVE in self.requests
            return Reply(
                CheckpointLeaseResponse("miss" if cancelled else "pending", "lease")
            )

    client = SlowStorageClient()
    worker = CheckpointTransferWorker(
        cast(MessageQueueClient, client),
        lambda job, lease: pytest.fail("a missed restore must not copy"),
        rpc_timeout=0.1,
    )
    entry = replace(make_manifest(), world_size=1)
    try:
        with patch.object(checkpoint_transfer.logger, "warning") as warning:
            completion = worker.submit(CheckpointTransferJob(entry, 0, "RETRIEVE", ()))
            assert completion is not None
            assert completion.result(timeout=5) is False
    finally:
        worker.close()
    assert RequestType.CHECKPOINT_CANCEL_RETRIEVE in client.requests
    message, tokens, rank, _elapsed = warning.call_args.args
    assert "still waiting for storage" in message
    assert (tokens, rank) == (entry.prefix.num_tokens, 0)


def test_cancel_pending_lookup_drains_locks_without_invalidating_payload(
    store: Any,
) -> None:
    service, index, storage, mapping = store
    entry = make_manifest()
    publish_all(service, index, mapping, entry)
    lease_id = service.begin_retrieve(entry, 0)
    service.cancel_retrieve(lease_id)
    assert poll(service, lease_id) is False
    assert index.find((entry.prefix,)) == entry
    keys = [key for group in checkpoint_object_keys(entry, 0) for key in group]
    assert storage.delete_l1_keys(keys)[0] == len(keys)


def test_generation_namespace_and_rank_have_disjoint_payload_keys() -> None:
    entry = make_manifest()
    variants = (
        (entry, 0),
        (entry, 1),
        (replace(entry, generation=uuid.uuid4().hex), 0),
        (replace(entry, prefix=replace(entry.prefix, namespace="different-salt")), 0),
    )
    seen: set[ObjectKey] = set()
    for manifest, rank in variants:
        keys = {
            key for group in checkpoint_object_keys(manifest, rank) for key in group
        }
        assert not (keys & seen)
        seen.update(keys)


@pytest.mark.parametrize("policy_type", [LRUEvictionPolicy, IsolatedLRUEvictionPolicy])
def test_checkpoint_payloads_remain_eligible_for_lru_eviction(policy_type: Any) -> None:
    """Variable-page checkpoint groups must not require nonexistent siblings."""
    policy = policy_type()
    entry = make_manifest()
    all_keys: list[ObjectKey] = []
    for rank in range(entry.world_size):
        for group in checkpoint_object_keys(entry, rank):
            policy.on_keys_created(list(group))
            all_keys.extend(group)

    actions = policy.get_eviction_actions(1.0, cache_salt="")
    assert {key for action in actions for key in action.keys} == set(all_keys)

    # The manifest/retrieval contract owns all-rank completeness. An unrelated
    # pinned page cannot make every other checkpoint object permanently resident.
    pinned = all_keys[0]
    actions = policy.get_eviction_actions(
        1.0, key_eligible_filter=lambda key: key != pinned, cache_salt=""
    )
    assert {key for action in actions for key in action.keys} == set(all_keys) - {
        pinned
    }


def test_sustained_checkpoint_stores_reclaim_capacity_without_losing_a_read(
    store: Any,
) -> None:
    """A bounded RAM pool accepts generations while preserving a live SHM lease."""
    service, index, storage, mapping = store
    pinned_entry = replace(make_manifest(), world_size=1)
    publish_all(service, index, mapping, pinned_entry)
    pinned = poll(service, service.begin_retrieve(pinned_entry, 0))
    assert isinstance(pinned, CheckpointSlots)
    payload = json.loads(pinned_entry.payload)
    for group in payload["page_groups"]:
        group["page_bytes"] = 128 * 1024
    try:
        for sequence in range(16):
            entry = replace(
                pinned_entry,
                generation=uuid.uuid4().hex,
                prefix=replace(pinned_entry.prefix, tail_tokens=(sequence + 100,)),
                payload=json.dumps(payload).encode(),
            )
            publish_all(service, index, mapping, entry)
            assert index.find((entry.prefix,)) == entry
        restored = poll(service, service.begin_retrieve(entry, 0))
        assert isinstance(restored, CheckpointSlots)
        service.finish_retrieve(restored.lease_id)
        for group_id, group in enumerate(pinned.groups):
            for page_id, slot in enumerate(group):
                assert slot is not None
                assert mapping[slot.offset : slot.offset + slot.length] == (
                    bytes([group_id * 4 + page_id]) * slot.length
                )
        status = storage.report_status()["l1_manager"]
        assert status["memory_used_bytes"] <= status["memory_total_bytes"]
        assert storage.get_admission_stats()["exhausted_timeouts"] == 0
    finally:
        service.finish_retrieve(pinned.lease_id)
    assert service.report_status()["store_leases"] == 0
    assert service.report_status()["retrieve_leases"] == 0


@pytest.mark.parametrize("native", [False, True])
def test_filesystem_restore_after_directory_and_storage_restart(
    tmp_path: Path, native: bool
) -> None:
    entry = make_manifest()
    with open_store(tmp_path, native) as (service, index, storage, mapping):
        publish_all(service, index, mapping, entry)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            status = storage.report_status()["store_controller"]
            if status["pending_keys_count"] == status["in_flight_task_count"] == 0:
                break
            time.sleep(0.01)
        else:
            raise AssertionError("filesystem checkpoint writes did not drain")
    with open_store(tmp_path, native) as (service, index, storage, mapping):
        restored = index.find((entry.prefix,))
        assert restored == entry
        for rank in range(entry.world_size):
            keys = [
                key for group in checkpoint_object_keys(entry, rank) for key in group
            ]
            assert storage.get_readable_keys(keys) == []
            lease_id = service.begin_retrieve(restored, rank)
            assert lease_id is not None
            lease = poll(service, lease_id)
            assert isinstance(lease, CheckpointSlots)
            for group_id, group in enumerate(lease.groups):
                for page_id, slot in enumerate(group):
                    assert slot is not None
                    assert (
                        mapping[slot.offset : slot.offset + slot.length]
                        == bytes([rank * 16 + group_id * 4 + page_id]) * slot.length
                    )
            service.finish_retrieve(lease.lease_id)


def drain_l2_stores(storage: StorageManager) -> None:
    """Wait until written and reused keys have reached their L2 adapters."""
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        status = storage.report_status()["store_controller"]
        if (
            status["pending_keys_count"]
            == status["pending_reused_keys_count"]
            == status["in_flight_task_count"]
            == 0
        ):
            return
        time.sleep(0.01)
    raise AssertionError("filesystem checkpoint writes did not drain")


@pytest.mark.parametrize("native", [False, True])
def test_reuse_admission_persists_only_restored_checkpoints(
    tmp_path: Path, native: bool
) -> None:
    """checkpoint_on_reuse writes a checkpoint to L2 only after a restore."""
    restored = make_manifest()
    unused = replace(
        make_manifest(), prefix=replace(restored.prefix, tail_tokens=(1, 2, 4))
    )
    with open_store(tmp_path, native, store_policy="checkpoint_on_reuse") as (
        service,
        index,
        storage,
        mapping,
    ):
        publish_all(service, index, mapping, restored)
        publish_all(service, index, mapping, unused)
        for rank in range(restored.world_size):
            lease_id = service.begin_retrieve(restored, rank)
            assert lease_id is not None
            lease = poll(service, lease_id)
            assert isinstance(lease, CheckpointSlots)
            service.finish_retrieve(lease.lease_id)
        drain_l2_stores(storage)
        time.sleep(0.2)
        drain_l2_stores(storage)
    with open_store(tmp_path, native, store_policy="checkpoint_on_reuse") as (
        service,
        index,
        storage,
        mapping,
    ):
        assert index.find((restored.prefix,)) == restored
        for rank in range(restored.world_size):
            lease_id = service.begin_retrieve(restored, rank)
            assert lease_id is not None
            lease = poll(service, lease_id)
            assert isinstance(lease, CheckpointSlots)
            for group_id, group in enumerate(lease.groups):
                for page_id, slot in enumerate(group):
                    assert slot is not None
                    assert (
                        mapping[slot.offset : slot.offset + slot.length]
                        == bytes([rank * 16 + group_id * 4 + page_id]) * slot.length
                    )
            service.finish_retrieve(lease.lease_id)
        # Never restored before the restart, so its pages never reached L2.
        assert index.find((unused.prefix,)) == unused
        lease_id = service.begin_retrieve(unused, 0)
        assert lease_id is not None
        assert poll(service, lease_id) is False
        assert index.find((unused.prefix,)) is None


def large_manifest(tail: int, page_bytes: int = 128 * 1024) -> CheckpointManifest:
    """A one-rank manifest whose pages fill a noticeable part of the 4 MiB L1."""
    entry = replace(make_manifest(), world_size=1)
    payload = json.loads(entry.payload)
    for group in payload["page_groups"]:
        group["page_bytes"] = page_bytes
    return replace(
        entry,
        generation=uuid.uuid4().hex,
        prefix=replace(entry.prefix, tail_tokens=(tail,)),
        payload=json.dumps(payload).encode(),
    )


def entry_keys(entry: CheckpointManifest) -> list[ObjectKey]:
    return [
        key
        for rank in range(entry.world_size)
        for group in checkpoint_object_keys(entry, rank)
        for key in group
    ]


def restores(
    service: CheckpointPayloadStore, mapping: mmap, entry: CheckpointManifest
) -> bool:
    """Whether every rank of ``entry`` restores with the bytes it published."""
    for rank in range(entry.world_size):
        lease_id = service.begin_retrieve(entry, rank)
        if lease_id is None:
            return False
        lease = poll(service, lease_id)
        if not isinstance(lease, CheckpointSlots):
            return False
        try:
            for group_id, group in enumerate(lease.groups):
                for page_id, slot in enumerate(group):
                    assert slot is not None
                    assert (
                        mapping[slot.offset : slot.offset + slot.length]
                        == bytes([rank * 16 + group_id * 4 + page_id]) * slot.length
                    )
        finally:
            service.finish_retrieve(lease.lease_id)
    return True


def wait_for(condition: Any, message: str, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.02)
    raise AssertionError(message)


@pytest.mark.parametrize("native", [False, True])
def test_evict_admission_writes_checkpoint_only_when_l1_evicts_it(
    tmp_path: Path, native: bool
) -> None:
    """checkpoint_on_evict keeps pages in L1 until eviction, then writes them."""
    first = large_manifest(100)
    with open_store(
        tmp_path, native, store_policy="checkpoint_on_evict", shutdown_flush_seconds=0
    ) as (service, index, storage, mapping):
        retention = storage.checkpoint_retention
        publish_all(service, index, mapping, first)
        drain_l2_stores(storage)
        assert not any(retention.is_l2_resident(key) for key in entry_keys(first))
        assert retention.report_status()["write_on_evict_requests"] == 0
        # Restores do not trigger writes under this policy.
        assert restores(service, mapping, first)
        drain_l2_stores(storage)
        assert retention.report_status()["write_on_evict_requests"] == 0

        later = [large_manifest(101 + i) for i in range(12)]
        for entry in later:
            publish_all(service, index, mapping, entry)
        wait_for(
            lambda: all(retention.is_l2_resident(key) for key in entry_keys(first)),
            "evicted checkpoint pages never reached L2",
        )
        status = retention.report_status()
        assert status["write_on_evict_persisted"] >= len(entry_keys(first))
        assert status["write_on_evict_timeouts"] == 0
        assert storage.get_admission_stats()["exhausted_timeouts"] == 0
        # The newest checkpoint still fits in L1 and has not been written.
        assert not any(retention.is_l2_resident(key) for key in entry_keys(later[-1]))
        drain_l2_stores(storage)
    with open_store(
        tmp_path, native, store_policy="checkpoint_on_evict", shutdown_flush_seconds=0
    ) as (service, index, storage, mapping):
        assert index.find((first.prefix,)) == first
        assert restores(service, mapping, first)


@pytest.mark.parametrize("native", [False, True])
def test_superseded_checkpoint_is_dropped_first_and_never_written(
    tmp_path: Path, native: bool
) -> None:
    """An older turn's unique pages leave L1 first and never reach L2."""
    older, newer = large_manifest(200), large_manifest(201)
    with open_store(
        tmp_path, native, store_policy="checkpoint_on_evict", shutdown_flush_seconds=0
    ) as (service, index, storage, mapping):
        retention = storage.checkpoint_retention
        publish_all(service, index, mapping, older)
        publish_all(service, index, mapping, newer)
        retention.mark_superseded(older.generation, entry_keys(older))
        for i in range(12):
            publish_all(service, index, mapping, large_manifest(300 + i))
        wait_for(
            lambda: all(retention.is_l2_resident(key) for key in entry_keys(newer)),
            "current checkpoint pages never reached L2",
        )
        drain_l2_stores(storage)
        assert not any(retention.is_l2_resident(key) for key in entry_keys(older))
        assert retention.report_status()["l1_superseded_drops"] >= len(
            entry_keys(older)
        )
    with open_store(
        tmp_path, native, store_policy="checkpoint_on_evict", shutdown_flush_seconds=0
    ) as (service, index, storage, mapping):
        assert restores(service, mapping, newer)
        assert not restores(service, mapping, older)


@pytest.mark.parametrize("flush_seconds", [0.0, 30.0])
def test_shutdown_flush_writes_current_checkpoints_for_the_next_start(
    tmp_path: Path, flush_seconds: float
) -> None:
    """Checkpoints still only in L1 reach L2 when the server stops."""
    current, superseded = make_manifest(), large_manifest(400, page_bytes=128)
    with open_store(
        tmp_path,
        True,
        store_policy="checkpoint_on_evict",
        shutdown_flush_seconds=flush_seconds,
    ) as (service, index, storage, mapping):
        publish_all(service, index, mapping, current)
        publish_all(service, index, mapping, superseded)
        storage.checkpoint_retention.mark_superseded(
            superseded.generation, entry_keys(superseded)
        )
        drain_l2_stores(storage)
    with open_store(
        tmp_path, True, store_policy="checkpoint_on_evict", shutdown_flush_seconds=0
    ) as (service, index, storage, mapping):
        assert restores(service, mapping, current) == (flush_seconds > 0)
        # A superseded checkpoint stays listed, so the flush writes it too,
        # after the current ones, and a branch can restore it after a restart.
        assert restores(service, mapping, superseded) == (flush_seconds > 0)


def publish_before_restart(
    path: Path, native: bool, entries: list[CheckpointManifest]
) -> None:
    """Publish checkpoints and let their pages reach L2 before a restart."""
    with open_store(path, native) as (service, index, storage, mapping):
        for entry in entries:
            publish_all(service, index, mapping, entry)
        drain_l2_stores(storage)


def begin_rank0_retrieve(
    service: CheckpointPayloadStore, entry: CheckpointManifest
) -> str:
    lease_id = service.begin_retrieve(entry, 0)
    assert lease_id is not None
    return lease_id


def hold_restores(
    service: CheckpointPayloadStore, entries: list[CheckpointManifest]
) -> list[CheckpointSlots]:
    """Restore rank 0 of each entry and keep its pages pinned in RAM."""
    leases = []
    for entry in entries:
        lease = poll(service, begin_rank0_retrieve(service, entry))
        assert isinstance(lease, CheckpointSlots)
        leases.append(lease)
    return leases


@pytest.mark.parametrize("native", [False, True])
def test_restore_after_restart_waits_for_ram_held_by_other_restores(
    tmp_path: Path, native: bool
) -> None:
    """A restore that finds RAM full waits for room instead of missing.

    After a restart, resumed conversations load their checkpoints from L2
    into RAM, where they stay until evicted. While other restores hold that
    RAM, no page of this one can be loaded. Its pages are still in L2, so it
    must complete once eviction makes room, and its checkpoint must stay
    listed.
    """
    entry = large_manifest(500)
    others = [large_manifest(501 + i) for i in range(4)]
    publish_before_restart(tmp_path, native, [entry, *others])
    with open_store(tmp_path, native, prefetch_policy="retain") as (
        service,
        index,
        storage,
        mapping,
    ):
        held = hold_restores(service, others)
        used, total = storage.get_l1_usage()
        assert total - used < len(entry_keys(entry)) * 128 * 1024

        def finish_held() -> None:
            for other in held:
                service.finish_retrieve(other.lease_id)

        release = threading.Timer(0.3, finish_held)
        release.start()
        try:
            lease = poll(service, begin_rank0_retrieve(service, entry))
        finally:
            release.join()
        assert isinstance(lease, CheckpointSlots)
        try:
            for group_id, group in enumerate(lease.groups):
                for page_id, slot in enumerate(group):
                    assert slot is not None
                    assert mapping[slot.offset : slot.offset + slot.length] == (
                        bytes([group_id * 4 + page_id]) * slot.length
                    )
        finally:
            service.finish_retrieve(lease.lease_id)
        assert index.find((entry.prefix,)) == entry


def test_restore_that_never_gets_ram_keeps_its_checkpoint_listed(
    tmp_path: Path,
) -> None:
    """Running out of RAM does not make a stored checkpoint missing.

    A restore that gets no room within the admission timeout misses, but its
    pages are still in L2: the directory keeps listing the checkpoint, and a
    later restore succeeds.
    """
    entry = large_manifest(500)
    others = [large_manifest(501 + i) for i in range(4)]
    publish_before_restart(tmp_path, True, [entry, *others])
    with open_store(
        tmp_path, True, admission_timeout_seconds=0.3, prefetch_policy="retain"
    ) as (service, index, storage, mapping):
        held = hold_restores(service, others)
        try:
            assert poll(service, begin_rank0_retrieve(service, entry)) is False
        finally:
            for lease in held:
                service.finish_retrieve(lease.lease_id)
        assert index.find((entry.prefix,)) == entry
        storage.delete_l1_keys([key for other in others for key in entry_keys(other)])
        assert restores(service, mapping, entry)


def conversation_manifest(
    tail: tuple[int, ...], label: str, kind: str = "response"
) -> CheckpointManifest:
    """One turn of a conversation: shared attention pages plus unique state."""
    digest = lambda text: hashlib.sha256(text.encode()).hexdigest()
    payload = json.dumps(
        {
            "schema_version": 2,
            "kind": kind,
            "page_groups": [
                {
                    "name": "target.attention.0",
                    "page_bytes": 128,
                    "positions": [0, 1],
                    "content_keys": [digest("shared-0"), digest(f"tail-{label}")],
                },
                {
                    "name": "target.recurrent.0",
                    "page_bytes": 256,
                    "positions": [16],
                    "content_keys": [digest(f"recurrent-{label}")],
                },
            ],
        }
    ).encode()
    return CheckpointManifest(
        uuid.uuid4().hex,
        CheckpointPrefix("weights-and-layout-and-salt", 4096, b"a" * 32, tail),
        1,
        payload,
    )


def test_index_lists_ancestors_of_a_sequence_without_touching_lru() -> None:
    index = CheckpointIndex(None)
    try:
        turns = [
            conversation_manifest((1,), "a"),
            conversation_manifest((1, 2), "b"),
            conversation_manifest((1, 2, 3), "c"),
            conversation_manifest((1, 9), "branch"),
        ]
        for turn in turns:
            assert index.begin(turn)
            assert index.acknowledge(turn.generation, 0)
        assert index.get(turns[1].generation) == turns[1]
        assert index.get("missing") is None
        query = replace(turns[2].prefix, tail_tokens=(1, 2, 3, 4))
        assert index.ancestors((query,), 4096 + 3) == turns[:2]
        assert index.ancestors((query,), 4096 + 4) == turns[:3]
        with pytest.raises(ValueError, match="mix namespaces"):
            index.ancestors((query, replace(query, namespace="other")), 10**6)
    finally:
        index.close()


def test_supersede_marks_only_pages_older_turns_alone_reference() -> None:
    """A new prompt supersedes earlier requests' checkpoints, not its own."""
    name = f"lmcache_l1_pool_checkpoint_supersede_{uuid.uuid4().hex}"
    with open_store(shm_name=name) as (_, _, storage, mapping):
        module = CheckpointModule(
            cast(
                MPCacheServerContext,
                SimpleNamespace(
                    storage_manager=storage,
                    shm_pool_info={"shm_name": name, "pool_size": 4 * 1024 * 1024},
                ),
            )
        )
        try:
            instruction = conversation_manifest((1,), "i", kind="instruction")
            # Turn 1: its response ends in tokens the chat template rewrites,
            # so the next prompt extends the prompt, not the response.
            prompt_1 = conversation_manifest((1, 2), "p1", kind="prompt")
            response_1 = conversation_manifest((1, 2, 5), "r1")
            tail_2 = conversation_manifest((1, 2, 3), "t2", kind="prefill_tail")
            prompt_2 = conversation_manifest((1, 2, 3, 4), "p2", kind="prompt")
            response_2 = conversation_manifest((1, 2, 3, 4, 6), "r2")
            turns = (instruction, prompt_1, response_1, tail_2, prompt_2, response_2)
            for turn in turns:
                # Shared pages already resident come back without a slot.
                assert module._index.begin(turn)
                lease = module._payloads.prepare_store(turn, 0)
                assert isinstance(lease, CheckpointSlots)
                for group in lease.groups:
                    for slot in group:
                        if slot is not None:
                            mapping[slot.offset : slot.offset + slot.length] = (
                                b"\x01" * slot.length
                            )
                assert module._payloads.finish_store(lease.lease_id, True)

            def roots(entry: CheckpointManifest) -> tuple[CheckpointPrefix, ...]:
                return (entry.prefix,)

            retention = storage.checkpoint_retention
            requests = ("r0", "r1", "r1", "r2", "r2", "r2")
            # Only a prompt supersedes, and turn 1 has no earlier request.
            for turn, request in zip(turns[:4], requests[:4], strict=True):
                assert module.supersede(roots(turn), turn.generation, request) == 0
            assert retention.superseded_keys() == []

            marked = module.supersede(roots(prompt_2), prompt_2.generation, "r2")
            current = set(entry_keys(prompt_2))
            expected = (set(entry_keys(prompt_1)) | set(entry_keys(response_1))) - (
                current
            )
            assert marked == len(expected) > 0
            assert set(retention.superseded_keys()) == expected
            for kept in (instruction, tail_2, prompt_2):
                assert not any(retention.is_superseded(k) for k in entry_keys(kept))
            # The same request's response supersedes nothing.
            assert module.supersede(roots(response_2), response_2.generation, "r2") == 0
            assert module.supersede(roots(prompt_2), "unknown-generation", "r2") == 0
            with pytest.raises(ValueError, match="mixes namespaces"):
                module.supersede(
                    (replace(prompt_2.prefix, namespace="other"),),
                    prompt_2.generation,
                    "r2",
                )
        finally:
            module.close()


def test_a_branch_keeps_the_checkpoints_it_branched_before() -> None:
    """A retry or resumed turn supersedes the prompt it extends, but not the
    longer response the original line continues from (lmcache-supersession-repro)."""
    name = f"lmcache_l1_pool_checkpoint_branch_{uuid.uuid4().hex}"
    with open_store(shm_name=name) as (_, _, storage, mapping):
        module = CheckpointModule(
            cast(
                MPCacheServerContext,
                SimpleNamespace(
                    storage_manager=storage,
                    shm_pool_info={"shm_name": name, "pool_size": 4 * 1024 * 1024},
                ),
            )
        )
        try:
            prompt_a = conversation_manifest((1, 2), "pa", kind="prompt")
            response_a = conversation_manifest((1, 2, 5, 6, 7), "ra")
            # An aborted turn resumed with one other token: shorter than A's
            # response, so it branched before it.
            prompt_b = conversation_manifest((1, 2, 9), "pb", kind="prompt")
            # The original line continues past A's response.
            prompt_c = conversation_manifest((1, 2, 5, 6, 7, 8), "pc", kind="prompt")
            turns = (prompt_a, response_a, prompt_b, prompt_c)
            for turn in turns:
                assert module._index.begin(turn)
                lease = module._payloads.prepare_store(turn, 0)
                assert isinstance(lease, CheckpointSlots)
                for group in lease.groups:
                    for slot in group:
                        if slot is not None:
                            mapping[slot.offset : slot.offset + slot.length] = (
                                b"\x01" * slot.length
                            )
                assert module._payloads.finish_store(lease.lease_id, True)
            retention = storage.checkpoint_retention
            assert module.supersede((prompt_a.prefix,), prompt_a.generation, "a") == 0
            assert (
                module.supersede((response_a.prefix,), response_a.generation, "a") == 0
            )

            module.supersede((prompt_b.prefix,), prompt_b.generation, "b")
            unique_a = set(entry_keys(prompt_a)) - set(entry_keys(prompt_b))
            assert unique_a and set(retention.superseded_keys()) == unique_a
            assert not any(retention.is_superseded(k) for k in entry_keys(response_a))

            module.supersede((prompt_c.prefix,), prompt_c.generation, "c")
            current = set(entry_keys(prompt_c))
            assert set(retention.superseded_keys()) == (
                unique_a | (set(entry_keys(response_a)) - current)
            )
        finally:
            module.close()


@pytest.mark.parametrize(
    ("persistent", "explicit", "expected"),
    [(False, None, 8192), (True, None, 65536), (False, 123, 123)],
)
def test_checkpoint_directory_capacity_follows_persistence(
    tmp_path: Path, persistent: bool, explicit: int | None, expected: int
) -> None:
    """RAM-only manifests live in server memory; persistent ones cover L2."""
    name = f"lmcache_l1_pool_checkpoint_capacity_{uuid.uuid4().hex}"
    with open_store(shm_name=name) as (_, _, storage, _mapping):
        module = CheckpointModule(
            cast(
                MPCacheServerContext,
                SimpleNamespace(
                    storage_manager=storage,
                    shm_pool_info={"shm_name": name, "pool_size": 4 * 1024 * 1024},
                ),
            ),
            tmp_path / "index.sqlite3" if persistent else None,
            index_max_entries=explicit,
        )
        try:
            assert module._index._max_entries == expected
        finally:
            module.close()
