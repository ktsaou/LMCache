# SPDX-License-Identifier: Apache-2.0
"""Atomic external checkpoint ownership with the real vLLM allocator and LMCache RPC."""

# Standard
from concurrent.futures import Future
from dataclasses import replace
from multiprocessing import shared_memory
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch
import json
import os
import time

# Third Party
import pytest
import torch
import zmq

pytest.importorskip("vllm")

# Third Party
from vllm.distributed.kv_transfer.kv_connector.v1.base import (  # noqa: E402
    KVConnectorRole,
)
from vllm.lora.request import LoRARequest  # noqa: E402
from vllm.sampling_params import SamplingParams  # noqa: E402
from vllm.utils.hashing import sha256  # noqa: E402
from vllm.v1.core.boundary_checkpoint import BoundaryCheckpoint  # noqa: E402
from vllm.v1.core.kv_cache_manager import KVCacheManager  # noqa: E402
from vllm.v1.core.kv_cache_utils import (  # noqa: E402
    get_request_block_hasher,
    init_none_hash,
)
from vllm.v1.kv_cache_interface import (  # noqa: E402
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.request import Request  # noqa: E402

# First Party
from lmcache.integration.vllm import checkpoint_scheduler  # noqa: E402
from lmcache.integration.vllm.checkpoint_copy import CheckpointPageCopier  # noqa: E402
from lmcache.integration.vllm.checkpoint_scheduler import (  # noqa: E402
    CheckpointEngineTask,
    CheckpointSchedulerBridge,
)
from lmcache.integration.vllm.recurrent_checkpoint_connector import (  # noqa: E402
    LMCacheRecurrentCheckpointConnector,
    RecurrentCheckpointMetadata,
    RecurrentCheckpointWorkerMetadata,
)
from lmcache.v1.multiprocess.checkpoint_storage import (  # noqa: E402
    checkpoint_object_keys,
)
from lmcache.v1.multiprocess.checkpoint_transfer import (  # noqa: E402
    CheckpointTransferJob,
    CheckpointTransferWorker,
)
from lmcache.v1.multiprocess.protocols.base import RequestType  # noqa: E402
from lmcache.v1.multiprocess.protocols.checkpoint import (  # noqa: E402
    CheckpointCapabilities,
)
from lmcache.v1.multiprocess.transfer_context.shm import ShmPoolMapping  # noqa: E402
from tests.v1.multiprocess.test_checkpoint_storage import (  # noqa: E402
    make_manifest,
    open_checkpoint_rpc,
)

pytestmark = pytest.mark.skipif(
    not hasattr(KVCacheManager, "reserve_external_boundary_checkpoint"),
    reason="vLLM requires the atomic boundary import allocator API",
)


def test_connector_capability_rejects_mutable_revision_names() -> None:
    identity = {
        "target_revision": "a" * 40,
        "source_revision": "b" * 40,
        "draft_revision": "c" * 40,
    }
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(get_from_extra_config=lambda *_: identity),
        speculative_config=SimpleNamespace(method="dflash"),
    )
    assert LMCacheRecurrentCheckpointConnector.supports_request_boundary_checkpoints(
        config
    )
    for key in tuple(identity):
        original = identity[key]
        identity[key] = "main"
        supported = (
            LMCacheRecurrentCheckpointConnector.supports_request_boundary_checkpoints(
                config
            )
        )
        assert not supported
        identity[key] = original


def test_worker_acknowledgements_preserve_rank_identity() -> None:
    left = RecurrentCheckpointWorkerMetadata(
        {0: {"page_bytes": 128}}, {"copy": {0: True}}
    )
    right = RecurrentCheckpointWorkerMetadata(
        {1: {"page_bytes": 128}}, {"copy": {1: False}}
    )
    merged = left.aggregate(right)
    assert merged.results == {"copy": {0: True, 1: False}}
    assert set(merged.layouts) == {0, 1}
    with pytest.raises(ValueError, match="Duplicate"):
        merged.aggregate(left)


@pytest.mark.parametrize("role", list(KVConnectorRole))
def test_checkpoint_connector_does_not_require_aligned_chunk_rpc(
    role: KVConnectorRole,
) -> None:
    """Qwen's 3008-token pages need no divisibility with a 4096-token service.

    The real RPC fixture implements only checkpoint operations. Any request
    for chunk geometry or aligned KV registration therefore fails this test.
    """
    init_none_hash(sha256)
    with open_checkpoint_rpc() as (client, _module, _mapping, _name):
        extras = {
            "lmcache.mp.checkpoint_identity": {
                "target_revision": "a" * 40,
                "source_revision": "b" * 40,
            },
            "lmcache.mp.mp_transfer_mode": "engine_driven",
            "lmcache.mp.mq_timeout": 2,
            "lmcache.mp.server_urls": [
                client.socket.getsockopt_string(zmq.LAST_ENDPOINT)
            ],
        }
        config = SimpleNamespace(
            use_request_boundary_checkpoints=True,
            speculative_config=None,
            cache_config=SimpleNamespace(block_size=3008),
            scheduler_config=SimpleNamespace(max_num_batched_tokens=6019),
            kv_transfer_config=SimpleNamespace(
                get_from_extra_config=lambda key, default: extras.get(key, default)
            ),
        )
        connector = LMCacheRecurrentCheckpointConnector(config, role, None)
        try:
            connector.register_kv_caches({"attention": torch.empty(2, 3008, 1)})
            assert connector.get_num_new_matched_tokens(
                make_request("no-chunks"), 0
            ) == (
                0,
                False,
            )
        finally:
            connector.shutdown()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA DMA")
def test_checkpoint_raw_pages_roundtrip_without_token_chunk_registration() -> None:
    """Target, recurrent, draft and auxiliary bytes survive D2H and H2D exactly."""
    with open_checkpoint_rpc() as (client, module, _memory, _name):
        capability: CheckpointCapabilities = client.submit_request(
            RequestType.CHECKPOINT_CAPABILITIES, []
        ).result(timeout=5)
        mapping = ShmPoolMapping(capability.shm_name, capability.pool_size)
        pool = (
            torch.arange(16 * 128, device="cuda", dtype=torch.int32)
            .to(torch.uint8)
            .reshape(16, 128)
        )
        # Distinguish equal-width pages as well as bytes within a page.
        pool.add_(torch.arange(16, device="cuda", dtype=torch.uint8)[:, None])
        expected = pool.clone()
        layout = {"schema_version": 1, "page_bytes": 128}
        validated: list[dict[str, Any]] = []
        copier = CheckpointPageCopier(
            pool, layout, validated.append, mapping, capability
        )
        entry = make_manifest()
        payload = json.loads(entry.payload)
        for group in payload["page_groups"]:
            group["page_bytes"] = 128
        payload["worker_layout"] = layout
        entry = replace(entry, world_size=1, payload=json.dumps(payload).encode())
        ids = ((1, 2, 3), (4,), (5, 6), (7,))
        event = torch.cuda.Event()
        event.record()
        worker = CheckpointTransferWorker(client, copier)
        try:
            assert validated == [layout]
            assert module.begin(entry)
            stored = worker.submit(CheckpointTransferJob(entry, 0, "STORE", ids, event))
            assert stored is not None and stored.result(timeout=5)
            pool[1:8].zero_()
            event = torch.cuda.Event()
            event.record()
            restored = worker.submit(
                CheckpointTransferJob(entry, 0, "RETRIEVE", ids, event)
            )
            assert restored is not None and restored.result(timeout=5)
            assert torch.equal(pool, expected)
            status = module.report_status()["recurrent_checkpoints"]
            assert status["store_leases"] == status["retrieve_leases"] == 0
        finally:
            worker.close()
            mapping.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA DMA")
def test_replaced_server_pool_fails_the_lease_closed_and_is_remapped() -> None:
    """A restarted LMCache server recreates its pool under the same name.

    A worker that kept the old mapping would publish or restore bytes the
    server never wrote. The first lease after the replacement must fail
    without publishing, and the worker must map the new pool.
    """
    with open_checkpoint_rpc() as (client, module, _memory, _name):
        capability: CheckpointCapabilities = client.submit_request(
            RequestType.CHECKPOINT_CAPABILITIES, []
        ).result(timeout=5)
        mapping = ShmPoolMapping(capability.shm_name, capability.pool_size)
        pool = torch.zeros(16, 128, device="cuda", dtype=torch.uint8)
        layout = {"schema_version": 1, "page_bytes": 128}
        copier = CheckpointPageCopier(pool, layout, lambda _: None, mapping, capability)
        entry = make_manifest()
        payload = json.loads(entry.payload)
        for group in payload["page_groups"]:
            group["page_bytes"] = 128
        payload["worker_layout"] = layout
        entry = replace(entry, world_size=1, payload=json.dumps(payload).encode())
        path = f"/dev/shm/{capability.shm_name.lstrip('/')}"
        os.rename(path, path + ".old")
        replacement = shared_memory.SharedMemory(
            name=capability.shm_name.lstrip("/"), create=True, size=capability.pool_size
        )
        worker = CheckpointTransferWorker(client, copier)
        try:
            assert not mapping.is_current()
            assert module.begin(entry)
            event = torch.cuda.Event()
            event.record()
            stored = worker.submit(
                CheckpointTransferJob(
                    entry, 0, "STORE", ((1, 2, 3), (4,), (5, 6), (7,)), event
                )
            )
            assert stored is not None
            with pytest.raises(ValueError, match="replaced"):
                stored.result(timeout=5)
            assert module.find((entry.prefix,)) is None
            status = module.report_status()["recurrent_checkpoints"]
            assert status["store_leases"] == 0
            assert mapping.is_current()
        finally:
            worker.close()
            mapping.close()
            replacement.close()
            replacement.unlink()
            os.rename(path + ".old", path)


def test_lora_requests_do_not_read_or_publish_the_base_weight_namespace() -> None:
    """Adapter IDs do not authenticate immutable adapter bytes across restarts."""
    with open_checkpoint_rpc() as (client, module, _mapping, _name):
        manager = make_manager()
        bridge = CheckpointSchedulerBridge(manager, client, {}, 4)
        layout = {"schema_version": 1, "page_bytes": 128}
        bridge.accept_layouts({rank: layout for rank in range(4)})
        request = make_request("adapter-request")
        request.lora_request = LoRARequest("adapter", 1, "/adapter")
        checkpoint = manager.reserve_external_boundary_checkpoint(
            request,
            11,
            manager.boundary_checkpoint_page_positions(11),
            draft_prefix_len=11,
            kind="prompt",
            num_ranks=4,
        )
        assert checkpoint is not None
        for rank in range(4):
            manager.acknowledge_external_boundary_checkpoint(
                checkpoint.checkpoint_id, rank
            )
        assert not bridge.handles(request)
        assert bridge.poll_prefix(request)
        bridge.store(request, checkpoint)
        assert bridge.take_tasks() == []
        assert manager.reset_prefix_cache()
        assert (
            module.report_status()["recurrent_checkpoints"]["pending_generations"] == 0
        )


def make_request(name: str, *, first_token: int = 0) -> Request:
    """Construct an exact eleven-token prompt with ordinary cache authentication."""
    params = SamplingParams(max_tokens=1)
    params.update_from_generation_config({}, eos_token_id=100)
    return Request(
        request_id=name,
        prompt_token_ids=list(range(first_token, first_token + 11)),
        sampling_params=params,
        pooling_params=None,
        block_hasher=get_request_block_hasher(4, sha256),
    )


def make_manager() -> KVCacheManager:
    """Mix full attention and an endpoint-only recurrent group in one block pool."""
    init_none_hash(sha256)
    config = KVCacheConfig(
        num_blocks=64,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["target.attention"],
                FullAttentionSpec(
                    block_size=4,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["target.recurrent"],
                MambaSpec(
                    block_size=4,
                    shapes=((1, 1),),
                    dtypes=(torch.float32,),
                ),
            ),
        ],
    )
    return KVCacheManager(
        config,
        max_model_len=128,
        hash_block_size=4,
        scheduler_block_size=4,
        enable_boundary_checkpoints=True,
    )


def test_failed_begin_releases_source_pin_after_request_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An RPC exception before dispatch owns no worker copy or source pin."""
    with open_checkpoint_rpc() as (client, _module, _mapping, _name):
        manager = make_manager()
        bridge = CheckpointSchedulerBridge(
            manager,
            client,
            {
                "target_revision": "a" * 40,
                "draft_revision": "",
                "source_revision": "b" * 40,
                "parallel": {"tp": 4, "dcp": 1},
            },
            4,
        )
        bridge.accept_layouts(
            {rank: {"schema_version": 1, "page_bytes": 128} for rank in range(4)}
        )
        request = make_request("failed-begin")
        checkpoint = manager.reserve_external_boundary_checkpoint(
            request,
            11,
            manager.boundary_checkpoint_page_positions(11),
            draft_prefix_len=11,
            kind="prompt",
            num_ranks=4,
        )
        assert checkpoint is not None
        for rank in range(4):
            manager.acknowledge_external_boundary_checkpoint(
                checkpoint.checkpoint_id, rank
            )
        failed: Future[bool] = Future()
        failed.set_exception(RuntimeError("checkpoint begin transport failure"))
        with monkeypatch.context() as patch:
            patch.setattr(
                client,
                "submit_request",
                lambda *_: SimpleNamespace(query=lambda: True, result=failed.result),
            )
            bridge.store(request, checkpoint)
        bridge.finish_request(request.request_id)
        assert not manager.reset_prefix_cache()
        assert bridge.take_tasks() == []
        assert not bridge.has_pending
        assert manager.reset_prefix_cache()


@pytest.mark.parametrize("failure_stage", ["submit", "event_create", "event_record"])
def test_submission_exception_reports_each_unsent_task(
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    """A rejected submission cannot strand the other ranks' source leases."""
    # First Party
    from lmcache.integration.vllm import recurrent_checkpoint_connector as connector
    from tests.v1.multiprocess.test_checkpoint_storage import make_manifest

    tasks = [
        CheckpointEngineTask(str(i), make_manifest(), "STORE", ((0,),))
        for i in range(3)
    ]
    submitted: list[str] = []
    completed: Future[bool] = Future()
    completed.set_result(True)

    def submit(job: CheckpointTransferJob) -> Future[bool]:
        submitted.append(job.manifest.generation)
        if failure_stage == "submit" and len(submitted) == 2:
            raise RuntimeError("executor rejected checkpoint submission")
        return completed

    event_calls = 0

    def make_event() -> SimpleNamespace:
        nonlocal event_calls
        event_calls += 1
        second_task = event_calls == 2
        if second_task and failure_stage == "event_create":
            raise RuntimeError("checkpoint event creation failed")

        def record() -> None:
            if second_task and failure_stage == "event_record":
                raise RuntimeError("checkpoint event recording failed")

        return SimpleNamespace(record=record)

    # Isolate the connector's worker protocol; real allocator/RPC ownership is
    # covered by the collective roundtrip test in this module.
    worker = cast(
        LMCacheRecurrentCheckpointConnector,
        SimpleNamespace(
            _connector_metadata=RecurrentCheckpointMetadata(tasks),
            _worker=SimpleNamespace(submit=submit),
            _rank=0,
            _pending={},
            _rejected=set(),
            _layout_sent=False,
            _worker_layout={"schema_version": 1, "page_bytes": 128},
        ),
    )
    monkeypatch.setattr(
        connector.torch_dev,
        "Event",
        make_event,
    )
    LMCacheRecurrentCheckpointConnector.start_load_kv(worker, None)
    result = LMCacheRecurrentCheckpointConnector.build_connector_worker_meta(worker)
    assert set(result.results) == {task.task_id for task in tasks}
    assert result.results["0"] == {0: True}
    assert result.results["1"] == {0: False}
    assert result.results["2"] == {0: True}
    assert not LMCacheRecurrentCheckpointConnector.build_connector_worker_meta(
        worker
    ).results


def test_worker_metadata_requires_bound_rank() -> None:
    """Unbound worker state must fail before emitting an invalid rank ID."""
    worker = cast(
        LMCacheRecurrentCheckpointConnector,
        SimpleNamespace(
            _rank=None,
            _pending={},
            _rejected=set(),
            _layout_sent=False,
            _worker_layout=None,
        ),
    )
    with pytest.raises(RuntimeError, match="bound"):
        LMCacheRecurrentCheckpointConnector.build_connector_worker_meta(worker)


@pytest.mark.parametrize(
    "outcome",
    [
        "success",
        "rank-miss",
        "cancelled",
        "rejected-store-reused-request-id",
        "inflight-store-reused-request-id",
        "capacity-retry",
        "capacity-timeout",
        "task-capacity",
        "local-hit-during-copy",
        "invalidated-before-use",
        "publication-invalidated",
    ],
)
def test_semantic_roundtrip_collective_visibility_and_cancellation(
    outcome: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with open_checkpoint_rpc() as (client, module, mapping, _name):
        manager = make_manager()
        bridge = CheckpointSchedulerBridge(
            manager,
            client,
            {
                "target_revision": "target-content",
                "draft_revision": "",
                "source_revision": "source-content",
                "parallel": {"tp": 4, "dcp": 1},
            },
            4,
            max_tasks=1 if outcome == "task-capacity" else 32,
            lookup_timeout=0.05 if outcome == "capacity-timeout" else 60.0,
        )
        layout = {"schema_version": 1, "page_bytes": 128}
        bridge.accept_layouts({rank: layout for rank in range(4)})
        producer = make_request("producer")
        checkpoint = manager.reserve_external_boundary_checkpoint(
            producer,
            11,
            manager.boundary_checkpoint_page_positions(11),
            draft_prefix_len=11,
            kind="prompt",
            num_ranks=4,
        )
        assert checkpoint is not None
        for rank in range(4):
            manager.acknowledge_external_boundary_checkpoint(
                checkpoint.checkpoint_id, rank
            )
        if outcome == "rejected-store-reused-request-id":
            # A producer can disconnect before the server rejects its store.
            # Reusing its request ID must not cancel an independent import.
            rejected = make_request("consumer")
            with monkeypatch.context() as patch:
                patch.setattr(
                    client,
                    "submit_request",
                    lambda *_: SimpleNamespace(
                        query=lambda: True, result=lambda: False
                    ),
                )
                bridge.store(rejected, checkpoint)
            bridge.finish_request(rejected.request_id)
            assert bridge.take_tasks() == []
            assert not bridge.has_pending
        bridge.store(producer, checkpoint)
        bridge.finish_request(producer.request_id)
        assert not manager.reset_prefix_cache()
        deadline = time.monotonic() + 5
        tasks: list[CheckpointEngineTask] = []
        while not tasks and time.monotonic() < deadline:
            tasks = bridge.take_tasks()
            time.sleep(0.001)
        assert len(tasks) == 1 and not bridge.take_tasks()
        store_task = tasks[0]

        def copy_pages(job, lease) -> None:
            for group_id, group in enumerate(lease.slots):
                for page_id, (offset, size) in enumerate(group):
                    pattern = bytes([job.rank * 16 + group_id * 4 + page_id]) * size
                    if job.direction == "STORE":
                        mapping[offset : offset + size] = pattern
                    else:
                        assert mapping[offset : offset + size] == pattern

        worker = CheckpointTransferWorker(client, copy_pages)
        supersessions: list[list[Any]] = []
        submit_request = client.submit_request

        def record(kind: RequestType, args: list[Any]) -> Any:
            if kind == RequestType.CHECKPOINT_SUPERSEDE:
                supersessions.append(args)
            return submit_request(kind, args)

        try:
            for rank in range(4):
                future = worker.submit(
                    CheckpointTransferJob(
                        store_task.manifest,
                        rank,
                        "STORE",
                        store_task.block_ids,
                    )
                )
                assert future is not None and future.result(timeout=5)
                with monkeypatch.context() as patch:
                    patch.setattr(client, "submit_request", record)
                    bridge.complete({store_task.task_id: {rank: True}})
                assert manager.reset_prefix_cache() == (rank == 3)
            assert not bridge.has_pending
            # Publication reports the producing sequence so older checkpoints
            # of it are superseded; this first checkpoint has none.
            assert len(supersessions) == 1
            roots, generation, request_id = supersessions[0]
            assert generation == store_task.manifest.generation
            assert request_id == producer.request_id
            prefix = store_task.manifest.prefix
            assert roots and all(root.namespace == prefix.namespace for root in roots)
            assert sum(len(root.tail_tokens) for root in roots) >= prefix.num_tokens
            consumer = make_request("consumer")
            if outcome in ("capacity-retry", "capacity-timeout"):
                # Resource pressure must not authorize cold admission.
                pressure = manager.block_pool.get_new_blocks(
                    manager.block_pool.get_num_free_blocks()
                )
                try:
                    for _ in range(100):
                        assert not bridge.poll_prefix(consumer)
                        time.sleep(0.001)
                    assert bridge.take_tasks() == []
                    assert not bridge.poll_prefix(consumer)
                finally:
                    manager.block_pool.free_blocks(pressure)
            inflight_store = None
            if outcome in ("inflight-store-reused-request-id", "task-capacity"):
                # This cancelled producer has different tokens but the same
                # public ID as the waiting consumer. Its admitted copy is still
                # live while the consumer looks up the first producer's data.
                predecessor = make_request(
                    "blocker" if outcome == "task-capacity" else "consumer",
                    first_token=20,
                )
                predecessor_checkpoint = manager.reserve_external_boundary_checkpoint(
                    predecessor,
                    11,
                    manager.boundary_checkpoint_page_positions(11),
                    draft_prefix_len=11,
                    kind="prompt",
                    num_ranks=4,
                )
                assert predecessor_checkpoint is not None
                for rank in range(4):
                    manager.acknowledge_external_boundary_checkpoint(
                        predecessor_checkpoint.checkpoint_id, rank
                    )
                bridge.store(predecessor, predecessor_checkpoint)
                bridge.finish_request(predecessor.request_id)
                deadline = time.monotonic() + 5
                pending: list[CheckpointEngineTask] = []
                while not pending and time.monotonic() < deadline:
                    pending = bridge.take_tasks()
                    time.sleep(0.001)
                assert len(pending) == 1
                inflight_store = pending[0]

            def drain_predecessor() -> None:
                nonlocal inflight_store
                assert inflight_store is not None
                for rank in range(4):
                    future = worker.submit(
                        CheckpointTransferJob(
                            inflight_store.manifest,
                            rank,
                            "STORE",
                            inflight_store.block_ids,
                        )
                    )
                    assert future is not None and future.result(timeout=5)
                    bridge.complete({inflight_store.task_id: {rank: True}})
                inflight_store = None

            before = manager.block_pool.get_num_free_blocks()
            deadline = time.monotonic() + 5
            drain_after = time.monotonic() + 0.1
            tasks = []
            while not tasks and time.monotonic() < deadline:
                assert not bridge.poll_prefix(consumer)
                tasks = bridge.take_tasks()
                if (
                    not tasks
                    and inflight_store is not None
                    and time.monotonic() >= drain_after
                ):
                    # Deferring reuse until the predecessor drains is valid.
                    # Admitting it sooner must not inherit cancellation or lose
                    # its lookup when the predecessor completes.
                    drain_predecessor()
                    before = manager.block_pool.get_num_free_blocks()
                time.sleep(0.001)
            assert len(tasks) == 1
            restore_task = tasks[0]
            assert manager.get_computed_blocks(consumer)[1] == 0
            if outcome == "local-hit-during-copy":
                local = make_request("local-producer")
                local_checkpoint = manager.reserve_external_boundary_checkpoint(
                    local,
                    11,
                    manager.boundary_checkpoint_page_positions(11),
                    draft_prefix_len=11,
                    kind="prompt",
                    num_ranks=1,
                )
                assert local_checkpoint is not None
                assert manager.acknowledge_external_boundary_checkpoint(
                    local_checkpoint.checkpoint_id, 0
                )
                assert not bridge.poll_prefix(consumer)
            if outcome == "cancelled":
                bridge.finish_request(consumer.request_id)
            if outcome == "publication-invalidated":
                assert manager.boundary_checkpoints is not None
                manager.boundary_checkpoints.invalidate_block(
                    restore_task.block_ids[-1][0]
                )
            for rank in range(4):
                future = worker.submit(
                    CheckpointTransferJob(
                        restore_task.manifest,
                        rank,
                        "RETRIEVE",
                        restore_task.block_ids,
                    )
                )
                assert future is not None and future.result(timeout=5)
                bridge.complete(
                    {
                        restore_task.task_id: {
                            rank: not (outcome == "rank-miss" and rank == 2)
                        }
                    }
                )
                if rank < 3:
                    if outcome != "local-hit-during-copy":
                        assert manager.get_computed_blocks(consumer)[1] == 0
                    assert manager.block_pool.get_num_free_blocks() < before
            expected_tokens = (
                0
                if outcome in ("rank-miss", "cancelled", "publication-invalidated")
                else 11
            )
            if expected_tokens:
                assert manager.block_pool.get_num_free_blocks() < before
                assert not manager.reset_prefix_cache()
            else:
                assert manager.block_pool.get_num_free_blocks() == before
            assert manager.get_computed_blocks(consumer)[1] == expected_tokens
            assert bridge.external_tokens(consumer) == expected_tokens
            if outcome == "invalidated-before-use":
                # Preemption releases request ownership while its connector
                # lookup state can still remember the previous successful import.
                checkpoint = consumer.boundary_checkpoint
                assert checkpoint is not None
                manager.free(consumer)
                assert manager.boundary_checkpoints is not None
                manager.boundary_checkpoints.invalidate(checkpoint.checkpoint_id)
                assert not bridge.poll_prefix(consumer)
            bridge.finish_request(consumer.request_id)
            assert manager.block_pool.get_num_free_blocks() == before
            if inflight_store is not None:
                drain_predecessor()
            assert not bridge.has_pending
            assert (
                module.report_status()["recurrent_checkpoints"]["retrieve_leases"] == 0
            )
        finally:
            worker.close()


def test_failed_restore_falls_back_to_the_longest_remaining_checkpoint() -> None:
    """A restore whose pages are gone retries the directory, not the prompt."""
    with open_checkpoint_rpc() as (client, module, mapping, _name):
        manager = make_manager()
        bridge = CheckpointSchedulerBridge(
            manager,
            client,
            {
                "target_revision": "target-content",
                "draft_revision": "",
                "source_revision": "source-content",
                "parallel": {"tp": 4, "dcp": 1},
            },
            4,
        )
        bridge.accept_layouts(
            {rank: {"schema_version": 1, "page_bytes": 128} for rank in range(4)}
        )

        def copy_pages(job, lease) -> None:
            for group_id, group in enumerate(lease.slots):
                for page_id, (offset, size) in enumerate(group):
                    pattern = bytes([job.rank * 16 + group_id * 4 + page_id]) * size
                    if job.direction == "STORE":
                        mapping[offset : offset + size] = pattern
                    else:
                        assert mapping[offset : offset + size] == pattern

        def run(task: CheckpointEngineTask) -> dict[int, bool]:
            results = {}
            for rank in range(4):
                future = worker.submit(
                    CheckpointTransferJob(
                        task.manifest, rank, task.direction, task.block_ids
                    )
                )
                results[rank] = future is not None and future.result(timeout=5)
            bridge.complete({task.task_id: results})
            return results

        worker = CheckpointTransferWorker(client, copy_pages)
        try:
            producer = make_request("producer")
            manifests = {}
            for length, kind in ((8, "instruction"), (11, "prompt")):
                checkpoint = manager.reserve_external_boundary_checkpoint(
                    producer,
                    length,
                    manager.boundary_checkpoint_page_positions(length),
                    draft_prefix_len=length,
                    kind=kind,
                    num_ranks=4,
                )
                assert checkpoint is not None
                for rank in range(4):
                    manager.acknowledge_external_boundary_checkpoint(
                        checkpoint.checkpoint_id, rank
                    )
                bridge.store(producer, checkpoint)
                deadline = time.monotonic() + 5
                tasks: list[CheckpointEngineTask] = []
                while not tasks and time.monotonic() < deadline:
                    tasks = bridge.take_tasks()
                    time.sleep(0.001)
                assert len(tasks) == 1
                assert all(run(tasks[0]).values())
                manifests[length] = tasks[0].manifest
            bridge.finish_request(producer.request_id)
            assert manager.reset_prefix_cache()

            # Pages only the longest checkpoint owns leave L1 without an L2
            # copy; content-addressed pages shared with the shorter one stay.
            storage = module.context.storage_manager
            shared = {
                key
                for rank in range(4)
                for group in checkpoint_object_keys(manifests[8], rank)
                for key in group
            }
            for rank in range(4):
                keys = [
                    key
                    for group in checkpoint_object_keys(manifests[11], rank)
                    for key in group
                    if key not in shared
                ]
                assert keys
                assert storage.delete_l1_keys(keys) == (len(keys), 0)

            consumer = make_request("consumer")
            attempts: list[tuple[int, bool]] = []
            deadline = time.monotonic() + 10
            with patch.object(checkpoint_scheduler.logger, "info") as info:
                while not bridge.poll_prefix(consumer):
                    assert time.monotonic() < deadline
                    for task in bridge.take_tasks():
                        results = run(task)
                        attempts.append(
                            (task.manifest.prefix.num_tokens, all(results.values()))
                        )
                    time.sleep(0.001)

            assert attempts == [(11, False), (8, True)]
            failure = next(
                call.args for call in info.call_args_list if "failed" in call.args[0]
            )
            assert failure[1:4] == (11, consumer.request_id, [0, 1, 2, 3])
            assert 0 <= failure[4] < 10
            assert manager.get_computed_blocks(consumer)[1] == 8
            assert bridge.external_tokens(consumer) == 8
            assert not bridge.has_pending
        finally:
            worker.close()


def make_bridge(client, manager, lookup_timeout: float) -> CheckpointSchedulerBridge:
    """Bridge with negotiated layouts and a short directory reply deadline."""
    bridge = CheckpointSchedulerBridge(
        manager,
        client,
        {
            "target_revision": "a" * 40,
            "draft_revision": "",
            "source_revision": "b" * 40,
            "parallel": {"tp": 4, "dcp": 1},
        },
        4,
        lookup_timeout=lookup_timeout,
    )
    bridge.accept_layouts(
        {rank: {"schema_version": 1, "page_bytes": 128} for rank in range(4)}
    )
    return bridge


@pytest.mark.parametrize("prefix, already_local", [(11, False), (8, True)])
def test_local_reuse_retries_external_restore_after_capacity_eviction(
    prefix: int, already_local: bool
) -> None:
    """Losing a local selection must not turn an external hit into cold prefill."""
    manager = make_manager()
    manifest = None
    lookups = 0

    def submit(kind: RequestType, args: list[Any]) -> SimpleNamespace:
        nonlocal manifest, lookups
        if kind == RequestType.CHECKPOINT_BEGIN:
            manifest = args[0]
        if kind == RequestType.CHECKPOINT_FIND:
            lookups += 1
        result = manifest if kind == RequestType.CHECKPOINT_FIND else True
        return SimpleNamespace(query=lambda: True, result=lambda: result)

    bridge = make_bridge(SimpleNamespace(submit_request=submit), manager, 60)
    producer = make_request("producer")

    def publish() -> BoundaryCheckpoint:
        checkpoint = manager.reserve_external_boundary_checkpoint(
            producer,
            prefix,
            manager.boundary_checkpoint_page_positions(prefix),
            draft_prefix_len=prefix,
            kind="prompt",
            num_ranks=1,
        )
        assert checkpoint is not None
        assert manager.acknowledge_external_boundary_checkpoint(
            checkpoint.checkpoint_id, 0
        )
        return checkpoint

    checkpoint = publish()
    bridge.store(producer, checkpoint)
    (store,) = bridge.take_tasks()
    bridge.complete({store.task_id: {rank: True for rank in range(4)}})
    bridge.finish_request(producer.request_id)
    assert manager.reset_prefix_cache()
    consumer = make_request("consumer")
    if already_local:
        checkpoint = publish()
    assert not bridge.poll_prefix(consumer)
    if not already_local:
        checkpoint = publish()
    # Other requests occupy everything except the unpinned cached bundle.
    pressure = manager.block_pool.get_new_blocks(
        manager.block_pool.get_num_free_blocks() - len(checkpoint.dependencies)
    )
    assert bridge.poll_prefix(consumer)
    blocks, tokens, _ = manager.get_computed_blocks(consumer)
    assert tokens == prefix
    assert bridge.external_tokens(consumer) == 0
    assert bridge.poll_prefix(consumer)
    assert lookups == 1
    assert bridge.take_tasks() == []
    assert (
        manager.allocate_slots(
            consumer,
            1,
            num_new_computed_tokens=tokens,
            new_computed_blocks=blocks,
            full_sequence_must_fit=True,
        )
        is None
    )
    # A running request grows into the cached bundle before capacity returns.
    growth = manager.block_pool.get_new_blocks(1)
    manager.block_pool.free_blocks(pressure + growth)
    assert manager.get_computed_blocks(consumer)[1] == 0
    assert not bridge.poll_prefix(consumer)
    assert lookups == 2
    assert not bridge.poll_prefix(consumer)
    (restore,) = bridge.take_tasks()
    bridge.complete({restore.task_id: {rank: True for rank in range(4)}})
    assert bridge.poll_prefix(consumer)
    blocks, tokens, _ = manager.get_computed_blocks(consumer)
    assert tokens == prefix
    assert bridge.external_tokens(consumer) == prefix
    assert (
        manager.allocate_slots(
            consumer,
            1,
            num_new_computed_tokens=tokens,
            new_computed_blocks=blocks,
            full_sequence_must_fit=True,
        )
        is not None
    )
    _, copies = manager.take_kv_cache_block_copies()
    manager.block_pool.free_blocks(copies)
    bridge.finish_request(consumer.request_id)
    manager.free(consumer)
    assert not bridge.has_pending
    assert manager.external_boundary_reserved_blocks() == 0
    assert manager.block_pool.get_num_free_blocks() == 63


@pytest.mark.parametrize("reply", ["never", "error", "miss"])
def test_lookup_without_a_usable_reply_admits_the_request(
    reply: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A lost or failed directory reply ends in recomputation, not a park."""
    with open_checkpoint_rpc() as (client, _module, _mapping, _name):
        manager = make_manager()
        bridge = make_bridge(client, manager, lookup_timeout=0.2)
        failed: Future[None] = Future()
        failed.set_exception(RuntimeError("checkpoint find handler failure"))
        future = (
            SimpleNamespace(query=lambda: False, result=failed.result)
            if reply == "never"
            else SimpleNamespace(
                query=lambda: True,
                result=(lambda: None) if reply == "miss" else failed.result,
            )
        )
        consumer = make_request("consumer")
        with monkeypatch.context() as patch:
            patch.setattr(client, "submit_request", lambda *_: future)
            assert not bridge.poll_prefix(consumer)
            if reply == "never":
                assert not bridge.poll_prefix(consumer)
                time.sleep(0.3)
            assert bridge.poll_prefix(consumer)
            assert bridge.poll_prefix(consumer)
        assert bridge.take_tasks() == []
        assert bridge.external_tokens(consumer) == 0
        assert not bridge.has_pending


def test_unanswered_store_begin_is_aborted_after_the_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A begin reply that never arrives cannot pin the source checkpoint."""
    with open_checkpoint_rpc() as (client, _module, _mapping, _name):
        manager = make_manager()
        bridge = make_bridge(client, manager, lookup_timeout=0.2)
        producer = make_request("producer")
        checkpoint = manager.reserve_external_boundary_checkpoint(
            producer,
            11,
            manager.boundary_checkpoint_page_positions(11),
            draft_prefix_len=11,
            kind="prompt",
            num_ranks=4,
        )
        assert checkpoint is not None
        for rank in range(4):
            manager.acknowledge_external_boundary_checkpoint(
                checkpoint.checkpoint_id, rank
            )
        submitted: list[RequestType] = []

        def submit(kind: RequestType, *_args) -> SimpleNamespace:
            submitted.append(kind)
            return SimpleNamespace(query=lambda: False, result=lambda: None)

        with monkeypatch.context() as patch:
            patch.setattr(client, "submit_request", submit)
            bridge.store(producer, checkpoint)
            bridge.finish_request(producer.request_id)
            assert bridge.take_tasks() == []
            assert bridge.has_pending
            assert not manager.reset_prefix_cache()
            time.sleep(0.3)
            assert bridge.take_tasks() == []
        assert submitted == [RequestType.CHECKPOINT_BEGIN, RequestType.CHECKPOINT_ABORT]
        assert not bridge.has_pending
        assert manager.reset_prefix_cache()
