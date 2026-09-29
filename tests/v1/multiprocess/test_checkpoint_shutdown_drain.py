# SPDX-License-Identifier: Apache-2.0
"""A stopping server finishes checkpoint stores in flight, then writes them.

When a container stops, the engine and the cache server get SIGTERM at the
same time. The engine may still be copying its last checkpoints; the server
keeps serving those stores for a bounded time before its message queue
closes, and its shutdown flush then writes them to L2.
"""

# Standard
from collections.abc import Iterator
from contextlib import contextmanager
from mmap import mmap
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, call, patch
import asyncio
import threading
import time
import uuid

# First Party
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.distributed.l2_adapters.config import L2AdaptersConfig
from lmcache.v1.distributed.l2_adapters.fs_native_l2_adapter import (
    FSNativeL2AdapterConfig,
)
from lmcache.v1.distributed.storage_manager import StorageManager
from lmcache.v1.multiprocess import http_server
from lmcache.v1.multiprocess.checkpoint_index import CheckpointManifest
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
from lmcache.v1.multiprocess.modules.checkpoint import CheckpointModule
from lmcache.v1.multiprocess.posix_shm import shm_open_pool_as_mmap
from lmcache.v1.multiprocess.server import MPCacheServer
from tests.v1.multiprocess.test_checkpoint_side_requests import (
    POOL_BYTES,
    checkpoint,
    content,
    page_keys,
    prefix,
    publish,
    restores,
)


@contextmanager
def open_server(
    path: Path, *, budget_seconds: float = 30.0
) -> Iterator[tuple[MPCacheServer, CheckpointModule, mmap]]:
    """A cache server with a checkpoint module over RAM and native-FS L2.

    The caller shuts it down with ``drain_for_shutdown`` and ``close``.
    """
    name = f"lmcache_l1_pool_checkpoint_drain_{uuid.uuid4().hex}"
    storage = StorageManager(
        StorageManagerConfig(
            L1ManagerConfig(L1MemoryManagerConfig(POOL_BYTES, False, shm_name=name)),
            EvictionConfig(eviction_policy="LRU"),
            l2_adapter_config=L2AdaptersConfig(
                [FSNativeL2AdapterConfig(str(path / "payloads"))]
            ),
            store_policy="checkpoint_on_evict",
            checkpoint_shutdown_flush_seconds=budget_seconds,
        )
    )
    context = SimpleNamespace(
        storage_manager=storage,
        shm_pool_info={"shm_name": name, "pool_size": POOL_BYTES},
        close=storage.close,
    )
    module = CheckpointModule(
        cast(MPCacheServerContext, context), path / "directory.sqlite3"
    )
    server = MPCacheServer(cast(MPCacheServerContext, context), [module])
    with shm_open_pool_as_mmap(name, POOL_BYTES) as mapping:
        yield server, module, mapping


def copy_pages(
    module: CheckpointModule, mapping: mmap, entry: CheckpointManifest
) -> str:
    """Admit a store and copy its pages, as a worker does before finishing."""
    assert module.begin(entry)
    lease = module.prepare_store(entry, 0)
    assert lease.status == "ready"
    slots = [slot for group in lease.slots for slot in group]
    for key, (offset, size) in zip(page_keys(entry), slots, strict=True):
        if offset >= 0:
            mapping[offset : offset + size] = content(key)
    return lease.lease_id


def restores_after_restart(path: Path, entry: CheckpointManifest) -> bool:
    with open_server(path) as (server, module, mapping):
        try:
            return module.find((entry.prefix,)) == entry and restores(
                module, mapping, entry
            )
        finally:
            server.close()


TOKENS = tuple(range(200, 213))


def test_shutdown_serves_a_store_in_flight_and_writes_it(tmp_path: Path) -> None:
    """The engine finishes its copy 0.4 s into the shutdown; the drain keeps
    the message queue's handlers serving until then, the checkpoint is
    published, and the flush writes it for the next start. Before, the
    server closed at once, the lease never finished, the checkpoint module
    refused to close and the storage manager never flushed."""
    entry = checkpoint(TOKENS, "response")
    with open_server(tmp_path) as (server, module, mapping):
        lease_id = copy_pages(module, mapping, entry)
        finisher = threading.Timer(0.4, module.finish_store, (lease_id, True))
        finisher.start()
        started = time.monotonic()
        server.drain_for_shutdown()
        elapsed = time.monotonic() - started
        finisher.join()
        assert 0.4 <= elapsed < 3.0
        assert module.find((entry.prefix,)) == entry
        server.close()
    assert restores_after_restart(tmp_path, entry)


def test_shutdown_drain_is_bounded_when_a_worker_never_finishes(
    tmp_path: Path,
) -> None:
    """A worker killed mid-copy never finishes its lease. The drain gives up
    within a third of the budget, closing still flushes what was published,
    and the unfinished checkpoint is never listed."""
    published = checkpoint(TOKENS, "response")
    stuck = checkpoint(TOKENS + (1, 2, 3), "prompt")
    with open_server(tmp_path, budget_seconds=1.5) as (server, module, mapping):
        publish(module, mapping, published, "a")
        copy_pages(module, mapping, stuck)
        started = time.monotonic()
        server.drain_for_shutdown()
        assert time.monotonic() - started < 1.0
        server.close()
    assert restores_after_restart(tmp_path, published)
    with open_server(tmp_path) as (server, module, _mapping):
        try:
            assert module.find((prefix(stuck.prefix.tail_tokens),)) == published
        finally:
            server.close()


def test_shutdown_drain_returns_at_once_when_idle(tmp_path: Path) -> None:
    with open_server(tmp_path) as (server, _module, _mapping):
        started = time.monotonic()
        server.drain_for_shutdown()
        assert time.monotonic() - started < 0.3
        server.close()


def test_shutdown_drain_is_bounded_while_stores_keep_arriving(
    tmp_path: Path,
) -> None:
    """An engine that keeps storing cannot hold the shutdown: the drain ends
    after a third of a 3 s budget and the flush keeps the rest."""
    stop = threading.Event()
    with open_server(tmp_path, budget_seconds=3.0) as (server, module, mapping):

        def keep_storing() -> None:
            # The same prefix again and again: every store is admitted at
            # once, since its pages are already in RAM.
            turn = 0
            while not stop.is_set():
                turn += 1
                publish(module, mapping, checkpoint(TOKENS[:4], "prompt"), f"r{turn}")
                time.sleep(0.05)

        producer = threading.Thread(target=keep_storing)
        producer.start()
        try:
            time.sleep(0.2)
            started = time.monotonic()
            server.drain_for_shutdown()
            elapsed = time.monotonic() - started
        finally:
            stop.set()
            producer.join()
        assert 0.8 <= elapsed < 1.5
        server.close()


def test_server_close_still_closes_storage_after_a_module_fails() -> None:
    """A module that refuses to close must not skip the storage manager's
    close, which runs the checkpoint flush and releases shared memory."""
    failing, other, context = MagicMock(), MagicMock(), MagicMock()
    failing.close.side_effect = RuntimeError("leases must drain")
    MPCacheServer(context, [failing, other]).close()
    other.close.assert_called_once_with()
    context.close.assert_called_once_with()


def test_http_lifespan_drains_before_the_message_queue_closes() -> None:
    """The HTTP server's shutdown serves stores in flight before it closes
    the message queue, then closes the engine."""
    order = MagicMock()
    zmq_server, engine = order.zmq_server, order.engine
    configs = {
        "mp": SimpleNamespace(runtime_plugin_config=SimpleNamespace(locations=[])),
        "storage_manager": MagicMock(),
        "observability": MagicMock(),
    }

    async def run() -> None:
        async with http_server.lifespan(MagicMock()):
            pass

    with (
        patch.dict(http_server._configs, configs, clear=True),
        patch.object(
            http_server, "run_cache_server", return_value=(zmq_server, engine)
        ),
        patch.object(http_server, "build_context", return_value=MagicMock()),
        patch.object(http_server, "get_event_bus", return_value=MagicMock()),
    ):
        asyncio.run(run())
    shutdown = [
        entry
        for entry in order.mock_calls
        if entry
        in (
            call.engine.drain_for_shutdown(),
            call.zmq_server.close(),
            call.engine.close(),
        )
    ]
    assert shutdown == [
        call.engine.drain_for_shutdown(),
        call.zmq_server.close(),
        call.engine.close(),
    ]
