# SPDX-License-Identifier: Apache-2.0
"""Side requests, branch points and lost pages of recurrent checkpoints.

A conversation's latest checkpoint can be extended by a request that is not
the conversation's next turn: a title or summary request, a sub-agent fork.
Supersession must keep what the conversation still continues from, and a
checkpoint whose pages are gone must stop being offered instead of failing a
restore.
"""

# Standard
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from mmap import mmap
from pathlib import Path
from types import SimpleNamespace
from typing import cast
import hashlib
import json
import math
import time
import uuid

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey
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
from lmcache.v1.multiprocess.checkpoint_index import (
    CheckpointManifest,
    CheckpointPrefix,
)
from lmcache.v1.multiprocess.checkpoint_storage import checkpoint_object_keys
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
from lmcache.v1.multiprocess.modules.checkpoint import CheckpointModule
from lmcache.v1.multiprocess.posix_shm import shm_open_pool_as_mmap

POOL_BYTES = 4 * 1024 * 1024
PAGE_BYTES = 64 * 1024
# Tokens per attention page.
BLOCK = 4
NAMESPACE = "weights-and-layout-and-salt"


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def prefix(tokens: tuple[int, ...]) -> CheckpointPrefix:
    return CheckpointPrefix(NAMESPACE, 4096, b"a" * 32, tokens)


def checkpoint(tokens: tuple[int, ...], kind: str) -> CheckpointManifest:
    """A one-rank bundle shaped like a request-boundary checkpoint.

    Attention pages are keyed by the tokens they cover, so a conversation's
    checkpoints share their full pages; the partial last page, the recurrent
    state and the auxiliary page belong to this endpoint alone.
    """
    positions = list(range(math.ceil(len(tokens) / BLOCK)))
    attention = [
        digest(f"attention:{tokens[: min((position + 1) * BLOCK, len(tokens))]}")
        for position in positions
    ]
    return CheckpointManifest(
        uuid.uuid4().hex,
        prefix(tokens),
        1,
        json.dumps(
            {
                "schema_version": 2,
                "kind": kind,
                "page_groups": [
                    {
                        "name": "target.attention.0",
                        "page_bytes": PAGE_BYTES,
                        "positions": positions,
                        "content_keys": attention,
                    },
                    {
                        "name": "target.recurrent.0",
                        "page_bytes": PAGE_BYTES,
                        "positions": [0],
                        "content_keys": [digest(f"recurrent:{tokens}")],
                    },
                    {
                        "name": "target-draft-auxiliary",
                        "page_bytes": PAGE_BYTES,
                        "positions": [0],
                        "content_keys": [digest(f"auxiliary:{kind}:{tokens}")],
                    },
                ],
            }
        ).encode(),
    )


def page_keys(entry: CheckpointManifest) -> list[ObjectKey]:
    return [key for group in checkpoint_object_keys(entry, 0) for key in group]


def content(key: ObjectKey) -> bytes:
    """The bytes a page holds, derived from its content key."""
    return bytes([key.chunk_hash[0]]) * PAGE_BYTES


@contextmanager
def open_module(
    path: Path,
    *,
    grace_seconds: float = 300.0,
    flush_seconds: float = 0.0,
) -> Iterator[tuple[CheckpointModule, StorageManager, mmap]]:
    """A checkpoint module over a 4 MiB L1 and a native-filesystem L2 whose
    inventory is complete, with checkpoint pages written on L1 eviction."""
    name = f"lmcache_l1_pool_checkpoint_side_{uuid.uuid4().hex}"
    storage = StorageManager(
        StorageManagerConfig(
            L1ManagerConfig(L1MemoryManagerConfig(POOL_BYTES, False, shm_name=name)),
            EvictionConfig(eviction_policy="LRU"),
            l2_adapter_config=L2AdaptersConfig(
                [FSNativeL2AdapterConfig(str(path / "payloads"))]
            ),
            store_policy="checkpoint_on_evict",
            checkpoint_shutdown_flush_seconds=flush_seconds,
            checkpoint_supersede_grace_seconds=grace_seconds,
        )
    )
    with ExitStack() as cleanup:
        cleanup.callback(storage.close)
        mapping = cleanup.enter_context(shm_open_pool_as_mmap(name, POOL_BYTES))
        module = CheckpointModule(
            cast(
                MPCacheServerContext,
                SimpleNamespace(
                    storage_manager=storage,
                    shm_pool_info={"shm_name": name, "pool_size": POOL_BYTES},
                ),
            ),
            path / "directory.sqlite3",
        )
        cleanup.callback(module.close)
        yield module, storage, mapping


def publish(
    module: CheckpointModule, mapping: mmap, entry: CheckpointManifest, request: str
) -> None:
    """Store, publish and report one checkpoint as the vLLM bridge does."""
    assert module.begin(entry)
    lease = module.prepare_store(entry, 0)
    assert lease.status == "ready", lease.status
    slots = [slot for group in lease.slots for slot in group]
    for key, (offset, size) in zip(page_keys(entry), slots, strict=True):
        if offset >= 0:
            mapping[offset : offset + size] = content(key)
    assert module.finish_store(lease.lease_id, True)
    module.supersede((entry.prefix,), entry.generation, request)


def restores(
    module: CheckpointModule, mapping: mmap, entry: CheckpointManifest
) -> bool:
    """Whether every page of ``entry`` restores with the bytes it published."""
    lease = module.begin_retrieve(entry, 0)
    if lease.status != "pending":
        return False
    deadline = time.monotonic() + 10
    while lease.status == "pending" and time.monotonic() < deadline:
        lease = module.poll_retrieve(lease.lease_id)
        time.sleep(0.001)
    if lease.status != "ready":
        return False
    try:
        slots = [slot for group in lease.slots for slot in group]
        for key, (offset, size) in zip(page_keys(entry), slots, strict=True):
            assert mapping[offset : offset + size] == content(key)
    finally:
        module.finish_retrieve(lease.lease_id)
    return True


def cycle_l1(module: CheckpointModule, mapping: mmap, conversations: int) -> None:
    """Publish other conversations until RAM has been refilled about twice."""
    for conversation in range(conversations):
        tokens = tuple(1000 * (conversation + 2) + i for i in range(4 * BLOCK))
        publish(module, mapping, checkpoint(tokens, "prompt"), f"filler-{conversation}")


def wait_for(
    condition: Callable[[], bool], message: str, timeout: float = 10.0
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.02)
    raise AssertionError(message)


# A conversation of two turns: prompt, then the response endpoint.
BASE = tuple(range(100, 100 + 2 * BLOCK))
PROMPT = checkpoint(BASE, "prompt")
RESPONSE_TOKENS = BASE + (7, 8, 9, 10, 11)


def test_side_request_keeps_the_turn_it_extended_under_ram_pressure(
    tmp_path: Path,
) -> None:
    """A title request extends the conversation's response; the user then
    continues the conversation after RAM pressure evicted everything.

    The side request continues from the response, so the response stays
    current: when RAM evicts it, it is written to L2 like any current page,
    and the conversation's next turn restores it. Before, the side request
    superseded it, its state pages were dropped without a write, and the
    listed checkpoint failed its restore ('K of M pages were readable').
    """
    response = checkpoint(RESPONSE_TOKENS, "response")
    with open_module(tmp_path) as (module, storage, mapping):
        publish(module, mapping, checkpoint(BASE, "prompt"), "a")
        publish(module, mapping, response, "a")
        title = checkpoint(RESPONSE_TOKENS + (50, 51, 52), "prompt")
        publish(module, mapping, title, "title")
        publish(
            module,
            mapping,
            checkpoint(title.prefix.tail_tokens + (53,), "response"),
            "title",
        )
        retention = storage.checkpoint_retention
        cycle_l1(module, mapping, 16)
        wait_for(
            lambda: retention.is_l2_resident(page_keys(response)[0]),
            "RAM pressure never reached the conversation",
        )

        # The conversation continues from its response.
        assert module.find((prefix(RESPONSE_TOKENS + (60, 61)),)) == response
        assert restores(module, mapping, response)
        # It stayed current: its pages left RAM with a write, none was dropped.
        assert not any(retention.is_superseded(key) for key in page_keys(response))
        assert retention.report_status()["retired_checkpoints"] >= 1


def test_a_superseded_checkpoint_is_retired_before_its_last_pages_go(
    tmp_path: Path,
) -> None:
    """Pages of a checkpoint the conversation moved past twice are dropped
    without a write; the checkpoint is delisted before they go, so a lookup
    misses it cleanly and falls back to a shorter one at once."""
    instruction = checkpoint(BASE[:BLOCK], "instruction")
    response = checkpoint(RESPONSE_TOKENS, "response")
    turn_2 = checkpoint(RESPONSE_TOKENS + (20, 21), "prompt")
    turn_3 = checkpoint(turn_2.prefix.tail_tokens + (22, 23, 24), "prompt")
    with open_module(tmp_path, grace_seconds=0.0) as (module, storage, mapping):
        publish(module, mapping, instruction, "system")
        publish(module, mapping, PROMPT, "a")
        publish(module, mapping, response, "a")
        publish(module, mapping, turn_2, "b")
        publish(module, mapping, turn_3, "c")
        retention = storage.checkpoint_retention
        # Turn 3 moved past turn 1's prompt and response; turn 2 is where it
        # continued from and stays current.
        assert all(retention.is_superseded(key) for key in page_keys(PROMPT)[-2:])
        assert all(retention.is_superseded(key) for key in page_keys(response)[-2:])
        assert not any(retention.is_superseded(key) for key in page_keys(turn_2))

        cycle_l1(module, mapping, 16)
        wait_for(
            lambda: retention.report_status()["retired_checkpoints"] >= 2,
            "superseded checkpoints were not retired when their pages were dropped",
        )
        assert not any(
            retention.is_l2_resident(key) for key in page_keys(response)[-2:]
        )
        # A branch from the end of turn 1 misses both turn-1 checkpoints at
        # once and gets the shared instruction.
        assert module.find((prefix(RESPONSE_TOKENS + (99,)),)) == instruction
        assert restores(module, mapping, instruction)
        # The conversation itself still continues from turn 3.
        assert module.find((prefix(turn_3.prefix.tail_tokens + (30,)),)) == turn_3
        assert restores(module, mapping, turn_3)


@pytest.mark.parametrize("grace", [300.0, 0.0])
def test_a_fork_point_is_not_dropped_first_during_the_grace_period(
    tmp_path: Path, grace: float
) -> None:
    """A sub-agent forks from the conversation's response and runs two turns,
    which supersedes the response. As a branch point it is not dropped first
    during the grace period; when the conversation finds it again it is
    current again and restores. With no grace, RAM pressure drops it first,
    without a write, and it is retired: the lookup misses it cleanly."""
    response = checkpoint(RESPONSE_TOKENS, "response")
    fork_prompt = checkpoint(RESPONSE_TOKENS + (40, 41), "prompt")
    fork_response = checkpoint(fork_prompt.prefix.tail_tokens + (42,), "response")
    fork_turn_2 = checkpoint(fork_response.prefix.tail_tokens + (43, 44), "prompt")
    continuation = prefix(RESPONSE_TOKENS + (60,))
    with open_module(tmp_path, grace_seconds=grace) as (module, storage, mapping):
        retention = storage.checkpoint_retention
        publish(module, mapping, PROMPT, "a")
        publish(module, mapping, response, "a")
        publish(module, mapping, fork_prompt, "fork")
        publish(module, mapping, fork_response, "fork")
        publish(module, mapping, fork_turn_2, "fork-2")
        state = page_keys(response)[-3:]
        assert all(retention.is_superseded(key) for key in state)
        droppable = set(retention.take_droppable_superseded())
        # The prompt and the fork's prompt were passed, not continued from.
        assert set(page_keys(PROMPT)[-2:]) <= droppable
        assert set(page_keys(fork_prompt)[-2:]) <= droppable
        if grace:
            assert not droppable & set(state)
            assert module.find((continuation,)) == response
            assert not any(retention.is_superseded(key) for key in state)
            assert restores(module, mapping, response)
        else:
            assert set(state) <= droppable
            cycle_l1(module, mapping, 16)
            wait_for(
                lambda: module.find((continuation,)) is None,
                "the dropped fork point stayed listed",
            )
            assert not any(retention.is_l2_resident(key) for key in state)


def test_lookup_retires_checkpoints_whose_pages_no_tier_holds(tmp_path: Path) -> None:
    """A current page lost outside supersession (an admin delete of a page
    that never reached L2) makes its checkpoint unrestorable. The lookup
    retires it and returns the next shorter complete checkpoint in the same
    call instead of offering a restore that fails."""
    response = checkpoint(RESPONSE_TOKENS, "response")
    turn_2 = checkpoint(RESPONSE_TOKENS + (20, 21), "prompt")
    with open_module(tmp_path) as (module, storage, mapping):
        publish(module, mapping, PROMPT, "a")
        publish(module, mapping, response, "a")
        publish(module, mapping, turn_2, "b")
        continuation = prefix(turn_2.prefix.tail_tokens + (30,))
        assert module.find((continuation,)) == turn_2
        deleted, _ = storage.delete_l1_keys(page_keys(turn_2)[-1:])
        assert deleted == 1
        assert module.find((continuation,)) == response
        assert restores(module, mapping, response)
        # Retired, not merely skipped: an exact lookup no longer lists it.
        assert module.find((turn_2.prefix,)) == response


def test_restart_retires_checkpoints_that_were_not_written(tmp_path: Path) -> None:
    """After a restart the directory lists checkpoints whose pages stayed in
    RAM. A lookup skips and retires them and restores the newest one that
    reached L2, without a failed restore first."""
    response = checkpoint(RESPONSE_TOKENS, "response")
    turn_2 = checkpoint(RESPONSE_TOKENS + (20, 21), "prompt")
    with open_module(tmp_path, flush_seconds=30.0) as (module, _storage, mapping):
        publish(module, mapping, PROMPT, "a")
        publish(module, mapping, response, "a")
    with open_module(tmp_path, flush_seconds=0.0) as (module, _storage, mapping):
        publish(module, mapping, turn_2, "b")
    with open_module(tmp_path) as (module, _storage, mapping):
        found = module.find((prefix(turn_2.prefix.tail_tokens + (30,)),))
        assert found == response
        assert restores(module, mapping, response)
        assert module.find((turn_2.prefix,)) == response


def test_ordinary_deletions_never_reach_the_directory() -> None:
    """Deleting KV chunks or current checkpoint pages retires nothing and
    takes no directory lock: only superseded pages carry owners."""
    name = f"lmcache_l1_pool_checkpoint_fastpath_{uuid.uuid4().hex}"
    storage = StorageManager(
        StorageManagerConfig(
            L1ManagerConfig(L1MemoryManagerConfig(POOL_BYTES, False, shm_name=name)),
            EvictionConfig(eviction_policy="LRU"),
        )
    )
    retired: list[list[str]] = []
    try:
        retention = storage.checkpoint_retention
        retention.set_retirement(retired.append)
        ordinary = [
            ObjectKey(ObjectKey.IntHash2Bytes(i), "some-model", 0) for i in range(3)
        ]
        current = page_keys(PROMPT)
        assert retention.retire_before_l1_delete(ordinary + current) == 0
        assert retention.retire_before_l2_delete(0, ordinary + current) == 0
        retention.mark_superseded("old", current[-1:])
        assert retention.retire_before_l1_delete(ordinary + current) == 1
        assert retired == [["old"]]
    finally:
        storage.close()


def test_lookup_trusts_pages_another_process_wrote(tmp_path: Path) -> None:
    """Two servers share one filesystem tier and directory. A checkpoint the
    other one wrote after this one started is not in this one's records, but
    its files exist: the lookup keeps it and it restores. Only pages that no
    tier holds make a lookup retire a checkpoint."""
    response = checkpoint(RESPONSE_TOKENS, "response")
    with open_module(tmp_path) as (module, _storage, mapping):
        with open_module(tmp_path, flush_seconds=30.0) as (writer, _, writer_map):
            publish(writer, writer_map, response, "a")
        assert module.find((prefix(RESPONSE_TOKENS + (60,)),)) == response
        assert restores(module, mapping, response)
