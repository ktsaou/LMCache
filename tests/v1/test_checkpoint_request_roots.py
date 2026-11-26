# SPDX-License-Identifier: Apache-2.0
"""The scheduler bridge keeps a text request's token roots between builds."""

# Standard
from types import SimpleNamespace
from typing import Any

# First Party
from lmcache.integration.vllm import checkpoint_scheduler
from lmcache.integration.vllm.checkpoint_scheduler import CheckpointSchedulerBridge
from lmcache.v1.multiprocess.checkpoint_identity import (
    CheckpointTokenRoots,
    checkpoint_namespace,
)

IDENTITY: dict[str, object] = dict(
    target_revision="target",
    draft_revision="draft",
    source_revision="source",
    layout={"dtype": "fp8", "page_bytes": 1024},
    parallel={"tp": 2, "dcp": 1},
)


def _bridge() -> Any:
    return SimpleNamespace(
        _identity=IDENTITY,
        _multimodal_salt=b"",
        _request_roots={},
        _reserve_admission=False,
        _lookups={},
        _tasks={},
        _cancelled=set(),
    )


def _request(request_id: str, tokens: list[int], salt: str | None = None) -> Any:
    return SimpleNamespace(
        request_id=request_id, all_token_ids=tokens, mm_features=[], cache_salt=salt
    )


def _full(tokens: list[int], salt: str = "") -> CheckpointTokenRoots:
    return CheckpointTokenRoots.build(checkpoint_namespace(IDENTITY, salt), tokens)


def test_roots_follow_a_growing_request_and_end_with_it() -> None:
    bridge = _bridge()
    tokens = list(range(9000))
    request = _request("a", tokens)

    first = CheckpointSchedulerBridge._roots(bridge, request)
    assert first == _full(tokens)
    tokens.extend(range(50))
    second = CheckpointSchedulerBridge._roots(bridge, request)
    assert second == _full(tokens)
    # The two complete chunks were not hashed again.
    assert second.roots[0] is first.roots[0] and second.roots[1] is first.roots[1]
    assert CheckpointSchedulerBridge._roots(bridge, request) is second

    CheckpointSchedulerBridge.finish_request(bridge, "a")
    assert bridge._request_roots == {}


def test_roots_are_rebuilt_for_another_salt_or_a_reused_request_id() -> None:
    bridge = _bridge()
    CheckpointSchedulerBridge._roots(bridge, _request("a", [1, 2, 3]))
    salted = CheckpointSchedulerBridge._roots(bridge, _request("a", [1, 2, 3], "t"))
    assert salted == _full([1, 2, 3], "t")
    other = CheckpointSchedulerBridge._roots(bridge, _request("a", [9, 2, 3, 4], "t"))
    assert other == _full([9, 2, 3, 4], "t")


def test_kept_roots_are_bounded(monkeypatch: Any) -> None:
    monkeypatch.setattr(checkpoint_scheduler, "_MAX_REQUEST_ROOTS", 2)
    bridge = _bridge()
    for name in ("a", "b"):
        CheckpointSchedulerBridge._roots(bridge, _request(name, [1, 2]))
    # Using "a" again makes "b" the least recently used.
    CheckpointSchedulerBridge._roots(bridge, _request("a", [1, 2]))
    CheckpointSchedulerBridge._roots(bridge, _request("c", [1, 2]))
    assert list(bridge._request_roots) == ["a", "c"]
