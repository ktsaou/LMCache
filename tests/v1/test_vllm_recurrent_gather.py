# SPDX-License-Identifier: Apache-2.0
"""CPU coverage for the optional vLLM all-rank output gathering hook."""

# Standard
from types import SimpleNamespace
from unittest.mock import Mock

# Third Party
import pytest

pytest.importorskip("vllm")

# First Party
from lmcache.integration.vllm.checkpoint_scheduler import (  # noqa: E402
    CheckpointSchedulerBridge,
)
from lmcache.integration.vllm.recurrent_checkpoint_connector import (  # noqa: E402
    LMCacheRecurrentCheckpointConnector,
)


def connector(
    monkeypatch: pytest.MonkeyPatch, scheduler: object = None
) -> LMCacheRecurrentCheckpointConnector:
    """Avoid starting RPC clients; inject only the scheduler-side dependency."""
    result = LMCacheRecurrentCheckpointConnector.__new__(
        LMCacheRecurrentCheckpointConnector
    )
    monkeypatch.setattr(result, "_scheduler", scheduler, raising=False)
    return result


def test_no_scheduler_gathers_conservatively(monkeypatch: pytest.MonkeyPatch) -> None:
    instance = connector(monkeypatch)
    assert all(instance.requires_all_worker_outputs() for _ in range(10))


def test_idle_after_all_layouts_arrive(monkeypatch: pytest.MonkeyPatch) -> None:
    bridge = CheckpointSchedulerBridge(Mock(), Mock(), {}, world_size=2)
    instance = connector(monkeypatch, bridge)
    assert instance.requires_all_worker_outputs()
    bridge.accept_layouts({0: {"page_bytes": 128}})
    assert instance.requires_all_worker_outputs()
    # Repeated metadata from one rank does not finish negotiation.
    bridge.accept_layouts({0: {"page_bytes": 128}})
    assert instance.requires_all_worker_outputs()
    bridge.accept_layouts({1: {"page_bytes": 128}})
    assert [instance.requires_all_worker_outputs() for _ in range(4)] == [
        True,
        True,
        False,
        False,
    ]


def test_ready_idle_bridge_skips_immediately(monkeypatch: pytest.MonkeyPatch) -> None:
    instance = connector(monkeypatch, SimpleNamespace(has_pending=False))
    assert not instance.requires_all_worker_outputs()


@pytest.mark.parametrize("pending_steps", [1, 2, 1000])
def test_pending_work_has_no_poll_expiry(
    monkeypatch: pytest.MonkeyPatch, pending_steps: int
) -> None:
    scheduler = SimpleNamespace(has_pending=True)
    instance = connector(monkeypatch, scheduler)
    assert all(instance.requires_all_worker_outputs() for _ in range(pending_steps))
    scheduler.has_pending = False
    # Deferred sample replies still need all-rank metadata after tasks drain.
    assert [instance.requires_all_worker_outputs() for _ in range(4)] == [
        True,
        True,
        False,
        False,
    ]


@pytest.mark.parametrize("idle_steps", [0, 1, 2, 10])
def test_new_work_rearms_gathering(
    monkeypatch: pytest.MonkeyPatch, idle_steps: int
) -> None:
    scheduler = SimpleNamespace(has_pending=True)
    instance = connector(monkeypatch, scheduler)
    assert instance.requires_all_worker_outputs()
    scheduler.has_pending = False
    for _ in range(idle_steps):
        instance.requires_all_worker_outputs()
    scheduler.has_pending = True
    assert instance.requires_all_worker_outputs()
    scheduler.has_pending = False
    assert [instance.requires_all_worker_outputs() for _ in range(3)] == [
        True,
        True,
        False,
    ]


def test_instances_do_not_share_trailing_state(monkeypatch: pytest.MonkeyPatch) -> None:
    busy = connector(monkeypatch, SimpleNamespace(has_pending=True))
    idle = connector(monkeypatch, SimpleNamespace(has_pending=False))
    assert busy.requires_all_worker_outputs()
    assert not idle.requires_all_worker_outputs()
