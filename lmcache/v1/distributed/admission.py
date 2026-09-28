# SPDX-License-Identifier: Apache-2.0
"""Bounded, condition-driven admission retry for atomic L1 stores."""

# Future
from __future__ import annotations

# Standard
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Generic, TypeVar
import time

T = TypeVar("T")


class AdmissionFailure(str, Enum):
    """Stable failure reasons returned by grouped store admission."""

    CAPACITY = "capacity"
    BUSY = "busy"
    CONFLICT = "conflict"
    INVALID_LAYOUT = "invalid_layout"
    TIMEOUT = "capacity_timeout"
    SHUTDOWN = "shutdown"


class AdmissionWaitResult(Enum):
    """Result of waiting for L1 capacity state to change."""

    CHANGED = "changed"
    TIMEOUT = "timeout"
    SHUTDOWN = "shutdown"


@dataclass(frozen=True)
class AdmissionAttempt(Generic[T]):
    """One atomic reservation attempt."""

    value: T | None
    failure_reason: AdmissionFailure | None

    @classmethod
    def success(cls, value: T) -> "AdmissionAttempt[T]":
        """Build a successful attempt."""
        return cls(value=value, failure_reason=None)

    @classmethod
    def failure(cls, reason: AdmissionFailure) -> "AdmissionAttempt[T]":
        """Build a failed attempt with a stable reason."""
        return cls(value=None, failure_reason=reason)


@dataclass(frozen=True)
class AdmissionOutcome(Generic[T]):
    """Final result of bounded admission backpressure."""

    value: T | None
    failure: AdmissionFailure | None
    retries: int
    waited: bool


def reserve_with_eviction_backpressure(
    *,
    attempt: Callable[[], AdmissionAttempt[T]],
    get_generation: Callable[[], int],
    request_eviction: Callable[[], None],
    wait_for_change: Callable[[int, float], tuple[AdmissionWaitResult, int]],
    timeout_seconds: float,
    retry_wait_seconds: float = 0.5,
    on_wait: Callable[[], None] = lambda: None,
    on_retry: Callable[[], None] = lambda: None,
    on_success_after_eviction: Callable[[], None] = lambda: None,
    on_timeout: Callable[[], None] = lambda: None,
) -> AdmissionOutcome[T]:
    """Retry capacity failures until eviction, shutdown, or a deadline.

    ``attempt`` must abort every partial reservation before returning failure.
    Conflict, busy-writer and layout failures return immediately without
    requesting eviction.

    Args:
        attempt: Atomic reservation attempt.
        get_generation: Return the current allocator-change generation.
        request_eviction: Wake the L1 eviction controller.
        wait_for_change: Wait for allocator change or shutdown.
        timeout_seconds: Total capacity-retry deadline.
        retry_wait_seconds: Maximum duration of each condition wait.
        on_wait: Called once when capacity backpressure begins.
        on_retry: Called before each retry attempt.
        on_success_after_eviction: Called when a retry succeeds.
        on_timeout: Called when the total deadline expires.

    Returns:
        Structured final outcome, including retries and whether it waited.

    Raises:
        ValueError: If a timeout is negative or retry wait is non-positive.
    """
    if timeout_seconds < 0:
        raise ValueError("timeout_seconds must be >= 0")
    if retry_wait_seconds <= 0:
        raise ValueError("retry_wait_seconds must be > 0")

    first = attempt()
    if first.failure_reason is None:
        return AdmissionOutcome(first.value, None, retries=0, waited=False)
    if first.failure_reason is not AdmissionFailure.CAPACITY:
        return AdmissionOutcome(None, first.failure_reason, retries=0, waited=False)
    # A failed atomic attempt rolls back its own partial allocations. Observe
    # capacity afterwards so that rollback cannot wake an immediate busy loop.
    generation = get_generation()

    on_wait()
    deadline = time.monotonic() + timeout_seconds
    retries = 0
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            on_timeout()
            return AdmissionOutcome(
                None, AdmissionFailure.TIMEOUT, retries=retries, waited=True
            )

        request_eviction()
        wait_result, observed_generation = wait_for_change(
            generation, min(remaining, retry_wait_seconds)
        )
        if wait_result is AdmissionWaitResult.SHUTDOWN:
            return AdmissionOutcome(
                None, AdmissionFailure.SHUTDOWN, retries=retries, waited=True
            )

        # A condition timeout is still a useful retry point: an existing
        # object may have become evictable without changing allocator usage.
        generation = observed_generation
        retries += 1
        on_retry()
        retried = attempt()
        if retried.failure_reason is None:
            on_success_after_eviction()
            return AdmissionOutcome(retried.value, None, retries=retries, waited=True)
        if retried.failure_reason is not AdmissionFailure.CAPACITY:
            return AdmissionOutcome(
                None, retried.failure_reason, retries=retries, waited=True
            )
        generation = get_generation()
