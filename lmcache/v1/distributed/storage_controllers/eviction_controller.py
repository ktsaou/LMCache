# SPDX-License-Identifier: Apache-2.0

# Future
from __future__ import annotations

# Standard
from abc import abstractmethod
from collections import Counter
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING
import enum
import inspect
import math
import threading
import time

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.config import EvictionConfig
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.eviction import L1EvictionPolicy, L2EvictionPolicy
from lmcache.v1.distributed.eviction_policy import CreateEvictionPolicy
from lmcache.v1.distributed.internal_api import (
    EvictionAction,
    EvictionDestination,
)
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.l2_adapters.base import L2AdapterInterface
from lmcache.v1.distributed.storage_controller import StorageControllerInterface
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import get_event_bus

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.distributed.checkpoint_retention import CheckpointRetention
    from lmcache.v1.distributed.quota_manager import QuotaManager

logger = init_logger(__name__)

_SyncStore = Callable[..., tuple[bool, int, int]]
_PendingStoreCheck = Callable[[list[ObjectKey]], bool]


class _SyncFlushResult(enum.Enum):
    SUCCESS = enum.auto()
    FAILURE = enum.auto()
    DEADLINE_EXHAUSTED = enum.auto()


class EvictionController(StorageControllerInterface):
    """
    Abstract base class for eviction controllers.

    Provides the shared eviction loop structure: background thread and stop
    flag. Subclasses implement eviction_loop and execute_eviction_action
    for their specific tier (L1 or L2).
    """

    def __init__(self):
        self._stop_flag = threading.Event()
        self._thread = threading.Thread(
            target=self.eviction_loop,
            daemon=True,
        )

    def start(self):
        logger.info("Starting %s...", self.__class__.__name__)
        self._thread.start()

    def stop(self):
        self._stop_flag.set()
        self._thread.join()

    @abstractmethod
    def report_status(self) -> dict:
        """Return a status dict for this controller.

        The child class needs to override this function to report
        controller-specific health and configuration information.
        """
        pass

    @abstractmethod
    def eviction_loop(self):
        """Run the eviction loop.

        The child class needs to override this function to implement
        internal eviction controlling logic.
        """
        pass

    @abstractmethod
    def execute_eviction_action(self, action: EvictionAction):
        """Execute a single eviction action.

        The child class needs to override this function to implement
        internal eviction controlling logic.
        """
        pass


# Superseded checkpoint pages dropped per eviction pass, before LRU victims.
_SUPERSEDED_DROP_BATCH = 4096


class L1EvictionController(EvictionController):
    """
    Eviction controller for L1 cache.

    Uses an L1EvictionPolicy bridge to keep the eviction policy up-to-date
    with L1 manager events, and periodically triggers eviction based on
    L1 memory usage.

    When writeback is enabled, eviction batches are synchronously persisted
    through a capable L2 adapter and removed from L1 only after the complete
    readable batch is durable. Missing, partial, or failed persistence keeps
    the L1 data resident.
    """

    _SYNC_FLUSH_BATCH_SIZE = 128
    _BACKUP_FLUSH_BATCH_SIZE = 128
    _EMERGENCY_FLUSH_TIMEOUT_SECONDS = 1.0
    _PERIODIC_FLUSH_TIMEOUT_SECONDS = 5.0

    def __init__(
        self,
        l1_manager: L1Manager,
        eviction_config: EvictionConfig,
        l2_adapters: Mapping[int, object] | None = None,
    ):
        super().__init__()
        self._eviction_config = eviction_config
        self._eviction_policy = CreateEvictionPolicy(eviction_config)
        self._l1_manager = l1_manager
        self._listener = L1EvictionPolicy(self._eviction_policy)
        self._l1_manager.register_listener(self._listener)
        self._event_bus = get_event_bus()
        self._last_extra_log = time.monotonic()
        self._immediate_request = threading.Event()
        self._pressure_since: float | None = None

        self._write_back_enabled = eviction_config.write_back_on_evict
        self._periodic_backup_enabled = eviction_config.periodic_flush_interval > 0
        self._l2_flush_adapters: dict[int, _SyncStore] = {}
        self._l2_adapter_supports_timeout: dict[int, bool] = {}
        self._l2_adapter_pending_store_checks: dict[int, _PendingStoreCheck] = {}
        self._l2_adapters_lock = threading.Lock()
        # Adapter replacement and synchronous persistence share this lock.
        # A runtime adapter removal can therefore wait for an active flush
        # before closing the adapter's native resources.
        self._flush_lock = threading.Lock()
        self._sync_flush_failures = 0
        self._sync_flush_backoff_until = 0.0
        self._last_backup_flush = time.monotonic()
        self._backup_flush_cursor = 0
        self._emergency_evict_lock = threading.Lock()
        self._retention: "CheckpointRetention | None" = None
        if self._write_back_enabled:
            # Register the durable destination even when no adapter is
            # currently compatible. This is the fail-closed invariant: policy
            # output never silently falls back to DISCARD.
            self._eviction_policy.register_eviction_destination(
                EvictionDestination.L2_CACHE
            )
        if l2_adapters:
            self.set_l2_adapters(l2_adapters)

    def set_checkpoint_retention(self, retention: "CheckpointRetention") -> None:
        """Order eviction by checkpoint supersession and write-on-evict state."""
        self._retention = retention

    def touch_keys(self, keys: list[ObjectKey]) -> None:
        """Move keys to the most recently used position without an L1 read.

        Keys that L1 does not hold are ignored.

        Args:
            keys: Keys a lookup proved are still wanted.
        """
        self._listener.on_l1_keys_accessed(keys)

    def _eligibility_filter(
        self, to_persist: list[ObjectKey]
    ) -> Callable[[ObjectKey], bool]:
        """Return the key filter for one eviction pass.

        With write-on-evict checkpoint storage, a current checkpoint page that
        is not in L2 yet is skipped and collected in ``to_persist``; the caller
        requests its L2 write, and a later pass evicts it once stored.
        """
        base = (
            self._is_writeback_evictable
            if self._write_back_enabled
            else self._l1_manager.is_key_evictable
        )
        retention = self._retention
        if retention is None or not retention.write_on_evict:
            return base

        def eligible(key: ObjectKey) -> bool:
            if not base(key):
                return False
            if retention.needs_persist_before_evict(key):
                to_persist.append(key)
                return False
            return True

        return eligible

    def _request_persist(self, to_persist: list[ObjectKey]) -> None:
        if self._retention is not None and to_persist:
            keys = list(dict.fromkeys(to_persist))
            retained = [key for key in keys if self._retention.is_retained(key)]
            orphaned = [key for key in keys if key not in retained]
            if orphaned:
                self._delete_l1(orphaned)
            self._retention.request_persist(retained)

    def _delete_l1(self, keys: list[ObjectKey]) -> dict[ObjectKey, L1Error]:
        if self._retention is not None:
            result = self._retention.evict(keys, "l1", self._l1_manager.delete)
        else:
            result = self._l1_manager.delete(keys)
        if any(error == L1Error.SUCCESS for error in result.values()):
            self._reclamation_generation = (
                getattr(self, "_reclamation_generation", 0) + 1
            )
        return result

    def _drop_superseded(self) -> int:
        """Evict superseded checkpoint pages from L1 before any LRU victim."""
        retention = self._retention
        if retention is None:
            return 0
        victims = [
            key
            for key in retention.superseded_keys()
            if self._l1_manager.is_key_evictable(key)
        ][:_SUPERSEDED_DROP_BATCH]
        if not victims:
            return 0
        to_persist = [
            key for key in victims if retention.needs_persist_before_evict(key)
        ]
        self._request_persist(to_persist)
        result = self._delete_l1([key for key in victims if key not in to_persist])
        dropped = sum(1 for error in result.values() if error == L1Error.SUCCESS)
        retention.record_l1_superseded_drops(dropped)
        return dropped

    def request_immediate_eviction(self) -> None:
        """Wake the eviction loop for a capacity-blocked store."""
        now = time.monotonic()
        if (
            getattr(self, "_pressure_since", None) is None
            or now - getattr(self, "_last_pressure_request", 0.0) > 1.0
        ):
            self._pressure_since = now
        self._last_pressure_request = now
        self._immediate_request.set()

    def set_pressure_grace(self, seconds: float) -> None:
        """Allocate part of the existing admission deadline to persistence."""
        self._pressure_grace = max(0.0, seconds)
        self._pressure_since = None

    @property
    def trigger_watermark(self) -> float:
        """L1 pressure threshold used to reserve publication headroom."""
        return self._eviction_config.trigger_watermark

    def stop(self) -> None:
        """Stop promptly even when the eviction loop is waiting."""
        self._stop_flag.set()
        self._immediate_request.set()
        self._thread.join()

    def set_l2_adapters(self, l2_adapters: Mapping[int, object]) -> None:
        """Replace the adapters eligible for synchronous L1 writeback.

        The replacement waits for an active flush, so callers may close any
        removed adapter immediately after this method returns.
        """
        if not (self._write_back_enabled or self._periodic_backup_enabled):
            with self._flush_lock:
                with self._l2_adapters_lock:
                    self._l2_flush_adapters = {}
                    self._l2_adapter_supports_timeout = {}
                    self._l2_adapter_pending_store_checks = {}
            return
        compatible: dict[int, _SyncStore] = {}
        supports_timeout: dict[int, bool] = {}
        pending_store_checks: dict[int, _PendingStoreCheck] = {}
        for adapter_id, adapter in l2_adapters.items():
            sync_store = getattr(adapter, "store_objects_sync", None)
            if callable(sync_store):
                compatible[adapter_id] = sync_store
                try:
                    parameters = inspect.signature(sync_store).parameters.values()
                    supports_timeout[adapter_id] = any(
                        parameter.name == "timeout"
                        or parameter.kind is inspect.Parameter.VAR_KEYWORD
                        for parameter in parameters
                    )
                except (TypeError, ValueError):
                    supports_timeout[adapter_id] = False
                pending_store_check = getattr(
                    adapter,
                    "has_inflight_store_for_keys",
                    None,
                )
                if callable(pending_store_check):
                    pending_store_checks[adapter_id] = pending_store_check
        ignored = sorted(set(l2_adapters) - set(compatible))
        if ignored:
            logger.warning(
                "L1 persistence ignored adapters without store_objects_sync: %s",
                ignored,
            )
        if (
            self._periodic_backup_enabled
            or self._eviction_config.emergency_evict_for_prefetch
        ):
            unbounded = sorted(
                adapter_id
                for adapter_id in compatible
                if not supports_timeout[adapter_id]
            )
            if unbounded:
                logger.warning(
                    "Bounded L1 persistence ignored adapters without a "
                    "timeout contract: %s",
                    unbounded,
                )
        with self._flush_lock:
            with self._l2_adapters_lock:
                self._l2_flush_adapters = compatible
                self._l2_adapter_supports_timeout = supports_timeout
                self._l2_adapter_pending_store_checks = pending_store_checks

    def _snapshot_l2_flush_adapters(self) -> list[tuple[int, _SyncStore]]:
        with self._l2_adapters_lock:
            return list(self._l2_flush_adapters.items())

    def _snapshot_periodic_flush_adapters(self) -> list[tuple[int, _SyncStore]]:
        """Return adapters whose sync-store contract accepts a timeout."""
        with self._l2_adapters_lock:
            return [
                (adapter_id, sync_store)
                for adapter_id, sync_store in self._l2_flush_adapters.items()
                if self._l2_adapter_supports_timeout.get(adapter_id, False)
            ]

    def _has_inflight_store_for_key(self, key: ObjectKey) -> bool:
        """Fail closed if an adapter may still consume this L1 buffer."""
        with self._l2_adapters_lock:
            checks = list(self._l2_adapter_pending_store_checks.items())
        for adapter_id, check in checks:
            try:
                if check([key]):
                    return True
            except Exception:
                logger.exception(
                    "L1 eviction could not query in-flight stores on adapter %d",
                    adapter_id,
                )
                return True
        return False

    def _is_writeback_evictable(self, key: ObjectKey) -> bool:
        return self._l1_manager.is_key_evictable(
            key
        ) and not self._has_inflight_store_for_key(key)

    def remove_l2_adapter(self, adapter_id: int) -> None:
        """Stop writeback to one adapter and wait for its active flush."""
        with self._flush_lock:
            with self._l2_adapters_lock:
                self._l2_flush_adapters.pop(adapter_id, None)
                self._l2_adapter_supports_timeout.pop(adapter_id, None)
                self._l2_adapter_pending_store_checks.pop(adapter_id, None)

    def has_l2_flush_adapter(self) -> bool:
        """Return whether a synchronous durability path is available."""
        return bool(self._snapshot_l2_flush_adapters())

    def has_periodic_flush_adapter(self) -> bool:
        """Return whether a bounded periodic-backup path is available."""
        return bool(self._snapshot_periodic_flush_adapters())

    def has_bounded_l2_flush_adapter(self) -> bool:
        """Return whether synchronous L2 persistence can honor a deadline."""
        return bool(self._snapshot_periodic_flush_adapters())

    def report_status(self) -> dict:
        adapters = self._snapshot_l2_flush_adapters()
        return {
            "is_healthy": self._thread.is_alive(),
            "thread_alive": self._thread.is_alive(),
            "eviction_policy": self._eviction_config.eviction_policy,
            "trigger_watermark": self._eviction_config.trigger_watermark,
            "eviction_ratio": self._eviction_config.eviction_ratio,
            "write_back_enabled": self._write_back_enabled,
            "periodic_flush_interval": (self._eviction_config.periodic_flush_interval),
            "periodic_flush_enabled": self.has_periodic_flush_adapter(),
            "l2_flush_enabled": bool(adapters),
            "l2_flush_adapter_ids": [adapter_id for adapter_id, _ in adapters],
            "sync_flush_failures": self._sync_flush_failures,
            "sync_flush_backoff_seconds": max(
                0.0, self._sync_flush_backoff_until - time.monotonic()
            ),
        }

    def _publish_skipped(self, usage: float, watermark: float) -> None:
        """Publish a below-watermark loop tick (no eviction this cycle)."""
        self._event_bus.publish(
            Event(
                event_type=EventType.L1_EVICTION_LOOP_TICK,
                metadata={
                    "usage": usage,
                    "watermark": watermark,
                    "triggered": False,
                },
            )
        )

    def _publish_triggered(self, usage: float, watermark: float) -> None:
        """Publish an above-watermark loop tick (eviction policy ran)."""
        self._event_bus.publish(
            Event(
                event_type=EventType.L1_EVICTION_LOOP_TICK,
                metadata={
                    "usage": usage,
                    "watermark": watermark,
                    "triggered": True,
                },
            )
        )

    def _maybe_log_memory_usage(self, used_bytes: int, total_bytes: int) -> None:
        """Emit the opt-in L1 memory usage INFO line, throttled to the interval."""
        now = time.monotonic()
        if now - self._last_extra_log < self._eviction_config.extra_logging_interval:
            return
        self._last_extra_log = now
        pct = 0.0 if total_bytes == 0 else used_bytes / total_bytes * 100.0
        logger.info(
            "L1 memory usage: %.2f/%.2f GiB (%.1f%%)",
            used_bytes / (1 << 30),
            total_bytes / (1 << 30),
            pct,
        )

    def eviction_loop(self):
        while not self._stop_flag.is_set():
            immediate = self._immediate_request.wait(timeout=1.0)
            self._immediate_request.clear()
            if self._stop_flag.is_set():
                break
            try:
                self._run_eviction_pass(immediate)
            except Exception:
                # An exception must not end the loop: nothing would ever be
                # evicted again and stores would fail once L1 is full.
                logger.exception("L1 eviction pass failed; retrying next pass")

    def _run_eviction_pass(self, immediate: bool) -> None:
        reclamation = getattr(self, "_reclamation_generation", 0)
        self._l1_manager.reclaim_abandoned_writes()
        watermark = self._eviction_config.trigger_watermark
        eviction_ratio = self._eviction_config.eviction_ratio
        backup_interval = self._eviction_config.periodic_flush_interval
        used_bytes, total_bytes = self._l1_manager.get_memory_usage()
        if self._eviction_config.extra_logging_enabled:
            self._maybe_log_memory_usage(used_bytes, total_bytes)
        usage = 0 if total_bytes == 0 else used_bytes / total_bytes
        if usage < watermark and not immediate:
            self._pressure_since = None
            now = time.monotonic()
            if (
                backup_interval > 0
                and self._snapshot_periodic_flush_adapters()
                and now - self._last_backup_flush >= backup_interval
            ):
                self._last_backup_flush = now
                self._backup_to_l2_no_delete(self._BACKUP_FLUSH_BATCH_SIZE)
            logger.debug(
                "L1 memory usage %.2f below watermark %.2f; skipping eviction.",
                usage,
                watermark,
            )
            self._publish_skipped(usage, watermark)
            return

        if (
            self._write_back_enabled
            and time.monotonic() < self._sync_flush_backoff_until
        ):
            self._publish_skipped(usage, watermark)
            return

        logger.info(
            "L1 memory usage %.2f above watermark %.2f; triggering eviction%s.",
            usage,
            watermark,
            " immediately" if immediate else "",
        )
        if self._drop_superseded():
            used_bytes, total_bytes = self._l1_manager.get_memory_usage()
            if total_bytes and used_bytes / total_bytes < watermark:
                self._pressure_since = None
                self._publish_triggered(usage, watermark)
                return
        to_persist: list[ObjectKey] = []
        actions = self._eviction_policy.get_eviction_actions(
            eviction_ratio,
            key_eligible_filter=self._eligibility_filter(to_persist),
        )
        for action in actions:
            self.execute_eviction_action(action)
        self._request_persist(to_persist)
        if getattr(self, "_reclamation_generation", 0) != reclamation:
            self._pressure_since = None
        pressure_since = getattr(self, "_pressure_since", None)
        if (
            immediate
            and self._retention is not None
            and pressure_since is not None
            and time.monotonic() - pressure_since
            >= getattr(self, "_pressure_grace", 1.0)
        ):
            retired_bytes = self._retention.retire_under_pressure(
                max(1, int(total_bytes * eviction_ratio))
            )
            if retired_bytes:
                self._pressure_since = None
        self._publish_triggered(usage, watermark)

    def execute_eviction_action(self, action: EvictionAction):
        if action.destination == EvictionDestination.L2_CACHE:
            if self._write_back_enabled:
                self._flush_to_l2_then_delete(action.keys)
            else:
                logger.error("L2 eviction destination requires writeback")
                logger.error("Treating it as DISCARD.")
                self._delete_l1(action.keys)
        elif action.destination == EvictionDestination.DISCARD:
            self._delete_l1(action.keys)
        else:
            logger.error("Unsupported eviction destination: %s", action.destination)
            logger.error("Treating it as DISCARD.")
            self._delete_l1(action.keys)

    def emergency_evict_bytes(
        self,
        target_free_bytes: int,
        requester: str = "",
        cache_salt: str | None = None,
    ) -> int:
        """Synchronously make room in L1 through bounded, ownership-safe eviction.

        The normal LRU policy chooses victims. A hard deadline bounds lock
        acquisition and L2 persistence, and deadline exhaustion does not count
        as a backend failure for the normal writeback circuit breaker.
        Write-through uses ordinary discard eviction without issuing L2 writes.
        """
        deadline = time.monotonic() + self._EMERGENCY_FLUSH_TIMEOUT_SECONDS
        if not self._emergency_evict_lock.acquire(
            timeout=self._EMERGENCY_FLUSH_TIMEOUT_SECONDS
        ):
            used, total = self._l1_manager.get_memory_usage()
            return max(0, total - used)
        try:
            used, total = self._l1_manager.get_memory_usage()
            target_free_bytes = min(max(0, target_free_bytes), total)
            free = max(0, total - used)
            if free >= target_free_bytes:
                return free
            if self._write_back_enabled and (
                not self.has_bounded_l2_flush_adapter()
                or time.monotonic() < self._sync_flush_backoff_until
            ):
                return free
            if self._eviction_policy.support_isolation and cache_salt is None:
                logger.warning(
                    "Emergency L1 eviction skipped: isolated eviction policy "
                    "requires a cache_salt"
                )
                return free

            start = time.monotonic()
            initial_free = free
            initial_objects = self._l1_manager.num_objects()
            to_persist: list[ObjectKey] = []
            if self._drop_superseded():
                used, total = self._l1_manager.get_memory_usage()
                free = max(0, total - used)
            while free < target_free_bytes and time.monotonic() < deadline:
                num_objects = self._l1_manager.num_objects()
                if num_objects <= 0:
                    break
                deficit = target_free_bytes - free
                average_size = max(1, used // num_objects)
                need_keys = math.ceil(deficit / average_size)
                # Modest headroom reduces a second pass for allocator
                # alignment while the loop still verifies actual free bytes.
                need_keys += max(1, math.ceil(need_keys / 8))
                tracked = num_objects
                get_tracked = getattr(
                    self._eviction_policy,
                    "get_num_tracked_keys",
                    None,
                )
                if callable(get_tracked):
                    tracked = max(
                        1,
                        int(
                            get_tracked(cache_salt)
                            if self._eviction_policy.support_isolation
                            else get_tracked()
                        ),
                    )
                actions = self._eviction_policy.get_eviction_actions(
                    min(1.0, need_keys / max(1, tracked)),
                    key_eligible_filter=self._eligibility_filter(to_persist),
                    cache_salt=cache_salt,
                )
                if not actions:
                    break

                free_before_pass = free
                for action in actions:
                    if (
                        self._write_back_enabled
                        and action.destination == EvictionDestination.L2_CACHE
                    ):
                        outcome = self._flush_to_l2_then_delete(
                            action.keys,
                            deadline=deadline,
                        )
                        if outcome is not _SyncFlushResult.SUCCESS:
                            break
                    else:
                        self.execute_eviction_action(action)
                    if time.monotonic() >= deadline:
                        break
                used, total = self._l1_manager.get_memory_usage()
                free = max(0, total - used)
                if free <= free_before_pass:
                    break

            self._request_persist(to_persist)
            if free < target_free_bytes:
                # Continue through the same bounded pressure policy used by
                # store admission while the prefetch caller retries for room.
                self.request_immediate_eviction()
            evicted_keys = max(0, initial_objects - self._l1_manager.num_objects())

            logger.info(
                "Emergency L1 eviction%s: wanted %.0f MB free, evicted %d keys "
                "in %.0f ms; free %.0f MB -> %.0f MB",
                (" for " + requester) if requester else "",
                target_free_bytes / 1e6,
                evicted_keys,
                (time.monotonic() - start) * 1000.0,
                initial_free / 1e6,
                free / 1e6,
            )
            return free
        finally:
            self._emergency_evict_lock.release()

    def _flush_to_l2_then_delete(
        self,
        keys: list[ObjectKey],
        deadline: float | None = None,
    ) -> _SyncFlushResult:
        """Persist bounded batches and delete only fully durable batches."""
        if deadline is None:
            self._flush_lock.acquire()
        else:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not self._flush_lock.acquire(timeout=remaining):
                return _SyncFlushResult.DEADLINE_EXHAUSTED
        try:
            if not keys or time.monotonic() < self._sync_flush_backoff_until:
                return _SyncFlushResult.SUCCESS

            outcome = _SyncFlushResult.SUCCESS
            for start in range(0, len(keys), self._SYNC_FLUSH_BATCH_SIZE):
                if deadline is not None and time.monotonic() >= deadline:
                    outcome = _SyncFlushResult.DEADLINE_EXHAUSTED
                    break
                batch = keys[start : start + self._SYNC_FLUSH_BATCH_SIZE]
                outcome = self._flush_one_l2_batch_then_delete(
                    batch,
                    deadline=deadline,
                )
                if outcome is not _SyncFlushResult.SUCCESS:
                    break

            if outcome is _SyncFlushResult.SUCCESS:
                self._sync_flush_failures = 0
                self._sync_flush_backoff_until = 0.0
            elif outcome is _SyncFlushResult.FAILURE:
                self._sync_flush_failures += 1
                delay = min(60.0, float(2 ** min(self._sync_flush_failures, 6)))
                self._sync_flush_backoff_until = time.monotonic() + delay
                logger.warning(
                    "L1-to-L2 flush circuit breaker: failure=%d, retry in %.0fs; "
                    "remaining L1 keys preserved",
                    self._sync_flush_failures,
                    delay,
                )
            else:
                logger.debug(
                    "L1-to-L2 emergency flush exhausted its deadline; "
                    "remaining L1 keys preserved"
                )
            return outcome
        finally:
            self._flush_lock.release()

    def _flush_one_l2_batch_then_delete(
        self,
        keys: list[ObjectKey],
        deadline: float | None = None,
    ) -> _SyncFlushResult:
        """Persist one readable batch, then delete only that durable set."""
        if not keys:
            return _SyncFlushResult.SUCCESS

        read_result = self._l1_manager.reserve_read(keys)
        readable_keys: list[ObjectKey] = []
        readable_objs = []
        for key in keys:
            entry = read_result.get(key)
            if (
                entry is not None
                and entry[0] == L1Error.SUCCESS
                and entry[1] is not None
                and not self._has_inflight_store_for_key(key)
            ):
                readable_keys.append(key)
                readable_objs.append(entry[1])

        if not readable_keys:
            return _SyncFlushResult.SUCCESS

        durable = False
        deadline_exhausted = False
        try:
            for adapter_id, sync_store in self._snapshot_l2_flush_adapters():
                try:
                    if deadline is None:
                        result = sync_store(readable_keys, readable_objs)
                    else:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            deadline_exhausted = True
                            break
                        if not self._l2_adapter_supports_timeout.get(adapter_id, False):
                            logger.warning(
                                "Emergency L1 writeback skipped adapter %d: "
                                "store_objects_sync has no timeout contract",
                                adapter_id,
                            )
                            continue
                        result = sync_store(
                            readable_keys,
                            readable_objs,
                            timeout=remaining,
                        )
                    ok, persisted_count, bytes_written = result
                except Exception:
                    if deadline is not None and time.monotonic() >= deadline:
                        deadline_exhausted = True
                        break
                    logger.exception(
                        "L1-to-L2 flush: sync store failed on adapter %d",
                        adapter_id,
                    )
                    continue
                if ok and persisted_count == len(readable_keys):
                    durable = True
                    logger.info(
                        "L1-to-L2 flush: persisted %d keys (%d bytes) via adapter %d",
                        persisted_count,
                        bytes_written,
                        adapter_id,
                    )
                    break
                logger.warning(
                    "L1-to-L2 flush: adapter %d reported %d/%d durable keys (%d bytes)",
                    adapter_id,
                    persisted_count,
                    len(readable_keys),
                    bytes_written,
                )
        finally:
            self._l1_manager.finish_read(readable_keys)

        if deadline_exhausted:
            return _SyncFlushResult.DEADLINE_EXHAUSTED
        if not durable:
            logger.warning(
                "L1-to-L2 flush: preserving %d readable keys after failed persistence",
                len(readable_keys),
            )
            return _SyncFlushResult.FAILURE

        delete_result = self._delete_l1(readable_keys)
        not_deleted = [
            key for key, error in delete_result.items() if error != L1Error.SUCCESS
        ]
        if not_deleted:
            logger.debug(
                "L1-to-L2 flush: %d durable keys remained locked in L1",
                len(not_deleted),
            )
        return _SyncFlushResult.SUCCESS

    def _backup_to_l2_no_delete(self, batch_limit: int) -> None:
        """Persist a bounded rotating L1 batch without deleting its L1 copy.

        Periodic work accepts only adapters with an explicit timeout contract,
        ensuring controller shutdown and runtime adapter replacement stay
        bounded even when an L2 backend is unhealthy.
        """
        with self._flush_lock:
            batch, self._backup_flush_cursor = self._l1_manager.get_evictable_keys(
                limit=batch_limit,
                cursor=self._backup_flush_cursor,
            )
            if not batch:
                return

            read_result = self._l1_manager.reserve_read(batch)
            readable_keys: list[ObjectKey] = []
            readable_objs = []
            for key in batch:
                entry = read_result.get(key)
                if (
                    entry is not None
                    and entry[0] == L1Error.SUCCESS
                    and entry[1] is not None
                ):
                    readable_keys.append(key)
                    readable_objs.append(entry[1])

            if not readable_keys:
                return

            try:
                for adapter_id, sync_store in self._snapshot_periodic_flush_adapters():
                    try:
                        ok, persisted_count, bytes_written = sync_store(
                            readable_keys,
                            readable_objs,
                            timeout=self._PERIODIC_FLUSH_TIMEOUT_SECONDS,
                        )
                    except Exception:
                        logger.exception(
                            "Periodic L2 backup: sync store failed on adapter %d",
                            adapter_id,
                        )
                        continue
                    if ok and persisted_count == len(readable_keys):
                        logger.info(
                            "Periodic L2 backup: persisted %d keys (%d bytes) "
                            "via adapter %d; L1 copies retained",
                            persisted_count,
                            bytes_written,
                            adapter_id,
                        )
                        break
                    logger.warning(
                        "Periodic L2 backup: adapter %d reported %d/%d durable "
                        "keys (%d bytes); L1 copies retained",
                        adapter_id,
                        persisted_count,
                        len(readable_keys),
                        bytes_written,
                    )
            finally:
                self._l1_manager.finish_read(readable_keys)


class L2AdapterEvictionState:
    """Per-adapter eviction state: its own policy, listener, and config."""

    def __init__(
        self,
        adapter_id: int,
        adapter: L2AdapterInterface,
        eviction_config: EvictionConfig,
    ):
        self.adapter_id = adapter_id
        self.adapter = adapter
        self.eviction_config = eviction_config
        self.eviction_policy = CreateEvictionPolicy(eviction_config)
        self.listener = L2EvictionPolicy(self.eviction_policy)
        adapter.register_listener(self.listener)
        # StorageManager constructs this state before exposing the adapter to
        # store/prefetch controllers. Seed one key at a time because normal
        # store batches are token-ordered and LRU intentionally reverses them;
        # the inventory itself is already ordered oldest to newest.
        for key, size in adapter.get_existing_key_sizes().items():
            self.listener.on_l2_keys_stored([key], [size])


class L2EvictionController(StorageControllerInterface):
    """
    Unified eviction controller for all L2 adapters.

    Each adapter gets its own eviction policy and listener bridge, but a
    single background thread loops over all of them.

    When the adapter's policy sets ``support_isolation == True``
    (e.g. :class:`IsolatedLRUEvictionPolicy`), the controller consults
    the injected :class:`QuotaManager` to decide which ``cache_salt``
    buckets are over budget and evicts from each one in isolation.
    Otherwise it uses the adapter's aggregate ``usage_fraction``
    against the configured watermark — unchanged from the pre-PR5
    behavior.
    """

    def __init__(
        self,
        l2_adapter_states: list[L2AdapterEvictionState],
        quota_manager: QuotaManager | None = None,
    ):
        self._adapter_states = l2_adapter_states
        self._quota_manager = quota_manager
        self._retention: "CheckpointRetention | None" = None
        # Guards _adapter_states against concurrent runtime add/remove.
        self._states_lock = threading.Lock()
        self._stop_flag = threading.Event()
        self._thread = threading.Thread(
            target=self._eviction_loop,
            daemon=True,
        )

    def start(self):
        logger.info("Starting %s...", self.__class__.__name__)
        self._thread.start()

    def stop(self):
        self._stop_flag.set()
        self._thread.join()

    def set_checkpoint_retention(self, retention: "CheckpointRetention") -> None:
        """Evict superseded checkpoint pages before LRU victims."""
        self._retention = retention

    def _evict_superseded(self, state: L2AdapterEvictionState) -> bool:
        """Delete superseded checkpoint pages first; return True if any went."""
        retention = self._retention
        if retention is None:
            return False
        usage = state.adapter.get_usage()
        budget = int(usage.total_bytes_used * state.eviction_config.eviction_ratio)
        victims, size = retention.superseded_in_adapter(
            state.adapter_id, max(1, budget)
        )
        if not victims:
            return False
        self._execute_eviction_action(
            state.adapter,
            EvictionAction(keys=victims, destination=EvictionDestination.DISCARD),
        )
        retention.record_l2_superseded_evictions(len(victims), size)
        logger.info(
            "L2 eviction removed %d superseded checkpoint pages (%.1f MB) first",
            len(victims),
            size / 1e6,
        )
        return True

    def touch_keys(self, keys: list[ObjectKey]) -> None:
        """Move keys to the most recently used position in every adapter.

        Only eviction order changes. An adapter ignores keys it does not
        hold, and store admission is not told that the keys are in L2.

        Args:
            keys: Keys a lookup proved are still wanted.
        """
        if not keys:
            return
        # The eviction loop holds _states_lock for a whole pass, so read the
        # list without it. Removal replaces the list; touching a removed
        # adapter's policy only reorders memory that is about to be dropped.
        for state in list(self._adapter_states):
            state.listener.on_l2_keys_accessed(keys)

    def add_adapter_state(self, state: L2AdapterEvictionState) -> None:
        """Register a new adapter's eviction state at runtime."""
        with self._states_lock:
            self._adapter_states.append(state)

    def remove_adapter_state(self, adapter_id: int) -> None:
        """Drop the eviction state for ``adapter_id``.

        Blocks until any in-progress eviction pass finishes (it holds the
        same lock), so the adapter is guaranteed idle here before the
        caller closes it. A no-op if the adapter has no eviction state.
        """
        with self._states_lock:
            self._adapter_states = [
                s for s in self._adapter_states if s.adapter_id != adapter_id
            ]

    def report_status(self) -> dict:
        # NOTE: ``usage.bytes_by_cache_salt`` is intentionally NOT
        # surfaced here. A deployment can have 10k+ salts, so embedding
        # the full bucket map in the status response would blow up the
        # payload. Per-salt inspection goes through the dedicated HTTP
        # quota endpoints (which pull from ``QuotaManager`` +
        # ``StorageManager.get_usage_bytes_by_cache_salt``).
        adapter_statuses = []
        with self._states_lock:
            states = list(self._adapter_states)
        for state in states:
            usage = state.adapter.get_usage()
            adapter_statuses.append(
                {
                    "eviction_policy": state.eviction_config.eviction_policy,
                    "trigger_watermark": state.eviction_config.trigger_watermark,
                    "eviction_ratio": state.eviction_config.eviction_ratio,
                    "current_usage": usage.usage_fraction,
                    "total_bytes_used": usage.total_bytes_used,
                    "total_capacity_bytes": usage.total_capacity_bytes,
                    "num_cache_salt_buckets": len(usage.bytes_by_cache_salt),
                }
            )
        return {
            "is_healthy": self._thread.is_alive(),
            "thread_alive": self._thread.is_alive(),
            "adapters": adapter_statuses,
        }

    def _eviction_loop(self):
        while not self._stop_flag.is_set():
            time.sleep(1)
            # Hold the lock across the whole pass so remove_adapter_state
            # cannot detach (and the caller close) an adapter while we are
            # calling into it.
            with self._states_lock:
                for state in self._adapter_states:
                    try:
                        self._check_and_evict(state)
                    except Exception:
                        # One failing pass must not end eviction for every
                        # adapter for the rest of the process.
                        logger.exception(
                            "L2 eviction pass failed for adapter %d; retrying",
                            state.adapter_id,
                        )

    def _check_and_evict(self, state: L2AdapterEvictionState):
        if state.eviction_policy.support_isolation and self._quota_manager is not None:
            self._check_and_evict_by_cache_salt(state)
        else:
            self._check_and_evict_global(state)

    def _check_and_evict_global(self, state: L2AdapterEvictionState):
        """Aggregate-usage eviction (``LRU`` / ``noop``)."""
        watermark = state.eviction_config.trigger_watermark
        eviction_ratio = state.eviction_config.eviction_ratio

        # ``usage_fraction == -1`` means the adapter doesn't support
        # usage-based eviction (no max_capacity_bytes declared), so we
        # do not trigger eviction. Adapters with ``supports_global_eviction ==
        # False`` should already have been filtered out at construction
        # time in ``StorageManager``; this check is a defensive belt.
        current_usage = state.adapter.get_usage().usage_fraction
        if current_usage < 0 or current_usage < watermark:
            logger.debug(
                "L2 usage %.2f below watermark %.2f; skipping eviction.",
                current_usage,
                watermark,
            )
            return

        logger.info(
            "L2 usage %.2f above watermark %.2f; triggering eviction.",
            current_usage,
            watermark,
        )
        if self._evict_superseded(state):
            current_usage = state.adapter.get_usage().usage_fraction
            if current_usage < watermark:
                return
        actions = state.eviction_policy.get_eviction_actions(eviction_ratio)
        for action in actions:
            self._execute_eviction_action(state.adapter, action)

    def _check_and_evict_by_cache_salt(self, state: L2AdapterEvictionState):
        """Per-``cache_salt`` eviction driven by :class:`QuotaManager`.

        For every salt with non-zero bytes, compare its usage against
        ``watermark * quota``. Salts over threshold get eviction scoped
        to their own LRU list. Salts with no quota registered have an
        effective limit of ``0`` and are therefore always over budget,
        so they get a full eviction (``effective_ratio=1.0``) — this
        enforces the allowlist rule: only registered salts retain data.

        Per-destination keys are batched across all over-budget salts
        before invoking the adapter — one ``adapter.delete(...)`` call
        per destination instead of one per (salt, destination) pair.
        Adapters with non-trivial per-call overhead (NIXL handle setup,
        FS sync, etc.) see this as a real win when many salts go over
        budget in the same cycle.
        """
        assert self._quota_manager is not None
        watermark = state.eviction_config.trigger_watermark
        eviction_ratio = state.eviction_config.eviction_ratio
        usage = state.adapter.get_usage()

        # destination -> accumulated keys across all over-budget salts.
        pending: dict[EvictionDestination, list[ObjectKey]] = {}

        for cache_salt, user_bytes in usage.bytes_by_cache_salt.items():
            if user_bytes <= 0:
                continue
            limit = self._quota_manager.get_limit_bytes(cache_salt)
            # Trigger on ``>=`` to match the global branch's ``usage <
            # watermark`` short-circuit. Salts with no quota (limit=0)
            # always land here because ``user_bytes > 0 >= 0``.
            if user_bytes < watermark * limit:
                continue

            # Unregistered / zero-quota salts: wipe everything.
            # Registered salts: evict the configured ratio of their list.
            effective_ratio = 1.0 if limit == 0 else eviction_ratio
            logger.info(
                "cache_salt=%r over quota (bytes=%d, limit=%d, "
                "watermark=%.2f); evicting ratio=%.2f.",
                cache_salt,
                user_bytes,
                limit,
                watermark,
                effective_ratio,
            )
            actions = state.eviction_policy.get_eviction_actions(
                effective_ratio, cache_salt=cache_salt
            )
            for action in actions:
                pending.setdefault(action.destination, []).extend(action.keys)

        for destination, keys in pending.items():
            self._execute_eviction_action(
                state.adapter,
                EvictionAction(keys=keys, destination=destination),
            )

    def _execute_eviction_action(
        self, adapter: L2AdapterInterface, action: EvictionAction
    ):
        def delete(keys: list[ObjectKey]) -> None:
            if self._retention is None:
                adapter.delete(keys)
                return
            adapter_id = next(
                state.adapter_id
                for state in self._adapter_states
                if state.adapter is adapter
            )
            self._retention.evict(keys, adapter_id, adapter.delete)

        if action.destination == EvictionDestination.DISCARD:
            delete(action.keys)
        else:
            logger.error("Unsupported eviction destination: %s", action.destination)
            logger.error("Treating it as DISCARD.")
            delete(action.keys)

        if action.keys:
            get_event_bus().publish(
                Event(
                    event_type=EventType.L2_KEYS_EVICTED,
                    metadata={
                        "key_count": len(action.keys),
                        "key_count_per_salt": Counter(
                            k.cache_salt for k in action.keys
                        ),
                    },
                )
            )
