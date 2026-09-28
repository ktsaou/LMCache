# SPDX-License-Identifier: Apache-2.0
"""
Distributed multi-tier storage manager for MP mode
"""

# Standard
from contextlib import contextmanager
from dataclasses import replace
from typing import Iterator, Literal, Optional
import threading
import time

# Third Party
from opentelemetry import metrics

# First Party
from lmcache.lmcache_native import Bitmap, PeriodicEventNotifier
from lmcache.logging import init_logger
from lmcache.v1.distributed.admission import AdmissionWaitResult
from lmcache.v1.distributed.api import (
    CapacitySnapshot,
    MemoryLayoutDesc,
    ModuleMemoryCapacity,
    ObjectKey,
    PrefetchHandle,
    PrefetchMode,
    PrefetchRequestSpec,
    Tier,
    TrimPolicy,
    is_recurrent_checkpoint_key,
)
from lmcache.v1.distributed.bitmap_ops import fold_unfold_ranked
from lmcache.v1.distributed.checkpoint_retention import CheckpointRetention
from lmcache.v1.distributed.config import (
    EvictionConfig,
    StorageManagerConfig,
    get_configured_capacity_bytes,
)
from lmcache.v1.distributed.error import L1Error, strerror
from lmcache.v1.distributed.internal_api import L1MemoryDesc, L2AdapterListener
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.l2_adapters import create_l2_adapter
from lmcache.v1.distributed.l2_adapters.base import AdapterUsage, L2AdapterInterface
from lmcache.v1.distributed.l2_adapters.config import L2AdapterConfigBase
from lmcache.v1.distributed.l2_adapters.reconfiguration import (
    L2ReconfigurableAdapter,
    L2ReconfigureError,
)
from lmcache.v1.distributed.l2_adapters.serde_wrapper import SerdeL2AdapterWrapper
from lmcache.v1.distributed.quota_manager import QuotaManager
from lmcache.v1.distributed.serde import create_serde_processor
from lmcache.v1.distributed.storage_controllers import (
    L1EvictionController,
    L2AdapterEvictionState,
    L2EvictionController,
    PrefetchController,
    StoreController,
)
from lmcache.v1.distributed.storage_controllers.prefetch_policy import (
    create_prefetch_policy,
)
from lmcache.v1.distributed.storage_controllers.store_policy import (
    AdapterDescriptor,
    create_store_policy,
)
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.mp_observability.errors import LMCacheTimeoutError
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import get_event_bus
from lmcache.v1.mp_observability.otel_init import register_gauge
from lmcache.v1.mp_observability.trace.decorator import (
    enable_tracing,
    is_tracing_enabled,
    publish_call_event,
)
from lmcache.v1.platform import HAS_EVENTFD

logger = init_logger(__name__)

# Seconds between progress lines of the shutdown checkpoint flush.
_FLUSH_PROGRESS_SECONDS = 2.0


class StorageManager:
    def __init__(self, config: StorageManagerConfig):
        self._eviction_config = config.eviction_config
        self._l1_manager = L1Manager(config.l1_manager_config)
        self._store_admission_timeout_seconds = config.store_admission_timeout_seconds
        self._admission_stats_lock = threading.Lock()
        self._admission_stats = {
            "waits": 0,
            "retries": 0,
            "successes_after_eviction": 0,
            "exhausted_timeouts": 0,
        }
        admission_meter = metrics.get_meter("lmcache.l1.admission")
        self._admission_waits = admission_meter.create_counter(
            "lmcache_mp.l1_admission_waits",
            description="Atomic L1 stores that waited for eviction capacity",
        )
        self._admission_retries = admission_meter.create_counter(
            "lmcache_mp.l1_admission_retries",
            description="Atomic L1 reservation attempts retried after waiting",
        )
        self._admission_successes = admission_meter.create_counter(
            "lmcache_mp.l1_admission_successes_after_eviction",
            description="Atomic L1 stores admitted after eviction backpressure",
        )
        self._admission_timeouts = admission_meter.create_counter(
            "lmcache_mp.l1_admission_exhausted_timeouts",
            description="Atomic L1 stores skipped after admission deadline",
        )
        # Retained for the L1 half of the capacity report; L1's configured
        # size is a pure function of it.
        self._l1_config = config.l1_manager_config
        self._event_bus = get_event_bus()

        # Checkpoint supersession and write-on-evict state, shared by the
        # eviction controllers, the store controller and checkpoint RPCs.
        store_policy = create_store_policy(config.store_policy)
        self._checkpoint_shutdown_flush_seconds = (
            config.checkpoint_shutdown_flush_seconds
        )
        # Monotonic deadline of the shutdown drain and flush; 0 until started.
        self._shutdown_deadline = 0.0
        self._checkpoint_retention = CheckpointRetention(
            write_on_evict=store_policy.writes_checkpoints_on_evict(),
            persist_timeout=config.checkpoint_write_timeout_seconds,
            supersede_grace_seconds=config.checkpoint_supersede_grace_seconds,
        )
        self._checkpoint_retention.set_l1_lookup(self._l1_manager.get_present_keys)
        self._checkpoint_retention.set_l2_absence(self._absent_from_l2)

        # L1 eviction controller
        self._eviction_controller = L1EvictionController(
            l1_manager=self._l1_manager,
            eviction_config=config.eviction_config,
        )
        self._eviction_controller.set_checkpoint_retention(self._checkpoint_retention)
        self._eviction_controller.start()

        # L2 adapters and store controller. When an adapter config carries
        # a ``serde_config``, the adapter is wrapped with
        # ``SerdeL2AdapterWrapper`` so controllers see a plain L2 adapter
        # and serde is transparent.
        self._l1_memory_desc = self._l1_manager.get_l1_memory_desc()
        self._next_adapter_id = 0
        # Serializes add_l2_adapter / delete_l2_adapter against each other.
        self._lifecycle_lock = threading.Lock()
        # Guards the _l2_adapters and _adapter_descriptors dicts.
        self._adapters_lock = threading.Lock()
        self._registered_l2_listeners: list[L2AdapterListener] = []
        self._l2_adapters: dict[int, L2AdapterInterface] = {}
        self._adapter_descriptors: dict[int, AdapterDescriptor] = {}
        for ac in config.l2_adapter_config.adapters:
            adapter_id, adapter, descriptor = self._build_l2_adapter(ac)
            self._l2_adapters[adapter_id] = adapter
            self._adapter_descriptors[adapter_id] = descriptor
            self._track_checkpoint_residency(adapter_id, adapter)

        PeriodicEventNotifier.create(
            interval_ms=config.periodic_notifier_interval_ms,
            use_eventfd=HAS_EVENTFD,
        )

        # Per-cache_salt quota registry. Shared across the L2 eviction
        # controller (reads quotas each cycle) and the HTTP quota
        # endpoints (CRUD). Present even when no adapter uses
        # IsolatedLRU so the HTTP layer has a stable ``quota_manager``
        # reference. No explicit cleanup on close — the registry is
        # just a dict protected by a lock and has no OS resources.
        self._quota_manager = QuotaManager()

        # Unified L2 eviction controller for all adapters with eviction
        # config. Aggregate-usage policies (``LRU``, ``noop``) need
        # ``max_capacity_bytes > 0`` to compute a usage fraction;
        # adapters without capacity are skipped for those. Isolated
        # policies (``IsolatedLRU``) operate on per cache_salt byte
        # counts which the base class tracks regardless of capacity,
        # so they are wired up unconditionally.
        l2_eviction_states: list[L2AdapterEvictionState] = []
        for adapter_id, ac in zip(
            self._l2_adapters, config.l2_adapter_config.adapters, strict=True
        ):
            adapter = self._l2_adapters[adapter_id]
            if self._should_enable_l2_eviction(adapter, ac.eviction_config):
                assert ac.eviction_config is not None  # make linter happy
                l2_eviction_states.append(
                    L2AdapterEvictionState(
                        adapter_id=adapter_id,
                        adapter=adapter,
                        eviction_config=ac.eviction_config,
                    )
                )
        self._l2_eviction_controller = L2EvictionController(
            l2_eviction_states, quota_manager=self._quota_manager
        )
        self._l2_eviction_controller.set_checkpoint_retention(
            self._checkpoint_retention
        )
        self._l2_eviction_controller.start()

        # Controllers receive the initial set as ordered lists; they key
        # their own copies by ``descriptor.index`` (== adapter_id) and learn
        # of later changes via add_adapter/request_remove_adapter.
        self._store_controller = StoreController(
            l1_manager=self._l1_manager,
            l2_adapters=list(self._l2_adapters.values()),
            adapter_descriptors=list(self._adapter_descriptors.values()),
            policy=store_policy,
        )
        self._store_controller.start()
        self._checkpoint_retention.set_persist(
            self._store_controller.submit_reused_keys
        )

        # Prefetch controller
        self._prefetch_controller = PrefetchController(
            l1_manager=self._l1_manager,
            l2_adapters=list(self._l2_adapters.values()),
            adapter_descriptors=list(self._adapter_descriptors.values()),
            policy=create_prefetch_policy(config.prefetch_policy),
            max_in_flight=config.prefetch_max_in_flight,
        )
        self._prefetch_controller.start()

        # Make adapters eligible for writeback only after their eviction state
        # and request-serving controllers are fully initialized.
        self._sync_l1_writeback_adapters()
        if self._eviction_config.write_back_on_evict and not (
            self._eviction_controller.has_l2_flush_adapter()
        ):
            logger.warning(
                "write_back_on_evict is enabled but no L2 adapter exposes "
                "store_objects_sync; L1 eviction will fail closed"
            )
        if self._eviction_config.periodic_flush_interval > 0 and not (
            self._eviction_controller.has_periodic_flush_adapter()
        ):
            logger.warning(
                "periodic_flush_interval is enabled but no L2 adapter exposes "
                "a timeout-capable store_objects_sync; periodic backup is idle"
            )
        if (
            self._eviction_config.emergency_evict_for_prefetch
            and self._eviction_config.write_back_on_evict
            and not self._eviction_controller.has_bounded_l2_flush_adapter()
        ):
            logger.warning(
                "emergency_evict_for_prefetch is enabled but no L2 adapter "
                "exposes a timeout-capable store_objects_sync; emergency "
                "eviction is disabled"
            )

        # L2 usage gauge — one observation per adapter, tagged by
        # ``l2_name``.  Parallel to L1Manager's ``l1_memory_usage_bytes``.
        register_gauge(
            "lmcache.l2",
            "lmcache_mp.l2_usage_bytes",
            (
                "Bytes currently held in each L2 adapter, tagged by "
                "``l2_name`` (one observation per adapter)."
            ),
            self.get_l2_usages,
        )
        register_gauge(
            "lmcache.l2",
            "lmcache_mp.checkpoint_retention",
            (
                "Recurrent checkpoint retention counters and sizes, tagged by "
                "``stat``: superseded pages, L1 drops and L2 evictions of "
                "superseded pages, write-on-evict requests, completions and "
                "timeouts, and checkpoint bytes held in L2."
            ),
            self._checkpoint_retention.observations,
        )

    # External APIs for serving engine integration code to call
    @enable_tracing()
    def reserve_write_detailed(
        self,
        keys: list[ObjectKey],
        layout_desc: MemoryLayoutDesc,
        mode: Literal["new", "update", "all"],
    ) -> dict[ObjectKey, tuple[L1Error, MemoryObj | None]]:
        """Reserve objects while preserving each L1 failure reason.

        Args:
            keys (list[ObjectKey]): List of object keys to reserve for writing.
            layout_desc (MemoryLayoutDesc): Description of the memory layout
                for the objects to be reserved.
            mode (Literal["new", "update", "all"]): Reservation mode.
            - "new": Reserve only new objects that do not exist.
            - "update": Reserve only existing objects for update.
            - "all": Reserve all writable objects regardless of existence.

        Returns:
            Per-key ``(L1Error, MemoryObj | None)`` results.
        """
        reserve_result = self._l1_manager.reserve_write(
            keys=keys,
            is_temporary=[False] * len(keys),
            layout_desc=layout_desc,
            mode=mode,
        )

        result = {k: m for k, (e, m) in reserve_result.items() if m is not None}
        successful_keys = list(result.keys())
        failed_keys = [k for k, (e, m) in reserve_result.items() if m is None]
        self._event_bus.publish(
            Event(
                event_type=EventType.SM_WRITE_RESERVED,
                metadata={
                    "succeeded_keys": successful_keys,
                    "failed_keys": failed_keys,
                },
            )
        )

        oom_keys = [
            k for k, (e, _) in reserve_result.items() if e == L1Error.OUT_OF_MEMORY
        ]
        if oom_keys:
            self._event_bus.publish(
                Event(
                    event_type=EventType.L1_ALLOCATION_FAILED,
                    metadata={"during": "l1_store", "keys": oom_keys},
                )
            )

        return reserve_result

    @enable_tracing()
    def reserve_write(
        self,
        keys: list[ObjectKey],
        layout_desc: MemoryLayoutDesc,
        mode: Literal["new", "update", "all"],
    ) -> dict[ObjectKey, MemoryObj]:
        """Reserve writable objects and return successful reservations."""
        reserve_result = self.reserve_write_detailed(keys, layout_desc, mode)
        return {
            key: memory_obj
            for key, (_error, memory_obj) in reserve_result.items()
            if memory_obj is not None
        }

    def get_readable_keys(self, keys: list[ObjectKey]) -> list[ObjectKey]:
        """Return complete readable keys without retaining their read locks."""
        results = self._l1_manager.reserve_read(keys)
        readable = [key for key, (_error, obj) in results.items() if obj is not None]
        if readable:
            self._l1_manager.finish_read(readable)
        return readable

    @property
    def store_admission_timeout_seconds(self) -> float:
        """Configured total deadline for capacity-only store retries."""
        return self._store_admission_timeout_seconds

    def get_capacity_generation(self) -> int:
        """Return the current L1 allocator-change generation."""
        return self._l1_manager.get_capacity_generation()

    def request_immediate_eviction(self) -> None:
        """Wake the L1 eviction loop for a capacity-blocked store."""
        self._eviction_controller.request_immediate_eviction()

    def wait_for_capacity_change(
        self, generation: int, timeout: float
    ) -> tuple[AdmissionWaitResult, int]:
        """Wait for L1 capacity change or storage-manager shutdown."""
        return self._l1_manager.wait_for_capacity_change(generation, timeout)

    def record_admission_wait(self) -> None:
        """Record an atomic store entering capacity backpressure."""
        with self._admission_stats_lock:
            self._admission_stats["waits"] += 1
        self._admission_waits.add(1)

    def record_admission_retry(self) -> None:
        """Record an atomic reservation retry."""
        with self._admission_stats_lock:
            self._admission_stats["retries"] += 1
        self._admission_retries.add(1)

    def record_admission_success_after_eviction(self) -> None:
        """Record admission that succeeded after capacity was released."""
        with self._admission_stats_lock:
            self._admission_stats["successes_after_eviction"] += 1
        self._admission_successes.add(1)

    def record_admission_timeout(self) -> None:
        """Record exhausted capacity backpressure."""
        with self._admission_stats_lock:
            self._admission_stats["exhausted_timeouts"] += 1
        self._admission_timeouts.add(1)

    def get_admission_stats(self) -> dict[str, int]:
        """Return process-level atomic admission counters."""
        with self._admission_stats_lock:
            return dict(self._admission_stats)

    @enable_tracing()
    def finish_write(
        self,
        keys: list[ObjectKey],
    ) -> None:
        """
        Finish writing the objects into the storage manager.

        Args:
            keys (list[ObjectKey]): List of object keys that have been written.
        """
        finish_result = self._l1_manager.finish_write(keys)
        successful_keys = [k for k, e in finish_result.items() if e == L1Error.SUCCESS]
        failed_keys = [k for k, e in finish_result.items() if e != L1Error.SUCCESS]
        self._event_bus.publish(
            Event(
                event_type=EventType.SM_WRITE_FINISHED,
                metadata={
                    "succeeded_keys": successful_keys,
                    "failed_keys": failed_keys,
                },
            )
        )

        # TODO: global key states update

    def abort_write(self, keys: list[ObjectKey]) -> None:
        """Discard newly allocated write reservations after a failed copy."""
        if keys:
            self._l1_manager.abort_write(keys)

    @contextmanager
    def read_prefetched_results(
        self,
        keys: list[ObjectKey],
    ) -> Iterator[list[MemoryObj] | None]:
        """
        Read the memory objects from L1 storage that has been prefetched beforehand.
        Yielding an optional list of memory objects corresponding to the requested
        keys. If any the object is not found in L1, None is yielded.

        Args:
            keys (list[ObjectKey]): List of object keys to reserve for reading.

        Returns:
            Iterator[list[MemoryObj] | None]: An iterator yielding an optional list of
                memory objects corresponding to the requested keys.

        Note:
            If any object is not found in L1 storage, None is yielded. In this case,
            this function will release release the read lock of all successfully read
            memory objects when exiting the context.

            If the caller raised exception during the processing of the yielded memory
            objects, this function will ensure that the read locks will be decreased.
        """
        # Manual TRACE_CALL emission for the context manager.  The
        # ``@enable_tracing`` decorator cannot wrap a ``@contextmanager``
        # generator function (it would publish the call to the wrapper
        # rather than to ``__enter__``).  Emit enter/exit events
        # directly, gated on the tracing flag for zero overhead when
        # disabled.
        if is_tracing_enabled():
            publish_call_event(
                "lmcache.v1.distributed.storage_manager."
                "StorageManager.read_prefetched_results.__enter__",
                {"keys": keys},
            )
        read_results = self._l1_manager.unsafe_read(keys)
        good_keys: list[ObjectKey] = []
        good_objs: list[MemoryObj] = []
        bad_keys: list[ObjectKey] = []
        not_found_keys: list[ObjectKey] = []
        write_locked_keys: list[ObjectKey] = []
        all_good = True
        for k, (e, o) in read_results.items():
            if o is None:
                logger.error(
                    "Failed to read prefetched object %s from L1 storage: %s",
                    k,
                    strerror(e),
                )
                bad_keys.append(k)
                all_good = False
                if e == L1Error.KEY_NOT_EXIST:
                    not_found_keys.append(k)
                elif e == L1Error.KEY_NOT_READABLE:
                    write_locked_keys.append(k)
                continue

            good_keys.append(k)
            good_objs.append(o)

        # L1 read-failure anomaly reporting: unsafe_read is required to be
        # called post-reserve_read, so any failure here is a lock/eviction
        # race, not a normal cache miss.
        if not_found_keys:
            self._event_bus.publish(
                Event(
                    event_type=EventType.L1_READ_FAILED,
                    metadata={
                        "during": "l1_retrieve",
                        "reason": "not_found",
                        "keys": not_found_keys,
                    },
                )
            )
        if write_locked_keys:
            self._event_bus.publish(
                Event(
                    event_type=EventType.L1_READ_FAILED,
                    metadata={
                        "during": "l1_retrieve",
                        "reason": "write_locked",
                        "keys": write_locked_keys,
                    },
                )
            )

        successfully_yielded = False

        try:
            yield good_objs if all_good else None
            successfully_yielded = True
        except Exception:
            logger.exception(
                "Exception occurred while processing read prefetched results",
            )
            raise
        finally:
            # Decrease the read lock for all successfully read memory objects
            # if None is yielded or exception occurs during caller's processing
            if not all_good or not successfully_yielded:
                self._l1_manager.finish_read(good_keys)
                self._event_bus.publish(
                    Event(
                        event_type=EventType.SM_READ_PREFETCHED_FINISHED,
                        metadata={
                            "succeeded_keys": good_keys,
                            "failed_keys": bad_keys,
                        },
                    )
                )
            if is_tracing_enabled():
                publish_call_event(
                    "lmcache.v1.distributed.storage_manager."
                    "StorageManager.read_prefetched_results.__exit__",
                    {"keys": keys},
                )

    @enable_tracing()
    def finish_read_prefetched(
        self,
        keys: list[ObjectKey],
        read_locks: int = 1,
    ) -> None:
        """Finish reading prefetched objects.

        Args:
            keys: Object keys that have been read.
            read_locks: Read locks to release per key (the whole
                reservation when releasing a lookup's locks).
        """
        finish_result = self._l1_manager.finish_read(keys, read_locks=read_locks)
        successful_keys = [k for k, e in finish_result.items() if e == L1Error.SUCCESS]
        failed_keys = [k for k, e in finish_result.items() if e != L1Error.SUCCESS]
        self._event_bus.publish(
            Event(
                event_type=EventType.SM_READ_PREFETCHED_FINISHED,
                metadata={
                    "succeeded_keys": successful_keys,
                    "failed_keys": failed_keys,
                },
            )
        )

    def notify_keys_reused(self, keys: list[ObjectKey]) -> None:
        """Report objects that a completed restore just read from L1.

        With a store policy that stores keys only after reuse (such as
        ``checkpoint_on_reuse``), the first report of a key not yet in L2
        stores it there. Write-through policies ignore the report. The call
        is non-blocking; the store controller's loop does the work.

        Args:
            keys: Keys read by a restore whose consumer copy has completed.
                Call this before releasing the restore's read locks.
        """
        if self._checkpoint_retention.write_on_evict:
            # Write-on-evict policies store checkpoint pages when L1 evicts
            # them, not because a restore read them.
            return
        self._store_controller.submit_reused_keys(keys)

    @property
    def checkpoint_retention(self) -> CheckpointRetention:
        """Checkpoint supersession and write-on-evict state."""
        return self._checkpoint_retention

    def _track_checkpoint_residency(
        self, adapter_id: int, adapter: L2AdapterInterface
    ) -> None:
        """Record which checkpoint pages ``adapter`` holds, now and later."""
        retention = self._checkpoint_retention
        listener = retention.listener_for(adapter_id)
        adapter.register_listener(listener)
        try:
            inventory = adapter.get_existing_key_sizes()
        except Exception:
            logger.exception(
                "Could not read the checkpoint inventory of L2 adapter %d",
                adapter_id,
            )
            return
        if inventory:
            retention.record_l2_present(
                adapter_id, list(inventory), list(inventory.values())
            )

    def _absent_from_l2(self, keys: list[ObjectKey]) -> list[ObjectKey]:
        """Return the keys every L2 adapter confirms it does not hold."""
        absent = keys
        for _adapter_id, _descriptor, adapter in self._snapshot_adapters():
            if not absent:
                break
            try:
                confirmed = set(adapter.absent_keys(absent))
            except Exception:
                logger.exception("L2 absence check failed; assuming keys present")
                return []
            absent = [key for key in absent if key in confirmed]
        return absent

    def start_shutdown(self) -> float:
        """Start the clean-shutdown budget for checkpoint work.

        The budget (``checkpoint_shutdown_flush_seconds``) covers waiting for
        checkpoint stores still in flight and then writing checkpoint pages
        that are only in L1 to L2. Idempotent.

        Returns:
            The monotonic deadline of the budget.
        """
        if not self._shutdown_deadline:
            self._shutdown_deadline = (
                time.monotonic() + self._checkpoint_shutdown_flush_seconds
            )
        return self._shutdown_deadline

    def _flush_current_checkpoints(self, deadline: float) -> None:
        """Write checkpoint pages still only in L1 to L2 on shutdown.

        Current pages go first, then superseded pages still in their grace
        period, then stale superseded pages, all within the same deadline:
        while serving, superseded pages leave L1 without a write to spare
        flash, but a shutdown writes each at most once, and their manifests
        stay listed, so a request that continues an older line after the
        restart (a branch, a resumed turn, a side request that extended the
        conversation) restores instead of finding its pages gone. Progress is
        logged every few seconds, so a flush cut short by SIGKILL still says
        how far it got.

        Args:
            deadline: Monotonic time at which the flush gives up.
        """
        retention = self._checkpoint_retention
        if (
            not retention.write_on_evict
            or not self._checkpoint_shutdown_flush_seconds
            or not self._has_l2_adapters()
        ):
            return
        count = self._l1_manager.num_objects()
        keys, _ = self._l1_manager.get_evictable_keys(limit=count, scan_limit=count)
        unwritten = [
            key
            for key in keys
            if is_recurrent_checkpoint_key(key) and not retention.is_l2_resident(key)
        ]
        superseded = set(retention.superseded_keys())
        current = [key for key in unwritten if key not in superseded]
        stale = set(retention.take_droppable_superseded())
        in_grace = [key for key in unwritten if key in superseded and key not in stale]
        old = [key for key in unwritten if key in superseded and key in stale]
        pending = current + in_grace + old
        if not pending:
            return
        start = time.monotonic()
        logger.info(
            "Writing %d current and %d superseded checkpoint pages to L2 before "
            "shutdown (%.1f s left)",
            len(current),
            len(in_grace) + len(old),
            max(0.0, deadline - start),
        )
        if retention.request_persist(pending) < len(pending):
            # Writes requested while serving may have failed; the store
            # controller skips the ones still in flight.
            self._store_controller.submit_reused_keys(pending)
        last_report = start
        while True:
            pending = [key for key in pending if not retention.is_l2_resident(key)]
            now = time.monotonic()
            if not pending or now >= deadline:
                break
            if now - last_report >= _FLUSH_PROGRESS_SECONDS:
                last_report = now
                logger.info(
                    "Shutdown checkpoint flush: %d pages left after %.1f s",
                    len(pending),
                    now - start,
                )
            time.sleep(0.05)
        if pending:
            left = set(pending)
            logger.warning(
                "Shutdown checkpoint flush left %d pages without an L2 copy after "
                "%.1f s (%d current, %d superseded); after the restart, lookups "
                "retire the checkpoints that need them and fall back to shorter ones",
                len(pending),
                time.monotonic() - start,
                sum(1 for key in current if key in left),
                sum(1 for key in in_grace + old if key in left),
            )
        else:
            logger.info(
                "Shutdown checkpoint flush completed in %.1f s",
                time.monotonic() - start,
            )

    @enable_tracing()
    def submit_prefetch_task(
        self,
        spec: PrefetchRequestSpec,
        external_request_id: str = "",
        skip_l2: bool = False,
    ) -> PrefetchHandle:
        """Prefetch objects into L1 asynchronously.

        Args:
            spec: The L2-fetch request inputs (see :class:`PrefetchRequestSpec`).
            external_request_id: Caller id for end-to-end log tracing.
            skip_l2: If True, do not load from L2. For ``LOOKUP`` only
                already-resident L1 keys are returned; for ``WARM`` nothing is
                loaded and an empty handle is returned.

        Returns:
            PrefetchHandle to track the task.
        """
        keys = spec.keys

        if spec.mode is PrefetchMode.WARM:
            # Warm path: load all keys, lock none. skip_l2 makes it a no-op.
            prefetch_request_id = -1
            if not skip_l2 and keys and self._l2_adapters:
                prefetch_request_id = self._prefetch_controller.submit_prefetch_request(
                    spec
                )
            return PrefetchHandle(
                prefetch_request_id=prefetch_request_id,
                external_request_id=external_request_id,
                l1_found_indices=(),
                l1_hit_chunks=0,
                total_requested_keys=len(keys),
                submit_time=time.monotonic(),
                l2_orig_indices=(
                    tuple(range(len(keys))) if prefetch_request_id != -1 else ()
                ),
            )

        # NOTE: now we only have L1, so the prefetch is essentially checking how many
        # objects are already in L1, and adding read locks to them.

        l1_read_result = self._l1_manager.reserve_read(
            keys, read_locks=spec.num_kv_readers
        )

        if spec.policy is TrimPolicy.SPARSE:
            # SPARSE: retain a read lock on every L1 hit (not just the leading
            # prefix) and send all L1 misses to L2 as one coalesced request.
            # reserve_read locks only SUCCESS keys, so the found-set already
            # equals the locked set -- nothing to release.
            l1_found_indices: list[int] = []
            succeeded_keys: list[ObjectKey] = []
            sparse_l2_indices: list[int] = []
            remaining_keys: list[ObjectKey] = []
            for i, key in enumerate(keys):
                ent = l1_read_result.get(key)
                if ent is not None and ent[0] == L1Error.SUCCESS and ent[1] is not None:
                    l1_found_indices.append(i)
                    succeeded_keys.append(key)
                else:
                    sparse_l2_indices.append(i)
                    remaining_keys.append(key)

            self._event_bus.publish(
                Event(
                    event_type=EventType.SM_READ_PREFETCHED,
                    metadata={
                        "succeeded_keys": succeeded_keys,
                        "failed_keys": remaining_keys,
                    },
                )
            )

            prefetch_request_id = -1
            if not skip_l2 and remaining_keys and self._has_l2_adapters():
                prefetch_request_id = self._prefetch_controller.submit_prefetch_request(
                    replace(spec, keys=remaining_keys)
                )
            return PrefetchHandle(
                prefetch_request_id=prefetch_request_id,
                external_request_id=external_request_id,
                l1_found_indices=tuple(l1_found_indices),
                l1_hit_chunks=0,
                total_requested_keys=len(keys),
                submit_time=time.monotonic(),
                l2_orig_indices=(
                    tuple(sparse_l2_indices) if prefetch_request_id != -1 else ()
                ),
            )

        # PREFIX: fold the per-(group, chunk, rank) L1 presence into the
        # model-wide hit and the per-object-group retain set (sliding-window
        # aware). All-full-attention reduces to the contiguous leading-ones
        # prefix. Keys past the L1 hit are sent to L2.
        elif spec.policy is TrimPolicy.PREFIX:
            return self._submit_prefix_fold(
                spec,
                l1_read_result,
                external_request_id,
                skip_l2,
            )

        raise ValueError(f"Unsupported trim policy: {spec.policy}")

    def _submit_prefix_fold(
        self,
        spec: PrefetchRequestSpec,
        l1_read_result: dict[ObjectKey, tuple[L1Error, "MemoryObj | None"]],
        external_request_id: str,
        skip_l2: bool,
    ) -> PrefetchHandle:
        """PREFIX path: fold L1 presence, retain in-window keys, submit rest to L2.

        Args:
            spec: The L2-fetch request inputs (see
                :class:`PrefetchRequestSpec`); the dispatcher only routes
                ``PREFIX``-policy specs here.
            l1_read_result: Per-key ``reserve_read`` results from the L1
                probe; SUCCESS entries count as L1-present and stay
                read-locked until the fold releases the out-of-window ones.
            external_request_id: Engine-side request id, for logging/trace.
            skip_l2: When True, serve from L1 only (no L2 prefetch).

        Returns:
            A :class:`PrefetchHandle` carrying the L1 hit (retained indices
            and hit chunks) and the pending L2 prefetch request id (``-1``
            when nothing was submitted to L2).
        """
        keys = spec.keys
        attn_desc = spec.attn_desc
        num_object_groups = attn_desc.num_object_groups
        stride = num_object_groups * attn_desc.world_size
        num_chunks = len(keys) // stride

        l1_presence = Bitmap(len(keys))
        for i, key in enumerate(keys):
            ent = l1_read_result.get(key)
            if ent is not None and ent[0] == L1Error.SUCCESS and ent[1] is not None:
                l1_presence.set(i)

        l1_hit_chunks, retain = fold_unfold_ranked(
            l1_presence,
            num_chunks,
            attn_desc.world_size,
            attn_desc.num_chunks_in_sw,
        )
        retained_indices = retain.get_indices_list()

        released_bitmap = l1_presence & (~retain)
        released = released_bitmap.gather(keys)
        if released:
            self._l1_manager.finish_read(released, read_locks=spec.num_kv_readers)

        # Keys from chunk l1_hit_chunks onwards are candidates for L2.
        l1_key_boundary = l1_hit_chunks * stride
        remaining_keys = keys[l1_key_boundary:]

        self._event_bus.publish(
            Event(
                event_type=EventType.SM_READ_PREFETCHED,
                metadata={
                    "succeeded_keys": retain.gather(keys),
                    "failed_keys": (~retain).gather(keys),
                },
            )
        )

        l1_only = skip_l2 or not self._has_l2_adapters()
        prefetch_request_id = -1
        l2_orig_indices: tuple[int, ...] = ()

        if not l1_only and remaining_keys:
            prefetch_request_id = self._prefetch_controller.submit_prefetch_request(
                replace(spec, keys=remaining_keys)
            )
            l2_orig_indices = tuple(range(l1_key_boundary, len(keys)))

        submit_time = time.monotonic()
        logger.debug(
            "Prefetch request submitted: "
            "%d total keys, %d L1 hit chunks (%d retained keys), "
            "%d remaining for L2 "
            "(external_request_id=%s, "
            "prefetch_request_id=%d)",
            len(keys),
            l1_hit_chunks,
            len(retained_indices),
            len(remaining_keys),
            external_request_id,
            prefetch_request_id,
        )

        return PrefetchHandle(
            prefetch_request_id=prefetch_request_id,
            external_request_id=external_request_id,
            l1_found_indices=tuple(retained_indices),
            l1_hit_chunks=l1_hit_chunks,
            total_requested_keys=len(keys),
            submit_time=submit_time,
            l2_orig_indices=l2_orig_indices,
        )

    def _combine_found(
        self, handle: PrefetchHandle, l2_local: "Bitmap | None"
    ) -> Bitmap:
        """Merge the L1 found indices with an L2 result bitmap into one bitmap
        over the original key positions.

        ``l2_local`` is indexed over the keys submitted to L2 (0-based); its
        set bits are mapped back to original positions via
        ``handle.l2_orig_indices``.
        """
        found = Bitmap(handle.total_requested_keys)
        found.batched_set(handle.l1_found_indices)
        if l2_local is not None:
            # gather maps each L2 set bit i to its original position
            # ``l2_orig_indices[i]``; batched_set drops any position >= size.
            found.batched_set(l2_local.gather(handle.l2_orig_indices))
        return found

    def query_prefetch_lookup_hits(
        self,
        handle: PrefetchHandle,
    ) -> int | None:
        """
        Query the number of prefix-hit chunks for a prefetch task before the
        L2 prefetching is done.

        Args:
            handle (PrefetchHandle): The handle of the lookup task.

        Returns:
            the number of prefix-hit chunks (L1 + L2) if the lookup is done,
            None if it's still in progress or the prefetch task is already done.

        Note:
            This function is designed for the scenario where the caller wants
            to check the L1 prefix hits as soon as possible without waiting for
            the whole prefetch task to be done.
            When the prefetch task is already done and the prefetch task result
            has already been queried by `query_prefetch_status`, this function
            will return None forever for the same prefetch handle.
            Therefore, it's the caller’s responsibility to make sure not calling
            this function after the prefetch task is done.
        """
        if handle.prefetch_request_id == -1:
            return handle.l1_hit_chunks

        l2_r = self._prefetch_controller.query_lookup_result(handle.prefetch_request_id)
        if l2_r is None:
            # Still in progress, or already consumed by query_prefetch_status.
            return None
        # Both l1_hit_chunks and l2_r are chunk-level counts.
        return handle.l1_hit_chunks + l2_r

    def wait_prefetch_status(
        self,
        handle: PrefetchHandle,
        timeout: float,
    ) -> bool:
        """
        Block until the prefetch task for ``handle`` has a result, or timeout.

        L1-only prefetches (``prefetch_request_id == -1``) have no L2 result to
        wait for and return immediately. This lets a caller avoid busy-polling
        query_prefetch_status; the status itself is still retrieved via
        query_prefetch_status afterwards.

        Args:
            handle (PrefetchHandle): The handle of the prefetch task.
            timeout: Maximum number of seconds to wait for the L2 result.

        Returns:
            True if a result is available within the timeout (always True for
            an L1-only prefetch), False if the wait timed out.
        """
        if handle.prefetch_request_id == -1:
            return True
        return self._prefetch_controller.wait_prefetch_result(
            handle.prefetch_request_id, timeout
        )

    def query_prefetch_status(
        self,
        handle: PrefetchHandle,
    ) -> Bitmap | None:
        """
        Query the status of the prefetch task.

        Args:
            handle (PrefetchHandle): The handle of the prefetch task.

        Returns:
            the found-key bitmap (over original positions) if the prefetch is
            done, None if it's still in progress. Derive the prefix hit count
            via ``count_leading_ones``.
        """
        l2_r: Bitmap | None = None
        if handle.prefetch_request_id != -1:
            l2_r = self._prefetch_controller.query_prefetch_result(
                handle.prefetch_request_id
            )
            if l2_r is None:
                return None

        found = self._combine_found(handle, l2_r)
        # popcount (not count_leading_ones) so the log is accurate for
        # non-contiguous policies (SEGMENTED_PREFIX / SPARSE) too.
        total_hits = found.popcount()
        elapsed_ms = (time.monotonic() - handle.submit_time) * 1000

        if total_hits > 0:
            # L1 and L2 sets are disjoint (only L1-misses go to L2).
            l1_hits = len(handle.l1_found_indices)
            l2_hits = l2_r.popcount() if l2_r is not None else 0
            logger.info(
                "Prefetch request completed (L1+L2): "
                "%d/%d retained keys (%d L1, %d L2) in %.1f ms "
                "(external_request_id=%s, prefetch_request_id=%d)",
                total_hits,
                handle.total_requested_keys,
                l1_hits,
                l2_hits,
                elapsed_ms,
                handle.external_request_id,
                handle.prefetch_request_id,
            )
        return found

    def touch_l1_keys(self, keys: list[ObjectKey]):
        """
        Touch the keys in L1 storage, marking the keys
        as accessed(retrieved or stored).

        Args:
            keys (list[ObjectKey]): List of object keys to touch.
        """
        self._l1_manager.touch_keys(keys)

    def touch_keys(self, keys: list[ObjectKey]) -> None:
        """Refresh keys in L1 and L2 eviction order without reading them.

        Use this when a lookup proves data is still wanted but no tier is
        read, for example because the engine already holds it. Each tier
        ignores keys it does not hold. No access event is published and
        store admission is unchanged.

        Args:
            keys: Keys to move to the most recently used position.
        """
        self._eviction_controller.touch_keys(keys)
        self._l2_eviction_controller.touch_keys(keys)

    def delete_l1_keys(
        self, keys: list[ObjectKey], force: bool = False
    ) -> tuple[int, int]:
        """Delete the given keys from L1.

        Args:
            keys (list[ObjectKey]): List of object keys to delete.
            force (bool): When True, delete even read/write-locked keys; else
                skip them.

        Returns:
            tuple[int, int]: ``(deleted, skipped)`` -- the number of keys removed
                and the number refused because they were locked (non-force only).
                Missing keys are a no-op, so the operation is idempotent.
        """
        self._checkpoint_retention.retire_before_l1_delete(keys)
        results = self._l1_manager.delete(keys, force=force)
        deleted = sum(1 for err in results.values() if err == L1Error.SUCCESS)
        skipped = sum(1 for err in results.values() if err == L1Error.KEY_IS_LOCKED)
        return deleted, skipped

    def unsafe_read(
        self, keys: list[ObjectKey]
    ) -> tuple[list[ObjectKey], list[MemoryObj]]:
        """Read already read-locked objects without acquiring new read locks."""
        read_results = self._l1_manager.unsafe_read(keys)
        good_keys: list[ObjectKey] = []
        good_objs: list[MemoryObj] = []
        for key in keys:
            err, obj = read_results.get(key, (L1Error.KEY_NOT_EXIST, None))
            if err != L1Error.SUCCESS or obj is None:
                continue
            good_keys.append(key)
            good_objs.append(obj)
        return good_keys, good_objs

    @property
    def quota_manager(self) -> QuotaManager:
        """Per-cache_salt quota registry.

        Exposed so the HTTP layer can serve CRUD endpoints without
        reaching into private state. Always non-``None`` — the
        storage manager creates the registry at construction time.
        """
        return self._quota_manager

    @property
    def l1_memory_desc(self) -> L1MemoryDesc:
        """Descriptor of the L1 memory buffer backing this storage manager."""
        return self._l1_memory_desc

    def get_l2_usages(
        self,
    ) -> list[tuple[int | float, dict[str, object]]]:
        """Per-adapter L2 usage in OTel-observation shape.

        Backing data for the ``lmcache_mp.l2_usage_bytes`` observable
        gauge.  One entry per configured adapter.

        Returns:
            A list of ``(total_bytes_used, {"l2_name": <type_name>})``
            tuples — empty when no L2 adapters are configured.  Adapters
            whose ``get_usage()`` raises are skipped (the gauge prefers
            silence over a poison observation).
        """
        return [
            (int(usage.total_bytes_used), {"l2_name": type_name})
            for type_name, usages in self.get_l2_usages_by_type().items()
            for usage in usages
        ]

    def get_l2_usages_by_type(self) -> dict[str, list[AdapterUsage]]:
        """Per-adapter usage snapshots grouped by adapter type name.

        Backing data for usage telemetry's per-connector presence and
        occupancy reporting and for the :meth:`get_l2_usages` gauge.
        Adapters whose ``get_usage()`` raises are skipped.

        Returns:
            Mapping from adapter type name (e.g. ``"dax"``) to one usage
            snapshot per active adapter of that type; empty when no L2
            adapters are configured.
        """
        out_by_type: dict[str, list[AdapterUsage]] = {}
        for _adapter_id, desc, adapter in self._snapshot_adapters():
            try:
                usage = adapter.get_usage()
            except Exception:
                logger.exception(
                    "L2 adapter %s get_usage() failed; skipping in usage snapshot",
                    desc.type_name,
                )
                continue
            out_by_type.setdefault(desc.type_name, []).append(usage)
        return out_by_type

    def publish_capacity(self) -> None:
        """Announce the current capacity topology on the event bus.

        Called after every coordinator registration, so a restarted
        coordinator relearns this server's capacities even if nothing is
        ever reconfigured. Later changes announce themselves.
        """
        self._publish_capacity_changed()

    def _build_capacities(self) -> list[ModuleMemoryCapacity]:
        """Assemble one capacity entry per memory compartment.

        Returns:
            L1 per backing medium, then one entry per L2 adapter.
        """
        capacities = [
            ModuleMemoryCapacity(
                tier=Tier.L1,
                backend=backend.value,
                capacity_bytes=configured,
                shared=False,
            )
            for backend, configured in get_configured_capacity_bytes(
                self._l1_config
            ).items()
        ]
        for _adapter_id, desc, adapter in self._snapshot_adapters():
            try:
                usage = adapter.get_usage()
            except Exception:
                logger.exception(
                    "L2 adapter %s get_usage() failed; omitting from the "
                    "capacity report",
                    desc.type_name,
                )
                continue
            capacities.append(
                ModuleMemoryCapacity(
                    tier=Tier.L2,
                    backend=desc.type_name,
                    capacity_bytes=int(usage.total_capacity_bytes),
                    shared=bool(desc.config.shared),
                )
            )
        return capacities

    def get_l1_usage(self) -> tuple[int, int]:
        """Current occupancy of the L1 memory pool.

        Backing data for usage telemetry's L1 occupancy reporting.

        Returns:
            Tuple of ``(used_bytes, total_bytes)``.
        """
        return self._l1_manager.get_memory_usage()

    def get_usage_bytes_by_cache_salt(self) -> dict[str, int]:
        """Aggregate ``cache_salt`` byte usage across every L2 adapter.

        Used by the HTTP quota endpoints to report ``current_usage_gb``
        alongside the configured limit. Aggregation is a simple sum:
        each adapter tracks the same salt independently so the totals
        are additive.
        """
        totals: dict[str, int] = {}
        for _adapter_id, _desc, adapter in self._snapshot_adapters():
            snap = adapter.get_usage().bytes_by_cache_salt
            for salt, used in snap.items():
                totals[salt] = totals.get(salt, 0) + used
        return totals

    # L2 APIs
    def get_l2_adapter_reconfigure_status(self) -> dict:
        """Return status for all runtime-reconfigurable L2 adapters.

        Returns:
            JSON-serializable status. If no reconfigurable adapter is configured,
            ``enabled`` is ``False`` and the adapter list is empty.
        """
        type_names = {
            adapter_id: desc.type_name
            for adapter_id, desc, _ in self._snapshot_adapters()
        }
        adapters = []
        for adapter_index, (
            l2_adapter_index,
            adapter,
        ) in enumerate(self._list_reconfigurable_l2_adapters()):
            status = dict(adapter.reconfigure_status())
            if l2_adapter_index in type_names:
                status["backend"] = type_names[l2_adapter_index]
            status["adapter_index"] = adapter_index
            status["l2_adapter_index"] = l2_adapter_index
            adapters.append(status)

        return {
            "enabled": bool(adapters),
            "num_adapters": len(adapters),
            "adapters": adapters,
        }

    def reconfigurable_l2_backends(self) -> set[str]:
        """Return the ``type_name`` of every L2 adapter that supports runtime
        reconfiguration.

        Returns:
            The set of reconfigurable adapter ``type_name`` strings (empty when
            none are reconfigurable). The ``{backend}`` path parameter the
            ``/reconfigure`` routes expect is the adapter's ``type_name``.
        """
        return {
            desc.type_name
            for _adapter_id, desc, adapter in self._snapshot_adapters()
            if self._unwrap_reconfigurable_l2_adapter(adapter) is not None
        }

    def _publish_capacity_changed(self) -> None:
        """Announce the current capacity topology on the event bus.

        Lock-free. ``_build_capacities`` guards its own reads
        (``_snapshot_adapters`` takes ``_adapters_lock``), and ordering is
        not this class's problem: the cache-event subscriber numbers
        declarations as it emits them, on the one bus drain thread, so a
        number cannot come apart from the topology it labels. Callers here
        are concurrent -- registration publishes from the event loop while a
        worker may be adding an adapter -- which is exactly why the counter
        does not live here.

        The event carries the whole topology, not a delta, so a dropped one
        is repaired by the next rather than leaving the coordinator
        permanently wrong.
        """
        self._event_bus.publish(
            Event(
                event_type=EventType.SM_CAPACITY_CHANGED,
                metadata={
                    "snapshot": CapacitySnapshot(
                        modules=tuple(self._build_capacities())
                    )
                },
            )
        )

    def reconfigure_l2_adapter(
        self,
        adapter_index: int,
        operation: str,
        payload: dict[str, object],
    ) -> dict:
        """Route a runtime reconfiguration request to one L2 adapter.

        Args:
            adapter_index: Zero-based reconfigurable-adapter index.
            operation: Adapter-specific operation name.
            payload: Adapter-specific operation payload.

        Returns:
            JSON-serializable operation result.
        """
        adapter = self._get_reconfigurable_l2_adapter(adapter_index)
        result = adapter.reconfigure(operation, payload)
        result["adapter_index"] = adapter_index
        # Lock-free: reconfigure did not serialize against adapter
        # add/delete before, and publishing is no reason to start.
        self._publish_capacity_changed()
        return result

    def add_l2_adapter(self, config: L2AdapterConfigBase) -> int:
        """Blocking function to add a new L2 adapter at runtime. Thread-safe.

        Args:
            config: The adapter configuration.

        Returns:
            The stable id assigned to the new adapter.
        """
        with self._lifecycle_lock:
            adapter_id, adapter, descriptor = self._build_l2_adapter(config)
            for listener in self._registered_l2_listeners:
                adapter.register_listener(listener)
            self._track_checkpoint_residency(adapter_id, adapter)
            with self._adapters_lock:
                self._l2_adapters[adapter_id] = adapter
                self._adapter_descriptors[adapter_id] = descriptor
            if self._should_enable_l2_eviction(adapter, config.eviction_config):
                assert config.eviction_config is not None  # make linter happy
                self._l2_eviction_controller.add_adapter_state(
                    L2AdapterEvictionState(
                        adapter_id=adapter_id,
                        adapter=adapter,
                        eviction_config=config.eviction_config,
                    )
                )
            # Expose the adapter to request-serving controllers only after its
            # persistent inventory has seeded eviction state. This makes runtime
            # add follow the same quiescent startup contract as initial adapters.
            self._store_controller.add_adapter(adapter_id, adapter, descriptor)
            self._prefetch_controller.add_adapter(adapter_id, adapter, descriptor)
            self._sync_l1_writeback_adapters()
            logger.info("Added L2 adapter %d (%s)", adapter_id, descriptor.type_name)
            self._publish_capacity_changed()
            return adapter_id

    def delete_l2_adapter(self, adapter_id: int, timeout: float = 30.0) -> None:
        """Blocking function to drain the L2 adapter gracefully at runtime.
        Thread-safe.

        Stops routing new stores/prefetches to the adapter, waits for its
        in-flight work to finish, removes it from the controllers, and
        closes it.

        Args:
            adapter_id: Stable id of the adapter to remove.
            timeout: Maximum seconds to wait for in-flight work to drain.

        Raises:
            ValueError: If no adapter with ``adapter_id`` is active.
            TimeoutError: If draining did not complete within ``timeout``;
                the adapter is left active (draining) so the caller can
                retry.
        """
        with self._lifecycle_lock:
            if adapter_id not in self._l2_adapters:
                raise ValueError(f"No L2 adapter with id {adapter_id}")

            deadline = time.monotonic() + timeout
            store_done = self._store_controller.request_remove_adapter(adapter_id)
            prefetch_done = self._prefetch_controller.request_remove_adapter(adapter_id)
            if not store_done.wait(timeout=max(0.0, deadline - time.monotonic())):
                raise LMCacheTimeoutError(
                    f"Timed out draining adapter {adapter_id} from store controller"
                )
            if not prefetch_done.wait(timeout=max(0.0, deadline - time.monotonic())):
                raise LMCacheTimeoutError(
                    f"Timed out draining adapter {adapter_id} from prefetch controller"
                )

            # Stop new writebacks and wait for any active synchronous flush
            # before detaching eviction state or closing native resources.
            self._eviction_controller.remove_l2_adapter(adapter_id)
            self._l2_eviction_controller.remove_adapter_state(adapter_id)
            self._checkpoint_retention.forget_adapter(adapter_id)
            with self._adapters_lock:
                adapter = self._l2_adapters.pop(adapter_id)
                self._adapter_descriptors.pop(adapter_id, None)
            self._sync_l1_writeback_adapters()
            adapter.close()
            logger.info("Deleted L2 adapter %d", adapter_id)
            self._publish_capacity_changed()

    def l2_adapters(self) -> list[tuple[AdapterDescriptor, L2AdapterInterface]]:
        """Return all active L2 adapters paired with descriptors, in
        ascending adapter-id order (== configuration order for the initial
        set, then runtime-added adapters). The list is empty when no L2 is
        configured.

        Do not cache the returned pairs — ``reconfigure_l2_adapter``,
        ``add_l2_adapter``, and ``delete_l2_adapter`` may change the set at
        runtime.
        """
        return [
            (desc, adapter) for _adapter_id, desc, adapter in self._snapshot_adapters()
        ]

    # Management APIs
    def clear(self, force: bool = False):
        """
        Clear data in the storage manager.

        Args:
            force: If True, clear ALL objects including locked ones.
                This may corrupt in-flight store/prefetch operations.
                If False (default), only clear unlocked objects, keeping
                write-locked and read-locked objects intact.
        """
        retention = self._checkpoint_retention
        retention.retire_before_l1_delete(retention.superseded_keys())
        self._l1_manager.clear(force=force)

    def close(self):
        """
        Close the storage manager and release all resources.

        With write-on-evict checkpoint storage, checkpoint pages still only
        in L1 are first written to L2 within what is left of the budget that
        :meth:`start_shutdown` started (it starts here if nothing did).
        """
        try:
            self._flush_current_checkpoints(self.start_shutdown())
        except Exception:
            logger.exception("Shutdown checkpoint flush failed")
        self._l1_manager.begin_shutdown()
        self._prefetch_controller.stop()
        self._store_controller.stop()
        self._eviction_controller.stop()
        self._l2_eviction_controller.stop()

        PeriodicEventNotifier.shutdown()

        for adapter in self._l2_adapters.values():
            adapter.close()

        self._l1_manager.close()

    def report_status(self) -> dict:
        """Return a status dict aggregating all sub-component statuses."""
        l1 = self._l1_manager.report_status()
        store = self._store_controller.report_status()
        prefetch = self._prefetch_controller.report_status()
        l1_eviction = self._eviction_controller.report_status()
        l2_eviction = self._l2_eviction_controller.report_status()
        adapters = [a.report_status() for _id, _desc, a in self._snapshot_adapters()]
        children = [l1, store, prefetch, l1_eviction, l2_eviction] + adapters
        return {
            "is_healthy": all(c["is_healthy"] for c in children),
            "l1_manager": l1,
            "store_controller": store,
            "prefetch_controller": prefetch,
            "l1_eviction_controller": l1_eviction,
            "l2_eviction_controller": l2_eviction,
            "checkpoint_retention": self._checkpoint_retention.report_status(),
            "l2_adapters": adapters,
            "num_l2_adapters": len(adapters),
            "store_admission": self.get_admission_stats(),
        }

    def register_l2_listener(self, listener: L2AdapterListener) -> None:
        """Register a listener on all current and future L2 adapters.

        The listener is recorded so that adapters added later via
        :meth:`add_l2_adapter` receive it too.

        Args:
            listener: The listener to register.
        """
        with self._lifecycle_lock:
            self._registered_l2_listeners.append(listener)
            for adapter in self._l2_adapters.values():
                adapter.register_listener(listener)

    # Functions for debugging and testing
    def memcheck(self) -> bool:
        """
        Perform memory check for all storage tiers.

        Returns:
            True if memory is consistent, False otherwise.
        """
        return self._l1_manager.memcheck()

    def _sync_l1_writeback_adapters(self) -> None:
        """Publish the active adapter set to the opt-in L1 writeback path."""
        if not (
            self._eviction_config.write_back_on_evict
            or self._eviction_config.periodic_flush_interval > 0
            or self._eviction_config.emergency_evict_for_prefetch
        ):
            return
        with self._adapters_lock:
            adapters = dict(self._l2_adapters)
        self._eviction_controller.set_l2_adapters(adapters)
        if self._eviction_config.emergency_evict_for_prefetch:
            self._prefetch_controller.set_l1_eviction_controller(
                self._eviction_controller
                if not self._eviction_config.write_back_on_evict
                or self._eviction_controller.has_bounded_l2_flush_adapter()
                else None
            )

    def _snapshot_adapters(
        self,
    ) -> list[tuple[int, AdapterDescriptor, L2AdapterInterface]]:
        """Snapshot the active adapters under the lock, in ascending
        adapter-id order. Iterate this instead of the live dicts so a
        concurrent add/delete cannot change them mid-iteration.

        Returns:
            A list of ``(adapter_id, descriptor, adapter)`` tuples.
        """
        with self._adapters_lock:
            return [
                (adapter_id, self._adapter_descriptors[adapter_id], adapter)
                for adapter_id, adapter in sorted(self._l2_adapters.items())
            ]

    def _has_l2_adapters(self) -> bool:
        """Return whether any L2 adapter is currently active."""
        with self._adapters_lock:
            return bool(self._l2_adapters)

    def _build_l2_adapter(
        self,
        config: L2AdapterConfigBase,
    ) -> tuple[int, L2AdapterInterface, AdapterDescriptor]:
        """Create a L2 adapter instance based on the config.

        Args:
            config: The adapter configuration.

        Returns:
            A ``(adapter_id, adapter, descriptor)`` tuple. ``adapter_id`` is
            the freshly allocated stable id, ``adapter`` is the new adapter
            instance, and ``descriptor`` is its descriptor carrying that id.
        """
        adapter_id = self._next_adapter_id
        self._next_adapter_id += 1
        adapter: L2AdapterInterface = create_l2_adapter(config, self._l1_memory_desc)
        if config.serde_config is not None:
            adapter = SerdeL2AdapterWrapper(
                inner=adapter,
                serde=create_serde_processor(config.serde_config),
                l1_manager=self._l1_manager,
            )
        descriptor = AdapterDescriptor(index=adapter_id, config=config)
        # Stamp the registered type name so the adapter's cache events on
        # the observability bus carry their backend identity.
        adapter.set_backend_identity(descriptor.type_name, shared=config.shared)
        return adapter_id, adapter, descriptor

    def _should_enable_l2_eviction(
        self,
        adapter: L2AdapterInterface,
        eviction_config: EvictionConfig | None,
    ) -> bool:
        """Whether to wire an adapter into the L2 eviction controller.

        Args:
            adapter: The adapter to evaluate.
            eviction_config: The adapter's eviction config, if any.

        Returns:
            True if an eviction state should be created for this adapter.
        """
        if eviction_config is None:
            return False
        policy_name = eviction_config.eviction_policy
        if policy_name != "IsolatedLRU" and not adapter.supports_global_eviction:
            logger.warning(
                "L2 adapter %s configured with '%s' eviction but does "
                "not support global eviction (max_capacity_bytes=0); "
                "skipping aggregate-usage eviction setup.",
                type(adapter).__name__,
                policy_name,
            )
            return False
        return True

    def _unwrap_reconfigurable_l2_adapter(
        self,
        adapter: L2AdapterInterface,
    ) -> Optional[L2ReconfigurableAdapter]:
        if isinstance(adapter, L2ReconfigurableAdapter):
            return adapter

        inner = getattr(adapter, "inner_adapter", None)
        if inner is not None and isinstance(inner, L2ReconfigurableAdapter):
            return inner

        return None

    def _list_reconfigurable_l2_adapters(
        self,
    ) -> list[tuple[int, L2ReconfigurableAdapter]]:
        with self._adapters_lock:
            items = sorted(self._l2_adapters.items())
        adapters: list[tuple[int, L2ReconfigurableAdapter]] = []
        for l2_adapter_index, adapter in items:
            reconfigurable_adapter = self._unwrap_reconfigurable_l2_adapter(adapter)
            if reconfigurable_adapter is not None:
                adapters.append((l2_adapter_index, reconfigurable_adapter))
        return adapters

    def _get_reconfigurable_l2_adapter(
        self,
        adapter_index: int,
    ) -> L2ReconfigurableAdapter:
        adapters = self._list_reconfigurable_l2_adapters()
        if adapter_index < 0 or adapter_index >= len(adapters):
            raise L2ReconfigureError(404, "L2 adapter not reconfigurable")
        return adapters[adapter_index][1]
