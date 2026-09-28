# SPDX-License-Identifier: Apache-2.0
"""Coordinate checkpoint persistence, replica accounting and coherent retirement.

Supersession is an eviction hint, never proof that an older branch is dead.
Deferred writes start under RAM pressure or shutdown. Directory lifecycle
callbacks retire affected checkpoints before their final known copy is reclaimed.

Supersession hints are bounded independently. Replica accounting is exact for
inventoried physical payloads and shrinks when adapters report deletion; it must
not forget a live replica because a telemetry history limit was reached.
"""

# Standard
from collections import OrderedDict
from collections.abc import Callable, Iterable
from typing import TypeVar
import threading
import time

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import ObjectKey, is_recurrent_checkpoint_key
from lmcache.v1.distributed.internal_api import L2AdapterListener

logger = init_logger(__name__)
_DeleteResult = TypeVar("_DeleteResult")


class _BoundedSet:
    """Insertion-ordered set that forgets its oldest entries beyond a bound."""

    def __init__(self, bound: int) -> None:
        self._bound = bound
        self._items: OrderedDict[object, None] = OrderedDict()

    def add(self, item: object) -> None:
        self._items[item] = None
        self._items.move_to_end(item)
        while len(self._items) > self._bound:
            self._items.popitem(last=False)

    def discard(self, item: object) -> None:
        self._items.pop(item, None)

    def __contains__(self, item: object) -> bool:
        return item in self._items

    def __len__(self) -> int:
        return len(self._items)

    def items(self) -> list:
        return list(self._items)


class _AdapterResidencyListener(L2AdapterListener):
    """Record which checkpoint pages one L2 adapter holds."""

    def __init__(self, retention: "CheckpointRetention", adapter_id: int) -> None:
        self._retention = retention
        self._adapter_id = adapter_id

    def on_l2_keys_stored(self, keys: list[ObjectKey], sizes: list[int]) -> None:
        self._retention.record_l2_present(self._adapter_id, keys, sizes)

    def on_l2_keys_accessed(self, keys: list[ObjectKey]) -> None:
        pass

    def on_l2_keys_deleted(self, keys: list[ObjectKey]) -> None:
        self._retention.record_l2_absent(self._adapter_id, keys)


class CheckpointRetention:
    """
    Track superseded and persisted recurrent checkpoint pages.

    Args:
        write_on_evict: Whether current checkpoint pages are written to L2
            only when L1 evicts them. Without it, checkpoint pages follow the
            store policy and this object only orders eviction.
        persist: Callback that asynchronously stores keys to L2; installed by
            the storage manager once its store controller exists.
        max_superseded: Bound on remembered superseded pages.
        max_tracked: Legacy compatibility argument; physical replica inventory
            is exact and bounded by stored payloads, not a telemetry history cap.
        persist_timeout: Legacy compatibility argument. Elapsed write time alone
            no longer permits losing a retained checkpoint page.
    """

    def __init__(
        self,
        *,
        write_on_evict: bool = False,
        persist: Callable[[list[ObjectKey]], list[ObjectKey] | None] | None = None,
        max_superseded: int = 262144,
        max_tracked: int = 1048576,
        max_generations: int = 65536,
        persist_timeout: float = 120.0,
    ) -> None:
        self._write_on_evict = write_on_evict
        self._persist = persist
        self._lock = threading.Lock()
        self._superseded = _BoundedSet(max_superseded)
        self._superseded_generations = _BoundedSet(max_generations)
        # adapter id -> key -> size in bytes
        self._resident: dict[int, OrderedDict[ObjectKey, int]] = {}
        self._unknown_inventories: set[int] = set()
        # key -> monotonic time its write was requested
        self._pending: dict[ObjectKey, float] = {}
        self._stats = {
            "superseded_checkpoints": 0,
            "superseded_pages": 0,
            "l1_superseded_drops": 0,
            "l2_superseded_evictions": 0,
            "l2_superseded_eviction_bytes": 0,
            "write_on_evict_requests": 0,
            "write_on_evict_persisted": 0,
            "write_on_evict_timeouts": 0,
            "retired_checkpoints": 0,
        }
        self._evict: Callable | None = None
        self._pressure: Callable[[int], int] | None = None
        self._retained: Callable[[ObjectKey], bool] | None = None
        self._order: Callable[[list[ObjectKey]], list[ObjectKey]] | None = None
        self._reconcile: Callable[[], None] | None = None

    def set_lifecycle(
        self,
        *,
        evict: Callable | None,
        pressure: Callable[[int], int] | None,
        retained: Callable[[ObjectKey], bool] | None,
        order: Callable[[list[ObjectKey]], list[ObjectKey]] | None = None,
        reconcile: Callable[[], None] | None = None,
    ) -> None:
        """Attach the directory authority; callbacks run outside retention locks."""
        self._evict, self._pressure, self._retained = evict, pressure, retained
        self._order = order
        self._reconcile = reconcile

    def evict(
        self,
        keys: list[ObjectKey],
        tier: str | int,
        delete: Callable[[list[ObjectKey]], _DeleteResult],
    ) -> _DeleteResult:
        """Retire last-copy owners before deletion; return the callback result.

        ``tier`` identifies L1 or an adapter. The callback receives only keys
        eligible for deletion, and its exceptions propagate to the caller.
        """
        if self._evict is not None:
            return self._evict(keys, tier, delete)
        return delete(keys)

    def retire_under_pressure(self, target_bytes: int) -> int:
        """Retire cold generations to free capacity after bounded admission waits."""
        return self._pressure(target_bytes) if self._pressure is not None else 0

    def is_retained(self, key: ObjectKey) -> bool:
        """Whether a checkpoint page still has a generation owner."""
        return self._retained(key) if self._retained is not None else True

    def resident_adapters(
        self, key: ObjectKey, expected_size: int | None = None
    ) -> set[int]:
        """Return known replicas, optionally requiring the manifest's byte size.

        Args:
            key: Checkpoint page to inspect.
            expected_size: Required payload bytes, or None for presence only.

        Returns:
            Adapter IDs with a matching recorded replica.
        """
        with self._lock:
            return {
                adapter
                for adapter, resident in self._resident.items()
                if key in resident
                and (expected_size is None or resident[key] == expected_size)
            }

    def record_retirement(self, count: int) -> None:
        """Count explicitly retired generations separately from failed restores."""
        with self._lock:
            self._stats["retired_checkpoints"] += count

    def set_inventory_complete(self, adapter_id: int) -> None:
        """Mark a successfully enumerated adapter inventory as authoritative."""
        with self._lock:
            self._unknown_inventories.discard(adapter_id)

    def inventories_complete(self) -> bool:
        """Whether missing residency is proof of absence across all adapters."""
        with self._lock:
            return not self._unknown_inventories

    @property
    def write_on_evict(self) -> bool:
        return self._write_on_evict

    def set_persist(
        self, persist: Callable[[list[ObjectKey]], list[ObjectKey] | None]
    ) -> None:
        """Install the asynchronous L2 store callback."""
        self._persist = persist

    def listener_for(self, adapter_id: int) -> L2AdapterListener:
        """Return a listener that records one adapter's checkpoint pages."""
        with self._lock:
            self._resident.setdefault(adapter_id, OrderedDict())
            self._unknown_inventories.add(adapter_id)
        return _AdapterResidencyListener(self, adapter_id)

    def forget_adapter(self, adapter_id: int) -> None:
        with self._lock:
            keys = list(self._resident.get(adapter_id, {}))
        self.evict(
            keys,
            adapter_id,
            lambda selected: self.record_l2_absent(adapter_id, selected),
        )
        with self._lock:
            self._resident.pop(adapter_id, None)
            self._unknown_inventories.discard(adapter_id)
        if self._reconcile is not None:
            self._reconcile()

    # ----- supersession ---------------------------------------------------

    def is_superseded_generation(self, generation: str) -> bool:
        with self._lock:
            return generation in self._superseded_generations

    def mark_superseded(self, generation: str, keys: Iterable[ObjectKey]) -> int:
        """Mark one older checkpoint's unique pages as superseded.

        Args:
            generation: The older checkpoint's generation, remembered so the
                same ancestor is not processed again.
            keys: Pages that no current checkpoint references.

        Returns:
            Number of pages newly marked.
        """
        added = 0
        with self._lock:
            self._superseded_generations.add(generation)
            for key in keys:
                if key not in self._superseded:
                    added += 1
                self._superseded.add(key)
            self._stats["superseded_checkpoints"] += 1
            self._stats["superseded_pages"] += added
        return added

    def mark_current(self, keys: Iterable[ObjectKey]) -> None:
        """Clear the superseded mark from pages a new checkpoint references."""
        with self._lock:
            for key in keys:
                self._superseded.discard(key)

    def is_superseded(self, key: ObjectKey) -> bool:
        with self._lock:
            return key in self._superseded

    def superseded_keys(self) -> list[ObjectKey]:
        with self._lock:
            return self._superseded.items()

    # ----- L2 residency ---------------------------------------------------

    def record_l2_present(
        self, adapter_id: int, keys: list[ObjectKey], sizes: list[int]
    ) -> None:
        with self._lock:
            resident = self._resident.setdefault(adapter_id, OrderedDict())
            for key, size in zip(keys, sizes, strict=False):
                if not is_recurrent_checkpoint_key(key):
                    continue
                if size or key not in resident:
                    resident[key] = size
                resident.move_to_end(key)
                if self._pending.pop(key, None) is not None:
                    self._stats["write_on_evict_persisted"] += 1
            # Residency is bounded by physical tier contents, not an LRU sample.
            # Forgetting a live replica would make retirement decisions unsound.

    def record_l2_absent(self, adapter_id: int, keys: list[ObjectKey]) -> None:
        with self._lock:
            resident = self._resident.get(adapter_id)
            if resident is None:
                return
            for key in keys:
                resident.pop(key, None)

    def is_l2_resident(self, key: ObjectKey, expected_size: int | None = None) -> bool:
        """Whether a replica exists with the optional required payload byte size."""
        with self._lock:
            return any(
                key in resident
                and (expected_size is None or resident[key] == expected_size)
                for resident in self._resident.values()
            )

    def replica_may_match(
        self, adapter_id: int, key: ObjectKey, expected_size: int
    ) -> bool:
        """Allow lookup unless inventory positively proves a byte-size mismatch.

        Args:
            adapter_id: Backend being considered for the load.
            key: Requested object key.
            expected_size: Required logical payload bytes.

        Returns:
            True for unknown or matching sizes; False for a known mismatch.
        """
        with self._lock:
            resident = self._resident.get(adapter_id)
            size = resident.get(key) if resident is not None else None
            return size is None or size == expected_size

    def superseded_in_adapter(
        self, adapter_id: int, max_bytes: int
    ) -> tuple[list[ObjectKey], int]:
        """Return superseded pages one adapter holds, up to ``max_bytes``."""
        victims: list[ObjectKey] = []
        total = 0
        with self._lock:
            resident: dict[ObjectKey, int] = self._resident.get(adapter_id, {})
            for key in self._superseded.items():
                size = resident.get(key)
                if size is None:
                    continue
                victims.append(key)
                total += size
                if total >= max_bytes:
                    break
        return victims, total

    def record_l2_superseded_evictions(self, count: int, size: int) -> None:
        with self._lock:
            self._stats["l2_superseded_evictions"] += count
            self._stats["l2_superseded_eviction_bytes"] += size

    def record_l1_superseded_drops(self, count: int) -> None:
        with self._lock:
            self._stats["l1_superseded_drops"] += count

    # ----- write on evict -------------------------------------------------

    def needs_persist_before_evict(self, key: ObjectKey) -> bool:
        """Whether L1 must keep ``key`` until its L2 write completes.

        True for a checkpoint page without a known L2 copy. Ordinary KV
        chunks follow their existing policy. Capacity-driven loss first retires
        the owning generations; age and supersession alone never allow loss.
        """
        if not self._write_on_evict or not is_recurrent_checkpoint_key(key):
            return False
        with self._lock:
            if any(key in resident for resident in self._resident.values()):
                self._pending.pop(key, None)
                return False
            return True

    def request_persist(self, keys: list[ObjectKey]) -> int:
        """Ask the store controller to write pages not already requested."""
        if self._order is not None:
            keys = self._order(keys)
        now = time.monotonic()
        with self._lock:
            fresh = [
                key
                for key in keys
                if key not in self._pending
                and not any(key in resident for resident in self._resident.values())
            ]
            for key in fresh:
                self._pending[key] = now
            self._stats["write_on_evict_requests"] += len(fresh)
        if fresh and self._persist is not None:
            accepted = self._persist(fresh)
            if accepted is not None:
                accepted_keys = set(accepted)
                with self._lock:
                    for key in fresh:
                        if key not in accepted_keys:
                            self._pending.pop(key, None)
        return (
            len(accepted_keys)
            if fresh and self._persist is not None and accepted is not None
            else len(fresh)
        )

    def forget_pending(self, keys: Iterable[ObjectKey]) -> None:
        """Forget requests for pages no retained generation needs."""
        with self._lock:
            for key in keys:
                self._pending.pop(key, None)

    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)

    def l2_checkpoint_bytes(self) -> int:
        """Bytes of tracked checkpoint pages held across L2 adapters."""
        with self._lock:
            return sum(sum(resident.values()) for resident in self._resident.values())

    def observations(self) -> list[tuple[int | float, dict[str, object]]]:
        """Counters and sizes in OTel-observation shape, one per ``stat``."""
        status = self.report_status()
        status.pop("write_on_evict")
        status["l2_checkpoint_bytes"] = self.l2_checkpoint_bytes()
        return [(value, {"stat": name}) for name, value in status.items()]

    def report_status(self) -> dict:
        with self._lock:
            return {
                "write_on_evict": self._write_on_evict,
                "superseded_pages_tracked": len(self._superseded),
                "l2_resident_pages_tracked": sum(
                    len(resident) for resident in self._resident.values()
                ),
                "write_on_evict_pending": len(self._pending),
                **self._stats,
            }
