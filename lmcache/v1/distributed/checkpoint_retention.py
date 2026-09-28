# SPDX-License-Identifier: Apache-2.0
"""
Decide which recurrent checkpoint pages deserve L1 memory and L2 storage.

A conversation publishes new request-boundary checkpoints every turn, and
each one supersedes the previous turn's endpoint state. Pages shared with the
newer checkpoint stay alive through it, so only the pages unique to the older
checkpoint (recurrent endpoint state, a partial attention page, auxiliary
state) become dead weight. This module keeps that knowledge:

* Superseded pages are never written to L2 when they leave L1. Once stale
  they are dropped first from L1 and L2. A page is stale as soon as it is
  superseded, unless its checkpoint is one a later prompt continued from: a
  branch point such as the end of a turn that a sub-agent forked from and
  ran several turns on. For a grace period such pages keep their LRU
  position instead, so the original line can still continue from them
  while they are in RAM.
* Before the last copy of a superseded page is deleted, the superseded
  checkpoints that need it are retired from the directory, so a lookup
  misses them cleanly and falls back to a shorter checkpoint at once.
* With write-on-evict storage, a current checkpoint page is written to L2
  once, when L1 is about to evict it, instead of on every request.
* L2 residency per adapter is tracked so a page that already reached L2 is
  not written again. A lookup uses it to find pages that may be lost, and
  treats one as lost only when L1 does not hold it and every L2 adapter
  confirms that it does not either.

Everything is bounded in memory and safe to call from the store, eviction and
request-handler threads. Forgetting a supersession only costs an extra write
or a later eviction. Where no adapter can confirm an absence, restores still
validate every page and fall back to a shorter checkpoint when one is gone.
"""

# Standard
from collections import OrderedDict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
import threading
import time

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import ObjectKey, is_recurrent_checkpoint_key
from lmcache.v1.distributed.internal_api import L2AdapterListener

logger = init_logger(__name__)


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


@dataclass
class _SupersededPage:
    """A superseded page: when it may be dropped first, and who needs it."""

    stale_at: float
    """Monotonic time from which the page is dropped first."""

    generations: set[str] = field(default_factory=set)
    """Superseded checkpoints that reference the page."""


def _all_present(keys: list[ObjectKey]) -> list[ObjectKey]:
    """Default L1 lookup: without one, assume L1 may hold every page."""
    return keys


def _retire_nothing(generations: list[str]) -> None:
    """Default retirement hook: without a directory there is nothing to delist."""


def _all_absent(keys: list[ObjectKey]) -> list[ObjectKey]:
    """Default L2 absence check: without L2 adapters no page is in L2."""
    return keys


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
        max_tracked: Bound on remembered L2-resident pages per adapter.
        max_generations: Bound on remembered superseded and continued
            checkpoints.
        persist_timeout: Seconds after which a page whose write never
            completed may be evicted from L1 without reaching L2.
        supersede_grace_seconds: How long a superseded checkpoint that a
            later prompt continued from keeps its LRU position before its
            pages are dropped first. 0 drops every superseded page first.
    """

    def __init__(
        self,
        *,
        write_on_evict: bool = False,
        persist: Callable[[list[ObjectKey]], None] | None = None,
        max_superseded: int = 262144,
        max_tracked: int = 1048576,
        max_generations: int = 65536,
        persist_timeout: float = 120.0,
        supersede_grace_seconds: float = 0.0,
    ) -> None:
        if supersede_grace_seconds < 0:
            raise ValueError("supersede_grace_seconds must be >= 0")
        self._write_on_evict = write_on_evict
        self._persist = persist
        self._max_superseded = max_superseded
        self._max_tracked = max_tracked
        self._max_generations = max_generations
        self._persist_timeout = persist_timeout
        self._grace = supersede_grace_seconds
        self._lock = threading.Lock()
        # Superseded pages in the order they were first marked.
        self._superseded: OrderedDict[ObjectKey, _SupersededPage] = OrderedDict()
        # Superseded checkpoints and the pages they alone referenced.
        self._superseded_generations: OrderedDict[str, tuple[ObjectKey, ...]] = (
            OrderedDict()
        )
        # Checkpoints a later prompt continued from.
        self._continued = _BoundedSet(max_generations)
        # adapter id -> key -> size in bytes
        self._resident: dict[int, OrderedDict[ObjectKey, int]] = {}
        # key -> monotonic time its write was requested
        self._pending: dict[ObjectKey, float] = {}
        self._l1_present: Callable[[list[ObjectKey]], list[ObjectKey]] = _all_present
        self._l2_absent: Callable[[list[ObjectKey]], list[ObjectKey]] = _all_absent
        self._retire: Callable[[list[str]], None] = _retire_nothing
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
            "restored_checkpoints": 0,
        }

    @property
    def write_on_evict(self) -> bool:
        return self._write_on_evict

    def set_persist(self, persist: Callable[[list[ObjectKey]], None]) -> None:
        """Install the asynchronous L2 store callback."""
        self._persist = persist

    def set_l1_lookup(
        self, present: Callable[[list[ObjectKey]], list[ObjectKey]]
    ) -> None:
        """Install the L1 lookup used to decide whether a copy is the last.

        Args:
            present: Returns the given keys that L1 holds. It must not call
                back into this object.
        """
        self._l1_present = present

    def set_l2_absence(
        self, absent: Callable[[list[ObjectKey]], list[ObjectKey]]
    ) -> None:
        """Install the check that confirms pages are in no L2 adapter.

        Args:
            absent: Returns the given keys that every L2 adapter confirms it
                does not hold (all of them without adapters). It must not
                call back into this object.
        """
        self._l2_absent = absent

    def set_retirement(self, retire: Callable[[list[str]], None]) -> None:
        """Install the hook that delists checkpoints whose pages are lost.

        Args:
            retire: Removes the given generations from the checkpoint
                directory. It is called without this object's lock held,
                before the page deletion that loses them, and must not raise.
        """
        with self._lock:
            self._retire = retire

    def clear_retirement(self, retire: Callable[[list[str]], None]) -> None:
        """Remove ``retire`` if it is still the installed retirement hook."""
        with self._lock:
            if self._retire == retire:
                self._retire = _retire_nothing

    def listener_for(self, adapter_id: int) -> L2AdapterListener:
        """Return a listener that records one adapter's checkpoint pages."""
        with self._lock:
            self._resident.setdefault(adapter_id, OrderedDict())
        return _AdapterResidencyListener(self, adapter_id)

    def forget_adapter(self, adapter_id: int) -> None:
        with self._lock:
            self._resident.pop(adapter_id, None)

    # ----- supersession ---------------------------------------------------

    def mark_continued(self, generation: str) -> None:
        """Record that a later prompt directly continued from ``generation``.

        Such a checkpoint may be a branch point: if it is superseded later,
        its pages keep their eviction order for the grace period.
        """
        with self._lock:
            self._continued.add(generation)

    def is_superseded_generation(self, generation: str) -> bool:
        with self._lock:
            return generation in self._superseded_generations

    def mark_superseded(self, generation: str, keys: Iterable[ObjectKey]) -> int:
        """Mark one older checkpoint's unique pages as superseded.

        The pages become stale at once, or after the grace period when a
        later prompt continued from ``generation``.

        Args:
            generation: The older checkpoint's generation, remembered so the
                same checkpoint is not processed again and can be retired
                when one of its pages is lost.
            keys: Pages that no current checkpoint references.

        Returns:
            Number of pages newly marked.
        """
        now = time.monotonic()
        added = 0
        with self._lock:
            stale_at = now + self._grace if generation in self._continued else now
            pages = tuple(keys)
            self._superseded_generations[generation] = pages
            self._superseded_generations.move_to_end(generation)
            while len(self._superseded_generations) > self._max_generations:
                self._superseded_generations.popitem(last=False)
            for key in pages:
                page = self._superseded.get(key)
                if page is None:
                    self._superseded[key] = _SupersededPage(stale_at, {generation})
                    added += 1
                else:
                    page.generations.add(generation)
                    page.stale_at = max(page.stale_at, stale_at)
            while len(self._superseded) > self._max_superseded:
                self._superseded.popitem(last=False)
            self._stats["superseded_checkpoints"] += 1
            self._stats["superseded_pages"] += added
        return added

    def mark_current(self, keys: Iterable[ObjectKey]) -> None:
        """Clear the superseded mark from pages a new checkpoint references."""
        with self._lock:
            for key in keys:
                self._superseded.pop(key, None)

    def mark_generation_current(
        self, generation: str, keys: Iterable[ObjectKey]
    ) -> bool:
        """Make a superseded checkpoint current again.

        Used when a lookup finds the checkpoint: a request continues from it,
        so it is no longer dead weight, and a later prompt may supersede it
        again.

        Args:
            generation: The found checkpoint.
            keys: All of its pages.

        Returns:
            True if the checkpoint was superseded.
        """
        with self._lock:
            was_superseded = (
                self._superseded_generations.pop(generation, None) is not None
            )
            for key in keys:
                self._superseded.pop(key, None)
            if was_superseded:
                self._stats["restored_checkpoints"] += 1
        return was_superseded

    def is_superseded(self, key: ObjectKey) -> bool:
        with self._lock:
            return key in self._superseded

    def superseded_keys(self) -> list[ObjectKey]:
        """Return every superseded page, stale or not, oldest mark first."""
        with self._lock:
            return list(self._superseded)

    def take_droppable_superseded(self) -> list[ObjectKey]:
        """Return stale superseded pages that L1 holds, oldest mark first.

        Stale pages that no tier holds any more are forgotten.
        """
        now = time.monotonic()
        with self._lock:
            stale = [
                key for key, page in self._superseded.items() if page.stale_at <= now
            ]
        if not stale:
            return []
        present = set(self._l1_present(stale))
        with self._lock:
            for key in stale:
                if key not in present and not self._is_resident_locked(key):
                    self._superseded.pop(key, None)
        return [key for key in stale if key in present]

    # ----- retirement -----------------------------------------------------

    def retire_before_l1_delete(self, keys: list[ObjectKey]) -> int:
        """Retire superseded checkpoints that deleting ``keys`` from L1 loses.

        Call it before L1 deletes ``keys``. A superseded page without an L2
        copy is its last copy, so every superseded checkpoint that references
        it is delisted first. Ordinary KV chunks return at once.

        Args:
            keys: Keys L1 is about to delete.

        Returns:
            Number of checkpoints retired.
        """
        tracked = self._tracked_superseded(keys)
        if not tracked:
            return 0
        with self._lock:
            lost = [key for key in tracked if not self._is_resident_locked(key)]
        return self._retire_owners(lost)

    def retire_before_l2_delete(self, adapter_id: int, keys: list[ObjectKey]) -> int:
        """Retire superseded checkpoints that deleting ``keys`` from L2 loses.

        Call it before adapter ``adapter_id`` deletes ``keys``. A superseded
        page that neither L1 nor another adapter holds is its last copy.

        Args:
            adapter_id: The adapter about to delete ``keys``.
            keys: Keys it is about to delete.

        Returns:
            Number of checkpoints retired.
        """
        tracked = self._tracked_superseded(keys)
        if not tracked:
            return 0
        in_l1 = set(self._l1_present(tracked))
        with self._lock:
            lost = [
                key
                for key in tracked
                if key not in in_l1
                and not any(
                    key in resident
                    for other, resident in self._resident.items()
                    if other != adapter_id
                )
            ]
        return self._retire_owners(lost)

    def _tracked_superseded(self, keys: list[ObjectKey]) -> list[ObjectKey]:
        checkpoint_keys = [key for key in keys if is_recurrent_checkpoint_key(key)]
        if not checkpoint_keys:
            return []
        with self._lock:
            return [key for key in checkpoint_keys if key in self._superseded]

    def _retire_owners(self, lost: list[ObjectKey]) -> int:
        """Delist the superseded checkpoints that reference ``lost`` pages."""
        if not lost:
            return 0
        now = time.monotonic()
        with self._lock:
            generations: set[str] = set()
            for key in lost:
                page = self._superseded.get(key)
                if page is not None:
                    generations.update(page.generations)
            for generation in generations:
                for key in self._superseded_generations.pop(generation, ()):
                    page = self._superseded.get(key)
                    if page is None:
                        continue
                    page.generations.discard(generation)
                    if not page.generations:
                        # Nothing listed needs it any more.
                        page.stale_at = min(page.stale_at, now)
                self._continued.discard(generation)
            self._stats["retired_checkpoints"] += len(generations)
            retire = self._retire
        if generations:
            retire(sorted(generations))
        return len(generations)

    # ----- L2 residency ---------------------------------------------------

    def record_l2_present(
        self, adapter_id: int, keys: list[ObjectKey], sizes: list[int]
    ) -> None:
        with self._lock:
            resident = self._resident.setdefault(adapter_id, OrderedDict())
            for key, size in zip(keys, sizes, strict=False):
                if not is_recurrent_checkpoint_key(key):
                    continue
                resident[key] = size
                resident.move_to_end(key)
                if self._pending.pop(key, None) is not None:
                    self._stats["write_on_evict_persisted"] += 1
            while len(resident) > self._max_tracked:
                resident.popitem(last=False)

    def record_l2_absent(self, adapter_id: int, keys: list[ObjectKey]) -> None:
        with self._lock:
            resident = self._resident.get(adapter_id)
            if resident is None:
                return
            for key in keys:
                resident.pop(key, None)

    def is_l2_resident(self, key: ObjectKey) -> bool:
        with self._lock:
            return self._is_resident_locked(key)

    def _is_resident_locked(self, key: ObjectKey) -> bool:
        return any(key in resident for resident in self._resident.values())

    def unavailable_pages(self, keys: list[ObjectKey]) -> list[ObjectKey]:
        """Return the checkpoint pages that no tier holds.

        Pages recorded in L2 count as held. Of the others, a page is
        unavailable when L1 does not hold it and every L2 adapter confirms
        that it does not either; an adapter that cannot confirm an absence
        keeps the page counted as held.

        Args:
            keys: Pages of one checkpoint.

        Returns:
            The unavailable pages, in the order given.
        """
        checkpoint_keys = [key for key in keys if is_recurrent_checkpoint_key(key)]
        with self._lock:
            candidates = [
                key for key in checkpoint_keys if not self._is_resident_locked(key)
            ]
        if not candidates:
            return []
        in_l1 = set(self._l1_present(candidates))
        candidates = [key for key in candidates if key not in in_l1]
        if not candidates:
            return []
        return self._l2_absent(candidates)

    def superseded_in_adapter(
        self, adapter_id: int, max_bytes: int
    ) -> tuple[list[ObjectKey], int]:
        """Return stale superseded pages one adapter holds, up to ``max_bytes``."""
        victims: list[ObjectKey] = []
        total = 0
        now = time.monotonic()
        with self._lock:
            resident: dict[ObjectKey, int] = self._resident.get(adapter_id, {})
            for key, page in self._superseded.items():
                if page.stale_at > now:
                    continue
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

        True for a current checkpoint page that is not in L2 yet, while its
        write is still expected to finish. A superseded page (stale or in its
        grace period), an ordinary KV chunk or a page whose write timed out
        may be evicted.
        """
        if not self._write_on_evict or not is_recurrent_checkpoint_key(key):
            return False
        now = time.monotonic()
        with self._lock:
            if key in self._superseded:
                return False
            if self._is_resident_locked(key):
                self._pending.pop(key, None)
                return False
            requested = self._pending.get(key)
            if requested is not None and now - requested > self._persist_timeout:
                del self._pending[key]
                self._stats["write_on_evict_timeouts"] += 1
                logger.warning(
                    "Checkpoint page write to L2 did not complete within %.0f s; "
                    "evicting it from L1 without an L2 copy",
                    self._persist_timeout,
                )
                return False
            return True

    def request_persist(self, keys: list[ObjectKey]) -> int:
        """Ask the store controller to write pages not already requested."""
        now = time.monotonic()
        with self._lock:
            fresh = [key for key in keys if key not in self._pending]
            for key in fresh:
                self._pending[key] = now
            self._stats["write_on_evict_requests"] += len(fresh)
        if fresh and self._persist is not None:
            self._persist(fresh)
        return len(fresh)

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
        now = time.monotonic()
        with self._lock:
            return {
                "write_on_evict": self._write_on_evict,
                "superseded_pages_tracked": len(self._superseded),
                "superseded_pages_in_grace": sum(
                    1 for page in self._superseded.values() if page.stale_at > now
                ),
                "l2_resident_pages_tracked": sum(
                    len(resident) for resident in self._resident.values()
                ),
                "write_on_evict_pending": len(self._pending),
                **self._stats,
            }
