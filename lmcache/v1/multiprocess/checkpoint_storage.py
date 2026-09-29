# SPDX-License-Identifier: Apache-2.0
"""SHM payload leases for atomic recurrent checkpoint generations.

The storage manager owns RAM and filesystem tiering. Workers own CUDA copies;
this service neither imports a CUDA context nor serializes tensor data over MQ.
Every lease remains pinned until its worker reports completion of the copy.
"""

# Standard
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
import hashlib
import json
import threading
import time
import uuid

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.admission import (
    AdmissionAttempt,
    AdmissionFailure,
    reserve_with_eviction_backpressure,
)
from lmcache.v1.distributed.api import (
    RECURRENT_CHECKPOINT_MODEL_PREFIX,
    MemoryLayoutDesc,
    ObjectKey,
    PrefetchHandle,
    PrefetchRequestSpec,
    TrimPolicy,
)
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.multiprocess.checkpoint_index import CheckpointIndex, CheckpointManifest
from lmcache.v1.multiprocess.transfer_context.shm import ShmSlotDescriptor

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.distributed.storage_manager import StorageManager
    from lmcache.v1.memory_management import MemoryObj

logger = init_logger(__name__)

# A retrieve waiting for RAM asks the L1 eviction loop again after this many
# seconds, as store admission does: a pass frees nothing while its victims are
# still pinned by other restores or L2 writes.
_ROOM_REQUEST_INTERVAL_SECONDS = 0.5


@dataclass(frozen=True)
class CheckpointPageGroup:
    """Persistent byte layout and logical positions for one cache storage.

    A logical position is relative to its KV group, never a GPU block ID.
    ``name`` identifies the storage's semantic role and layer independently of
    allocator addresses. Only the explicitly listed pages are transferred.
    """

    name: str
    page_bytes: int
    positions: tuple[int, ...]
    content_keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name or len(self.name) > 256:
            raise ValueError("checkpoint storage name must contain 1..256 characters")
        if not 0 < self.page_bytes <= 1024**3:
            raise ValueError("checkpoint page size must be in 1 byte..1 GiB")
        if (
            not self.positions
            or len(self.positions) > 1048576
            or any(type(i) is not int or i < 0 for i in self.positions)
            or tuple(sorted(set(self.positions))) != self.positions
        ):
            raise ValueError(
                "checkpoint page positions must be nonempty and increasing"
            )
        if self.content_keys and (
            len(self.content_keys) != len(self.positions)
            or any(
                len(key) != 64
                or any(character not in "0123456789abcdef" for character in key)
                for key in self.content_keys
            )
        ):
            raise ValueError(
                "checkpoint content keys must be one lowercase SHA256 per page"
            )


def checkpoint_page_groups(
    manifest: CheckpointManifest,
) -> tuple[CheckpointPageGroup, ...]:
    """Validate byte groups in a version-1 checkpoint payload manifest.

    Args:
        manifest: Manifest whose JSON payload contains ``page_groups`` entries
            with ``name``, ``page_bytes`` and ``positions`` fields.

    Returns:
        Ordered storage groups, whose order defines LMCache object-group IDs.

    Raises:
        ValueError: If required groups are absent, malformed or duplicated.
    """
    payload = json.loads(manifest.payload)
    rows = payload.get("page_groups")
    if not isinstance(rows, list) or not 1 <= len(rows) <= 4096:
        raise ValueError("checkpoint manifest requires 1..4096 storage groups")
    try:
        groups = tuple(
            CheckpointPageGroup(
                row["name"],
                row["page_bytes"],
                tuple(row["positions"]),
                tuple(row.get("content_keys", ())),
            )
            for row in rows
        )
    except (KeyError, TypeError) as exc:
        raise ValueError("invalid checkpoint storage group") from exc
    if len({group.name for group in groups}) != len(groups):
        raise ValueError("checkpoint storage names must be unique")
    return groups


def checkpoint_object_keys(
    manifest: CheckpointManifest, rank: int
) -> tuple[tuple[ObjectKey, ...], ...]:
    """Derive immutable payload keys without embedding tokens or file paths.

    Args:
        manifest: Complete generation descriptor and isolation namespace.
        rank: Physical tensor-parallel worker rank in the manifest's world.

    Returns:
        Per-group keys ordered by logical page position. Schema-1 keys include
        the generation; schema-2 keys include authenticated page content. Both
        include rank, storage role and namespace. Rank/storage pairs have
        separate eviction namespaces because their logical positions are not
        aligned distributed token-chunk families. The manifest and complete-
        payload retrieval enforce generation atomicity.

    Compatibility:
        Schema-1 manifests retain generation-scoped v2 object keys. Schema-2
        manifests use content-addressed v3 keys and intentionally miss the old
        filesystem objects. Missing payloads invalidate their manifest and
        require recomputation; generations never return partial state.

    Raises:
        ValueError: If rank or the manifest's byte layout is invalid.
    """
    if not 0 <= rank < manifest.world_size:
        raise ValueError("checkpoint payload rank is out of range")
    groups = checkpoint_page_groups(manifest)
    payload_version = json.loads(manifest.payload).get("schema_version")
    if payload_version == 2 and any(not group.content_keys for group in groups):
        raise ValueError("version-2 checkpoint groups require content keys")
    namespaces = tuple(
        hashlib.sha256(
            json.dumps([manifest.prefix.namespace, rank, group_id, group.name]).encode()
        ).hexdigest()
        for group_id, group in enumerate(groups)
    )
    return tuple(
        tuple(
            ObjectKey(
                (
                    bytes.fromhex(group.content_keys[page_id])
                    if group.content_keys
                    else hashlib.sha256(
                        json.dumps([manifest.generation, group.name, position]).encode()
                    ).digest()
                ),
                f"{RECURRENT_CHECKPOINT_MODEL_PREFIX}"
                f"v{3 if group.content_keys else 2}-{namespaces[group_id]}",
                rank,
                group_id,
            )
            for page_id, position in enumerate(group.positions)
        )
        for group_id, group in enumerate(groups)
    )


def _slot(memory: "MemoryObj", expected_bytes: int) -> ShmSlotDescriptor:
    tensor = memory.tensor
    if (
        tensor is None
        or tensor.dtype != torch.uint8
        or tuple(tensor.shape) != (expected_bytes,)
        or memory.shm_offset < 0
        or memory.shm_byte_length != expected_bytes
    ):
        raise ValueError("checkpoint payload does not match its SHM byte layout")
    return ShmSlotDescriptor(
        memory.shm_offset, memory.shm_byte_length, [expected_bytes], "uint8"
    )


@dataclass(frozen=True)
class CheckpointSlots:
    """Pinned SHM descriptors grouped in the manifest's storage/page order."""

    lease_id: str
    groups: tuple[tuple[ShmSlotDescriptor | None, ...], ...]


@dataclass
class _StoreLease:
    manifest: CheckpointManifest
    rank: int
    keys: list[ObjectKey]
    started: float = field(default_factory=time.monotonic)


@dataclass
class _RetrieveLease:
    manifest: CheckpointManifest
    rank: int
    keys: list[ObjectKey]
    # None while the lease waits for RAM before repeating its lookup.
    handle: PrefetchHandle | None
    slots: CheckpointSlots | None = None
    cancelled: bool = False
    started: float = field(default_factory=time.monotonic)
    lookups: int = 1
    # Readable pages of the last incomplete lookup, the RAM its unread pages
    # need, and when eviction was last asked to make that room.
    readable: int = 0
    unread_bytes: int = 0
    eviction_requested: float = 0.0


class CheckpointPayloadStore:
    """Lease checkpoint byte pages through an existing SHM storage manager.

    Args:
        storage: SHM-backed storage manager; the service does not own its life.
        index: Directory used for all-rank publication and stale invalidation.
        max_leases: Shared bound for pending stores and retrieves. Exhaustion
            rejects admission without recycling a worker's live SHM buffers.
        abandoned_after_seconds: Age after which a lease or pending generation
            is treated as abandoned by a dead worker and released; 0 disables
            reclaiming. It must be far longer than any live transfer. A worker
            whose copy could not drain (UnsafeCheckpointCopyError) keeps its
            lease on purpose because its DMA may still touch those pages;
            reclaiming releases that lease too, which is safe only because a
            copy still running after this long means a hung CUDA context.

    The caller stages a manifest with ``index.begin`` before rank stores. Each
    successful store acknowledgement follows a drained worker D2H transfer.
    Retrieval completion similarly follows H2D completion, not MQ delivery.
    """

    def __init__(
        self,
        storage: "StorageManager",
        index: CheckpointIndex,
        *,
        max_leases: int = 1024,
        abandoned_after_seconds: float = 600.0,
    ) -> None:
        if max_leases < 1:
            raise ValueError("checkpoint lease capacity must be positive")
        if abandoned_after_seconds < 0:
            raise ValueError("abandoned lease age must not be negative")
        self._storage = storage
        self._index = index
        self._max_leases = max_leases
        self._abandoned_after = abandoned_after_seconds
        self._last_reclaim = time.monotonic()
        self._stores: dict[str, _StoreLease] = {}
        self._store_ranks: set[tuple[str, int]] = set()
        self._retrieves: dict[str, _RetrieveLease] = {}
        self._retrieve_admissions: set[str] = set()
        self._lock = threading.Lock()

    def prepare_store(
        self, manifest: CheckpointManifest, rank: int
    ) -> CheckpointSlots | AdmissionFailure | None:
        """Reserve all pages for one rank or release the partial reservation.

        Args:
            manifest: Staged immutable generation descriptor.
            rank: Producer rank.

        Returns:
            A pinned writable lease, ``AdmissionFailure.BUSY`` when an
            immutable page has another writer, or None for a terminal miss.
            Busy admission retains the pending generation so the producer can
            retry; a terminal miss cancels its publication.

        Raises:
            ValueError: For invalid layouts or a duplicate producer rank.
        """
        self._maybe_reclaim()
        groups = checkpoint_page_groups(manifest)
        key_groups = checkpoint_object_keys(manifest, rank)
        if not self._index.is_pending(manifest):
            return None
        identity = (manifest.generation, rank)
        with self._lock:
            if identity in self._store_ranks:
                raise ValueError("checkpoint rank already has a store lease")
            if (
                len(self._store_ranks)
                + len(self._retrieves)
                + len(self._retrieve_admissions)
                >= self._max_leases
            ):
                self._index.abort(manifest.generation)
                return None
            self._store_ranks.add(identity)
        reserved: list[ObjectKey] = []

        def attempt() -> AdmissionAttempt[
            tuple[tuple[ShmSlotDescriptor | None, ...], ...]
        ]:
            if not self._index.is_pending(manifest):
                return AdmissionAttempt.failure(AdmissionFailure.CONFLICT)
            slots: list[tuple[ShmSlotDescriptor | None, ...]] = []
            for group, keys in zip(groups, key_groups, strict=True):
                detailed = self._storage.reserve_write_detailed(
                    list(keys),
                    MemoryLayoutDesc([torch.Size([group.page_bytes])], [torch.uint8]),
                    "new",
                )
                objects = {
                    key: obj
                    for key, (_error, obj) in detailed.items()
                    if obj is not None
                }
                reserved.extend(objects)
                existing_candidates = [
                    key
                    for key in keys
                    if detailed.get(key, (L1Error.KEY_NOT_EXIST, None))[0]
                    is L1Error.KEY_NOT_WRITABLE
                ]
                readable = set(self._storage.get_readable_keys(existing_candidates))
                if len(objects) + len(readable) != len(keys):
                    self._storage.abort_write(reserved)
                    reserved.clear()
                    missing = [
                        key
                        for key in keys
                        if key not in objects and key not in readable
                    ]
                    errors = [
                        detailed.get(key, (L1Error.KEY_NOT_EXIST, None))[0]
                        for key in missing
                    ]
                    capacity_only = all(
                        error is L1Error.OUT_OF_MEMORY for error in errors
                    )
                    busy_only = all(
                        error is L1Error.KEY_NOT_WRITABLE for error in errors
                    )
                    return AdmissionAttempt.failure(
                        AdmissionFailure.CAPACITY
                        if capacity_only
                        else (
                            AdmissionFailure.BUSY
                            if busy_only
                            else AdmissionFailure.CONFLICT
                        )
                    )
                slots.append(
                    tuple(
                        _slot(objects[key], group.page_bytes)
                        if key in objects
                        else None
                        for key in keys
                    )
                )
            return AdmissionAttempt.success(tuple(slots))

        admitted = False
        retryable = False
        try:
            outcome = reserve_with_eviction_backpressure(
                attempt=attempt,
                get_generation=self._storage.get_capacity_generation,
                request_eviction=self._storage.request_immediate_eviction,
                wait_for_change=self._storage.wait_for_capacity_change,
                timeout_seconds=self._storage.store_admission_timeout_seconds,
                on_wait=self._storage.record_admission_wait,
                on_retry=self._storage.record_admission_retry,
                on_success_after_eviction=(
                    self._storage.record_admission_success_after_eviction
                ),
                on_timeout=self._storage.record_admission_timeout,
            )
            if outcome.value is None:
                retryable = outcome.failure is AdmissionFailure.BUSY
                return AdmissionFailure.BUSY if retryable else None
            lease_id = uuid.uuid4().hex
            with self._lock:
                self._stores[lease_id] = _StoreLease(manifest, rank, reserved)
            admitted = True
            return CheckpointSlots(lease_id, outcome.value)
        finally:
            if not admitted:
                try:
                    self._storage.abort_write(reserved)
                finally:
                    try:
                        if not retryable:
                            self._index.abort(manifest.generation)
                    finally:
                        with self._lock:
                            self._store_ranks.discard(identity)

    def finish_store(self, lease_id: str, success: bool) -> bool:
        """Commit or discard pages after the producer's CUDA event completes.

        Args:
            lease_id: Writable lease returned by ``prepare_store``.
            success: Whether all page copies completed successfully.

        Returns:
            True only if this acknowledgement publishes the complete generation.
            Unknown or previously completed leases return False.
        """
        with self._lock:
            lease = self._stores.pop(lease_id, None)
        if lease is None:
            return False
        try:
            if not success:
                self._storage.abort_write(lease.keys)
                self._index.abort(lease.manifest.generation)
                return False
            self._storage.finish_write(lease.keys)
            if len(self._storage.get_readable_keys(lease.keys)) != len(lease.keys):
                self._index.abort(lease.manifest.generation)
                return False
            return self._index.acknowledge(lease.manifest.generation, lease.rank)
        except Exception:
            # Commit can fail before releasing every exclusive write lock.
            # abort_write leaves already-readable objects intact; none become
            # discoverable through this generation after publication is aborted.
            try:
                self._storage.abort_write(lease.keys)
            finally:
                self._index.abort(lease.manifest.generation)
            raise
        finally:
            with self._lock:
                self._store_ranks.remove((lease.manifest.generation, lease.rank))

    def begin_retrieve(self, manifest: CheckpointManifest, rank: int) -> str | None:
        """Start asynchronous RAM/filesystem lookup for every page of one rank.

        Args:
            manifest: Candidate selected by the checkpoint directory.
            rank: Consumer rank, using exactly the producer's parallel geometry.

        Returns:
            Lease identifier to poll, or None if the lease budget is exhausted.
            Admission does not indicate a successful cache hit.

        Raises:
            ValueError: If the manifest layout or rank is invalid.
        """
        self._maybe_reclaim()
        keys = [
            key for group in checkpoint_object_keys(manifest, rank) for key in group
        ]
        lease_id = uuid.uuid4().hex
        with self._lock:
            if (
                len(self._store_ranks)
                + len(self._retrieves)
                + len(self._retrieve_admissions)
                >= self._max_leases
            ):
                return None
            self._retrieve_admissions.add(lease_id)
        try:
            handle = self._lookup(lease_id, manifest, keys)
            with self._lock:
                self._retrieves[lease_id] = _RetrieveLease(manifest, rank, keys, handle)
                self._retrieve_admissions.remove(lease_id)
        finally:
            with self._lock:
                self._retrieve_admissions.discard(lease_id)
        return lease_id

    def _lookup(
        self, lease_id: str, manifest: CheckpointManifest, keys: list[ObjectKey]
    ) -> PrefetchHandle:
        """Submit the RAM/filesystem lookup that pins every page of one rank."""
        layouts = {
            group_id: MemoryLayoutDesc([torch.Size([group.page_bytes])], [torch.uint8])
            for group_id, group in enumerate(checkpoint_page_groups(manifest))
        }
        return self._storage.submit_prefetch_task(
            PrefetchRequestSpec(keys, layouts, policy=TrimPolicy.SPARSE),
            external_request_id=f"checkpoint-{lease_id}",
        )

    def _maybe_reclaim(self) -> None:
        if not self._abandoned_after:
            return
        now = time.monotonic()
        with self._lock:
            if now - self._last_reclaim < min(30.0, self._abandoned_after):
                return
            self._last_reclaim = now
        self.reclaim_abandoned()

    def reclaim_abandoned(self) -> int:
        """Release leases and pending generations that a dead worker left behind.

        A worker that dies between PREPARE_STORE and FINISH_STORE, or between
        a ready retrieve and its release, never returns its lease. The lease
        keeps its L1 locks and counts against ``max_leases``, and its pending
        generation can never be staged again. Once older than
        ``abandoned_after_seconds``, store leases are aborted, ready retrieve
        leases release their read locks, and pending lookups are cancelled and
        drained as their prefetch completes.

        Returns:
            Number of leases and pending generations released.
        """
        cutoff = time.monotonic() - self._abandoned_after
        with self._lock:
            stores = [
                (lease_id, lease)
                for lease_id, lease in self._stores.items()
                if lease.started < cutoff
            ]
            for lease_id, lease in stores:
                del self._stores[lease_id]
                self._store_ranks.discard((lease.manifest.generation, lease.rank))
            ready = [
                (lease_id, lease)
                for lease_id, lease in self._retrieves.items()
                if lease.slots is not None and lease.started < cutoff
            ]
            for lease_id, _lease in ready:
                del self._retrieves[lease_id]
            lookups = [
                lease_id
                for lease_id, lease in self._retrieves.items()
                if lease.slots is None and lease.started < cutoff
            ]
            for lease_id in lookups:
                self._retrieves[lease_id].cancelled = True
        for _lease_id, store_lease in stores:
            self._storage.abort_write(store_lease.keys)
            self._index.abort(store_lease.manifest.generation)
        for _lease_id, read_lease in ready:
            self._storage.finish_read_prefetched(read_lease.keys)
        for lease_id in lookups:
            try:
                # A cancelled lookup releases its locks once its prefetch ends.
                self.poll_retrieve(lease_id)
            except KeyError:
                pass
        stale = self._index.abort_stale(self._abandoned_after)
        reclaimed = len(stores) + len(ready) + len(lookups) + len(stale)
        if reclaimed:
            logger.warning(
                "Released %d store leases, %d retrieve leases, %d lookups and %d "
                "pending checkpoint generations abandoned for over %.0f s",
                len(stores),
                len(ready),
                len(lookups),
                len(stale),
                self._abandoned_after,
            )
        return reclaimed

    def report_status(self) -> dict[str, int]:
        """Return live lease counts without releasing worker-owned copy buffers.

        Store counts include reservations being prepared. Retrieve counts
        include prefetch submissions that have not yet returned their handle.
        A shutdown coordinator must drain these leases before closing storage.
        """
        with self._lock:
            return {
                "store_leases": len(self._store_ranks),
                "retrieve_leases": len(self._retrieves)
                + len(self._retrieve_admissions),
                "max_leases": self._max_leases,
            }

    def poll_retrieve(self, lease_id: str) -> CheckpointSlots | bool | None:
        """Return slots only when every required page is readable and pinned.

        Args:
            lease_id: Identifier from ``begin_retrieve``.

        Returns:
            None while prefetch is pending, False on a miss/cancellation, or
            the pinned slots. A miss releases all acquired locks and invalidates
            only the failed generation. It must not advance computed tokens.

            The storage manager loads pages from L2 only into RAM it reserved
            for all of them, so a full RAM leaves stored pages unread. A lookup
            with unread pages is therefore repeated once RAM has room for them,
            and only a repeated lookup without allocation failure invalidates
            the generation. The lease stays pending while eviction makes room, for
            at most the storage admission timeout; a lease that never gets
            room misses without invalidating its generation.

        Raises:
            KeyError: If the lease is unknown or already finished.
            ValueError: If stored payload byte layouts do not match the manifest.
        """
        with self._lock:
            lease = self._retrieves[lease_id]
            if lease.slots is not None:
                return lease.slots
            if lease.handle is None:
                return self._repeat_with_room(lease_id, lease)
            result = self._storage.query_prefetch_status_detailed(lease.handle)
            if result is None:
                return None
            found = result.found
            readable_keys = [key for i, key in enumerate(lease.keys) if found.test(i)]
            if len(readable_keys) != len(lease.keys) or lease.cancelled:
                self._storage.finish_read_prefetched(readable_keys)
                return self._repeat_or_miss(
                    lease_id, lease, readable_keys, result.reservation_failed
                )
            try:
                keys, objects = self._storage.unsafe_read(lease.keys)
                if keys != lease.keys or len(objects) != len(keys):
                    raise ValueError("checkpoint pages changed during pinned retrieval")
                slots = []
                offset = 0
                for group in checkpoint_page_groups(lease.manifest):
                    count = len(group.positions)
                    slots.append(
                        tuple(
                            _slot(obj, group.page_bytes)
                            for obj in objects[offset : offset + count]
                        )
                    )
                    offset += count
                lease.slots = CheckpointSlots(lease_id, tuple(slots))
                return lease.slots
            except Exception:
                self._storage.finish_read_prefetched(lease.keys)
                self._index.invalidate(lease.manifest.generation)
                del self._retrieves[lease_id]
                raise

    def _repeat_or_miss(
        self,
        lease_id: str,
        lease: _RetrieveLease,
        readable_keys: list[ObjectKey],
        reservation_failed: bool,
    ) -> bool | None:
        """Repeat or end a lookup whose pages were not all pinned.

        Called with the lookup's read locks already released. Returns None
        when the lookup will be repeated, otherwise False after forgetting the
        lease. Only a repeated lookup without allocation failure invalidates.
        """
        lease.readable = len(readable_keys)
        if lease.cancelled:
            return self._miss(lease_id, lease, "was cancelled by the engine")
        groups = checkpoint_page_groups(lease.manifest)
        readable = set(readable_keys)
        alignment = self._storage.l1_memory_desc.align_bytes
        lease.unread_bytes = sum(
            ((groups[key.object_group_id].page_bytes + alignment - 1) // alignment)
            * alignment
            for key in lease.keys
            if key not in readable
        )
        used, total = self._storage.get_l1_usage()
        if reservation_failed and lease.unread_bytes > total:
            return self._miss(
                lease_id, lease, "cannot fit in RAM; its checkpoint stays listed"
            )
        # Without L2, or with more pages than RAM can hold, no repeat loads them.
        if (
            self._storage.l2_adapters()
            and lease.unread_bytes <= total
            and (
                reservation_failed
                or lease.lookups == 1
                or total - used < lease.unread_bytes
            )
        ):
            lease.handle = None
            return self._repeat_with_room(lease_id, lease)
        self._index.invalidate(lease.manifest.generation)
        return self._miss(lease_id, lease, "missed; its checkpoint is no longer listed")

    def _repeat_with_room(self, lease_id: str, lease: _RetrieveLease) -> bool | None:
        """Repeat a waiting lookup once RAM can hold its unread pages.

        Until then the L1 eviction loop is asked to make room. Returns None
        while waiting or after resubmitting, or False once the lease is
        cancelled or the storage admission timeout expires. Its pages remain
        stored, so the generation stays listed.
        """
        if lease.cancelled:
            return self._miss(lease_id, lease, "was cancelled by the engine")
        now = time.monotonic()
        if now - lease.started >= self._storage.store_admission_timeout_seconds:
            return self._miss(
                lease_id, lease, "found no room in RAM; its checkpoint stays listed"
            )
        used, total = self._storage.get_l1_usage()
        if total - used >= lease.unread_bytes:
            lease.handle = self._lookup(lease_id, lease.manifest, lease.keys)
            lease.lookups += 1
            return None
        if now - lease.eviction_requested >= _ROOM_REQUEST_INTERVAL_SECONDS:
            lease.eviction_requested = now
            self._storage.request_immediate_eviction()
        return None

    def _miss(self, lease_id: str, lease: _RetrieveLease, outcome: str) -> bool:
        """Forget a lease whose read locks are released, and say why."""
        del self._retrieves[lease_id]
        logger.info(
            "Checkpoint retrieve of %d tokens for rank %d %s after %.1f s: "
            "%d of %d pages were readable",
            lease.manifest.prefix.num_tokens,
            lease.rank,
            outcome,
            time.monotonic() - lease.started,
            lease.readable,
            len(lease.keys),
        )
        return False

    def finish_retrieve(self, lease_id: str) -> None:
        """Release a prepared read lease after the consumer's CUDA event completes.

        Args:
            lease_id: Pinned lease whose H2D work has drained.

        Raises:
            ValueError: If lookup is still pending; use ``cancel_retrieve``.
        """
        with self._lock:
            lease = self._retrieves.get(lease_id)
            if lease is None:
                return
            if lease.slots is None:
                raise ValueError("checkpoint retrieval has not produced a read lease")
            del self._retrieves[lease_id]
        # A completed restore proves reuse; reuse-admission store policies
        # store checkpoint pages to L2 only after such a report.
        self._storage.notify_keys_reused(lease.keys)
        self._storage.finish_read_prefetched(lease.keys)

    def cancel_retrieve(self, lease_id: str) -> None:
        """Mark a pending lookup for draining without exposing its SHM slots.

        Args:
            lease_id: Lookup whose consumer was cancelled before H2D submission.
                The caller must keep polling until False releases lookup locks.

        Raises:
            ValueError: If slots were already exposed; their GPU copy must first
                drain and use ``finish_retrieve`` instead.
        """
        with self._lock:
            lease = self._retrieves[lease_id]
            if lease.slots is not None:
                raise ValueError(
                    "prepared checkpoint retrieval requires copy completion"
                )
            lease.cancelled = True
