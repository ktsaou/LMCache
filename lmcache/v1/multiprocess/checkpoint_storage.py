# SPDX-License-Identifier: Apache-2.0
"""SHM payload leases for atomic recurrent checkpoint generations.

The storage manager owns RAM and filesystem tiering. Workers own CUDA copies;
this service neither imports a CUDA context nor serializes tensor data over MQ.
Every lease remains pinned until its worker reports completion of the copy.
"""

# Standard
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypeVar
import hashlib
import json
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
_DeleteResult = TypeVar("_DeleteResult")

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
    reused: list[ObjectKey] = field(default_factory=list)
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
        abandoned_after_seconds: Age after which unexposed lookups and pending
            generations are cancelled; 0 disables automatic cancellation.
            Exposed copy buffers stay owned until the worker reports completion.
            Elapsed time cannot prove that a CUDA transfer has stopped.

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
        self._lock = index.lifecycle_lock
        self._completed: dict[str, list[ObjectKey]] = {}
        self._completed_ranks: dict[str, set[int]] = {}
        self._publication_bytes: dict[str, int] = {}
        self._generations: OrderedDict[str, dict[ObjectKey, int]] = OrderedDict()
        self._owners: dict[ObjectKey, set[str]] = {}
        self._published: set[str] = set()
        self._deleting: dict[ObjectKey, set[str | int]] = {}
        self._closed = False
        self._accepting = True
        index.add_lifecycle_listener(self._index_changed, self._before_index_close)
        storage.checkpoint_retention.set_lifecycle(
            evict=self._evict_pages,
            pressure=self._retire_under_pressure,
            retained=self._is_retained,
            order=self._persistence_order,
            reconcile=self._reconcile_replicas,
        )
        with self._lock:
            for manifest in index.manifests():
                self._register_manifest(manifest)
                self._published.add(manifest.generation)
                keys = list(self._generations[manifest.generation])
                readable = set(storage.get_readable_keys(keys, retained_only=True))
                if storage.checkpoint_retention.inventories_complete() and any(
                    key not in readable
                    and not storage.checkpoint_retention.is_l2_resident(
                        key, self._generations[manifest.generation][key]
                    )
                    for key in keys
                ):
                    index.invalidate(manifest.generation)

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
            if not self._accepting:
                return None
            if identity in self._store_ranks or rank in self._completed_ranks.get(
                manifest.generation, set()
            ):
                raise ValueError("checkpoint rank already has a store lease")
            if any(key in self._deleting for group in key_groups for key in group):
                return AdmissionFailure.BUSY
            self._register_manifest(manifest)
            if manifest.generation not in self._publication_bytes:
                footprint = manifest.world_size * sum(
                    self._storage.checkpoint_allocation_bytes(group.page_bytes)
                    * len(group.positions)
                    for group in groups
                )
                if (
                    footprint + sum(self._publication_bytes.values())
                    > self._storage.checkpoint_publication_budget()
                ):
                    self._index.abort(manifest.generation)
                    return None
                self._publication_bytes[manifest.generation] = footprint
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
        reused: list[ObjectKey] = []

        def attempt() -> AdmissionAttempt[
            tuple[tuple[ShmSlotDescriptor | None, ...], ...]
        ]:
            with self._lock:
                if not self._index.is_pending(manifest):
                    return AdmissionAttempt.failure(AdmissionFailure.CONFLICT)
                if any(key in self._deleting for group in key_groups for key in group):
                    return AdmissionAttempt.failure(AdmissionFailure.BUSY)
            slots: list[tuple[ShmSlotDescriptor | None, ...]] = []
            for group, keys in zip(groups, key_groups, strict=True):
                detailed = self._storage.reserve_write_detailed(
                    list(keys),
                    MemoryLayoutDesc([torch.Size([group.page_bytes])], [torch.uint8]),
                    "new",
                    internal=True,
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
                readable = set(
                    self._storage.pin_readable_keys(
                        existing_candidates, retained_only=True
                    )
                )
                reused.extend(readable)
                if len(objects) + len(readable) != len(keys):
                    self._storage.abort_write(reserved)
                    reserved.clear()
                    self._storage.release_read_pins(reused)
                    reused.clear()
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
                self._stores[lease_id] = _StoreLease(manifest, rank, reserved, reused)
            admitted = True
            return CheckpointSlots(lease_id, outcome.value)
        finally:
            if not admitted:
                try:
                    self._storage.abort_write(reserved)
                    self._storage.release_read_pins(reused)
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
        with self._lock:
            pinned = list(lease.reused)
            try:
                if not success or not self._index.is_pending(lease.manifest):
                    self._storage.abort_write(lease.keys)
                    self._index.abort(lease.manifest.generation)
                    return False
                committed = self._storage.finish_write_pinned(lease.keys)
                pinned.extend(committed)
                if len(committed) != len(lease.keys):
                    self._storage.abort_write(lease.keys)
                    self._index.abort(lease.manifest.generation)
                    return False
                self._completed.setdefault(lease.manifest.generation, []).extend(pinned)
                self._completed_ranks.setdefault(lease.manifest.generation, set()).add(
                    lease.rank
                )
                pinned = []
                return self._index.acknowledge(lease.manifest.generation, lease.rank)
            except Exception:
                self._storage.abort_write(lease.keys)
                self._index.abort(lease.manifest.generation)
                raise
            finally:
                self._storage.release_read_pins(pinned)
                self._store_ranks.discard((lease.manifest.generation, lease.rank))

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
            if not self._accepting:
                return None
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
        """Cancel stale publication and unexposed lookups without recycling DMA.

        Completed ranks release their publication pins when their generation
        aborts. Live writers and exposed retrieves require explicit completion;
        their buffers remain bounded by the pool and lease admission limits.

        Returns:
            Number of cancelled lookups and aborted pending generations.
        """
        cutoff = time.monotonic() - self._abandoned_after
        with self._lock:
            lookups = [
                lease_id
                for lease_id, lease in self._retrieves.items()
                if lease.slots is None and lease.started < cutoff
            ]
            for lease_id in lookups:
                self._retrieves[lease_id].cancelled = True
        for lease_id in lookups:
            try:
                # A cancelled lookup releases its locks once its prefetch ends.
                self.poll_retrieve(lease_id)
            except KeyError:
                pass
        stale = self._index.abort_stale(self._abandoned_after)
        reclaimed = len(lookups) + len(stale)
        if reclaimed:
            logger.warning(
                "Cancelled %d lookups and %d "
                "pending checkpoint generations abandoned for over %.0f s",
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

    def quiesce(self) -> None:
        """Close admission atomically with the idle check.

        Raises RuntimeError if a worker still owns copy buffers or an admission
        is in progress. A failed attempt leaves the service usable so workers
        can finish and the caller can retry closure.
        """
        with self._lock:
            if self._store_ranks or self._retrieves or self._retrieve_admissions:
                raise RuntimeError(
                    "Checkpoint worker copy leases must drain before close"
                )
            self._accepting = False

    def prepare_terminal_shutdown(self) -> None:
        """Stop admission and flush the directory without releasing live copies.

        Terminal process shutdown may leave leases from workers already killed
        by the supervisor. Those buffers remain owned until process termination;
        closing their metadata never authorizes recycling their memory.
        """
        with self._lock:
            self._accepting = False
        self._index.close()

    def poll_retrieve(self, lease_id: str) -> CheckpointSlots | bool | None:
        """Return slots only when every required page is readable and pinned.

        Args:
            lease_id: Identifier from ``begin_retrieve``.

        Returns:
            None while prefetch is pending, False on a miss/cancellation, or
            the pinned slots. A miss releases all acquired locks and must not
            advance computed tokens. Reservation failures retry within the
            original admission deadline, using aligned allocation sizes.
            Unknown backend misses preserve the generation: failed I/O is not
            proof of missing stored pages. Coordinated retirement and recovery
            remove manifests whose payloads are known to be lost.

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
            pinned = self._storage.pin_readable_keys(lease.keys)
            if pinned != lease.keys:
                self._storage.release_read_pins(pinned)
                self._storage.finish_read_prefetched(lease.keys)
                return self._repeat_or_miss(lease_id, lease, [], True)
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
                self._storage.release_read_pins(pinned)
                self._storage.finish_read_prefetched(lease.keys)
                self._index.invalidate(lease.manifest.generation)
                del self._retrieves[lease_id]
                raise

    def _repeat_or_miss(
        self,
        lease_id: str,
        lease: _RetrieveLease,
        readable_keys: list[ObjectKey],
        reservation_failed: bool = False,
    ) -> bool | None:
        """Repeat or end a lookup whose pages were not all pinned.

        Called with the lookup's read locks already released. Returns None
        when the lookup will be repeated, otherwise False after forgetting the
        lease. An unavailable page does not prove that its stored copy is lost.
        """
        lease.readable = len(readable_keys)
        if lease.cancelled:
            return self._miss(lease_id, lease, "was cancelled by the engine")
        if self._index.get(lease.manifest.generation) is None:
            return self._miss(
                lease_id, lease, "missed; its checkpoint is no longer listed"
            )
        groups = checkpoint_page_groups(lease.manifest)
        readable = set(readable_keys)
        lease.unread_bytes = sum(
            self._storage.checkpoint_allocation_bytes(
                groups[key.object_group_id].page_bytes
            )
            for key in lease.keys
            if key not in readable
        )
        _, total = self._storage.get_l1_usage()
        if (
            self._storage.l2_adapters()
            and lease.unread_bytes <= total
            and (reservation_failed or lease.lookups == 1)
        ):
            lease.handle = None
            return self._repeat_with_room(lease_id, lease)
        # Adapters can report a miss for failed I/O as well as absence. Only
        # coordinated retirement/recovery provides authoritative loss evidence.
        return self._miss(
            lease_id, lease, "was unavailable; its checkpoint stays listed"
        )

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
            # Keep admission/shutdown accounting until ownership cleanup ends.
            try:
                self._storage.notify_keys_reused(lease.keys)
                self._storage.finish_read_prefetched(lease.keys)
            finally:
                self._storage.release_read_pins(lease.keys)
                del self._retrieves[lease_id]

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

    def _index_changed(self, event: str, value: CheckpointManifest | str) -> None:
        if event in {"publish", "touch"}:
            assert isinstance(value, CheckpointManifest)
            self._register_manifest(value)
            self._generations.move_to_end(value.generation)
            if event == "publish":
                self._published.add(value.generation)
        if event not in {"publish", "abort", "remove"}:
            return
        generation = (
            value.generation if isinstance(value, CheckpointManifest) else value
        )
        if event in {"abort", "remove"}:
            # Aborting an already published generation is a no-op.
            if event != "abort" or generation not in self._published:
                self._published.discard(generation)
                for key in self._generations.pop(generation, {}):
                    owners = self._owners[key]
                    owners.discard(generation)
                    if not owners:
                        del self._owners[key]
                        self._storage.checkpoint_retention.forget_pending([key])
        self._storage.release_read_pins(self._completed.pop(generation, []))
        self._completed_ranks.pop(generation, None)
        self._publication_bytes.pop(generation, None)

    def _before_index_close(self) -> None:
        if self._closed:
            return
        with self._lock:
            self._accepting = False
            for generation in list(self._generations):
                if generation in self._published:
                    continue
                self._index.abort(generation)
        try:
            self._storage.flush_checkpoints()
        except Exception:
            logger.exception(
                "Checkpoint shutdown flush failed; retiring RAM-only entries"
            )
        with self._lock:
            retention = self._storage.checkpoint_retention
            lost = [
                generation
                for generation in self._published
                if any(
                    not retention.is_l2_resident(key, size)
                    for key, size in self._generations[generation].items()
                )
            ]
            for generation in lost:
                self._index.invalidate(generation)
            if lost:
                logger.warning("Shutdown retired %d RAM-only checkpoints", len(lost))
                retention.record_retirement(len(lost))
            self._closed = True
            retention.set_lifecycle(evict=None, pressure=None, retained=None)

    def _register_manifest(self, manifest: CheckpointManifest) -> None:
        if manifest.generation in self._generations:
            return
        sizes: dict[ObjectKey, int] = {}
        groups = checkpoint_page_groups(manifest)
        for rank in range(manifest.world_size):
            for group, keys in zip(
                groups, checkpoint_object_keys(manifest, rank), strict=True
            ):
                sizes.update((key, group.page_bytes) for key in keys)
        self._generations[manifest.generation] = sizes
        for key in sizes:
            self._owners.setdefault(key, set()).add(manifest.generation)

    def _is_retained(self, key: ObjectKey) -> bool:
        with self._lock:
            return key in self._owners

    def _reconcile_replicas(self) -> None:
        """Adapter removal can invalidate unenumerated recovered dependencies."""
        with self._lock:
            retention = self._storage.checkpoint_retention
            for generation in list(self._published):
                keys = list(self._generations[generation])
                readable = set(
                    self._storage.get_readable_keys(keys, retained_only=True)
                )
                if any(
                    key not in readable
                    and not retention.is_l2_resident(
                        key, self._generations[generation][key]
                    )
                    for key in keys
                ):
                    self._index.invalidate(generation)
                    retention.record_retirement(1)

    def _persistence_order(self, keys: list[ObjectKey]) -> list[ObjectKey]:
        """Prefer candidates that complete a checkpoint with fewer missing bytes."""
        with self._lock:
            retention = self._storage.checkpoint_retention
            candidates = {owner for key in keys for owner in self._owners.get(key, ())}
            missing = {
                generation: [
                    key
                    for key in self._generations[generation]
                    if not retention.is_l2_resident(
                        key, self._generations[generation][key]
                    )
                ]
                for generation in candidates
            }
            ranked = sorted(
                [
                    generation
                    for generation in reversed(self._generations)
                    if generation in candidates
                ],
                key=lambda generation: sum(
                    self._generations[generation][key] for key in missing[generation]
                ),
            )
            ordered = dict.fromkeys(
                key for generation in ranked for key in missing[generation]
            )
            return list(ordered)

    def _evict_pages(
        self,
        keys: list[ObjectKey],
        tier: str | int,
        delete: Callable[[list[ObjectKey]], _DeleteResult],
    ) -> _DeleteResult:
        """Fence payload deletion and retire affected last-copy generations."""
        keys = list(dict.fromkeys(keys))
        force = tier == "l1-force"
        if force:
            tier = "l1"
        with self._lock:
            if self._closed:
                return delete(keys)
            retention = self._storage.checkpoint_retention
            readable = (
                set(self._storage.get_readable_keys(keys, retained_only=True))
                if tier != "l1"
                else set()
            )
            selected = []
            retired: set[str] = set()
            for key in keys:
                if tier in self._deleting.get(key, set()):
                    continue
                if (
                    tier == "l1"
                    and not force
                    and not self._storage.is_l1_evictable(key)
                ):
                    continue
                deleting = self._deleting.get(key, set())
                owners = self._owners.get(key, set())
                retained_l1 = tier != "l1" and key in readable and "l1" not in deleting
                at_risk = {
                    generation
                    for generation in owners
                    if not retained_l1
                    and not (
                        retention.resident_adapters(
                            key, self._generations[generation][key]
                        )
                        - deleting
                        - {tier}
                    )
                }
                if at_risk - self._published and not force:
                    continue
                if force:
                    for generation in list(at_risk - self._published):
                        self._index.abort(generation)
                retired.update(at_risk & self._published)
                selected.append(key)
            for generation in retired:
                self._index.invalidate(generation)
            retention.record_retirement(len(retired))
            for key in selected:
                self._deleting.setdefault(key, set()).add(tier)
        try:
            return delete(selected)
        finally:
            with self._lock:
                for key in selected:
                    self._deleting[key].discard(tier)
                    if not self._deleting[key]:
                        del self._deleting[key]

    def _retire_under_pressure(self, target_bytes: int) -> int:
        """Free unique pages of cold, inactive generations, preserving sharing."""
        freed = 0
        with self._lock:
            candidates = list(self._generations)
        for generation in candidates:
            if freed >= target_bytes:
                break
            with self._lock:
                if generation not in self._published:
                    continue
                sizes = self._generations[generation]
                unique = [
                    key
                    for key in sizes
                    if self._owners.get(key) == {generation}
                    and (
                        self._storage.is_l1_evictable(key)
                        or self._storage.can_cancel_persistence(key)
                    )
                ]
                if not unique:
                    continue
                # Shared dependencies survive retirement and need no extra write.
                self._index.invalidate(generation)
                self._storage.checkpoint_retention.record_retirement(1)
                self._storage.cancel_queued_persistence(unique)
                self._storage.delete_l1_keys(unique)
                # Bound selection by bytes retired, including queued pins that
                # the store loop will release. Admission still checks actual RAM.
                freed += sum(
                    self._storage.checkpoint_allocation_bytes(sizes[key])
                    for key in unique
                )
                logger.info("Retired checkpoint under RAM pressure: %s", generation)
        return freed
