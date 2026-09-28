# SPDX-License-Identifier: Apache-2.0
"""CPU-only SHM service for atomic recurrent checkpoint generations."""

# Standard
from collections import OrderedDict
from pathlib import Path
import json
import threading
import time

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.admission import AdmissionFailure
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.multiprocess.checkpoint_index import (
    CheckpointIndex,
    CheckpointManifest,
    CheckpointPrefix,
)
from lmcache.v1.multiprocess.checkpoint_storage import (
    CheckpointPayloadStore,
    CheckpointSlots,
    checkpoint_object_keys,
    checkpoint_page_groups,
)
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
from lmcache.v1.multiprocess.engine_module import HandlerSpec, ThreadPoolType
from lmcache.v1.multiprocess.protocols.base import RequestType
from lmcache.v1.multiprocess.protocols.checkpoint import (
    CheckpointCapabilities,
    CheckpointLeaseResponse,
)

logger = init_logger(__name__)

# Checkpoints superseded per call, newest first. Older ones were handled by
# earlier turns or have long left the cache tiers.
_MAX_SUPERSEDED_ANCESTORS = 32
# Requests whose published generations are remembered for supersession.
_MAX_TRACKED_REQUESTS = 65536
# Default manifest capacity. A persistent directory should cover what L2
# retains; a RAM-only directory holds manifests in server memory (tens to
# hundreds of KiB each for long GLM prompts) while its payloads live only in
# L1, which holds far fewer checkpoints.
_PERSISTENT_INDEX_MAX_ENTRIES = 65536
_RAM_INDEX_MAX_ENTRIES = 8192
# Candidates a lookup may retire because no tier holds their pages any more
# before it gives up; each is one directory query and one residency check.
_MAX_LOOKUP_RETIREMENTS = 256
# While the server stops: with no store in flight, how long to wait after the
# last store RPC for a straggler (a request finishing as the engine stops
# begins its last store at once), and with stores in flight, how long without
# any store RPC before their producer is taken as gone.
_DRAIN_SETTLE_SECONDS = 0.5
_DRAIN_IDLE_SECONDS = 3.0


def _kind(manifest: CheckpointManifest) -> str | None:
    try:
        return json.loads(manifest.payload).get("kind")
    except (ValueError, AttributeError):
        return None


def _payload_keys(manifest: CheckpointManifest) -> list[ObjectKey]:
    return [
        key
        for rank in range(manifest.world_size)
        for group in checkpoint_object_keys(manifest, rank)
        for key in group
    ]


def _ready(slots: CheckpointSlots) -> CheckpointLeaseResponse:
    return CheckpointLeaseResponse(
        "ready",
        slots.lease_id,
        tuple(
            tuple(
                (slot.offset, slot.length) if slot is not None else (-1, 0)
                for slot in group
            )
            for group in slots.groups
        ),
    )


class CheckpointModule:
    """Publish complete all-rank manifests and lease their payload pages.

    Args:
        ctx: Existing CPU SHM storage context; no CUDA registration is created.
        index_path: SQLite file in a trusted, existing directory, or None for
            a RAM-only directory. Filesystem payloads alone do not imply a
            durable manifest directory.
        max_leases: Shared limit for pending rank stores and retrieves.
        abandoned_lease_seconds: Age after which leases and pending
            generations of a dead worker are released; 0 disables it.
        index_max_entries: Published manifests kept before LRU removal, or
            None for 65536 with ``index_path`` and 8192 for a RAM-only
            directory.

    At shutdown, :meth:`drain_stores` gives workers a bounded time to finish
    or abort their copy leases while the message queue still serves them.
    A lease left after that makes :meth:`close` raise; its buffers are never
    recycled while a worker can still access their bytes.
    """

    def __init__(
        self,
        ctx: MPCacheServerContext,
        index_path: Path | None = None,
        *,
        max_leases: int = 1024,
        index_max_entries: int | None = None,
        abandoned_lease_seconds: float = 600.0,
    ) -> None:
        if not ctx.shm_pool_info["shm_name"] or ctx.shm_pool_info["pool_size"] <= 0:
            raise ValueError("Recurrent checkpoint transfers require an SHM pool")
        self._ctx = ctx
        if index_max_entries is None:
            index_max_entries = (
                _PERSISTENT_INDEX_MAX_ENTRIES
                if index_path is not None
                else _RAM_INDEX_MAX_ENTRIES
            )
        self._index = CheckpointIndex(index_path, max_entries=index_max_entries)
        self._payloads = CheckpointPayloadStore(
            ctx.storage_manager,
            self._index,
            max_leases=max_leases,
            abandoned_after_seconds=abandoned_lease_seconds,
        )
        self._capabilities = CheckpointCapabilities(
            format_version=1,
            shm_name=ctx.shm_pool_info["shm_name"],
            pool_size=ctx.shm_pool_info["pool_size"],
            durable_index=index_path is not None,
        )
        # Generations each recent request published, and the reverse map.
        self._lineage_lock = threading.Lock()
        self._request_generations: OrderedDict[str, list[str]] = OrderedDict()
        self._generation_request: dict[str, str] = {}
        # Monotonic time of the last store RPC, for the shutdown drain.
        self._last_store_activity = 0.0
        ctx.storage_manager.checkpoint_retention.set_retirement(
            self._retire_generations
        )

    @property
    def context(self) -> MPCacheServerContext:
        """Return the shared server context without transferring ownership."""
        return self._ctx

    def get_handlers(self) -> list[HandlerSpec]:
        """Return metadata-only RPC handlers on the ordinary CPU thread pool."""
        handlers = (
            (RequestType.CHECKPOINT_FIND, self.find),
            (RequestType.CHECKPOINT_BEGIN, self.begin),
            (RequestType.CHECKPOINT_ABORT, self.abort),
            (RequestType.CHECKPOINT_PREPARE_STORE, self.prepare_store),
            (RequestType.CHECKPOINT_FINISH_STORE, self.finish_store),
            (RequestType.CHECKPOINT_BEGIN_RETRIEVE, self.begin_retrieve),
            (RequestType.CHECKPOINT_POLL_RETRIEVE, self.poll_retrieve),
            (RequestType.CHECKPOINT_FINISH_RETRIEVE, self.finish_retrieve),
            (RequestType.CHECKPOINT_CANCEL_RETRIEVE, self.cancel_retrieve),
            (RequestType.CHECKPOINT_SUPERSEDE, self.supersede),
        )
        return [
            HandlerSpec(
                RequestType.CHECKPOINT_CAPABILITIES,
                self.capabilities,
                ThreadPoolType.SYNC,
            ),
            *(
                HandlerSpec(request, handler, ThreadPoolType.NORMAL)
                for request, handler in handlers
            ),
        ]

    def capabilities(self) -> CheckpointCapabilities:
        """Return the supported format and exact SHM pool identity."""
        return self._capabilities

    def begin(self, manifest: CheckpointManifest) -> bool:
        """Stage a validated manifest; False means directory admission failed.

        The manifest's generation must be shared by all producer ranks. A
        changed or already published generation raises ValueError.
        """
        checkpoint_page_groups(manifest)
        self._last_store_activity = time.monotonic()
        return self._index.begin(manifest)

    def find(self, prefixes: tuple[CheckpointPrefix, ...]) -> CheckpointManifest | None:
        """Return the longest complete candidate for authenticated prefix roots.

        A candidate is not a cache hit until all payload ranks restore it.
        A listed candidate whose pages no tier holds any more (dropped,
        evicted or deleted since it was published, or not written before a
        restart) is retired from the directory, and the next shorter one is
        tried at once. This needs L2 adapters whose inventory is complete;
        otherwise the restore validates the pages.

        Every rank's payload pages of the returned candidate are refreshed in
        L1 and L2 eviction order: a resumed conversation is often served from
        the engine's own cache, so its pages are neither read nor rewritten,
        and its oldest pages, which every later checkpoint of the
        conversation needs, would otherwise keep the recency of their first
        write. A superseded candidate becomes current again: a request is
        continuing from it.
        """
        storage = self._ctx.storage_manager
        retention = storage.checkpoint_retention
        retired: list[tuple[int, int, int]] = []
        found: CheckpointManifest | None = None
        for _ in range(_MAX_LOOKUP_RETIREMENTS + 1):
            manifest = self._index.find(prefixes)
            if manifest is None:
                break
            try:
                keys = _payload_keys(manifest)
            except ValueError:
                # Retrieval rejects the same manifest and invalidates it.
                found = manifest
                break
            lost = retention.unavailable_pages(keys)
            if not lost:
                storage.touch_keys(keys)
                if retention.mark_generation_current(manifest.generation, keys):
                    logger.debug(
                        "Superseded checkpoint of %d tokens was found again; "
                        "it is current",
                        manifest.prefix.num_tokens,
                    )
                found = manifest
                break
            self._index.invalidate(manifest.generation)
            retired.append((manifest.prefix.num_tokens, len(lost), len(keys)))
        if retired:
            tokens, lost_pages, pages = retired[0]
            logger.info(
                "Checkpoint lookup retired %d listed checkpoints whose pages no "
                "tier holds (longest %d tokens, %d of %d pages lost); %s",
                len(retired),
                tokens,
                lost_pages,
                pages,
                f"found one of {found.prefix.num_tokens} tokens"
                if found is not None
                else "found none",
            )
        return found

    def supersede(
        self, prefixes: tuple[CheckpointPrefix, ...], generation: str, request: str
    ) -> int:
        """Mark the pages only superseded checkpoints still use.

        Called after ``generation`` is published by ``request``, with the
        roots of the token sequence that produced it. Only a ``prompt``
        checkpoint supersedes. Its longest published ancestor from an earlier
        request is the checkpoint the new prompt continues from; it stays
        current, because the new request may be a side request (a title or
        summary request, a sub-agent fork) while the conversation goes on
        from the same point later. The other published ancestors of the
        sequence from earlier requests are superseded: the conversation has
        moved past them twice. So are the other checkpoints of their requests
        that the new prompt has moved past (a response endpoint that the chat
        template rewrote, a prefill tail). A checkpoint of such a request
        that is at least as long as the new prompt is kept: the new request
        branched off before it (a retry, or an aborted turn resumed with
        other tokens), and the original line can still continue from it.
        Checkpoints of the same request and ``instruction`` checkpoints,
        which other conversations share, are kept. Pages the new checkpoint
        references are current again.

        Superseded pages are never written to L2 while serving. They are
        evicted first, at once, or after the grace period when a later prompt
        had continued from their checkpoint. When a superseded page's last
        copy is deleted, its checkpoints are retired from the directory.

        Returns:
            Number of pages newly marked superseded.
        """
        current = self._index.get(generation)
        if current is None or not prefixes:
            return 0
        if any(prefix.namespace != current.prefix.namespace for prefix in prefixes):
            raise ValueError("checkpoint supersession mixes namespaces")
        self._remember(request, generation)
        retention = self._ctx.storage_manager.checkpoint_retention
        current_keys = set(_payload_keys(current))
        retention.mark_current(current_keys)
        if _kind(current) != "prompt":
            return 0
        ancestors = [
            ancestor
            for ancestor in self._index.ancestors(prefixes, current.prefix.num_tokens)
            if self._request_of(ancestor.generation) != request
        ]
        if not ancestors:
            return 0
        parent = ancestors[-1]
        retention.mark_continued(parent.generation)
        retention.mark_generation_current(parent.generation, _payload_keys(parent))
        victims: dict[str, CheckpointManifest] = {
            ancestor.generation: ancestor for ancestor in ancestors[:-1]
        }
        for ancestor in ancestors:
            for sibling in self._generations_of(self._request_of(ancestor.generation)):
                if sibling in victims or sibling in (generation, parent.generation):
                    continue
                manifest = self._index.get(sibling)
                if (
                    manifest is not None
                    and manifest.prefix.num_tokens < current.prefix.num_tokens
                ):
                    victims[sibling] = manifest
        selected = sorted(
            (
                victim
                for victim in victims.values()
                if _kind(victim) != "instruction"
                and victim.prefix.namespace == current.prefix.namespace
                and not retention.is_superseded_generation(victim.generation)
            ),
            key=lambda victim: victim.prefix.num_tokens,
        )[-_MAX_SUPERSEDED_ANCESTORS:]
        marked = 0
        for victim in selected:
            unique = [key for key in _payload_keys(victim) if key not in current_keys]
            marked += retention.mark_superseded(victim.generation, unique)
        if selected:
            logger.debug(
                "Prompt checkpoint of %d tokens superseded %d older checkpoints "
                "(%d pages); it continues from one of %d tokens",
                current.prefix.num_tokens,
                len(selected),
                marked,
                parent.prefix.num_tokens,
            )
        return marked

    def _remember(self, request: str, generation: str) -> None:
        with self._lineage_lock:
            generations = self._request_generations.setdefault(request, [])
            self._request_generations.move_to_end(request)
            if generation not in generations:
                generations.append(generation)
                self._generation_request[generation] = request
            while len(self._request_generations) > _MAX_TRACKED_REQUESTS:
                _, forgotten = self._request_generations.popitem(last=False)
                for old in forgotten:
                    self._generation_request.pop(old, None)

    def _request_of(self, generation: str) -> str | None:
        with self._lineage_lock:
            return self._generation_request.get(generation)

    def _generations_of(self, request: str | None) -> list[str]:
        if request is None:
            return []
        with self._lineage_lock:
            return list(self._request_generations.get(request, ()))

    def abort(self, generation: str) -> bool:
        """Prevent publication; rank copy leases still require explicit finish."""
        self._last_store_activity = time.monotonic()
        self._index.abort(generation)
        return True

    def prepare_store(
        self, manifest: CheckpointManifest, rank: int
    ) -> CheckpointLeaseResponse:
        """Return all writable rank slots or miss without partial admission.

        Invalid layouts, ranks or duplicate active rank stores raise ValueError.
        """
        self._last_store_activity = time.monotonic()
        admission = self._payloads.prepare_store(manifest, rank)
        if admission is AdmissionFailure.BUSY:
            return CheckpointLeaseResponse("busy")
        return (
            _ready(admission)
            if isinstance(admission, CheckpointSlots)
            else CheckpointLeaseResponse("miss")
        )

    def finish_store(self, lease_id: str, success: bool) -> bool:
        """Finish drained D2H work; True means all ranks published the manifest."""
        self._last_store_activity = time.monotonic()
        try:
            return self._payloads.finish_store(lease_id, success)
        finally:
            self._last_store_activity = time.monotonic()

    def begin_retrieve(
        self, manifest: CheckpointManifest, rank: int
    ) -> CheckpointLeaseResponse:
        """Start an asynchronous all-page rank lookup or reject lease admission."""
        lease_id = self._payloads.begin_retrieve(manifest, rank)
        return (
            CheckpointLeaseResponse("pending", lease_id)
            if lease_id is not None
            else CheckpointLeaseResponse("miss")
        )

    def poll_retrieve(self, lease_id: str) -> CheckpointLeaseResponse:
        """Return pending, complete pinned slots, or an atomic rank miss.

        Unknown or completed leases raise KeyError. Missing data never yields
        partial slots; malformed byte layouts raise ValueError after cleanup.
        """
        result = self._payloads.poll_retrieve(lease_id)
        if isinstance(result, CheckpointSlots):
            return _ready(result)
        return CheckpointLeaseResponse(
            "pending" if result is None else "miss", lease_id
        )

    def finish_retrieve(self, lease_id: str) -> bool:
        """Release a read lease after H2D drains; pending lookup raises ValueError."""
        self._payloads.finish_retrieve(lease_id)
        return True

    def cancel_retrieve(self, lease_id: str) -> bool:
        """Cancel unexposed lookup; callers must poll until its locks drain.

        Once slots are exposed, use finish_retrieve after H2D completion instead.
        """
        self._payloads.cancel_retrieve(lease_id)
        return True

    def report_status(self) -> dict[str, dict[str, int]]:
        """Report capability and live leases without advancing or releasing work."""
        return {
            "recurrent_checkpoints": {
                "format_version": self._capabilities.format_version,
                "durable_index": self._capabilities.durable_index,
                **self._index.report_status(),
                **self._payloads.report_status(),
            }
        }

    def drain_stores(self, deadline: float) -> None:
        """Keep serving checkpoint stores until they settle or ``deadline``.

        Call it when the server starts to stop, while its message queue still
        serves requests. An engine stopping at the same time may still be
        copying its last checkpoints; each rank that finishes publishes them,
        so the shutdown flush can write them to L2.

        Returns once no store lease or unpublished generation remains and no
        store RPC arrived for a short settle time, once no store RPC arrived
        for a few seconds while stores are still in flight (their producer is
        gone), or at ``deadline``. New stores are admitted meanwhile.

        Args:
            deadline: Monotonic time at which to stop waiting.
        """
        start = time.monotonic()
        announced = False
        while True:
            now = time.monotonic()
            in_flight = (
                self._payloads.report_status()["store_leases"]
                + self._index.pending_count()
            )
            quiet = now - self._last_store_activity
            if (
                (not in_flight and quiet >= _DRAIN_SETTLE_SECONDS)
                or quiet >= _DRAIN_IDLE_SECONDS
                or now >= deadline
            ):
                break
            if not announced:
                announced = True
                logger.info(
                    "Serving checkpoint stores for up to %.1f s before shutdown "
                    "(%d in flight)",
                    max(0.0, deadline - now),
                    in_flight,
                )
            time.sleep(0.02)
        if in_flight:
            logger.warning(
                "%d checkpoint stores were still in flight after %.1f s of "
                "shutdown; those checkpoints stay unpublished",
                in_flight,
                time.monotonic() - start,
            )
        elif announced:
            logger.info(
                "Checkpoint stores settled after %.1f s of shutdown",
                time.monotonic() - start,
            )

    def _retire_generations(self, generations: list[str]) -> None:
        """Delist superseded checkpoints whose last page copy is being deleted."""
        try:
            self._index.retire(generations)
        except Exception:
            # The directory may already be closed during shutdown; a listed
            # checkpoint without pages is retired by the next lookup instead.
            logger.debug("Could not retire %d checkpoints", len(generations))
            return
        logger.debug(
            "Retired %d superseded checkpoints whose last pages left the cache",
            len(generations),
        )

    def close(self) -> None:
        """Close the directory after copy leases drain, otherwise raise RuntimeError.

        This module never frees buffers solely because a copy took too long.
        The owning process shutdown must coordinate GPU worker termination:
        :meth:`drain_stores` gives workers a bounded time first, and
        ``MPCacheServer.close`` logs this error and still closes the storage
        manager, so its shutdown flush runs and its shared memory is released
        without reusing a lease's buffers.
        """
        status = self._payloads.report_status()
        if status["store_leases"] or status["retrieve_leases"]:
            raise RuntimeError(
                f"Checkpoint worker copy leases must drain before close "
                f"({status['store_leases']} store, "
                f"{status['retrieve_leases']} retrieve)"
            )
        self._ctx.storage_manager.checkpoint_retention.clear_retirement(
            self._retire_generations
        )
        self._index.close()
