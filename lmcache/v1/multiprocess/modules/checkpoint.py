# SPDX-License-Identifier: Apache-2.0
"""CPU-only SHM service for atomic recurrent checkpoint generations."""

# Standard
from collections import OrderedDict
from pathlib import Path
import json
import threading

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

    Shutdown requires workers to finish or abort all submitted copy leases.
    A timeout must not recycle SHM while a worker can still access its bytes.
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
        return self._index.begin(manifest)

    def find(self, prefixes: tuple[CheckpointPrefix, ...]) -> CheckpointManifest | None:
        """Return the longest complete candidate for authenticated prefix roots.

        A candidate is not a cache hit until all payload ranks restore it.
        Every rank's payload pages are refreshed in L1 and L2 eviction order:
        a resumed conversation is often served from the engine's own cache,
        so its pages are neither read nor rewritten, and its oldest pages,
        which every later checkpoint of the conversation needs, would
        otherwise keep the recency of their first write.
        """
        manifest = self._index.find(prefixes)
        if manifest is not None:
            try:
                keys = _payload_keys(manifest)
            except ValueError:
                # Retrieval rejects the same manifest and invalidates it.
                return manifest
            self._ctx.storage_manager.touch_keys(keys)
        return manifest

    def supersede(
        self, prefixes: tuple[CheckpointPrefix, ...], generation: str, request: str
    ) -> int:
        """Mark the pages only superseded checkpoints still use.

        Called after ``generation`` is published by ``request``, with the
        roots of the token sequence that produced it. Only a ``prompt``
        checkpoint supersedes: the complete prompt of a new request replaces
        every published shorter checkpoint of its sequence that an earlier
        request produced, together with the other checkpoints of those
        requests that it has moved past (a response endpoint that the chat
        template rewrote, a prefill tail). A checkpoint of such a request
        that is at least as long as the new prompt is kept: the new request
        branched off before it (a retry, or an aborted turn resumed with
        other tokens), and the original line can still continue from it.
        Checkpoints of the same request and ``instruction`` checkpoints,
        which other conversations share, are kept. Pages the new checkpoint
        references are current again.

        Superseded pages are evicted first and are never written to L2 on
        eviction. Their manifests stay listed, so a request that branches
        from an older turn can still restore one while its pages last.

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
        victims: dict[str, CheckpointManifest] = {}
        for ancestor in self._index.ancestors(prefixes, current.prefix.num_tokens):
            owner = self._request_of(ancestor.generation)
            if owner == request:
                continue
            victims[ancestor.generation] = ancestor
            for sibling in self._generations_of(owner):
                if sibling not in victims and sibling != generation:
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
                "(%d pages)",
                current.prefix.num_tokens,
                len(selected),
                marked,
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
        self._index.abort(generation)
        return True

    def prepare_store(
        self, manifest: CheckpointManifest, rank: int
    ) -> CheckpointLeaseResponse:
        """Return all writable rank slots or miss without partial admission.

        Invalid layouts, ranks or duplicate active rank stores raise ValueError.
        """
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
        return self._payloads.finish_store(lease_id, success)

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

    def close(self) -> None:
        """Close the directory after copy leases drain, otherwise raise RuntimeError.

        This module never frees buffers solely because a copy took too long.
        The owning process shutdown must coordinate GPU worker termination.
        """
        status = self._payloads.report_status()
        if status["store_leases"] or status["retrieve_leases"]:
            raise RuntimeError("Checkpoint worker copy leases must drain before close")
        self._index.close()
