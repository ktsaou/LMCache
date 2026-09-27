# SPDX-License-Identifier: Apache-2.0
"""Scheduler ownership and all-rank publication of external recurrent checkpoints."""

# Standard
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal
import json
import math
import os
import time
import uuid

# First Party
from lmcache.logging import init_logger
from lmcache.v1.multiprocess.checkpoint_identity import (
    CheckpointTokenRoots,
    checkpoint_generation,
    checkpoint_namespace,
)
from lmcache.v1.multiprocess.checkpoint_index import CheckpointManifest
from lmcache.v1.multiprocess.checkpoint_storage import checkpoint_page_groups
from lmcache.v1.multiprocess.futures import MessagingFuture
from lmcache.v1.multiprocess.mq import MessageQueueClient
from lmcache.v1.multiprocess.protocols.base import RequestType

logger = init_logger(__name__)

# Directory lookups per request, including the first. A failed restore makes
# the server invalidate the missing generation, so each retry receives the
# longest candidate that is still listed rather than recomputing the prompt.
_MAX_LOOKUP_ATTEMPTS = 4

# Restores slower than this many seconds are logged even when they succeed;
# the request waits in the scheduler, deferred, for the whole restore.
_SLOW_RESTORE_SECONDS = 10.0

# Settings that change encoder outputs, and so the KV of multimodal spans,
# without changing the processor's content hash. They salt multimodal roots
# only, so text namespaces are unaffected.
_MULTIMODAL_ENVIRONMENT = ("VLLM_GLM53_VISION_MXFP8",)

if TYPE_CHECKING:
    # Third Party
    from vllm.v1.core.boundary_checkpoint import BoundaryCheckpoint
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.request import Request


@dataclass(frozen=True)
class CheckpointEngineTask:
    """Ephemeral scheduler-to-worker transfer command, never persisted to storage."""

    task_id: str
    manifest: CheckpointManifest
    direction: Literal["STORE", "RETRIEVE"]
    block_ids: tuple[tuple[int, ...], ...]


@dataclass
class _PendingTask:
    task: CheckpointEngineTask
    checkpoint: "BoundaryCheckpoint"
    request_id: str
    begin: MessagingFuture[bool] | None = None
    acknowledgements: dict[int, bool] = field(default_factory=dict)
    sent: bool = False
    created: float = field(default_factory=time.monotonic)
    # Roots of the producing sequence, for superseding its older checkpoints.
    roots: CheckpointTokenRoots | None = None


@dataclass
class _Lookup:
    roots: CheckpointTokenRoots
    future: MessagingFuture[CheckpointManifest | None]
    done: bool = False
    task_id: str | None = None
    checkpoint_id: int | None = None
    local_checkpoint_id: int | None = None
    attempts: int = 1
    started: float = field(default_factory=time.monotonic)
    capacity_refusal_logged: bool = False
    task_refusal_logged: bool = False


class CheckpointSchedulerBridge:
    """Keep imports private and store sources pinned until every rank completes.

    Args:
        manager: vLLM allocator with request-boundary checkpoint support enabled.
        client: Shared thread-safe LMCache queue client; no ownership transfer.
        identity: Immutable target/draft/source revisions and parallel geometry.
            Worker layout is added after all ranks report identical descriptors.
        world_size: Number of engine ranks contributing to one atomic generation.
        max_tasks: Admission limit for collective stores and restores.
        lookup_timeout: Seconds allowed for each directory reply (including
            retries) and for a store's begin reply. Capacity waiting does not
            consume this deadline. An unanswered lookup permits prompt
            recomputation; an unanswered store is aborted.
            A started worker copy is never abandoned.

    The scheduler must keep issuing connector-only steps while has_pending is
    true, including after a producer request finishes. Request cancellation
    never releases pages still referenced by an admitted worker copy.
    """

    def __init__(
        self,
        manager: "KVCacheManager",
        client: MessageQueueClient,
        identity: dict[str, object],
        world_size: int,
        *,
        max_tasks: int = 32,
        lookup_timeout: float = 60.0,
    ) -> None:
        if manager.boundary_checkpoints is None or world_size < 1 or max_tasks < 1:
            raise ValueError(
                "Semantic transfers require a boundary allocator and ranks"
            )
        if not 0 < lookup_timeout < math.inf:
            raise ValueError("Checkpoint lookup timeout must be finite and positive")
        self._lookup_timeout = lookup_timeout
        self._manager = manager
        self._cache = manager.boundary_checkpoints
        self._client = client
        self._identity = dict(identity)
        self._multimodal_salt = json.dumps(
            {name: os.environ.get(name) for name in _MULTIMODAL_ENVIRONMENT},
            sort_keys=True,
        ).encode()
        self._world_size = world_size
        self._max_tasks = max_tasks
        self._layouts: dict[int, dict[str, Any]] = {}
        self._layout: dict[str, Any] | None = None
        self._lookups: dict[str, _Lookup] = {}
        self._tasks: dict[str, _PendingTask] = {}
        self._cancelled: set[str] = set()

    @property
    def has_pending(self) -> bool:
        """Whether connector-only steps are needed to negotiate or drain copies."""
        return self._layout is None or bool(self._tasks)

    def handles(self, request: "Request") -> bool:
        """Require a supported request whose weight identity is authenticated.

        Per-request LoRA content revisions are not part of the manifest namespace;
        such requests may use GPU-local caching but never this external directory.
        """
        return request.lora_request is None and self._cache.supports_request(request)

    def accept_layouts(self, layouts: dict[int, dict[str, Any]]) -> None:
        """Validate identical byte interpretation across all contributing ranks.

        Args:
            layouts: Address-free descriptors emitted by worker initialization.

        Raises:
            ValueError: If any rank is invalid or changes its byte layout.
        """
        for rank, layout in layouts.items():
            if not 0 <= rank < self._world_size:
                raise ValueError("Checkpoint layout came from an unknown rank")
            if self._layouts and layout != next(iter(self._layouts.values())):
                raise ValueError("Checkpoint byte layouts must match on every rank")
            self._layouts[rank] = layout
        if len(self._layouts) == self._world_size:
            self._layout = next(iter(self._layouts.values()))
            self._identity["layout"] = self._layout

    def poll_prefix(self, request: "Request") -> bool:
        """Start/poll an import and permit scheduling only after collective completion.

        Args:
            request: Waiting request whose prefix has not been admitted yet.

        Returns:
            False while layout negotiation, lookup or H2D is pending. True when
            ordinary GPU admission may proceed. Capacity and copy-task pressure
            keep an available checkpoint waiting without another lookup.
            Reusing a finished request's ID waits for its admitted copies to
            drain, so their completion cannot cancel or erase another lookup.
        """
        if request.request_id in self._cancelled:
            return False
        if not self.handles(request) or not self._manager.prefix_cache_lookup_enabled(
            request
        ):
            return True
        local = self._cache.find(request, request.num_tokens)
        state = self._lookups.get(request.request_id)
        if (
            state is None
            and local is not None
            and local.num_tokens == request.num_tokens
        ):
            return True
        if self._layout is None:
            return False
        if state is None:
            roots = self._roots(request)
            state = _Lookup(
                roots,
                self._client.submit_request(RequestType.CHECKPOINT_FIND, [roots.roots]),
            )
            self._lookups[request.request_id] = state
            return False
        if state.done:
            # Local reuse can also disappear while ordinary admission waits.
            selected_id = (
                state.local_checkpoint_id
                if state.local_checkpoint_id is not None
                else state.checkpoint_id
            )
            if (
                selected_id is not None
                and not (
                    self._manager.external_boundary_admission_ready(request.request_id)
                )
                and (local is None or local.checkpoint_id != selected_id)
            ):
                logger.info(
                    "Recurrent checkpoint invalidated before admission for "
                    "request %s; looking up its prefix again",
                    request.request_id,
                )
                self._manager.release_external_boundary_admission(request.request_id)
                del self._lookups[request.request_id]
                return self.poll_prefix(request)
            return True
        if state.task_id is not None:
            return False
        if not state.future.query():
            if time.monotonic() - state.started < self._lookup_timeout:
                return False
            # An unanswered lookup must not park the request indefinitely.
            logger.warning(
                "Recurrent checkpoint lookup for request %s got no reply within "
                "%.0f s; recomputing its prompt",
                request.request_id,
                self._lookup_timeout,
            )
            state.done = True
            return True
        try:
            manifest = state.future.result()
        except Exception:
            logger.exception(
                "Recurrent checkpoint lookup failed for request %s; recomputing "
                "its prompt",
                request.request_id,
            )
            state.done = True
            return True
        if manifest is None:
            state.done = True
            return True
        if local is not None and manifest.prefix.num_tokens <= local.num_tokens:
            # Keep local selection separate from external cache-hit accounting.
            state.local_checkpoint_id = local.checkpoint_id
            state.done = True
            return True
        if len(self._tasks) >= self._max_tasks:
            if not state.task_refusal_logged:
                state.task_refusal_logged = True
                logger.info(
                    "Recurrent checkpoint restore of %d tokens waiting for "
                    "copy capacity for request %s (%d copies in flight)",
                    manifest.prefix.num_tokens,
                    request.request_id,
                    len(self._tasks),
                )
            return False
        try:
            payload, positions = self._validate_manifest(manifest, state.roots)
            checkpoint = self._manager.reserve_external_boundary_checkpoint(
                request,
                manifest.prefix.num_tokens,
                positions,
                draft_prefix_len=payload["draft_prefix_len"],
                kind=payload["kind"],
                num_ranks=self._world_size,
                reserve_admission=True,
            )
        except (ValueError, KeyError, TypeError):
            logger.warning(
                "Recurrent checkpoint restore of %d tokens rejected for request "
                "%s; recomputing its prompt",
                manifest.prefix.num_tokens,
                request.request_id,
                exc_info=True,
            )
            state.done = True
            return True
        if checkpoint is None:
            # Resource pressure is not a cache miss. Keep the answered manifest
            # while runnable owners release capacity.
            if not state.capacity_refusal_logged:
                state.capacity_refusal_logged = True
                logger.info(
                    "Recurrent checkpoint restore of %d tokens deferred for "
                    "request %s: waiting for restore and execution capacity "
                    "(%d free GPU blocks)",
                    manifest.prefix.num_tokens,
                    request.request_id,
                    self._manager.block_pool.get_num_free_blocks(),
                )
            return False
        task = self._make_task(manifest, checkpoint, "RETRIEVE")
        logger.debug(
            "Recurrent checkpoint restore of %d tokens copying for request %s",
            manifest.prefix.num_tokens,
            request.request_id,
        )
        self._tasks[task.task_id] = _PendingTask(task, checkpoint, request.request_id)
        state.task_id = task.task_id
        return False

    def external_tokens(self, request: "Request") -> int:
        """Attribute only this request's selected imported bundle to external cache."""
        state = self._lookups.get(request.request_id)
        checkpoint = request.boundary_checkpoint
        if (
            state is not None
            and checkpoint is not None
            and state.checkpoint_id == checkpoint.checkpoint_id
        ):
            return checkpoint.num_tokens
        return 0

    def store(self, request: "Request", checkpoint: "BoundaryCheckpoint") -> None:
        """Pin a committed checkpoint and asynchronously stage its storage generation.

        Args:
            request: Producer with exact target tokens, including committed response.
            checkpoint: Published immutable GPU bundle, not a live request block table.
        """
        if (
            not self.handles(request)
            or self._layout is None
            or len(self._tasks) >= self._max_tasks
        ):
            return
        pinned = self._cache.acquire(checkpoint.checkpoint_id)
        if pinned is None:
            return
        try:
            roots = self._roots(request)
            positions = self._manager.boundary_checkpoint_page_positions(
                checkpoint.num_tokens
            )
            content_keys, auxiliary_key = self._page_content_keys(
                roots,
                checkpoint.num_tokens,
                positions,
                checkpoint.draft_prefix_len,
                checkpoint.kind,
            )
            page_bytes = self._layout["page_bytes"]
            groups = [
                {
                    "name": f"engine-kv-group:{i}",
                    "page_bytes": page_bytes,
                    "positions": list(pages),
                    "content_keys": list(content_keys[i]),
                }
                for i, pages in enumerate(positions)
            ]
            groups.append(
                {
                    "name": "target-draft-auxiliary",
                    "page_bytes": page_bytes,
                    "positions": [0],
                    "content_keys": [auxiliary_key],
                }
            )
            prefix = roots.prefix(checkpoint.num_tokens)
            payload = json.dumps(
                {
                    "schema_version": 2,
                    "worker_layout": self._layout,
                    "page_groups": groups,
                    "draft_prefix_len": checkpoint.draft_prefix_len,
                    "kind": checkpoint.kind,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            manifest = CheckpointManifest(
                checkpoint_generation(prefix, payload),
                prefix,
                self._world_size,
                payload,
            )
            task = self._make_task(manifest, checkpoint, "STORE")
            self._tasks[task.task_id] = _PendingTask(
                task,
                pinned,
                request.request_id,
                begin=self._client.submit_request(
                    RequestType.CHECKPOINT_BEGIN, [manifest]
                ),
                roots=roots,
            )
        except BaseException:
            self._cache.release(pinned)
            raise

    def _page_content_keys(
        self,
        roots: CheckpointTokenRoots,
        num_tokens: int,
        positions: tuple[tuple[int, ...], ...],
        draft_prefix_len: int,
        kind: str,
    ) -> tuple[tuple[tuple[str, ...], ...], str]:
        """Authenticate reusable attention pages and unique recurrent endpoints."""
        # Import against the source-locked vLLM boundary allocator only when
        # semantic checkpoint storage is active.
        # Third Party
        from vllm.v1.kv_cache_interface import MambaSpec, iter_layer_specs

        managers = self._manager.coordinator.single_type_managers
        specs = self._manager.kv_cache_config.kv_cache_groups
        if len(managers) != len(positions) or len(specs) != len(positions):
            raise ValueError("Checkpoint cache-group geometry changed during store")
        boundaries: list[tuple[int, str]] = []
        widths = []
        for group_id, (manager, spec, pages) in enumerate(
            zip(managers, specs, positions, strict=True)
        ):
            recurrent = all(
                isinstance(layer_spec, MambaSpec)
                for layer_spec in iter_layer_specs(spec.kv_cache_spec)
            )
            widths.append(len(pages))
            for position in pages:
                page_end = (
                    num_tokens
                    if recurrent
                    else min((position + 1) * manager.block_size, num_tokens)
                )
                boundaries.append(
                    (
                        page_end,
                        f"{'recurrent' if recurrent else 'attention'}:{group_id}",
                    )
                )
        boundaries.append((num_tokens, f"auxiliary:{draft_prefix_len}:{kind}"))
        flat_keys = iter(roots.content_keys(boundaries))
        keys = tuple(tuple(next(flat_keys) for _ in range(width)) for width in widths)
        return keys, next(flat_keys)

    def take_tasks(self) -> list[CheckpointEngineTask]:
        """Return each admitted collective copy exactly once, without waiting on RPC."""
        tasks = []
        for task_id, pending in tuple(self._tasks.items()):
            if pending.sent:
                continue
            if pending.begin is not None:
                answered = pending.begin.query()
                if (
                    not answered
                    and time.monotonic() - pending.created < self._lookup_timeout
                ):
                    continue
                accepted = False
                abort = not answered
                if answered:
                    try:
                        accepted = pending.begin.result()
                    except Exception:
                        logger.exception("Recurrent checkpoint begin operation failed")
                        abort = True
                else:
                    logger.warning(
                        "Recurrent checkpoint store of %d tokens got no begin reply "
                        "within %.0f s; aborting it",
                        pending.task.manifest.prefix.num_tokens,
                        self._lookup_timeout,
                    )
                if abort:
                    # No worker has received this task. Its GPU source pin can
                    # be released even if the server's begin reply is lost.
                    try:
                        self._client.submit_request(
                            RequestType.CHECKPOINT_ABORT,
                            [pending.task.manifest.generation],
                        )
                    except Exception:
                        logger.exception("Recurrent checkpoint abort submission failed")
                if not accepted:
                    self._cache.release(pending.checkpoint)
                    del self._tasks[task_id]
                    if pending.request_id in self._cancelled:
                        self.finish_request(pending.request_id)
                    continue
            pending.sent = True
            tasks.append(pending.task)
        return tasks

    def complete(self, results: dict[str, dict[int, bool]]) -> None:
        """Aggregate drained worker acknowledgements and publish or discard atomically.

        Args:
            results: Task ID to distinct rank results. True proves completed bytes;
                False proves that rank will no longer access the transfer's pages.

        Raises:
            ValueError: For duplicate or unknown rank acknowledgements.
        """
        for task_id, ranks in results.items():
            pending = self._tasks.get(task_id)
            if pending is None:
                raise ValueError("Checkpoint completion identifies an unknown task")
            for rank, success in ranks.items():
                if rank in pending.acknowledgements or not 0 <= rank < self._world_size:
                    raise ValueError("Duplicate or invalid checkpoint rank completion")
                pending.acknowledgements[rank] = success
            if len(pending.acknowledgements) != self._world_size:
                continue
            if pending.task.direction == "STORE":
                if not all(pending.acknowledgements.values()):
                    self._client.submit_request(
                        RequestType.CHECKPOINT_ABORT, [pending.task.manifest.generation]
                    )
                elif pending.roots is not None:
                    # Published: a new prompt supersedes older checkpoints of
                    # its sequence, and the server links this request's
                    # checkpoints. The reply is not needed.
                    try:
                        self._client.submit_request(
                            RequestType.CHECKPOINT_SUPERSEDE,
                            [
                                pending.roots.roots,
                                pending.task.manifest.generation,
                                pending.request_id,
                            ],
                        )
                    except Exception:
                        logger.warning(
                            "Could not report superseded checkpoints", exc_info=True
                        )
                self._cache.release(pending.checkpoint)
            else:
                state = self._lookups.get(pending.request_id)
                retry = False
                if (
                    all(pending.acknowledgements.values())
                    and pending.request_id not in self._cancelled
                ):
                    for rank in range(self._world_size):
                        published = (
                            self._manager.acknowledge_external_boundary_checkpoint(
                                pending.checkpoint.checkpoint_id, rank
                            )
                        )
                    if published and state is not None:
                        state.checkpoint_id = pending.checkpoint.checkpoint_id
                        logger.debug(
                            "Recurrent checkpoint restore of %d tokens ready for "
                            "request %s; ownership retained until admission",
                            pending.checkpoint.num_tokens,
                            pending.request_id,
                        )
                    elif state is not None:
                        retry = state.attempts < _MAX_LOOKUP_ATTEMPTS
                        logger.info(
                            "Recurrent checkpoint publication invalidated for "
                            "request %s%s",
                            pending.request_id,
                            "; retrying lookup"
                            if retry
                            else "; recomputing its prompt",
                        )
                    elapsed = time.monotonic() - pending.created
                    if elapsed > _SLOW_RESTORE_SECONDS:
                        logger.info(
                            "Recurrent checkpoint restore of %d tokens for request "
                            "%s took %.1f s",
                            pending.task.manifest.prefix.num_tokens,
                            pending.request_id,
                            elapsed,
                        )
                else:
                    self._manager.discard_external_boundary_checkpoint(
                        pending.checkpoint.checkpoint_id
                    )
                    retry = (
                        state is not None
                        and pending.request_id not in self._cancelled
                        and state.attempts < _MAX_LOOKUP_ATTEMPTS
                    )
                    logger.info(
                        "Recurrent checkpoint restore of %d tokens failed for "
                        "request %s on ranks %s after %.1f s%s",
                        pending.task.manifest.prefix.num_tokens,
                        pending.request_id,
                        sorted(
                            rank
                            for rank, success in pending.acknowledgements.items()
                            if not success
                        ),
                        time.monotonic() - pending.created,
                        "" if retry else "; recomputing its prompt",
                    )
                if state is not None and retry:
                    # A shorter checkpoint can survive eviction or a restart
                    # when the longest one did not; look it up again.
                    logger.info(
                        "Recurrent checkpoint restore of %d tokens missed for "
                        "request %s; looking up a shorter checkpoint (attempt %d)",
                        pending.task.manifest.prefix.num_tokens,
                        pending.request_id,
                        state.attempts + 1,
                    )
                    state.task_id = None
                    state.attempts += 1
                    state.started = time.monotonic()
                    state.future = self._client.submit_request(
                        RequestType.CHECKPOINT_FIND, [state.roots.roots]
                    )
                elif state is not None:
                    state.done = True
            del self._tasks[task_id]
            if pending.request_id in self._cancelled:
                self.finish_request(pending.request_id)

    def finish_request(self, request_id: str) -> None:
        """Forget lookup state, retaining copy pins until every admitted task drains."""
        self._manager.release_external_boundary_admission(request_id)
        self._lookups.pop(request_id, None)
        if any(task.request_id == request_id for task in self._tasks.values()):
            self._cancelled.add(request_id)
        else:
            self._cancelled.discard(request_id)

    def _roots(self, request: "Request") -> CheckpointTokenRoots:
        tokens = request.all_token_ids
        if request.mm_features:
            # Placeholder IDs are equal for every image; name spans by content.
            # Third Party
            from vllm.v1.core.boundary_checkpoint import content_token_ids

            tokens = content_token_ids(request, 0, len(tokens), self._multimodal_salt)
        return CheckpointTokenRoots.build(
            checkpoint_namespace(self._identity, request.cache_salt or ""),
            tokens,
        )

    def _validate_manifest(
        self, manifest: CheckpointManifest, roots: CheckpointTokenRoots
    ) -> tuple[dict[str, Any], tuple[tuple[int, ...], ...]]:
        if manifest.world_size != self._world_size or manifest.prefix != roots.prefix(
            manifest.prefix.num_tokens
        ):
            raise ValueError("Checkpoint rank count or token prefix does not match")
        payload = json.loads(manifest.payload)
        groups = checkpoint_page_groups(manifest)
        if (
            payload.get("worker_layout") != self._layout
            or groups[-1].name != "target-draft-auxiliary"
            or groups[-1].positions != (0,)
        ):
            raise ValueError("Checkpoint layout or auxiliary payload is incompatible")
        if any(group.page_bytes != self._layout["page_bytes"] for group in groups):
            raise ValueError(
                "Checkpoint physical page width differs from the worker pool"
            )
        if [group.name for group in groups[:-1]] != [
            f"engine-kv-group:{i}" for i in range(len(groups) - 1)
        ]:
            raise ValueError("Checkpoint engine group ordering is incompatible")
        return payload, tuple(group.positions for group in groups[:-1])

    def _make_task(
        self,
        manifest: CheckpointManifest,
        checkpoint: "BoundaryCheckpoint",
        direction: Literal["STORE", "RETRIEVE"],
    ) -> CheckpointEngineTask:
        return CheckpointEngineTask(
            uuid.uuid4().hex,
            manifest,
            direction,
            tuple(
                tuple(block for block in group if block)
                for group in checkpoint.block_ids
            )
            + (checkpoint.auxiliary_block_ids,),
        )
