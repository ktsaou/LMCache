# SPDX-License-Identifier: Apache-2.0
"""Scheduler ownership and all-rank publication of external recurrent checkpoints."""

# Standard
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal
import inspect
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

# A restore waiting for GPU or copy capacity is logged when the wait starts
# and again at this interval, so a starved restore stays visible.
_WAIT_LOG_SECONDS = 30.0

# Allocator methods of vLLM's restore admission reservations. The scheduler
# gates admissions with the first; this bridge calls the other two.
_ADMISSION_METHODS = (
    "can_admit_external_boundary_request",
    "external_boundary_admission_ready",
    "release_external_boundary_admission",
)

# Settings that change encoder outputs, and so the KV of multimodal spans,
# without changing the processor's content hash. They salt multimodal roots
# only, so text namespaces are unaffected.
_MULTIMODAL_ENVIRONMENT = ("VLLM_GLM53_VISION_MXFP8",)

if TYPE_CHECKING:
    # Third Party
    from vllm.v1.core.boundary_checkpoint import BoundaryCheckpoint
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.request import Request


def _supports_restore_admission(manager: "KVCacheManager") -> bool:
    """Whether vLLM can hold GPU capacity for a restore until its admission.

    Args:
        manager: vLLM allocator bound to the bridge.

    Returns:
        True if the allocator has the admission methods and reservations
        accept ``reserve_admission``. Allocators older than the paired vLLM
        restore-admission change have neither.
    """
    if not all(callable(getattr(manager, name, None)) for name in _ADMISSION_METHODS):
        return False
    try:
        signature = inspect.signature(manager.reserve_external_boundary_checkpoint)
    except (TypeError, ValueError):
        return False
    return "reserve_admission" in signature.parameters


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
    # Directory lookup of the current attempt, sent at `started`.
    future: MessagingFuture[CheckpointManifest | None]
    done: bool = False
    # Restore copy in flight for this request, or None.
    task_id: str | None = None
    # Published import, counted as an external cache hit.
    checkpoint_id: int | None = None
    # Prefix length selected for reuse, local or imported; 0 before selection.
    selected_tokens: int = 0
    attempts: int = 1
    started: float = field(default_factory=time.monotonic)
    # Answered candidate of the current attempt, retained until its copy
    # starts, and its validated payload and page positions.
    manifest: CheckpointManifest | None = None
    plan: tuple[dict[str, Any], tuple[tuple[int, ...], ...]] | None = None
    # When the directory last answered or was asked again for `manifest`, and
    # the unanswered re-sent lookup, if any.
    refreshed: float = 0.0
    refresh: MessagingFuture[CheckpointManifest | None] | None = None
    # Start of the current wait for GPU or copy capacity, and its last log.
    waiting_since: float | None = None
    wait_logged: float = 0.0
    capacity_refusal_logged: bool = False


class CheckpointSchedulerBridge:
    """Keep imports private and store sources pinned until every rank completes.

    Args:
        manager: vLLM allocator with request-boundary checkpoint support enabled.
            Restores wait for GPU capacity only if it supports restore
            admission reservations; otherwise a restore that cannot reserve
            its pages permits admission and the prompt is recomputed.
        client: Shared thread-safe LMCache queue client; no ownership transfer.
        identity: Immutable target/draft/source revisions and parallel geometry.
            Worker layout is added after all ranks report identical descriptors.
        world_size: Number of engine ranks contributing to one atomic generation.
        max_tasks: Admission limit for collective stores and restores.
        lookup_timeout: Seconds allowed for each directory reply, separately
            for every retry after a failed restore, and for a store's begin
            reply. Capacity waits and copies do not consume it. An unanswered
            lookup permits prompt recomputation; an unanswered store is
            aborted. A restore waiting longer than half of it sends its lookup
            again, without waiting for the reply, to keep its pages recent.
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
        # Fixed for the allocator's lifetime; never call a missing method.
        self._reserve_admission = _supports_restore_admission(manager)
        if not self._reserve_admission:
            logger.warning(
                "vLLM cannot reserve admission capacity for recurrent checkpoint "
                "restores; a restore without enough free GPU blocks or copy slots "
                "is admitted to recompute its prompt. Deploy the paired vLLM "
                "restore-admission change to make restores wait for capacity."
            )
        # Restores that had to wait for GPU or copy capacity, since creation.
        self._waits_started = 0
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
            False while layout negotiation, lookup or H2D is pending, and, if
            vLLM supports restore admission reservations, while an answered
            checkpoint waits for GPU or copy capacity. True when ordinary GPU
            admission may proceed. Without those reservations, a restore that
            cannot reserve GPU blocks permits admission; an unadmitted request
            may poll again to retry it without another blocking lookup.
            A complete local hit proceeds without a directory reply, except
            behind this request's own reserved copy.
            Reusing a finished request's ID waits for its admitted copies to
            drain, so their completion cannot cancel or erase another lookup.
        """
        request_id = request.request_id
        if request_id in self._cancelled:
            return False
        if not self.handles(request) or not self._manager.prefix_cache_lookup_enabled(
            request
        ):
            return True
        local = self._cache.find(request, request.num_tokens)
        state = self._lookups.get(request_id)
        if (
            local is not None
            and local.num_tokens == request.num_tokens
            and (state is None or state.task_id is None or not self._reserve_admission)
        ):
            # A complete local hit needs no directory reply or capacity wait. A
            # reserved copy must still finish: its publication hands the
            # reservation to this request, which must still be waiting then.
            if self._reserve_admission and state is not None and not state.done:
                # Settle the lookup as a local selection. If this checkpoint
                # leaves the cache before admission, the prefix is looked up
                # again rather than resuming an old reply or its deadline.
                state.selected_tokens = local.num_tokens
                self._stop_waiting(request_id, state)
                state.done = True
            return True
        if self._layout is None:
            return False
        if state is None:
            roots = self._roots(request)
            self._lookups[request_id] = _Lookup(
                roots,
                self._client.submit_request(RequestType.CHECKPOINT_FIND, [roots.roots]),
            )
            return False
        if state.done:
            # A selected prefix can leave the GPU cache while ordinary admission
            # waits or after preemption; only a shorter or missing local
            # checkpoint needs another lookup. Without reservations a restored
            # bundle is not held for its consumer, so another lookup could
            # repeat the restore under the same pressure.
            if (
                self._reserve_admission
                and state.selected_tokens > 0
                and (local is None or local.num_tokens < state.selected_tokens)
                and not self._manager.external_boundary_admission_ready(request_id)
            ):
                logger.info(
                    "Recurrent checkpoint of %d tokens selected for request %s is "
                    "no longer cached; looking up its prefix again",
                    state.selected_tokens,
                    request_id,
                )
                self._manager.release_external_boundary_admission(request_id)
                del self._lookups[request_id]
                return self.poll_prefix(request)
            return True
        if state.task_id is not None:
            return False
        manifest = state.manifest
        if manifest is None:
            if not state.future.query():
                if time.monotonic() - state.started < self._lookup_timeout:
                    return False
                # An unanswered lookup must not park the request indefinitely.
                logger.warning(
                    "Recurrent checkpoint lookup for request %s got no reply within "
                    "%.0f s; recomputing its prompt",
                    request_id,
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
                    request_id,
                )
                state.done = True
                return True
            if manifest is None:
                state.done = True
                return True
            state.manifest = manifest
            state.refreshed = time.monotonic()
        else:
            manifest = self._refresh_manifest(request_id, state, manifest)
            if manifest is None:
                self._stop_waiting(request_id, state)
                state.done = True
                return True
        if local is not None and manifest.prefix.num_tokens <= local.num_tokens:
            # Reuse the local checkpoint; external_tokens counts imports only.
            state.selected_tokens = local.num_tokens
            self._stop_waiting(request_id, state)
            state.done = True
            return True
        if len(self._tasks) >= self._max_tasks:
            if not self._reserve_admission:
                logger.info(
                    "Recurrent checkpoint restore of %d tokens skipped for request "
                    "%s: %d checkpoint copies in flight; recomputing its prompt",
                    manifest.prefix.num_tokens,
                    request_id,
                    len(self._tasks),
                )
                state.done = True
                return True
            self._wait(
                request_id,
                state,
                manifest.prefix.num_tokens,
                "a copy slot (%d copies in flight)",
                len(self._tasks),
            )
            return False
        try:
            if state.plan is None:
                # Validated once per answer, not on every step of a wait.
                state.plan = self._validate_manifest(manifest, state.roots)
            payload, positions = state.plan
            checkpoint = self._reserve(
                request, manifest.prefix.num_tokens, payload, positions
            )
        except (ValueError, KeyError, TypeError):
            logger.warning(
                "Recurrent checkpoint restore of %d tokens rejected for request "
                "%s; recomputing its prompt",
                manifest.prefix.num_tokens,
                request_id,
                exc_info=True,
            )
            self._stop_waiting(request_id, state)
            state.done = True
            return True
        if checkpoint is None:
            if not self._reserve_admission:
                # Insufficient GPU capacity is not an external-cache miss. Permit
                # ordinary admission, but retain the manifest so an unadmitted
                # request can retry its import after other owners release pages.
                if not state.capacity_refusal_logged:
                    state.capacity_refusal_logged = True
                    logger.info(
                        "Recurrent checkpoint restore of %d tokens deferred for "
                        "request %s: not enough free GPU blocks (%d free); it is "
                        "admitted without the restore unless capacity returns first",
                        manifest.prefix.num_tokens,
                        request_id,
                        self._manager.block_pool.get_num_free_blocks(),
                    )
                return True
            # Resource pressure is not a cache miss. vLLM queues this request
            # for the next restore reservation; keep the answered manifest.
            self._wait(
                request_id,
                state,
                manifest.prefix.num_tokens,
                "GPU capacity (%d free blocks; the checkpoint needs %d plus "
                "execution headroom)",
                self._manager.block_pool.get_num_free_blocks(),
                sum(len(group) for group in positions) + 1,
            )
            return False
        if state.waiting_since is not None:
            waited = time.monotonic() - state.waiting_since
            state.waiting_since = None
            if waited >= _WAIT_LOG_SECONDS:
                logger.info(
                    "Recurrent checkpoint restore of %d tokens for request %s "
                    "starts after waiting %.0f s for capacity",
                    manifest.prefix.num_tokens,
                    request_id,
                    waited,
                )
        task = self._make_task(manifest, checkpoint, "RETRIEVE")
        logger.debug(
            "Recurrent checkpoint restore of %d tokens copying for request %s",
            manifest.prefix.num_tokens,
            request_id,
        )
        self._tasks[task.task_id] = _PendingTask(task, checkpoint, request_id)
        state.task_id = task.task_id
        # The copy carries the manifest; a retry looks the prefix up again.
        state.manifest = None
        state.plan = None
        state.refresh = None
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
                if state is not None:
                    state.task_id = None
                retry = False
                missed = False
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
                        state.selected_tokens = pending.checkpoint.num_tokens
                        logger.debug(
                            "Recurrent checkpoint restore of %d tokens ready for "
                            "request %s; ownership retained until admission",
                            pending.checkpoint.num_tokens,
                            pending.request_id,
                        )
                    elif state is not None:
                        # The GPU cache evicted a destination page before
                        # publication. The stored checkpoint is intact, so the
                        # retry may receive the same one.
                        retry = state.attempts < _MAX_LOOKUP_ATTEMPTS
                        logger.info(
                            "Recurrent checkpoint restore of %d tokens for request "
                            "%s was invalidated before publication; %s",
                            pending.task.manifest.prefix.num_tokens,
                            pending.request_id,
                            "looking up its prefix again"
                            if retry
                            else "recomputing its prompt",
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
                    missed = True
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
                    if missed:
                        # A shorter checkpoint can survive eviction or a restart
                        # when the longest one did not; look it up again.
                        logger.info(
                            "Recurrent checkpoint restore of %d tokens missed for "
                            "request %s; looking up a shorter checkpoint "
                            "(attempt %d)",
                            pending.task.manifest.prefix.num_tokens,
                            pending.request_id,
                            state.attempts + 1,
                        )
                    # Each attempt has its own reply deadline; capacity waits
                    # and copies of earlier attempts do not consume it.
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
        """Forget lookup state, retaining copy pins until every admitted task drains.

        With vLLM restore admission reservations, also release the request's
        reservation, or its place among restores waiting for capacity.
        """
        if self._reserve_admission:
            self._manager.release_external_boundary_admission(request_id)
        self._lookups.pop(request_id, None)
        if any(task.request_id == request_id for task in self._tasks.values()):
            self._cancelled.add(request_id)
        else:
            self._cancelled.discard(request_id)

    def report_status(self) -> dict[str, float]:
        """Report restores waiting for GPU or copy capacity, without advancing them.

        Returns:
            ``capacity_waits``: requests whose answered restore is waiting now.
            ``longest_capacity_wait_seconds``: age of the oldest such wait, or
            0.0 without one. ``capacity_waits_started``: restores that had to
            wait since this bridge was created. Requests that vLLM holds back
            before they reach this bridge are not counted.
        """
        now = time.monotonic()
        waits = [
            now - state.waiting_since
            for state in self._lookups.values()
            if state.waiting_since is not None
        ]
        return {
            "capacity_waits": len(waits),
            "longest_capacity_wait_seconds": max(waits, default=0.0),
            "capacity_waits_started": self._waits_started,
        }

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

    def _reserve(
        self,
        request: "Request",
        num_tokens: int,
        payload: dict[str, Any],
        positions: tuple[tuple[int, ...], ...],
    ) -> "BoundaryCheckpoint | None":
        """Reserve restore destinations, returning None under resource pressure."""
        if self._reserve_admission:
            # Also reserve a running slot and continuation blocks; vLLM then
            # keeps the published restore owned until this request is admitted.
            return self._manager.reserve_external_boundary_checkpoint(
                request,
                num_tokens,
                positions,
                draft_prefix_len=payload["draft_prefix_len"],
                kind=payload["kind"],
                num_ranks=self._world_size,
                reserve_admission=True,
            )
        return self._manager.reserve_external_boundary_checkpoint(
            request,
            num_tokens,
            positions,
            draft_prefix_len=payload["draft_prefix_len"],
            kind=payload["kind"],
            num_ranks=self._world_size,
        )

    def _refresh_manifest(
        self, request_id: str, state: _Lookup, manifest: CheckpointManifest
    ) -> CheckpointManifest | None:
        """Return the retained answer, asking the directory again while it waits.

        A lookup refreshes the eviction recency of its candidate's pages. A
        restore that has retained its answer for half the reply deadline sends
        the lookup again, one at a time. It keeps using the retained answer
        until the reply arrives, which then replaces it.

        Args:
            request_id: Request that owns the lookup.
            state: Lookup of the current attempt, retaining ``manifest``.
            manifest: Answered candidate of the current attempt.

        Returns:
            The current candidate, or None once the directory lists no
            checkpoint of the prefix.
        """
        now = time.monotonic()
        if state.refresh is None:
            if now - state.refreshed >= self._lookup_timeout / 2:
                state.refresh = self._client.submit_request(
                    RequestType.CHECKPOINT_FIND, [state.roots.roots]
                )
                state.refreshed = now
            return manifest
        if not state.refresh.query():
            return manifest
        refresh, state.refresh = state.refresh, None
        try:
            answer = refresh.result()
        except Exception:
            logger.warning(
                "Recurrent checkpoint lookup refresh failed for request %s; "
                "keeping its answered checkpoint",
                request_id,
                exc_info=True,
            )
            return manifest
        if answer is None:
            logger.info(
                "Recurrent checkpoint of %d tokens for request %s is no longer "
                "listed; recomputing its prompt",
                manifest.prefix.num_tokens,
                request_id,
            )
            return None
        if answer.generation != manifest.generation:
            state.manifest = answer
            state.plan = None
        return answer

    def _wait(
        self, request_id: str, state: _Lookup, num_tokens: int, reason: str, *args: int
    ) -> None:
        """Keep an answered restore waiting; log its start, then periodically.

        Args:
            request_id: Request whose restore cannot start yet.
            state: Its lookup, retaining the answered checkpoint.
            num_tokens: Length of that checkpoint.
            reason: Log format naming the missing capacity, filled from args.
            args: Current values for ``reason``.
        """
        now = time.monotonic()
        if state.waiting_since is None:
            state.waiting_since = now
            state.wait_logged = now
            self._waits_started += 1
            logger.info(
                "Recurrent checkpoint restore of %d tokens for request %s is "
                "waiting for " + reason,
                num_tokens,
                request_id,
                *args,
            )
        elif now - state.wait_logged >= _WAIT_LOG_SECONDS:
            state.wait_logged = now
            logger.info(
                "Recurrent checkpoint restore of %d tokens for request %s has "
                "waited %.0f s for " + reason,
                num_tokens,
                request_id,
                now - state.waiting_since,
                *args,
            )

    def _stop_waiting(self, request_id: str, state: _Lookup) -> None:
        """End a capacity wait of a request that proceeds without this restore."""
        if state.waiting_since is None:
            return
        state.waiting_since = None
        if self._reserve_admission:
            # vLLM would otherwise still count the request as waiting for a
            # restore reservation and hold other admissions back for it.
            self._manager.release_external_boundary_admission(request_id)
