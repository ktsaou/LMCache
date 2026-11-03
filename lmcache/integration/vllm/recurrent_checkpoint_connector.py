# SPDX-License-Identifier: Apache-2.0
"""External storage of complete target/draft request-boundary checkpoints.

Status: research-only until GLM serving, restart and performance qualification.
Ordinary aligned LMCache transfers remain in LMCacheMPConnector. This connector
maps the negotiated SHM byte pool directly: complete checkpoint pages do not
depend on the server's token-chunk size or aligned KV registration.
"""

# Standard
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any
import os
import re

# Third Party
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    KVConnectorWorkerMetadata,
    SupportsHMA,
)
from vllm.v1.core.boundary_checkpoint import BoundaryCheckpoint
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request
from vllm.v1.worker.gpu.boundary_checkpoint import BoundaryCheckpointState
import torch
import zmq

# First Party
from lmcache.integration.vllm.checkpoint_copy import CheckpointPageCopier
from lmcache.integration.vllm.checkpoint_scheduler import (
    CheckpointEngineTask,
    CheckpointSchedulerBridge,
)
from lmcache.integration.vllm.lmcache_mp_connector import (
    build_parallel_strategy_from_vllm_config,
)
from lmcache.logging import init_logger
from lmcache.v1.multiprocess.checkpoint_transfer import (
    CheckpointTransferJob,
    CheckpointTransferWorker,
    UnsafeCheckpointCopyError,
)
from lmcache.v1.multiprocess.futures import MessagingFuture
from lmcache.v1.multiprocess.mq import MessageQueueClient
from lmcache.v1.multiprocess.protocols.base import RequestType
from lmcache.v1.multiprocess.protocols.checkpoint import CheckpointCapabilities
from lmcache.v1.multiprocess.transfer_context.shm import ShmPoolMapping
from lmcache.v1.platform import torch_dev

logger = init_logger(__name__)

_CHECKPOINT_MAX_TASKS = 32
_CHECKPOINT_COPY_WORKERS = 2

# These precision overrides are outside VllmConfig's computation-graph hash.
# Changing one must not reuse states produced by another numeric representation.
_PRECISION_ENVIRONMENT = (
    "VLLM_B12X_MOE_FP4_FORCE_A16",
    "VLLM_B12X_DENSE_ACTIVATION_MODE",
    "VLLM_B12X_NVFP4_ACTIVATION_MODE",
    "VLLM_B12X_MXFP8_ACTIVATION_MODE",
    "VLLM_GLM53_ONLINE_DENSE_MXFP8",
    "VLLM_MXFP8_LM_HEAD",
    "VLLM_MTP_NVFP4_LM_HEAD",
    "VLLM_LM_HEAD_A16",
    "VLLM_GDN_DECODE_KERNEL",
)


@dataclass
class RecurrentCheckpointMetadata(KVConnectorMetadata):
    """Collective copy commands; physical IDs are valid only in this engine."""

    tasks: list[CheckpointEngineTask] = field(default_factory=list)


@dataclass
class RecurrentCheckpointWorkerMetadata(KVConnectorWorkerMetadata):
    """Distinct rank completion and address-free initialization descriptors."""

    layouts: dict[int, dict[str, Any]] = field(default_factory=dict)
    results: dict[str, dict[int, bool]] = field(default_factory=dict)

    def aggregate(self, other: KVConnectorWorkerMetadata) -> KVConnectorWorkerMetadata:
        """Merge distinct rank acknowledgements without treating counts as identity."""
        if not isinstance(other, RecurrentCheckpointWorkerMetadata):
            raise ValueError("Incompatible recurrent checkpoint worker metadata")
        if self.layouts.keys() & other.layouts.keys():
            raise ValueError("Duplicate recurrent checkpoint layout rank")
        results = {task: dict(ranks) for task, ranks in self.results.items()}
        for task, ranks in other.results.items():
            merged = results.setdefault(task, {})
            if merged.keys() & ranks.keys():
                raise ValueError("Duplicate recurrent checkpoint completion rank")
            merged.update(ranks)
        return RecurrentCheckpointWorkerMetadata(self.layouts | other.layouts, results)


class LMCacheRecurrentCheckpointConnector(KVConnectorBase_V1, SupportsHMA):
    """CPU-sidecar checkpoint storage with worker-owned, asynchronous SHM DMA.

    Requires one LMCache server, engine-driven SHM, immutable content revisions
    in ``lmcache.mp.checkpoint_identity``, and a supported vLLM boundary adapter.
    Requests unsupported by that adapter recompute rather than restoring an
    incomplete collection of target/draft tensors.
    """

    @classmethod
    def supports_request_boundary_checkpoints(cls, config: VllmConfig) -> bool:
        """Validate immutable namespace inputs before vLLM chooses retention policy."""
        transfer = config.kv_transfer_config
        if transfer is None:
            return False
        identity = transfer.get_from_extra_config(
            "lmcache.mp.checkpoint_identity", None
        )
        if not isinstance(identity, dict):
            return False
        keys = ["target_revision", "source_revision"]
        if config.speculative_config is not None:
            keys.append("draft_revision")
        return all(
            isinstance(identity.get(key), str)
            and re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", identity[key]) is not None
            for key in keys
        )

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig,
    ) -> None:
        super().__init__(vllm_config, role, kv_cache_config)
        if not self.supports_request_boundary_checkpoints(vllm_config):
            raise ValueError(
                "Semantic LMCache requires immutable target/draft/source revisions"
            )
        if not vllm_config.use_request_boundary_checkpoints:
            raise ValueError(
                "Semantic LMCache requires a supported request-boundary adapter"
            )
        self._scheduler: CheckpointSchedulerBridge | None = None
        self._worker: CheckpointTransferWorker | None = None
        self._mapping: ShmPoolMapping | None = None
        self._worker_layout: dict[str, Any] | None = None
        self._layout_sent = False
        self._pending: dict[str, Future[bool]] = {}
        self._rejected: set[str] = set()
        self._role = role
        self._rank: int | None = None
        config = vllm_config.kv_transfer_config
        mode = config.get_from_extra_config(
            "lmcache.mp.mp_transfer_mode", os.getenv("LMCACHE_MP_TRANSFER_MODE", "auto")
        )
        if mode != "engine_driven":
            raise ValueError("Semantic checkpoints require engine-driven SHM transport")
        urls = config.get_from_extra_config("lmcache.mp.server_urls", None)
        if urls:
            urls = urls if isinstance(urls, list) else str(urls).split(",")
            urls = [str(url).strip() for url in urls if str(url).strip()]
        else:
            host = config.get_from_extra_config("lmcache.mp.host", "tcp://localhost")
            port = config.get_from_extra_config("lmcache.mp.port", 5555)
            urls = [f"{host}:{port}"]
        if len(urls) != 1:
            raise ValueError("Semantic checkpoints require one shared LMCache server")
        url = urls[0] if "://" in urls[0] else f"tcp://{urls[0]}"
        timeout = float(config.get_from_extra_config("lmcache.mp.mq_timeout", 60.0))
        if not 0 < timeout < float("inf"):
            raise ValueError(
                "LMCache message-queue timeout must be finite and positive"
            )
        self._mq_timeout = timeout
        self._client = MessageQueueClient(url, zmq.Context.instance())
        try:
            capability_reply: MessagingFuture[CheckpointCapabilities] = (
                self._client.submit_request(RequestType.CHECKPOINT_CAPABILITIES, [])
            )
            self._capabilities = capability_reply.result(timeout=timeout)
            if (
                self._capabilities.format_version != 1
                or not self._capabilities.shm_name
                or self._capabilities.pool_size <= 0
            ):
                raise ValueError(
                    "LMCache server does not support atomic checkpoint SHM storage"
                )
        except BaseException:
            self._client.close()
            raise

    def bind_boundary_checkpoint_cache(self, manager: KVCacheManager) -> None:
        """Bind the scheduler's allocator and immutable content namespace."""
        config = self._vllm_config
        identity = dict(
            config.kv_transfer_config.get_from_extra_config(
                "lmcache.mp.checkpoint_identity", {}
            )
        )
        identity.setdefault("draft_revision", "")
        identity["runtime_config_hash"] = config.compute_hash()
        identity["model_config_hash"] = config.model_config.compute_hash()
        identity["precision_environment"] = {
            name: os.environ.get(name) for name in _PRECISION_ENVIRONMENT
        }
        parallel = config.parallel_config
        identity["parallel"] = {
            "tp": parallel.tensor_parallel_size,
            "dcp": parallel.decode_context_parallel_size,
            "cp_interleave": parallel.cp_kv_cache_interleave_size,
        }
        self._scheduler = CheckpointSchedulerBridge(
            manager,
            self._client,
            identity,
            parallel.world_size,
            max_tasks=_CHECKPOINT_MAX_TASKS,
            lookup_timeout=self._mq_timeout,
        )

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        """The boundary-state hook binds raw pages, not per-layer aligned chunks."""
        return

    def bind_boundary_checkpoint_state(self, state: BoundaryCheckpointState) -> None:
        """Validate the worker byte layout and bind raw-page DMA before serving."""
        if self._role != KVConnectorRole.WORKER or self._mapping is not None:
            raise RuntimeError("Checkpoint byte mapping must bind once on a worker")
        self._worker_layout = state.get_external_checkpoint_layout()
        strategy = build_parallel_strategy_from_vllm_config(self._vllm_config, 1)
        mapping = ShmPoolMapping(
            self._capabilities.shm_name, self._capabilities.pool_size
        )
        try:
            copier = CheckpointPageCopier(
                state.get_external_checkpoint_page_pool(),
                self._worker_layout,
                state.initialize_external_checkpoint_layout,
                mapping,
                self._capabilities,
            )
            # A completed Future can be reported before its release callback runs.
            # Each copy thread can retain one such untracked admission permit.
            self._worker = CheckpointTransferWorker(
                self._client,
                copier,
                workers=_CHECKPOINT_COPY_WORKERS,
                max_pending=_CHECKPOINT_MAX_TASKS + _CHECKPOINT_COPY_WORKERS,
            )
        except BaseException:
            mapping.close()
            raise
        self._mapping = mapping
        self._rank = strategy.vllm_worker_id

    def poll_boundary_checkpoint(self, request: Request) -> bool:
        """Defer request admission until a collective import finishes or misses."""
        assert self._scheduler is not None
        return self._scheduler.poll_prefix(request)

    def boundary_checkpoint_external_tokens(self, request: Request) -> int:
        """Report imported bytes as external cache hits, not GPU prefix-cache hits."""
        assert self._scheduler is not None
        return self._scheduler.external_tokens(request)

    def store_boundary_checkpoint(
        self, request: Request, checkpoint: BoundaryCheckpoint
    ) -> None:
        """Retain immutable source pages independently of the producer request."""
        assert self._scheduler is not None
        self._scheduler.store(request, checkpoint)

    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int, bool]:
        """Use the atomic import hook, never ordinary independent chunk retrieval."""
        return 0, False

    def update_state_after_alloc(
        self, request: Request, blocks: Any, num_external_tokens: int
    ) -> None:
        """Imports own a private bundle; normal request allocation needs no transfer."""
        return

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        """Emit each admitted task once, including on connector-only scheduler steps."""
        assert self._scheduler is not None
        return RecurrentCheckpointMetadata(self._scheduler.take_tasks())

    def start_load_kv(self, forward_context: Any, **kwargs: Any) -> None:
        """Submit both transfer directions for already immutable checkpoint pages."""
        metadata = self._connector_metadata
        if not isinstance(metadata, RecurrentCheckpointMetadata):
            raise ValueError("Missing recurrent checkpoint task metadata")
        assert self._worker is not None
        if self._rank is None:
            raise RuntimeError("Recurrent checkpoint worker state must be bound")
        for task in metadata.tasks:
            if task.task_id in self._pending or task.task_id in self._rejected:
                raise ValueError("Checkpoint copy task was submitted twice")
            try:
                event = torch_dev.Event()
                event.record()
                future = self._worker.submit(
                    CheckpointTransferJob(
                        task.manifest,
                        self._rank,
                        task.direction,
                        task.block_ids,
                        event,
                    )
                )
            except UnsafeCheckpointCopyError:
                raise
            except Exception:
                # Event setup and submit fail before accepting a copy. Report a terminal
                # result for this task and continue draining the batch; other
                # ranks may already hold leases for every task in the batch.
                logger.exception("Recurrent checkpoint copy setup or submission failed")
                future = None
            if future is None:
                self._rejected.add(task.task_id)
            else:
                self._pending[task.task_id] = future

    def wait_for_layer_load(self, layer_name: str) -> None:
        """Live requests never read an unpublished import destination."""
        return

    def save_kv_layer(
        self, layer_name: str, kv_layer: torch.Tensor, attn_metadata: Any, **kwargs: Any
    ) -> None:
        """Only committed bundles are stored; mutable layer outputs are not exported."""
        return

    def wait_for_save(self) -> None:
        """Source pins permit stores to continue asynchronously after forward."""
        return

    def get_finished(self, finished_req_ids: set[str]) -> tuple[None, None]:
        """Request blocks need no deferred ownership; copy tasks own separate pins."""
        return None, None

    def build_connector_worker_meta(self) -> KVConnectorWorkerMetadata:
        """Return only drained rank completions and the initialization descriptor."""
        if self._rank is None or self._worker_layout is None:
            raise RuntimeError("Recurrent checkpoint worker state must be bound")
        results = {task: {self._rank: False} for task in self._rejected}
        self._rejected.clear()
        for task, future in tuple(self._pending.items()):
            if not future.done():
                continue
            try:
                success = future.result()
            except UnsafeCheckpointCopyError:
                raise
            except Exception:
                logger.exception(
                    "Recurrent checkpoint rank transfer failed after draining"
                )
                success = False
            results[task] = {self._rank: success}
            del self._pending[task]
        layouts = {}
        if not self._layout_sent:
            layouts[self._rank] = self._worker_layout
            self._layout_sent = True
        return RecurrentCheckpointWorkerMetadata(layouts, results)

    def update_connector_output(self, connector_output: KVConnectorOutput) -> None:
        """Publish imports and release pins only after distinct all-rank results."""
        metadata = connector_output.kv_connector_worker_meta
        if not isinstance(metadata, RecurrentCheckpointWorkerMetadata):
            return
        assert self._scheduler is not None
        self._scheduler.accept_layouts(metadata.layouts)
        self._scheduler.complete(metadata.results)

    # Keep deferred sample replies aggregated after the last task drains.
    _GATHER_TRAILING_STEPS = 2
    _gather_steps_left = 0

    def requires_all_worker_outputs(self) -> bool:
        """Gather rank metadata until layouts and copies drain, plus two steps.

        Called after build_connector_meta by vLLM's optional gather hook.
        The trailing steps cover deferred sample_tokens replies from an earlier
        execute_model call. Older vLLM versions ignore this hook and gather
        every step. Returns True conservatively without a scheduler bridge.
        """
        # Layout negotiation and copies remain pending until every rank reports.
        if self._scheduler is None or self.has_pending_push_work():
            self._gather_steps_left = self._GATHER_TRAILING_STEPS
            return True
        if self._gather_steps_left > 0:
            self._gather_steps_left -= 1
            return True
        return False

    def has_pending_push_work(self) -> bool:
        """Keep connector-only engine steps alive until collective copies drain."""
        return self._scheduler is not None and self._scheduler.has_pending

    def request_finished(
        self, request: Request, block_ids: list[int]
    ) -> tuple[bool, None]:
        """Cancel pending admission without recycling any in-flight copy pages."""
        assert self._scheduler is not None
        self._scheduler.finish_request(request.request_id)
        return False, None

    def request_finished_all_groups(
        self, request: Request, block_ids: tuple[list[int], ...]
    ) -> tuple[bool, None]:
        """Apply request-lifetime cleanup independently of hybrid group count."""
        return self.request_finished(request, [])

    def shutdown(self) -> None:
        """Drain all background copies before unregistering or unmapping worker SHM."""
        if self._worker is not None:
            self._worker.close()
        if self._mapping is not None:
            self._mapping.close()
            self._mapping = None
        self._client.close()

    @classmethod
    def get_required_kvcache_layout(cls, vllm_config: VllmConfig) -> str | None:
        """Opaque byte-page copies preserve the engine's native tensor layout."""
        return None
