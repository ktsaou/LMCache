# SPDX-License-Identifier: Apache-2.0
"""RAM allocation failures must not delist intact disk checkpoints."""

# Standard
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import patch
import hashlib

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import (
    MemoryLayoutDesc,
    ObjectKey,
    PrefetchHandle,
    PrefetchRequestSpec,
    TrimPolicy,
)
from lmcache.v1.distributed.storage_controllers.prefetch_controller import (
    PrefetchResult,
)
from lmcache.v1.multiprocess.checkpoint_storage import (
    checkpoint_object_keys,
    checkpoint_page_groups,
)
from tests.v1.multiprocess.test_checkpoint_storage import (
    make_manifest,
    open_store,
    poll,
    publish_before_restart,
    restores,
)


@pytest.mark.parametrize("native", [False, True])
def test_restore_accounts_for_page_alignment(tmp_path: Path, native: bool) -> None:
    entry = replace(make_manifest(), world_size=1)
    publish_before_restart(tmp_path, native, [entry])
    with open_store(tmp_path, native, admission_timeout_seconds=0.15) as (
        service,
        index,
        storage,
        mapping,
    ):
        filler = ObjectKey(hashlib.sha256(b"alignment-filler").digest(), "filler", 0, 0)
        _, total = storage.get_l1_usage()
        alignment = storage.l1_memory_desc.align_bytes
        reserved = storage.reserve_write_detailed(
            [filler],
            MemoryLayoutDesc([torch.Size([total - alignment])], [torch.uint8]),
            "new",
        )
        assert reserved[filler][1] is not None
        try:
            # Raw payload bytes fit; seven separate aligned allocations do not.
            assert poll(service, service.begin_retrieve(entry, 0)) is False
            assert index.get(entry.generation) == entry
        finally:
            storage.abort_write([filler])
        assert restores(service, mapping, entry)


@pytest.mark.parametrize("native", [False, True])
def test_restore_uses_allocation_result_before_capacity_returns(
    tmp_path: Path, native: bool
) -> None:
    entry = replace(make_manifest(), world_size=1)
    publish_before_restart(tmp_path, native, [entry])
    with open_store(tmp_path, native, admission_timeout_seconds=0.15) as (
        service,
        index,
        storage,
        mapping,
    ):
        filler = ObjectKey(
            hashlib.sha256(b"concurrent-writer").digest(), "filler", 0, 0
        )
        _, total = storage.get_l1_usage()
        submit = storage.submit_prefetch_task
        query = storage.query_prefetch_status_detailed
        attempts = 0

        def reserve_then_submit(*args: Any, **kwargs: Any) -> PrefetchHandle:
            nonlocal attempts
            attempts += 1
            result = storage.reserve_write_detailed(
                [filler], MemoryLayoutDesc([torch.Size([total])], [torch.uint8]), "new"
            )
            assert result[filler][1] is not None
            return submit(*args, **kwargs)

        def release_before_consumption(handle: PrefetchHandle) -> PrefetchResult | None:
            result = query(handle)
            if result is not None:
                # Enforce a concurrent writer finishing after allocation failed.
                storage.abort_write([filler])
            return result

        try:
            with (
                patch.object(
                    storage, "submit_prefetch_task", side_effect=reserve_then_submit
                ),
                patch.object(
                    storage,
                    "query_prefetch_status_detailed",
                    side_effect=release_before_consumption,
                ),
            ):
                assert poll(service, service.begin_retrieve(entry, 0)) is False
            assert attempts >= 2
            assert index.get(entry.generation) == entry
        finally:
            storage.abort_write([filler])
        assert restores(service, mapping, entry)


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("detailed", [False, True])
def test_prefetch_queries_consume_capacity_failure_once(
    tmp_path: Path, native: bool, detailed: bool
) -> None:
    entry = replace(make_manifest(), world_size=1)
    publish_before_restart(tmp_path, native, [entry])
    with open_store(tmp_path, native) as (service, index, storage, mapping):
        filler = ObjectKey(hashlib.sha256(b"query-filler").digest(), "filler", 0, 0)
        _, total = storage.get_l1_usage()
        reserved = storage.reserve_write_detailed(
            [filler], MemoryLayoutDesc([torch.Size([total])], [torch.uint8]), "new"
        )
        assert reserved[filler][1] is not None
        keys = [key for group in checkpoint_object_keys(entry, 0) for key in group]
        layouts = {
            i: MemoryLayoutDesc([torch.Size([group.page_bytes])], [torch.uint8])
            for i, group in enumerate(checkpoint_page_groups(entry))
        }
        try:
            handle = storage.submit_prefetch_task(
                PrefetchRequestSpec(keys, layouts, policy=TrimPolicy.SPARSE)
            )
            assert storage.wait_prefetch_status(handle, timeout=5)
            if detailed:
                result = storage.query_prefetch_status_detailed(handle)
                assert result is not None and result.reservation_failed
                found = result.found
            else:
                found = storage.query_prefetch_status(handle)
            assert found is not None and found.popcount() == 0
            assert storage.query_prefetch_status_detailed(handle) is None
            assert storage.query_prefetch_status(handle) is None
            assert storage.query_prefetch_lookup_hits(handle) is None
            assert index.get(entry.generation) == entry
        finally:
            storage.abort_write([filler])
        assert restores(service, mapping, entry)
