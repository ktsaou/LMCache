# SPDX-License-Identifier: Apache-2.0
"""Measure payload writes and restore outcomes before and after final drain."""

# Standard
from dataclasses import replace
from pathlib import Path
import json

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.internal_api import L2AdapterListener
from tests.v1.multiprocess.test_checkpoint_storage import (
    drain_l2_stores,
    large_manifest,
    make_manifest,
    open_store,
    publish_all,
    restores,
)


class WriteCounter(L2AdapterListener):
    def __init__(self) -> None:
        self.payload_bytes = 0

    def on_l2_keys_stored(self, keys: list[ObjectKey], sizes: list[int]) -> None:
        self.payload_bytes += sum(sizes)

    def on_l2_keys_accessed(self, keys: list[ObjectKey]) -> None:
        pass

    def on_l2_keys_deleted(self, keys: list[ObjectKey]) -> None:
        pass


@pytest.mark.parametrize("case", ["replacement", "retained_branches", "disk_pressure"])
def test_payload_write_volume(tmp_path: Path, case: str) -> None:
    entries = (
        [large_manifest(900 + i) for i in range(16)]
        if case == "disk_pressure"
        else [replace(make_manifest(), world_size=1) for _ in range(12)]
    )
    entries = [
        replace(entry, prefix=replace(entry.prefix, tail_tokens=(900 + i,)))
        for i, entry in enumerate(entries)
    ]
    if case == "replacement":
        entries = [replace(entry, prefix=entries[0].prefix) for entry in entries]
    totals = {}
    for policy in ("default", "checkpoint_on_evict"):
        path = tmp_path / policy
        path.mkdir()
        counter = WriteCounter()
        with open_store(
            path,
            True,
            store_policy=policy,
            disk_capacity_gb=(6 / 1024 if case == "disk_pressure" else 0),
        ) as resources:
            service, index, storage, mapping = resources
            storage.l2_adapters()[0][1].register_listener(counter)
            for entry in entries:
                publish_all(service, index, mapping, entry)
                # A completed turn separates write bursts equally for both policies.
                drain_l2_stores(storage)
            before = counter.payload_bytes
            assert restores(service, mapping, entries[-1])
        after = counter.payload_bytes
        with open_store(path, True, store_policy=policy) as resources:
            service, index, _, mapping = resources
            retained = [
                entry for entry in entries if index.find((entry.prefix,)) == entry
            ]
            restored = sum(restores(service, mapping, entry) for entry in retained)
            assert restored == len(retained)
        totals[policy] = {
            "before_drain": before,
            "after_drain": after,
            "restored_after_restart": restored,
        }
    print(
        "WRITE_VOLUME " + json.dumps({"case": case, "policies": totals}, sort_keys=True)
    )
    if case == "replacement":
        assert (
            totals["checkpoint_on_evict"]["after_drain"]
            < totals["default"]["after_drain"]
        )
    if case == "retained_branches":
        assert totals["checkpoint_on_evict"]["before_drain"] == 0
        assert (
            totals["checkpoint_on_evict"]["after_drain"]
            == totals["default"]["after_drain"]
        )
