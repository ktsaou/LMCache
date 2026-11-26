# SPDX-License-Identifier: Apache-2.0
"""Checkpoint roots name multimodal placeholder spans by content."""

# Standard
from types import SimpleNamespace

# Third Party
import pytest

pytest.importorskip("vllm.v1.core.boundary_checkpoint")

# Third Party
from vllm.multimodal.inputs import PlaceholderRange  # noqa: E402

# First Party
from lmcache.integration.vllm.checkpoint_scheduler import (  # noqa: E402
    CheckpointSchedulerBridge,
)
from lmcache.v1.multiprocess.checkpoint_identity import (  # noqa: E402
    CheckpointTokenRoots,
    checkpoint_namespace,
)

IDENTITY = dict(
    target_revision="target",
    draft_revision="draft",
    source_revision="source",
    layout={"page_bytes": 1024},
    parallel={"tp": 2, "dcp": 2},
)
TOKENS = list(range(300))


def roots(images, salt=b"{}"):
    bridge = SimpleNamespace(
        _identity=IDENTITY, _multimodal_salt=salt, _request_roots={}
    )
    request = SimpleNamespace(
        request_id="request",
        all_token_ids=TOKENS,
        cache_salt=None,
        mm_features=[
            SimpleNamespace(
                identifier=identifier,
                mm_position=PlaceholderRange(offset=offset, length=40),
            )
            for identifier, offset in images
        ]
        or None,
    )
    return CheckpointSchedulerBridge._roots(bridge, request)


def test_text_roots_are_unchanged():
    expected = CheckpointTokenRoots.build(checkpoint_namespace(IDENTITY, ""), TOKENS)
    assert roots([]) == expected


def test_images_separate_checkpoints_by_content_and_encoder_settings():
    a = roots([("image-a", 20)])
    assert a == roots([("image-a", 20)])
    assert a.prefix(300) != roots([("image-b", 20)]).prefix(300)
    assert a.prefix(300) != roots([("image-a", 20)], salt=b"mxfp8").prefix(300)
    # Prefixes that end before the image are shared with the text request.
    assert a.prefix(20) == roots([]).prefix(20)
    # A later image does not change the identity of the earlier prefix.
    assert a.prefix(100) == roots([("image-a", 20), ("image-c", 200)]).prefix(100)
