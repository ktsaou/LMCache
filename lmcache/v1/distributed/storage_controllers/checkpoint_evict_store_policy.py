# SPDX-License-Identifier: Apache-2.0
"""
Store recurrent checkpoints to L2 only when L1 evicts them.

Every request publishes request-boundary checkpoints of the complete
recurrent state. Writing each one through to L2 wears flash storage with
states that the next turn of the same conversation supersedes within
seconds, and pushes still-useful checkpoints out of L2 by capacity.

This policy keeps new checkpoint pages in L1 only. When L1 is about to evict
a page that is still current (not superseded by a newer checkpoint of the
same conversation) and not yet in L2, the page is written to L2 first and
evicted after the write completes. A conversation that stays active never
writes its checkpoints; one that goes idle writes its latest checkpoint once.
Superseded pages are evicted without a write. Ordinary KV chunks keep
write-through behavior. On a clean shutdown, checkpoint pages still only in
L1 are written to L2 within the configured time budget, current pages first
and then superseded ones.

Select it with ``--l2-store-policy checkpoint_on_evict``.
"""

# First Party
from lmcache.v1.distributed.storage_controllers.checkpoint_reuse_store_policy import (  # noqa: E501
    CheckpointReuseStorePolicy,
)
from lmcache.v1.distributed.storage_controllers.store_policy import (
    register_store_policy,
)


class CheckpointEvictStorePolicy(CheckpointReuseStorePolicy):
    """
    Write ordinary keys through; store checkpoint pages when L1 evicts them.

    ``select_reuse_targets`` (inherited) stores the checkpoint pages that the
    eviction path asks to persist.
    """

    def writes_checkpoints_on_evict(self) -> bool:
        """
        Report that checkpoint pages reach L2 on L1 eviction.

        Returns:
            True.
        """
        return True


register_store_policy("checkpoint_on_evict", CheckpointEvictStorePolicy)
