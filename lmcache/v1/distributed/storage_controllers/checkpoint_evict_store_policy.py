# SPDX-License-Identifier: Apache-2.0
"""
Store recurrent checkpoints to L2 only when L1 evicts them.

Every request publishes request-boundary checkpoints of the complete
recurrent state. Writing each one through to L2 wears flash storage with
states that the next turn of the same conversation supersedes within
seconds, and pushes still-useful checkpoints out of L2 by capacity.

This policy keeps new checkpoint pages in L1 until memory pressure or shutdown.
Retained pages without an L2 copy are offered for bounded asynchronous
persistence before normal eviction. Supersession guides victim selection but
does not prove an older branch is dead. Sustained capacity pressure may retire
cold checkpoints coherently before reclaiming their last pages.

Ordinary KV chunks keep write-through behavior. Clean shutdown persists retained
pages within its time budget; unfinished checkpoints may be lost. Write savings
depend on retirement before persistence, not merely conversation activity.

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
