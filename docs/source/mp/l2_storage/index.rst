Secondary KV Storage
====================

LMCache multiprocess mode supports a two-tier storage architecture:

- **L1 (fast tier)** -- CPU memory by default, or an NVMe slab via GPUDirect
  Storage (cuFile) when ``--gds-l1-path`` is set, managed by the L1 Manager.
  All KV cache chunks live here during active use. (Byte-array L2 adapters are
  unsupported under the GDS L1 tier, which exposes no L1 memory buffer.)
- **L2 (persistent)** -- Durable storage backends (NIXL-based or plain
  file-system/raw-block).  The StoreController asynchronously pushes data from L1
  to L2, and the PrefetchController loads data from L2 back into L1 on
  cache misses.

.. contents::
   :local:
   :depth: 2

Data Flow
---------

**Write path (L1 -> L2):**

1. vLLM stores KV cache chunks into L1 via the ``STORE`` RPC.
2. The ``StoreController`` detects new objects (via eventfd) and
   asynchronously submits store tasks to each configured L2 adapter.
3. The L2 adapter writes the data to its backend (e.g., local SSD via GDS).

**Read path (L2 -> L1):**

1. A ``LOOKUP`` RPC checks L1 for prefix hits.
2. For keys not found in L1, the ``PrefetchController`` submits lookup
   requests to L2 adapters.
3. If found in L2, the data is loaded back into L1 and read-locked for the
   pending ``RETRIEVE`` RPC.

Adapter Types
-------------

LMCache ships several L2 storage backends, grouped by medium under
:doc:`Supported Backends <supported_storages>`. Select one or more with the
``--l2-adapter`` flag.

Every adapter accepts, alongside its type-specific keys, the common
``"shared": true`` option. Set it when the adapter mounts a storage
domain that several LMCache instances share (one S3 bucket, one NFS
export): with :doc:`coordinator event reporting <../coordinator>`
enabled, the coordinator's key directory then identifies the pool by
the adapter's type name — keep **one pool per adapter type** across the
fleet — deduplicates its placements across instances, and keeps them
across instance restarts (the bytes outlive any single reporter).
Storage private to the instance (the default, ``false``) needs nothing.

.. code-block:: bash

    --l2-adapter '{"type": "fs", "base_path": "/mnt/nfs/lmcache", "shared": true}'

.. toctree::
   :maxdepth: 2

   supported_storages

Multiple Adapters (Cascade)
---------------------------

You can configure multiple L2 adapters by repeating the ``--l2-adapter``
argument.  Adapters are used in the order they are specified.  The
``StoreController`` pushes data to all configured adapters, and the
``PrefetchController`` queries adapters in order during lookups.

.. code-block:: bash

    # SSD (fast, smaller) + NVMe GDS (larger capacity)
    --l2-adapter '{"type": "nixl_store", "backend": "POSIX", "backend_params": {"file_path": "/data/ssd/l2", "use_direct_io": "false"}, "pool_size": 64}' \
    --l2-adapter '{"type": "nixl_store", "backend": "GDS", "backend_params": {"file_path": "/data/nvme/l2", "use_direct_io": "true"}, "pool_size": 128}'

Store and Prefetch Policies
----------------------------

The **store policy** controls how keys flow from L1 to L2: which adapters
receive each key and whether keys are deleted from L1 after a successful
L2 store.  The **prefetch policy** controls how keys flow from L2 back to
L1: when multiple adapters have the same key, the policy decides which
adapter loads it.

Select policies via CLI:

.. code-block:: bash

    --l2-store-policy default \
    --l2-prefetch-policy default

**Built-in policies:**

.. list-table::
   :header-rows: 1
   :widths: 15 15 70

   * - Flag
     - Name
     - Behaviour
   * - ``--l2-store-policy``
     - ``default``
     - Store all keys to all adapters.  Never delete from L1.
   * - ``--l2-store-policy``
     - ``skip_l1``
     - Buffer-only mode.  Store all keys to all adapters, then
       **delete them from L1** immediately.  Pair with
       ``--eviction-policy noop`` to avoid useless LRU overhead.
   * - ``--l2-store-policy``
     - ``checkpoint_on_reuse``
     - Store ordinary keys like ``default``.  Keep new recurrent checkpoint
       pages in L1 and store each one to L2 the first time a restore reads
       it, once per page.  Checkpoints that are never restored never reach
       L2; after a restart or eviction, restores fall back to the longest
       checkpoint that still exists.
   * - ``--l2-store-policy``
     - ``checkpoint_on_evict``
     - Store ordinary keys like ``default``.  Keep new recurrent checkpoint
       pages in L1 and store a page to L2 once, when L1 is about to evict it
       and no newer checkpoint of the same conversation has superseded it.
       On shutdown, current pages still only in L1 are written within
       ``--checkpoint-shutdown-flush-seconds``.  See
       :ref:`checkpoint-retention`.
   * - ``--l2-prefetch-policy``
     - ``default``
     - For each key, pick the first (lowest-indexed) adapter that has it.
       Prefetched keys are **temporary** (deleted after the reader finishes).
   * - ``--l2-prefetch-policy``
     - ``retain``
     - Same load plan as ``default``, but prefetched keys are **retained**
       permanently in L1.  Useful when prefetched data is likely reused
       by subsequent requests (e.g. shared system-prompt chunks).

Prefetch Concurrency
~~~~~~~~~~~~~~~~~~~~~

The ``--l2-prefetch-max-in-flight`` flag limits the number of concurrent
prefetch requests that the ``PrefetchController`` can have in flight at
any time.  A higher value increases L2-to-L1 throughput but also
increases L1 memory pressure from in-flight data.

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Flag
     - Default
     - Description
   * - ``--l2-prefetch-max-in-flight``
     - ``8``
     - Maximum number of concurrent prefetch requests.

Buffer-Only Mode
~~~~~~~~~~~~~~~~~

When L1 is used purely as a write buffer (all data lives in L2), use
``--l2-store-policy skip_l1`` together with ``--eviction-policy noop``.
This combination deletes keys from L1 as soon as they are stored to L2
and disables the LRU eviction tracker entirely, reducing memory and CPU
overhead.

.. code-block:: bash

    --eviction-policy noop \
    --l2-store-policy skip_l1 \
    --l2-prefetch-policy default

Policies are extensible -- new policies can be added by creating a file
in ``storage_controllers/`` and calling ``register_store_policy()`` or
``register_prefetch_policy()`` at import time.  See the design doc
``l2_adapters/design_docs/overall.md`` for details.

Serde (compression / quantization)
----------------------------------

Each adapter can optionally run a **serde** (serializer / deserializer)
that transforms data on the way in and out of L2 — e.g. fp8 quantization
for disk backends, or encryption for remote adapters. See
:doc:`KV Cache Compression </mp/serde>` for details and configuration.

.. toctree::
   :maxdepth: 1

   /mp/serde

Eviction
--------

LMCache supports eviction at both storage tiers so that each tier
can operate within a fixed capacity budget.

L1 Eviction
~~~~~~~~~~~

L1 eviction runs a single background thread that monitors overall L1
memory usage. When usage exceeds ``trigger_watermark``, the eviction
policy evicts a fraction of the least-recently-used keys.

**CLI flags:**

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Flag
     - Default
     - Description
   * - ``--eviction-policy``
     - *(required)*
     - Policy name: ``LRU``, ``IsolatedLRU``, or ``noop``. ``IsolatedLRU``
       maintains one LRU list per ``cache_salt`` and relies on per-salt
       quotas registered via the ``/quota`` HTTP endpoints; see
       :doc:`/mp/configuration` for the full description.
   * - ``--eviction-trigger-watermark``
     - ``0.8``
     - L1 usage fraction [0, 1] above which eviction is triggered.
   * - ``--eviction-ratio``
     - ``0.2``
     - Fraction of currently allocated L1 memory to evict per cycle.
   * - ``--write-back-on-evict``
     - disabled
     - Synchronously persist an eviction batch to a capable L2 adapter before
       deleting it from L1. Persistence failure preserves the L1 copy.
   * - ``--periodic-flush-interval``
     - ``0`` (disabled)
     - Seconds between bounded backup scans below the L1 eviction watermark.
       Backups add an L2 copy without deleting the L1 object.
   * - ``--emergency-evict-for-prefetch``
     - disabled
     - Permit non-warm L2 restores to make room by synchronously writing back
       and evicting LRU objects. Requires ``--write-back-on-evict``.

**Example:**

.. code-block:: bash

    --eviction-policy LRU \
    --eviction-trigger-watermark 0.8 \
    --eviction-ratio 0.2 \
    --write-back-on-evict

Without ``--write-back-on-evict``, L1 eviction keeps its existing discard
behavior. With the option enabled, the policy always targets L2, including
when no compatible adapter is available, so loss of the persistence path
cannot silently degrade to discard. Objects are read-locked, persisted in
bounded batches, unlocked, and deleted only when one adapter reports every
readable object durable. Partial stores, exceptions, adapter removal, and
concurrent locks preserve the affected L1 objects. Repeated failures use a
bounded exponential backoff.

The option currently requires at least one configured L2 adapter that exposes
the synchronous ``store_objects_sync()`` durability contract, such as the
native filesystem adapter. Incompatible adapters are ignored and reported in
the controller status and logs.

Periodic Backup
~~~~~~~~~~~~~~~

``--periodic-flush-interval SECONDS`` independently enables bounded backup
scans while L1 usage is below the eviction watermark. Each pass advances a
rotating cursor through at most 128 currently evictable objects, persists one
read-locked batch, and retains every L1 copy. It does not change ordinary L1
eviction behavior; use ``--write-back-on-evict`` separately when capacity
evictions must be durable.

Periodic backup accepts only adapters whose ``store_objects_sync()`` contract
supports a timeout. This bounds controller shutdown and runtime adapter
replacement when an L2 backend is unhealthy. Missing, partial, timed-out, or
failed backup stores leave L1 unchanged and are retried on a later scan.

Emergency Prefetch Pressure
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``--emergency-evict-for-prefetch`` lets a non-warm L2 restore request bounded
L1 space before allocation and retry only its out-of-memory reservations once.
Victims follow the normal LRU order and are deleted only after complete L2
durability. Hybrid restores calculate the request from each object group's own
layout rather than assuming that every group has the first group's size.
With ``IsolatedLRU``, emergency victims are restricted to the restoring
request's ``cache_salt``; a missing salt fails closed without eviction.

The synchronous eviction is capped at 60 percent of total L1 capacity for one
restore and has a one-second persistence deadline. Deadline exhaustion keeps
the remaining L1 objects and does not trip the normal backend-failure circuit
breaker. Native buffers from a timed-out store remain ineligible for eviction
until that store completes, preventing allocator reuse while the backend may
still read them. Warm-up requests never evict other sessions. Runtime adapter removal
disconnects the emergency path before closing the last bounded synchronous
backend, and a later compatible adapter reconnects it.

L2 Eviction
~~~~~~~~~~~

L2 eviction is **per-adapter** and **opt-in**. Each adapter can
independently declare an eviction policy by adding an ``"eviction"``
sub-object to its ``--l2-adapter`` JSON spec. Adapters without an
``"eviction"`` key have no eviction controller.

When L2 eviction is enabled for an adapter, a dedicated background
thread monitors that adapter's ``get_usage()`` value. Once usage
exceeds ``trigger_watermark``, the policy evicts keys until usage
drops by ``eviction_ratio``.

**``"eviction"`` sub-object fields:**

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Field
     - Default
     - Description
   * - ``eviction_policy``
     - *(required)*
     - Policy name: ``"LRU"`` or ``"noop"``.
   * - ``trigger_watermark``
     - ``0.8``
     - Adapter usage fraction [0, 1] above which eviction is triggered.
   * - ``eviction_ratio``
     - ``0.2``
     - Fraction of used capacity to evict per cycle.

**Example — nixl_store with LRU eviction:**

.. code-block:: bash

    --l2-adapter '{
      "type": "nixl_store",
      "backend": "POSIX",
      "backend_params": {"file_path": "/data/lmcache/l2", "use_direct_io": "false"},
      "pool_size": 128,
      "eviction": {
        "eviction_policy": "LRU",
        "trigger_watermark": 0.8,
        "eviction_ratio": 0.2
      }
    }'

**Adapter support:**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Adapter
     - L2 Eviction Support
   * - ``nixl_store``
     - Full support. ``delete`` frees pool slots; pinned keys (in-flight
       loads) are skipped and retried on the next cycle.
   * - ``nixl_store_dynamic``
     - Full support. ``delete`` removes data files from disk; pinned
       keys are skipped. ``get_usage`` is byte-based
       (``_total_bytes / max_capacity_bytes``).
   * - ``mock``
     - Full support. Useful for testing eviction behaviour without
       real storage hardware.
   * - ``raw_block``
     - Full shared/global eviction support. ``delete`` recycles raw-block
       slots; locked entries are skipped and retried on the next cycle.
   * - ``s3``
     - ``delete`` removes objects from the bucket and frees aggregate
       byte accounting. ``get_usage`` reports ``usage_fraction == -1.0``
       when ``max_capacity_gb`` is ``0`` (disabled); set a non-zero
       ``max_capacity_gb`` to enable the watermark-triggered eviction
       controller.
   * - ``hfbucket``
     - ``delete`` removes objects from the bucket and frees aggregate
       byte accounting. ``get_usage`` reports ``usage_fraction == -1.0``
       when ``max_capacity_gb`` is ``0`` (disabled); set a non-zero
       ``max_capacity_gb`` to enable the watermark-triggered eviction
       controller. Locked keys (in-flight loads) are skipped.
   * - ``dax``
     - Full support. ``delete`` removes unlocked keys from the in-memory
       index immediately and recycles fixed slots once active read borrows
       drain. Usage is slot-based.
   * - ``mooncake_store``
     - No eviction support (native connector adapter).
   * - ``fs``
     - No eviction support (``delete`` and ``get_usage`` are no-ops).
   * - native connectors
     - No eviction support.

.. note::

   Each L2 adapter instance gets its own independent eviction
   controller and policy.  Two adapters of the same type can have
   different watermarks or policies.

Combined L1 + L2 Eviction Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    --l1-size-gb 100 \
    --eviction-policy LRU \
    --eviction-trigger-watermark 0.8 \
    --eviction-ratio 0.2 \
    --l2-adapter '{
      "type": "nixl_store",
      "backend": "GDS",
      "backend_params": {"file_path": "/data/nvme/l2", "use_direct_io": "true"},
      "pool_size": 256,
      "eviction": {
        "eviction_policy": "LRU",
        "trigger_watermark": 0.9,
        "eviction_ratio": 0.1
      }
    }'

In this setup:

- L1 evicts from memory when it is 80 % full, reclaiming 20 % of
  allocated memory per cycle.
- L2 (NIXL/GDS) evicts from the storage pool when 90 % of pool slots
  are occupied, reclaiming 10 % per cycle.
- Both tiers use independent LRU policies, so each evicts its own
  least-recently-used keys.

Verifying L2 Storage
--------------------

Set ``LMCACHE_LOG_LEVEL=DEBUG`` to see L2 activity in the server logs:

.. code-block:: bash

    LMCACHE_LOG_LEVEL=DEBUG lmcache server \
        --l1-size-gb 100 --eviction-policy LRU \
        --l2-adapter '{"type": "nixl_store", "backend": "POSIX", "backend_params": {"file_path": "/data/lmcache/l2", "use_direct_io": "false"}, "pool_size": 64}'

Expected log messages when L2 is active:

.. code-block:: text

    LMCache DEBUG: Submitted store task ...
    LMCache DEBUG: L2 store task N completed ...
    LMCache DEBUG: Prefetch request submitted: X total keys, Y L1 prefix hits, Z remaining for L2

.. _checkpoint-retention:

Recurrent Checkpoint Retention
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Hybrid models (for example Qwen3.8-Flash-Next and GLM-5.3-Flash with its
MTP draft state) publish recurrent checkpoints at request boundaries: the
complete prompt, the response endpoint, and sometimes a prefill tail and the
leading instructions.  Checkpoints of one conversation share attention pages
by content, but the recurrent state, the partial last attention page and
auxiliary state are unique to each.

The vLLM integration reports each published checkpoint with its token
sequence and request.  When the prompt checkpoint of a new request is
published, the longest earlier checkpoint that it extends is where it
continues from.  That checkpoint stays current: the new request may be a
side request (a title or summary request that appends a task to the whole
conversation, or a sub-agent forked from it) while the conversation itself
goes on from the same point later.  The server marks as *superseded* the
unique pages of every other shorter checkpoint of that sequence produced by
an earlier request (the conversation has moved past them twice), and of the
other checkpoints those requests published that the new prompt is longer
than.  That includes a response endpoint that the chat template rewrote and
the next prompt therefore does not extend.  A checkpoint at least as long as
the new prompt is kept: the new request branched off before it.
Checkpoints of the same request and ``instruction`` checkpoints, which other
conversations share, are never marked.  With every store policy:

* L1 eviction drops stale superseded pages before any LRU victim.
* L2 eviction deletes stale superseded pages before any LRU victim.

A superseded page is stale at once, unless a later prompt had continued from
its checkpoint (a possible branch point, for example the end of a turn that
a sub-agent forked from and ran several turns on).  Such pages keep their
normal eviction order for ``--checkpoint-supersede-grace-seconds`` (default
300), so the conversation can still continue from them.  A lookup that finds
a superseded checkpoint makes it current again.

With ``--l2-store-policy checkpoint_on_evict`` in addition:

* A current checkpoint page is written to L2 once, when L1 is about to
  evict it, instead of on every request.  L1 keeps the page until the
  write completes (at most ``--checkpoint-write-timeout-seconds``).
* A superseded page is never written to L2 while serving, also during its
  grace period.
* Shutdown writes pages still only in L1, current pages first, so they can
  be restored after a restart.

A checkpoint is never offered once a page it needs is gone.  Before the last
copy of a superseded page is deleted from L1 or L2, the superseded
checkpoints that reference it are retired from the directory.  A lookup also
checks the pages of the longest candidate against L1 and the L2 inventory:
when a page was lost another way (L2 eviction, an administrative delete, a
page that was not written before a restart), the lookup retires that
candidate and returns the next shorter complete one in the same call.  A page
counts as lost only when every L2 adapter confirms it does not hold it;
``fs`` and ``fs_native`` check the object file, so pages another server
sharing the directory wrote still count.  With adapters that cannot confirm
an absence, the restore still validates every page and falls back to a
shorter checkpoint.

**Shutdown.**  ``--checkpoint-shutdown-flush-seconds`` (default 30) is the
budget of a clean shutdown.  At SIGTERM the server first keeps serving
checkpoint stores that are still in flight, so an engine stopping at the
same time can publish the checkpoints it is copying.  This takes at most a
third of the budget (and at most 5 s) and ends as soon as no store is in
flight.  With ``checkpoint_on_evict`` it then writes pages still only in L1
to L2 within the rest of the budget, current pages first, and logs how many
it could not write.  Give the process that much time between SIGTERM and
SIGKILL; with a shorter stop timeout the most valuable pages are written
first, and the checkpoints left incomplete are retired by lookups after the
restart.

**Sizing.**  With write-through (``default``) the L2 retention time is about::

    L2 capacity / (requests per minute x checkpoint bytes per request)

With ``checkpoint_on_evict`` the L2 receives roughly one current checkpoint
per conversation that leaves L1, so retention is about::

    L2 capacity / (conversations leaving L1 per minute x checkpoint bytes)

and superseded pages are reclaimed first.  The
``lmcache_mp_checkpoint_retention`` gauge reports, per ``stat``, superseded
pages (and those in their grace period), L1 drops and L2 evictions of
superseded pages, retired checkpoints, superseded checkpoints found again,
write-on-evict requests, completions and timeouts, and the checkpoint bytes
held in L2.
