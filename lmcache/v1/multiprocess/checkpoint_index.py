# SPDX-License-Identifier: Apache-2.0
"""Atomic, CPU-only directory of immutable recurrent checkpoint manifests.

Payloads belong to the storage manager, not this directory. A caller may
acknowledge a rank only after all of that rank's payload writes have committed.
Lookup identifies a candidate, not a cache hit: every payload must still be
retrieved successfully before the serving engine can publish restored state.
"""

# Standard
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
import json
import sqlite3
import threading
import time

# Every checkpoint that ends inside the same hash block shares one lookup
# bucket, so a lookup must not scan the bucket. Each published tail also gets
# a polynomial hash in checkpoint_tails; a query hashes each of its own
# prefixes and probes only those. The hash only selects candidates.
_TAIL_HASH_BASE = 1_000_003
_TAIL_HASH_MODULUS = (1 << 61) - 1

# Stored tails are JSON token arrays. A candidate's tail is a token prefix of
# the :tail query exactly when the query repeats it without the closing
# bracket and continues with "," or "]". This confirms every hash candidate.
_TAIL_IS_PREFIX_OF_QUERY = (
    "substr(:tail, 1, length(c.tail) - 1) = substr(c.tail, 1, length(c.tail) - 1) "
    "AND substr(:tail, length(c.tail), 1) IN (x'2c', x'5d')"
)

# Published candidates whose tail hash equals one of the query's prefix hashes.
_CANDIDATES = (
    "SELECT c.tail,c.generation,c.world_size,c.payload FROM checkpoint_tails t "
    "JOIN checkpoints c ON c.generation=t.generation "
    "WHERE t.namespace=:namespace AND t.start_tokens=:start "
    "AND t.prefix_hash=:prefix_hash "
    "AND t.tail_hash IN (SELECT value FROM json_each(:prefix_hashes)) "
    f"AND c.num_tokens<=:num_tokens AND {_TAIL_IS_PREFIX_OF_QUERY}"
)


def _prefix_hashes(tokens: tuple[int, ...]) -> list[int]:
    """Return the tail hash of every nonempty prefix of ``tokens``."""
    hashes = []
    value = 0
    for token in tokens:
        value = (value * _TAIL_HASH_BASE + token + 1) % _TAIL_HASH_MODULUS
        hashes.append(value)
    return hashes


def _query_parameters(query: "CheckpointPrefix") -> dict[str, object]:
    return {
        "namespace": query.namespace,
        "start": query.start_tokens,
        "prefix_hash": query.prefix_hash,
        "num_tokens": query.num_tokens,
        "tail": json.dumps(query.tail_tokens).encode(),
        "prefix_hashes": json.dumps(_prefix_hashes(query.tail_tokens)),
    }


@dataclass(frozen=True)
class CheckpointPrefix:
    """A namespace-isolated prefix ending between ordinary hash boundaries.

    ``namespace`` authenticates weights, draft, tensor layout, parallel geometry,
    hash algorithm and cache salt. ``prefix_hash`` authenticates the first
    ``start_tokens`` tokens; ``tail_tokens`` identifies the remaining tokens.
    Neither worker addresses nor GPU block IDs belong in a persistent key.
    """

    namespace: str
    start_tokens: int
    prefix_hash: bytes
    tail_tokens: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.namespace or len(self.namespace) > 256:
            raise ValueError("checkpoint namespace must contain 1..256 characters")
        if self.start_tokens < 0 or not self.tail_tokens:
            raise ValueError("checkpoint prefix requires a nonempty token tail")
        if len(self.tail_tokens) > 65536 or any(
            type(token) is not int or not 0 <= token < 2**31
            for token in self.tail_tokens
        ):
            raise ValueError("checkpoint token tail exceeds the wire limits")
        if len(self.prefix_hash) not in (0, 32) or (
            self.start_tokens > 0 and not self.prefix_hash
        ):
            raise ValueError("a nonempty hashed prefix requires a 32-byte digest")

    @property
    def num_tokens(self) -> int:
        """Return the number of target tokens represented by the prefix."""
        return self.start_tokens + len(self.tail_tokens)


@dataclass(frozen=True)
class CheckpointManifest:
    """One all-rank payload generation for an exact token prefix.

    ``generation`` is an immutable identifier shared by all producer ranks; it
    may be content-derived to make exact publication idempotent. ``payload`` is
    versioned JSON describing immutable storage object keys and layouts;
    consumers validate its schema separately. The directory never merges
    manifests or payloads from different generations.
    """

    generation: str
    prefix: CheckpointPrefix
    world_size: int
    payload: bytes

    def __post_init__(self) -> None:
        if not self.generation or len(self.generation) > 128:
            raise ValueError("checkpoint generation must contain 1..128 characters")
        if not 1 <= self.world_size <= 1024:
            raise ValueError("checkpoint world size must be in 1..1024")
        if not self.payload or len(self.payload) > 16 * 1024 * 1024:
            raise ValueError("checkpoint manifest must contain at most 16 MiB")
        decoded = json.loads(self.payload)
        if not isinstance(decoded, dict) or decoded.get("schema_version") not in (1, 2):
            raise ValueError("unsupported checkpoint manifest schema")


@dataclass
class _PendingManifest:
    manifest: CheckpointManifest
    ranks: set[int] = field(default_factory=set)
    started: float = field(default_factory=time.monotonic)


class CheckpointIndex:
    """Publish complete checkpoint generations in RAM or a SQLite file.

    Pending generations remain memory-only. A process restart therefore cannot
    expose a partially acknowledged bundle. SQLite transactions atomically
    replace the entire manifest for an exact prefix. Evicted payloads may leave
    stale entries; failed retrieval must invalidate that specific generation.

    Args:
        path: SQLite database file, or ``None`` for a RAM-only directory. The
            containing directory must already exist and be trusted server data.
        max_entries: Maximum published manifests, evicting least-recently-used
            entries. Payload eviction is independently owned by storage tiers,
            so this should cover at least as many generations as L2 retains.
        max_pending: Maximum unpublished generations admitted concurrently.

    Raises:
        ValueError: If either capacity is not a positive integer or the schema
            is unknown.
        sqlite3.Error: If the database cannot be opened or committed.
    """

    def __init__(
        self,
        path: Path | None = None,
        *,
        max_entries: int = 65536,
        max_pending: int = 128,
    ) -> None:
        if any(
            type(capacity) is not int or capacity <= 0
            for capacity in (max_entries, max_pending)
        ):
            raise ValueError(
                "checkpoint directory capacities must be positive integers"
            )
        self._max_entries = max_entries
        self._max_pending = max_pending
        self._pending: dict[str, _PendingManifest] = {}
        self._lock = threading.RLock()
        self._listeners: list[Callable[[str, CheckpointManifest | str], None]] = []
        self._close_callbacks: list[Callable[[], None]] = []
        self._closed = False
        self._closing = False
        self._db = sqlite3.connect(
            str(path) if path is not None else ":memory:", check_same_thread=False
        )
        version = self._db.execute("PRAGMA user_version").fetchone()[0]
        if version not in (0, 1):
            self._db.close()
            raise ValueError(f"unsupported checkpoint directory schema {version}")
        # A committed directory entry must survive process failure independently
        # of the asynchronous L2 payload writer. Missing L2 payloads remain misses.
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=FULL")
        with self._db:
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS checkpoints ("
                "namespace TEXT NOT NULL, start_tokens INTEGER NOT NULL, "
                "prefix_hash BLOB NOT NULL, tail BLOB NOT NULL, "
                "num_tokens INTEGER NOT NULL, generation TEXT UNIQUE NOT NULL, "
                "world_size INTEGER NOT NULL, payload BLOB NOT NULL, "
                "access_order INTEGER NOT NULL, "
                "PRIMARY KEY(namespace, start_tokens, prefix_hash, tail))"
            )
            # Capacity trimming walks entries by recency on every publication.
            self._db.execute(
                "CREATE INDEX IF NOT EXISTS checkpoints_access_order "
                "ON checkpoints(access_order)"
            )
            # A separate table keeps the checkpoints schema readable and
            # writable by older builds. Rows they add or replace are
            # reconciled here on the next open; lookups join on generation,
            # so a stale row never matches.
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS checkpoint_tails ("
                "generation TEXT PRIMARY KEY, namespace TEXT NOT NULL, "
                "start_tokens INTEGER NOT NULL, prefix_hash BLOB NOT NULL, "
                "tail_hash INTEGER NOT NULL)"
            )
            self._db.execute(
                "CREATE INDEX IF NOT EXISTS checkpoint_tails_lookup ON "
                "checkpoint_tails(namespace, start_tokens, prefix_hash, tail_hash)"
            )
            self._db.execute(
                "DELETE FROM checkpoint_tails WHERE generation NOT IN "
                "(SELECT generation FROM checkpoints)"
            )
            missing = self._db.execute(
                "SELECT generation,namespace,start_tokens,prefix_hash,tail "
                "FROM checkpoints WHERE generation NOT IN "
                "(SELECT generation FROM checkpoint_tails)"
            ).fetchall()
            self._db.executemany(
                "INSERT INTO checkpoint_tails VALUES (?,?,?,?,?)",
                (
                    (*row[:4], _prefix_hashes(tuple(json.loads(row[4])))[-1])
                    for row in missing
                ),
            )
            self._db.execute("PRAGMA user_version=1")
        self._access_order = self._db.execute(
            "SELECT COALESCE(MAX(access_order), 0) FROM checkpoints"
        ).fetchone()[0]

    @property
    def lifecycle_lock(self) -> threading.RLock:
        """Serialize directory mutations with payload ownership transitions.

        Callers must not wait for capacity or backend I/O while holding this
        reentrant lock. L1 callbacks must never acquire it.
        """
        return self._lock

    def add_lifecycle_listener(
        self,
        listener: Callable[[str, CheckpointManifest | str], None],
        before_close: Callable[[], None],
    ) -> None:
        """Subscribe to committed changes under the lifecycle lock.

        Events are publish/touch (manifest) and remove/abort (generation).
        The close callback runs outside the lock while SQLite is still open.
        Listeners must be nonblocking and must not raise.
        """
        with self._lock:
            self._listeners.append(listener)
            self._close_callbacks.append(before_close)

    def manifests(self) -> list[CheckpointManifest]:
        """Return published manifests ordered from least to most recently used."""
        with self._lock:
            generations = self._db.execute(
                "SELECT generation FROM checkpoints ORDER BY access_order"
            ).fetchall()
            return [
                manifest
                for (generation,) in generations
                if (manifest := self.get(generation)) is not None
            ]

    def begin(self, manifest: CheckpointManifest) -> bool:
        """Stage a complete manifest before rank acknowledgements arrive.

        Args:
            manifest: Immutable generation and expected rank count.

        Returns:
            False if the unpublished-generation capacity is exhausted or the
            identical manifest is already pending/published; True only when a
            new generation is admitted. Duplicate suppression happens before
            any SHM reservation or GPU copy.

        Raises:
            ValueError: If the generation is reused for a different manifest or
                has already been published.
        """
        with self._lock:
            pending = self._pending.get(manifest.generation)
            if self._closing:
                return False
            if pending is not None:
                if pending.manifest != manifest:
                    raise ValueError("checkpoint generation changed during store")
                return False
            published = self._db.execute(
                "SELECT namespace,start_tokens,prefix_hash,tail,world_size,payload "
                "FROM checkpoints WHERE generation=?",
                (manifest.generation,),
            ).fetchone()
            if published is not None:
                prefix = manifest.prefix
                expected = (
                    prefix.namespace,
                    prefix.start_tokens,
                    prefix.prefix_hash,
                    json.dumps(prefix.tail_tokens).encode(),
                    manifest.world_size,
                    manifest.payload,
                )
                if published != expected:
                    raise ValueError("checkpoint generation changed after publication")
                return False
            if len(self._pending) >= self._max_pending:
                return False
            self._pending[manifest.generation] = _PendingManifest(manifest)
            return True

    def acknowledge(self, generation: str, rank: int) -> bool:
        """Publish after all distinct ranks confirm committed payload writes.

        Args:
            generation: Identifier passed to ``begin``.
            rank: Producer rank in ``[0, world_size)``.

        Returns:
            True only when this call publishes the complete manifest. Unknown,
            aborted and already-published generations return False.

        Raises:
            ValueError: If the rank is outside the declared world size.
            sqlite3.Error: If atomic publication fails; the pending generation
                remains available for a retry or explicit abort.
        """
        with self._lock:
            pending = self._pending.get(generation)
            if pending is None:
                return False
            manifest = pending.manifest
            if not 0 <= rank < manifest.world_size:
                raise ValueError("checkpoint acknowledgement rank is out of range")
            pending.ranks.add(rank)
            if len(pending.ranks) != manifest.world_size:
                return False
            prefix = manifest.prefix
            self._access_order += 1
            tail = json.dumps(prefix.tail_tokens).encode()
            with self._db:
                removed = [
                    row[0]
                    for row in self._db.execute(
                        "SELECT generation FROM checkpoints WHERE namespace=? "
                        "AND start_tokens=? AND prefix_hash=? AND tail=?",
                        (
                            prefix.namespace,
                            prefix.start_tokens,
                            prefix.prefix_hash,
                            tail,
                        ),
                    )
                ]
                # The generation replaced for this exact prefix, if any.
                self._db.execute(
                    "DELETE FROM checkpoint_tails WHERE generation IN ("
                    "SELECT generation FROM checkpoints WHERE namespace=? "
                    "AND start_tokens=? AND prefix_hash=? AND tail=?)",
                    (prefix.namespace, prefix.start_tokens, prefix.prefix_hash, tail),
                )
                self._db.execute(
                    "INSERT OR REPLACE INTO checkpoints VALUES (?,?,?,?,?,?,?,?,?)",
                    (
                        prefix.namespace,
                        prefix.start_tokens,
                        prefix.prefix_hash,
                        tail,
                        prefix.num_tokens,
                        generation,
                        manifest.world_size,
                        manifest.payload,
                        self._access_order,
                    ),
                )
                self._db.execute(
                    "INSERT OR REPLACE INTO checkpoint_tails VALUES (?,?,?,?,?)",
                    (
                        generation,
                        prefix.namespace,
                        prefix.start_tokens,
                        prefix.prefix_hash,
                        _prefix_hashes(prefix.tail_tokens)[-1],
                    ),
                )
                removed.extend(
                    row[0]
                    for row in self._db.execute(
                        "SELECT generation FROM checkpoints ORDER BY access_order DESC "
                        "LIMIT -1 OFFSET ?",
                        (self._max_entries,),
                    )
                )
                for table in ("checkpoint_tails", "checkpoints"):
                    self._db.execute(
                        f"DELETE FROM {table} WHERE generation IN ("
                        "SELECT generation FROM checkpoints "
                        "ORDER BY access_order DESC LIMIT -1 OFFSET ?)",
                        (self._max_entries,),
                    )
            del self._pending[generation]
            for retired in removed:
                self._notify("remove", retired)
            self._notify("publish", manifest)
            return True

    def is_pending(self, manifest: CheckpointManifest) -> bool:
        """Check an immutable store descriptor without reopening aborted work.

        Args:
            manifest: Descriptor supplied by a producer rank.

        Returns:
            True only if the identical generation remains staged.

        Raises:
            ValueError: If a rank changes any part of the staged descriptor.
        """
        with self._lock:
            pending = self._pending.get(manifest.generation)
            if pending is None:
                return False
            if pending.manifest != manifest:
                raise ValueError("checkpoint generation changed during store")
            return True

    def find(self, prefixes: tuple[CheckpointPrefix, ...]) -> CheckpointManifest | None:
        """Find the longest published prefix of any supplied hash-block tail.

        Args:
            prefixes: Query tails whose preceding tokens are authenticated by
                their chained hashes. All candidates must share one namespace.

        Returns:
            The complete manifest with the greatest matching token count, or
            None. Returned payloads still require all-rank retrieval validation.

        Raises:
            ValueError: If the query mixes isolation namespaces.
        """
        if len({prefix.namespace for prefix in prefixes}) > 1:
            raise ValueError("checkpoint lookup cannot mix namespaces")
        with self._lock:
            best: CheckpointManifest | None = None
            for query in sorted(
                prefixes, key=lambda prefix: prefix.num_tokens, reverse=True
            ):
                if best is not None and query.num_tokens <= best.prefix.num_tokens:
                    break
                row = self._db.execute(
                    f"{_CANDIDATES} ORDER BY c.num_tokens DESC LIMIT 1",
                    _query_parameters(query),
                ).fetchone()
                if row is None:
                    continue
                tail_blob, generation, world_size, payload = row
                prefix = CheckpointPrefix(
                    query.namespace,
                    query.start_tokens,
                    query.prefix_hash,
                    tuple(json.loads(tail_blob)),
                )
                if best is None or prefix.num_tokens > best.prefix.num_tokens:
                    best = CheckpointManifest(generation, prefix, world_size, payload)
            if best is not None:
                self._access_order += 1
                with self._db:
                    self._db.execute(
                        "UPDATE checkpoints SET access_order=? WHERE generation=?",
                        (self._access_order, best.generation),
                    )
                self._notify("touch", best)
            return best

    def get(self, generation: str) -> CheckpointManifest | None:
        """Return one published manifest by generation, without touching LRU."""
        with self._lock:
            row = self._db.execute(
                "SELECT namespace,start_tokens,prefix_hash,tail,world_size,payload "
                "FROM checkpoints WHERE generation=?",
                (generation,),
            ).fetchone()
        if row is None:
            return None
        namespace, start_tokens, prefix_hash, tail_blob, world_size, payload = row
        prefix = CheckpointPrefix(
            namespace, start_tokens, prefix_hash, tuple(json.loads(tail_blob))
        )
        return CheckpointManifest(generation, prefix, world_size, payload)

    def ancestors(
        self, prefixes: tuple[CheckpointPrefix, ...], below_tokens: int
    ) -> list[CheckpointManifest]:
        """Return every published manifest that is a proper prefix of a sequence.

        Args:
            prefixes: The sequence's hash-block roots, as for :meth:`find`.
            below_tokens: Only manifests shorter than this are returned.

        Returns:
            Matching manifests, shortest first. Access order is unchanged.

        Raises:
            ValueError: If the query mixes isolation namespaces.
        """
        if len({prefix.namespace for prefix in prefixes}) > 1:
            raise ValueError("checkpoint lookup cannot mix namespaces")
        found: list[CheckpointManifest] = []
        with self._lock:
            for query in prefixes:
                if query.start_tokens >= below_tokens:
                    continue
                rows = self._db.execute(
                    f"{_CANDIDATES} AND c.num_tokens<:below",
                    {**_query_parameters(query), "below": below_tokens},
                )
                for tail_blob, generation, world_size, payload in rows:
                    prefix = CheckpointPrefix(
                        query.namespace,
                        query.start_tokens,
                        query.prefix_hash,
                        tuple(json.loads(tail_blob)),
                    )
                    found.append(
                        CheckpointManifest(generation, prefix, world_size, payload)
                    )
        return sorted(found, key=lambda manifest: manifest.prefix.num_tokens)

    def abort_stale(self, older_than_seconds: float) -> list[str]:
        """Discard pending generations whose producers never finished.

        A producer that dies between ``begin`` and its last acknowledgement
        leaves its generation pending forever: the same generation can never
        be staged again, and it permanently uses one unit of pending capacity.

        Args:
            older_than_seconds: Minimum age of a discarded pending generation.

        Returns:
            The discarded generations.
        """
        cutoff = time.monotonic() - older_than_seconds
        with self._lock:
            stale = [
                generation
                for generation, pending in self._pending.items()
                if pending.started < cutoff
            ]
            for generation in stale:
                del self._pending[generation]
                self._notify("abort", generation)
        return stale

    def abort(self, generation: str) -> None:
        """Discard an unpublished generation after its payload writers drain.

        Args:
            generation: Pending generation identifier; unknown IDs are ignored.
        """
        with self._lock:
            self._pending.pop(generation, None)
            self._notify("abort", generation)

    def invalidate(self, generation: str) -> None:
        """Remove only the generation whose payload retrieval failed.

        Args:
            generation: Failed generation, not merely its shared token prefix.
                Invalidating a replaced generation cannot remove its replacement.
        """
        with self._lock:
            with self._db:
                for table in ("checkpoint_tails", "checkpoints"):
                    self._db.execute(
                        f"DELETE FROM {table} WHERE generation=?", (generation,)
                    )
            self._notify("remove", generation)

    def report_status(self) -> dict[str, int]:
        """Return directory counts without asserting that payload pages are resident.

        Returns:
            Numbers of published and pending generations. A published manifest
            still requires an all-page, all-rank retrieval before it is a hit.
        """
        with self._lock:
            return {
                "published_generations": self._db.execute(
                    "SELECT COUNT(*) FROM checkpoints"
                ).fetchone()[0],
                "pending_generations": len(self._pending),
            }

    def close(self) -> None:
        """Discard pending generations and close the database connection."""
        with self._lock:
            if self._closed:
                return
            self._closing = True
        for callback in self._close_callbacks:
            callback()
        with self._lock:
            for generation in list(self._pending):
                self._notify("abort", generation)
            self._pending.clear()
            self._db.close()
            self._closed = True

    def _notify(self, event: str, value: CheckpointManifest | str) -> None:
        for listener in self._listeners:
            listener(event, value)
