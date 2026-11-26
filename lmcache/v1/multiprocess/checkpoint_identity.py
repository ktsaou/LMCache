# SPDX-License-Identifier: Apache-2.0
"""Deterministic, restart-safe identities for recurrent checkpoint lookup."""

# Standard
from array import array
from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import json
import sys

# First Party
from lmcache.v1.multiprocess.checkpoint_index import CheckpointPrefix


def _content_digest(prefix: CheckpointPrefix, discriminator: bytes) -> str:
    encoded = json.dumps(
        {
            "namespace": prefix.namespace,
            "start_tokens": prefix.start_tokens,
            "prefix_hash": prefix.prefix_hash.hex(),
            "tail_tokens": prefix.tail_tokens,
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(
        b"lmcache-recurrent-content-v1\0" + discriminator + b"\0" + encoded
    ).hexdigest()


def checkpoint_page_content_key(prefix: CheckpointPrefix, discriminator: str) -> str:
    """Return a stable SHA256 key for one semantically immutable cache page.

    ``prefix`` must end at the last token represented by the page. The caller's
    discriminator separates attention pages from endpoint and auxiliary state;
    rank and storage-group isolation are added by the storage layer.
    """
    return _content_digest(prefix, discriminator.encode())


def checkpoint_generation(prefix: CheckpointPrefix, payload: bytes) -> str:
    """Return an idempotent generation ID for one exact immutable manifest.

    Args:
        prefix: Authenticated exact token-prefix identity.
        payload: Canonical manifest payload describing the byte layout.

    Returns:
        A stable versioned SHA256 identifier no longer than the wire limit.

    Generation identity is metadata-only and intentionally excludes transient
    request IDs, GPU block addresses, and storage capacity. Publishing the same
    token prefix and byte layout again can therefore stop before reserving SHM
    or copying GPU pages, including after a durable-index restart.
    """
    prefix_digest = _content_digest(prefix, "manifest".encode())
    digest = hashlib.sha256(
        b"lmcache-recurrent-generation-v1\0"
        + bytes.fromhex(prefix_digest)
        + b"\0"
        + payload
    ).hexdigest()
    return f"recurrent-content-v1:{digest}"


def checkpoint_namespace(identity: dict[str, object], cache_salt: str) -> str:
    """Authenticate the execution contract and a request's cache isolation salt.

    Args:
        identity: Immutable weights, source and layout contract. Required fields
            are target_revision, draft_revision (empty without speculation),
            source_revision, layout, and parallel. Revision values identify
            content, not a mutable branch name or a local directory path.
        cache_salt: Request isolation salt; an empty string is a distinct value.

    Returns:
        Versioned SHA256 namespace, stable across processes and pool capacities.

    Raises:
        ValueError: If any required identity field is missing or empty.
        TypeError: If the identity cannot be represented as canonical JSON.
    """
    required = {
        "target_revision",
        "draft_revision",
        "source_revision",
        "layout",
        "parallel",
    }
    if not required.issubset(identity) or any(
        not identity[key] for key in required - {"draft_revision"}
    ):
        raise ValueError(
            "Checkpoint identity requires immutable weights, source and layout"
        )
    encoded = json.dumps(
        {"schema_version": 1, "identity": identity, "cache_salt": cache_salt},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return "recurrent-sha256-v1:" + hashlib.sha256(encoded).hexdigest()


def _chain_seed(chunk_tokens: int) -> bytes:
    """Digest that starts the token hash chain of one hash unit size."""
    return hashlib.sha256(
        b"lmcache-recurrent-token-roots-v1\0" + chunk_tokens.to_bytes(4, "little")
    ).digest()


def _chain_roots(
    namespace: str,
    token_ids: Sequence[int],
    chunk_tokens: int,
    start: int,
    chain: bytes,
) -> tuple[list[CheckpointPrefix], list[bytes]]:
    """Hash ``token_ids`` from a chunk boundary onward.

    Args:
        namespace: Result of checkpoint_namespace for this request.
        token_ids: The whole token sequence.
        chunk_tokens: Hash unit; ``start`` is a multiple of it.
        start: First token to hash.
        chain: Chain digest of the ``start`` tokens before it.

    Returns:
        One root per chunk from ``start``, and the chain digest after each.

    Raises:
        ValueError: If a token is outside the wire domain.
    """
    roots = []
    end_hashes = []
    for chunk_start in range(start, len(token_ids), chunk_tokens):
        tail = tuple(token_ids[chunk_start : chunk_start + chunk_tokens])
        roots.append(CheckpointPrefix(namespace, chunk_start, chain, tail))
        packed = array("I", tail)
        if packed.itemsize != 4:
            raise ValueError("Checkpoint tokens require 32-bit unsigned integers")
        if sys.byteorder != "little":
            packed.byteswap()
        chain = hashlib.sha256(chain + packed.tobytes()).digest()
        end_hashes.append(chain)
    return roots, end_hashes


@dataclass(frozen=True)
class CheckpointTokenRoots:
    """Restart-stable hash roots with exact token tails for longest-prefix lookup.

    Unlike process-seeded GPU cache hashes, this format uses an explicit SHA256
    domain and unsigned little-endian token IDs. A root covers at most one
    chunk's tail; all query tails together contain each token only once.
    """

    namespace: str
    roots: tuple[CheckpointPrefix, ...]
    chunk_tokens: int
    end_hashes: tuple[bytes, ...]

    @classmethod
    def build(
        cls, namespace: str, token_ids: Sequence[int], *, chunk_tokens: int = 4096
    ) -> "CheckpointTokenRoots":
        """Build linear-size queries for an authenticated token sequence.

        Args:
            namespace: Result of checkpoint_namespace for this request.
            token_ids: Nonnegative token IDs below 2**31; empty inputs yield no roots.
            chunk_tokens: Hash unit, in 1..65536; part of the wire format.

        Returns:
            Roots matching checkpoints at any exact token offset in the sequence.

        Raises:
            ValueError: If the chunk size or any token is outside the wire domain.
        """
        if not 1 <= chunk_tokens <= 65536:
            raise ValueError("Checkpoint hash unit must be in 1..65536")
        roots, end_hashes = _chain_roots(
            namespace, token_ids, chunk_tokens, 0, _chain_seed(chunk_tokens)
        )
        return cls(namespace, tuple(roots), chunk_tokens, tuple(end_hashes))

    def extended(self, token_ids: Sequence[int]) -> "CheckpointTokenRoots":
        """Return the roots of a sequence that continues the covered one.

        Complete hash chunks keep their roots and digests; only the tokens
        from the last incomplete chunk onward are hashed. A request stores
        several checkpoints while its token list only grows, so each store
        costs the new tokens rather than the whole prompt.

        Args:
            token_ids: The covered tokens followed by zero or more new ones.
                The caller guarantees the earlier tokens are unchanged, as for
                the append-only token list of one request.

        Returns:
            Roots equal to ``build(namespace, token_ids)``. The last one or
            two covered chunks are compared with ``token_ids``; a shorter or
            differing sequence is built from the start instead.

        Raises:
            ValueError: If a new token is outside the wire domain.
        """
        if not self.roots:
            return self.build(self.namespace, token_ids, chunk_tokens=self.chunk_tokens)
        covered = self.roots[-1].num_tokens
        checked = self.roots[-2:]
        if len(token_ids) < covered or any(
            tuple(token_ids[root.start_tokens : root.num_tokens]) != root.tail_tokens
            for root in checked
        ):
            return self.build(self.namespace, token_ids, chunk_tokens=self.chunk_tokens)
        if len(token_ids) == covered:
            return self
        keep = covered // self.chunk_tokens
        roots, end_hashes = _chain_roots(
            self.namespace,
            token_ids,
            self.chunk_tokens,
            keep * self.chunk_tokens,
            self.end_hashes[keep - 1] if keep else _chain_seed(self.chunk_tokens),
        )
        return CheckpointTokenRoots(
            self.namespace,
            self.roots[:keep] + tuple(roots),
            self.chunk_tokens,
            self.end_hashes[:keep] + tuple(end_hashes),
        )

    def prefix(self, num_tokens: int) -> CheckpointPrefix:
        """Return the exact stored key for a nonempty covered token prefix.

        Args:
            num_tokens: Exclusive token boundary within the sequence.

        Returns:
            Root plus a nonempty exact tail, including chunk-aligned endpoints.

        Raises:
            ValueError: If the requested boundary is empty or exceeds the sequence.
        """
        if num_tokens <= 0 or not self.roots or num_tokens > self.roots[-1].num_tokens:
            raise ValueError("Checkpoint boundary is outside the token sequence")
        root = self.roots[(num_tokens - 1) // self.chunk_tokens]
        return CheckpointPrefix(
            self.namespace,
            root.start_tokens,
            root.prefix_hash,
            root.tail_tokens[: num_tokens - root.start_tokens],
        )

    def content_key(self, num_tokens: int, discriminator: str) -> str:
        """Hash an exact token boundary without serializing preceding pages.

        Full hash chunks reuse the chain digest produced by :meth:`build`.
        Only a final partial chunk is packed once per call, keeping page-key
        generation linear with a small constant even for long prompts.
        """
        return self.content_keys(((num_tokens, discriminator),))[0]

    def content_keys(self, boundaries: Sequence[tuple[int, str]]) -> tuple[str, ...]:
        """Hash several page identities, reusing each token-boundary digest."""
        digests: dict[int, bytes] = {}
        keys = []
        for num_tokens, discriminator in boundaries:
            token_digest = digests.get(num_tokens)
            if token_digest is None:
                token_digest = self._token_digest(num_tokens)
                digests[num_tokens] = token_digest
            encoded = json.dumps(
                [self.namespace, num_tokens, discriminator],
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode()
            keys.append(
                hashlib.sha256(
                    b"lmcache-recurrent-page-v1\0" + token_digest + b"\0" + encoded
                ).hexdigest()
            )
        return tuple(keys)

    def _token_digest(self, num_tokens: int) -> bytes:
        if num_tokens <= 0 or not self.roots or num_tokens > self.roots[-1].num_tokens:
            raise ValueError("Checkpoint boundary is outside the token sequence")
        root_index = (num_tokens - 1) // self.chunk_tokens
        if num_tokens == self.roots[root_index].num_tokens:
            return self.end_hashes[root_index]
        prefix = self.prefix(num_tokens)
        packed = array("I", prefix.tail_tokens)
        if packed.itemsize != 4:
            raise ValueError("Checkpoint tokens require 32-bit unsigned integers")
        if sys.byteorder != "little":
            packed.byteswap()
        return hashlib.sha256(prefix.prefix_hash + packed.tobytes()).digest()
