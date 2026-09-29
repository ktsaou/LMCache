# SPDX-License-Identifier: Apache-2.0
"""
File-system based L2 adapter using aiofiles for async I/O.

Stores KV cache objects as raw tensor bytes on disk (no metadata
header). Each ObjectKey maps to a separate ``.data`` file. Names that fit the
filesystem component limit use the flat layout. Oversized names use bounded,
reversible path components so restart inventory retains the complete key.
"""

# Future
from __future__ import annotations

# Standard
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union
import asyncio
import os
import threading
import uuid

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.distributed.internal_api import (
        L1MemoryDesc,
    )
    from lmcache.v1.memory_management import MemoryObj

# Third Party
import aiofiles
import aiofiles.os

# First Party
from lmcache.lmcache_native import Bitmap
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.internal_api import L2StoreResult
from lmcache.v1.distributed.l2_adapters.base import (
    L2AdapterInterface,
    L2TaskId,
)
from lmcache.v1.distributed.l2_adapters.config import (
    L2AdapterConfigBase,
    register_l2_adapter_type,
)
from lmcache.v1.distributed.l2_adapters.factory import (
    register_l2_adapter_factory,
)
from lmcache.v1.platform import create_event_notifier

logger = init_logger(__name__)

_KEY_SEP = "@"
# ``@`` in both ``model_name`` and ``cache_salt`` is rejected by
# ObjectKey.__post_init__, so splitting on ``@`` is unambiguous.
# Kept in sync with native_connector_l2_adapter.py and
# csrc/storage_backends/fs/connector.cpp.
_PATH_SLASH_REPLACEMENT = "-SEP-"
_FILE_EXT = ".data"
# Keep ObjectKey-to-filename mapping independent of local filesystem settings.
_LEGACY_FILENAME_MAX_BYTES = 255
_BOUNDED_PATH_VERSION = ".lmcache-objects-v1"
_ENCODED_COMPONENT_MAX_CHARS = 200


def _write_all(fd: int, buf: bytes | bytearray | memoryview) -> None:
    """Write all bytes to ``fd``, retrying short writes."""
    view = memoryview(buf)
    written = 0
    while written < len(view):
        count = os.write(fd, view[written:])
        if count <= 0:
            raise OSError("write returned 0 before the buffer was complete")
        written += count


def _publish_temp_file(tmp_path: Path, file_path: Path) -> bool:
    """Publish a complete inode without replacing an established value.

    Returns ``True`` when this writer created ``file_path`` and ``False`` when
    another writer already published the same key. The writer-owned temporary
    link is removed in both cases.
    """
    try:
        os.link(tmp_path, file_path)
        return True
    except FileExistsError:
        return False
    finally:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
        except OSError:
            logger.warning("Failed to remove temporary cache file %s", tmp_path)


def _readinto_full(
    f,  # typing: IO[bytes]
    buf: Union[bytearray, memoryview, bytes],
) -> int:
    """Loop readinto() until *buf* is full or EOF.

    A single ``readinto()`` may return fewer bytes than
    *len(buf)* even when more data is available.  This
    helper keeps reading until the buffer is completely
    filled or the file reaches EOF.

    Returns:
        Total number of bytes read.
    """
    mv = memoryview(buf) if not isinstance(buf, memoryview) else buf
    total = 0
    while total < len(mv):
        n = f.readinto(mv[total:])
        if n is None or n == 0:
            break
        total += n
    return total


async def _async_readinto_full(
    f,  # aiofiles async file handle
    buf: Union[bytearray, memoryview, bytes],
) -> int:
    """Async version of :func:`_readinto_full`."""
    mv = memoryview(buf) if not isinstance(buf, memoryview) else buf
    total = 0
    while total < len(mv):
        n = await f.readinto(mv[total:])
        if n is None or n == 0:
            break
        total += n
    return total


def _object_key_to_filename(key: ObjectKey) -> str:
    """Build a reversible, filesystem-safe filename.

    Unsalted::

        <safe_model>@0x<kv_rank_hex>@<object_group_id_hex>@<chunk_hash_hex>.data

    Salted (trailing ``cache_salt``)::

        <safe_model>@0x<kv_rank_hex>@<object_group_id_hex>@<chunk_hash_hex>@<cache_salt>.data

    ``kv_rank`` is written in ``0x`` prefixed hex so each byte
    of the bitmap ``(ws<<24)|(rank<<16)|(local_ws<<8)|local``
    is directly readable. ``object_group_id`` is written in plain hex.
    """
    safe_model = key.model_name.replace("/", _PATH_SLASH_REPLACEMENT)
    base = (
        f"{safe_model}{_KEY_SEP}{key.kv_rank:#010x}"
        f"{_KEY_SEP}{key.object_group_id:x}{_KEY_SEP}{key.chunk_hash.hex()}"
    )
    if key.cache_salt:
        return f"{base}{_KEY_SEP}{key.cache_salt}{_FILE_EXT}"
    return f"{base}{_FILE_EXT}"


def _filename_to_object_key(
    filename: str,
) -> Optional[ObjectKey]:
    """Reverse legacy filenames produced by ``_object_key_to_filename``.

    Accepts both the 4-field unsalted shape and the 5-field salted
    shape (trailing ``cache_salt``). Returns ``None`` for anything
    else. Since ``model_name`` is guaranteed not to contain ``@``,
    plain ``split`` suffices — no marker, no rsplit.
    """
    if not filename.endswith(_FILE_EXT):
        return None
    stem = filename[: -len(_FILE_EXT)]
    parts = stem.split(_KEY_SEP)
    if len(parts) == 4:
        safe_model, kv_rank_str, object_group_str, chunk_hash_hex = parts
        cache_salt = ""
    elif len(parts) == 5:
        safe_model, kv_rank_str, object_group_str, chunk_hash_hex, cache_salt = parts
    else:
        return None

    model_name = safe_model.replace(_PATH_SLASH_REPLACEMENT, "/")
    try:
        chunk_hash = bytes.fromhex(chunk_hash_hex)
        kv_rank = int(kv_rank_str, 16)
        object_group_id = int(object_group_str, 16)
        # ObjectKey.__post_init__ raises ValueError when the decoded
        # model_name / cache_salt violate the forbidden-char or length
        # invariants (e.g. a stray file from another tool on disk).
        # The contract here is to return None for anything unparsable,
        # so keep the constructor inside the try block.
        return ObjectKey(
            chunk_hash=chunk_hash,
            model_name=model_name,
            kv_rank=kv_rank,
            object_group_id=object_group_id,
            cache_salt=cache_salt,
        )
    except ValueError:
        return None


def _hex_path_components(value: str) -> list[str]:
    """Encode a key field into bounded, path-safe components."""
    encoded = value.encode("utf-8", errors="surrogatepass").hex()
    return [
        encoded[offset : offset + _ENCODED_COMPONENT_MAX_CHARS]
        for offset in range(0, len(encoded), _ENCODED_COMPONENT_MAX_CHARS)
    ]


def _ascii_path_components(value: str) -> list[str]:
    """Split a path-safe ASCII key field into bounded components."""
    return [
        value[offset : offset + _ENCODED_COMPONENT_MAX_CHARS]
        for offset in range(0, len(value), _ENCODED_COMPONENT_MAX_CHARS)
    ]


def _object_key_to_bounded_relative_path(key: ObjectKey) -> Path:
    """Map an oversized ObjectKey to a reversible bounded relative path.

    Model identity and tenant salt are preserved exactly so restart inventory
    can reconstruct capacity and per-tenant accounting without a sidecar.
    """
    model_parts = _hex_path_components(key.model_name)
    salt_parts = _hex_path_components(key.cache_salt)
    leaf = (
        f"{key.kv_rank:#010x}{_KEY_SEP}{key.object_group_id:x}"
        f"{_KEY_SEP}{key.chunk_hash.hex()}{_FILE_EXT}"
    )
    prefix = Path(
        _BOUNDED_PATH_VERSION,
        f"m{len(model_parts)}",
        *model_parts,
        f"s{len(salt_parts)}",
        *salt_parts,
    )
    if len(os.fsencode(leaf)) <= _LEGACY_FILENAME_MAX_BYTES:
        return prefix / leaf

    rank_parts = _ascii_path_components(f"{key.kv_rank:#010x}")
    group_parts = _ascii_path_components(f"{key.object_group_id:x}")
    hash_parts = _ascii_path_components(key.chunk_hash.hex())
    return Path(
        prefix,
        f"r{len(rank_parts)}",
        *rank_parts,
        f"g{len(group_parts)}",
        *group_parts,
        f"h{len(hash_parts)}",
        *hash_parts,
        f"object{_FILE_EXT}",
    )


def _object_key_to_relative_path(key: ObjectKey) -> Path:
    """Return a stable flat or bounded storage path for an ObjectKey."""
    legacy_filename = _object_key_to_filename(key)
    canonical_bytes = legacy_filename.encode("utf-8", errors="surrogatepass")
    if len(canonical_bytes) <= _LEGACY_FILENAME_MAX_BYTES:
        return Path(legacy_filename)
    return _object_key_to_bounded_relative_path(key)


def _bounded_relative_path_to_object_key(
    relative_path: Path,
) -> Optional[ObjectKey]:
    """Decode the reversible bounded path shape into its ObjectKey."""
    parts = relative_path.parts
    if len(parts) < 4 or parts[0] != _BOUNDED_PATH_VERSION:
        return None
    try:
        model_count = int(parts[1][1:]) if parts[1].startswith("m") else -1
        model_start = 2
        salt_marker = model_start + model_count
        if model_count < 0 or salt_marker >= len(parts):
            return None
        salt_header = parts[salt_marker]
        salt_count = int(salt_header[1:]) if salt_header.startswith("s") else -1
        salt_start = salt_marker + 1
        suffix_start = salt_start + salt_count
        if salt_count < 0 or suffix_start >= len(parts):
            return None

        model_name = bytes.fromhex("".join(parts[model_start:salt_marker])).decode(
            "utf-8", errors="surrogatepass"
        )
        cache_salt = bytes.fromhex("".join(parts[salt_start:suffix_start])).decode(
            "utf-8", errors="surrogatepass"
        )
        leaf = parts[-1]
        if not leaf.endswith(_FILE_EXT):
            return None
        key_parts = leaf[: -len(_FILE_EXT)].split(_KEY_SEP)
        if suffix_start == len(parts) - 1 and len(key_parts) == 3:
            kv_rank_str, object_group_str, chunk_hash_hex = key_parts
        else:
            rank_header = parts[suffix_start]
            rank_count = int(rank_header[1:]) if rank_header.startswith("r") else -1
            rank_start = suffix_start + 1
            group_marker = rank_start + rank_count
            if rank_count <= 0 or group_marker >= len(parts):
                return None
            group_header = parts[group_marker]
            group_count = int(group_header[1:]) if group_header.startswith("g") else -1
            group_start = group_marker + 1
            hash_marker = group_start + group_count
            if group_count <= 0 or hash_marker >= len(parts):
                return None
            hash_header = parts[hash_marker]
            hash_count = int(hash_header[1:]) if hash_header.startswith("h") else -1
            hash_start = hash_marker + 1
            leaf_index = hash_start + hash_count
            if (
                hash_count < 0
                or leaf_index != len(parts) - 1
                or leaf != f"object{_FILE_EXT}"
            ):
                return None
            kv_rank_str = "".join(parts[rank_start:group_marker])
            object_group_str = "".join(parts[group_start:hash_marker])
            chunk_hash_hex = "".join(parts[hash_start:leaf_index])
        return ObjectKey(
            chunk_hash=bytes.fromhex(chunk_hash_hex),
            model_name=model_name,
            kv_rank=int(kv_rank_str, 16),
            object_group_id=int(object_group_str, 16),
            cache_salt=cache_salt,
        )
    except (UnicodeDecodeError, ValueError):
        return None


class FSL2AdapterConfig(L2AdapterConfigBase):
    """
    Config for the filesystem-backed L2 adapter.

    Fields:
    - base_path: directory for storing KV cache files.
    - relative_tmp_dir: optional relative sub-dir for
      temp files (same as fs_connector_relative_tmp_dir).
    """

    def __init__(
        self,
        base_path: str,
        relative_tmp_dir: Optional[str] = None,
        read_ahead_size: Optional[int] = None,
        use_odirect: bool = False,
    ):
        """Initialize FSL2AdapterConfig.

        Args:
            base_path: Directory for storing KV cache files.
            relative_tmp_dir: Relative sub-dir under
                base_path for temp files during writes.
            read_ahead_size: If set, trigger filesystem
                readahead by issuing a small initial read
                of this many bytes before reading the rest.
            use_odirect: If True, bypass the OS page cache
                using O_DIRECT for both reads and writes.
                Requires buffer sizes aligned to the
                filesystem block size.
        """
        self.base_path = base_path
        self.relative_tmp_dir = relative_tmp_dir
        self.read_ahead_size = read_ahead_size
        self.use_odirect = use_odirect

    @classmethod
    def from_dict(cls, d: dict) -> "FSL2AdapterConfig":
        base_path = d.get("base_path")
        if not isinstance(base_path, str) or not base_path:
            raise ValueError("base_path must be a non-empty string")
        relative_tmp_dir = d.get("relative_tmp_dir", None)
        if relative_tmp_dir is not None:
            if not isinstance(relative_tmp_dir, str):
                raise ValueError("relative_tmp_dir must be a string")
        read_ahead_size = d.get("read_ahead_size", None)
        if read_ahead_size is not None:
            if not isinstance(read_ahead_size, int) or read_ahead_size <= 0:
                raise ValueError("read_ahead_size must be a positive integer")
        use_odirect = d.get("use_odirect", False)
        if not isinstance(use_odirect, bool):
            raise ValueError("use_odirect must be a boolean")
        return cls(
            base_path=base_path,
            relative_tmp_dir=relative_tmp_dir,
            read_ahead_size=read_ahead_size,
            use_odirect=use_odirect,
        )

    @classmethod
    def help(cls) -> str:
        return (
            "FS L2 adapter config fields:\n"
            "- base_path (str): directory for KV cache "
            "files (required)\n"
            "- relative_tmp_dir (str): relative sub-dir "
            "for temp files (optional, same as "
            "fs_connector_relative_tmp_dir)\n"
            "- read_ahead_size (int): trigger fs "
            "readahead by reading this many bytes first "
            "(optional)\n"
            "- use_odirect (bool): bypass page cache "
            "via O_DIRECT (optional, default false)"
        )


class FSL2Adapter(L2AdapterInterface):
    """
    File-system backed L2 adapter with async I/O via *aiofiles*.

    Each file stores **only** the raw tensor bytes (no metadata header), which
    gives maximum I/O throughput. Representable file names encode the complete
    ``ObjectKey``; oversized names use a reversible bounded path.

    Thread safety is ensured via a lock for shared bookkeeping
    and an asyncio event loop running on a dedicated daemon
    thread.
    """

    _DELETE_CONCURRENCY = 64

    def __init__(self, config: FSL2AdapterConfig):
        super().__init__()
        self._config = config
        base = config.base_path
        self._base_path = Path(base)
        self._base_path.mkdir(parents=True, exist_ok=True)

        # Temp-file strategy aligned with FSConnector:
        # if relative_tmp_dir is set, write to a sub-dir;
        # otherwise fall back to a .tmp suffix.
        self._relative_tmp_dir: Optional[Path] = None
        if config.relative_tmp_dir is not None:
            self._relative_tmp_dir = Path(config.relative_tmp_dir)
            if (
                self._relative_tmp_dir.is_absolute()
                or ".." in self._relative_tmp_dir.parts
            ):
                raise ValueError("Invalid relative_tmp_dir: " + config.relative_tmp_dir)
            (self._base_path / self._relative_tmp_dir).mkdir(
                parents=False, exist_ok=True
            )

        try:
            name_limits = [os.pathconf(self._base_path, "PC_NAME_MAX")]
            path_limits = [os.pathconf(self._base_path, "PC_PATH_MAX")]
            if self._relative_tmp_dir is not None:
                tmp_path = self._base_path / self._relative_tmp_dir
                name_limits.append(os.pathconf(tmp_path, "PC_NAME_MAX"))
                path_limits.append(os.pathconf(tmp_path, "PC_PATH_MAX"))
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                f"Failed to determine filesystem path limits for FS L2 adapter: {exc}"
            ) from exc
        finite_name_limits = [limit for limit in name_limits if limit >= 0]
        self._name_max = min(finite_name_limits, default=None)
        if self._name_max is not None and self._name_max < _LEGACY_FILENAME_MAX_BYTES:
            raise ValueError(
                "FS L2 adapter requires PC_NAME_MAX >= "
                f"{_LEGACY_FILENAME_MAX_BYTES}, got {self._name_max}"
            )
        finite_path_limits = [limit for limit in path_limits if limit >= 0]
        self._path_max = min(finite_path_limits, default=None)

        # I/O tuning options aligned with FSConnector
        self._read_ahead_size = config.read_ahead_size
        self._use_odirect = config.use_odirect
        self._os_disk_bs = 0
        if self._use_odirect:
            stat = os.statvfs(self._base_path)
            self._os_disk_bs = stat.f_bsize

        self._store_efd = create_event_notifier()
        self._lookup_efd = create_event_notifier()
        self._load_efd = create_event_notifier()

        # Task bookkeeping
        self._next_task_id: L2TaskId = 0
        self._completed_store_tasks: dict[L2TaskId, L2StoreResult] = {}
        self._completed_lookup_tasks: dict[L2TaskId, Bitmap] = {}
        self._completed_load_tasks: dict[L2TaskId, Bitmap] = {}
        self._lock = threading.Lock()

        # Background asyncio event loop
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self._loop_thread.start()

        logger.info(
            "Initialized FSL2Adapter with base_path=%s, "
            "relative_tmp_dir=%s, "
            "read_ahead_size=%s, use_odirect=%s",
            self._base_path,
            self._relative_tmp_dir,
            self._read_ahead_size,
            self._use_odirect,
        )

    # ------------------------------------------------------------------
    # Event Fd Interface
    # ------------------------------------------------------------------

    def get_store_event_fd(self) -> int:
        return self._store_efd.fileno()

    def get_lookup_and_lock_event_fd(self) -> int:
        return self._lookup_efd.fileno()

    def get_load_event_fd(self) -> int:
        return self._load_efd.fileno()

    # ------------------------------------------------------------------
    # Store Interface
    # ------------------------------------------------------------------

    def submit_store_task(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> L2TaskId:
        with self._lock:
            task_id = self._get_next_task_id()

        asyncio.run_coroutine_threadsafe(
            self._execute_store(keys, objects, task_id),
            self._loop,
        )
        return task_id

    def pop_completed_store_tasks(
        self,
    ) -> dict[L2TaskId, L2StoreResult]:
        """Pop all completed store tasks.

        Returns:
            dict[L2TaskId, L2StoreResult]: a dictionary mapping the task
            id to an ``L2StoreResult`` that encodes both the success flag
            and the bytes actually transferred.
        """
        with self._lock:
            completed = self._completed_store_tasks
            self._completed_store_tasks = {}
        return completed

    # ------------------------------------------------------------------
    # Lookup and Lock Interface
    # ------------------------------------------------------------------

    def submit_lookup_and_lock_task(
        self, keys: list[ObjectKey], group_layout_descs: dict[int, MemoryLayoutDesc]
    ) -> L2TaskId:
        with self._lock:
            task_id = self._get_next_task_id()

        asyncio.run_coroutine_threadsafe(
            self._execute_lookup(keys, task_id),
            self._loop,
        )
        return task_id

    def query_lookup_and_lock_result(self, task_id: L2TaskId) -> Bitmap | None:
        with self._lock:
            return self._completed_lookup_tasks.pop(task_id, None)

    def submit_unlock(self, keys: list[ObjectKey]) -> None:
        # No-op: the FS adapter tracks no per-key locks — a delete
        # racing the lookup→load window degrades the load to a miss.
        pass

    # ------------------------------------------------------------------
    # Load Interface
    # ------------------------------------------------------------------

    def submit_load_task(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> L2TaskId:
        with self._lock:
            task_id = self._get_next_task_id()

        asyncio.run_coroutine_threadsafe(
            self._execute_load(keys, objects, task_id),
            self._loop,
        )
        return task_id

    def query_load_result(self, task_id: L2TaskId) -> Bitmap | None:
        with self._lock:
            return self._completed_load_tasks.pop(task_id, None)

    # ------------------------------------------------------------------
    # Status Interface
    # ------------------------------------------------------------------

    def report_status(self) -> dict:
        """Return a status dict for the FS L2 adapter."""
        return {
            "is_healthy": self._loop_thread.is_alive(),
            "type": "FSL2Adapter",
            "base_path": str(self._base_path),
            "use_odirect": self._use_odirect,
            "event_loop_alive": self._loop_thread.is_alive(),
        }

    # ------------------------------------------------------------------
    # Eviction Interface
    # ------------------------------------------------------------------

    def delete(self, keys: list[ObjectKey]) -> None:
        """Delete each key's backing file and notify listeners.

        Args:
            keys: The object keys to delete.

        Note:
            No per-key locks: a delete racing a load turns that load
            into a miss; racing a store of the same key may leave the
            key re-stored.
        """
        if not keys:
            return
        try:
            fut = asyncio.run_coroutine_threadsafe(
                self._execute_delete(keys),
                self._loop,
            )
            deleted_keys, deleted_sizes = fut.result(timeout=30.0)
        except Exception as e:
            logger.warning("FSL2Adapter delete failed: %s", e)
            return
        if deleted_keys:
            self._notify_keys_deleted(deleted_keys, deleted_sizes)

    # ``get_usage()`` is inherited from ``L2AdapterInterface``. The base
    # class maintains byte totals via ``_notify_keys_stored`` /
    # ``_notify_keys_deleted``, but the FS adapter declares no max capacity
    # (default 0) so ``supports_global_eviction`` returns ``False`` and
    # ``usage_fraction == -1.0`` — the eviction controller treats this as
    # "no eviction signal" and skips the adapter entirely.

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        async def _stop_tasks():
            tasks = [
                t
                for t in asyncio.all_tasks(self._loop)
                if t is not asyncio.current_task()
            ]
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

        if self._loop.is_running():
            fut = asyncio.run_coroutine_threadsafe(_stop_tasks(), self._loop)
            try:
                fut.result(timeout=5)
            except Exception:
                pass
            self._loop.call_soon_threadsafe(self._loop.stop)

        self._loop_thread.join()
        self._loop.close()

        self._store_efd.close()
        self._lookup_efd.close()
        self._load_efd.close()
        logger.info("FSL2Adapter closed")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_event_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _get_next_task_id(self) -> L2TaskId:
        tid = self._next_task_id
        self._next_task_id += 1
        return tid

    def _key_to_path(self, key: ObjectKey) -> Path:
        path = self._base_path / _object_key_to_relative_path(key)
        self._require_representable_path(path)
        return path

    def _key_candidate_paths(self, key: ObjectKey) -> tuple[Path, ...]:
        """Return canonical-first paths accepted for one cache object.

        Oversized keys used flat filenames before bounded paths were
        introduced. Filesystems with a larger component limit can therefore
        contain a valid legacy object that must remain readable and must block
        a duplicate canonical store.
        """
        canonical = self._base_path / _object_key_to_relative_path(key)
        legacy = self._base_path / _object_key_to_filename(key)
        candidates: list[Path] = []
        if self._path_is_representable(canonical):
            candidates.append(canonical)
        if (
            canonical != legacy
            and (
                self._name_max is None
                or len(os.fsencode(legacy.name)) <= self._name_max
            )
            and self._path_is_representable(legacy)
        ):
            candidates.append(legacy)
        if not candidates:
            self._require_representable_path(canonical)
        return tuple(candidates)

    def absent_keys(self, keys: list[ObjectKey]) -> list[ObjectKey]:
        """Return keys with no object file under any accepted path.

        One ``stat`` per candidate path, so objects that other processes or
        nodes sharing the directory published count as present. Keys whose
        paths this filesystem cannot represent are not reported.
        """
        absent = []
        for key in keys:
            try:
                paths = self._key_candidate_paths(key)
            except (OSError, ValueError):
                continue
            if not any(os.path.exists(path) for path in paths):
                absent.append(key)
        return absent

    async def _existing_key_path(self, key: ObjectKey) -> Optional[Path]:
        """Resolve the first existing canonical or compatible legacy path."""
        for path in self._key_candidate_paths(key):
            if await aiofiles.os.path.exists(path):
                return path
        return None

    async def _key_exists_on_disk(
        self,
        key: ObjectKey,
    ) -> bool:
        """Check whether the file for *key* exists on disk.

        Uses ``aiofiles.os.path.exists`` so the check is
        non-blocking and always reflects the real FS state,
        which is critical for multi-node shared-FS setups.
        """
        return await self._existing_key_path(key) is not None

    def _key_to_file_and_tmp_path(self, key: ObjectKey) -> tuple[Path, Path]:
        """Return ``(final_path, tmp_path)``.

        When ``relative_tmp_dir`` is configured, the temp file
        is placed under that sub-directory (same behaviour as
        ``FSConnector._get_file_and_tmp_path``).  Otherwise a
        ``.tmp`` suffix is used.
        """
        relative_path = _object_key_to_relative_path(key)
        final = self._base_path / relative_path
        self._require_representable_path(final)
        if self._relative_tmp_dir is not None:
            tmp_dir = self._base_path / self._relative_tmp_dir
        else:
            tmp_dir = final.parent
        # The caller appends a PID and UUID. Keep the temporary basename
        # independent of the final key so a valid 255-byte legacy name cannot
        # overflow the same filesystem's component limit during publication.
        tmp = tmp_dir / ".lmcache-write.tmp"
        return final, tmp

    def _path_is_representable(self, path: Path) -> bool:
        """Return whether the encoded path fits the filesystem path limit."""
        return self._path_max is None or len(os.fsencode(path)) < self._path_max

    def _require_representable_path(self, path: Path) -> None:
        """Reject a path that cannot be passed to the target filesystem."""
        if not self._path_is_representable(path):
            path_bytes = len(os.fsencode(path))
            raise ValueError(
                f"FS L2 object path uses {path_bytes} bytes, but PC_PATH_MAX "
                f"is {self._path_max}"
            )

    # ---- O_DIRECT helpers -----------------------------------------------

    def _read_with_odirect(
        self,
        file_path: Path,
        dst_buf: Union[bytearray, memoryview, bytes],
    ) -> int:
        """Synchronous O_DIRECT read into *dst_buf*.

        Returns the number of bytes actually read.
        Runs in an executor (not on the event loop).
        """
        fd = -1
        size = len(dst_buf)
        try:
            aligned = self._os_disk_bs > 0 and size % self._os_disk_bs == 0
            if not aligned:
                logger.warning(
                    "Cannot use O_DIRECT for %s, size is not aligned.",
                    file_path,
                )
                with open(file_path, "rb") as f:
                    return _readinto_full(f, dst_buf)

            fd = os.open(
                str(file_path),
                os.O_RDONLY | getattr(os, "O_DIRECT", 0),
            )
            with os.fdopen(fd, "rb", buffering=0) as fdo:
                fd = -1  # now managed by fdopen
                return _readinto_full(fdo, dst_buf)
        except Exception:
            logger.exception("Failed to O_DIRECT read %s", file_path)
            return 0
        finally:
            if fd >= 0:
                try:
                    os.close(fd)
                except OSError:
                    pass

    def _write_with_odirect(
        self, file_path: Path, buf: Union[bytearray, memoryview, bytes]
    ) -> None:
        """Synchronous O_DIRECT write of *buf*.

        Runs in an executor (not on the event loop).
        """
        fd = -1
        try:
            fd = os.open(
                str(file_path),
                os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_DIRECT", 0),
                0o644,
            )
            _write_all(fd, buf)
            os.close(fd)
            fd = -1
        except Exception:
            logger.exception("Failed to O_DIRECT write %s", file_path)
            raise
        finally:
            if fd >= 0:
                try:
                    os.close(fd)
                except OSError:
                    pass

    # ---- store ----------------------------------------------------------

    async def _execute_store(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
        task_id: L2TaskId,
    ) -> None:
        success = True
        bytes_written = 0
        stored_keys: list[ObjectKey] = []
        stored_sizes: list[int] = []
        try:
            for key, obj in zip(keys, objects, strict=True):
                # Skip if already stored on disk
                if await self._existing_key_path(key) is not None:
                    continue
                file_path, tmp_template = self._key_to_file_and_tmp_path(key)
                buf = obj.byte_array
                size = len(buf)
                tmp_path = tmp_template.with_name(
                    f"{tmp_template.name}.{os.getpid()}.{uuid.uuid4().hex}"
                )
                self._require_representable_path(tmp_path)
                file_path.parent.mkdir(parents=True, exist_ok=True)

                try:
                    # Decide whether O_DIRECT is usable
                    do_odirect = self._use_odirect
                    if do_odirect:
                        aligned = self._os_disk_bs > 0 and size % self._os_disk_bs == 0
                        if not aligned:
                            logger.warning(
                                "Cannot use O_DIRECT for "
                                "writing size %d, not "
                                "aligned to block size "
                                "%d.",
                                size,
                                self._os_disk_bs,
                            )
                            do_odirect = False

                    if do_odirect:
                        await self._loop.run_in_executor(
                            None,
                            self._write_with_odirect,
                            tmp_path,
                            buf,
                        )
                    else:
                        async with aiofiles.open(tmp_path, "xb") as f:
                            await f.write(buf)

                    published = await self._loop.run_in_executor(
                        None, _publish_temp_file, tmp_path, file_path
                    )
                    if published:
                        bytes_written += size
                        stored_keys.append(key)
                        stored_sizes.append(size)
                        logger.debug(
                            "FSL2Adapter stored key %s (%d bytes)",
                            file_path.name,
                            size,
                        )
                    else:
                        logger.debug(
                            "FSL2Adapter duplicate store kept existing key %s",
                            file_path.name,
                        )
                except Exception:
                    logger.exception(
                        "FSL2Adapter failed to store %s",
                        file_path,
                    )
                    if await aiofiles.os.path.exists(tmp_path):
                        await aiofiles.os.unlink(tmp_path)
                    success = False
        except Exception:
            logger.exception(
                "FSL2Adapter store task %s failed",
                task_id,
            )
            success = False

        if stored_keys:
            self._notify_keys_stored(stored_keys, stored_sizes)

        with self._lock:
            self._completed_store_tasks[task_id] = L2StoreResult(success, bytes_written)
        self._store_efd.notify()

    # ---- lookup ---------------------------------------------------------

    async def _execute_lookup(
        self,
        keys: list[ObjectKey],
        task_id: L2TaskId,
    ) -> None:
        bitmap = Bitmap(len(keys))
        for i, key in enumerate(keys):
            try:
                key_exists = await self._key_exists_on_disk(key)
            except ValueError:
                logger.exception("FSL2Adapter rejected unrepresentable key %s", key)
                continue
            if not key_exists:
                continue
            bitmap.set(i)

        with self._lock:
            self._completed_lookup_tasks[task_id] = bitmap
        self._lookup_efd.notify()

    # ---- load -----------------------------------------------------------

    async def _execute_load(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
        task_id: L2TaskId,
    ) -> None:
        bitmap = Bitmap(len(keys))
        for i, key in enumerate(keys):
            try:
                file_path = await self._existing_key_path(key) or self._key_to_path(key)
                dst_buf = objects[i].byte_array
                expected = len(dst_buf)
                num_read: Optional[int] = None

                # O_DIRECT path (sync, via executor)
                if self._use_odirect:
                    num_read = await self._loop.run_in_executor(
                        None,
                        self._read_with_odirect,
                        file_path,
                        dst_buf,
                    )
                    if num_read != expected:
                        logger.warning(
                            "Incomplete O_DIRECT read for %s: expected %d, got %d",
                            file_path.name,
                            expected,
                            num_read or 0,
                        )
                    else:
                        bitmap.set(i)
                        logger.debug(
                            "FSL2Adapter loaded key %s (%d bytes, O_DIRECT)",
                            file_path.name,
                            num_read,
                        )
                    continue

                # Standard async path with optional
                # read-ahead
                expected = len(dst_buf)
                async with aiofiles.open(file_path, "rb") as f:
                    if self._read_ahead_size is None:
                        num_read = await _async_readinto_full(f, dst_buf)
                    else:
                        if not isinstance(dst_buf, memoryview):
                            dst_buf = memoryview(dst_buf)
                        # Trigger readahead with a
                        # small initial read
                        ra = self._read_ahead_size
                        n_head = await _async_readinto_full(f, dst_buf[:ra])
                        if n_head == ra:
                            n_tail = await _async_readinto_full(f, dst_buf[ra:])
                            num_read = n_head + n_tail
                        else:
                            num_read = n_head

                    if num_read != expected:
                        logger.warning(
                            "Incomplete read for %s: expected %d, got %d",
                            file_path.name,
                            expected,
                            num_read,
                        )
                        continue

                    bitmap.set(i)
                    logger.debug(
                        "FSL2Adapter loaded key %s (%d bytes)",
                        file_path.name,
                        num_read,
                    )
            except FileNotFoundError:
                continue
            except Exception:
                logger.exception(
                    "FSL2Adapter failed to load key %s",
                    key,
                )
                continue

        loaded_keys = [keys[i] for i in bitmap.get_indices_list()]
        if loaded_keys:
            self._notify_keys_accessed(loaded_keys)

        with self._lock:
            self._completed_load_tasks[task_id] = bitmap
        self._load_efd.notify()

    # ---- delete ---------------------------------------------------------

    async def _execute_delete(
        self, keys: list[ObjectKey]
    ) -> tuple[list[ObjectKey], list[int]]:
        """Unlink each key's file concurrently; return removed keys + sizes."""
        sem = asyncio.Semaphore(self._DELETE_CONCURRENCY)

        async def _delete_one(key: ObjectKey) -> tuple[ObjectKey, int] | None:
            try:
                file_path = await self._existing_key_path(key)
            except ValueError:
                logger.exception("FSL2Adapter rejected unrepresentable key %s", key)
                return None
            if file_path is None:
                return None
            async with sem:
                try:
                    size = (await aiofiles.os.stat(file_path)).st_size
                    await aiofiles.os.unlink(file_path)
                except FileNotFoundError:
                    return None
                except Exception:
                    logger.exception(
                        "FSL2Adapter failed to delete %s",
                        file_path,
                    )
                    return None
            return key, size

        results = await asyncio.gather(*(_delete_one(k) for k in keys))

        deleted_keys: list[ObjectKey] = []
        deleted_sizes: list[int] = []
        for res in results:
            if res is not None:
                key, size = res
                deleted_keys.append(key)
                deleted_sizes.append(size)
        return deleted_keys, deleted_sizes


# Self-register config type and adapter factory
register_l2_adapter_type("fs", FSL2AdapterConfig)


def _create_fs_adapter(
    config: L2AdapterConfigBase,
    l1_memory_desc: "Optional[L1MemoryDesc]" = None,
) -> L2AdapterInterface:
    """Create an FSL2Adapter from config."""
    return FSL2Adapter(config)  # type: ignore[arg-type]


register_l2_adapter_factory("fs", _create_fs_adapter)
