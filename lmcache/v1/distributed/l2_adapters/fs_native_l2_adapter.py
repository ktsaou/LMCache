# SPDX-License-Identifier: Apache-2.0
"""
Filesystem native L2 adapter config and factory.

Backed by the native C++ filesystem connector wrapped with
``NativeConnectorL2Adapter``.
"""

# Future
from __future__ import annotations

# Standard
from pathlib import Path
from typing import TYPE_CHECKING, Optional
import os
import stat

if TYPE_CHECKING:
    from lmcache.v1.distributed.internal_api import (
        L1MemoryDesc,
    )

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.l2_adapters.base import (
    L2AdapterInterface,
)
from lmcache.v1.distributed.l2_adapters.config import (
    L2AdapterConfigBase,
    register_l2_adapter_type,
)
from lmcache.v1.distributed.l2_adapters.factory import (
    register_l2_adapter_factory,
)
from lmcache.v1.distributed.l2_adapters.fs_l2_adapter import (
    _BOUNDED_PATH_VERSION,
    _bounded_relative_path_to_object_key,
    _filename_to_object_key,
    _object_key_to_filename,
    _object_key_to_relative_path,
)

logger = init_logger(__name__)

_FILE_EXT = ".data"
_IGNORED_FILE_SAMPLE_LIMIT = 5


def _scan_existing_key_sizes(base_path: str) -> dict[ObjectKey, int]:
    """Inventory complete native-FS objects before the client starts.

    Regular, positive-size ``.data`` objects in either the legacy flat layout
    or the bounded reversible layout are counted. Entries are returned
    oldest-to-newest by ``(mtime_ns, path)`` so an LRU policy can reconstruct a
    deterministic best-effort order. The cache directory is expected to be
    quiescent (or exclusively owned by this server) during startup: a file that
    disappears during the scan is skipped, while every other I/O error fails
    construction instead of silently undercounting capacity.
    """
    records: list[tuple[int, str, ObjectKey, int]] = []
    ignored_data_files: list[str] = []
    ignored_data_count = 0

    try:
        entries = os.scandir(base_path)
    except FileNotFoundError:
        return {}
    except OSError as exc:
        raise RuntimeError(
            f"Failed to scan native FS cache directory {base_path!r}: {exc}"
        ) from exc

    try:
        with entries:
            for entry in entries:
                try:
                    if not entry.is_file(follow_symlinks=False):
                        continue
                except FileNotFoundError:
                    continue
                except OSError as exc:
                    raise RuntimeError(
                        f"Failed to inspect native FS cache entry {entry.path!r}: {exc}"
                    ) from exc

                key = _filename_to_object_key(entry.name)
                if key is None:
                    if entry.name.endswith(_FILE_EXT):
                        ignored_data_count += 1
                        if len(ignored_data_files) < _IGNORED_FILE_SAMPLE_LIMIT:
                            ignored_data_files.append(entry.name)
                    continue

                try:
                    entry_stat = entry.stat(follow_symlinks=False)
                except FileNotFoundError:
                    # A concurrent cleanup can remove an entry between listing
                    # and stat. No controller is connected to this adapter yet.
                    continue
                except OSError as exc:
                    raise RuntimeError(
                        f"Failed to stat native FS cache entry {entry.path!r}: {exc}"
                    ) from exc

                if entry_stat.st_size <= 0:
                    ignored_data_count += 1
                    if len(ignored_data_files) < _IGNORED_FILE_SAMPLE_LIMIT:
                        ignored_data_files.append(entry.name)
                    continue
                records.append(
                    (entry_stat.st_mtime_ns, entry.name, key, entry_stat.st_size)
                )
    except RuntimeError:
        raise
    except OSError as exc:
        raise RuntimeError(
            f"Failed while scanning native FS cache directory {base_path!r}: {exc}"
        ) from exc

    bounded_root = os.path.join(base_path, _BOUNDED_PATH_VERSION)
    try:
        bounded_stat = os.stat(bounded_root, follow_symlinks=False)
    except FileNotFoundError:
        bounded_root_exists = False
    except OSError as exc:
        raise RuntimeError(
            f"Failed to inspect native FS bounded cache root {bounded_root!r}: {exc}"
        ) from exc
    else:
        if not stat.S_ISDIR(bounded_stat.st_mode):
            raise RuntimeError(
                f"Native FS bounded cache root is not a directory: {bounded_root!r}"
            )
        bounded_root_exists = True

    if bounded_root_exists:

        def _raise_walk_error(exc: OSError) -> None:
            raise RuntimeError(
                f"Failed while scanning native FS cache directory {base_path!r}: {exc}"
            ) from exc

        for root, directories, filenames in os.walk(
            bounded_root,
            topdown=True,
            onerror=_raise_walk_error,
            followlinks=False,
        ):
            directories[:] = [
                name
                for name in directories
                if not os.path.islink(os.path.join(root, name))
            ]
            for filename in filenames:
                if not filename.endswith(_FILE_EXT):
                    continue
                path = os.path.join(root, filename)
                try:
                    file_stat = os.lstat(path)
                except FileNotFoundError:
                    continue
                except OSError as exc:
                    raise RuntimeError(
                        f"Failed to stat native FS cache entry {path!r}: {exc}"
                    ) from exc
                if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_size <= 0:
                    ignored_data_count += 1
                    if len(ignored_data_files) < _IGNORED_FILE_SAMPLE_LIMIT:
                        ignored_data_files.append(path)
                    continue
                relative_path = Path(os.path.relpath(path, base_path))
                key = _bounded_relative_path_to_object_key(relative_path)
                if key is None:
                    ignored_data_count += 1
                    if len(ignored_data_files) < _IGNORED_FILE_SAMPLE_LIMIT:
                        ignored_data_files.append(str(relative_path))
                    continue
                records.append(
                    (
                        file_stat.st_mtime_ns,
                        relative_path.as_posix(),
                        key,
                        file_stat.st_size,
                    )
                )

    if ignored_data_count:
        logger.warning(
            "Ignored %d unrecognized or empty native FS .data files during "
            "restart inventory (sample=%s)",
            ignored_data_count,
            ignored_data_files,
        )

    records.sort(key=lambda record: (record[0], record[1]))
    key_sizes: dict[ObjectKey, int] = {}
    for _mtime_ns, filename, key, size in records:
        if key in key_sizes:
            raise RuntimeError(
                "Native FS restart inventory found multiple filenames for the "
                f"same ObjectKey (latest={filename!r})"
            )
        key_sizes[key] = size
    return key_sizes


class FSNativeL2AdapterConfig(L2AdapterConfigBase):
    """
    Config for an L2 adapter backed by the native C++
    filesystem connector.

    Fields:
    - base_path: directory for storing KV cache files.
    - num_workers: C++ worker threads for I/O (default 4).
    - relative_tmp_dir: relative sub-dir for temp files.
    - use_odirect: bypass page cache via O_DIRECT.
    - read_ahead_size: trigger filesystem readahead by
      reading this many bytes first (optional).
    """

    def __init__(
        self,
        base_path: str,
        num_workers: int = 4,
        relative_tmp_dir: str = "",
        use_odirect: bool = False,
        read_ahead_size: Optional[int] = None,
        max_capacity_gb: float = 0,
    ):
        self.base_path = base_path
        self.num_workers = num_workers
        self.relative_tmp_dir = relative_tmp_dir
        self.use_odirect = use_odirect
        self.read_ahead_size = read_ahead_size
        self.max_capacity_gb = max_capacity_gb

    @classmethod
    def from_dict(cls, d: dict) -> "FSNativeL2AdapterConfig":
        base_path = d.get("base_path")
        if not isinstance(base_path, str) or not base_path:
            raise ValueError("base_path must be a non-empty string")

        num_workers = d.get("num_workers", 4)
        if not isinstance(num_workers, int) or num_workers <= 0:
            raise ValueError("num_workers must be a positive integer")

        relative_tmp_dir = d.get("relative_tmp_dir", "")
        if not isinstance(relative_tmp_dir, str):
            raise ValueError("relative_tmp_dir must be a string")

        use_odirect = d.get("use_odirect", False)
        if not isinstance(use_odirect, bool):
            raise ValueError("use_odirect must be a boolean")

        read_ahead_size = d.get("read_ahead_size", None)
        if read_ahead_size is not None:
            if not isinstance(read_ahead_size, int) or read_ahead_size <= 0:
                raise ValueError("read_ahead_size must be a positive integer")

        max_capacity_gb = d.get("max_capacity_gb", 0)
        if not isinstance(max_capacity_gb, (int, float)) or max_capacity_gb < 0:
            raise ValueError("max_capacity_gb must be a non-negative number")

        return cls(
            base_path=base_path,
            num_workers=num_workers,
            relative_tmp_dir=str(relative_tmp_dir),
            use_odirect=use_odirect,
            read_ahead_size=read_ahead_size,
            max_capacity_gb=float(max_capacity_gb),
        )

    @classmethod
    def help(cls) -> str:
        return (
            "FS native L2 adapter config fields:\n"
            "- base_path (str): directory for KV "
            "cache files (required)\n"
            "- num_workers (int): C++ worker threads "
            "for I/O (default 4, >0)\n"
            "- relative_tmp_dir (str): relative "
            "sub-dir for temp files (default empty)\n"
            "- use_odirect (bool): bypass page cache "
            "via O_DIRECT (default false)\n"
            "- read_ahead_size (int): trigger fs "
            "readahead by reading this many bytes "
            "first (optional)\n"
            "- max_capacity_gb (float): max L2 capacity "
            "in GB for usage tracking / eviction "
            "(default 0 = disabled)"
        )


def _absent_native_fs_keys(base_path: str, keys: list[ObjectKey]) -> list[ObjectKey]:
    """Return keys with neither a canonical nor a legacy object file.

    One ``stat`` per path; a file another process published counts. Keys
    whose paths cannot be derived are not reported.
    """
    base = Path(base_path)
    absent = []
    for key in keys:
        try:
            paths = (
                base / _object_key_to_relative_path(key),
                base / _object_key_to_filename(key),
            )
        except ValueError:
            continue
        if not any(os.path.exists(path) for path in paths):
            absent.append(key)
    return absent


def _create_fs_native_l2_adapter(
    config: L2AdapterConfigBase,
    l1_memory_desc: "Optional[L1MemoryDesc]" = None,
) -> L2AdapterInterface:
    """Create a NativeConnectorL2Adapter backed by the
    C++ filesystem connector."""
    try:
        # First Party
        from lmcache.lmcache_fs import (
            LMCacheFSClient,
        )
    except ImportError as e:
        raise RuntimeError(
            "FS native L2 adapter requires the C++ FS "
            "extension. Build with: pip install -e ."
        ) from e

    # Lazy import to avoid circular dependency
    # First Party
    from lmcache.v1.distributed.l2_adapters.native_connector_l2_adapter import (  # noqa: E501
        NativeConnectorL2Adapter,
    )

    assert isinstance(config, FSNativeL2AdapterConfig)
    initial_key_sizes = _scan_existing_key_sizes(config.base_path)
    native_client = LMCacheFSClient(
        config.base_path,
        config.num_workers,
        config.relative_tmp_dir,
        config.use_odirect,
        config.read_ahead_size or 0,
    )
    try:
        adapter = NativeConnectorL2Adapter(
            native_client,
            max_capacity_gb=config.max_capacity_gb,
            type_name="FSNativeL2Adapter",
            extra_status={
                "base_path": config.base_path,
                "use_odirect": config.use_odirect,
                "num_workers": config.num_workers,
                "read_ahead_size": config.read_ahead_size,
            },
            initial_key_sizes=initial_key_sizes,
            absence_probe=lambda keys: _absent_native_fs_keys(config.base_path, keys),
        )
    except Exception:
        native_client.close()
        raise
    logger.info(
        "Created FS native L2 adapter: %s (workers=%d, odirect=%s, "
        "read_ahead=%s, restored_keys=%d, restored_bytes=%d)",
        config.base_path,
        config.num_workers,
        config.use_odirect,
        config.read_ahead_size,
        len(initial_key_sizes),
        sum(initial_key_sizes.values()),
    )
    return adapter


register_l2_adapter_type("fs_native", FSNativeL2AdapterConfig)
register_l2_adapter_factory("fs_native", _create_fs_native_l2_adapter)
