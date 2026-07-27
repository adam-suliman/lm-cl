from __future__ import annotations

import fcntl
import json
import os
import shutil
import time
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any


OWNED_MARKER = ".lm_cl_owned_root.json"


class DiskLimitError(RuntimeError):
    pass


def directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for entry in path.rglob("*"):
        if entry.is_file() and not entry.is_symlink():
            total += entry.stat().st_size
    return total


def _safe_root(path: str | Path) -> Path:
    root = Path(path).expanduser().resolve()
    if root == Path("/") or len(root.parts) < 4:
        raise ValueError(f"Refusing unsafe owned root: {root}")
    return root


def ensure_owned_root(path: str | Path, *, purpose: str) -> Path:
    root = _safe_root(path)
    root.mkdir(parents=True, exist_ok=True)
    marker = root / OWNED_MARKER
    if marker.exists():
        data = json.loads(marker.read_text(encoding="utf-8"))
        if data.get("owner") != "lm-cl" or data.get("purpose") != purpose:
            raise ValueError(f"Owned-root marker mismatch at {marker}")
        return root
    existing = [entry for entry in root.iterdir() if entry.name != OWNED_MARKER]
    if existing:
        raise ValueError(
            f"Refusing to claim non-empty unowned directory: {root}"
        )
    atomic_write_json(
        marker,
        {
            "owner": "lm-cl",
            "purpose": purpose,
            "format_version": 1,
        },
    )
    return root


def require_owned_root(path: str | Path, *, purpose: str) -> Path:
    root = _safe_root(path)
    marker = root / OWNED_MARKER
    if not marker.is_file():
        raise ValueError(f"Missing lm-cl owned-root marker: {marker}")
    data = json.loads(marker.read_text(encoding="utf-8"))
    if data.get("owner") != "lm-cl" or data.get("purpose") != purpose:
        raise ValueError(f"Owned-root marker mismatch at {marker}")
    return root


def enforce_disk_limit(path: Path, max_bytes: int, *, label: str) -> int:
    used = directory_size(path)
    if used > max_bytes:
        raise DiskLimitError(
            f"{label} disk cap exceeded: {used} > {max_bytes} bytes"
        )
    return used


def estimate_stage_bytes(
    *,
    max_output_tokens: int,
    max_input_documents: int,
    write_boundaries: bool,
) -> dict[str, int]:
    token_bytes = max_output_tokens * 4
    boundary_bytes = max_input_documents * 320 if write_boundaries else 0
    manifest_and_registry_bytes = max(1_048_576, max_input_documents * 256)
    temporary_bytes = max(
        1_048_576,
        min(manifest_and_registry_bytes, 2 * 1024 * 1024 * 1024),
    )
    total = (
        token_bytes
        + boundary_bytes
        + manifest_and_registry_bytes
        + temporary_bytes
    )
    return {
        "packed_token_bytes": token_bytes,
        "boundary_bytes_upper_bound": boundary_bytes,
        "manifest_and_registry_bytes_upper_bound": manifest_and_registry_bytes,
        "temporary_bytes_upper_bound": temporary_bytes,
        "total_upper_bound": total,
    }


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def clean_owned_root(path: str | Path, *, purpose: str) -> dict[str, int]:
    root = require_owned_root(path, purpose=purpose)
    before = directory_size(root)
    removed = 0
    for entry in root.iterdir():
        if entry.name == OWNED_MARKER:
            continue
        if entry.is_symlink() or entry.is_file():
            removed += entry.lstat().st_size
            entry.unlink()
        elif entry.is_dir():
            removed += directory_size(entry)
            shutil.rmtree(entry)
    return {
        "bytes_before": before,
        "bytes_removed": removed,
        "bytes_after": directory_size(root),
    }


class FileLock(AbstractContextManager["FileLock"]):
    def __init__(self, path: Path):
        self.path = path
        self._handle: Any = None

    def __enter__(self) -> "FileLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a+", encoding="utf-8")
        fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc_info: object) -> None:
        assert self._handle is not None
        fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        self._handle.close()
        self._handle = None


class RuntimeGuard:
    def __init__(self, max_seconds: int):
        self.max_seconds = max_seconds
        self.started = time.monotonic()

    def check(self) -> None:
        elapsed = time.monotonic() - self.started
        if elapsed > self.max_seconds:
            raise TimeoutError(
                f"Configured runtime cap exceeded: {elapsed:.2f}s > "
                f"{self.max_seconds}s"
            )
