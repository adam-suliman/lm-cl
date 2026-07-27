from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any


REGISTRY_SCHEMA_VERSION = 2


def _memory_setting(name: str, default_mib: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default_mib
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer MiB count") from exc
    if value <= 0 or value > 262_144:
        raise ValueError(f"{name} must be in [1, 262144] MiB")
    return value


class OverlapError(RuntimeError):
    pass


class OverlapRegistry:
    """Disk-backed global document registry. Raw document text is never stored."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        cache_mib = _memory_setting("LM_CL_REGISTRY_CACHE_MIB", 512)
        mmap_mib = _memory_setting("LM_CL_REGISTRY_MMAP_MIB", 4096)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute("PRAGMA temp_store=MEMORY")
        self.connection.execute(f"PRAGMA cache_size=-{cache_mib * 1024}")
        self.connection.execute(f"PRAGMA mmap_size={mmap_mib * 1024 * 1024}")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                content_sha256 TEXT PRIMARY KEY,
                token_ids_sha256 TEXT NOT NULL UNIQUE,
                stage_id TEXT NOT NULL,
                purpose TEXT NOT NULL,
                language TEXT NOT NULL,
                split_name TEXT NOT NULL,
                source_id TEXT NOT NULL,
                document_index INTEGER NOT NULL,
                token_start INTEGER NOT NULL,
                token_end INTEGER NOT NULL
            )
            """
        )
        self.connection.execute(
            "INSERT OR IGNORE INTO metadata(key, value) VALUES (?, ?)",
            ("schema_version", str(REGISTRY_SCHEMA_VERSION)),
        )
        self.connection.commit()
        version = self.connection.execute(
            "SELECT value FROM metadata WHERE key='schema_version'"
        ).fetchone()
        if version != (str(REGISTRY_SCHEMA_VERSION),):
            raise ValueError("Unknown overlap-registry schema version")

    def close(self) -> None:
        self.connection.close()

    def commit(self) -> None:
        """Durably commit the current ordered insertion batch."""
        self.connection.commit()

    def __enter__(self) -> "OverlapRegistry":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def lookup(self, content_sha256: str) -> dict[str, Any] | None:
        cursor = self.connection.execute(
            """
            SELECT content_sha256, token_ids_sha256, stage_id, purpose, language, split_name,
                   source_id, document_index, token_start, token_end
            FROM documents WHERE content_sha256=?
            """,
            (content_sha256,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        names = (
            "content_sha256",
            "token_ids_sha256",
            "stage_id",
            "purpose",
            "language",
            "split",
            "source_id",
            "document_index",
            "token_start",
            "token_end",
        )
        return dict(zip(names, row, strict=True))

    def register(
        self,
        *,
        content_sha256: str,
        token_ids_sha256: str,
        stage_id: str,
        purpose: str,
        language: str,
        split: str,
        source_id: str,
        document_index: int,
        token_start: int,
        token_end: int,
    ) -> None:
        existing = self.lookup(content_sha256)
        if existing is not None:
            if existing["stage_id"] == stage_id:
                return
            raise OverlapError(
                "Document overlap with stage "
                f"{existing['stage_id']}: {content_sha256}"
            )
        token_existing = self.lookup_token_ids(token_ids_sha256)
        if token_existing is not None:
            if token_existing["stage_id"] == stage_id:
                return
            raise OverlapError(
                "Tokenized-document overlap with stage "
                f"{token_existing['stage_id']}: {token_ids_sha256}"
            )
        try:
            with self.connection:
                self.connection.execute(
                    """
                    INSERT INTO documents(
                        content_sha256, token_ids_sha256, stage_id, purpose, language, split_name,
                        source_id, document_index, token_start, token_end
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        content_sha256,
                        token_ids_sha256,
                        stage_id,
                        purpose,
                        language,
                        split,
                        source_id,
                        document_index,
                        token_start,
                        token_end,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise OverlapError(
                f"Document already registered: {content_sha256}"
            ) from exc

    def insert_prechecked(
        self,
        *,
        content_sha256: str,
        token_ids_sha256: str,
        stage_id: str,
        purpose: str,
        language: str,
        split: str,
        source_id: str,
        document_index: int,
        token_start: int,
        token_end: int,
    ) -> None:
        """Insert into the current transaction after both keys were checked.

        Materialization owns the global overlap lock and processes documents in
        one canonical order.  Delaying the commit until the existing stage
        checkpoint boundary removes one synchronous filesystem transaction per
        document without changing lookup visibility or accepted-document order.
        """
        try:
            self.connection.execute(
                """
                INSERT INTO documents(
                    content_sha256, token_ids_sha256, stage_id, purpose,
                    language, split_name, source_id, document_index,
                    token_start, token_end
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    content_sha256,
                    token_ids_sha256,
                    stage_id,
                    purpose,
                    language,
                    split,
                    source_id,
                    document_index,
                    token_start,
                    token_end,
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise OverlapError(
                f"Document already registered: {content_sha256}"
            ) from exc

    def lookup_token_ids(self, token_ids_sha256: str) -> dict[str, Any] | None:
        cursor = self.connection.execute(
            """
            SELECT content_sha256, token_ids_sha256, stage_id, purpose,
                   language, split_name, source_id, document_index,
                   token_start, token_end
            FROM documents WHERE token_ids_sha256=?
            """,
            (token_ids_sha256,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        names = (
            "content_sha256",
            "token_ids_sha256",
            "stage_id",
            "purpose",
            "language",
            "split",
            "source_id",
            "document_index",
            "token_start",
            "token_end",
        )
        return dict(zip(names, row, strict=True))

    def hashes_for_stage(self, stage_id: str) -> set[str]:
        rows = self.connection.execute(
            "SELECT content_sha256 FROM documents WHERE stage_id=?",
            (stage_id,),
        )
        return {row[0] for row in rows}

    def count_for_stage(self, stage_id: str) -> int:
        return int(
            self.connection.execute(
                "SELECT COUNT(*) FROM documents WHERE stage_id=?",
                (stage_id,),
            ).fetchone()[0]
        )

    def max_document_index_for_stage(self, stage_id: str) -> int | None:
        value = self.connection.execute(
            "SELECT MAX(document_index) FROM documents WHERE stage_id=?",
            (stage_id,),
        ).fetchone()[0]
        return None if value is None else int(value)

    def truncate_stage(self, stage_id: str, *, keep_document_count: int) -> None:
        if keep_document_count < 0:
            raise ValueError("keep_document_count must be non-negative")
        with self.connection:
            self.connection.execute(
                """
                DELETE FROM documents
                WHERE stage_id=? AND document_index>=?
                """,
                (stage_id, keep_document_count),
            )

    def count(self) -> int:
        return int(
            self.connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        )
