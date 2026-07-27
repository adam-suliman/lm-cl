from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterable, Iterator, Mapping
from itertools import islice
from typing import Any

from lm_cl.config.data_schema import SelectionConfig
from lm_cl.data.types import SourceDocument


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_source_id(
    row: Mapping[str, Any],
    *,
    id_field: str | None,
    content_sha256: str,
) -> str:
    if (
        id_field is None
        or id_field not in row
        or row[id_field] is None
        or row[id_field] == ""
    ):
        return f"content-sha256:{content_sha256}"
    value = row[id_field]
    try:
        canonical = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
    except (TypeError, ValueError):
        return f"content-sha256:{content_sha256}"
    return f"source-id-sha256:{sha256_text(canonical)}"


def rows_to_documents(
    rows: Iterable[Mapping[str, Any]],
    *,
    text_field: str,
    id_field: str | None,
    rejection_counts: dict[str, int],
    counters: dict[str, int] | None = None,
) -> Iterator[SourceDocument]:
    for row in rows:
        if counters is not None:
            counters["input_documents_seen"] = (
                counters.get("input_documents_seen", 0) + 1
            )
        if not isinstance(row, Mapping):
            rejection_counts["row_not_mapping"] = (
                rejection_counts.get("row_not_mapping", 0) + 1
            )
            continue
        text = row.get(text_field)
        if not isinstance(text, str):
            rejection_counts["missing_or_non_string_text"] = (
                rejection_counts.get("missing_or_non_string_text", 0) + 1
            )
            continue
        if not text:
            rejection_counts["empty_text"] = (
                rejection_counts.get("empty_text", 0) + 1
            )
            continue
        digest = sha256_text(text)
        yield SourceDocument(
            source_id=stable_source_id(
                row, id_field=id_field, content_sha256=digest
            ),
            text=text,
            content_sha256=digest,
        )


def deterministic_buffered_order(
    documents: Iterable[SourceDocument],
    *,
    seed: int,
    buffer_size: int,
) -> Iterator[SourceDocument]:
    """Bounded-memory deterministic shuffle suitable for iterable datasets."""
    if buffer_size <= 0:
        raise ValueError("buffer_size must be positive")
    rng = random.Random(seed)
    iterator = iter(documents)
    buffer = list(islice(iterator, buffer_size))
    for document in iterator:
        index = rng.randrange(len(buffer))
        yield buffer[index]
        buffer[index] = document
    rng.shuffle(buffer)
    yield from buffer


def document_split(
    content_sha256: str,
    *,
    split_seed: int,
    validation_permyriad: int,
) -> str:
    digest = hashlib.sha256(
        f"{split_seed}:{content_sha256}".encode("ascii")
    ).digest()
    bucket = int.from_bytes(digest[:8], "big") % 10_000
    return "validation" if bucket < validation_permyriad else "train"


def purpose_split(purpose: str) -> str | None:
    if purpose in {"language_validation", "vietnamese_validation"}:
        return "validation"
    if purpose in {"continual_train", "vietnamese_train"}:
        return "train"
    return None


def select_documents(
    rows: Iterable[Mapping[str, Any]],
    *,
    text_field: str,
    id_field: str | None,
    purpose: str,
    config: SelectionConfig,
    rejection_counts: dict[str, int],
    counters: dict[str, int] | None = None,
) -> Iterator[tuple[SourceDocument, str]]:
    """Apply the source cap, stable order, and document split before tokenization."""
    capped_rows = islice(rows, config.max_input_documents)
    documents = rows_to_documents(
        capped_rows,
        text_field=text_field,
        id_field=id_field,
        rejection_counts=rejection_counts,
        counters=counters,
    )
    ordered = deterministic_buffered_order(
        documents,
        seed=config.document_order_seed,
        buffer_size=config.shuffle_buffer_documents,
    )
    required_split = purpose_split(purpose)
    for document in ordered:
        split = document_split(
            document.content_sha256,
            split_seed=config.split_seed,
            validation_permyriad=config.validation_permyriad,
        )
        if required_split is not None and split != required_split:
            reason = f"assigned_to_{split}"
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
            continue
        yield document, split
