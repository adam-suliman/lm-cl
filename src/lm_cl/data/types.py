from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Protocol

import numpy as np


@dataclass(frozen=True)
class TokenPosition:
    shard_index: int
    token_offset: int

    def validate(self) -> None:
        if self.shard_index < 0 or self.token_offset < 0:
            raise ValueError("Token position components must be non-negative")


@dataclass(frozen=True)
class TokenBatch:
    input_ids: np.ndarray
    labels: np.ndarray
    valid_target_count: int
    start_position: TokenPosition
    next_position: TokenPosition
    global_sequence_count: int | None = None
    local_slice_start: int | None = None
    local_slice_end: int | None = None
    global_sequence_start: int | None = None
    global_sequence_end: int | None = None


class TokenBatchSource(Protocol):
    def iter_batches(
        self,
        *,
        sequence_length: int,
        global_sequences_per_batch: int,
        start: TokenPosition | None = None,
        sequence_prefix_count: int | None = None,
    ) -> Iterator[TokenBatch]: ...


class DocumentTokenizer(Protocol):
    eos_token_id: int | None

    def encode(
        self, text: str, *, add_special_tokens: bool = False
    ) -> list[int]: ...


def normalize_token_ids(values: object) -> list[int]:
    if values is None:
        raise TypeError("Tokenizer returned null token IDs")
    result: list[int] = []
    for value in values:  # type: ignore[union-attr]
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError("Tokenizer IDs must be integers")
        result.append(int(value))
    return result


@dataclass(frozen=True)
class SourceDocument:
    source_id: str
    text: str
    content_sha256: str


@dataclass(frozen=True)
class DocumentBoundary:
    document_index: int
    source_id: str
    content_sha256: str
    token_start: int
    content_token_count: int
    token_end: int
    eos_after: bool
    truncated: bool
    split: str
