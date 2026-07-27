from __future__ import annotations

import hashlib
import json
import os
import re
from bisect import bisect_right
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from lm_cl.config.data_schema import PACKED_FORMAT_VERSION
from lm_cl.data.tokenizer import sha256_file
from lm_cl.data.types import TokenBatch, TokenPosition


UINT32_LE = np.dtype("<u4")


def manifest_content_hash(manifest: dict[str, Any]) -> str:
    content = {
        key: value
        for key, value in manifest.items()
        if key != "manifest_content_sha256"
    }
    return hashlib.sha256(
        json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def ordered_data_hash(manifest: dict[str, Any]) -> str:
    ordered_data = {
        "dataset": manifest["dataset"],
        "stage": {
            key: manifest["stage"][key]
            for key in (
                "stage_id",
                "purpose",
                "language",
                "task_index",
                "cycle_index",
                "require_exact_output_tokens",
            )
        },
        "tokenizer_content_sha256": manifest["tokenizer"][
            "tokenizer_content_sha256"
        ],
        "selection": {
            key: value
            for key, value in manifest["selection"].items()
            if key != "max_runtime_seconds"
        },
        "packing": manifest["packing"],
        "reader": manifest["reader"],
        "accepted_documents": manifest["accepted_documents"],
        "token_count": manifest["token_count"],
        "target_token_count": manifest["target_token_count"],
        "shards": manifest["shards"],
        "boundaries": manifest["boundaries"],
    }
    return hashlib.sha256(
        json.dumps(
            ordered_data, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


class PackedShardWriter:
    def __init__(
        self,
        stage_dir: Path,
        *,
        max_shard_tokens: int,
        checkpoint_shards: Sequence[dict[str, Any]] | None = None,
    ):
        self.stage_dir = stage_dir
        self.max_shard_tokens = max_shard_tokens
        self.stage_dir.mkdir(parents=True, exist_ok=True)
        self.shards: list[dict[str, Any]] = [
            {
                "filename": item["filename"],
                "token_count": int(item["token_count"]),
            }
            for item in (checkpoint_shards or [])
        ]
        for index, shard in enumerate(self.shards):
            if shard["filename"] != f"tokens-{index:05d}.bin":
                raise ValueError("Resume shard filename/order is not canonical")
            if (
                shard["token_count"] < 0
                or shard["token_count"] > self.max_shard_tokens
            ):
                raise ValueError("Resume shard token count is invalid")
            if (
                index + 1 < len(self.shards)
                and shard["token_count"] != self.max_shard_tokens
            ):
                raise ValueError("Non-final resume shard must be full")
        self._handle: Any = None
        self._recover_files()

    @property
    def total_tokens(self) -> int:
        return sum(int(shard["token_count"]) for shard in self.shards)

    def _recover_files(self) -> None:
        expected_names = {item["filename"] for item in self.shards}
        for path in self.stage_dir.glob("tokens-*.bin"):
            if path.name not in expected_names:
                path.unlink()
        for shard in self.shards:
            path = self.stage_dir / shard["filename"]
            expected_bytes = int(shard["token_count"]) * UINT32_LE.itemsize
            if not path.is_file():
                raise ValueError(f"Resume shard missing: {path}")
            actual_bytes = path.stat().st_size
            if actual_bytes < expected_bytes:
                raise ValueError(
                    f"Resume shard shorter than checkpoint: {path}"
                )
            if actual_bytes > expected_bytes:
                with path.open("r+b") as handle:
                    handle.truncate(expected_bytes)
                    handle.flush()
                    os.fsync(handle.fileno())

    def _open_append_shard(self) -> None:
        if self._handle is not None:
            return
        if not self.shards or self.shards[-1]["token_count"] >= self.max_shard_tokens:
            index = len(self.shards)
            self.shards.append(
                {
                    "filename": f"tokens-{index:05d}.bin",
                    "token_count": 0,
                }
            )
        self._handle = (self.stage_dir / self.shards[-1]["filename"]).open("ab")

    def append(self, token_ids: Sequence[int] | np.ndarray) -> None:
        array = np.asarray(token_ids)
        if array.ndim != 1:
            raise ValueError("Packed token IDs must be one-dimensional")
        if not np.issubdtype(array.dtype, np.integer):
            raise ValueError("Packed token IDs must be integers")
        if array.size and (
            int(array.min()) < 0
            or int(array.max()) > np.iinfo(np.uint32).max
        ):
            raise ValueError("Token IDs must fit uint32")
        array = np.asarray(array, dtype=UINT32_LE)
        cursor = 0
        while cursor < len(array):
            self._open_append_shard()
            available = self.max_shard_tokens - int(
                self.shards[-1]["token_count"]
            )
            count = min(available, len(array) - cursor)
            np.asarray(array[cursor : cursor + count], dtype=UINT32_LE).tofile(
                self._handle
            )
            self.shards[-1]["token_count"] += count
            cursor += count
            if self.shards[-1]["token_count"] == self.max_shard_tokens:
                self.flush()
                self._handle.close()
                self._handle = None

    def flush(self) -> None:
        if self._handle is not None:
            self._handle.flush()
            os.fsync(self._handle.fileno())

    def checkpoint(self) -> list[dict[str, Any]]:
        self.flush()
        return [
            {
                "filename": item["filename"],
                "token_count": int(item["token_count"]),
            }
            for item in self.shards
        ]

    def finalize(self) -> list[dict[str, Any]]:
        self.flush()
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        result = []
        for shard in self.shards:
            path = self.stage_dir / shard["filename"]
            result.append(
                {
                    "filename": shard["filename"],
                    "token_count": int(shard["token_count"]),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        return result

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


def load_packed_manifest(path: str | Path) -> tuple[Path, dict[str, Any]]:
    input_path = Path(path).expanduser().resolve()
    manifest_path = input_path / "manifest.json" if input_path.is_dir() else input_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != PACKED_FORMAT_VERSION:
        raise ValueError(
            f"Unknown packed format version: {manifest.get('format_version')}"
        )
    if manifest.get("completion_status") != "complete":
        raise ValueError("Packed stage manifest is not complete")
    expected = manifest.get("manifest_content_sha256")
    if expected != manifest_content_hash(manifest):
        raise ValueError("Packed manifest content checksum mismatch")
    return manifest_path.parent, manifest


def validate_packed_shards(
    path: str | Path,
    *,
    expected_tokenizer_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    stage_dir, manifest = load_packed_manifest(path)
    if manifest.get("ordered_data_sha256") != ordered_data_hash(manifest):
        raise ValueError("Packed ordered-data checksum mismatch")
    if (
        expected_tokenizer_manifest_sha256 is not None
        and manifest["tokenizer"]["manifest_content_sha256"]
        != expected_tokenizer_manifest_sha256
    ):
        raise ValueError("Packed stage tokenizer mismatch")
    tokenizer_contract = manifest["tokenizer"]
    for name in (
        "base_vocab_size",
        "effective_vocab_size",
        "model_embedding_vocab_size",
    ):
        if (
            not isinstance(tokenizer_contract.get(name), int)
            or tokenizer_contract[name] <= 0
        ):
            raise ValueError(f"Packed tokenizer {name} is invalid")
    maximum_token_id = tokenizer_contract.get("maximum_emitted_token_id")
    model_vocab_size = tokenizer_contract["model_embedding_vocab_size"]
    if (
        not isinstance(maximum_token_id, int)
        or maximum_token_id < 0
        or maximum_token_id >= model_vocab_size
    ):
        raise ValueError(
            "Packed tokenizer maximum emitted ID does not fit the model "
            "embedding vocabulary"
        )
    if tokenizer_contract.get("trailing_unused_embedding_rows") != (
        model_vocab_size - maximum_token_id - 1
    ):
        raise ValueError(
            "Packed tokenizer unused embedding-row count mismatch"
        )
    total_tokens = 0
    shard_arrays: list[np.memmap] = []
    for index, shard in enumerate(manifest["shards"]):
        if shard["filename"] != f"tokens-{index:05d}.bin":
            raise ValueError("Packed shard filenames/order are not canonical")
        shard_path = stage_dir / shard["filename"]
        if not shard_path.is_file():
            raise ValueError(f"Packed shard missing: {shard_path}")
        expected_size = int(shard["token_count"]) * UINT32_LE.itemsize
        if shard_path.stat().st_size != expected_size:
            raise ValueError(f"Packed shard size mismatch: {shard_path}")
        if sha256_file(shard_path) != shard["sha256"]:
            raise ValueError(f"Packed shard checksum mismatch: {shard_path}")
        total_tokens += int(shard["token_count"])
        shard_array = np.memmap(
            shard_path,
            dtype=UINT32_LE,
            mode="r",
            shape=(int(shard["token_count"]),),
        )
        if len(shard_array) and int(shard_array.max()) > maximum_token_id:
            raise ValueError(
                "Packed shard contains a token ID outside the inspected "
                "tokenizer ID space"
            )
        if len(shard_array) and int(shard_array.max()) >= model_vocab_size:
            raise ValueError(
                "Packed shard contains a token ID outside the model embedding "
                "vocabulary"
            )
        shard_arrays.append(shard_array)
    if not manifest["shards"] or total_tokens <= 0:
        raise ValueError("Complete packed stage must contain tokens")
    if total_tokens != manifest["token_count"]:
        raise ValueError("Packed manifest total token count mismatch")
    sequence_length = int(manifest["reader"]["sequence_length"])
    if sequence_length <= 1:
        raise ValueError("Packed reader sequence length is invalid")
    usable_sequences = total_tokens // sequence_length
    expected_targets = usable_sequences * (sequence_length - 1)
    if manifest["target_token_count"] != expected_targets:
        raise ValueError("Packed target-token count mismatch")
    if manifest["adjacent_target_token_count"] != max(total_tokens - 1, 0):
        raise ValueError("Packed adjacent-target count mismatch")
    hit_cap = total_tokens == manifest["selection"]["max_output_tokens"]
    if manifest.get("hit_output_token_cap") is not hit_cap:
        raise ValueError("Packed output-token cap status mismatch")
    if manifest["stage"]["require_exact_output_tokens"] and not hit_cap:
        raise ValueError("Exact-output stage is underfilled")

    boundaries = manifest.get("boundaries")
    boundary_count = 0
    expected_eos = manifest["tokenizer"]["special_token_ids"]["eos_token_id"]
    shard_starts = []
    running = 0
    for array in shard_arrays:
        shard_starts.append(running)
        running += len(array)

    def validate_boundary_record(
        record: dict[str, Any],
        *,
        expected_index: int,
        previous_end: int,
    ) -> int:
        if record["document_index"] != expected_index:
            raise ValueError("Boundary document index discontinuity")
        if record["token_start"] != previous_end:
            raise ValueError("Boundary token range discontinuity")
        if record["token_end"] <= record["token_start"]:
            raise ValueError("Empty or inverted document boundary")
        if record["token_end"] > total_tokens:
            raise ValueError("Boundary exceeds packed token count")
        if record["content_token_count"] <= 0:
            raise ValueError("Boundary content-token count must be positive")
        if (
            record["token_end"] - record["token_start"]
            != record["content_token_count"] + 1
        ):
            raise ValueError("Boundary range does not equal content plus EOS")
        if record.get("eos_after") is not True:
            raise ValueError("Boundary must record eos_after=true")
        if not isinstance(record.get("truncated"), bool):
            raise ValueError("Boundary truncated flag must be boolean")
        if record.get("split") not in {"train", "validation"}:
            raise ValueError("Boundary has an unknown document split")
        if not isinstance(record.get("source_id"), str) or not record["source_id"]:
            raise ValueError("Boundary source ID must be non-empty")
        for name in ("content_sha256", "token_ids_sha256"):
            value = record.get(name)
            if not isinstance(value, str) or not re.fullmatch(
                r"[0-9a-f]{64}", value
            ):
                raise ValueError(f"Boundary {name} is not lowercase SHA-256")
        eos_offset = record["token_end"] - 1
        shard_index = bisect_right(shard_starts, eos_offset) - 1
        if not shard_arrays or int(
            shard_arrays[shard_index][eos_offset - shard_starts[shard_index]]
        ) != expected_eos:
            raise ValueError("Packed document boundary does not contain EOS")
        return int(record["token_end"])

    if boundaries is not None:
        if boundaries.get("filename") != "boundaries.jsonl":
            raise ValueError("Boundary filename is not canonical")
        document_reference = manifest.get("accepted_documents")
        if not isinstance(document_reference, dict):
            raise ValueError(
                "Sidecar-backed manifest must reference accepted documents"
            )
        if (
            document_reference.get("storage") != "boundary_sidecar"
            or document_reference.get("encoding") != "jsonl"
            or boundaries.get("encoding") != "jsonl"
        ):
            raise ValueError("Unknown accepted-document sidecar encoding")
        for key in ("filename", "record_count", "size_bytes", "sha256"):
            if document_reference.get(key) != boundaries.get(key):
                raise ValueError(
                    "Accepted-document reference differs from boundary metadata"
                )
        boundary_path = stage_dir / boundaries["filename"]
        if not boundary_path.is_file():
            raise ValueError("Boundary metadata file is missing")
        if boundary_path.stat().st_size != boundaries["size_bytes"]:
            raise ValueError("Boundary metadata size mismatch")
        if sha256_file(boundary_path) != boundaries["sha256"]:
            raise ValueError("Boundary metadata checksum mismatch")
        previous_end = 0
        with boundary_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                previous_end = validate_boundary_record(
                    record,
                    expected_index=boundary_count,
                    previous_end=previous_end,
                )
                boundary_count += 1
        if boundary_count != boundaries["record_count"]:
            raise ValueError("Boundary record-count mismatch")
        if boundary_count and previous_end != total_tokens:
            raise ValueError("Boundaries do not reconstruct the token stream")
    if boundaries is None:
        records = manifest.get("accepted_documents", [])
        if not isinstance(records, list):
            raise ValueError("Inline accepted documents must be a list")
        previous_end = 0
        for index, record in enumerate(records):
            previous_end = validate_boundary_record(
                record,
                expected_index=index,
                previous_end=previous_end,
            )
        if records and previous_end != total_tokens:
            raise ValueError("Compact boundaries do not reconstruct token stream")
        boundary_count = len(records)
    if boundary_count != manifest["accepted_document_count"]:
        raise ValueError("Accepted-document count differs from boundaries")
    return {
        "status": "valid",
        "stage_id": manifest["stage"]["stage_id"],
        "token_count": total_tokens,
        "target_token_count": manifest["target_token_count"],
        "document_count": boundary_count,
        "shard_count": len(manifest["shards"]),
        "ordered_data_sha256": manifest["ordered_data_sha256"],
        "manifest_content_sha256": manifest["manifest_content_sha256"],
    }


class PackedShardSource:
    def __init__(
        self,
        path: str | Path,
        *,
        drop_incomplete_sequence: bool = True,
        expected_tokenizer_manifest_sha256: str | None = None,
    ):
        validate_packed_shards(
            path,
            expected_tokenizer_manifest_sha256=(
                expected_tokenizer_manifest_sha256
            ),
        )
        self.stage_dir, self.manifest = load_packed_manifest(path)
        self.drop_incomplete_sequence = drop_incomplete_sequence
        self._arrays = [
            np.memmap(
                self.stage_dir / shard["filename"],
                dtype=UINT32_LE,
                mode="r",
                shape=(int(shard["token_count"]),),
            )
            for shard in self.manifest["shards"]
        ]

    @property
    def token_count(self) -> int:
        return int(self.manifest["token_count"])

    def global_offset(self, position: TokenPosition) -> int:
        position.validate()
        if position.shard_index > len(self._arrays):
            raise ValueError("Shard index is past end of stream")
        if position.shard_index == len(self._arrays):
            if position.token_offset != 0:
                raise ValueError("EOF position must have token_offset=0")
            return self.token_count
        shard_size = len(self._arrays[position.shard_index])
        if position.token_offset > shard_size:
            raise ValueError("Token offset is past shard end")
        return sum(len(array) for array in self._arrays[: position.shard_index]) + (
            position.token_offset
        )

    def position_at(self, global_offset: int) -> TokenPosition:
        if global_offset < 0 or global_offset > self.token_count:
            raise ValueError("Global token offset is outside stream")
        remaining = global_offset
        for index, array in enumerate(self._arrays):
            if remaining < len(array):
                return TokenPosition(index, remaining)
            remaining -= len(array)
        return TokenPosition(len(self._arrays), 0)

    def read_tokens(
        self, count: int, *, start: TokenPosition | None = None
    ) -> tuple[np.ndarray, TokenPosition]:
        if count < 0:
            raise ValueError("count must be non-negative")
        position = start or TokenPosition(0, 0)
        offset = self.global_offset(position)
        remaining = min(count, self.token_count - offset)
        pieces = []
        cursor = position
        while remaining:
            array = self._arrays[cursor.shard_index]
            available = len(array) - cursor.token_offset
            take = min(remaining, available)
            pieces.append(
                np.asarray(
                    array[cursor.token_offset : cursor.token_offset + take],
                    dtype=np.uint32,
                )
            )
            remaining -= take
            cursor = self.position_at(offset + sum(len(piece) for piece in pieces))
        result = (
            np.concatenate(pieces)
            if pieces
            else np.empty((0,), dtype=np.uint32)
        )
        return result, cursor

    def position_for_global_sequence(
        self, sequence_index: int, *, sequence_length: int
    ) -> TokenPosition:
        if sequence_index < 0 or sequence_length <= 0:
            raise ValueError("Invalid global sequence position")
        return self.position_at(sequence_index * sequence_length)

    def iter_batches(
        self,
        *,
        sequence_length: int,
        global_sequences_per_batch: int,
        start: TokenPosition | None = None,
        sequence_prefix_count: int | None = None,
    ) -> Iterator[TokenBatch]:
        if sequence_length <= 1 or global_sequences_per_batch <= 0:
            raise ValueError("Invalid batch dimensions")
        position = start or TokenPosition(0, 0)
        if sequence_prefix_count is not None:
            if sequence_prefix_count <= 0:
                raise ValueError("sequence_prefix_count must be positive")
            available_complete = self.token_count // sequence_length
            if sequence_prefix_count > available_complete:
                raise ValueError(
                    "sequence_prefix_count exceeds available complete sequences"
                )
        while True:
            global_offset = self.global_offset(position)
            if global_offset % sequence_length:
                raise ValueError("Batch start is not sequence-aligned")
            available_tokens = self.token_count - global_offset
            possible_sequences = max(available_tokens // sequence_length, 0)
            if sequence_prefix_count is not None:
                possible_sequences = min(
                    possible_sequences,
                    max(
                        sequence_prefix_count
                        - global_offset // sequence_length,
                        0,
                    ),
                )
            if possible_sequences == 0:
                return
            sequences = min(global_sequences_per_batch, possible_sequences)
            flat, _ = self.read_tokens(
                sequences * sequence_length, start=position
            )
            inputs = np.asarray(
                flat.reshape(sequences, sequence_length), dtype=np.int64
            )
            labels = inputs.copy()
            next_position = self.position_at(
                global_offset + sequences * sequence_length
            )
            yield TokenBatch(
                input_ids=inputs,
                labels=labels,
                valid_target_count=sequences * (sequence_length - 1),
                start_position=position,
                next_position=next_position,
            )
            position = next_position
