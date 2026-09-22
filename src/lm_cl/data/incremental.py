"""Experimental immutable-block protocol; never accepted as a legacy manifest.

The producer owns shuffle/packing state. Consumers see only committed blocks.
Raw rows belong to a separate, immutable source cache, not publication records.
"""
from __future__ import annotations

import hashlib
import copy
from contextlib import nullcontext
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Protocol

import numpy as np

from lm_cl.data.selection import document_split, stable_source_id
from lm_cl.data.types import TokenBatch, TokenPosition


FORMAT = "lm-cl-incremental-blocks-v1"


def digest(value: Any) -> str:
    return hashlib.sha256(json_bytes(value)).hexdigest()


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False) + "\n").encode()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def immutable_write(path: Path, content: bytes) -> None:
    """Publish complete bytes without replacing any existing committed content."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if file_hash(path) != hashlib.sha256(content).hexdigest():
            raise ValueError(f"Existing immutable content differs: {path}")
        return
    temporary = path.with_name(path.name + ".partial")
    if temporary.exists():
        if file_hash(temporary) != hashlib.sha256(content).hexdigest():
            # Keep failed bytes. A new file is created for this retry.
            temporary = path.with_name(path.name + f".retry-{os.getpid()}.partial")
    with temporary.open("wb") as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    # link is exclusive; a racing publisher cannot replace an existing record.
    try:
        os.link(temporary, path)
    except FileExistsError:
        if file_hash(path) != hashlib.sha256(content).hexdigest():
            raise ValueError(f"Concurrent publication differs: {path}")
    # No automatic unlink: the temporary name is a hardlink, not a second copy.
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _tuples(value: Any) -> Any:
    return tuple(_tuples(x) for x in value) if isinstance(value, list) else value


class RowSource(Protocol):
    def row(self, index: int) -> dict[str, Any] | None: ...


@dataclass(frozen=True)
class Recipe:
    format: str
    source_identity: dict[str, Any]
    language: str
    purpose: str
    tokenizer_identity: dict[str, Any]
    max_input_documents: int
    output_tokens: int
    block_tokens: int
    sequence_length: int
    document_order_seed: int
    shuffle_buffer_documents: int
    split_seed: int
    validation_permyriad: int
    eos_token_id: int
    maximum_token_id: int
    text_field: str
    id_field: str | None
    predecessor_streams: list[dict[str, str]]
    order_algorithm: str = "bounded_buffer_python_v1"
    split_algorithm: str = "sha256_permyriad_v1"
    packing_algorithm: str = "uint32_eos_final_truncation_v1"

    def validate(self) -> None:
        if self.format != FORMAT:
            raise ValueError("Unknown incremental format")
        if self.language not in {"en", "zh_written", "fr", "ja", "es", "de", "pt", "ru", "vi"}:
            raise ValueError("Unknown language")
        if self.purpose not in {"train", "validation", "inspection", "timing_only"}:
            raise ValueError("Unknown purpose")
        if self.purpose == "timing_only" and self.source_identity.get("kind") != "completed_packed_timing_replay_v1":
            raise ValueError("Timing-only repetition requires an explicit frozen packed replay identity")
        for name in ("max_input_documents", "output_tokens", "block_tokens", "sequence_length", "shuffle_buffer_documents"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"Invalid {name}")
        if self.sequence_length < 2 or self.output_tokens % self.sequence_length:
            raise ValueError("Budget must contain complete sequences")
        if not 0 <= self.eos_token_id <= self.maximum_token_id < 2**32:
            raise ValueError("Invalid token bounds")
        if not 0 <= self.validation_permyriad <= 10000:
            raise ValueError("Invalid split fraction")
        if not self.source_identity or not self.tokenizer_identity:
            raise ValueError("Immutable source and tokenizer identities required")
        if (self.order_algorithm, self.split_algorithm, self.packing_algorithm) != (
            "bounded_buffer_python_v1", "sha256_permyriad_v1", "uint32_eos_final_truncation_v1"
        ):
            raise ValueError("Unknown preparation algorithm")
        for predecessor in self.predecessor_streams:
            if set(predecessor) != {"path", "recipe_sha256", "completion_sha256"}:
                raise ValueError("Predecessor must have frozen completion identity")

    @property
    def sha256(self) -> str:
        return digest(asdict(self))


def initialize(root: Path, recipe: Recipe) -> None:
    recipe.validate()
    root.mkdir(parents=True, exist_ok=True)
    immutable_write(root / "recipe.json", json_bytes(asdict(recipe)))
    (root / "blocks").mkdir(exist_ok=True)
    (root / "commits").mkdir(exist_ok=True)


def load_recipe(root: Path) -> Recipe:
    recipe = Recipe(**json.loads((root / "recipe.json").read_text()))
    recipe.validate()
    return recipe


def _record(root: Path, index: int, previous: str, recipe: Recipe) -> tuple[dict, str] | None:
    path = root / "commits" / f"{index:08d}.json"
    if not path.exists():
        later = [p for p in (root / "commits").glob("*.json") if p.stem.isdigit() and int(p.stem) > index]
        if later:
            raise ValueError("Missing/reordered committed block")
        return None
    record = json.loads(path.read_text())
    if set(record) != {"format", "recipe_sha256", "index", "previous", "start", "count", "data_sha256", "boundaries", "producer_state"}:
        raise ValueError("Invalid committed record fields")
    if record["format"] != FORMAT or record["recipe_sha256"] != recipe.sha256:
        raise ValueError("Committed block recipe mismatch")
    if record["index"] != index or record["previous"] != previous:
        raise ValueError("Duplicate/reordered committed block")
    expected_count = min(recipe.block_tokens, recipe.output_tokens - index * recipe.block_tokens)
    if record["start"] != index * recipe.block_tokens or record["count"] != expected_count or expected_count <= 0:
        raise ValueError("Noncontiguous or over-budget block")
    data = root / "blocks" / f"{index:08d}.bin"
    if not data.is_file() or data.stat().st_size != record["count"] * 4 or file_hash(data) != record["data_sha256"]:
        raise ValueError("Missing/corrupt committed token file")
    values = np.memmap(data, dtype="<u4", mode="r")
    if len(values) and int(values.max()) > recipe.maximum_token_id:
        raise ValueError("Committed token ID exceeds tokenizer")
    return record, file_hash(path)


class IncrementalSource:
    """Stable global offset cursor; publication does not change source identity."""
    def __init__(self, root: str | Path, *, recipe_sha256: str | None = None,
                 wait_seconds: float = 60, poll_seconds: float = .1):
        self.root = Path(root).resolve()
        self.recipe = load_recipe(self.root)
        if recipe_sha256 is not None and recipe_sha256 != self.recipe.sha256:
            raise ValueError("Frozen recipe checksum mismatch")
        if wait_seconds < 0 or poll_seconds <= 0:
            raise ValueError("Invalid waiting policy")
        self.wait_seconds, self.poll_seconds = wait_seconds, poll_seconds
        self.records: list[dict] = []
        self.hashes: list[str] = []
        self.arrays: list[np.ndarray] = []
        self.signatures: list[tuple] = []
        self.waited_seconds = 0.0
        self.refresh()

    @property
    def token_count(self) -> int:
        return self.recipe.output_tokens

    @property
    def ready_tokens(self) -> int:
        return sum(r["count"] for r in self.records)

    @property
    def identity(self) -> dict:
        return {"kind": FORMAT, "root": str(self.root), "recipe_sha256": self.recipe.sha256}

    def refresh(self) -> None:
        for i, signature in enumerate(self.signatures):
            paths = [self.root / "commits" / f"{i:08d}.json", self.root / "blocks" / f"{i:08d}.bin"]
            current = tuple((p.stat().st_ino, p.stat().st_size, p.stat().st_mtime_ns) for p in paths)
            if current != signature:
                raise ValueError("Previously committed data changed")
        while True:
            i = len(self.records)
            item = _record(self.root, i, self.hashes[-1] if self.hashes else self.recipe.sha256, self.recipe)
            if item is None:
                break
            record, checksum = item
            self.records.append(record)
            self.hashes.append(checksum)
            self.arrays.append(np.memmap(self.root / "blocks" / f"{i:08d}.bin", dtype="<u4", mode="r"))
            paths = [self.root / "commits" / f"{i:08d}.json", self.root / "blocks" / f"{i:08d}.bin"]
            self.signatures.append(tuple((p.stat().st_ino, p.stat().st_size, p.stat().st_mtime_ns) for p in paths))
        complete = self.root / "complete.json"
        if complete.exists():
            expected = {"format": FORMAT, "recipe_sha256": self.recipe.sha256,
                        "blocks": len(self.records), "tokens": self.ready_tokens,
                        "chain_sha256": self.hashes[-1] if self.hashes else self.recipe.sha256}
            if json.loads(complete.read_text()) != expected or self.ready_tokens != self.token_count:
                raise ValueError("Invalid completion record")

    def ensure(self, end: int) -> None:
        start = time.monotonic()
        while True:
            self.refresh()
            if self.ready_tokens >= end:
                self.waited_seconds += time.monotonic() - start
                return
            if time.monotonic() - start >= self.wait_seconds:
                raise TimeoutError(f"Data not ready: need {end}, committed {self.ready_tokens}; this is not EOF")
            time.sleep(self.poll_seconds)

    def proof(self, offset: int) -> dict:
        if not 0 <= offset <= self.token_count:
            raise ValueError("Invalid consumed prefix")
        self.ensure(offset)
        count = (offset + self.recipe.block_tokens - 1) // self.recipe.block_tokens
        return {"recipe_sha256": self.recipe.sha256, "consumed_tokens": offset,
                "covering_blocks": count, "chain_sha256": self.hashes[count-1] if count else self.recipe.sha256}

    def validate_proof(self, proof: dict) -> None:
        if self.proof(proof["consumed_tokens"]) != proof:
            raise ValueError("Consumed-prefix checkpoint proof mismatch")

    def position_at(self, offset: int) -> TokenPosition:
        if not 0 <= offset <= self.token_count:
            raise ValueError("Offset outside frozen budget")
        return TokenPosition(0, offset)

    def global_offset(self, position: TokenPosition) -> int:
        position.validate()
        if position.shard_index != 0:
            raise ValueError("Incremental cursor uses global token offset")
        self.position_at(position.token_offset)
        return position.token_offset

    def position_for_global_sequence(self, sequence_index: int, *, sequence_length: int) -> TokenPosition:
        return self.position_at(sequence_index * sequence_length)

    def read_tokens(self, count: int, *, start: TokenPosition | None = None) -> tuple[np.ndarray, TokenPosition]:
        begin = self.global_offset(start or TokenPosition(0, 0))
        end = min(begin + count, self.token_count)
        if count < 0:
            raise ValueError("Negative token count")
        self.ensure(end)
        parts = []
        offset = begin
        while offset < end:
            i, local = divmod(offset, self.recipe.block_tokens)
            take = min(end - offset, len(self.arrays[i]) - local)
            parts.append(np.asarray(self.arrays[i][local:local+take]))
            offset += take
        return (np.concatenate(parts) if parts else np.empty(0, dtype="<u4")), self.position_at(end)

    def iter_batches(self, *, sequence_length: int, global_sequences_per_batch: int,
                     start: TokenPosition | None = None, sequence_prefix_count: int | None = None) -> Iterator[TokenBatch]:
        if sequence_length != self.recipe.sequence_length or global_sequences_per_batch <= 0:
            raise ValueError("Reader dimensions differ from recipe")
        offset = self.global_offset(start or TokenPosition(0, 0))
        end = self.token_count if sequence_prefix_count is None else sequence_prefix_count * sequence_length
        if not 0 < end <= self.token_count or offset % sequence_length:
            raise ValueError("Invalid sequence window")
        while offset < end:
            count = min(global_sequences_per_batch * sequence_length, end - offset)
            values, next_position = self.read_tokens(count, start=self.position_at(offset))
            inputs = values.reshape(-1, sequence_length).astype(np.int64)
            yield TokenBatch(inputs, inputs.copy(), len(inputs)*(sequence_length-1), self.position_at(offset), next_position)
            offset += count


class Producer:
    """One ordered publisher, resumable after any complete block.

    This bounded prototype keeps document hash sets in RAM. It is not a
    replacement for a production-scale on-disk overlap registry.
    """
    def __init__(self, root: Path, source: RowSource, tokenizer: Any, *, publication_guard=None):
        self.root, self.source, self.tokenizer = root, source, tokenizer
        self.reader = IncrementalSource(root, wait_seconds=0)
        self.recipe = self.reader.recipe
        if self.recipe.purpose == "timing_only":
            raise ValueError("Timing replay is imported from completed packed bytes, never produced from documents")
        self.publication_guard = publication_guard or nullcontext
        self.content_hashes: set[str] = set()
        self.token_hashes: set[str] = set()
        for predecessor in self.recipe.predecessor_streams:
            prior = IncrementalSource(predecessor["path"], recipe_sha256=predecessor["recipe_sha256"], wait_seconds=0)
            completion = prior.root / "complete.json"
            if not completion.exists() or file_hash(completion) != predecessor["completion_sha256"]:
                raise ValueError("Earlier ownership stream is not frozen and complete")
            for record in prior.records:
                for b in record["boundaries"]:
                    self.content_hashes.add(b["content_sha256"])
                    self.token_hashes.add(b["token_ids_sha256"])
        for record in self.reader.records:
            for b in record["boundaries"]:
                self.content_hashes.add(b["content_sha256"])
                self.token_hashes.add(b["token_ids_sha256"])
        self.rng = random.Random(self.recipe.document_order_seed)
        if self.reader.records:
            self.state = copy.deepcopy(self.reader.records[-1]["producer_state"])
            self.rng.setstate(_tuples(self.state["rng"]))
        else:
            self.state = {"source_cursor": 0, "buffer": [], "initialized": False,
                          "draining": False, "pending_tokens": [], "pending_offset": 0,
                          "assigned_tokens": 0, "accepted_documents": 0,
                          "rejections": {}, "selected_documents": 0, "rng": self.rng.getstate()}

    def _reject(self, reason: str) -> None:
        r = self.state["rejections"]
        r[reason] = r.get(reason, 0) + 1

    def _valid_next(self) -> int | None:
        while self.state["source_cursor"] < self.recipe.max_input_documents:
            index = self.state["source_cursor"]
            row = self.source.row(index)
            if row is None:
                return None
            self.state["source_cursor"] += 1
            if not isinstance(row, dict):
                self._reject("row_not_mapping")
                continue
            text = row.get(self.recipe.text_field)
            if not isinstance(text, str):
                self._reject("missing_or_non_string_text")
                continue
            if not text:
                self._reject("empty_text")
                continue
            return index
        return None

    def _ordered_next(self) -> int | None:
        buffer = self.state["buffer"]
        if not self.state["initialized"]:
            while len(buffer) < self.recipe.shuffle_buffer_documents:
                item = self._valid_next()
                if item is None:
                    break
                buffer.append(item)
            self.state["initialized"] = True
        if not self.state["draining"]:
            item = self._valid_next()
            if item is not None:
                i = self.rng.randrange(len(buffer))
                chosen, buffer[i] = buffer[i], item
                return chosen
            self.rng.shuffle(buffer)
            self.state["draining"] = True
        return buffer.pop(0) if buffer else None

    def _accept(self) -> dict | None:
        while True:
            index = self._ordered_next()
            if index is None:
                raise RuntimeError("Source/document cap exhausted before exact token budget")
            row = self.source.row(index)
            assert row is not None
            text = row[self.recipe.text_field]
            content_hash = hashlib.sha256(text.encode()).hexdigest()
            split = document_split(content_hash, split_seed=self.recipe.split_seed,
                                   validation_permyriad=self.recipe.validation_permyriad)
            if self.recipe.purpose != "inspection" and split != self.recipe.purpose:
                self._reject(f"assigned_to_{split}")
                continue
            self.state["selected_documents"] += 1
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            if not ids:
                self._reject("empty_token_sequence")
                continue
            if any(type(i) is not int or not 0 <= i <= self.recipe.maximum_token_id for i in ids):
                raise ValueError("Tokenizer emitted invalid IDs")
            token_hash = hashlib.sha256(np.asarray(ids, dtype="<u4").tobytes()).hexdigest()
            if content_hash in self.content_hashes or token_hash in self.token_hashes:
                self._reject("duplicate_content_or_tokens")
                continue
            remaining = self.recipe.output_tokens - self.state["assigned_tokens"]
            if remaining < (1 if getattr(self, "allow_empty_final_document", False) else 2):
                raise RuntimeError("Exact budget cannot fit final document plus EOS")
            truncated = len(ids) + 1 > remaining
            kept = ids[:remaining-1]
            packed = kept + [self.recipe.eos_token_id]
            begin = self.state["assigned_tokens"]
            boundary = {"document_index": self.state["accepted_documents"],
                        "source_id": stable_source_id(row, id_field=self.recipe.id_field, content_sha256=content_hash),
                        "content_sha256": content_hash, "token_ids_sha256": token_hash,
                        "token_start": begin, "content_token_count": len(kept),
                        "token_end": begin+len(packed), "eos_after": True,
                        "truncated": truncated, "split": split}
            self.content_hashes.add(content_hash)
            self.token_hashes.add(token_hash)
            self.state["accepted_documents"] += 1
            self.state["assigned_tokens"] += len(packed)
            self.state["pending_tokens"], self.state["pending_offset"] = packed, 0
            return boundary

    def publish_one(self, *, fault: str | None = None) -> dict | None:
        i = len(self.reader.records)
        start = self.reader.ready_tokens
        if start == self.recipe.output_tokens:
            self.finish()
            return None
        wanted = min(self.recipe.block_tokens, self.recipe.output_tokens-start)
        values: list[int] = []
        boundaries = []
        while len(values) < wanted:
            if self.state["pending_offset"] == len(self.state["pending_tokens"]):
                boundaries.append(self._accept())
            pending = self.state["pending_tokens"]
            offset = self.state["pending_offset"]
            take = min(len(pending)-offset, wanted-len(values))
            values.extend(pending[offset:offset+take])
            self.state["pending_offset"] += take
        self.state["rng"] = self.rng.getstate()
        content = np.asarray(values, dtype="<u4").tobytes()
        record = {"format": FORMAT, "recipe_sha256": self.recipe.sha256,
                  "index": i, "previous": self.reader.hashes[-1] if i else self.recipe.sha256,
                  "start": start, "count": wanted, "data_sha256": hashlib.sha256(content).hexdigest(),
                  "boundaries": boundaries, "producer_state": self.state}
        with self.publication_guard():
            immutable_write(self.root / "blocks" / f"{i:08d}.bin", content)
            if fault == "after_data":
                raise InterruptedError("Injected producer interruption after data, before commit")
            immutable_write(self.root / "commits" / f"{i:08d}.json", json_bytes(record))
        if fault == "after_commit":
            raise InterruptedError("Injected producer interruption after commit")
        self.reader.refresh()
        if self.reader.ready_tokens == self.recipe.output_tokens:
            self.finish()
        return record

    def finish(self) -> None:
        if self.reader.ready_tokens != self.recipe.output_tokens:
            raise ValueError("Cannot complete an underfilled stream")
        immutable_write(self.root / "complete.json", json_bytes({
            "format": FORMAT, "recipe_sha256": self.recipe.sha256,
            "blocks": len(self.reader.records), "tokens": self.reader.ready_tokens,
            "chain_sha256": self.reader.hashes[-1]}))

    def run(self, *, max_blocks: int | None = None) -> None:
        n = 0
        while max_blocks is None or n < max_blocks:
            if self.publish_one() is None:
                return
            n += 1
