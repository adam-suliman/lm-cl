"""Production, recipe-bound packed blocks with a disposable, bounded cache.

Only the producer accesses the network. Readers copy verified completed blocks
under a cache lock; eviction never removes receipts, producer state or checkpoints.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import time
import tempfile

import numpy as np

from lm_cl.data.incremental import digest, json_bytes
from lm_cl.data.packed import PackedShardSource
from lm_cl.data.storage import atomic_write_json
from lm_cl.data.types import TokenBatch, TokenPosition

FORMAT = "lm-cl-production-streaming-v1"
ALTERNATING_FORMAT = "lm-cl-alternating-streaming-v1"


@contextmanager
def lock(path, *, nonblocking=False):
    if Path(path).is_symlink():
        raise ValueError("Symlinked streaming lock is forbidden")
    with Path(path).open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0))
        yield


@contextmanager
def connect(root, *, create=False):
    path = Path(root) / "receipts.sqlite"
    if not create and not path.is_file():
        raise ValueError("Streaming receipt database is missing; do not create a replacement")
    db = sqlite3.connect(path, timeout=60)
    db.execute("PRAGMA busy_timeout=60000")
    try:
        with db:
            yield db
    finally:
        db.close()


def validate_plan(plan):
    from lm_cl.data.incremental import Recipe
    from lm_cl.data.incremental_remote import REVISION, MAPPINGS
    from lm_cl.launcher.streaming import StreamingSettings, AlternatingSettings
    if (plan["format"] not in {FORMAT, ALTERNATING_FORMAT} or plan["policy"] != "serial_interleaved_global_dedup_v1"
            or plan["final_document_remainder"] != "eos_only_if_one_token_v1"):
        raise ValueError("Unknown streaming preparation policy/version")
    if (plan.get("checkpoint_retention") not in {None, "cycle_end_v1"}
            or (plan.get("checkpoint_retention") is not None
                and plan["format"] != ALTERNATING_FORMAT)):
        raise ValueError("Unknown streaming checkpoint-retention policy")
    offsets = {}
    for name, spec in plan["streams"].items():
        recipe = Recipe(**spec)
        recipe.validate()
        (AlternatingSettings if plan["format"] == ALTERNATING_FORMAT else StreamingSettings)(**plan["limits"]).validate(recipe.sequence_length)
        if (recipe.purpose not in {"train", "validation"} or name != f"{recipe.purpose}-{recipe.language}"
                or recipe.block_tokens != plan["limits"]["block_tokens"]
                or recipe.source_identity.get("revision") != REVISION
                or recipe.source_identity.get("mappings") != MAPPINGS
                or recipe.source_identity.get("repo_id") != "uonlp/CulturaX"):
            raise ValueError("Streaming source policy differs from the frozen production contract")
        offsets[name] = 0
    for block in plan["blocks"]:
        name, count = block["stream"], block["count"]
        spec = plan["streams"][name]
        if (type(count) is not int or not 0 < count <= spec["block_tokens"]
                or count % spec["sequence_length"] or block["start"] != offsets[name]):
            raise ValueError("Invalid streaming block coverage")
        offsets[name] += count
    if any(offsets[name] != spec["output_tokens"] for name,spec in plan["streams"].items()):
        raise ValueError("Streaming plan does not cover the exact frozen budgets")
    if plan["format"] == ALTERNATING_FORMAT:
        from lm_cl.data.alternating import validate_alternation
        validate_alternation(plan)


def load_plan(root):
    root = Path(root).resolve()
    plan = json.loads((root / "study.json").read_text())
    claimed = plan.pop("sha256")
    if plan["format"] not in {FORMAT, ALTERNATING_FORMAT} or digest(plan) != claimed:
        raise ValueError("Unknown/corrupt streaming study recipe")
    validate_plan(plan)
    owner = json.loads((root / ".streaming-owner.json").read_text())
    if owner != {"format": plan["format"], "recipe_sha256": claimed}:
        raise ValueError("Streaming cache ownership mismatch")
    plan["sha256"] = claimed
    return plan


def initialize(root, plan):
    root = Path(root).resolve()
    plan = dict(plan)
    plan.setdefault("format", FORMAT)
    validate_plan(plan)
    plan["sha256"] = digest(plan)
    if root.exists():
        if load_plan(root) != plan:
            raise ValueError("Existing streaming recipe differs; use a new data root")
        return plan
    destination = root
    root.parent.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix=f".initializing-{plan['sha256'][:12]}-", dir=root.parent))
    for name in ("cache", "requests", "raw"):
        (root / name).mkdir()
    atomic_write_json(root / "study.json", plan)
    atomic_write_json(root / ".streaming-owner.json", {"format": plan["format"], "recipe_sha256": plan["sha256"]})
    with connect(root, create=True) as db:
        db.executescript("""
        CREATE TABLE receipts (ordinal INTEGER PRIMARY KEY, stream TEXT NOT NULL,
          state TEXT NOT NULL, record TEXT NOT NULL, sha256 TEXT NOT NULL);
        CREATE INDEX receipts_stream ON receipts(stream, ordinal);
        CREATE TABLE content_hashes (hash TEXT PRIMARY KEY, ordinal INTEGER NOT NULL);
        CREATE TABLE token_hashes (hash TEXT PRIMARY KEY, ordinal INTEGER NOT NULL);
        """)
        if plan["format"] == ALTERNATING_FORMAT:
            from lm_cl.data.alternating import initialize_tables
            initialize_tables(root, plan, db)
    os.rename(root, destination)
    fd = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return plan


def receipt(root, ordinal):
    with connect(root) as db:
        row = db.execute("SELECT record,sha256 FROM receipts WHERE ordinal=?", (ordinal,)).fetchone()
    if row is None:
        return None
    record = json.loads(row[0])
    if digest(record) != row[1] or record["ordinal"] != ordinal:
        raise ValueError("Streaming receipt checksum mismatch")
    return record, row[1]


def stream_identity(root, stream):
    plan = load_plan(root)
    if stream not in plan["streams"]:
        raise ValueError("Unknown streaming source")
    return {"kind": "streaming_packed", "root": str(Path(root).resolve()),
            "stream": stream, "recipe_sha256": plan["sha256"]}


def source_from_pipeline(pipeline):
    stage = Path(pipeline.storage.generated_root) / "stages" / pipeline.stage.stage_id
    reference = json.loads((stage / "stream.json").read_text())
    if set(reference) != {"root", "stream", "recipe_sha256"}:
        raise ValueError("Invalid streaming reference")
    source = StreamingPackedSource(reference["root"], reference["stream"])
    if source.plan["sha256"] != reference["recipe_sha256"]:
        raise ValueError("Streaming reference identity differs")
    if pipeline.stage.stage_id != f"stream-{source.plan['sha256']}-{source.stream}":
        raise ValueError("Streaming config/reference mismatch")
    from dataclasses import asdict
    if asdict(pipeline.tokenizer) != source.plan["tokenizer_reference"]:
        raise ValueError("Streaming tokenizer identity differs")
    from lm_cl.data.incremental import file_hash
    if file_hash(Path(pipeline.tokenizer.manifest_path)) != source.spec["tokenizer_identity"]["manifest_file_sha256"]:
        raise ValueError("Streaming tokenizer manifest changed")
    expected = (source.spec["language"], source.plan["purposes"][source.stream], source.spec["output_tokens"],
                source.spec["document_order_seed"], source.spec["split_seed"], source.spec["validation_permyriad"],
                source.spec["shuffle_buffer_documents"], source.spec["max_input_documents"])
    actual = (pipeline.stage.language, pipeline.stage.purpose, pipeline.selection.max_output_tokens,
              pipeline.selection.document_order_seed, pipeline.selection.split_seed, pipeline.selection.validation_permyriad,
              pipeline.selection.shuffle_buffer_documents, pipeline.selection.max_input_documents)
    if actual != expected or pipeline.dataset.revision != source.spec["source_identity"]["revision"]:
        raise ValueError("Streaming pipeline differs from frozen recipe")
    if pipeline.reader.sequence_length != source.spec["sequence_length"]:
        raise ValueError("Streaming sequence length mismatch")
    if pipeline.tokenizer.manifest_path != source.plan["tokenizer_manifest"]:
        raise ValueError("Streaming tokenizer path differs")
    return source


def prefix_proof(identity, offset):
    source = StreamingPackedSource(identity["root"], identity["stream"])
    if source.identity != {k: identity[k] for k in source.identity}:
        raise ValueError("Streaming checkpoint recipe changed")
    covering = [b for b in source.blocks if b["start"] < offset]
    if offset < 0 or offset > source.token_count:
        raise ValueError("Invalid streaming checkpoint offset")
    proof = {"recipe_sha256": source.plan["sha256"], "stream": source.stream,
             "consumed_tokens": offset, "covering_receipts": []}
    for block in covering:
        item = receipt(source.root, block["ordinal"])
        if item is None:
            raise ValueError("Checkpoint references unpublished data")
        proof["covering_receipts"].append([block["ordinal"], item[1]])
    return proof


def validate_prefix(identity, proof):
    if proof is None or prefix_proof(identity, proof["consumed_tokens"]) != proof:
        raise ValueError("Streaming checkpoint consumed-prefix proof differs")


def cache_path(root, ordinal):
    return Path(root) / "cache" / f"{ordinal:08d}.bin"


def block_path(root, plan, ordinal):
    if plan["format"] == ALTERNATING_FORMAT and plan["purposes"][plan["blocks"][ordinal]["stream"]] != "continual_train":
        return Path(root) / "pinned" / f"{ordinal:08d}.bin"
    return cache_path(root, ordinal)


def publish_cache(root, plan, ordinal, content):
    """Only files in this recipe's newly owned disposable cache are evicted."""
    root = Path(root)
    if load_plan(root)["sha256"] != plan["sha256"]:
        raise ValueError("Cache owner changed")
    capacity = plan["limits"]["token_cache_bytes"]
    if len(content) > capacity:
        raise ValueError("Block exceeds token-cache capacity")
    if (root / "cache").is_symlink():
        raise ValueError("Symlinked disposable cache is forbidden")
    with lock(root / "cache.lock"):
        for entry in (root / "cache").iterdir():
            if entry.is_symlink() or not entry.is_file() or entry.suffix not in {".bin", ".tmp"} or len(entry.stem) != 8 or not entry.stem.isdigit() or not 0 <= int(entry.stem) < len(plan["blocks"]):
                raise ValueError("Unrecognized disposable cache entry")
        files = sorted((root / "cache").iterdir(), key=lambda p: p.stat().st_mtime_ns)
        total = sum(p.stat().st_size for p in files)
        for old in files:
            if total + len(content) <= capacity:
                break
            if old.is_symlink() or old.parent != root / "cache":
                raise ValueError("Unsafe disposable cache entry")
            total -= old.stat().st_size
            old.unlink()
        if shutil.disk_usage(root).free < len(content) + plan["limits"]["minimum_free_bytes"]:
            raise RuntimeError("Streaming free-space floor reached")
        path = cache_path(root, ordinal)
        temp = path.with_suffix(".tmp")
        with temp.open("wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)


class StreamingPackedSource(PackedShardSource):
    """Packed source interface used by the actual continual/probe trainers."""
    def __init__(self, root, stream):
        self.root = Path(root).resolve()
        self.plan = load_plan(self.root)
        self.stream = stream
        self.spec = self.plan["streams"][stream]
        self.blocks = [dict(b, ordinal=i) for i, b in enumerate(self.plan["blocks"]) if b["stream"] == stream]
        self.identity = stream_identity(self.root, stream)
        self.waited_seconds = 0.0

    @property
    def token_count(self):
        return self.spec["output_tokens"]

    def position_at(self, offset):
        if not 0 <= offset <= self.token_count:
            raise ValueError("Streaming position outside source")
        return TokenPosition(0, offset)

    def global_offset(self, position):
        position.validate()
        if position.shard_index != 0:
            raise ValueError("Streaming positions use global token offsets")
        self.position_at(position.token_offset)
        return position.token_offset

    def read_tokens(self, count, *, start=None):
        if count < 0:
            raise ValueError("Negative streaming read length")
        offset = self.global_offset(start or TokenPosition(0, 0))
        end = min(offset + count, self.token_count)
        selected = [b for b in self.blocks if b["start"] < end and b["start"]+b["count"] > offset]
        parts = []
        for block in selected if count else []:
            values = self._read_block(block)
            lo = max(offset, block["start"]) - block["start"]
            hi = min(end, block["start"]+block["count"]) - block["start"]
            parts.append(values[lo:hi])
        data = np.concatenate(parts) if parts else np.empty(0, dtype="<u4")
        if data.size != end-offset:
            raise ValueError("Streaming block coverage gap")
        return data, self.position_at(end)

    def prefetch_window(self, offset, end):
        following = [b for b in self.blocks if offset <= b["start"] < end]
        for block in following[:self.plan["limits"]["prefetch_blocks"]]:
            self._request(block)

    def position_for_global_sequence(self, sequence_index, *, sequence_length):
        if sequence_length != self.spec["sequence_length"] or not 0 <= sequence_index*sequence_length <= self.token_count:
            raise ValueError("Streaming position outside source")
        return TokenPosition(0, sequence_index*sequence_length)

    def _request(self, block):
        if self.plan["format"] == ALTERNATING_FORMAT:
            from lm_cl.data.alternating import request_allowed
            if not request_allowed(self.root, self.plan, block["ordinal"]):
                return
        path = self.root / "requests" / f"{block['ordinal']:08d}.json"
        with lock(self.root / "requests.lock"):
            if not path.exists():
                atomic_write_json(path, {"ordinal": block["ordinal"], "recipe_sha256": self.plan["sha256"]})

    def _read_block(self, block):
        start = time.monotonic()
        deadline = start + self.plan["limits"]["wait_seconds"]
        self._request(block)
        while True:
            if (self.root / "cache").is_symlink():
                raise ValueError("Symlinked disposable cache is forbidden")
            with lock(self.root / "cache.lock"):
                item = receipt(self.root, block["ordinal"])
                path = block_path(self.root, self.plan, block["ordinal"])
                if path.parent.is_symlink():
                    raise ValueError("Symlinked block directory is forbidden")
                if path.is_symlink():
                    raise ValueError("Symlinked block is forbidden")
                if item is not None and path.is_file():
                    content = path.read_bytes()
                    if len(content) != block["count"]*4 or hashlib.sha256(content).hexdigest() != item[0]["data_sha256"]:
                        raise ValueError("Corrupt published streaming block")
                    os.utime(path, None)
                    self.waited_seconds += time.monotonic()-start
                    return np.frombuffer(content, dtype="<u4")
            if (self.root / "producer-error.json").exists():
                raise RuntimeError("Streaming producer failed; see producer-error.json and resume instructions")
            ready = self.root / "producer-ready.json"
            if ready.exists():
                try:
                    os.kill(json.loads(ready.read_text())["pid"], 0)
                except ProcessLookupError as exc:
                    raise RuntimeError("Streaming producer exited; restart the same launcher with --resume auto") from exc
            if time.monotonic() >= deadline:
                raise TimeoutError("Streaming data temporarily unavailable; no short batch/EOF was emitted")
            self._request(block)  # service may have finished just before eviction
            time.sleep(.05)

    def iter_batches(self, *, sequence_length, global_sequences_per_batch, start=None, sequence_prefix_count=None):
        if sequence_length != self.spec["sequence_length"] or global_sequences_per_batch <= 0:
            raise ValueError("Streaming reader dimensions differ")
        start = start or TokenPosition(0, 0)
        offset = start.token_offset
        end = self.token_count if sequence_prefix_count is None else sequence_prefix_count*sequence_length
        if start.shard_index != 0 or offset % sequence_length or not 0 <= offset <= end <= self.token_count:
            raise ValueError("Invalid streaming sequence window")
        while offset < end:
            count = min(global_sequences_per_batch*sequence_length, end-offset)
            values, _ = self.read_tokens(count, start=TokenPosition(0, offset))
            self.prefetch_window(offset+count, end)
            data = values.reshape(-1, sequence_length).astype(np.int64)
            yield TokenBatch(data, data.copy(), len(data)*(sequence_length-1), TokenPosition(0,offset), TokenPosition(0,offset+count))
            offset += count


def checkpoint_fields(identity, position):
    if not identity or identity.get("kind") != "streaming_packed":
        return {}
    return {"streaming_prefix_proof": prefix_proof(identity, position["token_offset"])}


def validate_checkpoint_prefix(payload, *, probe=False):
    identity = payload.get("training_source_identity" if probe else "source_identity")
    if identity and identity.get("kind") == "streaming_packed":
        state = payload["probe_state" if probe else "trainer_state"]
        proof = payload.get("streaming_prefix_proof")
        if not proof or proof["consumed_tokens"] != state["source_position"]["token_offset"]:
            raise ValueError("Streaming checkpoint proof/source position differs")
        from lm_cl.config.continual_yaml import _data_pipeline_from_mapping
        config = payload["resolved_config"]
        configured = config["train_source"] if probe else config["tasks"][state["current_task_index"]]["train_source"]
        if configured["kind"] != "streaming_packed":
            raise ValueError("Streaming checkpoint has another configured source kind")
        actual = source_from_pipeline(_data_pipeline_from_mapping(configured["packed"])).identity
        if actual != {k:identity[k] for k in actual}:
            raise ValueError("Checkpoint source differs from configured streaming reference")
        validate_prefix(identity, proof)
        if probe:
            validation = payload["validation_source_identity"]
            validation_proof = payload.get("streaming_validation_prefix_proof")
            validate_prefix(validation, validation_proof)
