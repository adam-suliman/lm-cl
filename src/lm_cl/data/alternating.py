"""Compact receipts, pinned reuse data and checkpoint-acknowledged queues.

Only newly owned token-cache files are released. Checkpoints and permanent
scientific records are never deleted by this module.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import zlib

import numpy as np

from lm_cl.data.incremental import Recipe, digest, json_bytes
from lm_cl.data.storage import atomic_write_json
from lm_cl.data.streaming import (
    ALTERNATING_FORMAT, block_path, connect, load_plan, lock, receipt,
)
from lm_cl.data.streaming_producer import Engine, HashIndex, ProductionProducer


def make_turns(plan, global_batch, consumers):
    """Split within language tasks; only the task tail may be a short batch."""
    task = plan["task_budget"]["effective_input_tokens"]
    batch_tokens = global_batch * next(iter(plan["streams"].values()))["sequence_length"]
    chunk_tokens = plan["limits"]["chunk_batches"] * batch_tokens
    train = [(i, b) for i, b in enumerate(plan["blocks"])
             if plan["purposes"][b["stream"]] == "continual_train"]
    turns, snapshots, total, batches = [], set(), 0, 0
    index = 0
    while index < len(train):
        start_index = index
        first = train[index][1]
        start = first["start"]
        end = start + task
        while index < len(train) and train[index][1]["stream"] == first["stream"] and train[index][1]["start"] < end:
            index += 1
        blocks = train[start_index:index]
        if sum(b["count"] for _, b in blocks) != task:
            raise ValueError("Alternating task coverage differs")
        task_index = total // task
        for offset in range(start, end, chunk_tokens):
            stop = min(end, offset + chunk_tokens)
            last = next(i for i, b in blocks if b["start"] < stop <= b["start"] + b["count"])
            snapshots.add(last)
            allowed = last
            if stop == end:
                while allowed + 1 < len(plan["blocks"]) and plan["purposes"][plan["blocks"][allowed + 1]["stream"]] != "continual_train":
                    allowed += 1
                    if (allowed + 1 == len(plan["blocks"])
                            or plan["blocks"][allowed + 1]["stream"] != plan["blocks"][allowed]["stream"]):
                        snapshots.add(allowed)
            batches += (stop - offset + batch_tokens - 1) // batch_tokens
            turns.append({"index": len(turns), "task_index": task_index,
                          "end_tokens": total + stop - start, "end_batches": batches,
                          "task_boundary": stop == end, "last_ordinal": allowed})
        total += task
    return {"global_batch_sequences": global_batch, "consumers": consumers,
            "turns": turns, "snapshot_ordinals": sorted(snapshots)}


def configure_plan(plan, config):
    from lm_cl.launcher.schema import PUBLIC_MODEL_VARIANTS
    plan["format"] = ALTERNATING_FORMAT
    consumers = {f"{model}-seed-{seed}": {"variant": PUBLIC_MODEL_VARIANTS[model], "seed": seed,
                 "output_dir": str((Path(config.experiment.output_root) / config.experiment.name / model / f"seed-{seed}").resolve())}
                 for seed in config.experiment.seeds for model in config.experiment.models}
    plan["alternation"] = make_turns(plan, config.training.global_batch_sequences, consumers)


def validate_alternation(plan):
    value = plan.get("alternation", {})
    consumers = value.get("consumers", {})
    if not consumers or type(value.get("global_batch_sequences")) is not int or value["global_batch_sequences"] <= 0:
        raise ValueError("Invalid alternating consumer/batch contract")
    from lm_cl.launcher.schema import PUBLIC_MODEL_VARIANTS
    for key, item in consumers.items():
        if (set(item) != {"variant", "seed", "output_dir"} or type(item["seed"]) is not int
                or not Path(item["output_dir"]).is_absolute()
                or not any(key == f"{name}-seed-{item['seed']}" and item["variant"] == variant
                           for name, variant in PUBLIC_MODEL_VARIANTS.items())):
            raise ValueError("Invalid alternating consumer identity")
    if value != make_turns(plan, value["global_batch_sequences"], consumers):
        raise ValueError("Alternating turn schedule differs")


def initialize_tables(root, plan, db):
    (root / "pinned").mkdir()
    db.executescript("""
        CREATE TABLE heads (stream TEXT PRIMARY KEY, ordinal INTEGER NOT NULL, state BLOB NOT NULL);
        CREATE TABLE snapshots (stream TEXT NOT NULL, ordinal INTEGER NOT NULL, state BLOB NOT NULL,
                                PRIMARY KEY(stream, ordinal)) WITHOUT ROWID;
        DROP TABLE content_hashes;
        DROP TABLE token_hashes;
        CREATE TABLE content_hashes (hash BLOB PRIMARY KEY, ordinal INTEGER NOT NULL) WITHOUT ROWID;
        CREATE TABLE token_hashes (hash BLOB PRIMARY KEY, ordinal INTEGER NOT NULL) WITHOUT ROWID;
    """)
    atomic_write_json(root / "queue.json", {"recipe_sha256": plan["sha256"], "turn": 0,
                                            "released_tokens": 0, "consumers": {}})


def control(root, plan):
    value = json.loads((Path(root) / "queue.json").read_text())
    if (value.get("recipe_sha256") != plan["sha256"] or type(value.get("turn")) is not int
            or not 0 <= value["turn"] <= len(plan["alternation"]["turns"])
            or not set(value.get("consumers", {})).issubset(plan["alternation"]["consumers"])):
        raise ValueError("Invalid alternating queue state")
    expected = 0 if value["turn"] == 0 else plan["alternation"]["turns"][value["turn"] - 1]["end_tokens"]
    if value.get("released_tokens") != expected:
        raise ValueError("Queue release position differs from turn")
    return value


def request_allowed(root, plan, ordinal):
    state = control(root, plan)
    turns = plan["alternation"]["turns"]
    return state["turn"] == len(turns) or ordinal <= turns[state["turn"]]["last_ordinal"]


def _checked_checkpoint(root, plan, consumer, path, sha):
    from lm_cl.training.checkpoint import load_checkpoint, sha256_file
    from lm_cl.data.streaming import validate_checkpoint_prefix
    path = Path(path).resolve()
    if sha256_file(path) != sha:
        raise ValueError("Consumer checkpoint checksum mismatch")
    payload = load_checkpoint(path)
    identity = payload["source_identity"]
    expected = plan["alternation"]["consumers"][consumer]
    cfg = payload["resolved_config"]
    if (identity.get("root") != str(Path(root).resolve()) or identity.get("recipe_sha256") != plan["sha256"]
            or cfg["runtime"]["seed"] != expected["seed"] or cfg["variant"]["name"] != expected["variant"]
            or cfg["runtime"]["output_dir"] != expected["output_dir"]
            or path.parent != Path(expected["output_dir"]) / "checkpoints"):
        raise ValueError("Consumer checkpoint belongs to another job/study")
    validate_checkpoint_prefix(payload)
    return payload


def acknowledge(root, consumer, path, sha):
    """Only a verified on-disk checkpoint can move a consumer past data."""
    root = Path(root); plan = load_plan(root)
    if consumer not in plan["alternation"]["consumers"]:
        raise ValueError("Unknown alternating consumer")
    payload = _checked_checkpoint(root, plan, consumer, path, sha)
    state = payload["trainer_state"]
    with lock(root / "queue.lock"):
        value = control(root, plan)
        turn = plan["alternation"]["turns"][value["turn"]]
        if (state["global_input_tokens"] != turn["end_tokens"] or state["global_logical_batches"] != turn["end_batches"]
                or (turn["task_boundary"] and (state["phase"] != "task_boundary" or state["next_task_index"] != turn["task_index"] + 1))):
            raise ValueError("Consumer checkpoint does not cover the open turn")
        value["consumers"][consumer] = {"checkpoint": str(Path(path).resolve()), "sha256": sha,
                                         "end_tokens": turn["end_tokens"]}
        atomic_write_json(root / "queue.json", value)


def _release_files(root, plan, through):
    """Idempotent: a crash midway leaves extra old data, never missing needed data."""
    with lock(root / "cache.lock"):
        directory = root / "cache"
        if directory.is_symlink():
            raise ValueError("Symlinked queue is forbidden")
        ends, total = {}, 0
        for i, block in enumerate(plan["blocks"]):
            if plan["purposes"][block["stream"]] == "continual_train":
                total += block["count"]; ends[i] = total
        for path in directory.iterdir():
            if (path.is_symlink() or not path.is_file() or path.suffix not in {".bin", ".tmp"}
                    or len(path.stem) != 8 or not path.stem.isdigit() or int(path.stem) not in ends):
                raise ValueError("Unrecognized alternating queue entry")
            i = int(path.stem)
            if ends[i] <= through:
                record = receipt(root, i)
                if record is None:
                    raise ValueError("Queue release lacks immutable receipt")
                if path.suffix == ".bin" and hashlib.sha256(path.read_bytes()).hexdigest() != record[0]["data_sha256"]:
                    raise ValueError("Corrupt queue block at release")
                path.unlink()


def advance(root):
    root = Path(root); plan = load_plan(root)
    with lock(root / "queue.lock"):
        value = control(root, plan)
        turns = plan["alternation"]["turns"]
        if value["turn"] == len(turns):
            _release_files(root, plan, value["released_tokens"])
            return True
        turn = turns[value["turn"]]
        for consumer in plan["alternation"]["consumers"]:
            ack = value["consumers"].get(consumer)
            if ack is None or ack["end_tokens"] != turn["end_tokens"]:
                return False
            payload = _checked_checkpoint(root, plan, consumer, ack["checkpoint"], ack["sha256"])
            if payload["trainer_state"]["global_input_tokens"] != turn["end_tokens"]:
                raise ValueError("Acknowledged progress changed")
        # Publish the verified recovery watermark before releasing any files.
        value["released_tokens"] = turn["end_tokens"]
        value["turn"] += 1
        atomic_write_json(root / "queue.json", value)
        _release_files(root, plan, value["released_tokens"])
        return True


def recover_release(root, plan=None):
    """Complete a release interrupted after publishing its recovery watermark."""
    root = Path(root); plan = plan or load_plan(root)
    with lock(root / "queue.lock"):
        value = control(root, plan)
        if value["released_tokens"]:
            for consumer in plan["alternation"]["consumers"]:
                ack = value["consumers"].get(consumer)
                if ack is None or ack["end_tokens"] < value["released_tokens"]:
                    raise ValueError("Release watermark lacks consumer recovery coverage")
                payload = _checked_checkpoint(root, plan, consumer, ack["checkpoint"], ack["sha256"])
                if payload["trainer_state"]["global_input_tokens"] != ack["end_tokens"]:
                    raise ValueError("Recovery acknowledgment changed")
        _release_files(root, plan, value["released_tokens"])
        return value["released_tokens"]


def publish(root, plan, ordinal, content):
    path = block_path(root, plan, ordinal)
    # Same lock order as release. A stale request cannot republish old queue
    # bytes after the controller has advanced the durable recovery watermark.
    with lock(root / "queue.lock"), lock(root / "cache.lock"):
        if path.parent.name == "cache":
            end = sum(block["count"] for block in plan["blocks"][:ordinal + 1]
                      if plan["purposes"][block["stream"]] == "continual_train")
            if end <= control(root, plan)["released_tokens"]:
                return
        if path.parent.is_symlink() or path.is_symlink():
            raise ValueError("Symlinked alternating data is forbidden")
        if path.exists():
            if path.read_bytes() != content:
                raise ValueError("Published block bytes changed")
            return
        temp = path.with_suffix(".tmp")
        if temp.is_symlink():
            raise ValueError("Symlinked temporary block")
        if path.parent.name == "cache":
            occupied = 0
            for entry in path.parent.iterdir():
                if (entry.is_symlink() or not entry.is_file() or entry.suffix not in {".bin", ".tmp"}
                        or not entry.stem.isdigit() or len(entry.stem) != 8
                        or not 0 <= int(entry.stem) < len(plan["blocks"])
                        or plan["purposes"][plan["blocks"][int(entry.stem)]["stream"]] != "continual_train"):
                    raise ValueError("Unrecognized alternating queue entry")
                if entry != temp: occupied += entry.stat().st_size
            if occupied + len(content) > plan["limits"]["token_cache_bytes"]:
                raise RuntimeError("Alternating queue full; cannot evict unacknowledged data")
        if shutil.disk_usage(root).free < len(content) + plan["limits"]["minimum_free_bytes"]:
            raise RuntimeError("Streaming free-space floor reached")
        with temp.open("wb") as handle:
            handle.write(content); handle.flush(); os.fsync(handle.fileno())
        os.replace(temp, path)
        fd = os.open(path.parent, os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)


class BinaryHashIndex(HashIndex):
    def __contains__(self, value):
        return bytes.fromhex(value) in self.added or self.db.execute(
            f"SELECT 1 FROM {self.table} WHERE hash=? AND ordinal<?", (bytes.fromhex(value), self.ordinal)).fetchone() is not None

    def add(self, value): self.added.add(bytes.fromhex(value))


class AlternatingProducer(ProductionProducer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.recovered_through = None
        self.train_ends = {}
        total = 0
        for i, block in enumerate(self.plan["blocks"]):
            if self.plan["purposes"][block["stream"]] == "continual_train":
                total += block["count"]
                self.train_ends[i] = total

    def _restore(self, db, stream, ordinal):
        previous = db.execute("SELECT ordinal FROM receipts WHERE stream=? AND ordinal<? ORDER BY ordinal DESC LIMIT 1",
                              (stream, ordinal)).fetchone()
        if previous is None: return None, -1
        row = db.execute("SELECT ordinal,state FROM heads WHERE stream=? AND ordinal=?", (stream, previous[0])).fetchone()
        if row is None:
            row = db.execute("SELECT ordinal,state FROM snapshots WHERE stream=? AND ordinal<? ORDER BY ordinal DESC LIMIT 1",
                             (stream, ordinal)).fetchone()
        if row is None: return None, -1
        state = json.loads(zlib.decompress(row[1]))
        if digest(state) != receipt(self.root, row[0])[0]["producer_state_sha256"]:
            raise ValueError("Selected recovery state checksum differs")
        return state, row[0]

    def generate(self, ordinal):
        stream = self.plan["blocks"][ordinal]["stream"]
        with connect(self.root) as db:
            state, previous = self._restore(db, stream, ordinal)
        for i in range(previous + 1, ordinal + 1):
            if self.plan["blocks"][i]["stream"] == stream:
                state = self._produce(i, state, publish_bytes=i == ordinal)

    def _produce(self, ordinal, state, *, publish_bytes):
        block = self.plan["blocks"][ordinal]; spec = self.plan["streams"][block["stream"]]
        self.budget.check(reserve_bytes=block["count"] * 4)
        if self.tokenizer is None:
            from lm_cl.data.tokenizer import load_verified_tokenizer
            from lm_cl.config.data_schema import TokenizerReference
            self.tokenizer, _ = load_verified_tokenizer(TokenizerReference(**self.plan["tokenizer_reference"]))
        started = time.monotonic()
        with connect(self.root) as db:
            old = receipt(self.root, ordinal)
            engine = Engine(Recipe(**spec), state, self.source(spec["language"]), self.tokenizer, db, ordinal)
            engine.max_document_bytes = self.plan["limits"]["max_document_bytes"]
            engine.content_hashes = BinaryHashIndex(db, "content_hashes", ordinal)
            engine.token_hashes = BinaryHashIndex(db, "token_hashes", ordinal)
            values, boundaries = [], []
            while len(values) < block["count"]:
                if time.monotonic() - started > self.plan["limits"]["max_block_seconds"]:
                    raise TimeoutError("Bounded block preparation deadline reached")
                if engine.state["pending_offset"] == len(engine.state["pending_tokens"]):
                    boundaries.append(engine._accept())
                pending, offset = engine.state["pending_tokens"], engine.state["pending_offset"]
                take = min(len(pending) - offset, block["count"] - len(values))
                values.extend(pending[offset:offset + take]); engine.state["pending_offset"] += take
            engine.state["rng"] = engine.rng.getstate()
            content = np.asarray(values, dtype="<u4").tobytes()
            prior = receipt(self.root, ordinal - 1) if ordinal else None
            record = {"ordinal": ordinal, **block, "recipe_sha256": self.plan["sha256"],
                      "previous": prior[1] if prior else self.plan["sha256"],
                      "source_index_sha256": digest(getattr(engine.source, "identity", spec["source_identity"])),
                      "data_sha256": hashlib.sha256(content).hexdigest(),
                      "boundaries_sha256": digest(boundaries), "documents_started": len(boundaries),
                      "producer_state_sha256": digest(engine.state)}
            if old:
                if old[1] != digest(record):
                    raise ValueError("Regenerated compact receipt differs")
            else:
                if db.execute("SELECT COUNT(*) FROM receipts").fetchone()[0] != ordinal:
                    raise ValueError("Out-of-order compact publication")
                db.execute("INSERT INTO receipts VALUES(?,?,?,?,?)", (ordinal, block["stream"], "", json_bytes(record).decode(), digest(record)))
                for table, index in [("content_hashes", engine.content_hashes), ("token_hashes", engine.token_hashes)]:
                    db.executemany(f"INSERT INTO {table} VALUES(?,?)", [(h, ordinal) for h in index.added])
                packed = zlib.compress(json_bytes(engine.state))
                db.execute("INSERT OR REPLACE INTO heads VALUES(?,?,?)", (block["stream"], ordinal, packed))
                if ordinal in self.plan["alternation"]["snapshot_ordinals"]:
                    db.execute("INSERT INTO snapshots VALUES(?,?,?)", (block["stream"], ordinal, packed))
                db.commit()
            self._complete_stream(db, block["stream"], ordinal)
        if publish_bytes: publish(self.root, self.plan, ordinal, content)
        atomic_write_json(self.root / "producer-status.json", {"status": "running", "ordinal": ordinal,
                          "stream": block["stream"], "regenerated": old is not None,
                          "seconds": time.monotonic() - started,
                          "network_bytes_this_process": self.remote.network_bytes if self.remote else 0})
        return engine.state

    def ensure(self, ordinal):
        if not 0 <= ordinal < len(self.plan["blocks"]): raise ValueError("Invalid requested block")
        value = control(self.root, self.plan)
        if self.recovered_through != value["released_tokens"]:
            self.recovered_through = recover_release(self.root, self.plan)
        # Requests left by an earlier turn are operational messages, not readers
        # entitled to roll an acknowledged trajectory backwards.
        if self.train_ends.get(ordinal, float("inf")) <= value["released_tokens"]:
            return
        if not request_allowed(self.root, self.plan, ordinal): return
        if receipt(self.root, ordinal) is None:
            with connect(self.root) as db: start = db.execute("SELECT COUNT(*) FROM receipts").fetchone()[0]
            for i in range(start, ordinal + 1): self.generate(i)
        elif not block_path(self.root, self.plan, ordinal).exists():
            self.generate(ordinal)
