"""Study-scoped production producer. No GPU work; all publication is serial."""
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
import hashlib
import io
import json
import os
from pathlib import Path
import random
import shutil
import time
from urllib.parse import quote

import numpy as np

from lm_cl.data.incremental import Producer, Recipe, FORMAT as RECIPE_FORMAT, _tuples, digest, json_bytes
from lm_cl.data.incremental_remote import RemoteAccess, ParquetRows, REVISION, MAPPINGS
from lm_cl.data.storage import atomic_write_json
from lm_cl.data.streaming import load_plan, lock, connect, receipt, publish_cache, cache_path


class Budget:
    def __init__(self, root, plan):
        self.root = self.work = self.report = Path(root)
        self.plan = plan
        self.v = plan["limits"]
        self.requests = 0

    def check(self, *, reserve_bytes=0):
        if shutil.disk_usage(self.root).free < self.v["minimum_free_bytes"] + reserve_bytes:
            raise RuntimeError("Streaming free-space floor reached")
        metadata = 0
        for path in self.root.rglob("*"):
            if any(name in path.relative_to(self.root).parts for name in ("cache", "pinned")):
                continue
            try:
                if path.is_file() and not path.is_symlink():
                    metadata += path.stat().st_size
            except FileNotFoundError:
                # Concurrent atomic request publication may retire a temp name.
                if "requests" not in path.relative_to(self.root).parts:
                    raise
        if metadata > self.v["metadata_bytes"]:
            raise RuntimeError("Streaming metadata cap reached; preserve receipts and review the frozen study limits")

    def network_attempt(self):
        self.check()
        self.requests += 1
        if self.requests > self.v["max_network_requests"]:
            raise RuntimeError("Streaming network request cap reached")
        return self.requests


class MemoryRangeFile(io.RawIOBase):
    """Bounded HTTP ranges; no persistent copy of remote Parquet files."""
    def __init__(self, remote, identity, entry):
        self.remote, self.entry = remote, entry
        self.position, self.size = 0, entry["size"]
        self.url = f"https://huggingface.co/datasets/uonlp/CulturaX/resolve/{identity['revision']}/{quote(entry['path'])}"

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.position
    def seek(self, offset, whence=0):
        value = offset if whence == 0 else self.position+offset if whence == 1 else self.size+offset
        if not 0 <= value <= self.size: raise ValueError("Invalid Parquet seek")
        self.position = value
        return value
    def read(self, count=-1):
        end = self.size if count < 0 else min(self.position+count, self.size)
        if end == self.position: return b""
        length = end-self.position
        if length > self.remote.limits.v["max_row_group_bytes"]:
            raise ValueError("Parquet range exceeds configured cap")
        value = self.remote.fetch(self.url, maximum_bytes=length, byte_range=(self.position,end))
        self.position = end
        return value
    def readinto(self, buffer):
        value = self.read(len(buffer)); buffer[:len(value)] = value; return len(value)


class Rows(ParquetRows):
    def _open_next(self):
        import pyarrow.parquet as pq
        i = len(self.files)
        if i == len(self.identity["files"]): return False
        file = pq.ParquetFile(MemoryRangeFile(self.remote,self.identity,self.identity["files"][i]), pre_buffer=False)
        if not {"text","url"}.issubset(file.schema.names): raise ValueError("Pinned Parquet schema differs")
        self.files.append(file)
        for group in range(file.num_row_groups):
            count = file.metadata.row_group(group).num_rows
            self.groups.append((self.indexed_rows,self.indexed_rows+count,i,group))
            self.indexed_rows += count
        return True


class HashIndex:
    def __init__(self, db, table, ordinal):
        self.db,self.table,self.ordinal = db,table,ordinal
        self.added = set()
    def __contains__(self, value):
        return value in self.added or self.db.execute(
            f"SELECT 1 FROM {self.table} WHERE hash=? AND ordinal<?", (value,self.ordinal)).fetchone() is not None
    def add(self, value): self.added.add(value)


class Engine(Producer):
    allow_empty_final_document = True
    # Reuse the independently tested shuffle/selection/EOS packing algorithm,
    # replacing the demo's unbounded in-memory registry and all-block reader.
    def __init__(self, recipe, state, source, tokenizer, db, ordinal):
        self.recipe,self.source,self.tokenizer = recipe,source,tokenizer
        self.rng = random.Random(recipe.document_order_seed)
        self.state = state or {"source_cursor":0,"buffer":[],"initialized":False,"draining":False,
            "pending_tokens":[],"pending_offset":0,"assigned_tokens":0,"accepted_documents":0,
            "rejections":{},"selected_documents":0,"rng":self.rng.getstate()}
        self.rng.setstate(_tuples(self.state["rng"]))
        self.max_document_bytes = 4 * 1024**2
        self.content_hashes = HashIndex(db,"content_hashes",ordinal)
        self.token_hashes = HashIndex(db,"token_hashes",ordinal)

    def _valid_next(self):
        value = super()._valid_next()
        if value is not None:
            text = self.source.row(value)[self.recipe.text_field]
            if len(text.encode("utf-8")) > self.max_document_bytes:
                raise ValueError("Document exceeds configured UTF-8 cap; no silent truncation/filtering")
        return value


class ProductionProducer:
    def __init__(self, root, *, source_factory=None, tokenizer=None):
        self.root = Path(root).resolve()
        self.plan = load_plan(root)
        self.budget = Budget(root,self.plan)
        self.sources = OrderedDict()
        self.source_factory = source_factory
        self.tokenizer = tokenizer
        self.remote = None

    def source(self, language):
        if language in self.sources:
            self.sources.move_to_end(language)
            return self._activate(language)
        if self.source_factory:
            source = self.source_factory(language)
        else:
            if self.remote is None: self.remote = RemoteAccess(self.budget)
            directory = self.root / "source-index"
            directory.mkdir(exist_ok=True)
            path = directory / (language+".json")
            if path.exists():
                envelope = json.loads(path.read_text()); identity = envelope["identity"]
                if digest(identity) != envelope["sha256"]: raise ValueError("Source index checksum differs")
            else:
                files = []
                url = f"https://huggingface.co/api/datasets/uonlp/CulturaX/tree/{REVISION}/{MAPPINGS[language]}?recursive=false&expand=false&limit=1000"
                seen = set()
                while url:
                    if url in seen or len(seen) > self.plan["limits"]["max_source_files"]:
                        raise ValueError("Invalid source-index pagination")
                    seen.add(url)
                    items = json.loads(self.remote.fetch(url, maximum_bytes=4*1024**2))
                    for item in items:
                        if item.get("type") == "file" and item["path"].endswith(".parquet"):
                            files.append({"path":item["path"], "size":item["size"], "oid":item.get("oid")})
                        if len(files) > self.plan["limits"]["max_source_files"]:
                            raise ValueError("Pinned source file count exceeds configured cap")
                    url = self.remote.last_links.get("next", {}).get("url")
                    if url and not url.startswith(f"https://huggingface.co/api/datasets/uonlp/CulturaX/tree/{REVISION}/"):
                        raise ValueError("Unexpected metadata pagination destination")
                if not files: raise ValueError("Pinned source has no Parquet files")
                identity = {"repo_id":"uonlp/CulturaX","revision":REVISION,
                    "language_config":MAPPINGS[language],"columns":["text","url"],
                    "files":sorted(files,key=lambda x:x["path"])}
                atomic_write_json(path,{"identity":identity,"sha256":digest(identity)})
            source = Rows(identity,self.remote)
        self.sources[language] = source
        return self._activate(language)

    def _activate(self, language):
        # Retain small file/footer indices across lanes, but only one language's
        # decoded row groups. This avoids replaying all earlier footer requests
        # whenever retention/probe work switches languages.
        for name, source in self.sources.items():
            if name != language and hasattr(source, "loaded_groups"):
                source.loaded_groups.clear()
                source.loaded_bytes = 0
        return self.sources[language]

    def generate(self, ordinal):
        block = self.plan["blocks"][ordinal]
        spec = self.plan["streams"][block["stream"]]
        self.budget.check(reserve_bytes=block["count"]*4)
        if self.tokenizer is None:
            from lm_cl.data.tokenizer import load_verified_tokenizer
            from lm_cl.config.data_schema import TokenizerReference
            self.tokenizer,_ = load_verified_tokenizer(TokenizerReference(**self.plan["tokenizer_reference"]))
        with connect(self.root) as db:
            old = receipt(self.root,ordinal)
            previous = db.execute("SELECT state FROM receipts WHERE stream=? AND ordinal<? ORDER BY ordinal DESC LIMIT 1",
                                  (block["stream"],ordinal)).fetchone()
            state = None if previous is None else json.loads(previous[0])
            prior_record = db.execute("SELECT ordinal FROM receipts WHERE stream=? AND ordinal<? ORDER BY ordinal DESC LIMIT 1",
                                      (block["stream"],ordinal)).fetchone()
            if prior_record and state != receipt(self.root, prior_record[0])[0]["producer_state"]:
                raise ValueError("Producer state differs from checked receipt")
            recipe = Recipe(**spec)
            recipe.validate()
            engine = Engine(recipe,state,self.source(recipe.language),self.tokenizer,db,ordinal)
            engine.max_document_bytes = self.plan["limits"]["max_document_bytes"]
            values,boundaries = [],[]
            started = time.monotonic()
            while len(values)<block["count"]:
                if time.monotonic()-started > self.plan["limits"]["max_block_seconds"]:
                    raise TimeoutError("Bounded block preparation deadline reached")
                if engine.state["pending_offset"] == len(engine.state["pending_tokens"]):
                    boundaries.append(engine._accept())
                pending,offset = engine.state["pending_tokens"],engine.state["pending_offset"]
                take = min(len(pending)-offset,block["count"]-len(values))
                values.extend(pending[offset:offset+take]);engine.state["pending_offset"] += take
            engine.state["rng"] = engine.rng.getstate()
            content = np.asarray(values,dtype="<u4").tobytes()
            prior = receipt(self.root,ordinal-1) if ordinal else None
            source_identity = getattr(engine.source, "identity", spec["source_identity"])
            record = {"source_index_sha256":digest(source_identity), "recipe_sha256":self.plan["sha256"],"ordinal":ordinal,**block,
                "previous":prior[1] if prior else self.plan["sha256"],
                "data_sha256":hashlib.sha256(content).hexdigest(),"boundaries":boundaries,
                "producer_state":engine.state}
            if old:
                if old[1] != digest(record): raise ValueError("Regenerated block differs from immutable receipt")
            else:
                n = db.execute("SELECT COUNT(*) FROM receipts").fetchone()[0]
                if n != ordinal: raise ValueError("Out-of-order streaming publication")
                db.execute("INSERT INTO receipts VALUES(?,?,?,?,?)",(ordinal,block["stream"],
                    json.dumps(engine.state),json_bytes(record).decode(),digest(record)))
                for table,index in [("content_hashes",engine.content_hashes),("token_hashes",engine.token_hashes)]:
                    db.executemany(f"INSERT INTO {table} VALUES(?,?)",[(h,ordinal) for h in index.added])
                db.commit()
            self._complete_stream(db, block["stream"], ordinal)
            publish_cache(self.root,self.plan,ordinal,content)
            atomic_write_json(self.root / "producer-status.json", {"status":"running","ordinal":ordinal,
                "stream":block["stream"],"input_tokens":block["count"],"seconds":time.monotonic()-started,
                "regenerated":old is not None,"network_bytes_this_process":self.remote.network_bytes if self.remote else 0})

    def _complete_stream(self, db, stream, ordinal):
        expected = [i for i,b in enumerate(self.plan["blocks"]) if b["stream"] == stream]
        if ordinal != expected[-1]:
            return
        rows = db.execute("SELECT ordinal,sha256 FROM receipts WHERE stream=? ORDER BY ordinal", (stream,)).fetchall()
        if [r[0] for r in rows] != expected:
            return
        directory = self.root / "completed"
        directory.mkdir(exist_ok=True)
        value = {"format":self.plan["format"], "recipe_sha256":self.plan["sha256"], "stream":stream,
                 "token_count":self.plan["streams"][stream]["output_tokens"],
                 "ordered_receipts_sha256":digest(rows), "blocks":len(rows)}
        path = directory / (stream+".json")
        if path.exists():
            if json.loads(path.read_text()) != value: raise ValueError("Completed stream receipt differs")
        else:
            atomic_write_json(path,value)
        if db.execute("SELECT COUNT(*) FROM receipts").fetchone()[0] == len(self.plan["blocks"]):
            final = {"format":self.plan["format"], "recipe_sha256":self.plan["sha256"],
                     "last_receipt_sha256":receipt(self.root,len(self.plan["blocks"])-1)[1],
                     "blocks":len(self.plan["blocks"]), "status":"all_blocks_published_cache_may_be_evicted"}
            path = self.root / "complete.json"
            if path.exists() and json.loads(path.read_text()) != final: raise ValueError("Study completion differs")
            if not path.exists(): atomic_write_json(path, final)

    def ensure(self, ordinal):
        if not 0 <= ordinal < len(self.plan["blocks"]): raise ValueError("Invalid requested block")
        if receipt(self.root,ordinal) is None:
            with connect(self.root) as db: start = db.execute("SELECT COUNT(*) FROM receipts").fetchone()[0]
            for i in range(start,ordinal+1): self.generate(i)
        elif not cache_path(self.root,ordinal).exists():
            self.generate(ordinal)

    def serve(self):
        with lock(self.root / "producer.lock",nonblocking=True):
            error = self.root / "producer-error.json"
            if error.exists():
                # Keep each failed producer record; only this operational pointer moves.
                os.replace(error,self.root/f"producer-error-{time.time_ns()}.json")
            atomic_write_json(self.root / "producer-ready.json", {"pid":os.getpid(), "recipe_sha256":self.plan["sha256"]})
            try:
                while True:
                    requests = sorted((self.root / "requests").glob("*.json"))
                    for path in requests:
                        value = json.loads(path.read_text())
                        if value["recipe_sha256"] != self.plan["sha256"]: raise ValueError("Request recipe differs")
                        import signal
                        def expired(signum, frame):
                            raise TimeoutError("Streaming block/request deadline reached")
                        signal.signal(signal.SIGALRM, expired)
                        signal.alarm(self.plan["limits"]["max_block_seconds"])
                        try:
                            self.ensure(value["ordinal"])
                        finally:
                            signal.alarm(0)
                        path.unlink()  # request queue only, never a receipt/artifact
                    time.sleep(.05)
            except BaseException as exc:
                atomic_write_json(error,{"status":"failed","error_type":type(exc).__name__,
                    "message":"Producer failed; no unverified block was consumed. Inspect configuration, source access and resource caps; restart the same recipe."})
                raise


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("root");args=p.parse_args()
    from lm_cl.data.streaming import ALTERNATING_FORMAT
    if load_plan(args.root)["format"] == ALTERNATING_FORMAT:
        from lm_cl.data.alternating import AlternatingProducer
        AlternatingProducer(args.root).serve()
    else:
        ProductionProducer(args.root).serve()


if __name__ == "__main__": main()
