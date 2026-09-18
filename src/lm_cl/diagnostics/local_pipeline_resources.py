"""Resource admission and ownership for the bounded local demonstration."""
from __future__ import annotations

import contextlib
import fcntl
import json
import os
import shutil
import time
from pathlib import Path

from lm_cl.data.storage import atomic_write_json


def allocated_bytes(root: Path) -> int:
    seen = set()
    total = 0
    for parent, dirs, files in os.walk(root, followlinks=False):
        for name in files:
            path = Path(parent)/name
            if path.is_symlink():
                continue
            try:
                s = path.stat()
            except FileNotFoundError:
                # Atomic operational-record replacement may rename a temporary
                # file between walk and stat. Artifact checksum validation is a
                # separate gate; a vanished directory entry has no allocation.
                continue
            key = (s.st_dev, s.st_ino)
            if key not in seen:
                seen.add(key)
                total += s.st_blocks * 512
    return total


class Limits:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.v = json.loads(self.path.read_text())
        hard = {"max_benchmark_elapsed_seconds": 43200,
                "max_additional_allocated_bytes": 50*1024**3,
                "max_hf_cache_bytes": 20*1024**3,
                "max_unique_prepared_tokens": 250000000}
        if self.v["schema_version"] != 1:
            raise ValueError("Unknown limits version")
        for k, bound in hard.items():
            if type(self.v[k]) is not int or not 0 < self.v[k] <= bound:
                raise ValueError(f"Invalid or raised limit: {k}")
        if self.v["minimum_free_bytes"] < 100*1024**3:
            raise ValueError("Free-space floor is below authorized minimum")
        for k in ["max_input_documents_per_language", "max_network_requests", "max_network_attempts_per_request",
                  "network_timeout_seconds", "max_prepare_process_seconds", "max_train_process_seconds",
                  "max_source_cache_bytes", "max_row_group_bytes", "cpu_threads_per_trainer", "cpu_threads_producer"]:
            if type(self.v[k]) is not int or self.v[k] <= 0:
                raise ValueError(f"Invalid limit: {k}")
        self.work = Path(self.v["work_root"]).resolve()
        self.report = Path(self.v["report_root"]).resolve()
        for root in [self.work, self.report]:
            marker = json.loads((root/".local_pipeline_demo_owner.json").read_text())
            if marker["attempt"] != self.v["attempt"]:
                raise ValueError("Demo root ownership mismatch")
        (self.work/"processes").mkdir(exist_ok=True)

    def owned(self, path: str | Path) -> Path:
        path = Path(path).resolve()
        if not any(path.is_relative_to(root) for root in [self.work, self.report]):
            raise ValueError("Output is outside owned demonstration roots")
        return path

    def check(self, *, reserve_bytes=0) -> dict:
        free = shutil.disk_usage(self.work).free
        total = allocated_bytes(self.work)+allocated_bytes(self.report)
        cache = allocated_bytes(self.work/"hf-cache")
        source_cache = allocated_bytes(self.work/"source-cache")
        if free-reserve_bytes < self.v["minimum_free_bytes"]:
            raise RuntimeError("Free-space floor reached")
        if total+reserve_bytes > self.v["max_additional_allocated_bytes"]:
            raise RuntimeError("Additional allocated disk cap reached")
        if cache > self.v["max_hf_cache_bytes"] or source_cache > self.v["max_source_cache_bytes"]:
            raise RuntimeError("Owned cache cap reached")
        intervals = []
        for p in (self.work/"processes").glob("*.json"):
            v = json.loads(p.read_text())
            finished = v.get("finished_unix")
            if finished is None:
                observed = self.work/"processes"/"observed-dead"/(p.name+".json")
                if observed.exists():
                    evidence = json.loads(observed.read_text())
                    if evidence["pid"] != v["pid"] or evidence["create_time"] != v["create_time"]:
                        raise ValueError("Dead-process observation identity mismatch")
                    finished = evidence["observed_dead_unix"]
                else:
                    import psutil
                    try:
                        process = psutil.Process(v["pid"])
                        alive = process.create_time() == v["create_time"] and process.status() != psutil.STATUS_ZOMBIE
                    except psutil.NoSuchProcess:
                        alive = False
                    if not alive:
                        from lm_cl.data.incremental import immutable_write, json_bytes
                        # The actual exit time is unknown. The first verified
                        # dead observation is a conservative upper bound. Keep
                        # the original failed operational record unchanged.
                        finished = time.time()
                        evidence = {"pid":v["pid"],"create_time":v["create_time"],
                            "observed_dead_unix":finished,"original_record":str(p),
                            "reason":"PID absent, reused, or zombie; actual exit time unknown"}
                        # Concurrent observers serialize the first observation.
                        with (self.work/".process-observation.lock").open("a") as lock:
                            fcntl.flock(lock, fcntl.LOCK_EX)
                            if observed.exists():
                                finished = json.loads(observed.read_text())["observed_dead_unix"]
                            else:
                                immutable_write(observed,json_bytes(evidence))
                if finished is None:
                    finished = time.time()
            intervals.append((v["started_unix"], finished))
        end, elapsed = 0., 0.
        for a,b in sorted(intervals):
            elapsed += max(0., b-max(a,end))
            end = max(end,b)
        if elapsed >= self.v["max_benchmark_elapsed_seconds"]:
            raise RuntimeError("Aggregate elapsed benchmark budget reached")
        return {"free_bytes": free, "allocated_bytes": total, "hf_cache_bytes": cache,
                "source_cache_bytes": source_cache, "benchmark_elapsed_seconds": elapsed}

    @contextlib.contextmanager
    def allocation(self, reserve_bytes):
        with (self.work/".allocation.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            self.check(reserve_bytes=reserve_bytes)
            yield
            self.check()

    @contextlib.contextmanager
    def process(self, role: str, label: str):
        import psutil
        self.check()
        path = self.work/"processes"/f"{role}-{os.getpid()}-{time.time_ns()}.json"
        record = {"pid": os.getpid(), "create_time": psutil.Process().create_time(),
                  "role": role, "label": label, "started_unix": time.time(), "status": "running"}
        atomic_write_json(path, record)
        try:
            yield record
            record["status"] = "exited"
        except BaseException as e:
            record["status"], record["error_type"] = "failed", type(e).__name__
            raise
        finally:
            record["finished_unix"] = time.time()
            atomic_write_json(path, record)

    def network_attempt(self) -> int:
        path = self.work/"network-attempts.json"
        with (self.work/".network.lock").open("a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            count = json.loads(path.read_text())["count"] if path.exists() else 0
            if count >= self.v["max_network_requests"]:
                raise RuntimeError("Network attempt cap reached")
            atomic_write_json(path, {"count": count+1})
            return count+1
