"""Commands for the isolated, bounded local pipeline engineering study."""
from __future__ import annotations

import argparse
import fcntl
import importlib.metadata
import json
import os
import signal
import sys
import time
from dataclasses import asdict
from pathlib import Path

from lm_cl.data.incremental import (
    FORMAT, IncrementalSource, Producer, Recipe, file_hash, initialize,
    immutable_write, json_bytes, load_recipe,
)
from lm_cl.data.storage import atomic_write_json, ensure_owned_root
from lm_cl.diagnostics.local_pipeline_resources import Limits


def _deadline(seconds):
    def expired(_signal, _frame):
        raise RuntimeError("Configured process deadline reached")
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(seconds)


def _tokenizer_reference(path):
    from lm_cl.config.data_schema import TokenizerReference
    return TokenizerReference("Qwen/Qwen3-0.6B-Base", "da87bfb608c14b7cf20ba1ce41287e8de496c0cd",
                              str(Path(path).resolve()), 151643, 151669, 151668, 151680, 151643, 151643)


def _preflight(args, limits):
    import platform
    import psutil
    import subprocess
    record = {"time": time.time(), "python": sys.executable, "platform": platform.platform(),
              "cpu_affinity": psutil.Process().cpu_affinity(), "memory": dict(psutil.virtual_memory()._asdict()),
              "limits": limits.v, "resources": limits.check(), "packages": {}}
    for package in ["torch", "numpy", "datasets", "pyarrow", "transformers", "tokenizers", "huggingface-hub", "pytest", "psutil"]:
        try:
            record["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            record["packages"][package] = None
    record["gpu_inventory"] = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu,driver_version", "--format=csv"], text=True)
    record["gpu_topology"] = subprocess.check_output(["nvidia-smi", "topo", "-m"], text=True)
    if args.tokenizer_manifest:
        from lm_cl.data.tokenizer import load_verified_tokenizer
        tokenizer, manifest = load_verified_tokenizer(_tokenizer_reference(args.tokenizer_manifest))
        record["tokenizer"] = {"manifest_sha256": manifest["manifest_content_sha256"], "effective_size": len(tokenizer),
                               "test_tokens": tokenizer.encode("Xin chào Việt Nam", add_special_tokens=False)}
    out = limits.owned(args.output)
    immutable_write(out, json_bytes(record))
    print(json.dumps({"status": "preflight_complete", "output": str(out), "resources": record["resources"]}))


def _discover(args, limits):
    from lm_cl.data.incremental_remote import RemoteAccess
    _deadline(limits.v["max_prepare_process_seconds"])
    ensure_owned_root(limits.work/"hf-cache", purpose="huggingface-cache")
    with limits.process("discover", args.language):
        remote = RemoteAccess(limits)
        value = remote.discover(args.language, args.max_files)
        immutable_write(limits.owned(args.output), json_bytes(value))
        print(json.dumps({"status": "source_identity_frozen", "files": len(value["files"]), "network_bytes": remote.network_bytes}))


def _initialize(args, limits):
    recipe = Recipe(**json.loads(Path(args.recipe).read_text()))
    recipe.validate()
    root = limits.owned(args.root)
    count = recipe.output_tokens
    for p in (limits.work/"data").glob("*/recipe.json"):
        if p.parent.resolve() != root:
            count += load_recipe(p.parent).output_tokens
    if count > limits.v["max_unique_prepared_tokens"] or recipe.max_input_documents > limits.v["max_input_documents_per_language"]:
        raise ValueError("Preparation recipe exceeds aggregate token/document caps")
    with limits.allocation(recipe.block_tokens*4 + 1024**2):
        initialize(root, recipe)
    print(json.dumps({"status": "initialized", "root": str(root), "recipe_sha256": recipe.sha256}))


def _prepare(args, limits):
    from lm_cl.data.incremental_remote import RemoteAccess, ParquetRows
    from lm_cl.data.tokenizer import load_verified_tokenizer
    root = limits.owned(args.root)
    recipe = load_recipe(root)
    if args.consumer_metrics and (args.lookahead_tokens is None or args.lookahead_tokens < recipe.block_tokens):
        raise ValueError("Consumer lookahead must cover at least one complete block")
    if args.lookahead_tokens is not None and not args.consumer_metrics:
        raise ValueError("Lookahead requires explicit matching consumer metrics")
    if recipe.max_input_documents > limits.v["max_input_documents_per_language"]:
        raise ValueError("Recipe source-document cap exceeds envelope")
    _deadline(limits.v["max_prepare_process_seconds"])
    with limits.process("producer", root.name), (root/".producer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = time.monotonic()
        started_unix = time.time()
        tokenizer, manifest = load_verified_tokenizer(_tokenizer_reference(recipe.tokenizer_identity["manifest_path"]))
        if manifest["manifest_content_sha256"] != recipe.tokenizer_identity["manifest_sha256"]:
            raise ValueError("Tokenizer identity mismatch")
        remote = RemoteAccess(limits)
        source = ParquetRows(recipe.source_identity, remote)
        producer = Producer(root, source, tokenizer,
            publication_guard=lambda: limits.allocation(recipe.block_tokens*4 + 16*1024**2))
        initial_tokens = producer.reader.ready_tokens
        log_path = limits.report/"raw"/f"prepare-{root.name}-{os.getpid()}.jsonl"
        n = 0
        while args.max_blocks is None or n < args.max_blocks:
            if (root/"STOP_REQUESTED").exists() and (root/"STOP_REQUESTED").stat().st_mtime > started_unix:
                break
            if args.consumer_metrics and args.lookahead_tokens:
                consumed = shared_consumed_prefix(args.consumer_metrics, root, recipe.sha256, limits)
                if producer.reader.ready_tokens-consumed >= args.lookahead_tokens:
                    limits.check()
                    time.sleep(.25)
                    continue
            tick = time.monotonic()
            record = producer.publish_one()
            if record is None:
                break
            n += 1
            state = record["producer_state"]
            event = {"event": "block_committed", "root": str(root), "index": record["index"],
                     "ready_tokens": producer.reader.ready_tokens, "new_tokens": producer.reader.ready_tokens-initial_tokens,
                     "seconds": time.monotonic()-started, "block_seconds": time.monotonic()-tick,
                     "source_cursor": state["source_cursor"], "accepted_documents": state["accepted_documents"],
                     "rejections": state["rejections"], "network_bytes": remote.network_bytes,
                     "cache_read_bytes": remote.cache_bytes, "source_read_seconds": source.read_seconds,
                     "requests": remote.requests, "monotonic": time.monotonic(), "unix_time": time.time()}
            with log_path.open("a") as f:
                f.write(json.dumps(event)+"\n")
            print(json.dumps(event), flush=True)
        print(json.dumps({"status": "complete" if (root/"complete.json").exists() else "stopped_at_committed_block",
                          "root": str(root), "ready_tokens": producer.reader.ready_tokens,
                          "elapsed_seconds": time.monotonic()-started}), flush=True)


def shared_consumed_prefix(paths, root, recipe_sha256, limits):
    """Maximum cursor of matching readers; unrelated tasks and partial JSONL are ignored."""
    consumed = 0
    for name in paths:
        path = limits.owned(name)
        if not path.exists():
            continue
        with path.open() as stream:
            for line in stream:
                if not line.endswith("\n"):
                    break
                event = json.loads(line)
                if event.get("source_root") != str(root) or event.get("source_recipe_sha256") != recipe_sha256:
                    continue
                position = event.get("source_position", {})
                if position.get("shard_index") != 0 or type(position.get("token_offset")) is not int:
                    raise ValueError("Matching consumer record lacks a valid global stream cursor")
                consumed = max(consumed, position["token_offset"])
    return consumed


def _train(args, limits):
    import torch
    from lm_cl.diagnostics.local_pipeline_training import load_demo_training_config, DemoTrainer, DistributedDemoTrainer
    from lm_cl.training.distributed import DistributedContext
    cfg = load_demo_training_config(args.config)
    limits.owned(cfg.runtime.output_dir)
    _deadline(limits.v["max_train_process_seconds"])
    torch.set_num_threads(limits.v["cpu_threads_per_trainer"])
    os.environ["LM_CL_DEMO_LIMITS"] = str(limits.path)
    context = None
    try:
        with limits.process("trainer", cfg.run_name):
            if cfg.distributed:
                context = DistributedContext.initialize(cfg.distributed, runtime_device=cfg.runtime.device)
                trainer = DistributedDemoTrainer(cfg, context)
            else:
                trainer = DemoTrainer(cfg)
            result = trainer.run(resume_checkpoint=args.resume,
                                 stop_after_global_logical_batches=args.stop_after_batches,
                                 stop_after_task_boundaries=args.stop_after_boundaries)
            if context is None or context.is_primary:
                name = "complete.json" if result.status=="complete" else f"stopped-{time.time_ns()}.json"
                path = Path(cfg.runtime.output_dir)/name
                immutable_write(path, json_bytes(asdict(result)))
                print(json.dumps({"status": result.status, "record": str(path), "checkpoint": result.checkpoint_path}), flush=True)
    finally:
        if context:
            context.close()


def _verify(args, limits):
    import hashlib
    s = IncrementalSource(args.root, wait_seconds=0)
    if not (s.root/"complete.json").exists():
        raise ValueError("No valid completion record")
    h = hashlib.sha256()
    for array in s.arrays:
        h.update(array.tobytes())
    boundaries = [b for r in s.records for b in r["boundaries"]]
    timing_replay = None
    if s.recipe.source_identity.get("kind") == "completed_packed_timing_replay_v1":
        from lm_cl.data.incremental_timing import verify_timing_replay
        timing_replay = verify_timing_replay(s)
    end = 0
    contents, token_hashes = set(), set()
    for i, b in enumerate(boundaries):
        if b["document_index"] != i or b["token_start"] != end or b["token_end"] != end+b["content_token_count"]+1:
            raise ValueError("Canonical document boundary discontinuity")
        if timing_replay is None and (b["content_sha256"] in contents or b["token_ids_sha256"] in token_hashes):
            raise ValueError("Duplicate accepted document")
        contents.add(b["content_sha256"]); token_hashes.add(b["token_ids_sha256"])
        eos, _ = s.read_tokens(1, start=s.position_at(b["token_end"]-1))
        if int(eos[0]) != s.recipe.eos_token_id or not b["eos_after"]:
            raise ValueError("Document EOS differs")
        end = b["token_end"]
    if end != s.token_count:
        raise ValueError("Final document budget mismatch")
    record = {"status": "verified", "recipe_sha256": s.recipe.sha256, "token_count": s.token_count,
              "valid_targets": (s.token_count//s.recipe.sequence_length)*(s.recipe.sequence_length-1),
              "token_bytes_sha256": h.hexdigest(), "canonical_boundaries_sha256": hashlib.sha256(json_bytes(boundaries)).hexdigest(),
              "accepted_documents": len(boundaries), "completion_sha256": file_hash(s.root/"complete.json"),
              "timing_replay": timing_replay}
    immutable_write(limits.owned(args.output), json_bytes(record))
    print(json.dumps(record))


def _monitor(args, limits):
    import psutil
    records = []
    for p in (limits.work/"processes").glob("*.json"):
        v = json.loads(p.read_text())
        if v["status"] == "running":
            try:
                process = psutil.Process(v["pid"])
                v["same_process_alive"] = process.create_time() == v["create_time"]
                v["rss_bytes"] = process.memory_info().rss
            except psutil.NoSuchProcess:
                v["same_process_alive"] = False
            records.append(v)
    print(json.dumps({"resources": limits.check(), "processes": records}, indent=2))


def _stop(args, limits):
    root = limits.owned(args.root)
    if not ((root/"recipe.json").is_file() or (root/"resolved_config.yaml").is_file()):
        raise ValueError("Stop target is not an initialized demo producer/trainer")
    event = {"requested_unix": time.time(), "action": "stop_at_safe_boundary"}
    with (root/"control-requests.jsonl").open("a") as f:
        f.write(json.dumps(event)+"\n")
    atomic_write_json(root/"STOP_REQUESTED", event)
    print(json.dumps({"status": "stop_requested", "root": str(root), "note": "No signal sent; wait for a committed block or checkpoint"}))


def command():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limits", required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("preflight"); p.add_argument("--output", required=True); p.add_argument("--tokenizer-manifest")
    p = sub.add_parser("discover"); p.add_argument("--language", required=True); p.add_argument("--max-files", type=int, default=2); p.add_argument("--output", required=True)
    p = sub.add_parser("initialize"); p.add_argument("--recipe", required=True); p.add_argument("--root", required=True)
    p = sub.add_parser("prepare"); p.add_argument("--root", required=True); p.add_argument("--max-blocks", type=int); p.add_argument("--consumer-metrics", action="append"); p.add_argument("--lookahead-tokens", type=int)
    p = sub.add_parser("train"); p.add_argument("--config", required=True); p.add_argument("--resume"); p.add_argument("--stop-after-batches", type=int); p.add_argument("--stop-after-boundaries", type=int)
    p = sub.add_parser("verify"); p.add_argument("--root", required=True); p.add_argument("--output", required=True)
    sub.add_parser("monitor")
    p = sub.add_parser("stop"); p.add_argument("--root", required=True)
    args = parser.parse_args()
    limits = Limits(args.limits)
    globals()["_"+args.command](args, limits)


if __name__ == "__main__":
    command()
