"""Bounded fresh-run timing of the production continual trainer.

The supplied configuration retains its scientific schedule and data identity.
Only its output/log locations change. The child stops at a recorded logical
batch, producing an ordinary interrupted checkpoint in a dedicated directory.
This is performance evidence, never an additional scientific trajectory.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from lm_cl.cli._common import cli_entry, print_json
from lm_cl.config import load_continual_config, save_continual_config
from lm_cl.training.checkpoint import canonical_sha256, sha256_file


def validate_window(config, *, warmup: int, batches: int, world_size: int) -> None:
    k = config.variant.slow_update_period_k
    if warmup < 2 or batches > 64 or batches - warmup < 2 * k:
        raise ValueError("Calibration requires >=2 warmup batches, >=2 measured slow windows, <=64 batches")
    if warmup % k or batches % k:
        raise ValueError("Warmup and stop must end at complete slow-update windows")
    if config.tasks[0].planned_logical_batches(config.optimization.global_sequences_per_logical_batch) < batches:
        raise ValueError("The first task does not contain the requested calibration batches")
    if bool(config.distributed) != (world_size > 1):
        raise ValueError("Calibration GPU count differs from the config's distributed layout")


def summarize(records, *, warmup: int, batches: int):
    """Measure complete optimizer windows, including data wait and updates."""
    points = {r["global_logical_batches"]: r for r in records if r["event"] == "optimizer_step"}
    if warmup not in points or batches not in points:
        raise ValueError("Missing completed warmup/final optimizer window")
    ordered = [points[i] for i in sorted(points) if warmup <= i <= batches]
    intervals = []
    for left, right in zip(ordered, ordered[1:]):
        seconds = right["wall_time_seconds"] - left["wall_time_seconds"]
        tokens = right["global_input_tokens"] - left["global_input_tokens"]
        if seconds <= 0 or tokens <= 0:
            raise ValueError("Nonmonotonic calibration counters or clock")
        intervals.append({"first_exclusive_batch": left["global_logical_batches"],
                          "last_inclusive_batch": right["global_logical_batches"],
                          "input_tokens": tokens, "seconds": seconds,
                          "input_tokens_per_second": tokens / seconds})
    tokens = sum(r["input_tokens"] for r in intervals)
    seconds = sum(r["seconds"] for r in intervals)
    return {"warmup_batches": warmup, "stop_after_batches": batches,
            "input_tokens": tokens, "seconds": seconds,
            "input_tokens_per_second": tokens / seconds, "windows": intervals,
            "minimum_window_tokens_per_second": min(r["input_tokens_per_second"] for r in intervals),
            "maximum_window_tokens_per_second": max(r["input_tokens_per_second"] for r in intervals),
            "counting_convention": "global input tokens including masked inputs; not valid targets",
            "includes": ["forward", "backward", "fast updates", "slow updates", "data wait", "logging"],
            "excludes": ["initialization", "warmup", "final checkpoint", "retention evaluation", "probes"]}


def calibrate(config_path, output_dir, *, gpus, warmup=2, batches=8,
              max_seconds=1800, minimum_free_bytes=20 * 1024**3,
              maximum_output_bytes=10 * 1024**3, require_a100=True,
              minimum_gpu_memory_bytes=0):
    source = Path(config_path).resolve()
    source_hash = sha256_file(source)
    config = load_continual_config(source)
    if not 1 <= max_seconds <= 7200 or minimum_free_bytes < 0 or maximum_output_bytes <= 0:
        raise ValueError("Invalid bounded calibration resource limits")
    if len(gpus) not in {1, 2} or len(set(gpus)) != len(gpus) or any(x < 0 for x in gpus):
        raise ValueError("Select one or two distinct nonnegative GPU IDs")
    validate_window(config, warmup=warmup, batches=batches, world_size=len(gpus))
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError("Calibration requires a new dedicated output directory")
    if output == Path(config.runtime.output_dir).expanduser().resolve():
        raise ValueError("Calibration output must differ from the scientific run output")
    if config.runtime.device != "cuda":
        raise ValueError("This calibration entry point requires an explicit CUDA config")
    if any(task.train_source.kind not in {"packed_shards", "streaming_packed"} for task in config.tasks):
        raise ValueError("Performance calibration requires recorded packed data")
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise ValueError("Unset CUDA_VISIBLE_DEVICES; --gpus selects physical device IDs")
    import torch
    inventory = []
    busy = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                                   "--format=csv,noheader,nounits"], text=True)
    busy = {int(i): (int(m), int(u)) for i, m, u in (line.split(",") for line in busy.splitlines())}
    for index in gpus:
        if index >= torch.cuda.device_count():
            raise ValueError("Selected GPU is unavailable")
        name = torch.cuda.get_device_name(index)
        if require_a100 and "A100" not in name:
            raise ValueError("A100 calibration requires A100 hardware")
        if torch.cuda.get_device_properties(index).total_memory < minimum_gpu_memory_bytes:
            raise ValueError("GPU memory is below the requested calibration class")
        if busy[index][0] > 256 or busy[index][1] > 10:
            raise RuntimeError("Selected GPU is occupied; calibration refuses interference")
        inventory.append({"index": index, "name": name,
                          "total_memory_bytes": torch.cuda.get_device_properties(index).total_memory})
    parent = output.parent
    while not parent.exists():
        parent = parent.parent
    if shutil.disk_usage(parent).free < minimum_free_bytes + maximum_output_bytes:
        raise RuntimeError("Insufficient free space for calibration reserve and output cap")
    config = replace(config, run_name=config.run_name + "-calibration",
                     runtime=replace(config.runtime, output_dir=str(output / "job"),
                                     metrics_jsonl="metrics.jsonl", tensorboard_dir=None))
    config.validate()
    output.mkdir(parents=True, exist_ok=False)
    resolved = output / "config.yaml"
    save_continual_config(config, resolved)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(map(str, gpus)),
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               PYTHONDONTWRITEBYTECODE="1", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    command = [sys.executable]
    if len(gpus) > 1:
        command += ["-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={len(gpus)}"]
    command += ["-m", "lm_cl.cli.train_continual", str(resolved),
                "--stop-after-global-logical-batches", str(batches),
                "--result-json", str(output / "trainer-result.json")]
    evidence = {"schema_version": 1, "source_config": str(source), "source_sha256": source_hash,
                "resolved_config_sha256": canonical_sha256(config.to_dict()), "command": command,
                "model_name": config.model.name, "variant": config.variant.name,
                "backbone_total_parameters": config.model.expected_total_parameters,
                "global_batch_sequences": config.optimization.global_sequences_per_logical_batch,
                "sequence_length": config.tasks[0].train_source.sequence_length,
                "physical_microbatch_sequences": config.optimization.physical_microbatch_sequences,
                "precision": config.optimization.precision,
                "slow_update_period_k": config.variant.slow_update_period_k,
                "gpu_topology": subprocess.check_output(["nvidia-smi", "topo", "-m"], text=True),
                "gpu_inventory": inventory, "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda, "python_version": sys.version,
                "max_seconds": max_seconds, "minimum_free_bytes": minimum_free_bytes,
                "maximum_output_bytes": maximum_output_bytes, "status": "running"}
    (output / "launch.json").write_text(json.dumps(evidence, indent=2) + "\n")
    started = time.monotonic()
    with (output / "trainer.log").open("x") as log:
        child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while child.poll() is None:
                if time.monotonic() - started > max_seconds:
                    raise TimeoutError("Calibration deadline reached; partial evidence is preserved")
                written = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
                if written > maximum_output_bytes or shutil.disk_usage(output).free < minimum_free_bytes:
                    raise RuntimeError("Calibration disk limit reached; partial evidence is preserved")
                time.sleep(1)
            if child.returncode:
                raise RuntimeError(f"Calibration worker failed ({child.returncode}); inspect trainer.log")
            result = json.loads((output / "trainer-result.json").read_text())
            if result["status"] != "interrupted" or result["trainer_state"]["global_logical_batches"] != batches:
                raise RuntimeError("Worker did not stop at the requested calibration boundary")
            if sha256_file(source) != source_hash:
                raise RuntimeError("Source configuration changed during calibration")
            records = [json.loads(line) for line in (output / "job/metrics.jsonl").read_text().splitlines()]
            evidence.update(status="passed", measurement=summarize(records, warmup=warmup, batches=batches),
                            elapsed_seconds=time.monotonic() - started,
                            checkpoint_sha256=result["checkpoint_sha256"],
                            checkpoint_bytes=Path(result["checkpoint_path"]).stat().st_size)
        except BaseException as exc:
            evidence.update(status="failed", error_type=type(exc).__name__, error=str(exc),
                            elapsed_seconds=time.monotonic() - started)
            raise
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            (output / "calibration.json").write_text(json.dumps(evidence, indent=2) + "\n")
    return evidence


def command():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--gpus", default="0")
    p.add_argument("--warmup-batches", type=int, default=2)
    p.add_argument("--batches", type=int, default=8)
    p.add_argument("--max-seconds", type=int, default=1800)
    p.add_argument("--minimum-free-bytes", type=int, default=20 * 1024**3)
    p.add_argument("--maximum-output-bytes", type=int, default=10 * 1024**3)
    p.add_argument("--allow-non-a100", action="store_true", help="record actual hardware; never label it an A100 measurement")
    args = p.parse_args()
    print_json(calibrate(args.config, args.output_dir, gpus=[int(x) for x in args.gpus.split(",")],
                         warmup=args.warmup_batches, batches=args.batches, max_seconds=args.max_seconds,
                         minimum_free_bytes=args.minimum_free_bytes, maximum_output_bytes=args.maximum_output_bytes,
                         require_a100=not args.allow_non_a100))


def main():
    cli_entry(command)


if __name__ == "__main__":
    main()
