"""Isolated engineering adapter for incremental blocks and existing trainers.

No network reader or relaxed legacy manifest is introduced into production.
Demo checkpoint kinds are deliberately rejected by production/probe loaders.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from lm_cl.config.continual_schema import (
    ContinualOptimizationConfig, ContinualRuntimeConfig, DistributedConfig,
)
from lm_cl.config.schema import ModelConfig, VariantConfig
from lm_cl.data.incremental import IncrementalSource, file_hash, immutable_write, json_bytes
from lm_cl.data.types import TokenBatch, TokenPosition
from lm_cl.training.checkpoint import (
    CHECKPOINT_KIND, capture_rng_state, restore_rng_state, validate_checkpoint_payload,
)
from lm_cl.training.continual import ContinualTrainer, TrainerState, TrainingResult
from lm_cl.training.distributed import DistributedContext, plan_logical_batch_partition
from lm_cl.training.distributed_continual import DistributedContinualTrainer


DEMO_CHECKPOINT_KIND = "lm-cl-local-pipeline-demo-checkpoint-v1"


class DemoStopRequested(Exception):
    pass


@dataclass(frozen=True)
class BlockInput:
    root: str
    recipe_sha256: str
    sequence_length: int
    wait_seconds: float

    @property
    def synthetic(self):
        return None

    @property
    def packed(self):
        return None


@dataclass(frozen=True)
class DemoTask:
    language: str
    task_index: int
    cycle_index: int
    train_source: BlockInput
    input_token_budget: int
    train_sequence_prefix_count: int
    validation_source: BlockInput | None = None
    validation_logical_batches: int = 0
    train_sequence_offset_count: int = 0
    logical_batches: None = None

    def planned_logical_batches(self, batch: int) -> int:
        return math.ceil(self.train_sequence_prefix_count / batch)


@dataclass(frozen=True)
class DemoTrainingConfig:
    schema_version: int
    run_name: str
    model: ModelConfig
    variant: VariantConfig
    optimization: ContinualOptimizationConfig
    runtime: ContinualRuntimeConfig
    tasks: list[DemoTask]
    distributed: DistributedConfig | None = None
    source_checkpoint: str | None = None
    source_checkpoint_sha256: str | None = None
    warmup_logical_batches: int = 2
    save_final_checkpoint: bool = True
    ddp_reducer_priming_steps: int = 2
    collect_microbatch_timings: bool = True

    def validate(self):
        if self.schema_version != 101:
            raise ValueError("Unknown engineering training config version")
        self.model.validate()
        self.variant.validate()
        self.optimization.validate()
        self.runtime.validate()
        if not self.run_name or not self.tasks or self.warmup_logical_batches < 0:
            raise ValueError("Invalid engineering workload")
        if self.ddp_reducer_priming_steps != 2:
            raise ValueError("Demo DDP uses two discarded reducer-priming backwards")
        if type(self.collect_microbatch_timings) is not bool:
            raise ValueError("Microbatch timing switch must be Boolean")
        if self.distributed:
            self.distributed.validate(runtime_device=self.runtime.device)
        if self.runtime.device == "cuda" and self.optimization.precision == "bf16":
            raise ValueError("This Pascal demonstration prohibits BF16")
        if (self.source_checkpoint is None) != (self.source_checkpoint_sha256 is None):
            raise ValueError("Derived source requires both path and SHA256")
        for i, task in enumerate(self.tasks):
            if task.task_index != i or task.cycle_index < 0:
                raise ValueError("Invalid demo task ordering")
            s = IncrementalSource(task.train_source.root, recipe_sha256=task.train_source.recipe_sha256, wait_seconds=0)
            if s.recipe.language != task.language or s.recipe.purpose not in {"train", "timing_only"}:
                raise ValueError("Training recipe language/purpose differs")
            length = task.train_source.sequence_length
            if length != s.recipe.sequence_length or length > self.model.max_position_embeddings:
                raise ValueError("Invalid model sequence length")
            if s.recipe.maximum_token_id >= self.model.vocab_size:
                raise ValueError("Source tokens exceed model vocabulary")
            if self.variant.memory_enabled and length != 2*self.variant.segment_length:
                raise ValueError("AG requires exactly two segments per sequence")
            if task.input_token_budget != task.train_sequence_prefix_count*length or task.train_sequence_prefix_count <= 0:
                raise ValueError("Task budget must be complete ordered sequences")
            if (task.train_sequence_offset_count+task.train_sequence_prefix_count)*length > s.token_count:
                raise ValueError("Task exceeds frozen data budget")
            if task.validation_source:
                v = IncrementalSource(task.validation_source.root, recipe_sha256=task.validation_source.recipe_sha256, wait_seconds=0)
                if not (v.root/'complete.json').exists() or v.recipe.purpose != "validation" or v.recipe.language != task.language:
                    raise ValueError("Fixed validation must already be complete")
                if v.recipe.sequence_length != length or task.validation_logical_batches <= 0:
                    raise ValueError("Invalid validation dimensions")
            elif task.validation_logical_batches:
                raise ValueError("Validation budget without a source")

    def to_dict(self):
        return asdict(self)


def load_demo_training_config(path: str | Path) -> DemoTrainingConfig:
    v = json.loads(Path(path).read_text())
    v["model"] = ModelConfig(**v["model"])
    v["variant"] = VariantConfig(**v["variant"])
    v["optimization"] = ContinualOptimizationConfig(**v["optimization"])
    v["runtime"] = ContinualRuntimeConfig(**v["runtime"])
    v["distributed"] = None if v.get("distributed") is None else DistributedConfig(**v["distributed"])
    tasks = []
    for t in v["tasks"]:
        t["train_source"] = BlockInput(**t["train_source"])
        if t.get("validation_source"):
            t["validation_source"] = BlockInput(**t["validation_source"])
        tasks.append(DemoTask(**t))
    v["tasks"] = tasks
    cfg = DemoTrainingConfig(**v)
    cfg.validate()
    return cfg


def load_demo_checkpoint(path: str | Path) -> dict:
    path = Path(path)
    sidecar = json.loads(path.with_suffix(path.suffix + ".sha256.json").read_text())
    if file_hash(path) != sidecar["sha256"]:
        raise ValueError("Demo checkpoint checksum mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("checkpoint_kind") != DEMO_CHECKPOINT_KIND:
        raise ValueError("Not a demo checkpoint")
    validate_checkpoint_payload({**payload, "checkpoint_kind": CHECKPOINT_KIND})
    source = IncrementalSource(payload["source_identity"]["root"],
                               recipe_sha256=payload["source_identity"]["recipe_sha256"], wait_seconds=0)
    source.validate_proof(payload["incremental_prefix"])
    return payload


class _DemoMixin:
    def __init__(self, config, *args):
        self._demo_sources = {}
        self._micro_timings = []
        self._batch_start = None
        self._control_started_unix = time.time()
        self._recording_microbatches = False
        super().__init__(config, *args)
        if hasattr(self, "ddp_model"):
            # DDP normally rebuilds buckets after its first backwards. A fresh
            # reducer on resume can otherwise change reduction ordering. Prime
            # both fresh/resumed reducers without optimizer/state transitions.
            rng = capture_rng_state()
            dummy = torch.zeros((1, config.tasks[0].train_source.sequence_length), dtype=torch.long, device=self.device)
            priming_start = time.perf_counter()
            for _ in range(config.ddp_reducer_priming_steps):
                with self._autocast():
                    output = self.ddp_model(dummy, dummy)
                (output.loss_sum / output.target_count).backward()
                self.model.zero_grad(set_to_none=True)
            self._sync()
            self.reducer_priming_seconds = time.perf_counter()-priming_start
            restore_rng_state(rng)
        if config.source_checkpoint:
            source = Path(config.source_checkpoint)
            if file_hash(source) != config.source_checkpoint_sha256:
                raise ValueError("Immutable source checkpoint hash differs")
            p = torch.load(source, map_location="cpu", weights_only=False)
            if p["trainer_state"]["phase"] != "task_boundary" or p["trainer_state"]["window_logical_batches"]:
                raise ValueError("Derived timing run requires a stable boundary")
            if p["resolved_config"]["model"] != asdict(config.model):
                raise ValueError("Derived timing model differs from source")
            self.model.load_state_dict(p["model_state"])
            del p
            # All optimizer/RNG/counters/active state are fresh. Only slow weights/M0 loaded.
            if file_hash(source) != config.source_checkpoint_sha256:
                raise ValueError("Source checkpoint changed while cloning")

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _forward_model(self, input_ids, labels, **kwargs):
        if not self._recording_microbatches:
            return super()._forward_model(input_ids, labels, **kwargs)
        before = time.perf_counter()
        events = None
        if self.device.type == "cuda":
            events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            events[0].record()
        result = super()._forward_model(input_ids, labels, **kwargs)
        if events:
            events[1].record()
        self._micro_timings.append((before, int(input_ids.shape[0]), events))
        return result

    def _open_source(self, source_config):
        key = source_config.root
        if key not in self._demo_sources:
            self._demo_sources[key] = IncrementalSource(key, recipe_sha256=source_config.recipe_sha256,
                                                       wait_seconds=source_config.wait_seconds)
        source = self._demo_sources[key]
        source.refresh()
        return source, source.identity, -100

    def _batch_iterator(self, source, task, *, start):
        end = task.train_sequence_offset_count + task.train_sequence_prefix_count
        for batch in source.iter_batches(sequence_length=task.train_source.sequence_length,
                    global_sequences_per_batch=self.config.optimization.global_sequences_per_logical_batch,
                    start=start, sequence_prefix_count=end):
            stop = self.output_dir/"STOP_REQUESTED"
            if stop.exists() and stop.stat().st_mtime > self._control_started_unix:
                raise DemoStopRequested("Owned demo stop requested")
            if os.environ.get("LM_CL_DEMO_LIMITS"):
                from lm_cl.diagnostics.local_pipeline_resources import Limits
                Limits(os.environ["LM_CL_DEMO_LIMITS"]).check()
            if hasattr(self, "distributed"):
                n, length = batch.input_ids.shape
                p = plan_logical_batch_partition(n, self.distributed.rank, self.distributed.world_size)
                yield TokenBatch(batch.input_ids[p.start:p.end], batch.labels[p.start:p.end], p.size*(length-1),
                    batch.start_position, batch.next_position, n, p.start, p.end,
                    batch.start_position.token_offset//length, batch.next_position.token_offset//length)
            else:
                yield batch

    def _train_logical_batch(self, batch, *, ignore_index):
        self._sync()
        start = time.perf_counter()
        self._micro_timings = []
        self._recording_microbatches = self.config.collect_microbatch_timings
        try:
            result = super()._train_logical_batch(batch, ignore_index=ignore_index)
        finally:
            self._recording_microbatches = False
        self._sync()
        elapsed = time.perf_counter()-start
        cycles = [b[0]-a[0] for a,b in zip(self._micro_timings,self._micro_timings[1:])]
        forward_ms = [entry[2][0].elapsed_time(entry[2][1]) for entry in self._micro_timings if entry[2]]
        self._log("timing_train", seconds=elapsed,
                  source_root=self.source_identity["root"],
                  source_recipe_sha256=self.source_identity["recipe_sha256"],
                  measured=self.state.global_logical_batches > self.config.warmup_logical_batches,
                  input_tokens=result["input_token_count"], valid_targets=result["valid_target_count"],
                  loss_scale=self.scaler.get_scale(),
                  max_cuda_allocated_bytes=torch.cuda.max_memory_allocated(self.device) if self.device.type=="cuda" else 0,
                  consumed_token_sha256=hashlib.sha256(batch.input_ids.astype("<u4").tobytes()).hexdigest(),
                  physical_microbatch_count=len(self._micro_timings),
                  physical_microbatch_rows=[m[1] for m in self._micro_timings],
                  physical_cycle_seconds=cycles,
                  forward_device_milliseconds=forward_ms,
                  physical_cycle_definition="successive forward starts; includes backward, communication and host feeding; last omitted",
                  monotonic=time.monotonic(), unix_time=time.time())
        return result

    def _optimizer_step(self, *, tail_flush):
        self._sync()
        start = time.perf_counter()
        result = super()._optimizer_step(tail_flush=tail_flush)
        self._sync()
        self._log("timing_optimizer", seconds=time.perf_counter()-start, tail_flush=tail_flush,
                  measured=self.state.global_logical_batches > self.config.warmup_logical_batches,
                  loss_scale=self.scaler.get_scale(), unix_time=time.time())
        return result

    def _evaluate_source(self, task):
        self._sync()
        start = time.perf_counter()
        result = super()._evaluate_source(task)
        self._sync()
        if result is not None:
            self._log("timing_evaluation", seconds=time.perf_counter()-start, result=result)
        return result

    def _checkpoint_payload(self):
        payload = super()._checkpoint_payload()
        source = self._demo_sources[self.source_identity["root"]]
        payload["incremental_prefix"] = source.proof(self.state.source_position["token_offset"])
        payload["checkpoint_kind"] = DEMO_CHECKPOINT_KIND
        payload["engineering_only"] = True
        return payload

    def _save_checkpoint(self, filename):
        if not self.config.save_final_checkpoint and filename.endswith("-boundary.pt"):
            self._log("checkpoint_excluded_from_timing_run", reason="validated config disables final payload")
            return "", ""
        self._sync()
        start = time.perf_counter()
        primary = not hasattr(self, "distributed") or self.distributed.is_primary
        if hasattr(self, "distributed"):
            dist.barrier()
            self._checkpoint_state_digests = self._verify_shared_state_for_checkpoint()
            states = [None]*self.distributed.world_size
            topology = [None]*self.distributed.world_size
            dist.all_gather_object(states, {"rank": self.distributed.rank, "state": capture_rng_state()})
            dist.all_gather_object(topology, self.distributed.topology_record())
            self._checkpoint_rank_rng_states, self._checkpoint_rank_topology = states, topology
        path = self.checkpoint_dir / filename.replace(".pt", ".demo.pt")
        outcome = [None]
        if primary:
            try:
                if path.exists():
                    raise FileExistsError(f"Refusing existing checkpoint: {path}")
                temp = path.with_name(path.name+f".{os.getpid()}.partial")
                payload = self._checkpoint_payload()
                validate_checkpoint_payload({**payload, "checkpoint_kind": CHECKPOINT_KIND})
                guard = nullcontext()
                if os.environ.get("LM_CL_DEMO_LIMITS"):
                    from lm_cl.diagnostics.local_pipeline_resources import Limits
                    limits = Limits(os.environ["LM_CL_DEMO_LIMITS"])
                    guard = limits.allocation(sum(p.numel() for p in self.model.parameters())*20+16*1024**2)
                with guard:
                    with temp.open("xb") as f:
                        torch.save(payload, f)
                        f.flush()
                        os.fsync(f.fileno())
                    loaded = torch.load(temp, map_location="cpu", weights_only=False)
                    validate_checkpoint_payload({**loaded, "checkpoint_kind": CHECKPOINT_KIND})
                    checksum = file_hash(temp)
                    os.link(temp, path)
                    immutable_write(path.with_suffix(path.suffix+".sha256.json"), json_bytes({"sha256": checksum}))
                outcome[0] = {"path": str(path), "sha256": checksum}
            except BaseException as e:
                outcome[0] = {"error": f"{type(e).__name__}: {e}"}
        if hasattr(self, "distributed"):
            dist.broadcast_object_list(outcome, src=0)
        if "error" in outcome[0]:
            raise RuntimeError(outcome[0]["error"])
        self._log("timing_checkpoint", seconds=time.perf_counter()-start,
                  checkpoint_path=outcome[0]["path"], checkpoint_sha256=outcome[0]["sha256"],
                  bytes=path.stat().st_size)
        return outcome[0]["path"], outcome[0]["sha256"]

    def _load_resume(self, checkpoint_path):
        start = time.perf_counter()
        p = load_demo_checkpoint(checkpoint_path)
        if p["config_sha256"] != self.config_sha256:
            raise ValueError("Demo resume configuration differs")
        m = p["distributed_state"]
        if hasattr(self, "distributed"):
            expected = {"enabled": True, "world_size": self.distributed.world_size,
                        "backend": self.distributed.backend, "partition_rule": self.distributed.partition_rule,
                        "reduction_policy": self.config.distributed.reduction_policy,
                        "global_logical_batch_size": self.config.optimization.global_sequences_per_logical_batch,
                        "active_memory_sync_policy": self.config.distributed.active_memory_sync_policy}
            if any(m.get(k) != v for k,v in expected.items()):
                raise ValueError("Demo resume distributed layout differs")
            self._checkpoint_state_digests = m["state_digests"]
            rng = m["rank_rng_states"][self.distributed.rank]["state"]
        else:
            if m["enabled"] or m["world_size"] != 1:
                raise ValueError("Demo resume world size differs")
            rng = p["rng_state"]
        self.model.load_state_dict(p["model_state"])
        self.state = TrainerState.from_dict(p["trainer_state"])
        self._restore_memory_state(p["memory_state"])
        restore_rng_state(rng)
        self._log("timing_load", seconds=time.perf_counter()-start)
        return p

    def run(self, **kwargs):
        try:
            result = self._run_impl(**kwargs)
            if self.config.source_checkpoint and file_hash(Path(self.config.source_checkpoint)) != self.config.source_checkpoint_sha256:
                raise ValueError("Preserved source changed during benchmark")
            return result
        except (TimeoutError, DemoStopRequested) as e:
            path, checksum = self._save_checkpoint(f"waiting-step-{self.state.global_logical_batches:08d}-{time.time_ns()}.pt")
            status = "waiting_for_data" if isinstance(e, TimeoutError) else "stopped_by_request"
            return TrainingResult(status, path, checksum, self.state.to_dict(), list(self.state.loss_history))
        except BaseException as e:
            if self.logger is not None and not self.logger.closed:
                self._log("run_error", error_type=type(e).__name__, error_message=str(e))
            raise
        finally:
            if self.logger:
                self.logger.close()
            if getattr(self, "rank_logger", None):
                self.rank_logger.close()


class DemoTrainer(_DemoMixin, ContinualTrainer):
    pass


class DistributedDemoTrainer(_DemoMixin, DistributedContinualTrainer):
    pass
