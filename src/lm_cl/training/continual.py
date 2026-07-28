from __future__ import annotations

import math
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import torch

from lm_cl.config import (
    ContinualExperimentConfig,
    ContinualTaskConfig,
    TrainSourceConfig,
    save_continual_config,
)
from lm_cl.data import (
    TokenBatch,
    TokenPosition,
    open_token_batch_source,
    validate_packed_shards,
)
from lm_cl.metrics import JsonlMetricLogger, update_forgetting_metrics
from lm_cl.models import RMTZyphraTransformer, ZyphraTransformer
from lm_cl.training.checkpoint import (
    CHECKPOINT_KIND,
    CHECKPOINT_SCHEMA_VERSION,
    atomic_save_checkpoint,
    canonical_sha256,
    capture_rng_state,
    checkpoint_metadata,
    load_checkpoint,
    restore_rng_state,
)
from lm_cl.training.scheduler import LinearWarmupConstantScheduler
from lm_cl.training.seed import set_deterministic_seed


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(requested)


def tensor_norm(tensors: Iterator[torch.Tensor]) -> float:
    squares = torch.zeros((), dtype=torch.float64)
    for tensor in tensors:
        value = tensor.detach()
        squares += value.double().pow(2).sum().cpu()
    return float(squares.sqrt())


def conditioned_backward_reference_targets(
    *,
    slow_update_period_k: int,
    global_sequences_per_logical_batch: int,
    sequence_length: int,
    fp16_loss_conditioning_multiplier: int,
) -> int:
    """Return the algebraically neutral FP16 summed-loss divisor."""
    if slow_update_period_k <= 0:
        raise ValueError("slow_update_period_k must be positive")
    if global_sequences_per_logical_batch <= 0:
        raise ValueError(
            "global_sequences_per_logical_batch must be positive"
        )
    if sequence_length <= 1:
        raise ValueError("sequence_length must exceed one")
    if fp16_loss_conditioning_multiplier != 2:
        raise ValueError("fp16_loss_conditioning_multiplier must be 2")
    return (
        fp16_loss_conditioning_multiplier
        * slow_update_period_k
        * global_sequences_per_logical_batch
        * (sequence_length - 1)
    )


def _position_dict(position: TokenPosition) -> dict[str, int]:
    return {
        "shard_index": position.shard_index,
        "token_offset": position.token_offset,
    }


def _position_from_dict(value: dict[str, Any]) -> TokenPosition:
    position = TokenPosition(
        shard_index=int(value["shard_index"]),
        token_offset=int(value["token_offset"]),
    )
    position.validate()
    return position


def _source_identity(source_config: TrainSourceConfig, source: Any) -> dict[str, Any]:
    if source_config.synthetic is not None:
        return {
            "kind": "synthetic",
            "config_sha256": canonical_sha256(
                {
                    key: value
                    for key, value in source_config.synthetic.__dict__.items()
                }
            ),
        }
    assert source_config.packed is not None
    manifest = source.manifest
    manifest_path = (source.stage_dir / "manifest.json").resolve()
    return {
        "kind": "packed_shards",
        "manifest_path": str(manifest_path),
        "manifest_content_sha256": manifest["manifest_content_sha256"],
        "ordered_data_sha256": manifest["ordered_data_sha256"],
        "tokenizer_manifest_path": (
            source_config.packed.tokenizer.manifest_path
        ),
        "tokenizer_manifest_sha256": manifest["tokenizer"][
            "manifest_content_sha256"
        ],
    }


def build_source(
    source_config: TrainSourceConfig,
    *,
    model_vocab_size: int,
) -> tuple[Any, dict[str, Any], int]:
    if source_config.synthetic is not None:
        source = open_token_batch_source(source_config.synthetic)
        return (
            source,
            _source_identity(source_config, source),
            source_config.synthetic.ignore_index,
        )
    assert source_config.packed is not None
    packed = source_config.packed
    stage_dir = (
        Path(packed.storage.generated_root)
        / "stages"
        / packed.stage.stage_id
    )
    validate_packed_shards(stage_dir)
    source = open_token_batch_source(packed)
    if source.manifest["tokenizer"]["model_embedding_vocab_size"] != (
        model_vocab_size
    ):
        raise ValueError(
            "Packed source model vocabulary differs from configured model"
        )
    return source, _source_identity(source_config, source), -100


def _task_window_source_identity(
    source_identity: dict[str, Any], task: ContinualTaskConfig
) -> dict[str, Any]:
    identity = dict(source_identity)
    count = task.train_sequence_prefix_count
    if count is not None:
        start = task.train_sequence_offset_count
        window = {
            "sequence_start": start,
            "sequence_count": count,
            "sequence_end_exclusive": start + count,
            "input_token_count": count * task.train_source.sequence_length,
        }
        window["view_sha256"] = canonical_sha256(
            {
                "source": source_identity,
                "window": window,
            }
        )
        identity["sequence_window"] = window
    return identity


def _source_position_for_sequence(
    source: Any, *, sequence_index: int, sequence_length: int
) -> TokenPosition:
    if sequence_index < 0:
        raise ValueError("Sequence offset must be non-negative")
    method = getattr(source, "position_for_global_sequence", None)
    if method is None:
        if sequence_index:
            raise ValueError("Token source does not support sequence offsets")
        return TokenPosition(0, 0)
    return method(sequence_index, sequence_length=sequence_length)


@dataclass
class TrainerState:
    phase: str = "task_active"
    next_task_index: int = 0
    current_task_index: int = 0
    cycle_index: int = 0
    language: str = "en"
    logical_batch_within_task: int = 0
    task_slow_steps: int = 0
    global_logical_batches: int = 0
    global_slow_steps: int = 0
    global_input_tokens: int = 0
    global_valid_targets: int = 0
    global_fast_updates: int = 0
    task_input_tokens: int = 0
    task_valid_targets: int = 0
    task_loss_sum: float = 0.0
    task_fast_updates: int = 0
    memory_reset_count: int = 0
    window_logical_batches: int = 0
    window_valid_targets: int = 0
    window_loss_sum: float = 0.0
    source_position: dict[str, int] = field(
        default_factory=lambda: {"shard_index": 0, "token_offset": 0}
    )
    loss_history: list[float] = field(default_factory=list)
    fast_gradient_norm_history: list[float] = field(default_factory=list)
    fast_clipped_gradient_norm_history: list[float] = field(
        default_factory=list
    )
    fast_memory_norm_history: list[float] = field(default_factory=list)
    last_active_memory_grad_norm: float | None = None
    last_active_memory_clipped_grad_norm: float | None = None
    last_active_memory_norm: float | None = None
    last_encoded_write_memory_norm: float | None = None
    forgetting_first_mean_loss: dict[str, float] = field(default_factory=dict)
    forgetting_best_mean_loss: dict[str, float] = field(default_factory=dict)
    forgetting_latest_mean_loss: dict[str, float] = field(default_factory=dict)
    forgetting_evaluation_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in self.__dict__.items()
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "TrainerState":
        values = dict(values)
        defaults = cls()
        for name in (
            "forgetting_first_mean_loss",
            "forgetting_best_mean_loss",
            "forgetting_latest_mean_loss",
            "forgetting_evaluation_count",
        ):
            values.setdefault(name, getattr(defaults, name))
        expected = set(cls().__dict__)
        unknown = sorted(set(values) - expected)
        missing = sorted(expected - set(values))
        if unknown or missing:
            raise ValueError(
                f"Invalid trainer state; unknown={unknown}, missing={missing}"
            )
        return cls(**values)


@dataclass(frozen=True)
class TrainingResult:
    status: str
    checkpoint_path: str
    checkpoint_sha256: str
    state: dict[str, Any]
    loss_history: list[float]


def build_continual_model(
    config: ContinualExperimentConfig,
) -> ZyphraTransformer:
    if not config.variant.memory_enabled:
        return ZyphraTransformer(config.model)
    assert config.variant.segment_length is not None
    return RMTZyphraTransformer(
        config.model,
        memory_tokens=config.variant.memory_tokens,
        segment_length=config.variant.segment_length,
    )


class ContinualTrainer:
    def __init__(self, config: ContinualExperimentConfig):
        config.validate()
        self.config = config
        self.device = resolve_device(config.runtime.device)
        if self.device.type == "cpu" and config.optimization.precision == "fp16":
            raise ValueError("FP16 continual training requires CUDA")
        set_deterministic_seed(
            config.runtime.seed,
            deterministic_algorithms=config.runtime.deterministic_algorithms,
        )
        self.model = build_continual_model(config).to(self.device)
        self.output_dir = Path(config.runtime.output_dir).expanduser().resolve()
        metrics_path = Path(config.runtime.metrics_jsonl).expanduser()
        if not metrics_path.is_absolute():
            metrics_path = self.output_dir / metrics_path
        self.metrics_path = metrics_path.resolve()
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.logger: JsonlMetricLogger | None = None
        self.optimizer: torch.optim.AdamW | None = None
        self.scheduler: LinearWarmupConstantScheduler | None = None
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=(
                self.device.type == "cuda"
                and config.optimization.precision == "fp16"
            ),
        )
        self.state = TrainerState()
        self.source_identity: dict[str, Any] = {}
        self.provenance = checkpoint_metadata()
        self.started_at = time.monotonic()
        self._optimizer_generation = 0
        self._backward_reference_targets = 1
        self.active_memory: torch.Tensor | None = None

    @property
    def config_sha256(self) -> str:
        return canonical_sha256(self.config.to_dict())

    def _autocast(self):
        precision = self.config.optimization.precision
        if precision == "fp32":
            return nullcontext()
        dtype = torch.float16 if precision == "fp16" else torch.bfloat16
        return torch.autocast(device_type=self.device.type, dtype=dtype)

    def _prepare_output(self, *, resume: bool) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        resolved_path = self.output_dir / "resolved_config.yaml"
        if resume:
            if not resolved_path.is_file():
                raise ValueError("Resume output lacks resolved_config.yaml")
        else:
            if self.metrics_path.exists() or resolved_path.exists():
                raise FileExistsError(
                    "Fresh run refuses an existing metrics or resolved config file"
                )
            save_continual_config(self.config, resolved_path)
            with resolved_path.open("rb") as handle:
                os.fsync(handle.fileno())
        tensorboard_dir = self.config.runtime.tensorboard_dir
        if tensorboard_dir is not None:
            tensorboard_path = Path(tensorboard_dir).expanduser()
            if not tensorboard_path.is_absolute():
                tensorboard_path = self.output_dir / tensorboard_path
            tensorboard_dir = str(tensorboard_path.resolve())
        self.logger = JsonlMetricLogger(
            self.metrics_path,
            tensorboard_dir=tensorboard_dir,
            tensorboard_flush_seconds=(
                self.config.runtime.tensorboard_flush_seconds
            ),
            tensorboard_log_every_batches=(
                self.config.runtime.tensorboard_log_every_batches
            ),
        )

    def _log(self, event: str, **values: Any) -> None:
        assert self.logger is not None
        elapsed = max(time.monotonic() - self.started_at, 1e-12)
        record = {
            "event": event,
            "run_name": self.config.run_name,
            "variant": self.config.variant.name,
            "memory_token_count": self.config.variant.memory_tokens,
            "fast_lr": self.config.variant.fast_lr,
            "fast_update_count": self.state.global_fast_updates,
            "task_index": self.state.current_task_index,
            "cycle_index": self.state.cycle_index,
            "language": self.state.language,
            "logical_batch_within_task": (
                self.state.logical_batch_within_task
            ),
            "global_logical_batches": self.state.global_logical_batches,
            "task_slow_steps": self.state.task_slow_steps,
            "global_slow_steps": self.state.global_slow_steps,
            "global_input_tokens": self.state.global_input_tokens,
            "global_valid_targets": self.state.global_valid_targets,
            "wall_time_seconds": elapsed,
            "throughput_input_tokens_per_second": (
                self.state.global_input_tokens / elapsed
            ),
            "learning_rate": (
                None if self.scheduler is None else self.scheduler.current_lr
            ),
            "m0_norm": (
                float(
                    self._rmt_model()
                    .initial_memory.detach()
                    .double()
                    .norm()
                    .cpu()
                )
                if self.config.variant.memory_enabled
                else None
            ),
            "source_position": self.state.source_position,
            **values,
        }
        self.logger.log(record)

    def _planned_slow_steps(self, task: ContinualTaskConfig) -> int:
        logical = task.planned_logical_batches(
            self.config.optimization.global_sequences_per_logical_batch
        )
        return math.ceil(logical / self.config.variant.slow_update_period_k)

    def _create_optimizer_scheduler(
        self, task: ContinualTaskConfig
    ) -> tuple[torch.optim.AdamW, LinearWarmupConstantScheduler]:
        optimization = self.config.optimization
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.model.learning_rate,
            betas=(optimization.adam_beta1, optimization.adam_beta2),
            eps=optimization.adam_epsilon,
            weight_decay=optimization.weight_decay,
        )
        planned_steps = self._planned_slow_steps(task)
        warmup_steps = math.ceil(
            optimization.warmup_fraction * planned_steps
        )
        scheduler = LinearWarmupConstantScheduler(
            optimizer,
            peak_lr=self.config.model.learning_rate,
            planned_steps=planned_steps,
            warmup_steps=warmup_steps,
        )
        self._optimizer_generation += 1
        return optimizer, scheduler

    def _restore_gradients(self, gradients: dict[str, torch.Tensor | None]) -> None:
        parameters = dict(self.model.named_parameters())
        if set(gradients) != set(parameters):
            raise ValueError("Checkpoint gradient parameter names differ")
        for name, parameter in parameters.items():
            gradient = gradients[name]
            parameter.grad = (
                None
                if gradient is None
                else gradient.to(device=self.device, dtype=parameter.dtype)
            )

    def _gradient_state(self) -> dict[str, torch.Tensor | None]:
        return {
            name: (
                None
                if parameter.grad is None
                else parameter.grad.detach().cpu().clone()
            )
            for name, parameter in self.model.named_parameters()
        }

    def _rmt_model(self) -> RMTZyphraTransformer:
        if not isinstance(self.model, RMTZyphraTransformer):
            raise RuntimeError("Memory operation requested for a clean model")
        return self.model

    def _memory_checkpoint_state(self) -> dict[str, Any]:
        initial_memory = (
            self._rmt_model().initial_memory.detach().cpu().clone()
            if self.config.variant.memory_enabled
            else None
        )
        active = (
            None
            if self.active_memory is None
            else self.active_memory.detach().cpu().clone()
        )
        active_gradient = (
            None
            if self.active_memory is None or self.active_memory.grad is None
            else self.active_memory.grad.detach().cpu().clone()
        )
        return {
            "variant": self.config.variant.name,
            "initial_memory": initial_memory,
            "active_memory": active,
            "active_memory_gradient": active_gradient,
            "fast_update_phase": (
                "ready_for_next_logical_batch"
                if self.config.variant.persistent_fast_memory
                else "not_applicable"
            ),
            "fast_lr": self.config.variant.fast_lr,
            "fast_clip_threshold": (
                self.config.variant.fast_memory_grad_clip_norm
            ),
            "reset_policy": self.config.variant.reset_policy,
            "segment_length": self.config.variant.segment_length,
            "memory_token_count": self.config.variant.memory_tokens,
            "memory_evaluation_policy": (
                self.config.variant.memory_evaluation_policy
            ),
        }

    def _restore_memory_state(self, values: dict[str, Any]) -> None:
        if values["variant"] != self.config.variant.name:
            raise ValueError("Checkpoint model variant differs from configuration")
        if self.config.variant.memory_enabled:
            expected_m0 = self._rmt_model().initial_memory.detach().cpu()
            if values["initial_memory"] is None:
                raise ValueError("Memory checkpoint lacks M0")
            torch.testing.assert_close(
                values["initial_memory"],
                expected_m0,
                rtol=0,
                atol=0,
            )
        if self.config.variant.persistent_fast_memory:
            active = values["active_memory"]
            if active is None:
                raise ValueError("FastMem checkpoint lacks active memory")
            expected_shape = (
                self.config.variant.memory_tokens,
                self.config.model.hidden_size,
            )
            if tuple(active.shape) != expected_shape:
                raise ValueError("Checkpoint active-memory shape is invalid")
            self.active_memory = (
                active.to(
                    device=self.device,
                    dtype=self._rmt_model().initial_memory.dtype,
                )
                .detach()
                .clone()
                .requires_grad_(True)
            )
            gradient = values["active_memory_gradient"]
            if gradient is not None:
                self.active_memory.grad = gradient.to(
                    device=self.device,
                    dtype=self.active_memory.dtype,
                )
        else:
            if values["active_memory"] is not None:
                raise ValueError("Non-FastMem checkpoint contains active memory")
            self.active_memory = None

    def _checkpoint_payload(self) -> dict[str, Any]:
        rng_state = capture_rng_state()
        return {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "checkpoint_kind": CHECKPOINT_KIND,
            "model_state": {
                name: tensor.detach().cpu()
                for name, tensor in self.model.state_dict().items()
            },
            "optimizer_state": (
                None if self.optimizer is None else self.optimizer.state_dict()
            ),
            "scheduler_state": (
                None if self.scheduler is None else self.scheduler.state_dict()
            ),
            "gradients": self._gradient_state(),
            "scaler_state": self.scaler.state_dict(),
            "trainer_state": self.state.to_dict(),
            "rng_state": rng_state,
            "resolved_config": self.config.to_dict(),
            "config_sha256": self.config_sha256,
            "source_identity": self.source_identity,
            "provenance": self.provenance,
            "memory_state": self._memory_checkpoint_state(),
            "distributed_state": {
                "schema_version": 1,
                "enabled": False,
                "backend": None,
                "world_size": 1,
                "rank_topology": [
                    {
                        "rank": 0,
                        "local_rank": 0,
                        "device": str(self.device),
                    }
                ],
                "global_logical_batch_size": (
                    self.config.optimization
                    .global_sequences_per_logical_batch
                ),
                "partition_rule": None,
                "rank_rng_states": [
                    {"rank": 0, "state": rng_state}
                ],
                "global_source_position": self.state.source_position,
                "global_input_tokens": self.state.global_input_tokens,
                "global_valid_targets": self.state.global_valid_targets,
                "reduction_policy": "single_process_sum_then_normalize_v1",
                "ddp": None,
                "active_memory_sync_policy": None,
                "state_digests": None,
            },
        }

    def _save_checkpoint(self, filename: str) -> tuple[str, str]:
        path = self.checkpoint_dir / filename
        checksum = atomic_save_checkpoint(path, self._checkpoint_payload())
        self._log(
            "checkpoint",
            checkpoint_path=str(path),
            checkpoint_sha256=checksum,
            source_position=self.state.source_position,
        )
        return str(path), checksum

    def _load_resume(
        self, checkpoint_path: str | Path
    ) -> dict[str, Any]:
        payload = load_checkpoint(checkpoint_path, map_location="cpu")
        if payload["config_sha256"] != self.config_sha256:
            raise ValueError("Resume configuration does not match checkpoint")
        self.model.load_state_dict(payload["model_state"])
        self.state = TrainerState.from_dict(payload["trainer_state"])
        self._restore_memory_state(payload["memory_state"])
        restore_rng_state(payload["rng_state"])
        return payload

    def _normalized_gradient_norm(self) -> float:
        if self.state.window_valid_targets <= 0:
            raise RuntimeError("Cannot normalize an empty slow-gradient window")
        if self.scaler.is_enabled():
            multiplier = (
                self._backward_reference_targets
                / self.state.window_valid_targets
            )
        else:
            multiplier = 1.0 / self.state.window_valid_targets
        for parameter in self.model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(multiplier)
        norm = tensor_norm(
            parameter.grad
            for parameter in self.model.parameters()
            if parameter.grad is not None
        )
        if not math.isfinite(norm):
            raise FloatingPointError("Non-finite slow-gradient norm")
        return norm

    def _optimizer_step(self, *, tail_flush: bool) -> dict[str, Any]:
        assert self.optimizer is not None and self.scheduler is not None
        if self.scaler.is_enabled():
            self.scaler.unscale_(self.optimizer)
        gradient_norm = self._normalized_gradient_norm()
        learning_rate = self.scheduler.current_lr
        if self.scaler.is_enabled():
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        self.scheduler.step()
        self.state.task_slow_steps += 1
        self.state.global_slow_steps += 1
        result = {
            "loss_sum": self.state.window_loss_sum,
            "valid_target_count": self.state.window_valid_targets,
            "mean_loss": (
                self.state.window_loss_sum / self.state.window_valid_targets
            ),
            "learning_rate": learning_rate,
            "gradient_norm": gradient_norm,
            "parameter_norm": tensor_norm(self.model.parameters()),
            "window_logical_batches": self.state.window_logical_batches,
            "tail_flush": tail_flush,
        }
        self.optimizer.zero_grad(set_to_none=True)
        self.state.window_logical_batches = 0
        self.state.window_valid_targets = 0
        self.state.window_loss_sum = 0.0
        self._log("optimizer_step", **result)
        return result

    def _reset_task_memory(self) -> None:
        if not self.config.variant.memory_enabled:
            self.active_memory = None
            return
        m0 = self._rmt_model().initial_memory
        self.state.memory_reset_count += 1
        if self.config.variant.persistent_fast_memory:
            self.active_memory = (
                m0.detach().clone().requires_grad_(True)
            )
            active_norm = float(self.active_memory.detach().norm().cpu())
            self.state.last_active_memory_norm = active_norm
        else:
            self.active_memory = None
            active_norm = None
        self._log(
            "memory_reset",
            reset_event="task_boundary_from_m0_stopgrad",
            active_memory_norm=active_norm,
            initial_memory_norm=float(m0.detach().norm().cpu()),
        )

    def _fast_effective_memory(self) -> torch.Tensor:
        if not self.config.variant.persistent_fast_memory:
            raise RuntimeError("Fast effective memory requested outside FastMem")
        if self.active_memory is None:
            raise RuntimeError("FastMem active root is not initialized")
        m0 = self._rmt_model().initial_memory
        return self.active_memory + m0 - m0.detach()

    def _forward_model(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        *,
        ignore_index: int,
        evaluation_root: torch.Tensor | None = None,
    ):
        if not self.config.variant.memory_enabled:
            return self.model(
                input_ids,
                labels,
                ignore_index=ignore_index,
            )
        root = evaluation_root
        if root is None and self.config.variant.persistent_fast_memory:
            root = self._fast_effective_memory()
        return self._rmt_model()(
            input_ids,
            labels,
            root_memory=root,
            ignore_index=ignore_index,
        )

    def _apply_fast_update(self, *, valid_target_count: int) -> dict[str, float]:
        if not self.config.variant.persistent_fast_memory:
            raise RuntimeError("Fast update requested outside FastMem")
        if self.active_memory is None or self.active_memory.grad is None:
            raise RuntimeError("FastMem active root did not receive a gradient")
        if valid_target_count <= 0:
            raise RuntimeError("Fast update requires valid targets")
        gradient = self.active_memory.grad.detach().clone()
        if self.scaler.is_enabled():
            gradient.mul_(
                self._backward_reference_targets / self.scaler.get_scale()
            )
        gradient.div_(valid_target_count)
        before = float(gradient.double().norm().cpu())
        if not math.isfinite(before):
            raise FloatingPointError("Non-finite active-memory gradient norm")
        threshold = self.config.variant.fast_memory_grad_clip_norm
        if threshold is None:
            raise RuntimeError("FastMem requires a gradient clip threshold")
        scale = min(1.0, threshold / max(before, 1e-30))
        clipped = gradient.mul(scale)
        after = float(clipped.double().norm().cpu())
        with torch.no_grad():
            next_memory = self.active_memory - (
                self.config.variant.fast_lr * clipped
            )
        self.active_memory = (
            next_memory.detach().clone().requires_grad_(True)
        )
        active_norm = float(self.active_memory.detach().double().norm().cpu())
        self.state.global_fast_updates += 1
        self.state.task_fast_updates += 1
        self.state.last_active_memory_grad_norm = before
        self.state.last_active_memory_clipped_grad_norm = after
        self.state.last_active_memory_norm = active_norm
        self.state.fast_gradient_norm_history.append(before)
        self.state.fast_clipped_gradient_norm_history.append(after)
        self.state.fast_memory_norm_history.append(active_norm)
        return {
            "active_memory_gradient_norm_before_clip": before,
            "active_memory_gradient_norm_after_clip": after,
            "active_memory_norm": active_norm,
        }

    def _train_logical_batch(
        self,
        batch: TokenBatch,
        *,
        ignore_index: int,
    ) -> dict[str, Any]:
        started = time.monotonic()
        input_ids = torch.from_numpy(batch.input_ids)
        labels = torch.from_numpy(batch.labels)
        microbatch = self.config.optimization.physical_microbatch_sequences
        loss_sum = 0.0
        target_count = 0
        segment_norm_sums: list[float] = []
        microbatch_count = 0
        self.model.train()
        for start in range(0, len(input_ids), microbatch):
            end = min(start + microbatch, len(input_ids))
            micro_inputs = input_ids[start:end].to(self.device)
            micro_labels = labels[start:end].to(self.device)
            with self._autocast():
                output = self._forward_model(
                    micro_inputs,
                    micro_labels,
                    ignore_index=ignore_index,
                )
            if output.loss_sum is None:
                raise RuntimeError("Training forward did not return loss")
            if not torch.isfinite(output.loss_sum):
                raise FloatingPointError("Non-finite training loss")
            if self.scaler.is_enabled():
                self.scaler.scale(
                    output.loss_sum / self._backward_reference_targets
                ).backward()
            else:
                output.loss_sum.backward()
            loss_sum += float(output.loss_sum.detach().cpu())
            target_count += int(output.target_count.detach().cpu())
            if output.segment_write_memories is not None:
                if not segment_norm_sums:
                    segment_norm_sums = [
                        0.0 for _ in output.segment_write_memories
                    ]
                for index, memory in enumerate(
                    output.segment_write_memories
                ):
                    segment_norm_sums[index] += float(
                        memory.detach().double().norm().cpu()
                    )
                microbatch_count += 1
        if target_count != batch.valid_target_count:
            raise RuntimeError(
                "Model valid-target count differs from source contract"
            )
        if target_count <= 0:
            raise RuntimeError("Logical batch contains no valid targets")
        fast_metrics: dict[str, float | None]
        if self.config.variant.persistent_fast_memory:
            fast_metrics = self._apply_fast_update(
                valid_target_count=target_count
            )
        else:
            fast_metrics = {
                "active_memory_gradient_norm_before_clip": None,
                "active_memory_gradient_norm_after_clip": None,
                "active_memory_norm": None,
            }
        segment_norms = [
            {
                "segment_index": index,
                "encoded_write_memory_norm": value / microbatch_count,
            }
            for index, value in enumerate(segment_norm_sums)
        ]
        final_write_norm = (
            None
            if not segment_norms
            else segment_norms[-1]["encoded_write_memory_norm"]
        )
        self.state.last_encoded_write_memory_norm = final_write_norm
        elapsed = max(time.monotonic() - started, 1e-12)
        input_tokens = int(batch.input_ids.size)
        self.state.logical_batch_within_task += 1
        self.state.global_logical_batches += 1
        self.state.global_input_tokens += input_tokens
        self.state.global_valid_targets += target_count
        self.state.task_input_tokens += input_tokens
        self.state.task_valid_targets += target_count
        self.state.task_loss_sum += loss_sum
        self.state.window_logical_batches += 1
        self.state.window_valid_targets += target_count
        self.state.window_loss_sum += loss_sum
        self.state.source_position = _position_dict(batch.next_position)
        mean_loss = loss_sum / target_count
        self.state.loss_history.append(mean_loss)
        record = {
            "loss_sum": loss_sum,
            "valid_target_count": target_count,
            "mean_loss": mean_loss,
            "input_token_count": input_tokens,
            "throughput_input_tokens_per_second": input_tokens / elapsed,
            "parameter_norm": tensor_norm(self.model.parameters()),
            "gradient_norm": None,
            "learning_rate": (
                None if self.scheduler is None else self.scheduler.current_lr
            ),
            "source_start_position": _position_dict(batch.start_position),
            "source_position": self.state.source_position,
            "encoded_write_memory_norm": final_write_norm,
            "segment_memory_diagnostics": segment_norms,
            **fast_metrics,
        }
        self._log("logical_batch", **record)
        return record

    def _evaluation_root(self, mode: str) -> torch.Tensor:
        if mode not in {"reset", "carried"}:
            raise ValueError("Memory evaluation mode must be reset or carried")
        m0 = self._rmt_model().initial_memory
        if (
            mode == "carried"
            and self.config.variant.persistent_fast_memory
        ):
            if self.active_memory is None:
                raise RuntimeError("Carried evaluation lacks active memory")
            return self.active_memory.detach().clone()
        return m0.detach().clone()

    def _evaluate_source_once(
        self,
        task: ContinualTaskConfig,
        *,
        memory_evaluation_mode: str | None,
        evaluation_root: torch.Tensor | None,
    ) -> dict[str, Any] | None:
        if task.validation_source is None:
            return None
        source, identity, ignore_index = build_source(
            task.validation_source,
            model_vocab_size=self.config.model.vocab_size,
        )
        iterator = source.iter_batches(
            sequence_length=task.validation_source.sequence_length,
            global_sequences_per_batch=(
                self.config.optimization.global_sequences_per_logical_batch
            ),
        )
        loss_sum = 0.0
        targets = 0
        inputs = 0
        self.model.eval()
        with torch.no_grad():
            for _ in range(task.validation_logical_batches):
                try:
                    batch = next(iterator)
                except StopIteration as exc:
                    raise RuntimeError(
                        "Validation source ended before its batch budget"
                    ) from exc
                physical = (
                    self.config.optimization.physical_microbatch_sequences
                )
                for start in range(0, batch.input_ids.shape[0], physical):
                    stop = min(start + physical, batch.input_ids.shape[0])
                    tensor_inputs = torch.from_numpy(
                        batch.input_ids[start:stop]
                    ).to(self.device)
                    tensor_labels = torch.from_numpy(
                        batch.labels[start:stop]
                    ).to(self.device)
                    with self._autocast():
                        output = self._forward_model(
                            tensor_inputs,
                            tensor_labels,
                            ignore_index=ignore_index,
                            evaluation_root=evaluation_root,
                        )
                    assert output.loss_sum is not None
                    loss_sum += float(output.loss_sum.detach().cpu())
                    targets += int(output.target_count.detach().cpu())
                    inputs += int(tensor_inputs.numel())
        result = {
            "loss_sum": loss_sum,
            "valid_target_count": targets,
            "mean_loss": loss_sum / targets,
            "input_token_count": inputs,
            "source_identity": identity,
            "memory_evaluation_mode": memory_evaluation_mode,
        }
        self._log("validation", **result)
        return result

    def _evaluate_source(
        self,
        task: ContinualTaskConfig,
    ) -> dict[str, Any] | None:
        if task.validation_source is None:
            return None
        if not self.config.variant.memory_enabled:
            return self._evaluate_source_once(
                task,
                memory_evaluation_mode=None,
                evaluation_root=None,
            )
        training_active_before = (
            None
            if self.active_memory is None
            else self.active_memory.detach().clone()
        )
        model_before = {
            name: value.detach().clone()
            for name, value in self.model.state_dict().items()
        }
        results: dict[str, Any] = {}
        for mode in ("reset", "carried"):
            results[mode] = self._evaluate_source_once(
                task,
                memory_evaluation_mode=mode,
                evaluation_root=self._evaluation_root(mode),
            )
        for name, before in model_before.items():
            torch.testing.assert_close(
                self.model.state_dict()[name],
                before,
                rtol=0,
                atol=0,
            )
        if training_active_before is not None:
            assert self.active_memory is not None
            torch.testing.assert_close(
                self.active_memory,
                training_active_before,
                rtol=0,
                atol=0,
            )
        return results

    def _forgetting_primary_result(
        self, result: dict[str, Any]
    ) -> dict[str, Any]:
        if self.config.variant.memory_enabled:
            primary = result.get("reset")
            if not isinstance(primary, dict):
                raise RuntimeError(
                    "Memory forgetting evaluation lacks reset result"
                )
            return primary
        return result

    def _evaluate_forgetting(
        self,
        *,
        completed_task_index: int,
        current_validation: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if current_validation is None:
            return None
        representatives: dict[str, ContinualTaskConfig] = {}
        for candidate in self.config.tasks[: completed_task_index + 1]:
            if candidate.validation_source is not None:
                representatives.setdefault(candidate.language, candidate)
        if not representatives:
            return None

        current_task = self.config.tasks[completed_task_index]
        evaluations: dict[str, dict[str, Any]] = {}
        for language, representative in representatives.items():
            if language == current_task.language:
                evaluations[language] = self._forgetting_primary_result(
                    current_validation
                )
                continue
            mode = "reset" if self.config.variant.memory_enabled else None
            result = self._evaluate_source_once(
                representative,
                memory_evaluation_mode=mode,
                evaluation_root=(
                    self._evaluation_root("reset")
                    if self.config.variant.memory_enabled
                    else None
                ),
            )
            if result is None:
                raise RuntimeError("Forgetting validation source disappeared")
            evaluations[language] = result

        current_mean_ce = {
            language: float(result["mean_loss"])
            for language, result in evaluations.items()
        }
        summary, first, best = update_forgetting_metrics(
            current_mean_ce,
            first_mean_ce=self.state.forgetting_first_mean_loss,
            best_mean_ce=self.state.forgetting_best_mean_loss,
        )
        for language, result in evaluations.items():
            summary["languages"][language]["validation"] = result
        prior_language_rows = [
            values
            for language, values in summary["languages"].items()
            if language != current_task.language
        ]
        summary.update(
            {
                "boundary_task_index": completed_task_index,
                "boundary_task_number": completed_task_index + 1,
                "cycle_index": current_task.cycle_index,
                "just_trained_language": current_task.language,
                "memory_evaluation_mode": (
                    "reset"
                    if self.config.variant.memory_enabled
                    else "not_applicable"
                ),
                "prior_language_count": len(prior_language_rows),
                "average_prior_language_forgetting_from_best_ce": (
                    None
                    if not prior_language_rows
                    else sum(
                        row["forgetting_from_best_ce"]
                        for row in prior_language_rows
                    )
                    / len(prior_language_rows)
                ),
                "average_prior_language_ce_change_from_first": (
                    None
                    if not prior_language_rows
                    else sum(
                        row["ce_change_from_first"]
                        for row in prior_language_rows
                    )
                    / len(prior_language_rows)
                ),
            }
        )
        self.state.forgetting_first_mean_loss = first
        self.state.forgetting_best_mean_loss = best
        self.state.forgetting_latest_mean_loss = current_mean_ce
        self.state.forgetting_evaluation_count += 1
        self._log("forgetting_evaluation", **summary)
        return summary

    def _batch_iterator(
        self,
        source: Any,
        task: ContinualTaskConfig,
        *,
        start: TokenPosition,
    ):
        sequence_end = (
            None
            if task.train_sequence_prefix_count is None
            else task.train_sequence_offset_count
            + task.train_sequence_prefix_count
        )
        return source.iter_batches(
            sequence_length=task.train_source.sequence_length,
            global_sequences_per_batch=(
                self.config.optimization.global_sequences_per_logical_batch
            ),
            start=start,
            sequence_prefix_count=sequence_end,
        )

    def _run_impl(
        self,
        *,
        resume_checkpoint: str | Path | None = None,
        stop_after_global_logical_batches: int | None = None,
        stop_after_task_boundaries: int | None = None,
    ) -> TrainingResult:
        if (
            stop_after_global_logical_batches is not None
            and stop_after_global_logical_batches <= 0
        ):
            raise ValueError("Stop-after count must be positive")
        if (
            stop_after_task_boundaries is not None
            and stop_after_task_boundaries <= 0
        ):
            raise ValueError("Task-boundary stop count must be positive")
        if (
            stop_after_task_boundaries is not None
            and stop_after_task_boundaries > len(self.config.tasks)
        ):
            raise ValueError(
                "Task-boundary stop count exceeds configured task count"
            )
        self._prepare_output(resume=resume_checkpoint is not None)
        resume_payload = (
            None
            if resume_checkpoint is None
            else self._load_resume(resume_checkpoint)
        )
        if resume_payload is None:
            backbone_parameters = self.config.model.expected_total_parameters
            initial_memory_parameters = (
                self.config.variant.memory_tokens
                * self.config.model.hidden_size
                if self.config.variant.memory_enabled
                else 0
            )
            active_memory_elements = (
                initial_memory_parameters
                if self.config.variant.persistent_fast_memory
                else 0
            )
            self._log(
                "run_start",
                config_sha256=self.config_sha256,
                parameter_count=sum(
                    parameter.numel() for parameter in self.model.parameters()
                ),
                device=str(self.device),
                precision=self.config.optimization.precision,
                fp16_loss_conditioning_multiplier=(
                    self.config.optimization
                    .fp16_loss_conditioning_multiplier
                ),
                backbone_parameter_count=backbone_parameters,
                initial_memory_parameter_count=initial_memory_parameters,
                active_memory_state_elements=active_memory_elements,
                active_memory_in_optimizer=False,
            )
            start_task_index = 0
        else:
            self._log(
                "resume",
                checkpoint_path=str(Path(resume_checkpoint).resolve()),
                checkpoint_config_sha256=resume_payload["config_sha256"],
                source_position=self.state.source_position,
            )
            start_task_index = self.state.next_task_index
            if self.state.phase == "task_active":
                start_task_index = self.state.current_task_index

        completed_boundaries = (
            self.state.current_task_index
            if self.state.phase == "task_active"
            else self.state.next_task_index
        )
        if (
            stop_after_task_boundaries is not None
            and start_task_index < len(self.config.tasks)
            and stop_after_task_boundaries <= completed_boundaries
        ):
            raise ValueError(
                "Task-boundary stop count must exceed boundaries already "
                "completed by the resume checkpoint"
            )

        last_checkpoint = ""
        last_checksum = ""
        if start_task_index >= len(self.config.tasks):
            self._log("run_end", status="complete")
            return TrainingResult(
                status="complete",
                checkpoint_path=str(Path(resume_checkpoint).resolve()),
                checkpoint_sha256="",
                state=self.state.to_dict(),
                loss_history=list(self.state.loss_history),
            )

        for task_index in range(start_task_index, len(self.config.tasks)):
            task = self.config.tasks[task_index]
            self._backward_reference_targets = (
                conditioned_backward_reference_targets(
                    slow_update_period_k=(
                        self.config.variant.slow_update_period_k
                    ),
                    global_sequences_per_logical_batch=(
                        self.config.optimization
                        .global_sequences_per_logical_batch
                    ),
                    sequence_length=task.train_source.sequence_length,
                    fp16_loss_conditioning_multiplier=(
                        self.config.optimization
                        .fp16_loss_conditioning_multiplier
                    ),
                )
            )
            continuing = (
                resume_payload is not None
                and self.state.phase == "task_active"
                and self.state.current_task_index == task_index
            )
            source, source_identity, ignore_index = build_source(
                task.train_source,
                model_vocab_size=self.config.model.vocab_size,
            )
            source_identity = _task_window_source_identity(
                source_identity, task
            )
            if continuing:
                if resume_payload["source_identity"] != source_identity:
                    raise ValueError(
                        "Resume source manifest/config identity mismatch"
                    )
                self.source_identity = source_identity
                self.optimizer, self.scheduler = (
                    self._create_optimizer_scheduler(task)
                )
                if resume_payload["optimizer_state"] is None:
                    raise ValueError("Active-task checkpoint lacks optimizer")
                self.optimizer.load_state_dict(
                    resume_payload["optimizer_state"]
                )
                assert resume_payload["scheduler_state"] is not None
                self.scheduler.load_state_dict(
                    resume_payload["scheduler_state"]
                )
                self.scaler.load_state_dict(resume_payload["scaler_state"])
                self._restore_gradients(resume_payload["gradients"])
                start_position = _position_from_dict(
                    self.state.source_position
                )
            else:
                self.state.phase = "task_active"
                self.state.next_task_index = task_index
                self.state.current_task_index = task_index
                self.state.cycle_index = task.cycle_index
                self.state.language = task.language
                self.state.logical_batch_within_task = 0
                self.state.task_slow_steps = 0
                self.state.task_input_tokens = 0
                self.state.task_valid_targets = 0
                self.state.task_loss_sum = 0.0
                self.state.task_fast_updates = 0
                self.state.window_logical_batches = 0
                self.state.window_valid_targets = 0
                self.state.window_loss_sum = 0.0
                initial_position = _source_position_for_sequence(
                    source,
                    sequence_index=task.train_sequence_offset_count,
                    sequence_length=task.train_source.sequence_length,
                )
                self.state.source_position = _position_dict(initial_position)
                self.source_identity = source_identity
                self.optimizer, self.scheduler = (
                    self._create_optimizer_scheduler(task)
                )
                self.optimizer.zero_grad(set_to_none=True)
                start_position = initial_position
                self._reset_task_memory()
                self._log(
                    "task_start",
                    optimizer_generation=self._optimizer_generation,
                    planned_logical_batches=task.planned_logical_batches(
                        self.config.optimization.global_sequences_per_logical_batch
                    ),
                    planned_slow_steps=self.scheduler.planned_steps,
                    warmup_slow_steps=self.scheduler.warmup_steps,
                    learning_rate=self.scheduler.current_lr,
                    backward_reference_targets=(
                        self._backward_reference_targets
                    ),
                    fp16_loss_conditioning_multiplier=(
                        self.config.optimization
                        .fp16_loss_conditioning_multiplier
                    ),
                    source_identity=source_identity,
                )
            resume_payload = None
            planned_batches = task.planned_logical_batches(
                self.config.optimization.global_sequences_per_logical_batch
            )
            iterator = self._batch_iterator(
                source,
                task,
                start=start_position,
            )
            while self.state.logical_batch_within_task < planned_batches:
                try:
                    batch = next(iterator)
                except StopIteration as exc:
                    raise RuntimeError(
                        f"Task {task_index} source ended before configured budget"
                    ) from exc
                self._train_logical_batch(batch, ignore_index=ignore_index)
                if self.state.window_logical_batches == (
                    self.config.variant.slow_update_period_k
                ):
                    self._optimizer_step(tail_flush=False)
                checkpoint_every = (
                    self.config.optimization.checkpoint_every_logical_batches
                )
                if (
                    checkpoint_every
                    and self.state.global_logical_batches % checkpoint_every == 0
                ):
                    last_checkpoint, last_checksum = self._save_checkpoint(
                        f"step-{self.state.global_logical_batches:08d}.pt"
                    )
                if (
                    stop_after_global_logical_batches is not None
                    and self.state.global_logical_batches
                    >= stop_after_global_logical_batches
                ):
                    filename = (
                        f"interrupted-step-"
                        f"{self.state.global_logical_batches:08d}.pt"
                    )
                    if (
                        last_checkpoint
                        and Path(last_checkpoint).name
                        == f"step-{self.state.global_logical_batches:08d}.pt"
                    ):
                        filename = (
                            f"interrupted-copy-step-"
                            f"{self.state.global_logical_batches:08d}.pt"
                        )
                    last_checkpoint, last_checksum = self._save_checkpoint(
                        filename
                    )
                    self._log("run_end", status="interrupted")
                    return TrainingResult(
                        status="interrupted",
                        checkpoint_path=last_checkpoint,
                        checkpoint_sha256=last_checksum,
                        state=self.state.to_dict(),
                        loss_history=list(self.state.loss_history),
                    )

            if self.state.window_logical_batches:
                self._optimizer_step(tail_flush=True)
            validation = self._evaluate_source(task)
            forgetting = self._evaluate_forgetting(
                completed_task_index=task_index,
                current_validation=validation,
            )
            self._log(
                "task_end",
                loss_sum=self.state.task_loss_sum,
                valid_target_count=self.state.task_valid_targets,
                mean_loss=(
                    self.state.task_loss_sum / self.state.task_valid_targets
                ),
                input_token_count=self.state.task_input_tokens,
                optimizer_generation=self._optimizer_generation,
                validation=validation,
                forgetting=forgetting,
            )
            self.state.phase = "task_boundary"
            self.state.next_task_index = task_index + 1
            last_checkpoint, last_checksum = self._save_checkpoint(
                f"task-{task_index:04d}-{task.language}-boundary.pt"
            )
            if (
                stop_after_task_boundaries is not None
                and task_index + 1 >= stop_after_task_boundaries
                and task_index + 1 < len(self.config.tasks)
            ):
                self._log("run_end", status="interrupted_at_task_boundary")
                return TrainingResult(
                    status="interrupted",
                    checkpoint_path=last_checkpoint,
                    checkpoint_sha256=last_checksum,
                    state=self.state.to_dict(),
                    loss_history=list(self.state.loss_history),
                )

        self._log("run_end", status="complete")
        return TrainingResult(
            status="complete",
            checkpoint_path=last_checkpoint,
            checkpoint_sha256=last_checksum,
            state=self.state.to_dict(),
            loss_history=list(self.state.loss_history),
        )

    def run(
        self,
        *,
        resume_checkpoint: str | Path | None = None,
        stop_after_global_logical_batches: int | None = None,
        stop_after_task_boundaries: int | None = None,
    ) -> TrainingResult:
        try:
            return self._run_impl(
                resume_checkpoint=resume_checkpoint,
                stop_after_global_logical_batches=(
                    stop_after_global_logical_batches
                ),
                stop_after_task_boundaries=stop_after_task_boundaries,
            )
        except BaseException as exc:
            if self.logger is not None and not self.logger.closed:
                self._log(
                    "run_error",
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
            raise
        finally:
            if self.logger is not None:
                self.logger.close()


def evaluate_clean_checkpoint(
    config: ContinualExperimentConfig,
    checkpoint_path: str | Path,
    *,
    task_index: int,
) -> dict[str, Any]:
    config.validate()
    if task_index < 0 or task_index >= len(config.tasks):
        raise ValueError("Evaluation task index is outside configuration")
    task = config.tasks[task_index]
    if task.validation_source is None:
        raise ValueError("Selected task has no configured validation source")
    device = resolve_device(config.runtime.device)
    payload = load_checkpoint(checkpoint_path, map_location="cpu")
    if payload["config_sha256"] != canonical_sha256(config.to_dict()):
        raise ValueError("Evaluation configuration differs from checkpoint")
    model = build_continual_model(config).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()

    def evaluate_once(
        mode: str | None,
        root: torch.Tensor | None,
    ) -> dict[str, Any]:
        source, identity, ignore_index = build_source(
            task.validation_source,
            model_vocab_size=config.model.vocab_size,
        )
        iterator = source.iter_batches(
            sequence_length=task.validation_source.sequence_length,
            global_sequences_per_batch=(
                config.optimization.global_sequences_per_logical_batch
            ),
        )
        loss_sum = 0.0
        targets = 0
        inputs = 0
        precision = config.optimization.precision

        def evaluation_autocast():
            if precision == "fp32":
                return nullcontext()
            return torch.autocast(
                device_type=device.type,
                dtype=(
                    torch.float16
                    if precision == "fp16"
                    else torch.bfloat16
                ),
            )

        with torch.no_grad():
            for _ in range(task.validation_logical_batches):
                try:
                    batch = next(iterator)
                except StopIteration as exc:
                    raise RuntimeError(
                        "Evaluation source ended before budget"
                    ) from exc
                physical = config.optimization.physical_microbatch_sequences
                for start in range(0, batch.input_ids.shape[0], physical):
                    stop = min(start + physical, batch.input_ids.shape[0])
                    tensor_inputs = torch.from_numpy(
                        batch.input_ids[start:stop]
                    ).to(device)
                    tensor_labels = torch.from_numpy(
                        batch.labels[start:stop]
                    ).to(device)
                    with evaluation_autocast():
                        if isinstance(model, RMTZyphraTransformer):
                            output = model(
                                tensor_inputs,
                                tensor_labels,
                                root_memory=root,
                                ignore_index=ignore_index,
                            )
                        else:
                            output = model(
                                tensor_inputs,
                                tensor_labels,
                                ignore_index=ignore_index,
                            )
                    assert output.loss_sum is not None
                    loss_sum += float(output.loss_sum.detach().cpu())
                    targets += int(output.target_count.detach().cpu())
                    inputs += int(tensor_inputs.numel())
        return {
            "checkpoint_path": str(Path(checkpoint_path).resolve()),
            "task_index": task_index,
            "language": task.language,
            "loss_sum": loss_sum,
            "valid_target_count": targets,
            "mean_loss": loss_sum / targets,
            "input_token_count": inputs,
            "source_identity": identity,
            "memory_evaluation_mode": mode,
        }

    if not config.variant.memory_enabled:
        return evaluate_once(None, None)
    assert isinstance(model, RMTZyphraTransformer)
    reset_root = model.initial_memory.detach().clone()
    active = payload["memory_state"]["active_memory"]
    carried_root = (
        reset_root.clone()
        if active is None
        else active.to(device=device, dtype=reset_root.dtype).detach().clone()
    )
    return {
        "checkpoint_path": str(Path(checkpoint_path).resolve()),
        "task_index": task_index,
        "language": task.language,
        "variant": config.variant.name,
        "reset": evaluate_once("reset", reset_root),
        "carried": evaluate_once("carried", carried_root),
    }
