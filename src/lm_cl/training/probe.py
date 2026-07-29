from __future__ import annotations

import math
import os
import time
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist

from lm_cl.config import ProbeExperimentConfig, VariantConfig, save_probe_config
from lm_cl.config.schema import strict_dataclass
from lm_cl.data import TokenPosition, validate_packed_shards
from lm_cl.data.packed import PackedShardSource
from lm_cl.data.tokenizer import load_tokenizer_manifest
from lm_cl.metrics import (
    JsonlMetricLogger,
    compute_probe_auc_report,
)
from lm_cl.training.checkpoint import (
    canonical_sha256,
    capture_rng_state,
    checkpoint_metadata,
    restore_rng_state,
    sha256_file,
    validate_checkpoint_payload,
)
from lm_cl.training.continual import (
    ContinualTrainer,
    TrainerState,
    TrainingResult,
    _position_dict,
    _position_from_dict,
    build_continual_model,
    build_source,
    conditioned_backward_reference_targets,
    resolve_device,
    tensor_norm,
)
from lm_cl.training.distributed import (
    DistributedContext,
    all_reduce_float,
    all_reduce_int,
    assert_digest_equal,
    collective_raise_if_any,
    iter_partitioned_batches,
    state_digest,
)
from lm_cl.training.distributed_continual import DistributedContinualTrainer
from lm_cl.training.probe_checkpoint import (
    PROBE_CHECKPOINT_KIND,
    PROBE_CHECKPOINT_SCHEMA_VERSION,
    atomic_save_probe_checkpoint,
    atomic_write_json,
    load_probe_checkpoint,
)
from lm_cl.training.scheduler import LinearWarmupConstantScheduler
from lm_cl.training.seed import set_deterministic_seed


PROBE_INITIALIZATION_POLICY = "boundary_model_m0_fresh_probe_state_v1"
LEGACY_CLEAN_SOURCE_KIND = "lm-cl-clean-continual-checkpoint"


def _load_probe_source_payload(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ValueError(f"Checkpoint cannot be loaded: {path}") from exc
    if (
        isinstance(payload, dict)
        and payload.get("checkpoint_schema_version") == 1
        and payload.get("checkpoint_kind") == LEGACY_CLEAN_SOURCE_KIND
    ):
        required = {
            "model_state",
            "optimizer_state",
            "scheduler_state",
            "gradients",
            "scaler_state",
            "trainer_state",
            "rng_state",
            "resolved_config",
            "config_sha256",
            "source_identity",
            "provenance",
        }
        missing = sorted(required - set(payload))
        if missing:
            raise ValueError(
                f"Legacy clean checkpoint is missing fields: {missing}"
            )
        if canonical_sha256(payload["resolved_config"]) != (
            payload["config_sha256"]
        ):
            raise ValueError(
                "Legacy clean source configuration checksum mismatch"
            )
        variant = payload["resolved_config"].get("variant")
        state = payload["trainer_state"]
        if not isinstance(state, dict):
            raise ValueError("Legacy clean source trainer state is invalid")
        for counter in (
            "next_task_index",
            "global_logical_batches",
            "global_slow_steps",
            "global_input_tokens",
            "global_valid_targets",
            "window_logical_batches",
            "window_valid_targets",
        ):
            value = state.get(counter)
            if not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"Legacy clean source counter {counter} is invalid"
                )
        if (
            not isinstance(payload["model_state"], dict)
            or not isinstance(payload["gradients"], dict)
            or not set(payload["gradients"]).issubset(
                payload["model_state"]
            )
        ):
            raise ValueError(
                "Legacy clean source model/gradient names are inconsistent"
            )
        if (
            not isinstance(variant, dict)
            or variant.get("memory_enabled")
            or variant.get("persistent_fast_memory")
            or variant.get("name")
            not in {"backbone_clean", "backbone_matched_k"}
            or "initial_memory" in payload["model_state"]
        ):
            raise ValueError("Legacy clean source has invalid memory metadata")
        payload = dict(payload)
        payload["memory_state"] = {
            "variant": variant["name"],
            "initial_memory": None,
            "active_memory": None,
            "active_memory_gradient": None,
            "fast_update_phase": "not_applicable",
            "fast_lr": 0.0,
            "fast_clip_threshold": None,
            "reset_policy": None,
            "segment_length": None,
            "memory_token_count": 0,
            "memory_evaluation_policy": None,
        }
        return payload
    return validate_checkpoint_payload(payload)


def _validate_source_data_identity(payload: dict[str, Any]) -> None:
    identity = payload.get("source_identity")
    if not isinstance(identity, dict):
        raise ValueError("Probe source checkpoint lacks source identity")
    base_identity = dict(identity)
    sequence_window = base_identity.pop("sequence_window", None)
    if sequence_window is not None:
        required_window_fields = {
            "sequence_start",
            "sequence_count",
            "sequence_end_exclusive",
            "input_token_count",
            "view_sha256",
        }
        if (
            not isinstance(sequence_window, dict)
            or set(sequence_window) != required_window_fields
        ):
            raise ValueError("Probe source sequence-window fields are invalid")
        values = {
            key: sequence_window[key]
            for key in (
                "sequence_start",
                "sequence_count",
                "sequence_end_exclusive",
                "input_token_count",
            )
        }
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in values.values()
        ):
            raise ValueError("Probe source sequence-window values must be integers")
        if values["sequence_start"] < 0 or values["sequence_count"] <= 0:
            raise ValueError("Probe source sequence window is empty or negative")
        if values["sequence_end_exclusive"] != (
            values["sequence_start"] + values["sequence_count"]
        ):
            raise ValueError("Probe source sequence-window bounds are inconsistent")
        task_index = payload["trainer_state"]["current_task_index"]
        tasks = payload["resolved_config"]["tasks"]
        if (
            isinstance(task_index, bool)
            or not isinstance(task_index, int)
            or task_index < 0
            or task_index >= len(tasks)
        ):
            raise ValueError("Probe source task index is invalid")
        task = tasks[task_index]
        source = task["train_source"]
        source_config = (
            source.get("synthetic")
            if source.get("kind") == "synthetic"
            else source.get("packed")
        )
        if not isinstance(source_config, dict):
            raise ValueError("Probe source task data configuration is invalid")
        reader = source_config.get("reader", source_config)
        sequence_length = reader.get("sequence_length")
        if (
            isinstance(sequence_length, bool)
            or not isinstance(sequence_length, int)
            or sequence_length <= 1
        ):
            raise ValueError("Probe source sequence length is invalid")
        if values["input_token_count"] != (
            values["sequence_count"] * sequence_length
        ):
            raise ValueError("Probe source sequence-window token count differs")
        unhashed_window = dict(sequence_window)
        claimed_view_sha256 = unhashed_window.pop("view_sha256")
        expected_view_sha256 = canonical_sha256(
            {"source": base_identity, "window": unhashed_window}
        )
        if claimed_view_sha256 != expected_view_sha256:
            raise ValueError("Probe source sequence-window SHA-256 differs")
    if identity.get("kind") == "synthetic":
        if set(base_identity) != {"kind", "config_sha256"}:
            raise ValueError("Probe synthetic source identity fields are invalid")
        checksum = identity.get("config_sha256")
        if (
            not isinstance(checksum, str)
            or len(checksum) != 64
            or any(character not in "0123456789abcdef" for character in checksum)
        ):
            raise ValueError("Probe synthetic source identity is invalid")
        return
    if identity.get("kind") != "packed_shards":
        raise ValueError("Probe source checkpoint has unknown source identity")
    required = {
        "kind",
        "manifest_path",
        "manifest_content_sha256",
        "ordered_data_sha256",
        "tokenizer_manifest_path",
        "tokenizer_manifest_sha256",
    }
    if set(base_identity) != required:
        raise ValueError("Probe packed source identity fields are incomplete")
    manifest_path = Path(identity["manifest_path"]).expanduser().resolve()
    if manifest_path.name != "manifest.json" or not manifest_path.is_file():
        raise ValueError("Probe packed source manifest path is invalid")
    report = validate_packed_shards(
        manifest_path.parent,
        expected_tokenizer_manifest_sha256=(
            identity["tokenizer_manifest_sha256"]
        ),
    )
    if report["manifest_content_sha256"] != (
        identity["manifest_content_sha256"]
    ):
        raise ValueError("Probe packed source manifest hash differs")
    if report["ordered_data_sha256"] != identity["ordered_data_sha256"]:
        raise ValueError("Probe packed source ordered-data hash differs")
    if sequence_window is not None and sequence_window[
        "sequence_end_exclusive"
    ] > (report["token_count"] // sequence_length):
        raise ValueError("Probe source sequence window exceeds packed data")
    tokenizer_path = Path(
        identity["tokenizer_manifest_path"]
    ).expanduser().resolve()
    tokenizer = load_tokenizer_manifest(tokenizer_path)
    if tokenizer["manifest_content_sha256"] != (
        identity["tokenizer_manifest_sha256"]
    ):
        raise ValueError("Probe source tokenizer-manifest hash differs")


class _ProbeTask:
    def __init__(self, config: ProbeExperimentConfig):
        self.train_source = config.train_source
        self.validation_source = config.validation_source
        self.language = "vi"
        self.task_index = 0
        self.cycle_index = 0
        self._planned = config.planned_logical_batches

    def planned_logical_batches(self, _: int) -> int:
        return self._planned


def validate_probe_source_checkpoint(
    config: ProbeExperimentConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if config.source_checkpoint_status == "pending":
        raise ValueError(
            "Probe source checkpoint identity is pending; freeze the Russian "
            "boundary path and SHA-256 before launch"
        )
    path = Path(config.source_checkpoint).expanduser().resolve()
    checksum = sha256_file(path)
    if (
        config.source_checkpoint_status == "frozen"
        and checksum != config.source_checkpoint_sha256
    ):
        raise ValueError("Probe source checkpoint SHA-256 differs")
    payload = _load_probe_source_payload(path)
    _validate_source_data_identity(payload)
    state = payload["trainer_state"]
    memory = payload["memory_state"]
    if state["phase"] != "task_boundary":
        raise ValueError("Probe source must be a task-boundary checkpoint")
    if (
        state["window_logical_batches"] != 0
        or state["window_valid_targets"] != 0
        or float(state["window_loss_sum"]) != 0.0
    ):
        raise ValueError("Probe source contains a partial slow K-window")
    if any(value is not None for value in payload["gradients"].values()):
        raise ValueError("Probe source contains partial slow gradients")
    if memory["fast_update_phase"] not in {
        "not_applicable",
        "ready_for_next_logical_batch",
    }:
        raise ValueError("Probe source FastMem phase is not stable")
    if payload["resolved_config"]["model"] != asdict(config.model):
        raise ValueError("Probe model architecture differs from source")
    source_variant = strict_dataclass(
        VariantConfig,
        payload["resolved_config"]["variant"],
        "source variant",
    )
    source_variant.validate()
    if asdict(source_variant) != asdict(config.variant):
        raise ValueError("Probe variant differs from source checkpoint")
    boundary = {
        "phase": state["phase"],
        "task_index": state["current_task_index"],
        "next_task_index": state["next_task_index"],
        "cycle_index": state["cycle_index"],
        "language": state["language"],
        "global_logical_batches": state["global_logical_batches"],
        "global_slow_steps": state["global_slow_steps"],
        "global_fast_updates": state.get("global_fast_updates", 0),
        "global_input_tokens": state["global_input_tokens"],
        "global_valid_targets": state["global_valid_targets"],
        "source_config_sha256": payload["config_sha256"],
        "source_identity": payload["source_identity"],
        "fast_update_phase": memory["fast_update_phase"],
    }
    identity = {
        "path": str(path),
        "sha256": checksum,
        "checkpoint_kind": payload["checkpoint_kind"],
        "checkpoint_schema_version": payload[
            "checkpoint_schema_version"
        ],
        "continual_boundary": boundary,
    }
    return payload, identity


class ProbeTrainer(ContinualTrainer):
    """Isolated Vietnamese probe derived from a stable continual boundary."""

    def __init__(self, config: ProbeExperimentConfig):
        config.validate()
        self.probe_config = config
        effective_variant = replace(
            config.variant,
            fast_lr=config.effective_fast_lr,
        )
        self.config = SimpleNamespace(
            run_name=config.run_name,
            model=config.model,
            variant=effective_variant,
            optimization=config.optimization,
            runtime=config.runtime,
            distributed=config.distributed,
        )
        self.device = resolve_device(config.runtime.device)
        if self.device.type == "cpu" and config.optimization.precision == "fp16":
            raise ValueError("FP16 probe training requires CUDA")
        set_deterministic_seed(
            config.runtime.seed,
            deterministic_algorithms=config.runtime.deterministic_algorithms,
        )
        source_payload, source_identity = validate_probe_source_checkpoint(
            config
        )
        self.source_checkpoint_identity = source_identity
        self.source_checkpoint_hash_before = source_identity["sha256"]
        self.model = build_continual_model(self.config).to(self.device)
        self.model.load_state_dict(source_payload["model_state"])
        self.output_dir = Path(config.runtime.output_dir).expanduser().resolve()
        metrics_path = Path(config.runtime.metrics_jsonl).expanduser()
        if not metrics_path.is_absolute():
            metrics_path = self.output_dir / metrics_path
        self.metrics_path = metrics_path.resolve()
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.results_path = self.output_dir / "probe_results.json"
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
        self.state = TrainerState(phase="probe_active", language="vi")
        self.provenance = checkpoint_metadata()
        self.started_at = time.monotonic()
        self._optimizer_generation = 0
        self._backward_reference_targets = (
            conditioned_backward_reference_targets(
                slow_update_period_k=config.variant.slow_update_period_k,
                global_sequences_per_logical_batch=(
                    config.optimization.global_sequences_per_logical_batch
                ),
                sequence_length=config.sequence_length,
                fp16_loss_conditioning_multiplier=(
                    config.optimization.fp16_loss_conditioning_multiplier
                ),
            )
        )
        self.active_memory: torch.Tensor | None = None
        self.completed_evaluation_steps: list[int] = []
        self.curve_records: list[dict[str, Any]] = []
        self._task = _ProbeTask(config)
        (
            self.train_source,
            self.training_source_identity,
            self.train_ignore_index,
        ) = build_source(
            config.train_source,
            model_vocab_size=config.model.vocab_size,
        )
        (
            self.validation_source,
            self.validation_source_identity,
            self.validation_ignore_index,
        ) = build_source(
            config.validation_source,
            model_vocab_size=config.model.vocab_size,
        )
        prefix = config.train_sequence_prefix_count
        if prefix is not None:
            if not isinstance(self.train_source, PackedShardSource):
                raise ValueError(
                    "Probe sequence prefix requires a packed shard source"
                )
            available = self.train_source.token_count // config.sequence_length
            if prefix > available:
                raise ValueError(
                    "train_sequence_prefix_count exceeds available complete sequences"
                )

    @property
    def config_sha256(self) -> str:
        from lm_cl.training.checkpoint import canonical_sha256

        return canonical_sha256(self.probe_config.to_dict())

    def _assert_source_unchanged(self) -> str:
        actual = sha256_file(self.probe_config.source_checkpoint)
        if actual != self.source_checkpoint_hash_before:
            raise RuntimeError(
                "Immutable source checkpoint changed during probing"
            )
        return actual

    def _prepare_output(self, *, resume: bool) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        resolved = self.output_dir / "resolved_probe_config.yaml"
        if resume:
            if not resolved.is_file():
                raise ValueError("Probe resume output lacks resolved config")
        else:
            if (
                self.metrics_path.exists()
                or resolved.exists()
                or self.results_path.exists()
            ):
                raise FileExistsError(
                    "Fresh probe refuses existing output files"
                )
            save_probe_config(self.probe_config, resolved)
            with resolved.open("rb") as handle:
                os.fsync(handle.fileno())
        tensorboard_dir = self.probe_config.runtime.tensorboard_dir
        if tensorboard_dir is not None:
            tensorboard_path = Path(tensorboard_dir).expanduser()
            if not tensorboard_path.is_absolute():
                tensorboard_path = self.output_dir / tensorboard_path
            tensorboard_dir = str(tensorboard_path.resolve())
        self.logger = JsonlMetricLogger(
            self.metrics_path,
            tensorboard_dir=tensorboard_dir,
            tensorboard_flush_seconds=(
                self.probe_config.runtime.tensorboard_flush_seconds
            ),
            tensorboard_log_every_batches=(
                self.probe_config.runtime.tensorboard_log_every_batches
            ),
        )

    def _log(self, event: str, **values: Any) -> None:
        values.setdefault("phase7_probe", True)
        values.setdefault("probe_mode", self.probe_config.probe_mode)
        values.setdefault(
            "configured_fast_lr",
            self.probe_config.variant.fast_lr,
        )
        values.setdefault(
            "effective_probe_fast_lr",
            self.probe_config.effective_fast_lr,
        )
        values.setdefault(
            "source_checkpoint_sha256",
            self.source_checkpoint_hash_before,
        )
        super()._log(
            event,
            **values,
        )

    def _create_probe_optimizer_scheduler(self) -> None:
        optimization = self.probe_config.optimization
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.probe_config.model.learning_rate,
            betas=(optimization.adam_beta1, optimization.adam_beta2),
            eps=optimization.adam_epsilon,
            weight_decay=optimization.weight_decay,
        )
        planned = self.probe_config.planned_slow_steps
        warmup = math.ceil(optimization.warmup_fraction * planned)
        self.scheduler = LinearWarmupConstantScheduler(
            self.optimizer,
            peak_lr=self.probe_config.model.learning_rate,
            planned_steps=planned,
            warmup_steps=warmup,
        )
        self._optimizer_generation += 1

    def _memory_checkpoint_state(self) -> dict[str, Any]:
        values = super()._memory_checkpoint_state()
        values.update(
            {
                "configured_fast_lr": self.probe_config.variant.fast_lr,
                "fast_lr_override": self.probe_config.fast_lr_override,
            }
        )
        return values

    def _checkpoint_payload(self) -> dict[str, Any]:
        rng = capture_rng_state()
        return {
            "checkpoint_schema_version": PROBE_CHECKPOINT_SCHEMA_VERSION,
            "checkpoint_kind": PROBE_CHECKPOINT_KIND,
            "model_state": {
                name: value.detach().cpu()
                for name, value in self.model.state_dict().items()
            },
            "optimizer_state": (
                None if self.optimizer is None else self.optimizer.state_dict()
            ),
            "scheduler_state": (
                None if self.scheduler is None else self.scheduler.state_dict()
            ),
            "gradients": self._gradient_state(),
            "scaler_state": self.scaler.state_dict(),
            "probe_state": self.state.to_dict(),
            "rng_state": rng,
            "resolved_config": self.probe_config.to_dict(),
            "config_sha256": self.config_sha256,
            "source_checkpoint": self.source_checkpoint_identity,
            "initialization_policy": PROBE_INITIALIZATION_POLICY,
            "training_source_identity": self.training_source_identity,
            "validation_source_identity": self.validation_source_identity,
            "memory_state": self._memory_checkpoint_state(),
            "curve_records": self.curve_records,
            "completed_evaluation_steps": self.completed_evaluation_steps,
            "auc_policy": asdict(self.probe_config.auc),
            "provenance": self.provenance,
            "distributed_state": {
                "schema_version": 1,
                "enabled": False,
                "backend": None,
                "world_size": 1,
                "partition_rule": None,
                "rank_topology": [
                    {
                        "rank": 0,
                        "local_rank": 0,
                        "device": str(self.device),
                    }
                ],
                "rank_rng_states": [{"rank": 0, "state": rng}],
                "global_source_position": self.state.source_position,
                "global_input_tokens": self.state.global_input_tokens,
                "global_valid_targets": self.state.global_valid_targets,
                "reduction_policy": "single_process_sum_then_normalize_v1",
                "active_memory_sync_policy": None,
                "state_digests": None,
            },
        }

    def _save_checkpoint(self, filename: str) -> tuple[str, str]:
        self._assert_source_unchanged()
        path = self.checkpoint_dir / filename
        checksum = atomic_save_probe_checkpoint(
            path,
            self._checkpoint_payload(),
        )
        self._log(
            "probe_checkpoint",
            checkpoint_path=str(path),
            checkpoint_sha256=checksum,
        )
        return str(path), checksum

    def _load_resume(self, checkpoint_path: str | Path) -> dict[str, Any]:
        payload = load_probe_checkpoint(checkpoint_path)
        if payload["config_sha256"] != self.config_sha256:
            raise ValueError("Probe resume configuration differs")
        if payload["source_checkpoint"] != self.source_checkpoint_identity:
            raise ValueError("Probe resume source-checkpoint identity differs")
        if payload["training_source_identity"] != (
            self.training_source_identity
        ):
            raise ValueError("Probe training manifest identity differs")
        if payload["validation_source_identity"] != (
            self.validation_source_identity
        ):
            raise ValueError("Probe validation manifest identity differs")
        distributed = payload["distributed_state"]
        if distributed["enabled"] or distributed["world_size"] != 1:
            raise ValueError("Single-process probe cannot resume DDP state")
        self.model.load_state_dict(payload["model_state"])
        self.state = TrainerState.from_dict(payload["probe_state"])
        self._restore_memory_state(payload["memory_state"])
        assert self.optimizer is not None and self.scheduler is not None
        self.optimizer.load_state_dict(payload["optimizer_state"])
        self.scheduler.load_state_dict(payload["scheduler_state"])
        self.scaler.load_state_dict(payload["scaler_state"])
        self._restore_gradients(payload["gradients"])
        self.curve_records = list(payload["curve_records"])
        self.completed_evaluation_steps = list(
            payload["completed_evaluation_steps"]
        )
        restore_rng_state(payload["rng_state"])
        self._assert_source_unchanged()
        return payload

    def _training_iterator(self, *, start: TokenPosition):
        distributed = getattr(self, "distributed", None)
        if distributed is None:
            return self.train_source.iter_batches(
                sequence_length=self.probe_config.sequence_length,
                global_sequences_per_batch=(
                    self.probe_config.optimization
                    .global_sequences_per_logical_batch
                ),
                start=start,
                sequence_prefix_count=(
                    self.probe_config.train_sequence_prefix_count
                ),
            )
        return iter_partitioned_batches(
            self.train_source,
            sequence_length=self.probe_config.sequence_length,
            global_sequences_per_batch=(
                self.probe_config.optimization
                .global_sequences_per_logical_batch
            ),
            rank=distributed.rank,
            world_size=distributed.world_size,
            start=start,
            sequence_prefix_count=(
                self.probe_config.train_sequence_prefix_count
            ),
        )

    def _evaluation_modes(self) -> list[str]:
        if not self.probe_config.variant.memory_enabled:
            return ["not_applicable"]
        return ["reset", "carried"]

    def _evaluation_root_for_probe(
        self,
        mode: str,
    ) -> torch.Tensor | None:
        if mode == "not_applicable":
            return None
        m0 = self._rmt_model().initial_memory.detach().clone()
        if (
            mode == "carried"
            and self.probe_config.variant.persistent_fast_memory
        ):
            if self.active_memory is None:
                raise RuntimeError("Carried probe evaluation lacks active root")
            return self.active_memory.detach().clone()
        return m0

    def _evaluation_state_digest(self) -> str:
        return state_digest(
            {
                "model": self.model.state_dict(),
                "optimizer": (
                    None
                    if self.optimizer is None
                    else self.optimizer.state_dict()
                ),
                "scheduler": (
                    None
                    if self.scheduler is None
                    else self.scheduler.state_dict()
                ),
                "scaler": self.scaler.state_dict(),
                "gradients": self._gradient_state(),
                "active": self.active_memory,
                "trainer": self.state.to_dict(),
            }
        )

    def _evaluate_one_mode(self, mode: str) -> dict[str, Any]:
        root = self._evaluation_root_for_probe(mode)
        distributed: DistributedContext | None = getattr(
            self,
            "distributed",
            None,
        )
        if distributed is not None and root is not None:
            assert_digest_equal(
                root,
                distributed,
                label=f"{mode} probe evaluation root",
            )
        global_batch = (
            self.probe_config.optimization.global_sequences_per_logical_batch
        )
        batches = self.probe_config.validation_sequences // global_batch
        if distributed is None:
            iterator = self.validation_source.iter_batches(
                sequence_length=self.probe_config.sequence_length,
                global_sequences_per_batch=global_batch,
            )
        else:
            iterator = iter_partitioned_batches(
                self.validation_source,
                sequence_length=self.probe_config.sequence_length,
                global_sequences_per_batch=global_batch,
                rank=distributed.rank,
                world_size=distributed.world_size,
            )
        loss_sum = 0.0
        valid_targets = 0
        input_tokens = 0
        self.model.eval()
        with torch.no_grad():
            for _ in range(batches):
                batch = None
                local_error = None
                try:
                    batch = next(iterator)
                except StopIteration as exc:
                    local_error = (
                        "RuntimeError: Probe validation source ended "
                        "before budget"
                    )
                    if distributed is None:
                        raise RuntimeError(
                            "Probe validation source ended before budget"
                        ) from exc
                except BaseException as exc:
                    local_error = f"{type(exc).__name__}: {exc}"
                if distributed is not None:
                    collective_raise_if_any(
                        local_error,
                        distributed,
                        prefix=(
                            "Distributed probe validation source "
                            "iteration failed"
                        ),
                    )
                assert batch is not None
                input_ids = torch.from_numpy(batch.input_ids)
                labels = torch.from_numpy(batch.labels)
                local_examples = len(input_ids)
                microbatch = (
                    self.probe_config.optimization
                    .physical_microbatch_sequences
                )
                if distributed is None:
                    maximum_local_examples = local_examples
                else:
                    maximum_local_examples = all_reduce_int(
                        local_examples,
                        distributed,
                        op=dist.ReduceOp.MAX,
                    )
                microbatch_slots = max(
                    1,
                    math.ceil(maximum_local_examples / microbatch),
                )
                local_loss = 0.0
                local_targets = 0
                for slot in range(microbatch_slots):
                    start = slot * microbatch
                    end = min(start + microbatch, local_examples)
                    has_examples = start < end
                    if has_examples:
                        inputs = input_ids[start:end].to(self.device)
                        micro_labels = labels[start:end].to(self.device)
                    else:
                        inputs = torch.zeros(
                            (1, self.probe_config.sequence_length),
                            dtype=torch.long,
                            device=self.device,
                        )
                        micro_labels = inputs
                    output = None
                    local_error = None
                    try:
                        with self._autocast():
                            output = self._forward_model(
                                inputs,
                                micro_labels,
                                ignore_index=self.validation_ignore_index,
                                evaluation_root=root,
                            )
                    except BaseException as exc:
                        local_error = f"{type(exc).__name__}: {exc}"
                    if distributed is None:
                        if local_error is not None:
                            raise RuntimeError(
                                "Probe validation forward failed: "
                                + local_error
                            )
                    else:
                        collective_raise_if_any(
                            local_error,
                            distributed,
                            prefix=(
                                "Distributed probe validation forward "
                                "failed"
                            ),
                        )
                    assert output is not None
                    assert output.loss_sum is not None
                    local_nonfinite = int(
                        has_examples
                        and not bool(torch.isfinite(output.loss_sum))
                    )
                    if distributed is None:
                        if local_nonfinite:
                            raise FloatingPointError(
                                "Probe validation loss is non-finite"
                            )
                    elif all_reduce_int(
                        local_nonfinite,
                        distributed,
                        op=dist.ReduceOp.MAX,
                    ):
                        raise FloatingPointError(
                            "A rank produced non-finite probe "
                            "validation loss"
                        )
                    if has_examples:
                        local_loss += float(
                            output.loss_sum.detach().cpu()
                        )
                        local_targets += int(
                            output.target_count.detach().cpu()
                        )
                target_mismatch = (
                    local_targets != batch.valid_target_count
                )
                if distributed is None:
                    if target_mismatch:
                        raise RuntimeError(
                            "Probe validation target count differs from "
                            "the source contract"
                        )
                else:
                    collective_raise_if_any(
                        (
                            "RuntimeError: Probe validation target count "
                            "differs from the source contract"
                            if target_mismatch
                            else None
                        ),
                        distributed,
                        prefix=(
                            "Distributed probe validation accounting "
                            "failed"
                        ),
                    )
                local_inputs = int(batch.input_ids.size)
                if distributed is None:
                    loss_sum += local_loss
                    valid_targets += local_targets
                    input_tokens += local_inputs
                else:
                    loss_sum += all_reduce_float(local_loss, distributed)
                    valid_targets += all_reduce_int(
                        local_targets,
                        distributed,
                    )
                    input_tokens += all_reduce_int(
                        local_inputs,
                        distributed,
                    )
        if input_tokens != (
            self.probe_config.validation_sequences
            * self.probe_config.sequence_length
        ):
            raise RuntimeError("Validation sequence coverage is incomplete")
        if valid_targets <= 0:
            raise RuntimeError("Validation contains no valid targets")
        record = {
            "probe_logical_step": self.state.global_logical_batches,
            "cumulative_input_tokens": self.state.global_input_tokens,
            "cumulative_valid_target_tokens": (
                self.state.global_valid_targets
            ),
            "validation_loss_sum": loss_sum,
            "validation_valid_target_count": valid_targets,
            "mean_validation_ce": loss_sum / valid_targets,
            "validation_input_token_count": input_tokens,
            "probe_mode": self.probe_config.probe_mode,
            "memory_evaluation_mode": mode,
            "variant": self.probe_config.variant.name,
            "source_checkpoint_sha256": (
                self.source_checkpoint_hash_before
            ),
            "training_manifest_identity": self.training_source_identity,
            "validation_manifest_identity": self.validation_source_identity,
            "model_parameter_norm": tensor_norm(self.model.parameters()),
            "m0_norm": (
                float(self._rmt_model().initial_memory.detach().norm().cpu())
                if self.probe_config.variant.memory_enabled
                else None
            ),
            "active_memory_norm": (
                None
                if self.active_memory is None
                else float(self.active_memory.detach().norm().cpu())
            ),
        }
        return record

    def _evaluate_point(self) -> None:
        step = self.state.global_logical_batches
        if step in self.completed_evaluation_steps:
            raise RuntimeError(f"Probe evaluation step {step} is duplicated")
        before = self._evaluation_state_digest()
        rng = capture_rng_state()
        records = [
            self._evaluate_one_mode(mode)
            for mode in self._evaluation_modes()
        ]
        restore_rng_state(rng)
        after = self._evaluation_state_digest()
        if before != after:
            raise RuntimeError("Probe evaluation mutated training state")
        distributed: DistributedContext | None = getattr(
            self,
            "distributed",
            None,
        )
        if (
            distributed is not None
            and self.probe_config.distributed is not None
            and self.probe_config.distributed.debug_assert_synced
        ):
            assert_digest_equal(
                {
                    "model": self.model.state_dict(),
                    "optimizer": (
                        None
                        if self.optimizer is None
                        else self.optimizer.state_dict()
                    ),
                    "scheduler": (
                        None
                        if self.scheduler is None
                        else self.scheduler.state_dict()
                    ),
                    "scaler": self.scaler.state_dict(),
                    "gradients": self._gradient_state(),
                    "active": self.active_memory,
                    "trainer": self.state.to_dict(),
                },
                distributed,
                label=f"probe state after evaluation step {step}",
            )
        self.curve_records.extend(records)
        self.completed_evaluation_steps.append(step)
        for record in records:
            self._log("probe_evaluation", **record)

    def _should_evaluate(self, step: int) -> bool:
        return (
            step == 0
            or step == self.probe_config.planned_logical_batches
            or step in self.probe_config.early_milestones
            or step
            % self.probe_config.evaluation_interval_logical_steps
            == 0
        )

    def _pairing_identity(self) -> dict[str, Any]:
        return {
            "training_manifest_identity": self.training_source_identity,
            "validation_manifest_identity": self.validation_source_identity,
            "source_boundary_class": "stable_task_boundary",
            "probe_seed": self.probe_config.runtime.seed,
            "evaluation_schedule": {
                "interval_logical_steps": (
                    self.probe_config.evaluation_interval_logical_steps
                ),
                "early_milestones": self.probe_config.early_milestones,
                "validation_sequences": (
                    self.probe_config.validation_sequences
                ),
                "sequence_length": self.probe_config.sequence_length,
            },
            "token_budget": {
                "planned_logical_batches": (
                    self.probe_config.planned_logical_batches
                ),
                "configured_input_token_budget": (
                    self.probe_config.input_token_budget
                ),
                "requested_input_token_budget": (
                    self.probe_config.requested_input_token_budget
                ),
                "token_budget_policy": (
                    self.probe_config.token_budget_policy
                ),
                "global_sequences_per_logical_batch": (
                    self.probe_config.optimization
                    .global_sequences_per_logical_batch
                ),
                "train_sequence_prefix_count": (
                    self.probe_config.train_sequence_prefix_count
                ),
            },
        }

    def _results(self, final_checkpoint: str, checksum: str) -> dict[str, Any]:
        auc = compute_probe_auc_report(
            self.curve_records,
            early_milestones=self.probe_config.early_milestones,
            policy=asdict(self.probe_config.auc),
        )
        return {
            "probe_results_schema_version": 1,
            "status": "complete",
            "run_name": self.probe_config.run_name,
            "variant": self.probe_config.variant.name,
            "probe_mode": self.probe_config.probe_mode,
            "configured_fast_lr": self.probe_config.variant.fast_lr,
            "effective_probe_fast_lr": self.probe_config.effective_fast_lr,
            "source_checkpoint": self.source_checkpoint_identity,
            "source_checkpoint_sha256_before": (
                self.source_checkpoint_hash_before
            ),
            "source_checkpoint_sha256_after": self._assert_source_unchanged(),
            "initialization_policy": PROBE_INITIALIZATION_POLICY,
            "probe_config_sha256": self.config_sha256,
            "training_source_identity": self.training_source_identity,
            "validation_source_identity": self.validation_source_identity,
            "pairing_identity": self._pairing_identity(),
            "curve_records": self.curve_records,
            "auc_report": auc,
            "final_checkpoint_path": final_checkpoint,
            "final_checkpoint_sha256": checksum,
            "probe_state": self.state.to_dict(),
        }

    def _run_probe_impl(
        self,
        *,
        resume_checkpoint: str | Path | None = None,
        stop_after_global_logical_batches: int | None = None,
        stop_after_task_boundaries: int | None = None,
    ) -> TrainingResult:
        del stop_after_task_boundaries
        if (
            stop_after_global_logical_batches is not None
            and stop_after_global_logical_batches <= 0
        ):
            raise ValueError("Stop-after count must be positive")
        self._prepare_output(resume=resume_checkpoint is not None)
        self._create_probe_optimizer_scheduler()
        assert self.optimizer is not None
        if resume_checkpoint is None:
            self.optimizer.zero_grad(set_to_none=True)
            self._reset_task_memory()
            self._log(
                "probe_start",
                initialization_policy=PROBE_INITIALIZATION_POLICY,
                source_checkpoint=self.source_checkpoint_identity,
                optimizer_state_imported=False,
                scheduler_state_imported=False,
                continual_rng_state_imported=False,
                continual_active_root_imported=False,
                planned_logical_batches=(
                    self.probe_config.planned_logical_batches
                ),
                planned_slow_steps=self.probe_config.planned_slow_steps,
                warmup_slow_steps=self.scheduler.warmup_steps,
                backward_reference_targets=(
                    self._backward_reference_targets
                ),
                fp16_loss_conditioning_multiplier=(
                    self.probe_config.optimization
                    .fp16_loss_conditioning_multiplier
                ),
                training_source_identity=self.training_source_identity,
                validation_source_identity=self.validation_source_identity,
            )
            self._evaluate_point()
            start = TokenPosition(0, 0)
        else:
            payload = self._load_resume(resume_checkpoint)
            self._log(
                "probe_resume",
                checkpoint_path=str(Path(resume_checkpoint).resolve()),
                completed_evaluation_steps=self.completed_evaluation_steps,
            )
            start = _position_from_dict(self.state.source_position)
            if self.state.phase == "probe_complete":
                raise ValueError("Completed probe checkpoints cannot be resumed")
            del payload

        iterator = self._training_iterator(start=start)
        last_checkpoint = ""
        last_checksum = ""
        planned = self.probe_config.planned_logical_batches
        while self.state.global_logical_batches < planned:
            try:
                batch = next(iterator)
            except StopIteration as exc:
                raise RuntimeError(
                    "Probe training source ended before configured budget"
                ) from exc
            self._train_logical_batch(
                batch,
                ignore_index=self.train_ignore_index,
            )
            if self.state.window_logical_batches == (
                self.probe_config.variant.slow_update_period_k
            ):
                self._optimizer_step(tail_flush=False)
            final = self.state.global_logical_batches == planned
            if final and self.state.window_logical_batches:
                self._optimizer_step(tail_flush=True)
            if self._should_evaluate(self.state.global_logical_batches):
                self._evaluate_point()
            every = (
                self.probe_config.optimization
                .checkpoint_every_logical_batches
            )
            if (
                every
                and self.state.global_logical_batches % every == 0
                and not final
            ):
                last_checkpoint, last_checksum = self._save_checkpoint(
                    f"step-{self.state.global_logical_batches:08d}.pt"
                )
            if (
                stop_after_global_logical_batches is not None
                and self.state.global_logical_batches
                >= stop_after_global_logical_batches
                and not final
            ):
                filename = (
                    "interrupted-step-"
                    f"{self.state.global_logical_batches:08d}.pt"
                )
                if (
                    last_checkpoint
                    and Path(last_checkpoint).name
                    == f"step-{self.state.global_logical_batches:08d}.pt"
                ):
                    filename = (
                        "interrupted-copy-step-"
                        f"{self.state.global_logical_batches:08d}.pt"
                    )
                last_checkpoint, last_checksum = self._save_checkpoint(
                    filename
                )
                self._log("probe_end", status="interrupted")
                return TrainingResult(
                    status="interrupted",
                    checkpoint_path=last_checkpoint,
                    checkpoint_sha256=last_checksum,
                    state=self.state.to_dict(),
                    loss_history=list(self.state.loss_history),
                )

        self.state.phase = "probe_complete"
        last_checkpoint, last_checksum = self._save_checkpoint(
            f"probe-complete-step-{planned:08d}.pt"
        )
        results = self._results(last_checkpoint, last_checksum)
        distributed: DistributedContext | None = getattr(
            self,
            "distributed",
            None,
        )
        if distributed is None or distributed.is_primary:
            atomic_write_json(self.results_path, results)
        self._log(
            "probe_end",
            status="complete",
            results_path=str(self.results_path),
            auc_report=results["auc_report"],
        )
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
            return self._run_probe_impl(
                resume_checkpoint=resume_checkpoint,
                stop_after_global_logical_batches=(
                    stop_after_global_logical_batches
                ),
                stop_after_task_boundaries=stop_after_task_boundaries,
            )
        except BaseException as exc:
            if self.logger is not None and not self.logger.closed:
                self._log(
                    "probe_error",
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
            raise
        finally:
            if self.logger is not None:
                self.logger.close()


class DistributedProbeTrainer(DistributedContinualTrainer, ProbeTrainer):
    """DDP probe with Phase 6 global-batch and rank-0 write semantics."""

    def __init__(
        self,
        config: ProbeExperimentConfig,
        context: DistributedContext,
    ):
        super().__init__(config, context)

    def _checkpoint_payload(self) -> dict[str, Any]:
        payload = ProbeTrainer._checkpoint_payload(self)
        payload["distributed_state"] = (
            self._distributed_checkpoint_metadata()
        )
        return payload

    def _save_checkpoint(self, filename: str) -> tuple[str, str]:
        self._assert_source_unchanged()
        self._log(
            "probe_checkpoint_barrier",
            checkpoint_filename=filename,
        )
        dist.barrier()
        self._checkpoint_state_digests = (
            self._verify_shared_state_for_checkpoint()
        )
        rng_states: list[dict[str, Any] | None] = [
            None for _ in range(self.distributed.world_size)
        ]
        dist.all_gather_object(
            rng_states,
            {"rank": self.distributed.rank, "state": capture_rng_state()},
        )
        topology: list[dict[str, Any] | None] = [
            None for _ in range(self.distributed.world_size)
        ]
        dist.all_gather_object(
            topology,
            self.distributed.topology_record(),
        )
        self._checkpoint_rank_rng_states = [
            value for value in rng_states if value is not None
        ]
        self._checkpoint_rank_topology = [
            value for value in topology if value is not None
        ]
        outcome: list[dict[str, str] | None] = [None]
        path = self.checkpoint_dir / filename
        if self.distributed.is_primary:
            try:
                checksum = atomic_save_probe_checkpoint(
                    path,
                    self._checkpoint_payload(),
                )
                outcome[0] = {
                    "status": "ok",
                    "path": str(path),
                    "sha256": checksum,
                }
            except BaseException as exc:
                outcome[0] = {
                    "status": "error",
                    "message": f"{type(exc).__name__}: {exc}",
                }
        dist.broadcast_object_list(outcome, src=0)
        assert outcome[0] is not None
        if outcome[0]["status"] != "ok":
            raise RuntimeError(
                "Rank-0 probe checkpoint write failed: "
                + outcome[0]["message"]
            )
        dist.barrier()
        self._log(
            "probe_checkpoint",
            checkpoint_path=outcome[0]["path"],
            checkpoint_sha256=outcome[0]["sha256"],
            writer_rank=0,
        )
        return outcome[0]["path"], outcome[0]["sha256"]

    def _load_resume(self, checkpoint_path: str | Path) -> dict[str, Any]:
        payload = load_probe_checkpoint(checkpoint_path)
        identities: list[str | None] = [
            None for _ in range(self.distributed.world_size)
        ]
        dist.all_gather_object(
            identities,
            sha256_file(checkpoint_path),
        )
        if len(set(identities)) != 1:
            raise ValueError("Ranks disagree about probe checkpoint identity")
        if payload["config_sha256"] != self.config_sha256:
            raise ValueError("Probe resume configuration differs")
        if payload["source_checkpoint"] != self.source_checkpoint_identity:
            raise ValueError("Probe resume source-checkpoint identity differs")
        if payload["training_source_identity"] != (
            self.training_source_identity
        ):
            raise ValueError("Probe training manifest identity differs")
        if payload["validation_source_identity"] != (
            self.validation_source_identity
        ):
            raise ValueError("Probe validation manifest identity differs")
        metadata = payload["distributed_state"]
        if (
            not metadata["enabled"]
            or metadata["world_size"] != self.distributed.world_size
            or metadata["backend"] != self.distributed.backend
            or metadata["partition_rule"]
            != self.distributed.partition_rule
        ):
            raise ValueError(
                "Probe resume requires the same DDP world/backend/partition"
            )
        by_rank = {
            item["rank"]: item["state"]
            for item in metadata["rank_rng_states"]
        }
        if set(by_rank) != set(range(self.distributed.world_size)):
            raise ValueError("Probe checkpoint rank RNG state is incomplete")
        self.model.load_state_dict(payload["model_state"])
        self.state = TrainerState.from_dict(payload["probe_state"])
        self._restore_memory_state(payload["memory_state"])
        assert self.optimizer is not None and self.scheduler is not None
        self.optimizer.load_state_dict(payload["optimizer_state"])
        self.scheduler.load_state_dict(payload["scheduler_state"])
        self.scaler.load_state_dict(payload["scaler_state"])
        self._checkpoint_state_digests = metadata["state_digests"]
        self._restore_gradients(payload["gradients"])
        self.curve_records = list(payload["curve_records"])
        self.completed_evaluation_steps = list(
            payload["completed_evaluation_steps"]
        )
        restore_rng_state(by_rank[self.distributed.rank])
        self._assert_source_unchanged()
        return payload
