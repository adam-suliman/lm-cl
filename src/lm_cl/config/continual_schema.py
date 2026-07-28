from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil
from typing import Any

from lm_cl.config.data_schema import DataPipelineConfig
from lm_cl.config.schema import DataConfig, ModelConfig, VariantConfig


CONTINUAL_SCHEMA_VERSION = 1
CONTINUAL_LANGUAGE_ORDER = (
    "en",
    "zh_written",
    "fr",
    "ja",
    "es",
    "de",
    "pt",
    "ru",
)


@dataclass(frozen=True)
class TrainSourceConfig:
    kind: str
    synthetic: DataConfig | None
    packed: DataPipelineConfig | None

    def validate(self, *, allow_pending_packed: bool = False) -> None:
        if self.kind not in {"synthetic", "packed_shards"}:
            raise ValueError(
                "Continual training supports only synthetic and packed_shards"
            )
        if self.kind == "synthetic":
            if self.synthetic is None or self.packed is not None:
                raise ValueError(
                    "Synthetic source requires synthetic and prohibits packed"
                )
            self.synthetic.validate()
        else:
            if self.packed is None or self.synthetic is not None:
                raise ValueError(
                    "Packed source requires packed and prohibits synthetic"
                )
            self.packed.validate()
            self.packed.require_access_ready()
            if self.packed.mode != "packed_shards":
                raise ValueError(
                    "Continual training requires data mode packed_shards"
                )
            if self.packed.stage.purpose not in {
                "continual_train",
                "language_validation",
            }:
                raise ValueError(
                    "Packed training/evaluation source has an invalid purpose"
                )
            if not allow_pending_packed:
                self.packed.require_packed_launch_ready()

    @property
    def sequence_length(self) -> int:
        if self.synthetic is not None:
            return self.synthetic.sequence_length
        assert self.packed is not None
        return self.packed.reader.sequence_length


@dataclass(frozen=True)
class ContinualTaskConfig:
    language: str
    task_index: int
    cycle_index: int
    logical_batches: int | None
    input_token_budget: int | None
    train_source: TrainSourceConfig
    validation_source: TrainSourceConfig | None
    validation_logical_batches: int
    train_sequence_prefix_count: int | None = None
    train_sequence_offset_count: int = 0

    def validate(
        self,
        *,
        global_sequences_per_logical_batch: int,
        allow_pending_packed: bool = False,
    ) -> None:
        if self.language not in CONTINUAL_LANGUAGE_ORDER:
            raise ValueError(f"Unknown continual language: {self.language}")
        if self.task_index < 0 or self.cycle_index < 0:
            raise ValueError("Task and cycle indices must be non-negative")
        if (self.logical_batches is None) == (self.input_token_budget is None):
            raise ValueError(
                "Exactly one of logical_batches and input_token_budget is required"
            )
        if self.logical_batches is not None and self.logical_batches <= 0:
            raise ValueError("logical_batches must be positive")
        if self.input_token_budget is not None and self.input_token_budget <= 0:
            raise ValueError("input_token_budget must be positive")
        if self.train_sequence_prefix_count is not None:
            prefix = self.train_sequence_prefix_count
            if prefix <= 0:
                raise ValueError(
                    "train_sequence_prefix_count must be positive"
                )
            if self.input_token_budget is None or self.logical_batches is not None:
                raise ValueError(
                    "train_sequence_prefix_count requires input_token_budget"
                )
            expected_input_tokens = prefix * self.train_source.sequence_length
            if self.input_token_budget != expected_input_tokens:
                raise ValueError(
                    "input_token_budget must equal complete-sequence prefix "
                    "times sequence length"
                )
        if self.train_sequence_offset_count < 0:
            raise ValueError(
                "train_sequence_offset_count must be non-negative"
            )
        if (
            self.train_sequence_offset_count
            and self.train_sequence_prefix_count is None
        ):
            raise ValueError(
                "A non-zero sequence offset requires an explicit sequence count"
            )
        self.train_source.validate(allow_pending_packed=allow_pending_packed)
        if self.validation_logical_batches < 0:
            raise ValueError("validation_logical_batches must be non-negative")
        if self.validation_source is None:
            if self.validation_logical_batches != 0:
                raise ValueError(
                    "validation_logical_batches must be zero without a source"
                )
        else:
            self.validation_source.validate(
                allow_pending_packed=allow_pending_packed
            )
            if self.validation_logical_batches <= 0:
                raise ValueError(
                    "A validation source requires a positive batch budget"
                )
            if self.validation_source.sequence_length != (
                self.train_source.sequence_length
            ):
                raise ValueError(
                    "Training and validation sequence lengths must match"
                )
            if (
                self.validation_source.packed is not None
                and self.validation_source.packed.stage.language
                != self.language
            ):
                raise ValueError(
                    "Validation source language differs from its continual task"
                )
        if (
            self.train_source.synthetic is not None
            and (
                (
                    self.train_sequence_offset_count
                    + self.train_sequence_prefix_count
                )
                if self.train_sequence_prefix_count is not None
                else (
                    self.planned_logical_batches(
                        global_sequences_per_logical_batch
                    )
                    * global_sequences_per_logical_batch
                )
            )
            > self.train_source.synthetic.num_sequences
        ):
            raise ValueError(
                "Synthetic source has fewer sequences than the task budget"
            )
        if self.train_source.packed is not None:
            stage = self.train_source.packed.stage
            if self.train_sequence_offset_count == 0:
                if (
                    stage.language != self.language
                    or stage.task_index != self.task_index
                    or stage.cycle_index != self.cycle_index
                ):
                    raise ValueError(
                        "Packed stage language/task/cycle differs from task"
                    )
            elif (
                stage.language != self.language
                or stage.task_index
                != CONTINUAL_LANGUAGE_ORDER.index(self.language)
                or stage.cycle_index != 0
            ):
                raise ValueError(
                    "Windowed reuse requires the matching cycle-0 language stage"
                )
            identity = self.train_source.packed.packed_manifest_identity
            if (
                identity is not None
                and self.train_sequence_prefix_count is not None
                and identity.expected_complete_sequence_count
                < (
                    self.train_sequence_offset_count
                    + self.train_sequence_prefix_count
                )
            ):
                raise ValueError(
                    "Packed stage has fewer complete sequences than the task window"
                )

    def planned_logical_batches(
        self, global_sequences_per_logical_batch: int
    ) -> int:
        if self.logical_batches is not None:
            return self.logical_batches
        if self.train_sequence_prefix_count is not None:
            return ceil(
                self.train_sequence_prefix_count
                / global_sequences_per_logical_batch
            )
        assert self.input_token_budget is not None
        input_tokens_per_batch = (
            global_sequences_per_logical_batch
            * self.train_source.sequence_length
        )
        return ceil(self.input_token_budget / input_tokens_per_batch)


@dataclass(frozen=True)
class ContinualOptimizationConfig:
    optimizer: str
    adam_beta1: float
    adam_beta2: float
    adam_epsilon: float
    weight_decay: float
    warmup_fraction: float
    global_sequences_per_logical_batch: int
    physical_microbatch_sequences: int
    slow_gradient_clip_norm: float | None
    precision: str
    fp16_loss_conditioning_multiplier: int
    checkpoint_every_logical_batches: int

    def validate(self) -> None:
        if self.optimizer != "adamw":
            raise ValueError("optimizer must be adamw")
        if (self.adam_beta1, self.adam_beta2) != (0.9, 0.95):
            raise ValueError("Continual AdamW betas must be (0.9, 0.95)")
        if self.adam_epsilon != 1e-8:
            raise ValueError("Continual AdamW epsilon must be 1e-8")
        if self.weight_decay != 0.1:
            raise ValueError("Continual AdamW weight decay must be 0.1")
        if self.warmup_fraction != 0.05:
            raise ValueError("Continual warmup_fraction must be 0.05")
        if self.global_sequences_per_logical_batch <= 0:
            raise ValueError(
                "global_sequences_per_logical_batch must be positive"
            )
        if self.physical_microbatch_sequences <= 0:
            raise ValueError("physical_microbatch_sequences must be positive")
        if self.slow_gradient_clip_norm is not None:
            raise ValueError("Primary continual runs prohibit slow-gradient clipping")
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("precision must be fp32, fp16, or bf16")
        if self.fp16_loss_conditioning_multiplier != 2:
            raise ValueError(
                "fp16_loss_conditioning_multiplier must be 2"
            )
        if self.checkpoint_every_logical_batches < 0:
            raise ValueError(
                "checkpoint_every_logical_batches must be non-negative"
            )


@dataclass(frozen=True)
class ContinualRuntimeConfig:
    seed: int
    device: str
    deterministic_algorithms: bool
    output_dir: str
    metrics_jsonl: str
    tensorboard_dir: str | None = None
    tensorboard_flush_seconds: int = 30
    tensorboard_log_every_batches: int = 1
    diagnostic_norm_interval_steps: int = 1

    def validate(self) -> None:
        if self.device not in {"cpu", "cuda", "auto"}:
            raise ValueError("runtime.device must be cpu, cuda, or auto")
        if not self.output_dir:
            raise ValueError("runtime.output_dir must not be empty")
        if not self.metrics_jsonl:
            raise ValueError("runtime.metrics_jsonl must not be empty")
        if self.tensorboard_dir is not None and not self.tensorboard_dir:
            raise ValueError("runtime.tensorboard_dir must be null or non-empty")
        if self.tensorboard_flush_seconds <= 0:
            raise ValueError("runtime.tensorboard_flush_seconds must be positive")
        if self.tensorboard_log_every_batches <= 0:
            raise ValueError(
                "runtime.tensorboard_log_every_batches must be positive"
            )
        if self.diagnostic_norm_interval_steps <= 0:
            raise ValueError(
                "runtime.diagnostic_norm_interval_steps must be positive"
            )


@dataclass(frozen=True)
class DistributedConfig:
    enabled: bool
    backend: str
    timeout_seconds: int
    partition_rule: str
    reduction_policy: str
    active_memory_sync_policy: str
    ddp_broadcast_buffers: bool
    ddp_find_unused_parameters: bool
    debug_assert_synced: bool
    per_rank_diagnostic_logs: bool

    def validate(self, *, runtime_device: str) -> None:
        if not self.enabled:
            raise ValueError(
                "An explicit distributed section must set enabled=true"
            )
        if self.backend not in {"auto", "gloo", "nccl"}:
            raise ValueError("distributed.backend must be auto, gloo, or nccl")
        if not 10 <= self.timeout_seconds <= 3600:
            raise ValueError(
                "distributed.timeout_seconds must be between 10 and 3600"
            )
        if self.partition_rule != "contiguous_floor_v1":
            raise ValueError("Unknown distributed partition rule")
        if (
            self.reduction_policy
            != "ddp_average_world_scaled_global_sum_v1"
        ):
            raise ValueError("Unknown distributed reduction policy")
        if (
            self.active_memory_sync_policy
            != "sum_unscale_normalize_clip_rank0_broadcast_v1"
        ):
            raise ValueError("Unknown active-memory synchronization policy")
        if self.ddp_broadcast_buffers:
            raise ValueError("Phase 6 DDP requires broadcast_buffers=false")
        if self.ddp_find_unused_parameters:
            raise ValueError(
                "Phase 6 DDP requires find_unused_parameters=false"
            )
        if self.backend == "nccl" and runtime_device != "cuda":
            raise ValueError("NCCL requires runtime.device=cuda")
        if self.backend == "gloo" and runtime_device == "cuda":
            raise ValueError("CUDA distributed runs must use NCCL or auto")


@dataclass(frozen=True)
class ContinualExperimentConfig:
    schema_version: int
    run_name: str
    model: ModelConfig
    variant: VariantConfig
    optimization: ContinualOptimizationConfig
    runtime: ContinualRuntimeConfig
    tasks: list[ContinualTaskConfig]
    distributed: DistributedConfig | None = None

    def validate(self, *, allow_pending_packed: bool = False) -> None:
        if self.schema_version != CONTINUAL_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {CONTINUAL_SCHEMA_VERSION}"
            )
        if not self.run_name:
            raise ValueError("run_name must not be empty")
        self.model.validate()
        self.variant.validate()
        self.optimization.validate()
        self.runtime.validate()
        if self.distributed is not None:
            self.distributed.validate(runtime_device=self.runtime.device)
        if not self.tasks:
            raise ValueError("At least one continual task is required")
        for index, task in enumerate(self.tasks):
            expected_language = CONTINUAL_LANGUAGE_ORDER[
                index % len(CONTINUAL_LANGUAGE_ORDER)
            ]
            expected_cycle = index // len(CONTINUAL_LANGUAGE_ORDER)
            if (
                task.task_index != index
                or task.cycle_index != expected_cycle
                or task.language != expected_language
            ):
                raise ValueError(
                    "Tasks must follow exact en→zh→fr→ja→es→de→pt→ru "
                    "order with contiguous indices"
                )
            task.validate(
                global_sequences_per_logical_batch=(
                    self.optimization.global_sequences_per_logical_batch
                ),
                allow_pending_packed=allow_pending_packed,
            )
            if task.train_source.sequence_length > (
                self.model.max_position_embeddings
            ):
                raise ValueError(
                    "Task sequence length exceeds model maximum positions"
                )
            if self.variant.memory_enabled:
                assert self.variant.segment_length is not None
                if (
                    task.train_source.sequence_length
                    % self.variant.segment_length
                    != 0
                ):
                    raise ValueError(
                        "Memory task sequence length must be divisible by "
                        "segment_length"
                    )
                if (
                    task.train_source.sequence_length
                    // self.variant.segment_length
                    != 2
                ):
                    raise ValueError(
                        "Primary memory variants require two segments per sequence"
                    )
            if (
                task.train_source.synthetic is not None
                and task.train_source.synthetic.vocab_size
                > self.model.vocab_size
            ):
                raise ValueError(
                    "Synthetic source vocabulary exceeds model vocabulary"
                )
            if task.train_source.packed is not None:
                if (
                    task.train_source.packed.reader.global_sequences_per_batch
                    != self.optimization.global_sequences_per_logical_batch
                ):
                    raise ValueError(
                        "Packed reader logical batch differs from training"
                    )

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        if values["runtime"].get("diagnostic_norm_interval_steps") == 1:
            values["runtime"].pop("diagnostic_norm_interval_steps", None)
        # Preserve schema-v1 checkpoint identities for legacy tasks.  The
        # offset was added for windowed packed views; zero has exactly the old
        # semantics and therefore must not perturb historical config hashes.
        for task in values["tasks"]:
            if task.get("train_sequence_offset_count", 0) == 0:
                task.pop("train_sequence_offset_count", None)
        return values
