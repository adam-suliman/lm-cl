from __future__ import annotations

from dataclasses import MISSING, asdict, dataclass, fields
from math import ceil
from types import UnionType
from typing import Any, TypeVar, Union, get_args, get_origin, get_type_hints


T = TypeVar("T")


def _matches_type(value: Any, annotation: Any) -> bool:
    if annotation is Any:
        return True
    origin = get_origin(annotation)
    if origin in {Union, UnionType}:
        return any(_matches_type(value, option) for option in get_args(annotation))
    if origin is dict:
        if not isinstance(value, dict):
            return False
        key_type, value_type = get_args(annotation)
        return all(
            _matches_type(key, key_type) and _matches_type(item, value_type)
            for key, item in value.items()
        )
    if origin is list:
        (item_type,) = get_args(annotation)
        return isinstance(value, list) and all(
            _matches_type(item, item_type) for item in value
        )
    if annotation is int:
        return isinstance(value, int) and not isinstance(value, bool)
    if annotation is float:
        return isinstance(value, float)
    if annotation is bool:
        return isinstance(value, bool)
    if annotation is type(None):
        return value is None
    return isinstance(value, annotation)


def strict_dataclass(cls: type[T], values: dict[str, Any], context: str) -> T:
    if not isinstance(values, dict):
        raise ValueError(f"{context} must be a mapping")
    allowed = {field.name for field in fields(cls)}
    unknown = sorted(set(values) - allowed)
    missing = sorted(
        field.name
        for field in fields(cls)
        if field.default is MISSING and field.default_factory is MISSING
        and field.name not in values
    )
    if unknown:
        raise ValueError(f"Unknown {context} fields: {unknown}")
    if missing:
        raise ValueError(f"Missing {context} fields: {missing}")
    annotations = get_type_hints(cls)
    wrong_types = sorted(
        name
        for name, value in values.items()
        if not _matches_type(value, annotations[name])
    )
    if wrong_types:
        raise ValueError(
            f"Incorrectly typed {context} fields: {wrong_types}"
        )
    try:
        return cls(**values)
    except TypeError as exc:
        raise ValueError(f"Invalid {context}: {exc}") from exc


@dataclass(frozen=True)
class ModelConfig:
    name: str
    layers: int
    hidden_size: int
    attention_heads: int
    head_dim: int
    mlp_hidden_size: int
    vocab_size: int
    max_position_embeddings: int
    expected_non_embedding_parameters: int
    expected_total_parameters: int
    learning_rate: float
    dropout: float
    initializer_std: float
    layer_norm_epsilon: float
    use_bias: bool
    activation: str
    gelu_approximation: str
    position_embedding_type: str
    tie_word_embeddings: bool

    def validate(self) -> None:
        positive = {
            "layers": self.layers,
            "hidden_size": self.hidden_size,
            "attention_heads": self.attention_heads,
            "head_dim": self.head_dim,
            "mlp_hidden_size": self.mlp_hidden_size,
            "vocab_size": self.vocab_size,
            "max_position_embeddings": self.max_position_embeddings,
            "expected_non_embedding_parameters": self.expected_non_embedding_parameters,
            "expected_total_parameters": self.expected_total_parameters,
            "learning_rate": self.learning_rate,
            "initializer_std": self.initializer_std,
            "layer_norm_epsilon": self.layer_norm_epsilon,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"model.{name} must be positive")
        if self.hidden_size != self.attention_heads * self.head_dim:
            raise ValueError("hidden_size must equal attention_heads * head_dim")
        if self.mlp_hidden_size != 4 * self.hidden_size:
            raise ValueError("mlp_hidden_size must equal 4 * hidden_size")
        if self.dropout != 0.0:
            raise ValueError("Faithful Zyphra model dropout must be 0.0")
        if not self.use_bias:
            raise ValueError("Faithful Zyphra projections require biases")
        if self.activation != "gelu":
            raise ValueError("Faithful Zyphra activation must be gelu")
        if self.gelu_approximation != "none":
            raise ValueError("Faithful Zyphra GeLU approximation must be none")
        if self.position_embedding_type != "learned_absolute":
            raise ValueError("Position embedding type must be learned_absolute")
        if not self.tie_word_embeddings:
            raise ValueError("Token and output weights must be tied")

        expected_non_embedding = (
            self.layers * (12 * self.hidden_size**2 + 13 * self.hidden_size)
            + 2 * self.hidden_size
        )
        embedding = (
            self.vocab_size + self.max_position_embeddings
        ) * self.hidden_size
        if self.expected_non_embedding_parameters != expected_non_embedding:
            raise ValueError(
                "Configured non-embedding count does not match architecture: "
                f"{self.expected_non_embedding_parameters} != {expected_non_embedding}"
            )
        if self.expected_total_parameters != expected_non_embedding + embedding:
            raise ValueError(
                "Configured total count does not match architecture: "
                f"{self.expected_total_parameters} != {expected_non_embedding + embedding}"
            )


@dataclass(frozen=True)
class VariantConfig:
    name: str
    memory_enabled: bool
    persistent_fast_memory: bool
    fast_lr: float
    memory_tokens: int
    slow_update_period_k: int
    fast_memory_grad_clip_norm: float | None
    segment_length: int | None = None
    reset_policy: str | None = None
    memory_evaluation_policy: str | None = None

    def validate(self) -> None:
        valid = {
            "backbone_clean",
            "backbone_matched_k",
            "base_rmt",
            "fastmem_rmt_zero",
            "fastmem_rmt",
        }
        if self.name not in valid:
            raise ValueError(f"Unknown variant: {self.name}")
        if self.slow_update_period_k <= 0:
            raise ValueError("slow_update_period_k must be positive")
        if self.fast_lr < 0:
            raise ValueError("fast_lr must be non-negative")
        if self.memory_tokens < 0:
            raise ValueError("memory_tokens must be non-negative")
        if not self.memory_enabled and (
            self.persistent_fast_memory
            or self.fast_lr != 0
            or self.memory_tokens != 0
            or self.fast_memory_grad_clip_norm is not None
            or self.segment_length is not None
            or self.reset_policy is not None
            or self.memory_evaluation_policy is not None
        ):
            raise ValueError("Memory-disabled variants cannot configure memory state")
        if self.name == "backbone_clean" and self.slow_update_period_k != 1:
            raise ValueError("backbone_clean requires K=1")
        if self.name != "backbone_clean" and self.slow_update_period_k != 2:
            raise ValueError("Primary non-clean variants require K=2")
        if self.name in {"backbone_clean", "backbone_matched_k"} and (
            self.memory_enabled
        ):
            raise ValueError("Backbone variants cannot enable memory")
        if self.memory_enabled:
            if self.memory_tokens != 8:
                raise ValueError("Primary memory variants require 8 memory tokens")
            if self.segment_length is None or self.segment_length <= 0:
                raise ValueError("Memory variants require a positive segment_length")
            if self.reset_policy != "task_boundary_from_m0_stopgrad":
                raise ValueError(
                    "Memory variants require task-boundary reset from stopgrad(M0)"
                )
            if self.memory_evaluation_policy != "reset_and_carried":
                raise ValueError(
                    "Memory variants require reset_and_carried evaluation"
                )
        if self.name == "base_rmt":
            if (
                not self.memory_enabled
                or self.persistent_fast_memory
                or self.fast_lr != 0
                or self.fast_memory_grad_clip_norm is not None
            ):
                raise ValueError("base_rmt memory/update semantics are fixed")
        if self.name == "fastmem_rmt_zero":
            if (
                not self.memory_enabled
                or not self.persistent_fast_memory
                or self.fast_lr != 0
                or self.fast_memory_grad_clip_norm != 1.0
            ):
                raise ValueError("fastmem_rmt_zero semantics are fixed")
        if self.name == "fastmem_rmt":
            if (
                not self.memory_enabled
                or not self.persistent_fast_memory
                or self.fast_lr <= 0
                or self.fast_memory_grad_clip_norm != 1.0
            ):
                raise ValueError(
                    "fastmem_rmt requires persistent memory, a positive "
                    "explicit fast LR, and active-only clipping at 1.0"
                )


@dataclass(frozen=True)
class DataConfig:
    backend: str
    vocab_size: int
    sequence_length: int
    num_sequences: int
    seed: int
    ignore_index: int
    mask_probability: float

    def validate(self) -> None:
        if self.backend != "synthetic":
            raise ValueError("DataConfig supports only the synthetic backend")
        if self.vocab_size <= 1:
            raise ValueError("data.vocab_size must be greater than one")
        if self.sequence_length <= 1:
            raise ValueError("data.sequence_length must be greater than one")
        if self.num_sequences <= 0:
            raise ValueError("data.num_sequences must be positive")
        if not 0.0 <= self.mask_probability < 1.0:
            raise ValueError("data.mask_probability must be in [0, 1)")


@dataclass(frozen=True)
class TrainingConfig:
    optimizer: str
    adam_beta1: float
    adam_beta2: float
    adam_epsilon: float
    weight_decay: float
    warmup_fraction: float
    global_sequences_per_logical_batch: int
    max_logical_steps_per_task: int
    nominal_task_tokens: int
    slow_gradient_clip_norm: float | None
    precision: str

    def validate(self) -> None:
        if self.optimizer != "adamw":
            raise ValueError("optimizer must be adamw")
        if not 0 <= self.adam_beta1 < 1 or not 0 <= self.adam_beta2 < 1:
            raise ValueError("Adam betas must be in [0, 1)")
        if self.adam_epsilon <= 0 or self.weight_decay < 0:
            raise ValueError("Adam epsilon must be positive and weight decay non-negative")
        if not 0 <= self.warmup_fraction <= 1:
            raise ValueError("warmup_fraction must be in [0, 1]")
        if self.global_sequences_per_logical_batch <= 0:
            raise ValueError("global_sequences_per_logical_batch must be positive")
        if self.max_logical_steps_per_task <= 0 or self.nominal_task_tokens <= 0:
            raise ValueError("Task step and token budgets must be positive")
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("precision must be fp32, fp16, or bf16")

    def planned_slow_steps(self, k: int) -> int:
        return ceil(self.max_logical_steps_per_task / k)

    def warmup_slow_steps(self, k: int) -> int:
        return ceil(self.warmup_fraction * self.planned_slow_steps(k))


@dataclass(frozen=True)
class RuntimeConfig:
    seed: int
    device: str
    deterministic_algorithms: bool
    metrics_jsonl: str

    def validate(self) -> None:
        if self.device not in {"cpu", "cuda", "auto"}:
            raise ValueError("runtime.device must be cpu, cuda, or auto")
        if not self.metrics_jsonl:
            raise ValueError("runtime.metrics_jsonl must not be empty")


@dataclass(frozen=True)
class ExperimentConfig:
    schema_version: int
    run_name: str
    model: ModelConfig
    variant: VariantConfig
    data: DataConfig
    training: TrainingConfig
    runtime: RuntimeConfig

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError("schema_version must be 1")
        if not self.run_name:
            raise ValueError("run_name must not be empty")
        self.model.validate()
        self.variant.validate()
        self.data.validate()
        self.training.validate()
        self.runtime.validate()
        if self.data.vocab_size > self.model.vocab_size:
            raise ValueError("Synthetic vocabulary cannot exceed model vocabulary")
        if self.data.sequence_length > self.model.max_position_embeddings:
            raise ValueError("Data sequence length exceeds model maximum positions")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
