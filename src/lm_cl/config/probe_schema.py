from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil
from pathlib import Path
from typing import Any

from lm_cl.config.continual_schema import (
    ContinualOptimizationConfig,
    ContinualRuntimeConfig,
    DistributedConfig,
    TrainSourceConfig,
)
from lm_cl.config.schema import ModelConfig, VariantConfig


PROBE_SCHEMA_VERSION = 1
PHASE7_CULTURAX_REPO = "uonlp/CulturaX"
PHASE7_CULTURAX_REVISION = (
    "6a8734bc69fefcbb7735f4f9250f43e4cd7a442e"
)
PHASE7_TOKENIZER_REPO = "Qwen/Qwen3-0.6B-Base"
PHASE7_TOKENIZER_REVISION = (
    "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"
)


@dataclass(frozen=True)
class ProbeAUCPolicy:
    primary: str
    interpolation: str
    x_axis: str
    normalization: str
    smoothing: str
    percentage_reference: str

    def validate(self) -> None:
        expected = {
            "primary": "normalized_trapezoidal_ce_v1",
            "interpolation": "trapezoidal",
            "x_axis": "cumulative_input_tokens",
            "normalization": "observed_span_to_unit_interval",
            "smoothing": "none",
            "percentage_reference": "first_probe_auc",
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(f"auc.{name} must be {value}")


@dataclass(frozen=True)
class ProbeExperimentConfig:
    schema_version: int
    run_name: str
    run_kind: str
    source_checkpoint: str
    model: ModelConfig
    variant: VariantConfig
    probe_mode: str
    fast_lr_override: float | None
    optimization: ContinualOptimizationConfig
    runtime: ContinualRuntimeConfig
    train_source: TrainSourceConfig
    validation_source: TrainSourceConfig
    train_logical_batches: int | None
    input_token_budget: int | None
    validation_sequences: int
    evaluation_interval_logical_steps: int
    early_milestones: list[int]
    auc: ProbeAUCPolicy
    train_sequence_prefix_count: int | None = None
    requested_input_token_budget: int | None = None
    token_budget_policy: str | None = None
    source_checkpoint_status: str = "legacy_runtime"
    source_checkpoint_sha256: str | None = None
    distributed: DistributedConfig | None = None

    @property
    def effective_fast_lr(self) -> float:
        return (
            self.variant.fast_lr
            if self.fast_lr_override is None
            else self.fast_lr_override
        )

    @property
    def sequence_length(self) -> int:
        return self.train_source.sequence_length

    @property
    def planned_logical_batches(self) -> int:
        if self.train_logical_batches is not None:
            return self.train_logical_batches
        assert self.input_token_budget is not None
        per_batch = (
            self.optimization.global_sequences_per_logical_batch
            * self.sequence_length
        )
        return ceil(self.input_token_budget / per_batch)

    @property
    def planned_slow_steps(self) -> int:
        return ceil(
            self.planned_logical_batches
            / self.variant.slow_update_period_k
        )

    def _validate_source(
        self,
        source: TrainSourceConfig,
        *,
        role: str,
        allow_pending_packed: bool,
    ) -> None:
        if source.kind not in {"synthetic", "packed_shards"}:
            raise ValueError(f"Probe {role} source kind is invalid")
        if source.kind == "synthetic":
            if source.synthetic is None or source.packed is not None:
                raise ValueError(
                    f"Synthetic probe {role} requires only synthetic data"
                )
            source.synthetic.validate()
            if source.synthetic.vocab_size > self.model.vocab_size:
                raise ValueError(
                    f"Probe {role} synthetic vocabulary exceeds the model"
                )
            return
        if source.packed is None or source.synthetic is not None:
            raise ValueError(f"Packed probe {role} requires only packed data")
        source.packed.validate()
        source.packed.require_access_ready()
        if not allow_pending_packed:
            source.packed.require_packed_launch_ready()
        if source.packed.mode != "packed_shards":
            raise ValueError("Probe execution requires completed packed shards")
        expected_purpose = (
            "vietnamese_train"
            if role == "training"
            else "vietnamese_validation"
        )
        if (
            source.packed.stage.language != "vi"
            or source.packed.stage.purpose != expected_purpose
        ):
            raise ValueError(
                f"Packed probe {role} must use vi/{expected_purpose}"
            )
        dataset = source.packed.dataset
        tokenizer = source.packed.tokenizer
        expected_dataset = {
            "repo_id": PHASE7_CULTURAX_REPO,
            "revision": PHASE7_CULTURAX_REVISION,
            "split": "train",
            "text_field": "text",
            "id_field": "url",
            "source_id_policy": "sha256_canonical_json",
            "missing_id_policy": "content_sha256",
        }
        dataset_mismatches = [
            name
            for name, value in expected_dataset.items()
            if getattr(dataset, name) != value
        ]
        if dataset.language_configs.get("vi") != "vi":
            dataset_mismatches.append("language_configs.vi")
        if dataset_mismatches:
            raise ValueError(
                "Packed Phase 7 CulturaX identity differs: "
                + ", ".join(dataset_mismatches)
            )
        expected_tokenizer = {
            "repo_id": PHASE7_TOKENIZER_REPO,
            "revision": PHASE7_TOKENIZER_REVISION,
            "base_vocab_size": 151_643,
            "effective_vocab_size": 151_669,
            "maximum_emitted_token_id": 151_668,
            "model_embedding_vocab_size": 151_680,
            "expected_eos_token_id": 151_643,
            "expected_pad_token_id": 151_643,
        }
        tokenizer_mismatches = [
            name
            for name, value in expected_tokenizer.items()
            if getattr(tokenizer, name) != value
        ]
        if tokenizer_mismatches:
            raise ValueError(
                "Packed Phase 7 tokenizer identity differs: "
                + ", ".join(tokenizer_mismatches)
            )

    def validate(self, *, allow_pending_packed: bool = False) -> None:
        if self.schema_version != PROBE_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {PROBE_SCHEMA_VERSION}"
            )
        if not self.run_name:
            raise ValueError("run_name must not be empty")
        if self.run_kind not in {"smoke", "production", "scaled_budget"}:
            raise ValueError(
                "run_kind must be smoke, production, or scaled_budget"
            )
        source = Path(self.source_checkpoint).expanduser()
        if not source.is_absolute():
            raise ValueError("source_checkpoint must resolve to an absolute path")
        if self.source_checkpoint_status not in {
            "legacy_runtime",
            "pending",
            "frozen",
        }:
            raise ValueError(
                "source_checkpoint_status must be legacy_runtime, pending, or frozen"
            )
        if self.source_checkpoint_status in {"legacy_runtime", "pending"}:
            if self.source_checkpoint_sha256 is not None:
                raise ValueError(
                    "An unfrozen source checkpoint must not declare SHA-256"
                )
        elif (
            not isinstance(self.source_checkpoint_sha256, str)
            or len(self.source_checkpoint_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.source_checkpoint_sha256
            )
        ):
            raise ValueError(
                "A frozen source checkpoint requires lowercase SHA-256"
            )
        self.model.validate()
        self.variant.validate()
        self.optimization.validate()
        self.runtime.validate()
        self.auc.validate()
        if self.distributed is not None:
            self.distributed.validate(runtime_device=self.runtime.device)
        if self.probe_mode not in {"system", "slow_only"}:
            raise ValueError("probe_mode must be system or slow_only")
        if self.fast_lr_override is not None and self.fast_lr_override < 0:
            raise ValueError("fast_lr_override must be null or non-negative")
        if self.probe_mode == "system" and self.fast_lr_override is not None:
            raise ValueError("System probes prohibit a fast-LR override")
        if (
            self.probe_mode == "slow_only"
            and self.variant.name == "fastmem_rmt"
            and self.fast_lr_override != 0.0
        ):
            raise ValueError(
                "FastMem slow_only requires explicit fast_lr_override=0.0"
            )
        if (
            self.probe_mode == "slow_only"
            and self.variant.name != "fastmem_rmt"
            and self.fast_lr_override is not None
        ):
            raise ValueError(
                "Only nonzero FastMem needs a slow_only fast-LR override"
            )
        if (self.train_logical_batches is None) == (
            self.input_token_budget is None
        ):
            raise ValueError(
                "Exactly one train_logical_batches/input_token_budget is required"
            )
        if (
            self.train_logical_batches is not None
            and self.train_logical_batches <= 0
        ):
            raise ValueError("train_logical_batches must be positive")
        if self.input_token_budget is not None and self.input_token_budget <= 0:
            raise ValueError("input_token_budget must be positive")
        if self.requested_input_token_budget is not None:
            if self.requested_input_token_budget <= 0:
                raise ValueError(
                    "requested_input_token_budget must be positive"
                )
            if self.token_budget_policy != "floor_complete_sequences_v1":
                raise ValueError(
                    "Resolved probe token_budget_policy must be "
                    "floor_complete_sequences_v1"
                )
        elif self.token_budget_policy is not None:
            raise ValueError(
                "token_budget_policy requires requested_input_token_budget"
            )
        if self.train_sequence_prefix_count is not None:
            prefix = self.train_sequence_prefix_count
            if (
                not isinstance(prefix, int)
                or isinstance(prefix, bool)
                or prefix <= 0
            ):
                raise ValueError(
                    "train_sequence_prefix_count must be a positive integer"
                )
            if self.train_source.kind != "packed_shards":
                raise ValueError(
                    "train_sequence_prefix_count requires packed training data"
                )
            if self.train_logical_batches is not None:
                raise ValueError(
                    "A sequence-prefix probe must use input_token_budget"
                )
            expected_input_tokens = prefix * self.sequence_length
            if self.input_token_budget != expected_input_tokens:
                raise ValueError(
                    "Probe sequence-prefix/input-token budgets are inconsistent"
                )
        self._validate_source(
            self.train_source,
            role="training",
            allow_pending_packed=allow_pending_packed,
        )
        self._validate_source(
            self.validation_source,
            role="validation",
            allow_pending_packed=allow_pending_packed,
        )
        for role, source in (
            ("training", self.train_source),
            ("validation", self.validation_source),
        ):
            if source.packed is not None and (
                source.packed.reader.global_sequences_per_batch
                != self.optimization.global_sequences_per_logical_batch
            ):
                raise ValueError(
                    f"Packed probe {role} global batch differs from optimization"
                )
        if self.train_source.sequence_length != (
            self.validation_source.sequence_length
        ):
            raise ValueError("Probe train/validation sequence lengths differ")
        if self.sequence_length > self.model.max_position_embeddings:
            raise ValueError("Probe sequence length exceeds model positions")
        if self.variant.memory_enabled:
            assert self.variant.segment_length is not None
            if (
                self.sequence_length % self.variant.segment_length != 0
                or self.sequence_length // self.variant.segment_length != 2
            ):
                raise ValueError(
                    "Primary probe memory variants require exactly two segments"
                )
        if self.validation_sequences <= 0:
            raise ValueError("validation_sequences must be positive")
        batch_size = self.optimization.global_sequences_per_logical_batch
        if self.validation_sequences % batch_size:
            raise ValueError(
                "validation_sequences must be divisible by global batch size"
            )
        if self.evaluation_interval_logical_steps <= 0:
            raise ValueError("Evaluation interval must be positive")
        if (
            not isinstance(self.early_milestones, list)
            or any(
                not isinstance(step, int)
                or isinstance(step, bool)
                or step <= 0
                or step > self.planned_logical_batches
                for step in self.early_milestones
            )
            or self.early_milestones != sorted(set(self.early_milestones))
        ):
            raise ValueError(
                "early_milestones must be sorted unique in-budget steps"
            )
        if self.train_source.synthetic is not None:
            needed = self.planned_logical_batches * batch_size
            if self.train_source.synthetic.num_sequences < needed:
                raise ValueError("Synthetic probe training source is too short")
        if self.validation_source.synthetic is not None:
            if (
                self.validation_source.synthetic.num_sequences
                < self.validation_sequences
            ):
                raise ValueError("Synthetic probe validation source is too short")
        if self.train_source.packed is not None:
            required_tokens = (
                self.input_token_budget
                if self.input_token_budget is not None
                else (
                    self.planned_logical_batches
                    * batch_size
                    * self.sequence_length
                )
            )
            if (
                self.train_source.packed.selection.max_output_tokens
                < required_tokens
            ):
                raise ValueError("Packed probe training token cap is too small")
        if self.validation_source.packed is not None:
            required_validation_tokens = (
                self.validation_sequences * self.sequence_length
            )
            if (
                self.validation_source.packed.selection.max_output_tokens
                < required_validation_tokens
            ):
                raise ValueError(
                    "Packed probe validation token cap is too small"
                )
        if self.run_kind in {"production", "scaled_budget"}:
            if (
                self.train_source.packed is None
                or self.validation_source.packed is None
            ):
                raise ValueError(
                    "Production and scaled-budget probes require packed sources"
                )
        if self.run_kind == "production":
            requested_budget = (
                self.input_token_budget
                if self.requested_input_token_budget is None
                else self.requested_input_token_budget
            )
            if requested_budget != 5_000_000_000:
                raise ValueError(
                    "Production probe requested input-token budget must be 5B"
                )
            if self.validation_sequences != 1_280:
                raise ValueError(
                    "Production probe validation requires 1,280 sequences"
                )
            if self.evaluation_interval_logical_steps != 95:
                raise ValueError(
                    "Production probe evaluation interval must be 95"
                )

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        if values["runtime"].get("diagnostic_norm_interval_steps") == 1:
            values["runtime"].pop("diagnostic_norm_interval_steps", None)
        return values
