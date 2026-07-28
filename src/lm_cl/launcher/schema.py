from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


LAUNCHER_SCHEMA_VERSION = 1
PUBLIC_LANGUAGE_ORDER = (
    "en",
    "zh_written",
    "fr",
    "ja",
    "es",
    "de",
    "pt",
    "ru",
)
PUBLIC_MODEL_VARIANTS = {
    "transformer": "backbone_clean",
    "fastmem_rmt": "fastmem_rmt",
}
TOKEN_BUDGET_POLICY = "floor_complete_sequences_v1"
CYCLE_MANIFEST_POLICY = "fresh_disjoint_v1"
WINDOWED_CYCLE_MANIFEST_POLICY = "disjoint_sequence_windows_v1"
CYCLE_MANIFEST_POLICIES = {
    CYCLE_MANIFEST_POLICY,
    WINDOWED_CYCLE_MANIFEST_POLICY,
}
PRIMARY_PROBE_MODE = "system"


@dataclass(frozen=True)
class TokenBudget:
    policy: str
    requested_input_tokens: int
    sequence_length: int
    effective_complete_sequences: int
    effective_input_tokens: int
    effective_valid_targets: int
    discarded_remainder_tokens: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_token_budget(
    requested_input_tokens: int,
    sequence_length: int,
    *,
    policy: str = TOKEN_BUDGET_POLICY,
) -> TokenBudget:
    if policy != TOKEN_BUDGET_POLICY:
        raise ValueError(
            f"token_budget_policy must be {TOKEN_BUDGET_POLICY}"
        )
    if requested_input_tokens <= 0:
        raise ValueError("tokens_per_task must be positive")
    if sequence_length <= 1:
        raise ValueError("sequence_length must be greater than one")
    sequences, remainder = divmod(requested_input_tokens, sequence_length)
    if sequences <= 0:
        raise ValueError(
            "tokens_per_task must contain at least one complete sequence"
        )
    effective_input = sequences * sequence_length
    return TokenBudget(
        policy=policy,
        requested_input_tokens=requested_input_tokens,
        sequence_length=sequence_length,
        effective_complete_sequences=sequences,
        effective_input_tokens=effective_input,
        effective_valid_targets=sequences * (sequence_length - 1),
        discarded_remainder_tokens=remainder,
    )


@dataclass(frozen=True)
class ExperimentSettings:
    name: str
    output_root: str
    repository_root: str
    seed: int | list[int]
    models: list[str]
    model_size: str
    cycles: int
    tokens_per_task: int
    token_budget_policy: str
    sequence_length: int
    languages: list[str]
    probe_schedule: str
    resume: str
    run_kind: str = "production"

    @property
    def seeds(self) -> list[int]:
        return [self.seed] if isinstance(self.seed, int) else list(self.seed)

    def validate(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", self.name):
            raise ValueError("experiment.name must be a safe path component")
        if self.run_kind not in {
            "production",
            "scaled_budget",
            "functional_smoke",
        }:
            raise ValueError(
                "experiment.run_kind must be production, scaled_budget, or "
                "functional_smoke"
            )
        valid_sizes = {"5m", "12m"}
        if self.run_kind == "functional_smoke":
            valid_sizes.add("tiny")
        if self.model_size not in valid_sizes:
            raise ValueError(
                f"experiment.model_size must be one of {sorted(valid_sizes)}"
            )
        if self.cycles <= 0:
            raise ValueError("experiment.cycles must be positive")
        if self.languages != list(PUBLIC_LANGUAGE_ORDER):
            raise ValueError(
                "experiment.languages must be exactly "
                "en,zh_written,fr,ja,es,de,pt,ru"
            )
        if not self.models or len(set(self.models)) != len(self.models):
            raise ValueError("experiment.models must be non-empty and unique")
        invalid_models = sorted(set(self.models) - set(PUBLIC_MODEL_VARIANTS))
        if invalid_models:
            raise ValueError(
                "Only transformer and fastmem_rmt are public model names; "
                f"invalid={invalid_models}"
            )
        seeds = self.seeds
        if not seeds or len(set(seeds)) != len(seeds):
            raise ValueError("experiment.seed must define unique seeds")
        if any(seed < 0 for seed in seeds):
            raise ValueError("experiment seeds must be non-negative")
        if self.probe_schedule != "cycle_end_after_ru":
            raise ValueError(
                "experiment.probe_schedule must be cycle_end_after_ru"
            )
        if self.resume not in {"never", "auto", "required"}:
            raise ValueError("experiment.resume must be never, auto, or required")
        resolve_token_budget(
            self.tokens_per_task,
            self.sequence_length,
            policy=self.token_budget_policy,
        )


@dataclass(frozen=True)
class DataSettings:
    mode: str
    manifest_root: str
    manifest_template: str
    cycle_manifest_policy: str
    tokenizer_manifest: str
    prepare_if_missing: bool
    dataset_cache_root: str
    generated_root: str
    probe_training_manifest: str | None
    probe_validation_manifest: str | None
    language_validation_manifest_template: str | None = None
    max_cache_bytes: int = 21_474_836_480
    max_generated_bytes: int = 429_496_729_600
    max_temporary_bytes: int = 2_147_483_648
    max_input_documents: int = 5_000_000
    max_runtime_seconds: int = 172_800
    shuffle_buffer_documents: int = 8_192
    document_order_seed: int = 31_003
    split_seed: int = 41_001
    validation_permyriad: int = 100
    max_shard_tokens: int = 250_000_000
    window_source_tokens_per_task: int | None = None

    def validate(self, experiment: ExperimentSettings) -> None:
        if self.mode not in {"packed", "synthetic"}:
            raise ValueError("data.mode must be packed or synthetic")
        if self.cycle_manifest_policy not in CYCLE_MANIFEST_POLICIES:
            raise ValueError(
                "data.cycle_manifest_policy must be one of "
                f"{sorted(CYCLE_MANIFEST_POLICIES)}"
            )
        if self.mode == "packed":
            required = {
                "manifest_root": self.manifest_root,
                "manifest_template": self.manifest_template,
                "tokenizer_manifest": self.tokenizer_manifest,
                "dataset_cache_root": self.dataset_cache_root,
                "generated_root": self.generated_root,
            }
            empty = sorted(name for name, value in required.items() if not value)
            if empty:
                raise ValueError(f"Packed data paths are empty: {empty}")
            if self.probe_training_manifest is None:
                raise ValueError("Packed data requires probe_training_manifest")
            if self.probe_validation_manifest is None:
                raise ValueError("Packed data requires probe_validation_manifest")
            if not self.manifest_template.endswith("manifest.json"):
                raise ValueError(
                    "data.manifest_template must end in manifest.json"
                )
            try:
                rendered = self.manifest_template.format(
                    cycle=0,
                    language="en",
                    task_index=0,
                    source_task_index=0,
                    effective_tokens=experiment.tokens_per_task,
                )
            except (KeyError, IndexError, ValueError) as exc:
                raise ValueError(
                    "data.manifest_template has an unsupported placeholder"
                ) from exc
            if Path(rendered).is_absolute() or ".." in Path(rendered).parts:
                raise ValueError(
                    "data.manifest_template must be a relative safe path"
                )
            if self.language_validation_manifest_template is not None:
                if not self.language_validation_manifest_template.endswith(
                    "manifest.json"
                ):
                    raise ValueError(
                        "data.language_validation_manifest_template must end "
                        "in manifest.json"
                    )
                try:
                    validation_rendered = (
                        self.language_validation_manifest_template.format(
                            language="en",
                            effective_tokens=experiment.sequence_length,
                        )
                    )
                except (KeyError, IndexError, ValueError) as exc:
                    raise ValueError(
                        "data.language_validation_manifest_template has an "
                        "unsupported placeholder"
                    ) from exc
                if (
                    Path(validation_rendered).is_absolute()
                    or ".." in Path(validation_rendered).parts
                ):
                    raise ValueError(
                        "data.language_validation_manifest_template must be "
                        "a relative safe path"
                    )
        elif experiment.run_kind != "functional_smoke":
            raise ValueError("Synthetic data is limited to functional_smoke runs")
        if self.cycle_manifest_policy == WINDOWED_CYCLE_MANIFEST_POLICY:
            if (
                self.window_source_tokens_per_task is not None
                and (
                    not isinstance(self.window_source_tokens_per_task, int)
                    or self.window_source_tokens_per_task <= 0
                )
            ):
                raise ValueError(
                    "data.window_source_tokens_per_task must be a positive integer"
                )
            source_tokens = (
                self.window_source_tokens_per_task
                if self.window_source_tokens_per_task is not None
                else experiment.tokens_per_task * experiment.cycles
            )
            source_budget = resolve_token_budget(
                source_tokens,
                experiment.sequence_length,
                policy=experiment.token_budget_policy,
            )
            task_budget = resolve_token_budget(
                experiment.tokens_per_task,
                experiment.sequence_length,
                policy=experiment.token_budget_policy,
            )
            if source_budget.effective_complete_sequences < (
                experiment.cycles
                * task_budget.effective_complete_sequences
            ):
                raise ValueError(
                    "data.window_source_tokens_per_task does not contain all "
                    "configured disjoint cycle windows"
                )
        elif self.window_source_tokens_per_task is not None:
            raise ValueError(
                "data.window_source_tokens_per_task is only valid for "
                "disjoint_sequence_windows_v1"
            )
        for name, value in (
            ("max_cache_bytes", self.max_cache_bytes),
            ("max_generated_bytes", self.max_generated_bytes),
            ("max_temporary_bytes", self.max_temporary_bytes),
            ("max_input_documents", self.max_input_documents),
            ("max_runtime_seconds", self.max_runtime_seconds),
            ("shuffle_buffer_documents", self.shuffle_buffer_documents),
            ("max_shard_tokens", self.max_shard_tokens),
        ):
            if value <= 0:
                raise ValueError(f"data.{name} must be positive")


@dataclass(frozen=True)
class TrainingSettings:
    global_batch_sequences: int
    physical_microbatch_sequences: int
    device: str
    precision: str
    fp16_loss_conditioning_multiplier: int
    optimizer: str
    peak_lr_5m: float
    peak_lr_12m: float
    warmup_fraction: float
    checkpoint_frequency: int
    retain_last_checkpoints: int
    deterministic: bool

    def validate(self) -> None:
        if self.global_batch_sequences <= 0:
            raise ValueError("training.global_batch_sequences must be positive")
        if self.physical_microbatch_sequences <= 0:
            raise ValueError(
                "training.physical_microbatch_sequences must be positive"
            )
        if self.physical_microbatch_sequences > self.global_batch_sequences:
            raise ValueError("Physical microbatch exceeds global batch")
        if self.device not in {"cpu", "cuda", "auto"}:
            raise ValueError("training.device must be cpu, cuda, or auto")
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("training.precision must be fp32, fp16, or bf16")
        if self.device == "cpu" and self.precision == "fp16":
            raise ValueError("FP16 is unsupported for CPU training")
        if self.fp16_loss_conditioning_multiplier != 2:
            raise ValueError(
                "training.fp16_loss_conditioning_multiplier must be 2"
            )
        if self.optimizer != "adamw":
            raise ValueError("training.optimizer must be adamw")
        if self.peak_lr_5m <= 0 or self.peak_lr_12m <= 0:
            raise ValueError("Configured peak learning rates must be positive")
        if self.warmup_fraction != 0.05:
            raise ValueError("training.warmup_fraction must be 0.05")
        if self.checkpoint_frequency < 0:
            raise ValueError("training.checkpoint_frequency cannot be negative")
        if self.retain_last_checkpoints <= 0:
            raise ValueError("training.retain_last_checkpoints must be positive")


@dataclass(frozen=True)
class FastMemSettings:
    memory_tokens: int
    segment_length: int
    fast_lr: float
    fast_clip: float
    slow_accumulation_k: int

    def validate(self, experiment: ExperimentSettings) -> None:
        if self.memory_tokens != 8:
            raise ValueError("fastmem.memory_tokens must be 8")
        expected_segment = experiment.sequence_length // 2
        if experiment.sequence_length % 2 or self.segment_length != expected_segment:
            raise ValueError(
                "fastmem.segment_length must divide each sequence into two segments"
            )
        if experiment.run_kind == "production" and (
            experiment.sequence_length != 2048 or self.segment_length != 1024
        ):
            raise ValueError(
                "Production FastMem requires 2,048-token sequences and 1,024-token segments"
            )
        if self.fast_lr <= 0:
            raise ValueError("fastmem.fast_lr must be positive")
        if self.fast_clip != 1.0:
            raise ValueError("fastmem.fast_clip must be 1.0")
        if self.slow_accumulation_k != 2:
            raise ValueError("fastmem.slow_accumulation_k must be 2")


@dataclass(frozen=True)
class LauncherSettings:
    gpu_ids: list[int]
    jobs_per_gpu: int
    gpus_per_job: int
    max_parallel_jobs: int
    start_method: str
    fail_fast: bool
    retry_count: int
    rendezvous_port_base: int = 29500
    disk_free_floor_bytes: int = 8_589_934_592
    checkpoint_bytes_per_parameter: int = 20
    checkpoint_fixed_overhead_bytes: int = 67_108_864
    ddp_debug_assert_synced: bool = True

    def validate(self, training: TrainingSettings) -> None:
        if len(set(self.gpu_ids)) != len(self.gpu_ids):
            raise ValueError("launcher.gpu_ids must be unique")
        if any(gpu < 0 for gpu in self.gpu_ids):
            raise ValueError("launcher.gpu_ids cannot contain negative IDs")
        if self.jobs_per_gpu <= 0:
            raise ValueError("launcher.jobs_per_gpu must be positive")
        if self.max_parallel_jobs <= 0:
            raise ValueError("launcher.max_parallel_jobs must be positive")
        if self.retry_count < 0:
            raise ValueError("launcher.retry_count cannot be negative")
        if self.start_method != "subprocess":
            raise ValueError("launcher.start_method must be subprocess")
        if training.device == "cpu":
            if self.gpu_ids or self.gpus_per_job != 0:
                raise ValueError(
                    "CPU jobs require gpu_ids=[] and gpus_per_job=0"
                )
        else:
            if not self.gpu_ids or self.gpus_per_job <= 0:
                raise ValueError("CUDA/auto jobs require GPUs")
            if self.gpus_per_job > len(self.gpu_ids):
                raise ValueError("gpus_per_job exceeds available GPU IDs")
            if len(self.gpu_ids) % self.gpus_per_job:
                raise ValueError(
                    "GPU count must be divisible by gpus_per_job for disjoint groups"
                )
        if not 1024 <= self.rendezvous_port_base <= 65000:
            raise ValueError("launcher.rendezvous_port_base is outside the safe range")
        if self.disk_free_floor_bytes < 0:
            raise ValueError("launcher.disk_free_floor_bytes cannot be negative")
        if self.checkpoint_bytes_per_parameter <= 0:
            raise ValueError(
                "launcher.checkpoint_bytes_per_parameter must be positive"
            )
        if self.checkpoint_fixed_overhead_bytes < 0:
            raise ValueError(
                "launcher.checkpoint_fixed_overhead_bytes cannot be negative"
            )


@dataclass(frozen=True)
class TrackingSettings:
    jsonl: bool
    tensorboard: bool
    tensorboard_flush_seconds: int
    log_every_batches: int
    save_summary_csv: bool
    save_summary_json: bool

    def validate(self) -> None:
        if not self.jsonl:
            raise ValueError("Release jobs require tracking.jsonl=true")
        if self.tensorboard_flush_seconds <= 0:
            raise ValueError("TensorBoard flush interval must be positive")
        if self.log_every_batches <= 0:
            raise ValueError("tracking.log_every_batches must be positive")


@dataclass(frozen=True)
class ProbeSettings:
    enabled: bool
    mode: str
    at_cycle_end: bool
    training_tokens: int
    validation_sequences: int
    evaluation_interval: int
    evaluation_milestones: list[int]

    def validate(self, experiment: ExperimentSettings) -> None:
        if self.mode != PRIMARY_PROBE_MODE:
            raise ValueError(f"probe.mode must be {PRIMARY_PROBE_MODE}")
        if self.enabled and not self.at_cycle_end:
            raise ValueError("Enabled release probes must run at every cycle end")
        if self.training_tokens <= 0 or self.validation_sequences <= 0:
            raise ValueError("Probe token and validation budgets must be positive")
        if self.evaluation_interval <= 0:
            raise ValueError("probe.evaluation_interval must be positive")
        if not self.evaluation_milestones:
            raise ValueError("probe.evaluation_milestones must not be empty")
        if any(step <= 0 for step in self.evaluation_milestones):
            raise ValueError("Probe milestones must be positive")
        resolve_token_budget(
            self.training_tokens,
            experiment.sequence_length,
            policy=experiment.token_budget_policy,
        )


@dataclass(frozen=True)
class ForgettingSettings:
    enabled: bool
    evaluation_schedule: str
    validation_sequences_per_language: int
    primary_memory_evaluation_mode: str
    metric: str

    @classmethod
    def disabled(cls) -> "ForgettingSettings":
        return cls(
            enabled=False,
            evaluation_schedule="after_each_task_boundary",
            validation_sequences_per_language=0,
            primary_memory_evaluation_mode="reset",
            metric="mean_validation_ce_from_best_v1",
        )

    def validate(
        self,
        experiment: ExperimentSettings,
        data: DataSettings,
        training: TrainingSettings,
    ) -> None:
        if self.evaluation_schedule != "after_each_task_boundary":
            raise ValueError(
                "forgetting.evaluation_schedule must be "
                "after_each_task_boundary"
            )
        if self.primary_memory_evaluation_mode != "reset":
            raise ValueError(
                "Forgetting must use reset memory for slow/backbone comparability"
            )
        if self.metric != "mean_validation_ce_from_best_v1":
            raise ValueError(
                "forgetting.metric must be mean_validation_ce_from_best_v1"
            )
        if not self.enabled:
            if self.validation_sequences_per_language != 0:
                raise ValueError(
                    "Disabled forgetting requires zero validation sequences"
                )
            return
        if data.mode != "packed":
            raise ValueError(
                "Forgetting evaluation currently requires packed held-out data"
            )
        if data.language_validation_manifest_template is None:
            raise ValueError(
                "Enabled forgetting requires "
                "data.language_validation_manifest_template"
            )
        if self.validation_sequences_per_language <= 0:
            raise ValueError(
                "Enabled forgetting requires positive validation sequences"
            )
        if (
            self.validation_sequences_per_language
            % training.global_batch_sequences
        ):
            raise ValueError(
                "Forgetting validation sequences must divide the global "
                "logical batch exactly"
            )
        resolve_token_budget(
            self.validation_sequences_per_language
            * experiment.sequence_length,
            experiment.sequence_length,
            policy=experiment.token_budget_policy,
        )


@dataclass(frozen=True)
class LauncherConfig:
    schema_version: int
    experiment: ExperimentSettings
    data: DataSettings
    training: TrainingSettings
    fastmem: FastMemSettings
    launcher: LauncherSettings
    tracking: TrackingSettings
    probe: ProbeSettings
    forgetting: ForgettingSettings | None = None

    def validate(self) -> None:
        if self.schema_version != LAUNCHER_SCHEMA_VERSION:
            raise ValueError(
                f"launcher schema_version must be {LAUNCHER_SCHEMA_VERSION}"
            )
        self.experiment.validate()
        self.data.validate(self.experiment)
        self.training.validate()
        self.fastmem.validate(self.experiment)
        self.launcher.validate(self.training)
        self.tracking.validate()
        self.probe.validate(self.experiment)
        forgetting = self.forgetting or ForgettingSettings.disabled()
        forgetting.validate(self.experiment, self.data, self.training)
        if (
            self.experiment.run_kind in {"production", "scaled_budget"}
            and not self.probe.enabled
        ):
            raise ValueError(
                "Production and scaled-budget runs require cycle-end probes"
            )

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        # These optional fields postdate schema version 1.  Omitting inactive
        # defaults keeps existing launcher/checkpoint identities resumable.
        if values["data"].get("language_validation_manifest_template") is None:
            values["data"].pop("language_validation_manifest_template", None)
        if values["data"].get("window_source_tokens_per_task") is None:
            values["data"].pop("window_source_tokens_per_task", None)
        if values.get("forgetting") is None:
            values.pop("forgetting", None)
        if values["launcher"].get("ddp_debug_assert_synced") is True:
            values["launcher"].pop("ddp_debug_assert_synced", None)
        return values
