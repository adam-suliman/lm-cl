from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from lm_cl.config import (
    ContinualExperimentConfig,
    ContinualOptimizationConfig,
    ContinualRuntimeConfig,
    ContinualTaskConfig,
    DataConfig,
    DistributedConfig,
    ProbeAUCPolicy,
    ProbeExperimentConfig,
    TrainSourceConfig,
    VariantConfig,
    load_model_config,
    save_continual_config,
    save_probe_config,
)
from lm_cl.launcher.config import save_yaml
from lm_cl.launcher.data import data_pipeline_from_identity
from lm_cl.launcher.schema import (
    PRIMARY_PROBE_MODE,
    PUBLIC_LANGUAGE_ORDER,
    PUBLIC_MODEL_VARIANTS,
    LauncherConfig,
)
from lm_cl.training.checkpoint import canonical_sha256


@dataclass(frozen=True)
class JobSpec:
    public_model: str
    internal_variant: str
    seed: int
    output_dir: str
    resolved_experiment: dict[str, Any]
    resolved_sha256: str
    scientific_sha256: str

    @property
    def job_id(self) -> str:
        return f"{self.public_model}-seed-{self.seed}"


def _scientific_identity(
    config: LauncherConfig,
    *,
    public_model: str,
    seed: int,
    data_contract: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "lm-cl-public-scientific-identity-v1",
        "public_model": public_model,
        "internal_variant": PUBLIC_MODEL_VARIANTS[public_model],
        "model_size": config.experiment.model_size,
        "seed": seed,
        "language_order": list(PUBLIC_LANGUAGE_ORDER),
        "task_token_budget": data_contract["task_token_budget"],
        "sequence_length": config.experiment.sequence_length,
        "training": {
            key: value
            for key, value in asdict(config.training).items()
            if key not in {"checkpoint_frequency", "retain_last_checkpoints"}
        },
        "fastmem": asdict(config.fastmem),
        "probe": asdict(config.probe),
        "tokenizer": data_contract.get("tokenizer"),
        "cycle_manifest_policy": data_contract["cycle_manifest_policy"],
    }


def expand_job_specs(
    config: LauncherConfig,
    data_contract: dict[str, Any],
) -> list[JobSpec]:
    output_root = Path(config.experiment.output_root).resolve()
    jobs: list[JobSpec] = []
    seen_paths: set[Path] = set()
    for public_model in config.experiment.models:
        for seed in config.experiment.seeds:
            output_dir = (
                output_root
                / config.experiment.name
                / public_model
                / f"seed-{seed}"
            ).resolve()
            if output_dir in seen_paths:
                raise ValueError(f"Job output collision: {output_dir}")
            seen_paths.add(output_dir)
            launcher_mapping = copy.deepcopy(config.to_dict())
            launcher_mapping["experiment"]["models"] = [public_model]
            launcher_mapping["experiment"]["seed"] = seed
            launcher_mapping["experiment"]["output_root"] = str(output_root)
            scientific = _scientific_identity(
                config,
                public_model=public_model,
                seed=seed,
                data_contract=data_contract,
            )
            resolved = {
                "resolved_experiment_schema_version": 1,
                "public_model": public_model,
                "internal_variant": PUBLIC_MODEL_VARIANTS[public_model],
                "seed": seed,
                "output_dir": str(output_dir),
                "requested_horizon_cycles": config.experiment.cycles,
                "launcher_config": launcher_mapping,
                "data_contract": data_contract,
                "scientific_identity": scientific,
                "scientific_sha256": canonical_sha256(scientific),
            }
            resolved_sha = canonical_sha256(resolved)
            resolved["resolved_experiment_sha256"] = resolved_sha
            jobs.append(
                JobSpec(
                    public_model=public_model,
                    internal_variant=PUBLIC_MODEL_VARIANTS[public_model],
                    seed=seed,
                    output_dir=str(output_dir),
                    resolved_experiment=resolved,
                    resolved_sha256=resolved_sha,
                    scientific_sha256=resolved["scientific_sha256"],
                )
            )
    return jobs


def write_resolved_job(spec: JobSpec) -> Path:
    output_dir = Path(spec.output_dir)
    path = output_dir / "resolved_experiment.yaml"
    save_yaml(spec.resolved_experiment, path)
    return path


def _model_config(config: LauncherConfig):
    repository = Path(config.experiment.repository_root)
    model_files = {
        "5m": repository / "configs/models/zyphra_5m.yaml",
        "12m": repository / "configs/models/zyphra_12m.yaml",
        "tiny": repository / "configs/models/tiny_test.yaml",
    }
    model = load_model_config(model_files[config.experiment.model_size])
    learning_rate = (
        config.training.peak_lr_12m
        if config.experiment.model_size == "12m"
        else config.training.peak_lr_5m
    )
    model = replace(model, learning_rate=learning_rate)
    model.validate()
    return model


def _variant(config: LauncherConfig, public_model: str) -> VariantConfig:
    if public_model == "transformer":
        variant = VariantConfig(
            name="backbone_clean",
            memory_enabled=False,
            persistent_fast_memory=False,
            fast_lr=0.0,
            memory_tokens=0,
            slow_update_period_k=1,
            fast_memory_grad_clip_norm=None,
        )
    elif public_model == "fastmem_rmt":
        variant = VariantConfig(
            name="fastmem_rmt",
            memory_enabled=True,
            persistent_fast_memory=True,
            fast_lr=config.fastmem.fast_lr,
            memory_tokens=config.fastmem.memory_tokens,
            slow_update_period_k=config.fastmem.slow_accumulation_k,
            fast_memory_grad_clip_norm=config.fastmem.fast_clip,
            segment_length=config.fastmem.segment_length,
            reset_policy="task_boundary_from_m0_stopgrad",
            memory_evaluation_policy="reset_and_carried",
        )
    else:
        raise ValueError(f"Unknown public model: {public_model}")
    variant.validate()
    return variant


def _distributed(config: LauncherConfig) -> DistributedConfig | None:
    if config.launcher.gpus_per_job <= 1:
        return None
    distributed = DistributedConfig(
        enabled=True,
        backend="nccl",
        timeout_seconds=300,
        partition_rule="contiguous_floor_v1",
        reduction_policy="ddp_average_world_scaled_global_sum_v1",
        active_memory_sync_policy=(
            "sum_unscale_normalize_clip_rank0_broadcast_v1"
        ),
        ddp_broadcast_buffers=False,
        ddp_find_unused_parameters=False,
        debug_assert_synced=True,
        per_rank_diagnostic_logs=False,
    )
    distributed.validate(runtime_device="cuda")
    return distributed


def _optimization(config: LauncherConfig) -> ContinualOptimizationConfig:
    return ContinualOptimizationConfig(
        optimizer=config.training.optimizer,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-8,
        weight_decay=0.1,
        warmup_fraction=config.training.warmup_fraction,
        global_sequences_per_logical_batch=(
            config.training.global_batch_sequences
        ),
        physical_microbatch_sequences=(
            config.training.physical_microbatch_sequences
        ),
        slow_gradient_clip_norm=None,
        precision=config.training.precision,
        fp16_loss_conditioning_multiplier=(
            config.training.fp16_loss_conditioning_multiplier
        ),
        checkpoint_every_logical_batches=(
            config.training.checkpoint_frequency
        ),
    )


def _runtime(
    config: LauncherConfig,
    *,
    seed: int,
    output_dir: Path,
    tensorboard_dir: Path | None,
) -> ContinualRuntimeConfig:
    return ContinualRuntimeConfig(
        seed=seed,
        device=(
            "cuda" if config.training.device == "auto" else config.training.device
        ),
        deterministic_algorithms=config.training.deterministic,
        output_dir=str(output_dir),
        metrics_jsonl="metrics.jsonl",
        tensorboard_dir=(
            None if tensorboard_dir is None else str(tensorboard_dir)
        ),
        tensorboard_flush_seconds=(
            config.tracking.tensorboard_flush_seconds
        ),
        tensorboard_log_every_batches=config.tracking.log_every_batches,
    )


def build_continual_job_config(
    config: LauncherConfig,
    spec: JobSpec,
) -> ContinualExperimentConfig:
    model = _model_config(config)
    variant = _variant(config, spec.public_model)
    data_contract = spec.resolved_experiment["data_contract"]
    budget = data_contract["task_token_budget"]
    tasks: list[ContinualTaskConfig] = []
    for cycle, cycle_items in enumerate(data_contract["data_manifests"]):
        for language_index, language in enumerate(PUBLIC_LANGUAGE_ORDER):
            task_index = cycle * len(PUBLIC_LANGUAGE_ORDER) + language_index
            identity = cycle_items[language]
            if config.data.mode == "synthetic":
                synthetic = DataConfig(
                    backend="synthetic",
                    vocab_size=model.vocab_size,
                    sequence_length=config.experiment.sequence_length,
                    num_sequences=budget["effective_complete_sequences"],
                    seed=spec.seed * 1_000_003 + task_index,
                    ignore_index=-100,
                    mask_probability=0.0,
                )
                source = TrainSourceConfig(
                    kind="synthetic", synthetic=synthetic, packed=None
                )
            else:
                packed = data_pipeline_from_identity(
                    config,
                    identity,
                    purpose=identity["purpose"],
                    global_sequences_per_batch=(
                        config.training.global_batch_sequences
                    ),
                )
                source = TrainSourceConfig(
                    kind="packed_shards", synthetic=None, packed=packed
                )
            tasks.append(
                ContinualTaskConfig(
                    language=language,
                    task_index=task_index,
                    cycle_index=cycle,
                    logical_batches=None,
                    input_token_budget=budget["effective_input_tokens"],
                    train_source=source,
                    validation_source=None,
                    validation_logical_batches=0,
                    train_sequence_prefix_count=budget[
                        "effective_complete_sequences"
                    ],
                )
            )
    job_dir = Path(spec.output_dir)
    continual = ContinualExperimentConfig(
        schema_version=1,
        run_name=f"{config.experiment.name}-{spec.job_id}",
        model=model,
        variant=variant,
        optimization=_optimization(config),
        runtime=_runtime(
            config,
            seed=spec.seed,
            output_dir=job_dir,
            tensorboard_dir=(
                job_dir / "tensorboard"
                if config.tracking.tensorboard
                else None
            ),
        ),
        tasks=tasks,
        distributed=_distributed(config),
    )
    continual.validate()
    return continual


def save_internal_continual_config(
    continual: ContinualExperimentConfig,
    job_dir: str | Path,
) -> Path:
    path = Path(job_dir) / "internal" / "continual.yaml"
    save_continual_config(continual, path)
    return path


def build_probe_job_config(
    config: LauncherConfig,
    spec: JobSpec,
    *,
    cycle_index: int,
    source_checkpoint: str | Path,
    source_checkpoint_sha256: str,
) -> ProbeExperimentConfig:
    model = _model_config(config)
    variant = _variant(config, spec.public_model)
    contract = spec.resolved_experiment["data_contract"]
    probe_budget = contract["probe_token_budget"]
    if config.data.mode == "synthetic":
        train_source = TrainSourceConfig(
            kind="synthetic",
            synthetic=DataConfig(
                backend="synthetic",
                vocab_size=model.vocab_size,
                sequence_length=config.experiment.sequence_length,
                num_sequences=probe_budget["effective_complete_sequences"],
                seed=spec.seed + 100_000,
                ignore_index=-100,
                mask_probability=0.0,
            ),
            packed=None,
        )
        validation_source = TrainSourceConfig(
            kind="synthetic",
            synthetic=DataConfig(
                backend="synthetic",
                vocab_size=model.vocab_size,
                sequence_length=config.experiment.sequence_length,
                num_sequences=config.probe.validation_sequences,
                seed=spec.seed + 200_000,
                ignore_index=-100,
                mask_probability=0.0,
            ),
            packed=None,
        )
        if (
            probe_budget["effective_complete_sequences"]
            % config.training.global_batch_sequences
        ):
            raise ValueError(
                "Synthetic probe sequences must divide the global batch exactly"
            )
        train_logical_batches = (
            probe_budget["effective_complete_sequences"]
            // config.training.global_batch_sequences
        )
        input_token_budget = None
        prefix = None
    else:
        train_identity = contract["probe_training_manifest"]
        validation_identity = contract["probe_validation_manifest"]
        train_source = TrainSourceConfig(
            kind="packed_shards",
            synthetic=None,
            packed=data_pipeline_from_identity(
                config,
                train_identity,
                purpose=train_identity["purpose"],
                global_sequences_per_batch=(
                    config.training.global_batch_sequences
                ),
            ),
        )
        validation_source = TrainSourceConfig(
            kind="packed_shards",
            synthetic=None,
            packed=data_pipeline_from_identity(
                config,
                validation_identity,
                purpose=validation_identity["purpose"],
                global_sequences_per_batch=(
                    config.training.global_batch_sequences
                ),
            ),
        )
        train_logical_batches = None
        input_token_budget = probe_budget["effective_input_tokens"]
        prefix = probe_budget["effective_complete_sequences"]
    probe_dir = Path(spec.output_dir) / "probes" / f"cycle-{cycle_index + 1:04d}"
    probe = ProbeExperimentConfig(
        schema_version=1,
        run_name=f"{config.experiment.name}-{spec.job_id}-probe-cycle-{cycle_index + 1}",
        run_kind=(
            "production"
            if config.experiment.run_kind == "production"
            else "smoke"
        ),
        source_checkpoint=str(Path(source_checkpoint).resolve()),
        source_checkpoint_status="frozen",
        source_checkpoint_sha256=source_checkpoint_sha256,
        model=model,
        variant=variant,
        probe_mode=PRIMARY_PROBE_MODE,
        fast_lr_override=None,
        optimization=_optimization(config),
        runtime=_runtime(
            config,
            seed=spec.seed + 10_000,
            output_dir=probe_dir,
            tensorboard_dir=(
                Path(spec.output_dir)
                / "tensorboard"
                / f"probe-cycle-{cycle_index + 1:04d}"
                if config.tracking.tensorboard
                else None
            ),
        ),
        train_source=train_source,
        validation_source=validation_source,
        train_logical_batches=train_logical_batches,
        input_token_budget=input_token_budget,
        validation_sequences=config.probe.validation_sequences,
        evaluation_interval_logical_steps=(
            config.probe.evaluation_interval
        ),
        early_milestones=list(config.probe.evaluation_milestones),
        auc=ProbeAUCPolicy(
            primary="normalized_trapezoidal_ce_v1",
            interpolation="trapezoidal",
            x_axis="cumulative_input_tokens",
            normalization="observed_span_to_unit_interval",
            smoothing="none",
            percentage_reference="first_probe_auc",
        ),
        train_sequence_prefix_count=prefix,
        requested_input_token_budget=(
            config.probe.training_tokens
            if input_token_budget is not None
            else None
        ),
        token_budget_policy=(
            config.experiment.token_budget_policy
            if input_token_budget is not None
            else None
        ),
        distributed=_distributed(config),
    )
    probe.validate()
    return probe


def save_internal_probe_config(
    probe: ProbeExperimentConfig,
    job_dir: str | Path,
    *,
    cycle_index: int,
) -> Path:
    path = (
        Path(job_dir)
        / "internal"
        / "probes"
        / f"cycle-{cycle_index + 1:04d}.yaml"
    )
    save_probe_config(probe, path)
    return path
