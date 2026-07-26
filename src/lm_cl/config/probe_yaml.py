from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from lm_cl.config.continual_schema import (
    ContinualOptimizationConfig,
    ContinualRuntimeConfig,
    DistributedConfig,
)
from lm_cl.config.continual_yaml import _model, _source, _variant
from lm_cl.config.probe_schema import (
    ProbeAUCPolicy,
    ProbeExperimentConfig,
)
from lm_cl.config.schema import strict_dataclass


def _read(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle)
    if not isinstance(values, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    if set(values) == {"base", "overrides"}:
        if not isinstance(values["overrides"], dict):
            raise ValueError("Composed probe overrides must be a mapping")
        base = _read((path.parent / values["base"]).resolve())

        def merge(left: dict, right: dict) -> dict:
            result = dict(left)
            for key, value in right.items():
                if isinstance(value, dict) and isinstance(result.get(key), dict):
                    result[key] = merge(result[key], value)
                else:
                    result[key] = value
            return result

        values = merge(base, values["overrides"])
    return values


def load_probe_config(
    path: str | Path,
    *,
    allow_pending_packed: bool = False,
) -> ProbeExperimentConfig:
    path = Path(path).expanduser().resolve()
    root = _read(path)
    allowed = {
        "schema_version",
        "run_name",
        "run_kind",
        "source_checkpoint",
        "source_checkpoint_status",
        "source_checkpoint_sha256",
        "model",
        "variant",
        "probe_mode",
        "fast_lr_override",
        "optimization",
        "runtime",
        "train_source",
        "validation_source",
        "train_logical_batches",
        "input_token_budget",
        "validation_sequences",
        "evaluation_interval_logical_steps",
        "early_milestones",
        "auc",
        "train_sequence_prefix_count",
        "requested_input_token_budget",
        "token_budget_policy",
        "distributed",
    }
    required = allowed - {
        "distributed",
        "train_sequence_prefix_count",
        "source_checkpoint_status",
        "source_checkpoint_sha256",
        "requested_input_token_budget",
        "token_budget_policy",
    }
    unknown = sorted(set(root) - allowed)
    missing = sorted(required - set(root))
    if unknown or missing:
        raise ValueError(
            f"Invalid probe config fields; unknown={unknown}, missing={missing}"
        )
    source_checkpoint = Path(root["source_checkpoint"]).expanduser()
    if not source_checkpoint.is_absolute():
        source_checkpoint = (path.parent / source_checkpoint).resolve()
    config = ProbeExperimentConfig(
        schema_version=root["schema_version"],
        run_name=root["run_name"],
        run_kind=root["run_kind"],
        source_checkpoint=str(source_checkpoint),
        source_checkpoint_status=root.get(
            "source_checkpoint_status", "legacy_runtime"
        ),
        source_checkpoint_sha256=root.get("source_checkpoint_sha256"),
        model=_model(path, root["model"]),
        variant=_variant(path, root["variant"]),
        probe_mode=root["probe_mode"],
        fast_lr_override=root["fast_lr_override"],
        optimization=strict_dataclass(
            ContinualOptimizationConfig,
            root["optimization"],
            "optimization",
        ),
        runtime=strict_dataclass(
            ContinualRuntimeConfig,
            root["runtime"],
            "runtime",
        ),
        train_source=_source(path, root["train_source"], "train_source"),
        validation_source=_source(
            path,
            root["validation_source"],
            "validation_source",
        ),
        train_logical_batches=root["train_logical_batches"],
        input_token_budget=root["input_token_budget"],
        validation_sequences=root["validation_sequences"],
        evaluation_interval_logical_steps=(
            root["evaluation_interval_logical_steps"]
        ),
        early_milestones=root["early_milestones"],
        auc=strict_dataclass(ProbeAUCPolicy, root["auc"], "auc"),
        train_sequence_prefix_count=root.get(
            "train_sequence_prefix_count"
        ),
        requested_input_token_budget=root.get(
            "requested_input_token_budget"
        ),
        token_budget_policy=root.get("token_budget_policy"),
        distributed=(
            None
            if root.get("distributed") is None
            else strict_dataclass(
                DistributedConfig,
                root["distributed"],
                "distributed",
            )
        ),
    )
    config.validate(allow_pending_packed=allow_pending_packed)
    return config


def save_probe_config(
    config: ProbeExperimentConfig,
    path: str | Path,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(asdict(config), handle, sort_keys=False)
