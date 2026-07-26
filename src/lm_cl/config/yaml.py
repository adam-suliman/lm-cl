from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from lm_cl.config.schema import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    RuntimeConfig,
    TrainingConfig,
    VariantConfig,
    strict_dataclass,
)


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle)
    if not isinstance(values, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return values


def _section(
    root_path: Path,
    root: dict[str, Any],
    name: str,
) -> dict[str, Any]:
    if name not in root:
        raise ValueError(f"Missing experiment section: {name}")
    value = root[name]
    if isinstance(value, str):
        section_path = (root_path.parent / value).resolve()
        return _read_yaml(section_path)
    if not isinstance(value, dict):
        raise ValueError(f"Experiment section {name} must be a mapping or YAML path")
    return value


def load_model_config(path: str | Path) -> ModelConfig:
    path = Path(path).resolve()
    config = strict_dataclass(ModelConfig, _read_yaml(path), "model")
    config.validate()
    return config


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    path = Path(path).resolve()
    root = _read_yaml(path)
    expected = {
        "schema_version",
        "run_name",
        "model",
        "variant",
        "data",
        "training",
        "runtime",
    }
    unknown = sorted(set(root) - expected)
    missing = sorted(expected - set(root))
    if unknown:
        raise ValueError(f"Unknown experiment fields: {unknown}")
    if missing:
        raise ValueError(f"Missing experiment fields: {missing}")

    config = ExperimentConfig(
        schema_version=root["schema_version"],
        run_name=root["run_name"],
        model=strict_dataclass(ModelConfig, _section(path, root, "model"), "model"),
        variant=strict_dataclass(
            VariantConfig, _section(path, root, "variant"), "variant"
        ),
        data=strict_dataclass(DataConfig, _section(path, root, "data"), "data"),
        training=strict_dataclass(
            TrainingConfig, _section(path, root, "training"), "training"
        ),
        runtime=strict_dataclass(
            RuntimeConfig, _section(path, root, "runtime"), "runtime"
        ),
    )
    config.validate()
    return config


def save_resolved_config(config: ExperimentConfig, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            asdict(config),
            handle,
            sort_keys=False,
            default_flow_style=False,
        )
