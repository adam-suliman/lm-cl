from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any

import yaml

from lm_cl.config.schema import strict_dataclass
from lm_cl.launcher.schema import (
    DataSettings,
    ExperimentSettings,
    FastMemSettings,
    ForgettingSettings,
    LauncherConfig,
    LauncherSettings,
    ProbeSettings,
    TrackingSettings,
    TrainingSettings,
)


_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_PATH_FIELDS = {
    ("experiment", "output_root"),
    ("experiment", "repository_root"),
    ("data", "manifest_root"),
    ("data", "tokenizer_manifest"),
    ("data", "dataset_cache_root"),
    ("data", "generated_root"),
    ("data", "probe_training_manifest"),
    ("data", "probe_validation_manifest"),
}


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        root = yaml.safe_load(handle)
    if not isinstance(root, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return root


def _expand_environment(value: str, *, context: str) -> str:
    missing: list[str] = []

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in os.environ:
            missing.append(name)
            return match.group(0)
        return os.environ[name]

    expanded = _ENV_RE.sub(replace, value)
    if missing:
        raise ValueError(
            f"Missing environment variables for {context}: {sorted(set(missing))}"
        )
    if "${" in expanded:
        raise ValueError(f"Malformed environment interpolation in {context}")
    return expanded


def _resolve_paths(root: dict[str, Any], config_path: Path) -> None:
    experiment = root.get("experiment")
    if not isinstance(experiment, dict):
        return
    repository_value = experiment.get("repository_root")
    if not isinstance(repository_value, str):
        return
    expanded_repo = _expand_environment(
        repository_value, context="experiment.repository_root"
    )
    repository_root = Path(expanded_repo).expanduser()
    if not repository_root.is_absolute():
        repository_root = config_path.parent / repository_root
    repository_root = repository_root.resolve()
    experiment["repository_root"] = str(repository_root)

    for section, field in sorted(_PATH_FIELDS):
        values = root.get(section)
        if not isinstance(values, dict):
            continue
        value = values.get(field)
        if value is None:
            continue
        if not isinstance(value, str):
            continue
        if value == "":
            continue
        expanded = _expand_environment(value, context=f"{section}.{field}")
        path = Path(expanded).expanduser()
        if not path.is_absolute():
            path = repository_root / path
        values[field] = str(path.resolve())


def _validate_safe_roots(config: LauncherConfig) -> None:
    repository = Path(config.experiment.repository_root).resolve()
    output = Path(config.experiment.output_root).resolve()
    forbidden = {Path("/"), Path.home().resolve(), repository}
    if output in forbidden:
        raise ValueError("experiment.output_root is an unsafe broad path")
    for data_root_value in (
        config.data.manifest_root,
        config.data.dataset_cache_root,
        config.data.generated_root,
    ):
        if not data_root_value:
            continue
        data_root = Path(data_root_value).resolve()
        if output == data_root or output in data_root.parents:
            raise ValueError(
                "experiment.output_root cannot contain or equal a data root"
            )
    if config.data.mode == "packed":
        manifest_root = Path(config.data.manifest_root).resolve()
        generated_root = Path(config.data.generated_root).resolve()
        expected = generated_root / "stages"
        if manifest_root != expected:
            raise ValueError(
                "data.manifest_root must equal data.generated_root/stages so "
                "the validated packed reader opens the exact frozen stages"
            )


def config_from_mapping(root: dict[str, Any]) -> LauncherConfig:
    expected = {
        "schema_version",
        "experiment",
        "data",
        "training",
        "fastmem",
        "launcher",
        "tracking",
        "probe",
        "forgetting",
    }
    unknown = sorted(set(root) - expected)
    missing = sorted((expected - {"forgetting"}) - set(root))
    if unknown or missing:
        raise ValueError(
            f"Invalid launcher fields; unknown={unknown}, missing={missing}"
        )
    config = LauncherConfig(
        schema_version=root["schema_version"],
        experiment=strict_dataclass(
            ExperimentSettings, root["experiment"], "experiment"
        ),
        data=strict_dataclass(DataSettings, root["data"], "data"),
        training=strict_dataclass(
            TrainingSettings, root["training"], "training"
        ),
        fastmem=strict_dataclass(
            FastMemSettings, root["fastmem"], "fastmem"
        ),
        launcher=strict_dataclass(
            LauncherSettings, root["launcher"], "launcher"
        ),
        tracking=strict_dataclass(
            TrackingSettings, root["tracking"], "tracking"
        ),
        probe=strict_dataclass(ProbeSettings, root["probe"], "probe"),
        forgetting=(
            None
            if root.get("forgetting") is None
            else strict_dataclass(
                ForgettingSettings, root["forgetting"], "forgetting"
            )
        ),
    )
    config.validate()
    _validate_safe_roots(config)
    return config


def load_launcher_config(
    path: str | Path,
    *,
    overrides: dict[str, Any] | None = None,
) -> LauncherConfig:
    config_path = Path(path).expanduser().resolve()
    root = _read_yaml(config_path)
    if overrides:
        root = apply_launcher_overrides(root, overrides)
    _resolve_paths(root, config_path)
    return config_from_mapping(root)


def apply_launcher_overrides(
    root: dict[str, Any], overrides: dict[str, Any]
) -> dict[str, Any]:
    result = copy.deepcopy(root)
    allowed = {
        "name": ("experiment", "name"),
        "models": ("experiment", "models"),
        "seeds": ("experiment", "seed"),
        "cycles": ("experiment", "cycles"),
        "tokens_per_task": ("experiment", "tokens_per_task"),
        "gpu_ids": ("launcher", "gpu_ids"),
        "jobs_per_gpu": ("launcher", "jobs_per_gpu"),
        "gpus_per_job": ("launcher", "gpus_per_job"),
        "output_root": ("experiment", "output_root"),
        "precision": ("training", "precision"),
        "resume": ("experiment", "resume"),
        "probe_enabled": ("probe", "enabled"),
    }
    unknown = sorted(set(overrides) - set(allowed))
    if unknown:
        raise ValueError(f"Unknown launcher overrides: {unknown}")
    for name, value in overrides.items():
        if value is None:
            continue
        section, field = allowed[name]
        result[section][field] = value
    return result


def save_yaml(value: dict[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(value, handle, sort_keys=False)
