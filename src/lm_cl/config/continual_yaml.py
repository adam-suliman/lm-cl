from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from lm_cl.config.continual_schema import (
    ContinualExperimentConfig,
    DistributedConfig,
    ContinualOptimizationConfig,
    ContinualRuntimeConfig,
    ContinualTaskConfig,
    TrainSourceConfig,
)
from lm_cl.config.data_schema import (
    DataPipelineConfig,
    DatasetReference,
    PackingConfig,
    PackedManifestIdentity,
    ReaderConfig,
    SelectionConfig,
    StageConfig,
    StorageConfig,
    TokenizerReference,
)
from lm_cl.config.data_yaml import load_data_pipeline_config
from lm_cl.config.schema import (
    DataConfig,
    ModelConfig,
    VariantConfig,
    strict_dataclass,
)
from lm_cl.config.yaml import load_model_config


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle)
    if not isinstance(values, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    if set(values) == {"base", "overrides"}:
        if not isinstance(values["overrides"], dict):
            raise ValueError("Composed continual overrides must be a mapping")
        base = _read_yaml((path.parent / values["base"]).resolve())

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


def _mapping_or_path(
    root_path: Path, value: Any, context: str
) -> tuple[Path, dict[str, Any]]:
    if isinstance(value, str):
        path = (root_path.parent / value).resolve()
        return path, _read_yaml(path)
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a mapping or YAML path")
    return root_path, value


def _model(root_path: Path, value: Any) -> ModelConfig:
    if isinstance(value, str):
        return load_model_config((root_path.parent / value).resolve())
    return strict_dataclass(ModelConfig, value, "model")


def _variant(root_path: Path, value: Any) -> VariantConfig:
    _, values = _mapping_or_path(root_path, value, "variant")
    return strict_dataclass(VariantConfig, values, "variant")


def _data_pipeline_from_mapping(values: dict[str, Any]) -> DataPipelineConfig:
    expected = {
        "schema_version",
        "name",
        "mode",
        "run_kind",
        "dataset",
        "tokenizer",
        "selection",
        "storage",
        "packing",
        "stage",
        "reader",
        "packed_manifest_identity",
    }
    unknown = sorted(set(values) - expected)
    missing = sorted((expected - {"packed_manifest_identity"}) - set(values))
    if unknown or missing:
        raise ValueError(
            f"Invalid packed data fields; unknown={unknown}, missing={missing}"
        )
    return DataPipelineConfig(
        schema_version=values["schema_version"],
        name=values["name"],
        mode=values["mode"],
        run_kind=values["run_kind"],
        dataset=strict_dataclass(
            DatasetReference, values["dataset"], "dataset"
        ),
        tokenizer=strict_dataclass(
            TokenizerReference, values["tokenizer"], "tokenizer"
        ),
        selection=strict_dataclass(
            SelectionConfig, values["selection"], "selection"
        ),
        storage=strict_dataclass(
            StorageConfig, values["storage"], "storage"
        ),
        packing=strict_dataclass(
            PackingConfig, values["packing"], "packing"
        ),
        stage=strict_dataclass(StageConfig, values["stage"], "stage"),
        reader=strict_dataclass(ReaderConfig, values["reader"], "reader"),
        packed_manifest_identity=(
            None
            if values.get("packed_manifest_identity") is None
            else strict_dataclass(
                PackedManifestIdentity,
                values["packed_manifest_identity"],
                "packed_manifest_identity",
            )
        ),
    )


def _source(root_path: Path, values: Any, context: str) -> TrainSourceConfig:
    if not isinstance(values, dict):
        raise ValueError(f"{context} must be a mapping")
    expected = {"kind", "synthetic", "packed"}
    unknown = sorted(set(values) - expected)
    missing = sorted(expected - set(values))
    if unknown or missing:
        raise ValueError(
            f"Invalid {context} fields; unknown={unknown}, missing={missing}"
        )
    synthetic = (
        None
        if values["synthetic"] is None
        else strict_dataclass(DataConfig, values["synthetic"], f"{context}.synthetic")
    )
    packed_value = values["packed"]
    packed = None
    if packed_value is not None:
        if isinstance(packed_value, str):
            packed = load_data_pipeline_config(
                (root_path.parent / packed_value).resolve()
            )
        elif isinstance(packed_value, dict):
            packed = _data_pipeline_from_mapping(packed_value)
        else:
            raise ValueError(f"{context}.packed must be null, mapping, or path")
    return TrainSourceConfig(
        kind=values["kind"],
        synthetic=synthetic,
        packed=packed,
    )


def _task(root_path: Path, values: Any, index: int) -> ContinualTaskConfig:
    if not isinstance(values, dict):
        raise ValueError(f"tasks[{index}] must be a mapping")
    expected = {
        "language",
        "task_index",
        "cycle_index",
        "logical_batches",
        "input_token_budget",
        "train_source",
        "validation_source",
        "validation_logical_batches",
        "train_sequence_prefix_count",
        "train_sequence_offset_count",
    }
    unknown = sorted(set(values) - expected)
    missing = sorted(
        (
            expected
            - {"train_sequence_prefix_count", "train_sequence_offset_count"}
        )
        - set(values)
    )
    if unknown or missing:
        raise ValueError(
            f"Invalid tasks[{index}] fields; unknown={unknown}, missing={missing}"
        )
    validation = values["validation_source"]
    return ContinualTaskConfig(
        language=values["language"],
        task_index=values["task_index"],
        cycle_index=values["cycle_index"],
        logical_batches=values["logical_batches"],
        input_token_budget=values["input_token_budget"],
        train_source=_source(
            root_path, values["train_source"], f"tasks[{index}].train_source"
        ),
        validation_source=(
            None
            if validation is None
            else _source(
                root_path,
                validation,
                f"tasks[{index}].validation_source",
            )
        ),
        validation_logical_batches=values["validation_logical_batches"],
        train_sequence_prefix_count=values.get(
            "train_sequence_prefix_count"
        ),
        train_sequence_offset_count=values.get(
            "train_sequence_offset_count", 0
        ),
    )


def load_continual_config(
    path: str | Path,
    *,
    allow_pending_packed: bool = False,
) -> ContinualExperimentConfig:
    path = Path(path).resolve()
    root = _read_yaml(path)
    allowed = {
        "schema_version",
        "run_name",
        "model",
        "variant",
        "optimization",
        "runtime",
        "tasks",
        "distributed",
    }
    required = allowed - {"distributed"}
    unknown = sorted(set(root) - allowed)
    missing = sorted(required - set(root))
    if unknown or missing:
        raise ValueError(
            f"Invalid continual config fields; unknown={unknown}, missing={missing}"
        )
    if not isinstance(root["tasks"], list):
        raise ValueError("tasks must be a list")
    config = ContinualExperimentConfig(
        schema_version=root["schema_version"],
        run_name=root["run_name"],
        model=_model(path, root["model"]),
        variant=_variant(path, root["variant"]),
        optimization=strict_dataclass(
            ContinualOptimizationConfig,
            root["optimization"],
            "optimization",
        ),
        runtime=strict_dataclass(
            ContinualRuntimeConfig, root["runtime"], "runtime"
        ),
        tasks=[
            _task(path, values, index)
            for index, values in enumerate(root["tasks"])
        ],
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


def save_continual_config(
    config: ContinualExperimentConfig, path: str | Path
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(asdict(config), handle, sort_keys=False)
