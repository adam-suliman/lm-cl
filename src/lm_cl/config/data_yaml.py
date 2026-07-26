from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import yaml

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
from lm_cl.config.schema import strict_dataclass


def load_data_pipeline_config(path: str | Path) -> DataPipelineConfig:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        root = yaml.safe_load(handle)
    if not isinstance(root, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    if set(root) == {"base", "overrides"}:
        base_path = (path.parent / root["base"]).resolve()
        with base_path.open("r", encoding="utf-8") as handle:
            base = yaml.safe_load(handle)
        if not isinstance(base, dict) or not isinstance(root["overrides"], dict):
            raise ValueError("Composed data config requires mapping base/overrides")

        def merge(left: dict, right: dict) -> dict:
            result = dict(left)
            for key, value in right.items():
                if isinstance(value, dict) and isinstance(result.get(key), dict):
                    result[key] = merge(result[key], value)
                else:
                    result[key] = value
            return result

        root = merge(base, root["overrides"])
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
    unknown = sorted(set(root) - expected)
    missing = sorted((expected - {"packed_manifest_identity"}) - set(root))
    if unknown:
        raise ValueError(f"Unknown data-pipeline fields: {unknown}")
    if missing:
        raise ValueError(f"Missing data-pipeline fields: {missing}")

    config = DataPipelineConfig(
        schema_version=root["schema_version"],
        name=root["name"],
        mode=root["mode"],
        run_kind=root["run_kind"],
        dataset=strict_dataclass(
            DatasetReference, root["dataset"], "dataset"
        ),
        tokenizer=strict_dataclass(
            TokenizerReference, root["tokenizer"], "tokenizer"
        ),
        selection=strict_dataclass(
            SelectionConfig, root["selection"], "selection"
        ),
        storage=strict_dataclass(StorageConfig, root["storage"], "storage"),
        packing=strict_dataclass(PackingConfig, root["packing"], "packing"),
        stage=strict_dataclass(StageConfig, root["stage"], "stage"),
        reader=strict_dataclass(ReaderConfig, root["reader"], "reader"),
        packed_manifest_identity=(
            None
            if root.get("packed_manifest_identity") is None
            else strict_dataclass(
                PackedManifestIdentity,
                root["packed_manifest_identity"],
                "packed_manifest_identity",
            )
        ),
    )
    config.validate()
    return config


def save_data_pipeline_config(
    config: DataPipelineConfig, path: str | Path
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(asdict(config), handle, sort_keys=False)
