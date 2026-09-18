from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from typing import Any

from lm_cl.cli._common import cli_entry, print_json, write_json
from lm_cl.config import DataPipelineConfig, load_data_pipeline_config
from lm_cl.data import load_packed_manifest, validate_packed_shards
from lm_cl.training.checkpoint import sha256_file


def _stage_dir(config: DataPipelineConfig) -> Path:
    return (
        Path(config.storage.generated_root).expanduser().resolve()
        / "stages"
        / config.stage.stage_id
    )


def _validate_role(
    config: DataPipelineConfig,
    *,
    purpose: str,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    config.require_access_ready()
    if config.mode != "packed_shards":
        raise ValueError("Probe data-pair validation requires packed_shards")
    if config.stage.language != "vi" or config.stage.purpose != purpose:
        raise ValueError(f"Probe data role must be vi/{purpose}")
    stage_dir = _stage_dir(config)
    report = validate_packed_shards(stage_dir)
    _, manifest = load_packed_manifest(stage_dir)
    if report["token_count"] != config.selection.max_output_tokens:
        raise ValueError("Probe stage does not contain its exact token budget")
    if manifest["stage"]["purpose"] != purpose:
        raise ValueError("Probe stage manifest purpose differs")
    return stage_dir, manifest, report


def validate_probe_data_pair(
    train_config_path: str | Path,
    validation_config_path: str | Path,
) -> dict[str, Any]:
    train_config = load_data_pipeline_config(train_config_path)
    validation_config = load_data_pipeline_config(validation_config_path)
    train_dir, train_manifest, train_report = _validate_role(
        train_config,
        purpose="vietnamese_train",
    )
    validation_dir, validation_manifest, validation_report = _validate_role(
        validation_config,
        purpose="vietnamese_validation",
    )
    if train_config.storage.generated_root != (
        validation_config.storage.generated_root
    ):
        raise ValueError("Probe train/validation generated roots differ")
    identity_fields = (
        ("dataset", "repo_id"),
        ("dataset", "revision"),
        ("dataset", "configuration"),
        ("dataset", "split"),
        ("dataset", "text_field"),
        ("dataset", "id_field"),
        ("dataset", "missing_id_policy"),
        ("dataset", "source_id_policy"),
        ("tokenizer", "repo_id"),
        ("tokenizer", "revision"),
        ("tokenizer", "manifest_content_sha256"),
        ("tokenizer", "model_embedding_vocab_size"),
        ("reader", "sequence_length"),
    )
    mismatches = []
    for section, field in identity_fields:
        if train_manifest[section][field] != validation_manifest[section][field]:
            mismatches.append(f"{section}.{field}")
    if train_manifest["selection"]["split_seed"] != (
        validation_manifest["selection"]["split_seed"]
    ):
        mismatches.append("selection.split_seed")
    if train_manifest["selection"]["validation_permyriad"] != (
        validation_manifest["selection"]["validation_permyriad"]
    ):
        mismatches.append("selection.validation_permyriad")
    if mismatches:
        raise ValueError(
            "Probe train/validation identity mismatch: "
            + ", ".join(mismatches)
        )
    registry_path = (
        Path(train_config.storage.generated_root).expanduser().resolve()
        / "overlap.sqlite3"
    )
    if not registry_path.is_file():
        raise ValueError("Global overlap registry is missing")
    uri = f"file:{registry_path}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        stage_counts = dict(
            connection.execute(
                "SELECT stage_id, COUNT(*) FROM documents "
                "WHERE stage_id IN (?, ?) GROUP BY stage_id",
                (
                    train_config.stage.stage_id,
                    validation_config.stage.stage_id,
                ),
            )
        )
        content_overlap = int(
            connection.execute(
                "SELECT COUNT(*) FROM documents a JOIN documents b "
                "ON a.content_sha256=b.content_sha256 "
                "WHERE a.stage_id=? AND b.stage_id=?",
                (
                    train_config.stage.stage_id,
                    validation_config.stage.stage_id,
                ),
            ).fetchone()[0]
        )
        token_overlap = int(
            connection.execute(
                "SELECT COUNT(*) FROM documents a JOIN documents b "
                "ON a.token_ids_sha256=b.token_ids_sha256 "
                "WHERE a.stage_id=? AND b.stage_id=?",
                (
                    train_config.stage.stage_id,
                    validation_config.stage.stage_id,
                ),
            ).fetchone()[0]
        )
        purpose_rows = list(
            connection.execute(
                "SELECT stage_id, purpose, COUNT(*) FROM documents "
                "WHERE stage_id IN (?, ?) "
                "GROUP BY stage_id, purpose ORDER BY stage_id, purpose",
                (
                    train_config.stage.stage_id,
                    validation_config.stage.stage_id,
                ),
            )
        )
    expected_counts = {
        train_config.stage.stage_id: train_manifest[
            "accepted_document_count"
        ],
        validation_config.stage.stage_id: validation_manifest[
            "accepted_document_count"
        ],
    }
    if stage_counts != expected_counts:
        raise ValueError("Probe overlap-registry stage counts differ")
    expected_purpose_rows = sorted(
        (
            (
                train_config.stage.stage_id,
                "vietnamese_train",
                expected_counts[train_config.stage.stage_id],
            ),
            (
                validation_config.stage.stage_id,
                "vietnamese_validation",
                expected_counts[validation_config.stage.stage_id],
            ),
        )
    )
    if purpose_rows != expected_purpose_rows:
        raise ValueError("Probe registry purposes are invalid")
    if content_overlap or token_overlap:
        raise ValueError("Probe train/validation documents overlap")
    return {
        "probe_data_pair_schema_version": 1,
        "status": "valid",
        "train": {
            "stage_dir": str(train_dir),
            "manifest_file_sha256": sha256_file(
                train_dir / "manifest.json"
            ),
            "manifest_content_sha256": train_report[
                "manifest_content_sha256"
            ],
            "ordered_data_sha256": train_report["ordered_data_sha256"],
            "token_count": train_report["token_count"],
            "target_token_count": train_report["target_token_count"],
            "document_count": train_report["document_count"],
        },
        "validation": {
            "stage_dir": str(validation_dir),
            "manifest_file_sha256": sha256_file(
                validation_dir / "manifest.json"
            ),
            "manifest_content_sha256": validation_report[
                "manifest_content_sha256"
            ],
            "ordered_data_sha256": validation_report[
                "ordered_data_sha256"
            ],
            "token_count": validation_report["token_count"],
            "target_token_count": validation_report["target_token_count"],
            "document_count": validation_report["document_count"],
        },
        "registry": {
            "path": str(registry_path),
            "stage_document_counts": stage_counts,
            "content_overlap": content_overlap,
            "token_overlap": token_overlap,
            "purpose_rows": [
                {
                    "stage_id": stage_id,
                    "purpose": purpose,
                    "document_count": document_count,
                }
                for stage_id, purpose, document_count in purpose_rows
            ],
        },
    }


def command() -> None:
    parser = argparse.ArgumentParser(
        description="Validate fixed, non-overlapping Vietnamese probe data"
    )
    parser.add_argument("train_config")
    parser.add_argument("validation_config")
    parser.add_argument("--output-report")
    args = parser.parse_args()
    report = validate_probe_data_pair(
        args.train_config,
        args.validation_config,
    )
    if args.output_report:
        write_json(args.output_report, report)
    print_json(report)


def main() -> None:
    cli_entry(command)


if __name__ == "__main__":
    main()
