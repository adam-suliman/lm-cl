from __future__ import annotations

import argparse
import json
from pathlib import Path

from lm_cl.cli._common import cli_entry, print_json
from lm_cl.metrics.probe import pairing_identity


def load_probe_results(path: str | Path) -> dict:
    resolved = Path(path).expanduser().resolve()
    with resolved.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if (
        not isinstance(value, dict)
        or value.get("probe_results_schema_version") != 1
        or value.get("status") != "complete"
        or not isinstance(value.get("curve_records"), list)
        or not isinstance(value.get("auc_report"), dict)
    ):
        raise ValueError("Invalid or incomplete probe results")
    pairing_identity(value)
    return value


def command() -> None:
    parser = argparse.ArgumentParser(description="Inspect probe results")
    parser.add_argument("results")
    args = parser.parse_args()
    value = load_probe_results(args.results)
    print_json(
        {
            "path": str(Path(args.results).expanduser().resolve()),
            "run_name": value["run_name"],
            "variant": value["variant"],
            "probe_mode": value["probe_mode"],
            "source_checkpoint": value["source_checkpoint"],
            "training_source_identity": value[
                "training_source_identity"
            ],
            "validation_source_identity": value[
                "validation_source_identity"
            ],
            "pairing_identity": value["pairing_identity"],
            "curve_record_count": len(value["curve_records"]),
            "evaluation_steps": sorted(
                {
                    item["probe_logical_step"]
                    for item in value["curve_records"]
                }
            ),
            "auc_report": value["auc_report"],
            "probe_state": value["probe_state"],
            "source_hash_unchanged": (
                value["source_checkpoint_sha256_before"]
                == value["source_checkpoint_sha256_after"]
            ),
        }
    )


def main() -> None:
    cli_entry(command)


if __name__ == "__main__":
    main()
