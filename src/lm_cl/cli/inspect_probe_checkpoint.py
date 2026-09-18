from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from lm_cl.cli._common import cli_entry, print_json, write_json
from lm_cl.training.checkpoint import sha256_file
from lm_cl.training.probe_checkpoint import load_probe_checkpoint


def _tensor_summary(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "shape"):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    return value


def command() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect a Phase 7 probe checkpoint"
    )
    parser.add_argument("checkpoint")
    parser.add_argument("--output-report")
    args = parser.parse_args()
    path = Path(args.checkpoint).expanduser().resolve()
    payload = load_probe_checkpoint(path)
    distributed = dict(payload["distributed_state"])
    rank_rng = distributed.pop("rank_rng_states")
    distributed["rank_rng_state_count"] = len(rank_rng)
    distributed["rank_rng_state_ranks"] = [
        item["rank"] for item in rank_rng
    ]
    report = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "checkpoint_kind": payload["checkpoint_kind"],
        "checkpoint_schema_version": payload[
            "checkpoint_schema_version"
        ],
        "config_sha256": payload["config_sha256"],
        "source_checkpoint": payload["source_checkpoint"],
        "initialization_policy": payload["initialization_policy"],
        "probe_mode": payload["resolved_config"]["probe_mode"],
        "variant": payload["resolved_config"]["variant"]["name"],
        "probe_state": payload["probe_state"],
        "completed_evaluation_steps": payload[
            "completed_evaluation_steps"
        ],
        "curve_record_count": len(payload["curve_records"]),
        "training_source_identity": payload[
            "training_source_identity"
        ],
        "validation_source_identity": payload[
            "validation_source_identity"
        ],
        "memory_state": {
            key: _tensor_summary(value)
            for key, value in payload["memory_state"].items()
        },
        "scheduler": payload["scheduler_state"],
        "optimizer_present": payload["optimizer_state"] is not None,
        "distributed_state": distributed,
        "auc_policy": payload["auc_policy"],
        "provenance": payload["provenance"],
    }
    if args.output_report:
        write_json(args.output_report, report)
    print_json(report)


def main() -> None:
    cli_entry(command)


if __name__ == "__main__":
    main()
