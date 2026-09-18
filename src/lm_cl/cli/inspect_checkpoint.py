from __future__ import annotations

import argparse
from pathlib import Path

from lm_cl.cli._common import cli_entry, print_json, write_json
from lm_cl.training.checkpoint import load_checkpoint, sha256_file


def command() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect a Phase 4/5 continual checkpoint"
    )
    parser.add_argument("checkpoint")
    parser.add_argument("--output-report")
    args = parser.parse_args()
    path = Path(args.checkpoint).expanduser().resolve()
    payload = load_checkpoint(path)
    model_state = payload["model_state"]
    distributed_state = payload.get("distributed_state")
    distributed_summary = None
    if isinstance(distributed_state, dict):
        rank_rng_states = distributed_state.get("rank_rng_states")
        distributed_summary = {
            key: value
            for key, value in distributed_state.items()
            if key != "rank_rng_states"
        }
        distributed_summary["rank_rng_state_count"] = (
            len(rank_rng_states)
            if isinstance(rank_rng_states, list)
            else None
        )
        distributed_summary["rank_rng_state_ranks"] = (
            [
                item.get("rank")
                for item in rank_rng_states
                if isinstance(item, dict)
            ]
            if isinstance(rank_rng_states, list)
            else None
        )
    report = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "checkpoint_schema_version": payload[
                "checkpoint_schema_version"
            ],
            "checkpoint_kind": payload["checkpoint_kind"],
            "config_sha256": payload["config_sha256"],
            "model_tensor_count": len(model_state),
            "model_state_numel_with_aliases": sum(
                tensor.numel() for tensor in model_state.values()
            ),
            "configured_model_parameter_count": payload["resolved_config"][
                "model"
            ]["expected_total_parameters"],
            "initial_memory_parameter_count": (
                0
                if payload["memory_state"]["initial_memory"] is None
                else payload["memory_state"]["initial_memory"].numel()
            ),
            "active_memory_state_elements": (
                0
                if payload["memory_state"]["active_memory"] is None
                else payload["memory_state"]["active_memory"].numel()
            ),
            "optimizer_present": payload["optimizer_state"] is not None,
            "scheduler": payload["scheduler_state"],
            "trainer_state": payload["trainer_state"],
            "memory_state": {
                key: (
                    None
                    if value is None
                    else {
                        "shape": list(value.shape),
                        "dtype": str(value.dtype),
                    }
                    if hasattr(value, "shape")
                    else value
                )
                for key, value in payload["memory_state"].items()
            },
            "source_identity": payload["source_identity"],
            "provenance": payload["provenance"],
            "distributed_state": distributed_summary,
        }
    if args.output_report:
        write_json(args.output_report, report)
    print_json(report)


def main() -> None:
    cli_entry(command)


if __name__ == "__main__":
    main()
