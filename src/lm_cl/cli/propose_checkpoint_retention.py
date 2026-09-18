from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lm_cl.cli._common import cli_entry, print_json
from lm_cl.training.checkpoint import load_checkpoint, sha256_file
from lm_cl.training.probe_checkpoint import atomic_write_json


ACTION = "delete_file_only_after_explicit_approval"


def _allocated_bytes(path: Path) -> int:
    return int(path.stat().st_blocks * 512)


def _summary(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    state = payload["trainer_state"]
    memory = payload["memory_state"]
    distributed = payload["distributed_state"]
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "allocated_bytes": _allocated_bytes(path),
        "sha256": sha256_file(path),
        "checkpoint_kind": payload["checkpoint_kind"],
        "checkpoint_schema_version": payload["checkpoint_schema_version"],
        "config_sha256": payload["config_sha256"],
        "phase": state["phase"],
        "next_task_index": state["next_task_index"],
        "current_task_index": state["current_task_index"],
        "cycle_index": state["cycle_index"],
        "language": state["language"],
        "global_logical_batches": state["global_logical_batches"],
        "global_slow_steps": state["global_slow_steps"],
        "global_fast_updates": state["global_fast_updates"],
        "global_input_tokens": state["global_input_tokens"],
        "global_valid_targets": state["global_valid_targets"],
        "source_position": state["source_position"],
        "window_logical_batches": state["window_logical_batches"],
        "window_valid_targets": state["window_valid_targets"],
        "source_identity": payload["source_identity"],
        "memory": {
            "variant": memory["variant"],
            "fast_update_phase": memory["fast_update_phase"],
            "has_active_memory": memory["active_memory"] is not None,
            "memory_token_count": memory["memory_token_count"],
        },
        "distributed": {
            "enabled": distributed["enabled"],
            "backend": distributed["backend"],
            "world_size": distributed["world_size"],
            "partition_rule": distributed["partition_rule"],
            "writer_rank": 0,
        },
    }


def _validate_boundary(summary: dict[str, Any], *, label: str) -> None:
    if summary["phase"] != "task_boundary":
        raise ValueError(f"{label} checkpoint is not a task boundary")
    if summary["window_logical_batches"] or summary["window_valid_targets"]:
        raise ValueError(f"{label} checkpoint has a partial slow window")
    if summary["memory"]["fast_update_phase"] not in {
        "ready_for_next_logical_batch",
        "not_applicable",
    }:
        raise ValueError(f"{label} checkpoint has an unstable memory phase")


def build_proposal(
    replacement_path: str | Path,
    superseded_path: str | Path,
    *,
    free_space_floor_bytes: int,
) -> dict[str, Any]:
    replacement = Path(replacement_path).expanduser().resolve()
    superseded = Path(superseded_path).expanduser().resolve()
    if replacement == superseded:
        raise ValueError("Replacement and superseded checkpoints must differ")
    for path in (replacement, superseded):
        if path.suffix != ".pt" or not path.is_file():
            raise ValueError(f"Checkpoint must be an existing .pt file: {path}")
    replacement_summary = _summary(replacement, load_checkpoint(replacement))
    superseded_summary = _summary(superseded, load_checkpoint(superseded))
    _validate_boundary(replacement_summary, label="Replacement")
    _validate_boundary(superseded_summary, label="Superseded")
    if replacement_summary["config_sha256"] != superseded_summary["config_sha256"]:
        raise ValueError("Retention checkpoints have different configurations")
    if replacement_summary["next_task_index"] != (
        superseded_summary["next_task_index"] + 1
    ):
        raise ValueError("Replacement is not the immediately next task boundary")
    for counter in (
        "global_logical_batches",
        "global_slow_steps",
        "global_fast_updates",
        "global_input_tokens",
        "global_valid_targets",
    ):
        if replacement_summary[counter] < superseded_summary[counter]:
            raise ValueError(f"Replacement counter regressed: {counter}")
    filesystem = os.statvfs(replacement.parent)
    available = filesystem.f_bavail * filesystem.f_frsize
    atomic_copy_reserve = replacement_summary["allocated_bytes"]
    admission_required = free_space_floor_bytes + atomic_copy_reserve
    return {
        "schema_version": 1,
        "kind": "lm-cl-checkpoint-retention-proposal",
        "status": "proposal_only",
        "deletion_executed": False,
        "explicit_execution_approval_required": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "replacement_checkpoint": replacement_summary,
        "deletion_candidates": [
            {
                **superseded_summary,
                "action": ACTION,
                "evidence_preserved_by": "this_retention_proposal",
            }
        ],
        "verification": {
            "both_payloads_load_and_validate": True,
            "same_resolved_configuration": True,
            "replacement_is_immediately_next_boundary": True,
            "stable_boundary_states": True,
            "counter_monotonicity": True,
            "automatic_deletion_performed": False,
        },
        "storage_admission": {
            "available_bytes": available,
            "free_space_floor_bytes": free_space_floor_bytes,
            "atomic_temporary_checkpoint_reserve_bytes": atomic_copy_reserve,
            "required_available_bytes": admission_required,
            "passes": available >= admission_required,
        },
        "retained_evidence": [
            "resolved configuration and config SHA-256",
            "metrics JSONL",
            "compact checkpoint identity and counters in this proposal",
            "packed manifests and source identities",
        ],
    }


def command() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Construct a compact proposal for a superseded boundary checkpoint; "
            "this command never deletes files"
        )
    )
    parser.add_argument("replacement")
    parser.add_argument("superseded")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--free-space-floor-bytes",
        type=int,
        default=8 * 1024**3,
    )
    args = parser.parse_args()
    if args.free_space_floor_bytes <= 0:
        raise ValueError("Free-space floor must be positive")
    proposal = build_proposal(
        args.replacement,
        args.superseded,
        free_space_floor_bytes=args.free_space_floor_bytes,
    )
    output = Path(args.output).expanduser().resolve()
    atomic_write_json(output, proposal)
    print_json({"status": "proposal_only", "output": str(output), **proposal})


def main() -> None:
    cli_entry(command)


if __name__ == "__main__":
    main()
