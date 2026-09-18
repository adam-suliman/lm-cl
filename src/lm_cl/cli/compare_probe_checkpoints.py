from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

from lm_cl.cli._common import cli_entry, print_json, write_json
from lm_cl.cli.compare_distributed_runs import _compare_tensors
from lm_cl.training.checkpoint import sha256_file
from lm_cl.training.probe_checkpoint import load_probe_checkpoint


def _curve_comparison(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    *,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    if len(left) != len(right):
        return {
            "passed": False,
            "reason": "curve record counts differ",
            "left_count": len(left),
            "right_count": len(right),
        }
    exact_fields = (
        "probe_logical_step",
        "cumulative_input_tokens",
        "cumulative_valid_target_tokens",
        "validation_valid_target_count",
        "validation_input_token_count",
        "probe_mode",
        "memory_evaluation_mode",
        "variant",
        "source_checkpoint_sha256",
        "training_manifest_identity",
        "validation_manifest_identity",
    )
    float_fields = (
        "validation_loss_sum",
        "mean_validation_ce",
        "model_parameter_norm",
        "m0_norm",
        "active_memory_norm",
    )
    mismatches: list[str] = []
    maximum_absolute = 0.0
    for index, (left_record, right_record) in enumerate(zip(left, right)):
        for field in exact_fields:
            if left_record.get(field) != right_record.get(field):
                mismatches.append(f"record[{index}].{field}")
        for field in float_fields:
            left_value = left_record.get(field)
            right_value = right_record.get(field)
            if left_value is None or right_value is None:
                if left_value != right_value:
                    mismatches.append(f"record[{index}].{field}")
                continue
            difference = abs(float(left_value) - float(right_value))
            maximum_absolute = max(maximum_absolute, difference)
            if not math.isclose(
                float(left_value),
                float(right_value),
                rel_tol=rtol,
                abs_tol=atol,
            ):
                mismatches.append(f"record[{index}].{field}")
    return {
        "passed": not mismatches,
        "record_count": len(left),
        "rtol": rtol,
        "atol": atol,
        "maximum_absolute_difference": maximum_absolute,
        "mismatch_fields": mismatches,
    }


def compare_probe_checkpoints(
    left_path: str | Path,
    right_path: str | Path,
    *,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    if (
        not math.isfinite(rtol)
        or not math.isfinite(atol)
        or rtol < 0
        or atol < 0
    ):
        raise ValueError("Tolerances must be finite and non-negative")
    left_resolved = Path(left_path).expanduser().resolve()
    right_resolved = Path(right_path).expanduser().resolve()
    left = load_probe_checkpoint(left_resolved)
    right = load_probe_checkpoint(right_resolved)
    identity_fields = (
        "source_checkpoint",
        "training_source_identity",
        "validation_source_identity",
        "initialization_policy",
        "auc_policy",
    )
    identity_mismatches = [
        field
        for field in identity_fields
        if left[field] != right[field]
    ]
    for path in (
        ("variant",),
        ("probe_mode",),
        ("model",),
        ("optimization", "global_sequences_per_logical_batch"),
        ("train_logical_batches",),
        ("input_token_budget",),
        ("train_sequence_prefix_count",),
        ("source_checkpoint_status",),
        ("source_checkpoint_sha256",),
        ("validation_sequences",),
        ("evaluation_interval_logical_steps",),
        ("early_milestones",),
    ):
        left_value: Any = left["resolved_config"]
        right_value: Any = right["resolved_config"]
        for part in path:
            left_value = left_value[part]
            right_value = right_value[part]
        if left_value != right_value:
            identity_mismatches.append("resolved_config." + ".".join(path))
    exact_counter_fields = (
        "global_logical_batches",
        "global_slow_steps",
        "global_fast_updates",
        "global_input_tokens",
        "global_valid_targets",
        "source_position",
        "window_logical_batches",
        "window_valid_targets",
        "completed_evaluation_steps",
    )
    counters: dict[str, Any] = {}
    for field in exact_counter_fields:
        if field == "completed_evaluation_steps":
            left_value = left[field]
            right_value = right[field]
        else:
            left_value = left["probe_state"][field]
            right_value = right["probe_state"][field]
        counters[field] = {
            "left": left_value,
            "right": right_value,
            "passed": left_value == right_value,
        }
    comparisons = {
        "model": _compare_tensors(
            left["model_state"],
            right["model_state"],
            rtol=rtol,
            atol=atol,
        ),
        "optimizer": _compare_tensors(
            left["optimizer_state"],
            right["optimizer_state"],
            rtol=rtol,
            atol=atol,
        ),
        "partial_gradients": _compare_tensors(
            left["gradients"],
            right["gradients"],
            rtol=rtol,
            atol=atol,
        ),
        "m0": _compare_tensors(
            left["memory_state"]["initial_memory"],
            right["memory_state"]["initial_memory"],
            rtol=rtol,
            atol=atol,
        ),
        "active_memory": _compare_tensors(
            left["memory_state"]["active_memory"],
            right["memory_state"]["active_memory"],
            rtol=rtol,
            atol=atol,
        ),
        "curve": _curve_comparison(
            left["curve_records"],
            right["curve_records"],
            rtol=rtol,
            atol=atol,
        ),
    }
    passed = (
        not identity_mismatches
        and all(item["passed"] for item in counters.values())
        and all(item["passed"] for item in comparisons.values())
    )
    return {
        "probe_checkpoint_comparison_schema_version": 1,
        "passed": passed,
        "left": {
            "path": str(left_resolved),
            "sha256": sha256_file(left_resolved),
            "world_size": left["distributed_state"]["world_size"],
        },
        "right": {
            "path": str(right_resolved),
            "sha256": sha256_file(right_resolved),
            "world_size": right["distributed_state"]["world_size"],
        },
        "rtol": rtol,
        "atol": atol,
        "identity_mismatches": identity_mismatches,
        "counters": counters,
        "comparisons": comparisons,
    }


def command() -> None:
    parser = argparse.ArgumentParser(
        description="Compare ordinary/one-rank/multi-rank probe checkpoints"
    )
    parser.add_argument("left_checkpoint")
    parser.add_argument("right_checkpoint")
    parser.add_argument("--rtol", type=float, required=True)
    parser.add_argument("--atol", type=float, required=True)
    parser.add_argument("--output-report")
    args = parser.parse_args()
    report = compare_probe_checkpoints(
        args.left_checkpoint,
        args.right_checkpoint,
        rtol=args.rtol,
        atol=args.atol,
    )
    if args.output_report:
        write_json(args.output_report, report)
    print_json(report)
    if not report["passed"]:
        raise SystemExit(1)


def main() -> None:
    cli_entry(command)


if __name__ == "__main__":
    main()
