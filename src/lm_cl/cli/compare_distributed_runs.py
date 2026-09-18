from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

from lm_cl.cli._common import cli_entry, print_json, write_json
from lm_cl.training.checkpoint import load_checkpoint, sha256_file


def _tensor_items(value: Any, prefix: str = ""):
    if torch.is_tensor(value):
        yield prefix, value.detach().cpu()
    elif isinstance(value, dict):
        for key in sorted(value, key=lambda item: repr(item)):
            child = f"{prefix}.{key}" if prefix else str(key)
            yield from _tensor_items(value[key], child)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            child = f"{prefix}[{index}]"
            yield from _tensor_items(item, child)


def _compare_tensors(
    left: Any,
    right: Any,
    *,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    left_items = dict(_tensor_items(left))
    right_items = dict(_tensor_items(right))
    if set(left_items) != set(right_items):
        return {
            "passed": False,
            "reason": "tensor paths differ",
            "left_only": sorted(set(left_items) - set(right_items)),
            "right_only": sorted(set(right_items) - set(left_items)),
        }
    maximum_absolute = 0.0
    maximum_relative = 0.0
    maximum_absolute_path = None
    maximum_relative_path = None
    mismatch_paths: list[str] = []
    compared_elements = 0
    for path in sorted(left_items):
        left_tensor = left_items[path]
        right_tensor = right_items[path]
        if left_tensor.shape != right_tensor.shape:
            mismatch_paths.append(path)
            continue
        left_float = left_tensor.double()
        right_float = right_tensor.double()
        difference = (left_float - right_float).abs()
        compared_elements += difference.numel()
        if difference.numel():
            absolute = float(difference.max())
            denominator = torch.maximum(
                left_float.abs(), right_float.abs()
            ).clamp_min(torch.finfo(torch.float64).tiny)
            relative = float((difference / denominator).max())
            if absolute > maximum_absolute:
                maximum_absolute = absolute
                maximum_absolute_path = path
            if relative > maximum_relative:
                maximum_relative = relative
                maximum_relative_path = path
        if not torch.allclose(
            left_tensor,
            right_tensor,
            rtol=rtol,
            atol=atol,
            equal_nan=False,
        ):
            mismatch_paths.append(path)
    return {
        "passed": not mismatch_paths,
        "tensor_count": len(left_items),
        "compared_elements": compared_elements,
        "rtol": rtol,
        "atol": atol,
        "maximum_absolute_difference": maximum_absolute,
        "maximum_absolute_difference_path": maximum_absolute_path,
        "maximum_relative_difference": maximum_relative,
        "maximum_relative_difference_path": maximum_relative_path,
        "mismatch_paths": mismatch_paths,
    }


def _maximum_sequence_difference(
    left: list[float],
    right: list[float],
) -> float | None:
    if len(left) != len(right):
        return None
    return max(
        (abs(a - b) for a, b in zip(left, right)),
        default=0.0,
    )


def _compare_sequence(
    left: list[float],
    right: list[float],
    *,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    difference = _maximum_sequence_difference(left, right)
    return {
        "left_count": len(left),
        "right_count": len(right),
        "maximum_absolute_difference": difference,
        "passed": (
            difference is not None
            and all(
                math.isclose(a, b, rel_tol=rtol, abs_tol=atol)
                for a, b in zip(left, right)
            )
        ),
    }


def _metrics(payload: dict[str, Any]) -> list[dict[str, Any]]:
    runtime = payload["resolved_config"]["runtime"]
    metrics = Path(runtime["metrics_jsonl"]).expanduser()
    if not metrics.is_absolute():
        metrics = (
            Path(runtime["output_dir"]).expanduser() / metrics
        )
    if not metrics.is_file():
        return []
    return [
        json.loads(line)
        for line in metrics.read_text(encoding="utf-8").splitlines()
    ]


def command() -> None:
    parser = argparse.ArgumentParser(
        description="Compare scientific state from one- and multi-rank runs"
    )
    parser.add_argument("left_checkpoint")
    parser.add_argument("right_checkpoint")
    parser.add_argument("--rtol", type=float, required=True)
    parser.add_argument("--atol", type=float, required=True)
    parser.add_argument("--output-report")
    args = parser.parse_args()
    if (
        not math.isfinite(args.rtol)
        or not math.isfinite(args.atol)
        or args.rtol < 0
        or args.atol < 0
    ):
        raise ValueError("Tolerances must be finite and non-negative")
    left_path = Path(args.left_checkpoint).expanduser().resolve()
    right_path = Path(args.right_checkpoint).expanduser().resolve()
    left = load_checkpoint(left_path)
    right = load_checkpoint(right_path)
    model = _compare_tensors(
        left["model_state"],
        right["model_state"],
        rtol=args.rtol,
        atol=args.atol,
    )
    optimizer = _compare_tensors(
        left["optimizer_state"],
        right["optimizer_state"],
        rtol=args.rtol,
        atol=args.atol,
    )
    gradients = _compare_tensors(
        left["gradients"],
        right["gradients"],
        rtol=args.rtol,
        atol=args.atol,
    )
    initial_memory = _compare_tensors(
        left["memory_state"]["initial_memory"],
        right["memory_state"]["initial_memory"],
        rtol=args.rtol,
        atol=args.atol,
    )
    active_memory = _compare_tensors(
        left["memory_state"]["active_memory"],
        right["memory_state"]["active_memory"],
        rtol=args.rtol,
        atol=args.atol,
    )
    counters = {}
    for key in (
        "global_logical_batches",
        "global_slow_steps",
        "global_fast_updates",
        "global_input_tokens",
        "global_valid_targets",
        "source_position",
    ):
        left_value = left["trainer_state"][key]
        right_value = right["trainer_state"][key]
        counters[key] = {
            "left": left_value,
            "right": right_value,
            "equal": left_value == right_value,
        }
    histories = {}
    for key in (
        "loss_history",
        "fast_gradient_norm_history",
        "fast_clipped_gradient_norm_history",
        "fast_memory_norm_history",
    ):
        left_values = left["trainer_state"][key]
        right_values = right["trainer_state"][key]
        histories[key] = _compare_sequence(
            left_values,
            right_values,
            rtol=args.rtol,
            atol=args.atol,
        )
    left_logical = [
        item for item in _metrics(left) if item.get("event") == "logical_batch"
    ]
    right_logical = [
        item for item in _metrics(right) if item.get("event") == "logical_batch"
    ]
    left_optimizer = [
        item for item in _metrics(left) if item.get("event") == "optimizer_step"
    ]
    right_optimizer = [
        item for item in _metrics(right) if item.get("event") == "optimizer_step"
    ]
    metric_diagnostics = {
        "slow_gradient_norms": _compare_sequence(
            [float(item["gradient_norm"]) for item in left_optimizer],
            [float(item["gradient_norm"]) for item in right_optimizer],
            rtol=args.rtol,
            atol=args.atol,
        ),
        "fast_gradient_norms": _compare_sequence(
            [
                float(item["active_memory_gradient_norm_before_clip"])
                for item in left_logical
                if item.get("active_memory_gradient_norm_before_clip")
                is not None
            ],
            [
                float(item["active_memory_gradient_norm_before_clip"])
                for item in right_logical
                if item.get("active_memory_gradient_norm_before_clip")
                is not None
            ],
            rtol=args.rtol,
            atol=args.atol,
        ),
    }
    left_intervals = [
        item.get("source_global_sequence_range")
        for item in left_logical
        if "source_global_sequence_range" in item
    ]
    right_intervals = [
        item.get("source_global_sequence_range")
        for item in right_logical
        if "source_global_sequence_range" in item
    ]
    source_intervals_equal = (
        bool(left_intervals)
        and left_intervals == right_intervals
    )
    sections = (
        model,
        optimizer,
        gradients,
        initial_memory,
        active_memory,
    )
    passed = (
        all(section["passed"] for section in sections)
        and all(item["equal"] for item in counters.values())
        and all(item["passed"] for item in histories.values())
        and all(item["passed"] for item in metric_diagnostics.values())
        and (
            source_intervals_equal
            or not left_intervals
            or not right_intervals
        )
    )
    report = {
        "kind": "lm-cl-distributed-equivalence-report",
        "schema_version": 1,
        "passed": passed,
        "tolerances": {"rtol": args.rtol, "atol": args.atol},
        "left": {
            "path": str(left_path),
            "sha256": sha256_file(left_path),
            "world_size": left["distributed_state"]["world_size"],
        },
        "right": {
            "path": str(right_path),
            "sha256": sha256_file(right_path),
            "world_size": right["distributed_state"]["world_size"],
        },
        "model": model,
        "optimizer": optimizer,
        "partial_slow_gradients": gradients,
        "initial_memory": initial_memory,
        "active_memory": active_memory,
        "counters": counters,
        "histories": histories,
        "metric_diagnostics": metric_diagnostics,
        "source_intervals": {
            "left": left_intervals,
            "right": right_intervals,
            "equal": source_intervals_equal,
        },
    }
    if args.output_report:
        write_json(args.output_report, report)
    print_json(report)
    if not passed:
        raise RuntimeError("Distributed equivalence comparison failed")


def main() -> None:
    cli_entry(command)


if __name__ == "__main__":
    main()
