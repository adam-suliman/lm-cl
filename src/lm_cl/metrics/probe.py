from __future__ import annotations

import math
from collections import defaultdict
from typing import Any


def trapezoidal_auc(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("AUC requires at least two paired curve points")
    if any(not math.isfinite(value) for value in xs + ys):
        raise ValueError("AUC inputs must be finite")
    if any(right <= left for left, right in zip(xs, xs[1:])):
        raise ValueError("AUC x-coordinates must be strictly increasing")
    return sum(
        (right_x - left_x) * (left_y + right_y) / 2.0
        for left_x, right_x, left_y, right_y in zip(
            xs,
            xs[1:],
            ys,
            ys[1:],
        )
    )


def percent_change_from_first(
    current_auc: float,
    first_auc: float,
) -> float:
    if not math.isfinite(current_auc) or not math.isfinite(first_auc):
        raise ValueError("AUC values must be finite")
    if first_auc == 0:
        raise ValueError("First-probe AUC cannot be zero")
    return 100.0 * (current_auc / first_auc - 1.0)


def _one_curve(
    records: list[dict[str, Any]],
    *,
    early_milestones: list[int],
) -> dict[str, Any]:
    ordered = sorted(records, key=lambda item: item["probe_logical_step"])
    steps = [float(item["probe_logical_step"]) for item in ordered]
    tokens = [float(item["cumulative_input_tokens"]) for item in ordered]
    losses = [float(item["mean_validation_ce"]) for item in ordered]
    if len(set(steps)) != len(steps):
        raise ValueError("Curve contains a duplicate logical step")
    if len(set(tokens)) != len(tokens):
        raise ValueError("Curve contains a duplicate token coordinate")
    raw_step_auc = trapezoidal_auc(steps, losses)
    raw_token_auc = trapezoidal_auc(tokens, losses)
    token_span = tokens[-1] - tokens[0]
    if token_span <= 0:
        raise ValueError("Curve token span must be positive")
    normalized_tokens = [
        (value - tokens[0]) / token_span
        for value in tokens
    ]
    normalized_auc = trapezoidal_auc(normalized_tokens, losses)
    by_step = {
        int(item["probe_logical_step"]): float(item["mean_validation_ce"])
        for item in ordered
    }
    missing = sorted(set(early_milestones) - set(by_step))
    if missing:
        raise ValueError(
            f"Curve lacks configured early milestones: {missing}"
        )
    return {
        "point_count": len(ordered),
        "primary_normalized_trapezoidal_auc": normalized_auc,
        "arithmetic_mean_recorded_validation_ce": (
            sum(losses) / len(losses)
        ),
        "raw_step_trapezoidal_auc": raw_step_auc,
        "raw_input_token_trapezoidal_auc": raw_token_auc,
        "step_0_validation_ce": by_step[0],
        "final_validation_ce": losses[-1],
        "final_probe_logical_step": int(steps[-1]),
        "final_cumulative_input_tokens": int(tokens[-1]),
        "early_milestone_validation_ce": {
            str(step): by_step[step]
            for step in early_milestones
        },
    }


def compute_probe_auc_report(
    records: list[dict[str, Any]],
    *,
    early_milestones: list[int],
    policy: dict[str, Any],
) -> dict[str, Any]:
    if not records:
        raise ValueError("Probe curve is empty")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        mode = record.get("memory_evaluation_mode")
        if mode not in {"not_applicable", "reset", "carried"}:
            raise ValueError("Curve has an invalid memory evaluation mode")
        grouped[mode].append(record)
    results = {
        mode: _one_curve(
            curve,
            early_milestones=early_milestones,
        )
        for mode, curve in sorted(grouped.items())
    }
    return {
        "auc_schema_version": 1,
        "policy": policy,
        "curves": results,
    }


def pairing_identity(result: dict[str, Any]) -> dict[str, Any]:
    required = {
        "training_manifest_identity",
        "validation_manifest_identity",
        "source_boundary_class",
        "probe_seed",
        "evaluation_schedule",
        "token_budget",
    }
    values = result.get("pairing_identity")
    if not isinstance(values, dict) or set(values) != required:
        raise ValueError("Probe result pairing identity is incomplete")
    return values


def compare_probe_results(
    left: dict[str, Any],
    right: dict[str, Any],
) -> dict[str, Any]:
    left_pairing = pairing_identity(left)
    right_pairing = pairing_identity(right)
    mismatches = {
        key: {
            "left": left_pairing[key],
            "right": right_pairing[key],
        }
        for key in sorted(left_pairing)
        if left_pairing[key] != right_pairing[key]
    }
    if mismatches:
        raise ValueError(
            "Probe results are not paired: "
            + ", ".join(mismatches)
        )
    left_curves = left["auc_report"].get("curves")
    right_curves = right["auc_report"].get("curves")
    if not isinstance(left_curves, dict) or not isinstance(
        right_curves,
        dict,
    ):
        raise ValueError("Probe comparison lacks AUC curves")
    shared_modes = sorted(set(left_curves) & set(right_curves))
    percent_changes = {
        mode: percent_change_from_first(
            float(
                right_curves[mode][
                    "primary_normalized_trapezoidal_auc"
                ]
            ),
            float(
                left_curves[mode][
                    "primary_normalized_trapezoidal_auc"
                ]
            ),
        )
        for mode in shared_modes
    }
    return {
        "comparison_schema_version": 1,
        "paired": True,
        "pairing_identity": left_pairing,
        "right_vs_left_primary_auc_percent_change": percent_changes,
        "left": {
            "run_name": left["run_name"],
            "variant": left["variant"],
            "probe_mode": left["probe_mode"],
            "auc": left["auc_report"],
        },
        "right": {
            "run_name": right["run_name"],
            "variant": right["variant"],
            "probe_mode": right["probe_mode"],
            "auc": right["auc_report"],
        },
    }
