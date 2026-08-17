from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from lm_cl.training.checkpoint import canonical_sha256, sha256_file


LANGUAGE_COUNT = 8
EXPECTED_VARIANTS = {
    "transformer": "backbone_clean",
    "backbone_matched_k": "backbone_matched_k",
    "fastmem_rmt_zero": "fastmem_rmt_zero",
    "fastmem_rmt": "fastmem_rmt",
}
VARIANT_K = {
    "backbone_clean": 1,
    "backbone_matched_k": 2,
    "fastmem_rmt_zero": 2,
    "fastmem_rmt": 2,
}
PERSISTENT_FAST_VARIANTS = {"fastmem_rmt_zero", "fastmem_rmt"}


def _read_summary(path: str | Path) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Summary not found: {resolved}")
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Summary is not an object: {resolved}")
    return resolved, value


def _read_resolved_experiment(summary_path: Path) -> dict[str, Any]:
    path = summary_path.with_name("resolved_experiment.yaml")
    if not path.is_file():
        raise FileNotFoundError(
            f"Resolved experiment is required beside the summary: {path}"
        )
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Resolved experiment is not an object: {path}")
    return value


def _portable_identity(value: Any) -> Any:
    """Remove machine-local paths while retaining scientific identities."""

    if isinstance(value, list):
        return [_portable_identity(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {}
    for key, item in value.items():
        lowered = str(key).lower()
        if (
            lowered == "path"
            or lowered.endswith("_path")
            or lowered.endswith("_root")
            or lowered in {"cache_root", "generated_root", "manifest_root"}
        ):
            continue
        result[str(key)] = _portable_identity(item)
    return result


def _causal_identity(resolved: dict[str, Any]) -> dict[str, Any]:
    identity = resolved.get("scientific_identity")
    if not isinstance(identity, dict):
        raise ValueError("Resolved experiment lacks scientific_identity")
    result = {
        key: value
        for key, value in identity.items()
        if key not in {"public_model", "internal_variant"}
    }
    training = result.get("training")
    if isinstance(training, dict):
        training = dict(training)
        # Physical microbatching is an execution partition. The trainer's
        # global target normalization is invariant to it, but we still report
        # differences as an execution warning below.
        training.pop("physical_microbatch_sequences", None)
        result["training"] = training
    return _portable_identity(result)


def _execution_identity(resolved: dict[str, Any]) -> dict[str, Any]:
    launcher_config = resolved.get("launcher_config", {})
    training = launcher_config.get("training", {})
    launcher = launcher_config.get("launcher", {})
    return {
        "physical_microbatch_sequences": training.get(
            "physical_microbatch_sequences"
        ),
        "precision": training.get("precision"),
        "gpus_per_job": launcher.get("gpus_per_job"),
    }


def _gpu_signature(summary: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "name": item.get("name"),
            "compute_capability": item.get("compute_capability"),
            "total_memory_bytes": item.get("total_memory_bytes"),
        }
        for item in summary.get("gpu_identity", [])
    ]


def _completion_evidence(
    name: str, summary: dict[str, Any]
) -> dict[str, Any]:
    expected_internal = EXPECTED_VARIANTS[name]
    if summary.get("status") != "complete":
        raise ValueError(f"{name} summary is not complete")
    if summary.get("model") != name:
        raise ValueError(f"{name} summary has another public model")
    if summary.get("internal_variant") != expected_internal:
        raise ValueError(f"{name} summary has another internal variant")

    cycles = summary.get("cycles_completed")
    if not isinstance(cycles, int) or cycles <= 0:
        raise ValueError(f"{name} has an invalid completed cycle count")
    expected_tasks = cycles * LANGUAGE_COUNT
    task_count = summary.get("completed_language_tasks")
    if task_count != expected_tasks:
        raise ValueError(
            f"{name} completed tasks {task_count} != {expected_tasks}"
        )
    total_logical = summary.get("total_logical_batches")
    if not isinstance(total_logical, int) or total_logical <= 0:
        raise ValueError(f"{name} has invalid logical-batch count")
    logical_per_task, remainder = divmod(total_logical, expected_tasks)
    if remainder:
        raise ValueError(f"{name} logical batches are not task-divisible")
    k = VARIANT_K[expected_internal]
    expected_slow = (
        (logical_per_task + k - 1) // k
    ) * expected_tasks
    expected_fast = (
        total_logical if expected_internal in PERSISTENT_FAST_VARIANTS else 0
    )
    checks = {
        "total_slow_updates": expected_slow,
        "total_fast_updates": expected_fast,
    }
    for field, expected in checks.items():
        if summary.get(field) != expected:
            raise ValueError(
                f"{name} {field} {summary.get(field)} != {expected}"
            )
    probes = summary.get("per_cycle_probe_auc")
    if not isinstance(probes, list) or len(probes) != cycles:
        raise ValueError(f"{name} does not contain one probe per cycle")
    forgetting = summary.get("final_forgetting")
    if (
        not isinstance(forgetting, dict)
        or forgetting.get("evaluation_count") != expected_tasks
    ):
        raise ValueError(
            f"{name} does not contain one forgetting evaluation per task"
        )

    gate = summary.get("completion_invariants")
    if gate is None:
        gate_status = "legacy_summary_reconstructed"
    elif not isinstance(gate, dict) or gate.get("status") != "valid":
        raise ValueError(f"{name} has invalid completion invariants")
    else:
        actual = gate.get("actual", {})
        gate_pairs = {
            "completed_language_tasks": task_count,
            "total_logical_batches": total_logical,
            "total_input_tokens": summary.get("total_input_tokens"),
            "total_target_tokens": summary.get("total_target_tokens"),
            "total_slow_updates": summary.get("total_slow_updates"),
            "total_fast_updates": summary.get("total_fast_updates"),
            "cycle_end_probes": len(probes),
            "forgetting_evaluations": forgetting["evaluation_count"],
        }
        for field, expected in gate_pairs.items():
            if actual.get(field) != expected:
                raise ValueError(
                    f"{name} completion gate disagrees on {field}"
                )
        gate_status = "embedded_gate_validated"
    return {
        "status": gate_status,
        "cycles": cycles,
        "completed_language_tasks": task_count,
        "logical_batches_per_task": logical_per_task,
        "expected_slow_updates": expected_slow,
        "expected_fast_updates": expected_fast,
        "cycle_end_probes": len(probes),
        "forgetting_evaluations": forgetting["evaluation_count"],
    }


def _curve_metric(
    probe: dict[str, Any], mode: str, field: str
) -> float | None:
    curve = probe.get("auc_report", {}).get("curves", {}).get(mode)
    if not isinstance(curve, dict):
        return None
    value = curve.get(field)
    return None if value is None else float(value)


def _metric_row(summary: dict[str, Any]) -> dict[str, Any]:
    forgetting = summary["final_forgetting"]
    probes = summary["per_cycle_probe_auc"]
    modes = ("not_applicable", "reset", "carried")
    return {
        "seed": summary["seed"],
        "cycles": summary["cycles_completed"],
        "total_logical_batches": summary["total_logical_batches"],
        "total_slow_updates": summary["total_slow_updates"],
        "total_fast_updates": summary["total_fast_updates"],
        "final_average_forgetting_ce": forgetting[
            "average_forgetting_from_best_ce"
        ],
        "final_prior_language_forgetting_ce": forgetting[
            "average_prior_language_forgetting_from_best_ce"
        ],
        "probe_primary_modes": [
            item["primary_memory_evaluation_mode"] for item in probes
        ],
        "probe_primary_normalized_auc_by_cycle": [
            item["normalized_token_auc"] for item in probes
        ],
        "probe_primary_final_ce_by_cycle": [
            item["final_validation_ce"] for item in probes
        ],
        "probe_normalized_auc_by_mode": {
            mode: [
                _curve_metric(
                    item, mode, "primary_normalized_trapezoidal_auc"
                )
                for item in probes
            ]
            for mode in modes
            if any(
                mode in item.get("auc_report", {}).get("curves", {})
                for item in probes
            )
        },
        "probe_final_ce_by_mode": {
            mode: [
                _curve_metric(item, mode, "final_validation_ce")
                for item in probes
            ]
            for mode in modes
            if any(
                mode in item.get("auc_report", {}).get("curves", {})
                for item in probes
            )
        },
        "reset_carried_max_abs_ce_difference_by_cycle": [
            item.get("reset_carried_max_abs_ce_difference")
            for item in probes
        ],
    }


def _delta(right: dict[str, Any], left: dict[str, Any]) -> dict[str, Any]:
    result = {
        field: float(right[field]) - float(left[field])
        for field in (
            "final_average_forgetting_ce",
            "final_prior_language_forgetting_ce",
        )
    }
    for field in (
        "probe_primary_normalized_auc_by_cycle",
        "probe_primary_final_ce_by_cycle",
    ):
        if len(right[field]) != len(left[field]):
            raise ValueError(f"Cannot contrast unequal {field} lengths")
        result[field] = [
            float(right_item) - float(left_item)
            for right_item, left_item in zip(right[field], left[field])
        ]
    return result


def build_control_comparison(
    *,
    transformer: str | Path,
    backbone_matched_k: str | Path,
    fastmem_rmt_zero: str | Path,
    fastmem_rmt: str | Path,
) -> dict[str, Any]:
    paths_and_summaries = {
        name: _read_summary(path)
        for name, path in {
            "transformer": transformer,
            "backbone_matched_k": backbone_matched_k,
            "fastmem_rmt_zero": fastmem_rmt_zero,
            "fastmem_rmt": fastmem_rmt,
        }.items()
    }
    summaries = {
        name: value for name, (_, value) in paths_and_summaries.items()
    }
    completion = {
        name: _completion_evidence(name, summary)
        for name, summary in summaries.items()
    }
    seeds = {summary.get("seed") for summary in summaries.values()}
    cycles = {summary.get("cycles_completed") for summary in summaries.values()}
    if len(seeds) != 1 or len(cycles) != 1:
        raise ValueError("Control summaries do not share seed and horizon")

    count_fields = (
        "completed_language_tasks",
        "total_input_tokens",
        "total_target_tokens",
        "total_logical_batches",
    )
    for field in count_fields:
        values = {summary.get(field) for summary in summaries.values()}
        if len(values) != 1:
            raise ValueError(f"Control summaries disagree on {field}: {values}")

    portable_data = {
        name: _portable_identity(summary.get("data_manifest_identities"))
        for name, summary in summaries.items()
    }
    portable_hashes = {
        name: canonical_sha256(value)
        for name, value in portable_data.items()
    }
    if len(set(portable_hashes.values())) != 1:
        raise ValueError(
            "Control summaries do not share portable data identities: "
            f"{portable_hashes}"
        )

    resolved = {
        name: _read_resolved_experiment(path)
        for name, (path, _) in paths_and_summaries.items()
    }
    for name, value in resolved.items():
        if value.get("public_model") != name:
            raise ValueError(f"{name} resolved experiment has another model")
        if value.get("internal_variant") != EXPECTED_VARIANTS[name]:
            raise ValueError(
                f"{name} resolved experiment has another internal variant"
            )
        if value.get("resolved_experiment_sha256") != summaries[name].get(
            "resolved_experiment_sha256"
        ):
            raise ValueError(f"{name} summary/resolved identity mismatch")
    causal_hashes = {
        name: canonical_sha256(_causal_identity(value))
        for name, value in resolved.items()
    }
    if len(set(causal_hashes.values())) != 1:
        raise ValueError(
            "Control arms differ in non-variant scientific settings: "
            f"{causal_hashes}"
        )

    metrics = {
        name: _metric_row(summary) for name, summary in summaries.items()
    }
    gpu_signatures = {
        name: _gpu_signature(summary) for name, summary in summaries.items()
    }
    execution_identities = {
        name: _execution_identity(value) for name, value in resolved.items()
    }
    source_trees = {
        name: summary.get("source_tree_identity", {})
        for name, summary in summaries.items()
    }
    warnings = []
    if len({canonical_sha256(value) for value in gpu_signatures.values()}) != 1:
        warnings.append(
            "GPU hardware or world size differs across arms; rerun all four "
            "arms on one hardware layout for the strictest ablation."
        )
    if len(
        {canonical_sha256(value) for value in execution_identities.values()}
    ) != 1:
        warnings.append(
            "Physical microbatch or GPU-group execution settings differ; "
            "global update semantics match, but numerical trajectories may differ."
        )
    if len(
        {canonical_sha256(value) for value in source_trees.values()}
    ) != 1:
        warnings.append(
            "Source-tree identities differ across arms; inspect commits, dirty "
            "flags, and tree hashes before causal attribution."
        )
    if any(
        item["status"] == "legacy_summary_reconstructed"
        for item in completion.values()
    ):
        warnings.append(
            "Legacy summaries predate the embedded completion gate; their "
            "task, update, probe, and forgetting counts were reconstructed."
        )
    warnings.append(
        "A single seed is descriptive evidence; run paired seeds 81011 and "
        "81012 before reporting uncertainty or a general mechanism claim."
    )

    return {
        "control_comparison_schema_version": 1,
        "status": "valid",
        "seed": next(iter(seeds)),
        "cycles": next(iter(cycles)),
        "portable_data_identity_sha256": next(iter(portable_hashes.values())),
        "causal_settings_sha256": next(iter(causal_hashes.values())),
        "completion_evidence": completion,
        "sources": {
            name: {
                "summary_path": str(path),
                "summary_sha256": sha256_file(path),
                "resolved_experiment_path": str(
                    path.with_name("resolved_experiment.yaml")
                ),
                "source_tree_identity": source_trees[name],
                "gpu_signature": gpu_signatures[name],
                "execution_identity": execution_identities[name],
            }
            for name, (path, _) in paths_and_summaries.items()
        },
        "metrics": metrics,
        "contrasts": {
            "backbone_matched_k_minus_transformer": {
                "interpretation": (
                    "K=2 slow-update-cadence effect on the no-memory backbone; "
                    "negative CE/AUC deltas are better"
                ),
                "delta": _delta(
                    metrics["backbone_matched_k"], metrics["transformer"]
                ),
            },
            "fastmem_rmt_zero_minus_backbone_matched_k": {
                "interpretation": (
                    "RMT memory-token architecture and persistent-state path at "
                    "fast_lr=0; negative CE/AUC deltas are better"
                ),
                "delta": _delta(
                    metrics["fastmem_rmt_zero"],
                    metrics["backbone_matched_k"],
                ),
            },
            "fastmem_rmt_minus_fastmem_rmt_zero": {
                "interpretation": (
                    "positive explicit fast-update contribution on the same "
                    "FastMem code path; negative CE/AUC deltas are better"
                ),
                "delta": _delta(
                    metrics["fastmem_rmt"], metrics["fastmem_rmt_zero"]
                ),
            },
        },
        "warnings": warnings,
    }
