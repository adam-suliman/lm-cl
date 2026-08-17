from __future__ import annotations

import json
from typing import Any

from lm_cl.launcher.schema import (
    MEMORY_INTERNAL_VARIANTS,
    PUBLIC_LANGUAGE_ORDER,
    LauncherConfig,
    resolve_token_budget,
)


def _variant_expectations(
    config: LauncherConfig, internal_variant: str
) -> dict[str, Any]:
    expectations = {
        "backbone_clean": {
            "memory_enabled": False,
            "persistent_fast_memory": False,
            "fast_lr": 0.0,
            "memory_tokens": 0,
            "slow_update_period_k": 1,
            "fast_memory_grad_clip_norm": None,
            "segment_length": None,
            "reset_policy": None,
            "memory_evaluation_policy": None,
        },
        "backbone_matched_k": {
            "memory_enabled": False,
            "persistent_fast_memory": False,
            "fast_lr": 0.0,
            "memory_tokens": 0,
            "slow_update_period_k": config.fastmem.slow_accumulation_k,
            "fast_memory_grad_clip_norm": None,
            "segment_length": None,
            "reset_policy": None,
            "memory_evaluation_policy": None,
        },
        "fastmem_rmt_zero": {
            "memory_enabled": True,
            "persistent_fast_memory": True,
            "fast_lr": 0.0,
            "memory_tokens": config.fastmem.memory_tokens,
            "slow_update_period_k": config.fastmem.slow_accumulation_k,
            "fast_memory_grad_clip_norm": config.fastmem.fast_clip,
            "segment_length": config.fastmem.segment_length,
            "reset_policy": "task_boundary_from_m0_stopgrad",
            "memory_evaluation_policy": "reset_and_carried",
        },
        "fastmem_rmt": {
            "memory_enabled": True,
            "persistent_fast_memory": True,
            "fast_lr": config.fastmem.fast_lr,
            "memory_tokens": config.fastmem.memory_tokens,
            "slow_update_period_k": config.fastmem.slow_accumulation_k,
            "fast_memory_grad_clip_norm": config.fastmem.fast_clip,
            "segment_length": config.fastmem.segment_length,
            "reset_policy": "task_boundary_from_m0_stopgrad",
            "memory_evaluation_policy": "reset_and_carried",
        },
    }
    if internal_variant not in expectations:
        raise ValueError(
            f"No completion contract exists for {internal_variant}"
        )
    return expectations[internal_variant]


def validate_completion_invariants(
    config: LauncherConfig,
    *,
    internal_variant: str,
    state: dict[str, Any],
    probe_summaries: list[dict[str, Any]],
    resolved_variant: dict[str, Any],
) -> dict[str, Any]:
    """Reject a complete-job summary unless its scientific counters agree.

    This gate is deliberately derived from the resolved budget and variant,
    rather than hard-coding the five-cycle production numbers.  The resulting
    report still records those numbers explicitly in every completed summary.
    """

    task_count = config.experiment.cycles * len(PUBLIC_LANGUAGE_ORDER)
    budget = resolve_token_budget(
        config.experiment.tokens_per_task,
        config.experiment.sequence_length,
        policy=config.experiment.token_budget_policy,
    )
    global_batch = config.training.global_batch_sequences
    logical_batches_per_task = (
        budget.effective_complete_sequences + global_batch - 1
    ) // global_batch
    variant = _variant_expectations(config, internal_variant)
    k = int(variant["slow_update_period_k"])
    expected = {
        "completed_language_tasks": task_count,
        "logical_batches_per_task": logical_batches_per_task,
        "total_logical_batches": logical_batches_per_task * task_count,
        "total_input_tokens": budget.effective_input_tokens * task_count,
        "total_target_tokens": budget.effective_valid_targets * task_count,
        "total_slow_updates": (
            (logical_batches_per_task + k - 1) // k
        )
        * task_count,
        "total_fast_updates": (
            logical_batches_per_task * task_count
            if variant["persistent_fast_memory"]
            else 0
        ),
        "memory_resets": (
            task_count if variant["memory_enabled"] else 0
        ),
        "cycle_end_probes": (
            config.experiment.cycles if config.probe.enabled else 0
        ),
        "forgetting_evaluations": (
            task_count
            if config.forgetting is not None
            and config.forgetting.enabled
            else 0
        ),
    }
    actual = {
        "completed_language_tasks": state.get("next_task_index"),
        "total_logical_batches": state.get("global_logical_batches"),
        "total_input_tokens": state.get("global_input_tokens"),
        "total_target_tokens": state.get("global_valid_targets"),
        "total_slow_updates": state.get("global_slow_steps"),
        "total_fast_updates": state.get("global_fast_updates"),
        "memory_resets": state.get("memory_reset_count"),
        "cycle_end_probes": len(probe_summaries),
        "forgetting_evaluations": state.get(
            "forgetting_evaluation_count", 0
        ),
    }
    mismatches: dict[str, dict[str, Any]] = {}
    for key, value in expected.items():
        if key == "logical_batches_per_task":
            continue
        if actual.get(key) != value:
            mismatches[key] = {
                "expected": value,
                "actual": actual.get(key),
            }

    if state.get("phase") != "task_boundary":
        mismatches["phase"] = {
            "expected": "task_boundary",
            "actual": state.get("phase"),
        }
    for key, value in variant.items():
        if resolved_variant.get(key) != value:
            mismatches[f"variant.{key}"] = {
                "expected": value,
                "actual": resolved_variant.get(key),
            }

    expected_modes = (
        {"reset", "carried"}
        if internal_variant in MEMORY_INTERNAL_VARIANTS
        else {"not_applicable"}
    )
    for cycle_index, summary in enumerate(probe_summaries):
        if summary.get("cycle_index") != cycle_index:
            mismatches[f"probe.{cycle_index}.cycle_index"] = {
                "expected": cycle_index,
                "actual": summary.get("cycle_index"),
            }
        curves = summary.get("auc_report", {}).get("curves", {})
        actual_modes = set(curves)
        if actual_modes != expected_modes:
            mismatches[f"probe.{cycle_index}.modes"] = {
                "expected": sorted(expected_modes),
                "actual": sorted(actual_modes),
            }
        expected_primary = (
            "carried"
            if internal_variant in MEMORY_INTERNAL_VARIANTS
            else "not_applicable"
        )
        if summary.get("primary_memory_evaluation_mode") != expected_primary:
            mismatches[f"probe.{cycle_index}.primary_mode"] = {
                "expected": expected_primary,
                "actual": summary.get(
                    "primary_memory_evaluation_mode"
                ),
            }

    if mismatches:
        raise ValueError(
            "Completion invariant mismatch: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return {
        "completion_invariant_schema_version": 1,
        "status": "valid",
        "internal_variant": internal_variant,
        "expected": expected,
        "actual": actual,
        "probe_memory_modes": sorted(expected_modes),
    }
