from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch

from lm_cl.training.checkpoint import canonical_sha256, sha256_file


PROBE_CHECKPOINT_SCHEMA_VERSION = 1
PROBE_CHECKPOINT_KIND = "lm-cl-probe-checkpoint"


def validate_probe_checkpoint_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Probe checkpoint payload must be a mapping")
    if payload.get("checkpoint_schema_version") != (
        PROBE_CHECKPOINT_SCHEMA_VERSION
    ):
        raise ValueError("Unknown probe checkpoint schema version")
    if payload.get("checkpoint_kind") != PROBE_CHECKPOINT_KIND:
        raise ValueError("Unknown probe checkpoint kind")
    required = {
        "model_state",
        "optimizer_state",
        "scheduler_state",
        "gradients",
        "scaler_state",
        "probe_state",
        "rng_state",
        "resolved_config",
        "config_sha256",
        "source_checkpoint",
        "initialization_policy",
        "training_source_identity",
        "validation_source_identity",
        "memory_state",
        "curve_records",
        "completed_evaluation_steps",
        "auc_policy",
        "provenance",
        "distributed_state",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"Probe checkpoint is missing fields: {missing}")
    if canonical_sha256(payload["resolved_config"]) != payload["config_sha256"]:
        raise ValueError("Probe resolved-configuration checksum mismatch")
    source = payload["source_checkpoint"]
    source_required = {
        "path",
        "sha256",
        "checkpoint_kind",
        "checkpoint_schema_version",
        "continual_boundary",
    }
    if not isinstance(source, dict) or set(source) != source_required:
        raise ValueError("Probe source-checkpoint identity is incomplete")
    if (
        not isinstance(source["sha256"], str)
        or len(source["sha256"]) != 64
    ):
        raise ValueError("Probe source-checkpoint hash is invalid")
    state = payload["probe_state"]
    if not isinstance(state, dict):
        raise ValueError("Probe state must be a mapping")
    for counter in (
        "global_logical_batches",
        "global_slow_steps",
        "global_input_tokens",
        "global_valid_targets",
        "window_logical_batches",
        "window_valid_targets",
    ):
        value = state.get(counter)
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"Probe counter {counter} is invalid")
    if state.get("phase") not in {"probe_active", "probe_complete"}:
        raise ValueError("Probe checkpoint phase is invalid")
    completed = payload["completed_evaluation_steps"]
    curve = payload["curve_records"]
    if (
        not isinstance(completed, list)
        or completed != sorted(set(completed))
        or not all(isinstance(step, int) and step >= 0 for step in completed)
    ):
        raise ValueError("Completed probe evaluations are invalid")
    if not isinstance(curve, list):
        raise ValueError("Probe curve records must be a list")
    curve_steps = sorted(
        {
            item.get("probe_logical_step")
            for item in curve
            if isinstance(item, dict)
        }
    )
    if curve_steps != completed:
        raise ValueError("Probe curve/evaluation-step coverage differs")
    memory = payload["memory_state"]
    if not isinstance(memory, dict):
        raise ValueError("Probe memory state must be a mapping")
    for name in (
        "variant",
        "initial_memory",
        "active_memory",
        "active_memory_gradient",
        "fast_update_phase",
        "fast_lr",
        "configured_fast_lr",
        "fast_lr_override",
        "fast_clip_threshold",
        "reset_policy",
        "segment_length",
        "memory_token_count",
        "memory_evaluation_policy",
    ):
        if name not in memory:
            raise ValueError(f"Probe memory state lacks {name}")
    model_m0 = payload["model_state"].get("initial_memory")
    checkpoint_m0 = memory["initial_memory"]
    if (model_m0 is None) != (checkpoint_m0 is None):
        raise ValueError("Probe M0 copies are inconsistent")
    if model_m0 is not None and not torch.equal(model_m0, checkpoint_m0):
        raise ValueError("Probe M0 copies differ")
    distributed = payload["distributed_state"]
    if not isinstance(distributed, dict):
        raise ValueError("Probe distributed state must be a mapping")
    world_size = distributed.get("world_size")
    rank_rng_states = distributed.get("rank_rng_states")
    if (
        not isinstance(world_size, int)
        or world_size <= 0
        or not isinstance(rank_rng_states, list)
        or len(rank_rng_states) != world_size
    ):
        raise ValueError("Probe distributed RNG coverage is incomplete")
    ranks = {
        item.get("rank")
        for item in rank_rng_states
        if isinstance(item, dict)
    }
    if ranks != set(range(world_size)):
        raise ValueError("Probe rank RNG mapping is invalid")
    return payload


def load_probe_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint = Path(path).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Probe checkpoint not found: {checkpoint}")
    try:
        payload = torch.load(
            checkpoint,
            map_location=map_location,
            weights_only=False,
        )
    except Exception as exc:
        raise ValueError(
            f"Probe checkpoint cannot be loaded: {checkpoint}"
        ) from exc
    return validate_probe_checkpoint_payload(payload)


def atomic_save_probe_checkpoint(
    path: str | Path,
    payload: dict[str, Any],
) -> str:
    output = Path(path).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    validate_probe_checkpoint_payload(payload)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"Temporary checkpoint exists: {temporary}")
    try:
        with temporary.open("xb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        load_probe_checkpoint(temporary)
        os.replace(temporary, output)
        directory = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    return sha256_file(output)


def atomic_write_json(
    path: str | Path,
    value: dict[str, Any],
    *,
    overwrite: bool = False,
) -> None:
    output = Path(path).expanduser().resolve()
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite JSON: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"Temporary JSON exists: {temporary}")
    encoded = (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
        directory = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
