from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lm_cl.data.storage import atomic_write_json
from lm_cl.launcher.schema import PUBLIC_LANGUAGE_ORDER
from lm_cl.training.checkpoint import (
    CHECKPOINT_KIND,
    CHECKPOINT_SCHEMA_VERSION,
    atomic_save_checkpoint,
    canonical_sha256,
    load_checkpoint,
    sha256_file,
)


LATEST_POINTER_SCHEMA_VERSION = 1
LATEST_POINTER_KIND = "lm-cl-public-latest-checkpoint"


def _completed_tasks(state: dict[str, Any]) -> int:
    if state["phase"] == "task_boundary":
        return int(state["next_task_index"])
    return int(state["current_task_index"])


def pointer_for_checkpoint(
    checkpoint_path: str | Path,
    *,
    job_dir: str | Path,
    scientific_sha256: str,
    resolved_experiment_sha256: str,
    requested_horizon_cycles: int,
) -> dict[str, Any]:
    job_root = Path(job_dir).resolve()
    path = Path(checkpoint_path).resolve()
    try:
        relative = path.relative_to(job_root)
    except ValueError as exc:
        raise ValueError("Latest checkpoint must live inside its job directory") from exc
    payload = load_checkpoint(path, map_location="cpu")
    state = payload["trainer_state"]
    completed_tasks = _completed_tasks(state)
    return {
        "schema_version": LATEST_POINTER_SCHEMA_VERSION,
        "kind": LATEST_POINTER_KIND,
        "checkpoint_path": relative.as_posix(),
        "checkpoint_sha256": sha256_file(path),
        "checkpoint_schema_version": payload["checkpoint_schema_version"],
        "checkpoint_kind": payload["checkpoint_kind"],
        "checkpoint_config_sha256": payload["config_sha256"],
        "phase": state["phase"],
        "completed_task_count": completed_tasks,
        "completed_cycle_count": completed_tasks // len(PUBLIC_LANGUAGE_ORDER),
        "current_task_index": state["current_task_index"],
        "next_task_index": state["next_task_index"],
        "global_logical_batches": state["global_logical_batches"],
        "global_input_tokens": state["global_input_tokens"],
        "global_valid_targets": state["global_valid_targets"],
        "scientific_sha256": scientific_sha256,
        "resolved_experiment_sha256": resolved_experiment_sha256,
        "requested_horizon_cycles": requested_horizon_cycles,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def write_latest_pointer(job_dir: str | Path, pointer: dict[str, Any]) -> Path:
    path = Path(job_dir).resolve() / "latest_checkpoint.json"
    atomic_write_json(path, pointer)
    validate_latest_pointer(path, expected_job_dir=job_dir)
    return path


def validate_latest_pointer(
    path: str | Path,
    *,
    expected_job_dir: str | Path | None = None,
    expected_scientific_sha256: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    pointer_path = Path(path).resolve()
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "kind",
        "checkpoint_path",
        "checkpoint_sha256",
        "checkpoint_schema_version",
        "checkpoint_kind",
        "checkpoint_config_sha256",
        "phase",
        "completed_task_count",
        "completed_cycle_count",
        "current_task_index",
        "next_task_index",
        "global_logical_batches",
        "global_input_tokens",
        "global_valid_targets",
        "scientific_sha256",
        "resolved_experiment_sha256",
        "requested_horizon_cycles",
        "updated_at_utc",
    }
    if not isinstance(pointer, dict) or set(pointer) != required:
        raise ValueError("latest_checkpoint.json has an invalid field set")
    if pointer["schema_version"] != LATEST_POINTER_SCHEMA_VERSION:
        raise ValueError("Unknown latest-checkpoint pointer schema")
    if pointer["kind"] != LATEST_POINTER_KIND:
        raise ValueError("Unknown latest-checkpoint pointer kind")
    job_dir = (
        Path(expected_job_dir).resolve()
        if expected_job_dir is not None
        else pointer_path.parent
    )
    relative = Path(pointer["checkpoint_path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Latest-checkpoint path must be a safe relative path")
    checkpoint_path = (job_dir / relative).resolve()
    try:
        checkpoint_path.relative_to(job_dir)
    except ValueError as exc:
        raise ValueError("Latest-checkpoint path escapes the job directory") from exc
    if sha256_file(checkpoint_path) != pointer["checkpoint_sha256"]:
        raise ValueError("Latest checkpoint SHA-256 mismatch")
    payload = load_checkpoint(checkpoint_path, map_location="cpu")
    state = payload["trainer_state"]
    completed_tasks = _completed_tasks(state)
    expected_values = {
        "checkpoint_schema_version": payload["checkpoint_schema_version"],
        "checkpoint_kind": payload["checkpoint_kind"],
        "checkpoint_config_sha256": payload["config_sha256"],
        "phase": state["phase"],
        "completed_task_count": completed_tasks,
        "completed_cycle_count": completed_tasks // len(PUBLIC_LANGUAGE_ORDER),
        "current_task_index": state["current_task_index"],
        "next_task_index": state["next_task_index"],
        "global_logical_batches": state["global_logical_batches"],
        "global_input_tokens": state["global_input_tokens"],
        "global_valid_targets": state["global_valid_targets"],
    }
    mismatches = {
        key: {"pointer": pointer[key], "checkpoint": value}
        for key, value in expected_values.items()
        if pointer[key] != value
    }
    if mismatches:
        raise ValueError(f"Latest pointer/checkpoint mismatch: {mismatches}")
    if pointer["checkpoint_schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("Latest checkpoint schema is not release-compatible")
    if pointer["checkpoint_kind"] != CHECKPOINT_KIND:
        raise ValueError("Latest pointer references an incompatible checkpoint")
    if (
        expected_scientific_sha256 is not None
        and pointer["scientific_sha256"] != expected_scientific_sha256
    ):
        raise ValueError("Latest checkpoint scientific identity mismatch")
    return pointer, payload, checkpoint_path


def validate_horizon_extension(
    old_resolved: dict[str, Any],
    new_resolved: dict[str, Any],
    *,
    completed_cycles: int,
) -> None:
    if old_resolved.get("resolved_experiment_schema_version") != 1:
        raise ValueError("Existing resolved experiment schema is incompatible")
    if new_resolved.get("resolved_experiment_schema_version") != 1:
        raise ValueError("Requested resolved experiment schema is incompatible")
    if old_resolved["public_model"] != new_resolved["public_model"]:
        raise ValueError("Horizon extension cannot change the public model")
    if old_resolved["seed"] != new_resolved["seed"]:
        raise ValueError("Horizon extension cannot change the seed")
    if old_resolved["scientific_sha256"] != new_resolved["scientific_sha256"]:
        raise ValueError("Horizon extension changes scientific semantics")
    new_horizon = int(new_resolved["requested_horizon_cycles"])
    old_horizon = int(old_resolved["requested_horizon_cycles"])
    if new_horizon < completed_cycles:
        raise ValueError("Requested horizon is below completed cycles")
    if new_horizon < old_horizon:
        raise ValueError("Decreasing a configured cycle horizon is prohibited")
    old_matrix = old_resolved["data_contract"]["data_manifests"]
    new_matrix = new_resolved["data_contract"]["data_manifests"]
    if len(new_matrix) < new_horizon:
        raise ValueError("Future cycle manifests are missing")
    if len(old_matrix) > len(new_matrix):
        raise ValueError("Requested manifest matrix truncates existing cycles")
    for cycle, old_cycle in enumerate(old_matrix):
        if old_cycle != new_matrix[cycle]:
            raise ValueError(
                f"Manifest identities changed for configured cycle {cycle}"
            )


def _validate_internal_extension(
    old_config: dict[str, Any], new_config: dict[str, Any]
) -> None:
    for field in ("model", "variant", "optimization", "distributed"):
        if old_config.get(field) != new_config.get(field):
            raise ValueError(f"Internal horizon extension changes {field}")
    old_runtime = old_config["runtime"]
    new_runtime = new_config["runtime"]
    for field in ("seed", "device", "deterministic_algorithms", "output_dir"):
        if old_runtime.get(field) != new_runtime.get(field):
            raise ValueError(
                f"Internal horizon extension changes runtime.{field}"
            )
    old_tasks = old_config["tasks"]
    new_tasks = new_config["tasks"]
    if len(new_tasks) < len(old_tasks) or new_tasks[: len(old_tasks)] != old_tasks:
        raise ValueError("Internal horizon extension changes existing tasks")


def migrate_checkpoint_horizon(
    checkpoint_path: str | Path,
    *,
    new_internal_config: dict[str, Any],
    output_path: str | Path,
    extension_record: dict[str, Any],
) -> tuple[str, str]:
    payload = load_checkpoint(checkpoint_path, map_location="cpu")
    if payload["trainer_state"]["phase"] != "task_boundary":
        raise ValueError("Horizon extension requires a stable task boundary")
    completed_tasks = int(payload["trainer_state"]["next_task_index"])
    if completed_tasks % len(PUBLIC_LANGUAGE_ORDER):
        raise ValueError("Horizon extension requires a completed cycle boundary")
    _validate_internal_extension(
        payload["resolved_config"], new_internal_config
    )
    migrated = copy.deepcopy(payload)
    migrated["resolved_config"] = new_internal_config
    migrated["config_sha256"] = canonical_sha256(new_internal_config)
    history = list(migrated.get("horizon_extension_history", []))
    history.append(extension_record)
    migrated["horizon_extension_history"] = history
    checksum = atomic_save_checkpoint(output_path, migrated)
    return str(Path(output_path).resolve()), checksum


def augment_cycle_checkpoint(
    checkpoint_path: str | Path,
    *,
    output_path: str | Path,
    experiment_state: dict[str, Any],
) -> tuple[str, str]:
    payload = load_checkpoint(checkpoint_path, map_location="cpu")
    if payload["trainer_state"]["phase"] != "task_boundary":
        raise ValueError("Cycle checkpoint augmentation requires task boundary")
    completed_tasks = int(payload["trainer_state"]["next_task_index"])
    if completed_tasks % len(PUBLIC_LANGUAGE_ORDER):
        raise ValueError("Cycle checkpoint augmentation requires Russian boundary")
    augmented = copy.deepcopy(payload)
    augmented["experiment_state"] = experiment_state
    checksum = atomic_save_checkpoint(output_path, augmented)
    return str(Path(output_path).resolve()), checksum


def discover_unambiguous_latest_checkpoint(
    job_dir: str | Path,
    *,
    expected_config_sha256: str,
) -> Path | None:
    checkpoint_dir = Path(job_dir).resolve() / "checkpoints"
    if not checkpoint_dir.is_dir():
        return None
    candidates: list[tuple[tuple[int, int, int], Path, str, bool]] = []
    for path in checkpoint_dir.glob("*.pt"):
        try:
            payload = load_checkpoint(path, map_location="cpu")
        except (OSError, ValueError):
            continue
        if payload["config_sha256"] != expected_config_sha256:
            continue
        state = payload["trainer_state"]
        progress = (
            _completed_tasks(state),
            int(state["global_logical_batches"]),
            1 if state["phase"] == "task_boundary" else 0,
        )
        completed_tasks = _completed_tasks(state)
        completed_cycles = completed_tasks // len(PUBLIC_LANGUAGE_ORDER)
        experiment_state = payload.get("experiment_state")
        is_augmented_cycle_checkpoint = (
            state["phase"] == "task_boundary"
            and completed_tasks > 0
            and completed_tasks % len(PUBLIC_LANGUAGE_ORDER) == 0
            and path.name == f"cycle-{completed_cycles:04d}-complete.pt"
            and isinstance(experiment_state, dict)
            and experiment_state.get("completed_cycle_count")
            == completed_cycles
        )
        candidates.append(
            (
                progress,
                path.resolve(),
                sha256_file(path),
                is_augmented_cycle_checkpoint,
            )
        )
    if not candidates:
        return None
    maximum = max(item[0] for item in candidates)
    winners = [item for item in candidates if item[0] == maximum]
    unique_hashes = {item[2] for item in winners}
    if len(unique_hashes) != 1:
        augmented = [item for item in winners if item[3]]
        if len({item[2] for item in augmented}) == 1:
            return sorted((item[1] for item in augmented), key=str)[0]
        raise ValueError(
            "Ambiguous latest checkpoints share progress but differ in content"
        )
    return sorted((item[1] for item in winners), key=str)[0]
