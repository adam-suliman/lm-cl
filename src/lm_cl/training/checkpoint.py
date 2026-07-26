from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lm_cl.environment import inspect_environment


CHECKPOINT_SCHEMA_VERSION = 3
LEGACY_CHECKPOINT_SCHEMA_VERSION = 2
CHECKPOINT_KIND = "lm-cl-continual-checkpoint"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def source_tree_sha256(root: Path | None = None) -> str:
    root = root or repository_root()
    paths = [root / "pyproject.toml"]
    paths.extend(sorted((root / "src" / "lm_cl").rglob("*.py")))
    digest = hashlib.sha256()
    for path in paths:
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        content = path.read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def git_provenance(root: Path | None = None) -> dict[str, Any]:
    root = root or repository_root()
    if not (root / ".git").exists():
        return {
            "commit": None,
            "dirty": None,
            "status": "not_a_git_worktree",
            "source_tree_sha256": source_tree_sha256(root),
        }
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return {
        "commit": commit,
        "dirty": bool(status),
        "status": "ok",
        "source_tree_sha256": source_tree_sha256(root),
    }


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda_all": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda_all"])


def checkpoint_metadata() -> dict[str, Any]:
    return {
        "git": git_provenance(),
        "environment": inspect_environment(),
    }


def validate_checkpoint_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint payload must be a mapping")
    schema_version = payload.get("checkpoint_schema_version")
    if schema_version not in {
        LEGACY_CHECKPOINT_SCHEMA_VERSION,
        CHECKPOINT_SCHEMA_VERSION,
    }:
        raise ValueError("Unknown checkpoint schema version")
    if payload.get("checkpoint_kind") != CHECKPOINT_KIND:
        raise ValueError("Unknown checkpoint kind")
    required = {
        "model_state",
        "optimizer_state",
        "scheduler_state",
        "gradients",
        "scaler_state",
        "trainer_state",
        "rng_state",
        "resolved_config",
        "config_sha256",
        "source_identity",
        "provenance",
        "memory_state",
    }
    if schema_version == CHECKPOINT_SCHEMA_VERSION:
        required.add("distributed_state")
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"Checkpoint is missing fields: {missing}")
    if canonical_sha256(payload["resolved_config"]) != payload["config_sha256"]:
        raise ValueError("Checkpoint resolved-configuration checksum mismatch")
    trainer_state = payload["trainer_state"]
    if not isinstance(trainer_state, dict):
        raise ValueError("Checkpoint trainer_state must be a mapping")
    for counter in (
        "next_task_index",
        "global_logical_batches",
        "global_slow_steps",
        "global_input_tokens",
        "global_valid_targets",
    ):
        value = trainer_state.get(counter)
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"Checkpoint counter {counter} is invalid")
    if trainer_state.get("phase") not in {"task_active", "task_boundary"}:
        raise ValueError("Checkpoint trainer phase is invalid")
    memory_state = payload["memory_state"]
    if not isinstance(memory_state, dict):
        raise ValueError("Checkpoint memory_state must be a mapping")
    memory_required = {
        "variant",
        "initial_memory",
        "active_memory",
        "active_memory_gradient",
        "fast_update_phase",
        "fast_lr",
        "fast_clip_threshold",
        "reset_policy",
        "segment_length",
        "memory_token_count",
        "memory_evaluation_policy",
    }
    memory_missing = sorted(memory_required - set(memory_state))
    if memory_missing:
        raise ValueError(
            f"Checkpoint memory_state is missing fields: {memory_missing}"
        )
    if memory_state["fast_update_phase"] not in {
        "not_applicable",
        "ready_for_next_logical_batch",
        "backward_complete",
    }:
        raise ValueError("Checkpoint fast-update phase is invalid")
    initial_memory = memory_state["initial_memory"]
    model_initial_memory = payload["model_state"].get("initial_memory")
    if (initial_memory is None) != (model_initial_memory is None):
        raise ValueError("Checkpoint M0 copies are inconsistent")
    if initial_memory is not None:
        if not torch.equal(initial_memory, model_initial_memory):
            raise ValueError("Checkpoint M0 copies differ")
    if schema_version == CHECKPOINT_SCHEMA_VERSION:
        distributed_state = payload["distributed_state"]
        if not isinstance(distributed_state, dict):
            raise ValueError("Checkpoint distributed_state must be a mapping")
        distributed_required = {
            "schema_version",
            "enabled",
            "backend",
            "world_size",
            "rank_topology",
            "global_logical_batch_size",
            "partition_rule",
            "rank_rng_states",
            "global_source_position",
            "global_input_tokens",
            "global_valid_targets",
            "reduction_policy",
            "ddp",
            "active_memory_sync_policy",
            "state_digests",
        }
        missing_distributed = sorted(
            distributed_required - set(distributed_state)
        )
        if missing_distributed:
            raise ValueError(
                "Checkpoint distributed_state is missing fields: "
                f"{missing_distributed}"
            )
        if distributed_state["schema_version"] != 1:
            raise ValueError("Unknown distributed checkpoint schema version")
        world_size = distributed_state["world_size"]
        if not isinstance(world_size, int) or world_size <= 0:
            raise ValueError("Distributed checkpoint world size is invalid")
        enabled = distributed_state["enabled"]
        if not isinstance(enabled, bool):
            raise ValueError("Distributed checkpoint enabled flag is invalid")
        if not enabled and world_size != 1:
            raise ValueError("Non-distributed checkpoint must have world size one")
        rank_rng_states = distributed_state["rank_rng_states"]
        topology = distributed_state["rank_topology"]
        if (
            not isinstance(rank_rng_states, list)
            or len(rank_rng_states) != world_size
            or not isinstance(topology, list)
            or len(topology) != world_size
        ):
            raise ValueError(
                "Distributed checkpoint rank metadata is incomplete"
            )
        ranks = {
            item.get("rank")
            for item in rank_rng_states
            if isinstance(item, dict)
        }
        if ranks != set(range(world_size)):
            raise ValueError(
                "Distributed checkpoint rank RNG mapping is corrupt"
            )
        topology_ranks = {
            item.get("rank")
            for item in topology
            if isinstance(item, dict)
        }
        if topology_ranks != set(range(world_size)):
            raise ValueError(
                "Distributed checkpoint rank topology is corrupt"
            )
        state_digests = distributed_state["state_digests"]
        if state_digests is not None:
            expected_digests = {
                "model",
                "gradients",
                "optimizer",
                "scheduler",
                "scaler",
                "trainer",
                "memory",
            }
            if (
                not isinstance(state_digests, dict)
                or set(state_digests) != expected_digests
                or any(
                    not isinstance(value, str)
                    or len(value) != 64
                    or any(
                        character not in "0123456789abcdef"
                        for character in value
                    )
                    for value in state_digests.values()
                )
            ):
                raise ValueError(
                    "Distributed checkpoint state digests are corrupt"
                )
        if distributed_state["global_source_position"] != (
            trainer_state["source_position"]
        ):
            raise ValueError(
                "Distributed checkpoint source position is inconsistent"
            )
        if distributed_state["global_input_tokens"] != (
            trainer_state["global_input_tokens"]
        ):
            raise ValueError(
                "Distributed checkpoint input-token counter is inconsistent"
            )
        if distributed_state["global_valid_targets"] != (
            trainer_state["global_valid_targets"]
        ):
            raise ValueError(
                "Distributed checkpoint target counter is inconsistent"
            )
    return payload


def load_checkpoint(
    path: str | Path, *, map_location: str | torch.device = "cpu"
) -> dict[str, Any]:
    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    try:
        payload = torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=False,
        )
    except Exception as exc:
        raise ValueError(f"Checkpoint cannot be loaded: {checkpoint_path}") from exc
    return validate_checkpoint_payload(payload)


def atomic_save_checkpoint(path: str | Path, payload: dict[str, Any]) -> str:
    output = Path(path).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    validate_checkpoint_payload(payload)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"Temporary checkpoint already exists: {temporary}")
    try:
        with temporary.open("xb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        load_checkpoint(temporary, map_location="cpu")
        os.replace(temporary, output)
        directory_fd = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    return sha256_file(output)
