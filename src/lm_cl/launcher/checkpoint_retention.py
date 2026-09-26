"""Opt-in retention of cycle evidence and the current alternating recovery point.

This runs only between completed, checkpoint-acknowledged turns. It never
targets a checkpoint from an existing study or an unacknowledged consumer.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re

from lm_cl.data.alternating import control
from lm_cl.data.storage import atomic_write_json
from lm_cl.launcher.schema import PUBLIC_LANGUAGE_ORDER
from lm_cl.training.checkpoint import sha256_file


_TASK_BOUNDARY = re.compile(r"task-(\d{4})-([a-z_]+)-boundary\.pt\Z")
_LEDGER_KIND = "lm-cl-cycle-checkpoint-retention-v1"


def retire_completed_turn_checkpoints(root: Path, plan: dict, jobs: list) -> None:
    """Retire old non-Russian task files after every consumer has advanced.

    Russian source checkpoints remain byte-identical for the derived probes.
    Augmented cycle checkpoints and completed probe checkpoints are never
    candidates. A durable hash ledger precedes each file removal so a crash
    cannot hide which exact file was retired.
    """
    if plan.get("checkpoint_retention") != "cycle_end_v1":
        return
    state = control(root, plan)
    if state["turn"] == 0:
        return
    turn = plan["alternation"]["turns"][state["turn"] - 1]
    consumers = plan["alternation"]["consumers"]
    if {job.job_id for job in jobs} != set(consumers):
        raise ValueError("Retention consumer group changed")
    for job in jobs:
        job_dir = Path(job.output_dir).resolve()
        if job_dir != Path(consumers[job.job_id]["output_dir"]):
            raise ValueError("Retention output path changed")
        ack = state["consumers"].get(job.job_id)
        if ack is None or ack["end_tokens"] != turn["end_tokens"]:
            raise ValueError("Retention requires a fully acknowledged turn")
        checkpoint_dir = job_dir / "checkpoints"
        if checkpoint_dir.is_symlink() or not checkpoint_dir.is_dir():
            raise ValueError("Retention checkpoint directory is invalid")
        current = Path(ack["checkpoint"]).resolve()
        if current.parent != checkpoint_dir or not current.is_file():
            raise ValueError("Retention recovery checkpoint is missing")
        pointer_path = job_dir / "latest_checkpoint.json"
        if pointer_path.is_symlink():
            raise ValueError("Retention latest pointer is symlinked")
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        if ((job_dir / pointer["checkpoint_path"]).resolve() != current
                or pointer["checkpoint_sha256"] != ack["sha256"]):
            # A child may have committed its next checkpoint before the other
            # consumers finished their wave. Keep the older acknowledged file
            # until the next barrier advances; recovery will validate it.
            continue
        ledger_path = job_dir / "checkpoint_retention.json"
        if ledger_path.is_symlink():
            raise ValueError("Retention ledger is symlinked")
        if ledger_path.is_file():
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
            if (ledger.get("kind") != _LEDGER_KIND
                    or ledger.get("recipe_sha256") != plan["sha256"]
                    or not isinstance(ledger.get("retired"), dict)):
                raise ValueError("Retention ledger identity differs")
        else:
            ledger = {"kind": _LEDGER_KIND, "recipe_sha256": plan["sha256"], "retired": {}}
        for path in sorted(checkpoint_dir.glob("task-*-boundary.pt")):
            if path.is_symlink():
                raise ValueError("Symlinked checkpoint cannot be retired")
            match = _TASK_BOUNDARY.fullmatch(path.name)
            if match is None:
                continue
            index = int(match[1])
            if (index >= turn["task_index"] + 1
                    or match[2] != PUBLIC_LANGUAGE_ORDER[index % len(PUBLIC_LANGUAGE_ORDER)]
                    or match[2] == "ru" or path == current):
                continue
            relative = path.relative_to(job_dir).as_posix()
            digest = sha256_file(path)
            previous = ledger["retired"].get(relative)
            if previous is not None and previous["sha256"] != digest:
                raise ValueError("Retired checkpoint bytes changed")
            if previous is None:
                ledger["retired"][relative] = {
                    "sha256": digest,
                    "released_turn": state["turn"],
                    "released_tokens": state["released_tokens"],
                }
                atomic_write_json(ledger_path, ledger)
            path.unlink()
            directory_fd = os.open(checkpoint_dir, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
