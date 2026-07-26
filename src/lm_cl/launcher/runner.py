from __future__ import annotations

import csv
import json
import os
import signal
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from lm_cl.config import save_continual_config
from lm_cl.data.storage import atomic_write_json
from lm_cl.environment import inspect_environment
from lm_cl.launcher.config import config_from_mapping
from lm_cl.launcher.jobs import (
    JobSpec,
    build_continual_job_config,
    build_probe_job_config,
    save_internal_continual_config,
    save_internal_probe_config,
)
from lm_cl.launcher.schema import PUBLIC_LANGUAGE_ORDER
from lm_cl.launcher.state import (
    augment_cycle_checkpoint,
    discover_unambiguous_latest_checkpoint,
    migrate_checkpoint_horizon,
    pointer_for_checkpoint,
    validate_latest_pointer,
    write_latest_pointer,
)
from lm_cl.metrics import TensorBoardTracker, percent_change_from_first
from lm_cl.metrics.jsonl import JsonlMetricLogger
from lm_cl.training.checkpoint import (
    canonical_sha256,
    git_provenance,
    load_checkpoint,
    sha256_file,
)
from lm_cl.training.probe_checkpoint import load_probe_checkpoint


class StageProcessController:
    def __init__(self):
        self.child: subprocess.Popen[bytes] | None = None
        self.received_signal: int | None = None

    def handler(self, signum: int, _frame: Any) -> None:
        self.received_signal = signum
        child = self.child
        if child is not None and child.poll() is None:
            child.send_signal(signum)

    def run(self, command: list[str], *, environment: dict[str, str]) -> None:
        self.child = subprocess.Popen(command, env=environment)
        try:
            status = self.child.wait()
        finally:
            self.child = None
        if self.received_signal is not None:
            raise KeyboardInterrupt(
                f"Job received signal {self.received_signal}"
            )
        if status != 0:
            raise RuntimeError(
                f"Child stage exited with status {status}: {command}"
            )


def _read_resolved(path: Path) -> dict[str, Any]:
    root = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(root, dict):
        raise ValueError("Resolved experiment must be a YAML mapping")
    claimed = root.get("resolved_experiment_sha256")
    unhashed = dict(root)
    unhashed.pop("resolved_experiment_sha256", None)
    if claimed != canonical_sha256(unhashed):
        raise ValueError("Resolved experiment SHA-256 mismatch")
    return root


def _job_spec(resolved: dict[str, Any]) -> JobSpec:
    return JobSpec(
        public_model=resolved["public_model"],
        internal_variant=resolved["internal_variant"],
        seed=resolved["seed"],
        output_dir=resolved["output_dir"],
        resolved_experiment=resolved,
        resolved_sha256=resolved["resolved_experiment_sha256"],
        scientific_sha256=resolved["scientific_sha256"],
    )


def _atomic_replace_continual_config(config: Any, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    save_continual_config(config, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _stage_command(
    *,
    module: str,
    arguments: list[str],
    world_size: int,
    rendezvous_port: int,
) -> list[str]:
    if world_size == 1:
        return [sys.executable, "-m", module, *arguments]
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc-per-node",
        str(world_size),
        "--master-port",
        str(rendezvous_port),
        "--module",
        module,
        *arguments,
    ]


def _next_attempt_path(root: Path, stem: str) -> tuple[Path, int]:
    attempt = 1
    while True:
        path = root / f"{stem}-attempt-{attempt:03d}.json"
        if not path.exists():
            return path, attempt
        attempt += 1


def _load_job_metadata(path: Path, spec: JobSpec) -> dict[str, Any]:
    if path.is_file():
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("job_id") != spec.job_id:
            raise ValueError("Existing job_metadata.json has another job identity")
        return value
    return {
        "schema_version": 1,
        "job_id": spec.job_id,
        "public_model": spec.public_model,
        "internal_variant": spec.internal_variant,
        "seed": spec.seed,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "attempts": [],
        "resume_history": [],
        "failure_history": [],
    }


def _archive_uncheckpointed_retry(job_dir: Path) -> list[str]:
    candidates = [
        job_dir / "resolved_config.yaml",
        job_dir / "metrics.jsonl",
        job_dir / "tensorboard",
    ]
    existing = [path for path in candidates if path.exists()]
    if not existing:
        return []
    archive = (
        job_dir
        / "internal"
        / "failed-attempts"
        / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    )
    archive.mkdir(parents=True, exist_ok=False)
    moved = []
    for path in existing:
        destination = archive / path.name
        shutil.move(str(path), str(destination))
        moved.append(str(destination))
    return moved


def _run_continual_stage(
    controller: StageProcessController,
    *,
    internal_config_path: Path,
    checkpoint: Path | None,
    stop_after_task_boundaries: int,
    result_path: Path,
    world_size: int,
    rendezvous_port: int,
    environment: dict[str, str],
) -> dict[str, Any]:
    if checkpoint is None:
        module = "lm_cl.cli.train_continual"
        args = [str(internal_config_path)]
    else:
        module = "lm_cl.cli.resume_continual"
        args = [str(internal_config_path), str(checkpoint)]
    args.extend(
        [
            "--stop-after-task-boundaries",
            str(stop_after_task_boundaries),
            "--result-json",
            str(result_path),
        ]
    )
    controller.run(
        _stage_command(
            module=module,
            arguments=args,
            world_size=world_size,
            rendezvous_port=rendezvous_port,
        ),
        environment=environment,
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    checkpoint_path = Path(result["checkpoint_path"]).resolve()
    if sha256_file(checkpoint_path) != result["checkpoint_sha256"]:
        raise ValueError("Continual stage result checkpoint SHA-256 mismatch")
    return result


def _latest_probe_checkpoint(probe_dir: Path, config_sha256: str) -> Path | None:
    checkpoint_dir = probe_dir / "checkpoints"
    if not checkpoint_dir.is_dir():
        return None
    candidates: list[tuple[int, Path, str]] = []
    for path in checkpoint_dir.glob("*.pt"):
        try:
            payload = load_probe_checkpoint(path, map_location="cpu")
        except (OSError, ValueError):
            continue
        if payload["config_sha256"] != config_sha256:
            continue
        step = int(payload["probe_state"]["global_logical_batches"])
        candidates.append((step, path.resolve(), sha256_file(path)))
    if not candidates:
        return None
    maximum = max(item[0] for item in candidates)
    winners = [item for item in candidates if item[0] == maximum]
    if len({item[2] for item in winners}) != 1:
        raise ValueError("Ambiguous probe checkpoints at the same step")
    return sorted((item[1] for item in winners), key=str)[0]


def _validate_probe_results(
    results_path: Path,
    *,
    source_checkpoint: Path,
    source_sha256: str,
) -> dict[str, Any]:
    result = json.loads(results_path.read_text(encoding="utf-8"))
    if result.get("status") != "complete":
        raise ValueError("Probe results are not complete")
    if result.get("source_checkpoint_sha256_before") != source_sha256:
        raise ValueError("Probe results source-before hash mismatch")
    if result.get("source_checkpoint_sha256_after") != source_sha256:
        raise ValueError("Probe results source-after hash mismatch")
    if sha256_file(source_checkpoint) != source_sha256:
        raise ValueError("Probe mutated its continual source checkpoint")
    return result


def _run_or_resume_probe(
    controller: StageProcessController,
    *,
    config: Any,
    spec: JobSpec,
    cycle_index: int,
    source_checkpoint: Path,
    source_sha256: str,
    world_size: int,
    rendezvous_port: int,
    environment: dict[str, str],
) -> dict[str, Any]:
    probe = build_probe_job_config(
        config,
        spec,
        cycle_index=cycle_index,
        source_checkpoint=source_checkpoint,
        source_checkpoint_sha256=source_sha256,
    )
    config_path = save_internal_probe_config(
        probe, spec.output_dir, cycle_index=cycle_index
    )
    probe_dir = Path(probe.runtime.output_dir)
    results_path = probe_dir / "probe_results.json"
    if results_path.is_file():
        return _validate_probe_results(
            results_path,
            source_checkpoint=source_checkpoint,
            source_sha256=source_sha256,
        )
    result_root = Path(spec.output_dir) / "internal" / "results"
    result_path, _ = _next_attempt_path(
        result_root, f"probe-cycle-{cycle_index + 1:04d}"
    )
    resume_checkpoint = _latest_probe_checkpoint(
        probe_dir, canonical_sha256(probe.to_dict())
    )
    if resume_checkpoint is None:
        module = "lm_cl.cli.run_probe"
        arguments = [str(config_path)]
    else:
        payload = load_probe_checkpoint(resume_checkpoint, map_location="cpu")
        if payload["probe_state"]["phase"] == "probe_complete":
            raise ValueError(
                "Complete probe checkpoint exists without probe_results.json"
            )
        module = "lm_cl.cli.resume_probe"
        arguments = [str(config_path), str(resume_checkpoint)]
    arguments.extend(["--result-json", str(result_path)])
    controller.run(
        _stage_command(
            module=module,
            arguments=arguments,
            world_size=world_size,
            rendezvous_port=rendezvous_port,
        ),
        environment=environment,
    )
    return _validate_probe_results(
        results_path,
        source_checkpoint=source_checkpoint,
        source_sha256=source_sha256,
    )


def _probe_summary(
    result: dict[str, Any], *, public_model: str, cycle_index: int
) -> dict[str, Any]:
    mode = "carried" if public_model == "fastmem_rmt" else "not_applicable"
    curves = result["auc_report"]["curves"]
    if mode not in curves:
        raise ValueError(f"Primary probe curve {mode} is missing")
    curve = curves[mode]
    return {
        "cycle_index": cycle_index,
        "cycle_number": cycle_index + 1,
        "primary_memory_evaluation_mode": mode,
        "step_0_validation_ce": curve["step_0_validation_ce"],
        "full_validation_curve": [
            item
            for item in result["curve_records"]
            if item["memory_evaluation_mode"] == mode
        ],
        "final_validation_ce": curve["final_validation_ce"],
        "normalized_token_auc": curve[
            "primary_normalized_trapezoidal_auc"
        ],
        "raw_token_auc": curve["raw_input_token_trapezoidal_auc"],
        "raw_step_auc": curve["raw_step_trapezoidal_auc"],
        "arithmetic_mean_validation_ce": curve[
            "arithmetic_mean_recorded_validation_ce"
        ],
        "source_checkpoint": result["source_checkpoint"],
        "source_checkpoint_sha256_before": result[
            "source_checkpoint_sha256_before"
        ],
        "source_checkpoint_sha256_after": result[
            "source_checkpoint_sha256_after"
        ],
        "probe_results_path": str(
            Path(result.get("results_path", "probe_results.json"))
        ),
        "auc_report": result["auc_report"],
    }


def _load_completed_probe_summaries(checkpoint_payload: dict[str, Any]) -> list[dict[str, Any]]:
    state = checkpoint_payload.get("experiment_state")
    if not isinstance(state, dict):
        return []
    summaries = state.get("completed_probe_summaries", [])
    if not isinstance(summaries, list):
        raise ValueError("Checkpoint experiment probe summaries are invalid")
    return list(summaries)


def _record_cycle_tensorboard(
    config: Any,
    job_dir: Path,
    probe_summaries: list[dict[str, Any]],
) -> None:
    if not config.tracking.tensorboard or not probe_summaries:
        return
    first_auc = float(probe_summaries[0]["normalized_token_auc"])
    tracker = TensorBoardTracker(
        job_dir / "tensorboard",
        flush_seconds=config.tracking.tensorboard_flush_seconds,
    )
    try:
        for item in probe_summaries:
            step = int(item["cycle_number"])
            tracker.scalar(
                "probe/normalized_token_auc",
                item["normalized_token_auc"],
                step,
            )
            tracker.scalar(
                "probe/raw_token_auc", item["raw_token_auc"], step
            )
            tracker.scalar("probe/raw_step_auc", item["raw_step_auc"], step)
            tracker.scalar(
                "probe/auc_percent_change_from_cycle1",
                percent_change_from_first(
                    float(item["normalized_token_auc"]), first_auc
                ),
                step,
            )
    finally:
        tracker.close()


def _summary(
    *,
    spec: JobSpec,
    config: Any,
    checkpoint_path: Path,
    payload: dict[str, Any],
    probes: list[dict[str, Any]],
    metadata: dict[str, Any],
    started: float,
) -> dict[str, Any]:
    state = payload["trainer_state"]
    first_auc = (
        None if not probes else float(probes[0]["normalized_token_auc"])
    )
    probe_rows = []
    for item in probes:
        row = dict(item)
        row["auc_percent_change_from_cycle1"] = (
            None
            if first_auc is None
            else percent_change_from_first(
                float(item["normalized_token_auc"]), first_auc
            )
        )
        probe_rows.append(row)
    return {
        "summary_schema_version": 1,
        "status": "complete",
        "model": spec.public_model,
        "internal_variant": spec.internal_variant,
        "seed": spec.seed,
        "cycles_requested": config.experiment.cycles,
        "cycles_completed": state["next_task_index"]
        // len(PUBLIC_LANGUAGE_ORDER),
        "completed_language_tasks": state["next_task_index"],
        "total_input_tokens": state["global_input_tokens"],
        "total_target_tokens": state["global_valid_targets"],
        "total_logical_batches": state["global_logical_batches"],
        "total_slow_updates": state["global_slow_steps"],
        "total_fast_updates": state["global_fast_updates"],
        "final_checkpoint_path": str(checkpoint_path),
        "final_checkpoint_sha256": sha256_file(checkpoint_path),
        "elapsed_seconds": time.monotonic() - started,
        "per_cycle_probe_auc": probe_rows,
        "final_validation_losses": [
            item["final_validation_ce"] for item in probe_rows
        ],
        "failure_history": metadata["failure_history"],
        "resume_history": metadata["resume_history"],
        "environment_identity": inspect_environment(),
        "gpu_identity": inspect_environment()["gpus"],
        "source_tree_identity": git_provenance(),
        "scientific_sha256": spec.scientific_sha256,
        "resolved_experiment_sha256": spec.resolved_sha256,
        "data_manifest_identities": spec.resolved_experiment[
            "data_contract"
        ],
    }


def run_resolved_job(
    resolved_path: str | Path,
    *,
    rendezvous_port: int,
    retry_resume: bool = False,
) -> dict[str, Any]:
    started = time.monotonic()
    resolved_file = Path(resolved_path).resolve()
    resolved = _read_resolved(resolved_file)
    config = config_from_mapping(resolved["launcher_config"])
    spec = _job_spec(resolved)
    job_dir = Path(spec.output_dir).resolve()
    if resolved_file != job_dir / "resolved_experiment.yaml":
        raise ValueError("Resolved job configuration is outside its job directory")
    job_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = job_dir / "job_metadata.json"
    metadata = _load_job_metadata(metadata_path, spec)
    metadata["attempts"].append(
        {
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "resolved_experiment_sha256": spec.resolved_sha256,
            "requested_horizon_cycles": config.experiment.cycles,
        }
    )
    atomic_write_json(metadata_path, metadata)
    event_logger: JsonlMetricLogger | None = None
    controller = StageProcessController()
    old_sigint = signal.signal(signal.SIGINT, controller.handler)
    old_sigterm = signal.signal(signal.SIGTERM, controller.handler)
    world_size = max(config.launcher.gpus_per_job, 1)
    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    continual = build_continual_job_config(config, spec)
    internal_path = save_internal_continual_config(continual, job_dir)
    internal_sha = canonical_sha256(continual.to_dict())
    pointer_path = job_dir / "latest_checkpoint.json"
    checkpoint: Path | None = None
    checkpoint_payload: dict[str, Any] | None = None
    resume_mode = "auto" if retry_resume else config.experiment.resume
    try:
        if resume_mode == "never":
            if pointer_path.exists() or (job_dir / "resolved_config.yaml").exists():
                raise FileExistsError("resume=never refuses existing job state")
        elif pointer_path.is_file():
            pointer, checkpoint_payload, checkpoint = validate_latest_pointer(
                pointer_path,
                expected_job_dir=job_dir,
                expected_scientific_sha256=spec.scientific_sha256,
            )
            metadata["resume_history"].append(
                {
                    "at_utc": datetime.now(timezone.utc).isoformat(),
                    "checkpoint_path": str(checkpoint),
                    "checkpoint_sha256": pointer["checkpoint_sha256"],
                    "completed_cycle_count": pointer[
                        "completed_cycle_count"
                    ],
                }
            )
            if checkpoint_payload["config_sha256"] != internal_sha:
                if checkpoint_payload["trainer_state"]["phase"] != "task_boundary":
                    raise ValueError(
                        "Horizon extension from an active task is prohibited"
                    )
                old_horizon = pointer["requested_horizon_cycles"]
                extension_path = (
                    job_dir
                    / "checkpoints"
                    / f"horizon-extension-{old_horizon:04d}-to-{config.experiment.cycles:04d}.pt"
                )
                extension_identity = {
                    "old_horizon_cycles": old_horizon,
                    "new_horizon_cycles": config.experiment.cycles,
                    "old_resolved_experiment_sha256": pointer[
                        "resolved_experiment_sha256"
                    ],
                    "new_resolved_experiment_sha256": spec.resolved_sha256,
                    "source_checkpoint_sha256": pointer["checkpoint_sha256"],
                }
                if extension_path.is_file():
                    checkpoint_payload = load_checkpoint(extension_path)
                    if checkpoint_payload["config_sha256"] != internal_sha:
                        raise ValueError(
                            "Existing horizon-extension checkpoint has the wrong configuration"
                        )
                    history = checkpoint_payload.get(
                        "horizon_extension_history", []
                    )
                    if not history or any(
                        history[-1].get(key) != value
                        for key, value in extension_identity.items()
                    ):
                        raise ValueError(
                            "Existing horizon-extension checkpoint has the wrong source identity"
                        )
                    checkpoint = extension_path
                else:
                    checkpoint_string, _ = migrate_checkpoint_horizon(
                        checkpoint,
                        new_internal_config=continual.to_dict(),
                        output_path=extension_path,
                        extension_record={
                            "extended_at_utc": datetime.now(
                                timezone.utc
                            ).isoformat(),
                            **extension_identity,
                        },
                    )
                    checkpoint = Path(checkpoint_string)
                    checkpoint_payload = load_checkpoint(checkpoint)
                _atomic_replace_continual_config(
                    continual, job_dir / "resolved_config.yaml"
                )
        elif resume_mode == "required":
            raise FileNotFoundError("resume=required needs latest_checkpoint.json")
        elif resume_mode == "auto":
            discovered = discover_unambiguous_latest_checkpoint(
                job_dir, expected_config_sha256=internal_sha
            )
            if discovered is not None:
                checkpoint = discovered
                checkpoint_payload = load_checkpoint(checkpoint)

        discovered = discover_unambiguous_latest_checkpoint(
            job_dir, expected_config_sha256=internal_sha
        )
        if discovered is not None:
            discovered_payload = load_checkpoint(discovered)
            discovered_progress = (
                discovered_payload["trainer_state"]["global_logical_batches"]
            )
            current_progress = (
                -1
                if checkpoint_payload is None
                else checkpoint_payload["trainer_state"][
                    "global_logical_batches"
                ]
            )
            if discovered_progress > current_progress:
                checkpoint = discovered
                checkpoint_payload = discovered_payload

        completed_tasks = (
            0
            if checkpoint_payload is None
            else (
                checkpoint_payload["trainer_state"]["next_task_index"]
                if checkpoint_payload["trainer_state"]["phase"]
                == "task_boundary"
                else checkpoint_payload["trainer_state"]["current_task_index"]
            )
        )
        if completed_tasks > len(continual.tasks):
            raise ValueError("Checkpoint is beyond the requested horizon")
        if checkpoint is None and retry_resume:
            archived = _archive_uncheckpointed_retry(job_dir)
            if archived:
                metadata["resume_history"].append(
                    {
                        "at_utc": datetime.now(timezone.utc).isoformat(),
                        "action": "archive_uncheckpointed_failed_attempt",
                        "archived_paths": archived,
                    }
                )
                atomic_write_json(metadata_path, metadata)
        probe_summaries = (
            []
            if checkpoint_payload is None
            else _load_completed_probe_summaries(checkpoint_payload)
        )
        if (job_dir / "metrics.jsonl").is_file():
            event_logger = JsonlMetricLogger(job_dir / "metrics.jsonl")
            event_logger.log(
                {
                    "event": "job_start_or_resume",
                    "job_id": spec.job_id,
                    "public_model": spec.public_model,
                    "seed": spec.seed,
                    "completed_tasks": completed_tasks,
                    "requested_cycles": config.experiment.cycles,
                    "resolved_experiment_sha256": spec.resolved_sha256,
                }
            )
        total_tasks = len(continual.tasks)
        while completed_tasks < total_tasks:
            target_boundary = min(
                ((completed_tasks // len(PUBLIC_LANGUAGE_ORDER)) + 1)
                * len(PUBLIC_LANGUAGE_ORDER),
                total_tasks,
            )
            cycle_index = target_boundary // len(PUBLIC_LANGUAGE_ORDER) - 1
            result_root = job_dir / "internal" / "results"
            result_path, stage_attempt = _next_attempt_path(
                result_root, f"continual-cycle-{cycle_index + 1:04d}"
            )
            result = _run_continual_stage(
                controller,
                internal_config_path=internal_path,
                checkpoint=checkpoint,
                stop_after_task_boundaries=target_boundary,
                result_path=result_path,
                world_size=world_size,
                rendezvous_port=rendezvous_port + cycle_index * 4,
                environment=environment,
            )
            checkpoint = Path(result["checkpoint_path"]).resolve()
            checkpoint_payload = load_checkpoint(checkpoint)
            state = checkpoint_payload["trainer_state"]
            if (
                state["phase"] != "task_boundary"
                or state["next_task_index"] != target_boundary
            ):
                raise ValueError("Continual stage did not reach its cycle boundary")
            completed_tasks = target_boundary
            source_sha = sha256_file(checkpoint)
            if config.probe.enabled:
                probe_result = _run_or_resume_probe(
                    controller,
                    config=config,
                    spec=spec,
                    cycle_index=cycle_index,
                    source_checkpoint=checkpoint,
                    source_sha256=source_sha,
                    world_size=world_size,
                    rendezvous_port=rendezvous_port + cycle_index * 4 + 1,
                    environment=environment,
                )
                probe_summary = _probe_summary(
                    probe_result,
                    public_model=spec.public_model,
                    cycle_index=cycle_index,
                )
                probe_summary["probe_results_path"] = str(
                    job_dir
                    / "probes"
                    / f"cycle-{cycle_index + 1:04d}"
                    / "probe_results.json"
                )
                if len(probe_summaries) == cycle_index:
                    probe_summaries.append(probe_summary)
                elif len(probe_summaries) > cycle_index:
                    if probe_summaries[cycle_index] != probe_summary:
                        raise ValueError("Existing cycle probe summary changed")
                else:
                    raise ValueError("Probe summary cycle sequence has a gap")
            experiment_state = {
                "schema_version": 1,
                "launcher_job_identity": {
                    "job_id": spec.job_id,
                    "public_model": spec.public_model,
                    "internal_variant": spec.internal_variant,
                    "seed": spec.seed,
                },
                "scientific_sha256": spec.scientific_sha256,
                "resolved_experiment_sha256": spec.resolved_sha256,
                "requested_horizon_cycles": config.experiment.cycles,
                "completed_cycle_count": cycle_index + 1,
                "data_manifest_identities": resolved["data_contract"],
                "completed_probe_summaries": probe_summaries,
            }
            augmented_path = (
                job_dir
                / "checkpoints"
                / f"cycle-{cycle_index + 1:04d}-complete.pt"
            )
            if augmented_path.is_file():
                existing = load_checkpoint(augmented_path)
                if existing.get("experiment_state") != experiment_state:
                    raise ValueError("Existing augmented cycle checkpoint differs")
                checkpoint = augmented_path
            else:
                checkpoint_string, _ = augment_cycle_checkpoint(
                    checkpoint,
                    output_path=augmented_path,
                    experiment_state=experiment_state,
                )
                checkpoint = Path(checkpoint_string)
            checkpoint_payload = load_checkpoint(checkpoint)
            pointer = pointer_for_checkpoint(
                checkpoint,
                job_dir=job_dir,
                scientific_sha256=spec.scientific_sha256,
                resolved_experiment_sha256=spec.resolved_sha256,
                requested_horizon_cycles=config.experiment.cycles,
            )
            write_latest_pointer(job_dir, pointer)
            _record_cycle_tensorboard(
                config, job_dir, probe_summaries
            )
            if event_logger is None:
                event_logger = JsonlMetricLogger(job_dir / "metrics.jsonl")
            event_logger.log(
                {
                    "event": "cycle_complete",
                    "cycle_index": cycle_index,
                    "completed_tasks": completed_tasks,
                    "checkpoint_path": str(checkpoint),
                    "checkpoint_sha256": pointer["checkpoint_sha256"],
                    "stage_attempt": stage_attempt,
                }
            )

        assert checkpoint is not None and checkpoint_payload is not None
        summary = _summary(
            spec=spec,
            config=config,
            checkpoint_path=checkpoint,
            payload=checkpoint_payload,
            probes=probe_summaries,
            metadata=metadata,
            started=started,
        )
        atomic_write_json(job_dir / "summary.json", summary)
        metadata["status"] = "complete"
        metadata["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_write_json(metadata_path, metadata)
        if event_logger is None:
            event_logger = JsonlMetricLogger(job_dir / "metrics.jsonl")
        event_logger.log(
            {
                "event": "job_complete",
                "job_id": spec.job_id,
                "final_checkpoint_path": str(checkpoint),
                "final_checkpoint_sha256": sha256_file(checkpoint),
            }
        )
        return summary
    except BaseException as exc:
        metadata["status"] = "failed"
        metadata["failure_history"].append(
            {
                "at_utc": datetime.now(timezone.utc).isoformat(),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
        )
        atomic_write_json(metadata_path, metadata)
        atomic_write_json(
            job_dir / "summary.json",
            {
                "summary_schema_version": 1,
                "status": "failed",
                "model": spec.public_model,
                "seed": spec.seed,
                "cycles_requested": config.experiment.cycles,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "failure_history": metadata["failure_history"],
                "resume_history": metadata["resume_history"],
            },
        )
        if (job_dir / "metrics.jsonl").is_file():
            if event_logger is None:
                event_logger = JsonlMetricLogger(job_dir / "metrics.jsonl")
            event_logger.log(
                {
                    "event": "job_failed",
                    "job_id": spec.job_id,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
        raise
    finally:
        if event_logger is not None:
            event_logger.close()
        signal.signal(signal.SIGINT, old_sigint)
        signal.signal(signal.SIGTERM, old_sigterm)
