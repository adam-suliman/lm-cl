from __future__ import annotations

import csv
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from lm_cl.environment import inspect_environment
from lm_cl.launcher.config import save_yaml
from lm_cl.launcher.jobs import JobSpec, _model_config
from lm_cl.launcher.schema import LauncherConfig, PUBLIC_LANGUAGE_ORDER
from lm_cl.launcher.state import (
    validate_horizon_extension,
    validate_latest_pointer,
)
from lm_cl.metrics import JsonlMetricLogger


@dataclass(frozen=True)
class JobAssignment:
    job_index: int
    job_id: str
    gpu_ids: list[int]
    slot_index: int
    rendezvous_port: int
    output_dir: str
    command: list[str]
    launch_state: str


def allocate_job_slots(
    config: LauncherConfig, jobs: list[JobSpec]
) -> list[JobAssignment]:
    launcher = config.launcher
    if config.training.device == "cpu":
        slots = [([], index) for index in range(launcher.max_parallel_jobs)]
    else:
        groups = [
            launcher.gpu_ids[index : index + launcher.gpus_per_job]
            for index in range(0, len(launcher.gpu_ids), launcher.gpus_per_job)
        ]
        slots = []
        for replica in range(launcher.jobs_per_gpu):
            for group_index, group in enumerate(groups):
                slots.append((list(group), replica * len(groups) + group_index))
    if not slots:
        raise ValueError("No launcher job slots are available")
    assignments = []
    port_stride = config.experiment.cycles * 4 + 4
    for index, job in enumerate(jobs):
        gpu_ids, slot_index = slots[index % len(slots)]
        port = launcher.rendezvous_port_base + index * port_stride
        if port + config.experiment.cycles * 4 + 1 > 65535:
            raise ValueError("Configured rendezvous port range exceeds 65535")
        resolved_path = Path(job.output_dir) / "resolved_experiment.yaml"
        command = [
            sys.executable,
            "-m",
            "lm_cl.cli.run_experiment_job",
            "--resolved-config",
            str(resolved_path),
            "--rendezvous-port",
            str(port),
        ]
        pointer_path = Path(job.output_dir) / "latest_checkpoint.json"
        launch_state = "resumable" if pointer_path.is_file() else "fresh"
        assignments.append(
            JobAssignment(
                job_index=index,
                job_id=job.job_id,
                gpu_ids=gpu_ids,
                slot_index=slot_index,
                rendezvous_port=port,
                output_dir=job.output_dir,
                command=command,
                launch_state=launch_state,
            )
        )
    return assignments


def _existing_resolved(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Existing resolved experiment is invalid: {path}")
    return value


def _checkpoint_estimate(config: LauncherConfig, job_count: int) -> dict[str, int]:
    model = _model_config(config)
    per_checkpoint = (
        model.expected_total_parameters
        * config.launcher.checkpoint_bytes_per_parameter
        + config.launcher.checkpoint_fixed_overhead_bytes
    )
    continual_checkpoints = config.experiment.cycles * (
        len(PUBLIC_LANGUAGE_ORDER) + 1
    )
    probe_checkpoints = (
        config.experiment.cycles if config.probe.enabled else 0
    )
    per_job_count = continual_checkpoints + probe_checkpoints
    return {
        "estimated_bytes_per_checkpoint": per_checkpoint,
        "estimated_checkpoints_per_job": per_job_count,
        "estimated_bytes_per_job": per_checkpoint * per_job_count,
        "estimated_total_checkpoint_bytes": (
            per_checkpoint * per_job_count * job_count
        ),
    }


def preflight_launch(
    config: LauncherConfig,
    jobs: list[JobSpec],
    assignments: list[JobAssignment],
) -> dict[str, Any]:
    environment = inspect_environment()
    if config.training.device != "cpu":
        available = {gpu["index"] for gpu in environment["gpus"]}
        missing = sorted(set(config.launcher.gpu_ids) - available)
        if missing:
            raise ValueError(f"Requested GPU IDs do not exist: {missing}")
        precision_key = {
            "fp32": "fp32",
            "fp16": "fp16_cuda_native",
            "bf16": "bf16_cuda_native",
        }[config.training.precision]
        if not environment["supported_precision_modes"][precision_key]:
            raise ValueError(
                f"Requested precision {config.training.precision} is unsupported"
            )
        if (
            config.training.global_batch_sequences
            < config.launcher.gpus_per_job
        ):
            raise ValueError(
                "Global batch must contain at least one sequence per GPU"
            )
    old_resolved: dict[str, dict[str, Any] | None] = {}
    for job in jobs:
        output = Path(job.output_dir)
        existing_path = output / "resolved_experiment.yaml"
        existing = _existing_resolved(existing_path)
        old_resolved[job.job_id] = existing
        pointer_path = output / "latest_checkpoint.json"
        if config.experiment.resume == "never":
            if existing is not None or pointer_path.exists():
                raise FileExistsError(
                    f"resume=never refuses existing job: {output}"
                )
        elif config.experiment.resume == "required" and not pointer_path.is_file():
            raise FileNotFoundError(
                f"resume=required lacks latest_checkpoint.json: {output}"
            )
        if pointer_path.is_file():
            pointer, _, _ = validate_latest_pointer(
                pointer_path,
                expected_job_dir=output,
                expected_scientific_sha256=job.scientific_sha256,
            )
            if existing is None:
                raise ValueError("Latest pointer exists without resolved experiment")
            validate_horizon_extension(
                existing,
                job.resolved_experiment,
                completed_cycles=pointer["completed_cycle_count"],
            )
        elif existing is not None:
            if config.experiment.resume == "required":
                raise ValueError("Required resume state is incomplete")
            if existing.get("scientific_sha256") != job.scientific_sha256:
                raise ValueError(
                    "Existing incomplete output has another scientific identity"
                )
    estimates = _checkpoint_estimate(config, len(jobs))
    output_root = Path(config.experiment.output_root)
    probe_path = output_root
    while not probe_path.exists() and probe_path != probe_path.parent:
        probe_path = probe_path.parent
    disk = shutil.disk_usage(probe_path)
    required = (
        estimates["estimated_total_checkpoint_bytes"]
        + config.launcher.disk_free_floor_bytes
    )
    if disk.free < required:
        raise ValueError(
            "Disk admission failed: "
            f"free={disk.free}, required={required}, "
            f"estimated_checkpoints={estimates['estimated_total_checkpoint_bytes']}"
        )
    return {
        "status": "ready",
        "environment": environment,
        "assignments": [asdict(item) for item in assignments],
        "manifest_count_per_job": (
            config.experiment.cycles * len(PUBLIC_LANGUAGE_ORDER)
        ),
        "disk": {
            "path": str(probe_path),
            "free_bytes": disk.free,
            "required_bytes": required,
            "free_floor_bytes": config.launcher.disk_free_floor_bytes,
            **estimates,
        },
        "old_resolved": old_resolved,
    }


def _atomic_save_yaml(value: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    save_yaml(value, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def write_job_configurations(
    jobs: list[JobSpec], preflight: dict[str, Any]
) -> None:
    old_resolved = preflight["old_resolved"]
    for job in jobs:
        output = Path(job.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        old = old_resolved[job.job_id]
        path = output / "resolved_experiment.yaml"
        if old is not None and old != job.resolved_experiment:
            old_horizon = old["requested_horizon_cycles"]
            old_sha = old["resolved_experiment_sha256"]
            history_path = (
                output
                / "internal"
                / "resolved-history"
                / f"horizon-{old_horizon:04d}-{old_sha[:12]}.yaml"
            )
            if not history_path.exists():
                _atomic_save_yaml(old, history_path)
        _atomic_save_yaml(job.resolved_experiment, path)


@dataclass
class _Running:
    assignment: JobAssignment
    process: subprocess.Popen[bytes]
    stdout_handle: Any
    stderr_handle: Any
    attempt: int


class LocalJobScheduler:
    def __init__(
        self,
        config: LauncherConfig,
        jobs: list[JobSpec],
        assignments: list[JobAssignment],
    ):
        self.config = config
        self.jobs = jobs
        self.assignments = assignments
        self.running: dict[int, _Running] = {}
        self.interrupted = False

    def _signal(self, signum: int, _frame: Any) -> None:
        self.interrupted = True
        for item in list(self.running.values()):
            if item.process.poll() is None:
                os.killpg(item.process.pid, signum)

    def _start(self, assignment: JobAssignment, attempt: int) -> _Running:
        output = Path(assignment.output_dir)
        stdout_handle = (output / "stdout.log").open("ab")
        stderr_handle = (output / "stderr.log").open("ab")
        environment = dict(os.environ)
        if assignment.gpu_ids:
            environment["CUDA_VISIBLE_DEVICES"] = ",".join(
                str(value) for value in assignment.gpu_ids
            )
        else:
            environment.pop("CUDA_VISIBLE_DEVICES", None)
        command = list(assignment.command)
        if attempt > 0:
            command.append("--retry-resume")
        process = subprocess.Popen(
            command,
            env=environment,
            stdout=stdout_handle,
            stderr=stderr_handle,
            start_new_session=True,
        )
        return _Running(
            assignment=assignment,
            process=process,
            stdout_handle=stdout_handle,
            stderr_handle=stderr_handle,
            attempt=attempt,
        )

    def run(self) -> list[dict[str, Any]]:
        experiment_dir = (
            Path(self.config.experiment.output_root)
            / self.config.experiment.name
        )
        index_logger = JsonlMetricLogger(experiment_dir / "jobs.jsonl")
        queue = [(assignment, 0) for assignment in self.assignments]
        results: list[dict[str, Any]] = []
        old_sigint = signal.signal(signal.SIGINT, self._signal)
        old_sigterm = signal.signal(signal.SIGTERM, self._signal)
        max_running = min(
            self.config.launcher.max_parallel_jobs,
            max(1, len({item.slot_index for item in self.assignments})),
        )
        try:
            while queue or self.running:
                used_slots = {
                    item.assignment.slot_index for item in self.running.values()
                }
                queue_index = 0
                while len(self.running) < max_running and queue_index < len(queue):
                    assignment, attempt = queue[queue_index]
                    if assignment.slot_index in used_slots:
                        queue_index += 1
                        continue
                    queue.pop(queue_index)
                    running = self._start(assignment, attempt)
                    self.running[running.process.pid] = running
                    used_slots.add(assignment.slot_index)
                    index_logger.log(
                        {
                            "event": "job_started",
                            "job_id": assignment.job_id,
                            "attempt": attempt,
                            "gpu_ids": assignment.gpu_ids,
                            "pid": running.process.pid,
                            "at_utc": datetime.now(timezone.utc).isoformat(),
                        }
                    )
                if not self.running:
                    if queue:
                        raise RuntimeError("Scheduler queue cannot make progress")
                    break
                finished_pid = None
                while finished_pid is None and not self.interrupted:
                    for pid, item in self.running.items():
                        if item.process.poll() is not None:
                            finished_pid = pid
                            break
                    if finished_pid is None:
                        time.sleep(0.1)
                if self.interrupted:
                    raise KeyboardInterrupt("Launcher interrupted")
                assert finished_pid is not None
                item = self.running.pop(finished_pid)
                status = item.process.returncode
                item.stdout_handle.close()
                item.stderr_handle.close()
                if status == 0:
                    summary_path = Path(item.assignment.output_dir) / "summary.json"
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    results.append(summary)
                    index_logger.log(
                        {
                            "event": "job_completed",
                            "job_id": item.assignment.job_id,
                            "attempt": item.attempt,
                            "status": status,
                        }
                    )
                elif item.attempt < self.config.launcher.retry_count:
                    queue.append((item.assignment, item.attempt + 1))
                    index_logger.log(
                        {
                            "event": "job_retry_queued",
                            "job_id": item.assignment.job_id,
                            "attempt": item.attempt + 1,
                            "child_status": status,
                        }
                    )
                else:
                    failure = {
                        "status": "failed",
                        "model": item.assignment.job_id.split("-seed-")[0],
                        "job_id": item.assignment.job_id,
                        "child_exit_status": status,
                    }
                    results.append(failure)
                    index_logger.log({"event": "job_failed", **failure})
                    if self.config.launcher.fail_fast:
                        for running in self.running.values():
                            os.killpg(running.process.pid, signal.SIGTERM)
                        raise RuntimeError(
                            f"Job failed: {item.assignment.job_id}"
                        )
            return results
        finally:
            for item in self.running.values():
                if item.process.poll() is None:
                    os.killpg(item.process.pid, signal.SIGTERM)
                item.process.wait()
                item.stdout_handle.close()
                item.stderr_handle.close()
            self.running.clear()
            index_logger.close()
            signal.signal(signal.SIGINT, old_sigint)
            signal.signal(signal.SIGTERM, old_sigterm)


def write_launcher_summaries(
    config: LauncherConfig, results: list[dict[str, Any]]
) -> tuple[Path, Path]:
    experiment_dir = (
        Path(config.experiment.output_root) / config.experiment.name
    )
    summary_path = experiment_dir / "summary.json"
    summary = {
        "summary_schema_version": 1,
        "experiment": config.experiment.name,
        "status": (
            "complete"
            if results and all(item.get("status") == "complete" for item in results)
            else "failed"
        ),
        "job_count": len(results),
        "jobs": results,
    }
    from lm_cl.data.storage import atomic_write_json

    atomic_write_json(summary_path, summary)
    csv_path = experiment_dir / "summary.csv"
    fields = [
        "status",
        "model",
        "seed",
        "cycles_requested",
        "cycles_completed",
        "completed_language_tasks",
        "total_input_tokens",
        "total_target_tokens",
        "total_logical_batches",
        "total_slow_updates",
        "total_fast_updates",
        "final_checkpoint_path",
        "final_checkpoint_sha256",
        "elapsed_seconds",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
        handle.flush()
        os.fsync(handle.fileno())
    return summary_path, csv_path
