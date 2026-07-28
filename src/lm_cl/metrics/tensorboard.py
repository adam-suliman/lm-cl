from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from lm_cl.data.storage import atomic_write_json


class TensorBoardTracker:
    """SummaryWriter wrapper with an atomic last-committed-step sidecar."""

    STATE_FILENAME = "tensorboard_state.json"

    def __init__(self, log_dir: str | Path, *, flush_seconds: int):
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as exc:
            raise ImportError(
                "TensorBoard tracking requires the optional dependency; install "
                "with: python -m pip install 'lm-cl[tracking]'"
            ) from exc
        self.log_dir = Path(log_dir).expanduser().resolve()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.log_dir / self.STATE_FILENAME
        self.last_steps: dict[str, int] = {}
        if self.state_path.is_file():
            values = json.loads(self.state_path.read_text(encoding="utf-8"))
            if (
                not isinstance(values, dict)
                or values.get("schema_version") != 1
                or not isinstance(values.get("last_steps"), dict)
            ):
                raise ValueError("TensorBoard step sidecar is invalid")
            self.last_steps = {
                str(tag): int(step)
                for tag, step in values["last_steps"].items()
            }
        self.writer = SummaryWriter(
            log_dir=str(self.log_dir),
            flush_secs=flush_seconds,
        )
        self.closed = False

    def _commit(self) -> None:
        atomic_write_json(
            self.state_path,
            {"schema_version": 1, "last_steps": self.last_steps},
        )

    def scalar(self, tag: str, value: float | int, step: int) -> bool:
        if self.closed:
            raise RuntimeError("TensorBoard tracker is closed")
        if step < 0:
            raise ValueError("TensorBoard steps must be non-negative")
        if step <= self.last_steps.get(tag, -1):
            return False
        self.writer.add_scalar(tag, value, global_step=step)
        self.last_steps[tag] = step
        self._commit()
        return True

    def text(self, tag: str, value: str, step: int) -> bool:
        text_tag = f"__text__/{tag}"
        if step <= self.last_steps.get(text_tag, -1):
            return False
        self.writer.add_text(tag, value, global_step=step)
        self.last_steps[text_tag] = step
        self._commit()
        return True

    def flush(self) -> None:
        if not self.closed:
            self.writer.flush()
            self._commit()

    def close(self) -> None:
        if self.closed:
            return
        self.flush()
        self.writer.close()
        self.closed = True


def _primary_probe_mode(record: dict[str, Any]) -> bool:
    mode = record.get("memory_evaluation_mode")
    return mode in {"not_applicable", "carried"}


def log_record_to_tensorboard(
    tracker: TensorBoardTracker,
    record: dict[str, Any],
    *,
    log_every_batches: int,
) -> None:
    event = record.get("event")
    logical_step = int(record.get("global_logical_batches", 0))
    if event == "logical_batch" and logical_step % log_every_batches == 0:
        scalar_fields = {
            "train/loss": record.get("mean_loss"),
            "train/learning_rate": record.get("learning_rate"),
            "train/parameter_norm": record.get("parameter_norm"),
            "train/input_tokens": record.get("global_input_tokens"),
            "train/valid_targets": record.get("global_valid_targets"),
            "train/tokens_per_second": record.get(
                "throughput_input_tokens_per_second"
            ),
            "continual/cycle_index": record.get("cycle_index"),
            "continual/task_index": record.get("task_index"),
            "fastmem/active_memory_norm": record.get("active_memory_norm"),
            "fastmem/fast_gradient_norm": record.get(
                "active_memory_gradient_norm_before_clip"
            ),
            "fastmem/fast_updates": record.get("fast_update_count"),
            "fastmem/m0_norm": record.get("m0_norm"),
        }
        for tag, value in scalar_fields.items():
            if value is not None:
                tracker.scalar(tag, value, logical_step)
    elif event == "optimizer_step":
        value = record.get("gradient_norm")
        if value is not None:
            tracker.scalar("train/slow_gradient_norm", value, logical_step)
        learning_rate = record.get("learning_rate")
        if learning_rate is not None:
            tracker.scalar("train/learning_rate", learning_rate, logical_step)
    elif event in {"resume", "probe_resume"}:
        tracker.text(
            "run/resume_event",
            json.dumps(record, sort_keys=True, default=str),
            logical_step,
        )
    elif event == "probe_evaluation":
        token_step = int(record["cumulative_input_tokens"])
        mode = str(record["memory_evaluation_mode"])
        value = float(record["mean_validation_ce"])
        tracker.scalar(f"probe/validation_ce/{mode}", value, token_step)
        if _primary_probe_mode(record):
            tracker.scalar("probe/validation_ce", value, token_step)
    elif event == "probe_end" and isinstance(record.get("auc_report"), dict):
        curves = record["auc_report"].get("curves", {})
        for mode, values in curves.items():
            step = int(values["final_cumulative_input_tokens"])
            metrics = {
                "normalized_token_auc": values[
                    "primary_normalized_trapezoidal_auc"
                ],
                "raw_token_auc": values["raw_input_token_trapezoidal_auc"],
                "raw_step_auc": values["raw_step_trapezoidal_auc"],
            }
            for name, value in metrics.items():
                tracker.scalar(f"probe/{name}/{mode}", value, step)
                if mode in {"not_applicable", "carried"}:
                    tracker.scalar(f"probe/{name}", value, step)
    elif event == "forgetting_evaluation":
        boundary = int(record["boundary_task_number"])
        tracker.scalar(
            "continual/average_forgetting_from_best_ce",
            float(record["average_forgetting_from_best_ce"]),
            boundary,
        )
        tracker.scalar(
            "continual/average_ce_change_from_first",
            float(record["average_ce_change_from_first"]),
            boundary,
        )
        prior = record.get(
            "average_prior_language_forgetting_from_best_ce"
        )
        if prior is not None:
            tracker.scalar(
                "continual/average_prior_language_forgetting_from_best_ce",
                float(prior),
                boundary,
            )
        for language, values in record.get("languages", {}).items():
            tracker.scalar(
                f"continual/validation_ce/{language}",
                float(values["mean_validation_ce"]),
                boundary,
            )
            tracker.scalar(
                f"continual/forgetting_from_best_ce/{language}",
                float(values["forgetting_from_best_ce"]),
                boundary,
            )
