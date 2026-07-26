from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from lm_cl.metrics.tensorboard import (
    TensorBoardTracker,
    log_record_to_tensorboard,
)


class JsonlMetricLogger:
    def __init__(
        self,
        path: str | Path,
        *,
        tensorboard_dir: str | Path | None = None,
        tensorboard_flush_seconds: int = 30,
        tensorboard_log_every_batches: int = 1,
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.closed = False
        self.tensorboard_log_every_batches = tensorboard_log_every_batches
        self.tensorboard = (
            None
            if tensorboard_dir is None
            else TensorBoardTracker(
                tensorboard_dir,
                flush_seconds=tensorboard_flush_seconds,
            )
        )

    def log(self, metrics: dict[str, Any]) -> None:
        if self.closed:
            raise RuntimeError("Metric logger is closed")
        record = _json_safe(metrics)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
        if self.tensorboard is not None:
            log_record_to_tensorboard(
                self.tensorboard,
                record,
                log_every_batches=self.tensorboard_log_every_batches,
            )

    def close(self) -> None:
        if self.closed:
            return
        if self.tensorboard is not None:
            self.tensorboard.close()
        self.closed = True


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"Metric value is not JSON serializable: {type(value).__name__}")
