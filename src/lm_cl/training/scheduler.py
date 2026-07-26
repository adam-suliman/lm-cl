from __future__ import annotations

from typing import Any

import torch


class LinearWarmupConstantScheduler:
    """Task-local warm-up in optimizer-step units followed by constant LR."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        peak_lr: float,
        planned_steps: int,
        warmup_steps: int,
    ):
        if peak_lr <= 0 or planned_steps <= 0:
            raise ValueError("Scheduler peak LR and planned steps must be positive")
        if warmup_steps <= 0 or warmup_steps > planned_steps:
            raise ValueError("Scheduler warmup steps must be in [1, planned_steps]")
        self.optimizer = optimizer
        self.peak_lr = peak_lr
        self.planned_steps = planned_steps
        self.warmup_steps = warmup_steps
        self.completed_steps = 0
        self._set_lr(self.lr_for_step(1))

    def lr_for_step(self, one_based_step: int) -> float:
        if one_based_step <= 0:
            raise ValueError("Scheduler step index must be positive")
        return self.peak_lr * min(one_based_step / self.warmup_steps, 1.0)

    @property
    def current_lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def _set_lr(self, learning_rate: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate

    def step(self) -> None:
        if self.completed_steps >= self.planned_steps:
            raise RuntimeError("Scheduler advanced past its planned task steps")
        self.completed_steps += 1
        next_step = min(self.completed_steps + 1, self.planned_steps)
        self._set_lr(self.lr_for_step(next_step))

    def state_dict(self) -> dict[str, Any]:
        return {
            "kind": "linear_warmup_constant_v1",
            "peak_lr": self.peak_lr,
            "planned_steps": self.planned_steps,
            "warmup_steps": self.warmup_steps,
            "completed_steps": self.completed_steps,
            "current_lr": self.current_lr,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        expected = {
            "kind": "linear_warmup_constant_v1",
            "peak_lr": self.peak_lr,
            "planned_steps": self.planned_steps,
            "warmup_steps": self.warmup_steps,
        }
        mismatches = [
            f"{key}: checkpoint={state.get(key)!r}, configured={value!r}"
            for key, value in expected.items()
            if state.get(key) != value
        ]
        if mismatches:
            raise ValueError(
                "Scheduler checkpoint mismatch: " + "; ".join(mismatches)
            )
        completed = state.get("completed_steps")
        if (
            not isinstance(completed, int)
            or completed < 0
            or completed > self.planned_steps
        ):
            raise ValueError("Scheduler completed-step count is invalid")
        self.completed_steps = completed
        expected_lr = self.lr_for_step(
            min(self.completed_steps + 1, self.planned_steps)
        )
        if abs(float(state.get("current_lr", -1.0)) - expected_lr) > 1e-15:
            raise ValueError("Scheduler current LR is inconsistent")
        self._set_lr(expected_lr)
