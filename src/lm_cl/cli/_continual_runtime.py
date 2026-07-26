from __future__ import annotations

import os
from pathlib import Path

from lm_cl.config import ContinualExperimentConfig
from lm_cl.training import (
    ContinualTrainer,
    DistributedContext,
    DistributedContinualTrainer,
    TrainingResult,
)


def run_continual(
    config: ContinualExperimentConfig,
    *,
    resume_checkpoint: str | Path | None = None,
    stop_after_global_logical_batches: int | None = None,
    stop_after_task_boundaries: int | None = None,
) -> tuple[TrainingResult, bool]:
    """Dispatch a validated run without changing legacy single-process use."""
    if config.distributed is None:
        if int(os.environ.get("WORLD_SIZE", "1")) != 1:
            raise RuntimeError(
                "torchrun world size exceeds one but the configuration has "
                "no distributed section"
            )
        trainer = ContinualTrainer(config)
        return (
            trainer.run(
                resume_checkpoint=resume_checkpoint,
                stop_after_global_logical_batches=(
                    stop_after_global_logical_batches
                ),
                stop_after_task_boundaries=stop_after_task_boundaries,
            ),
            True,
        )

    context = DistributedContext.initialize(
        config.distributed,
        runtime_device=config.runtime.device,
    )
    trainer: DistributedContinualTrainer | None = None
    try:
        trainer = DistributedContinualTrainer(config, context)
        result = trainer.run(
            resume_checkpoint=resume_checkpoint,
            stop_after_global_logical_batches=(
                stop_after_global_logical_batches
            ),
            stop_after_task_boundaries=stop_after_task_boundaries,
        )
        return result, context.is_primary
    except BaseException as exc:
        if (
            trainer is not None
            and trainer.logger is not None
            and not trainer.logger.closed
        ):
            trainer._log(
                "distributed_error",
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
        raise
    finally:
        context.close()
