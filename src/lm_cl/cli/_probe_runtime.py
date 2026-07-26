from __future__ import annotations

import os
from pathlib import Path

from lm_cl.config import ProbeExperimentConfig
from lm_cl.training import (
    DistributedContext,
    DistributedProbeTrainer,
    ProbeTrainer,
    TrainingResult,
)


def run_probe_runtime(
    config: ProbeExperimentConfig,
    *,
    resume_checkpoint: str | Path | None = None,
    stop_after_global_logical_batches: int | None = None,
) -> tuple[TrainingResult, bool]:
    if config.distributed is None:
        if int(os.environ.get("WORLD_SIZE", "1")) != 1:
            raise RuntimeError(
                "torchrun world size exceeds one without probe DDP config"
            )
        trainer = ProbeTrainer(config)
        return (
            trainer.run(
                resume_checkpoint=resume_checkpoint,
                stop_after_global_logical_batches=(
                    stop_after_global_logical_batches
                ),
            ),
            True,
        )
    context = DistributedContext.initialize(
        config.distributed,
        runtime_device=config.runtime.device,
    )
    trainer: DistributedProbeTrainer | None = None
    try:
        trainer = DistributedProbeTrainer(config, context)
        result = trainer.run(
            resume_checkpoint=resume_checkpoint,
            stop_after_global_logical_batches=(
                stop_after_global_logical_batches
            ),
        )
        return result, context.is_primary
    except BaseException as exc:
        if (
            trainer is not None
            and trainer.logger is not None
            and not trainer.logger.closed
        ):
            trainer._log(
                "distributed_probe_error",
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
        raise
    finally:
        context.close()
