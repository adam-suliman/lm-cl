from __future__ import annotations

import argparse
from pathlib import Path

from lm_cl.cli._common import cli_entry, print_json
from lm_cl.cli._continual_runtime import run_continual
from lm_cl.config import load_continual_config
from lm_cl.data.storage import atomic_write_json


def command() -> None:
    parser = argparse.ArgumentParser(
        description="Resume a continual checkpoint, optionally with Phase 6 DDP"
    )
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument(
        "--stop-after-global-logical-batches",
        type=int,
        default=None,
    )
    parser.add_argument("--result-json", default=None)
    parser.add_argument(
        "--stop-after-task-boundaries",
        type=int,
        default=None,
        help=(
            "Return after this absolute number of completed task boundaries; "
            "it must exceed the checkpoint's completed-boundary count"
        ),
    )
    args = parser.parse_args()
    result, should_print = run_continual(
        load_continual_config(args.config),
        resume_checkpoint=args.checkpoint,
        stop_after_global_logical_batches=(
            args.stop_after_global_logical_batches
        ),
        stop_after_task_boundaries=args.stop_after_task_boundaries,
    )
    if should_print:
        payload = {
            "status": result.status,
            "checkpoint_path": result.checkpoint_path,
            "checkpoint_sha256": result.checkpoint_sha256,
            "trainer_state": result.state,
        }
        if args.result_json is not None:
            atomic_write_json(Path(args.result_json).resolve(), payload)
        print_json(payload)


def main() -> None:
    cli_entry(command)


if __name__ == "__main__":
    main()
