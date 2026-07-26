from __future__ import annotations

import argparse
from pathlib import Path

from lm_cl.cli._common import cli_entry, print_json
from lm_cl.cli._probe_runtime import run_probe_runtime
from lm_cl.config import load_probe_config
from lm_cl.data.storage import atomic_write_json


def command() -> None:
    parser = argparse.ArgumentParser(description="Resume a Phase 7 probe")
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument(
        "--stop-after-global-logical-batches",
        type=int,
    )
    parser.add_argument("--result-json", default=None)
    args = parser.parse_args()
    result, primary = run_probe_runtime(
        load_probe_config(args.config),
        resume_checkpoint=args.checkpoint,
        stop_after_global_logical_batches=(
            args.stop_after_global_logical_batches
        ),
    )
    if primary:
        payload = {
            "status": result.status,
            "checkpoint_path": result.checkpoint_path,
            "checkpoint_sha256": result.checkpoint_sha256,
            "probe_state": result.state,
        }
        if args.result_json is not None:
            atomic_write_json(Path(args.result_json).resolve(), payload)
        print_json(payload)


def main() -> None:
    cli_entry(command)


if __name__ == "__main__":
    main()
