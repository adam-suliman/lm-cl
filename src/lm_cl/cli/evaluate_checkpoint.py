from __future__ import annotations

import argparse

from lm_cl.cli._common import cli_entry, print_json
from lm_cl.config import load_continual_config
from lm_cl.training import evaluate_clean_checkpoint


def command() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a continual checkpoint on a configured source"
    )
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument("--task-index", type=int, required=True)
    args = parser.parse_args()
    print_json(
        evaluate_clean_checkpoint(
            load_continual_config(args.config),
            args.checkpoint,
            task_index=args.task_index,
        )
    )


def main() -> None:
    cli_entry(command)


if __name__ == "__main__":
    main()
