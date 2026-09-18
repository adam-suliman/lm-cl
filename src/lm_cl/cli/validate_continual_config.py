from __future__ import annotations

import argparse

from lm_cl.cli._common import cli_entry, print_json
from lm_cl.config import load_continual_config, save_continual_config


def command() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a Phase 4/5 continual-training YAML"
    )
    parser.add_argument("config")
    parser.add_argument("--output")
    args = parser.parse_args()
    config = load_continual_config(args.config)
    if args.output:
        save_continual_config(config, args.output)
    print_json(
        {
            "status": "valid",
            "run_name": config.run_name,
            "variant": config.variant.name,
            "model": config.model.name,
            "task_count": len(config.tasks),
        }
    )


def main() -> None:
    cli_entry(command)


if __name__ == "__main__":
    main()
