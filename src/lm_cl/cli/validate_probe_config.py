from __future__ import annotations

import argparse

from lm_cl.cli._common import cli_entry, print_json
from lm_cl.config import load_probe_config, save_probe_config
from lm_cl.training import validate_probe_source_checkpoint


def command() -> None:
    parser = argparse.ArgumentParser(description="Validate a Phase 7 probe YAML")
    parser.add_argument("config")
    parser.add_argument("--output")
    parser.add_argument(
        "--require-source-checkpoint",
        action="store_true",
    )
    args = parser.parse_args()
    config = load_probe_config(args.config)
    source = None
    if args.require_source_checkpoint:
        _, source = validate_probe_source_checkpoint(config)
    if args.output:
        save_probe_config(config, args.output)
    print_json(
        {
            "status": "valid",
            "config": config.to_dict(),
            "planned_logical_batches": config.planned_logical_batches,
            "planned_slow_steps": config.planned_slow_steps,
            "effective_fast_lr": config.effective_fast_lr,
            "source_checkpoint": source,
        }
    )


def main() -> None:
    cli_entry(command)


if __name__ == "__main__":
    main()
