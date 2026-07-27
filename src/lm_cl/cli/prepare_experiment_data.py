from __future__ import annotations

import argparse

from lm_cl.cli._common import cli_entry, print_json
from lm_cl.launcher.config import load_launcher_config
from lm_cl.launcher.data import prepare_or_validate_data


def command() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate missing or checksum-validate the fresh cycle×language "
            "packed-manifest matrix for a public experiment"
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--cycles", type=int, default=None)
    parser.add_argument("--tokens-per-task", type=int, default=None)
    parser.add_argument(
        "--parallel-languages",
        type=int,
        default=1,
        help=(
            "materialize independent language lanes concurrently, then "
            "require a checked global overlap-registry merge"
        ),
    )
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="validate identities without rereading every packed shard",
    )
    args = parser.parse_args()
    overrides = {
        "cycles": args.cycles,
        "tokens_per_task": args.tokens_per_task,
    }
    config = load_launcher_config(args.config, overrides=overrides)
    result = prepare_or_validate_data(
        config,
        full_checksum_validation=not args.manifest_only,
        parallel_languages=args.parallel_languages,
    )
    print_json(
        {
            "status": "ready",
            "experiment": config.experiment.name,
            "cycles": config.experiment.cycles,
            "data_contract": result,
        }
    )


def main() -> None:
    cli_entry(command)


if __name__ == "__main__":
    main()
