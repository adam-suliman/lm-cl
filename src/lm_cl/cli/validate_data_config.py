from __future__ import annotations

import argparse

from lm_cl.cli._common import cli_entry, print_json
from lm_cl.config import load_data_pipeline_config
from lm_cl.data.materialize import dry_run_materialization


def run() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a strict Phase 3 data-pipeline YAML"
    )
    parser.add_argument("config")
    parser.add_argument(
        "--require-access-ready",
        action="store_true",
        help="also require immutable dataset/tokenizer/configuration pins",
    )
    args = parser.parse_args()
    config = load_data_pipeline_config(args.config)
    if args.require_access_ready:
        config.require_access_ready()
    print_json(
        {
            "status": "valid",
            "name": config.name,
            "mode": config.mode,
            "access_ready": (
                _access_ready(config)
            ),
            "dry_run": dry_run_materialization(config),
        }
    )


def _access_ready(config: object) -> bool:
    try:
        config.require_access_ready()  # type: ignore[attr-defined]
        return True
    except ValueError:
        return False


def main() -> None:
    cli_entry(run)


if __name__ == "__main__":
    main()

