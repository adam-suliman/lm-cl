from __future__ import annotations

import argparse

from lm_cl.cli._common import cli_entry, print_json
from lm_cl.launcher.runner import run_resolved_job


def command() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run one complete public model×seed continual trajectory, including probes"
        )
    )
    parser.add_argument("--resolved-config", required=True)
    parser.add_argument("--rendezvous-port", type=int, required=True)
    parser.add_argument(
        "--retry-resume",
        action="store_true",
        help="retry this unchanged resolved job from its latest validated state",
    )
    args = parser.parse_args()
    print_json(
        run_resolved_job(
            args.resolved_config,
            rendezvous_port=args.rendezvous_port,
            retry_resume=args.retry_resume,
        )
    )


def main() -> None:
    cli_entry(command)


if __name__ == "__main__":
    main()
