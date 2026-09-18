from __future__ import annotations

import argparse

from lm_cl.cli._common import cli_entry, print_json, write_json
from lm_cl.cli.inspect_probe_results import load_probe_results
from lm_cl.metrics import compare_probe_results


def command() -> None:
    parser = argparse.ArgumentParser(
        description="Compare strictly paired Phase 7 probes"
    )
    parser.add_argument("left_results")
    parser.add_argument("right_results")
    parser.add_argument("--output-report")
    args = parser.parse_args()
    report = compare_probe_results(
        load_probe_results(args.left_results),
        load_probe_results(args.right_results),
    )
    if args.output_report:
        write_json(args.output_report, report)
    print_json(report)


def main() -> None:
    cli_entry(command)


if __name__ == "__main__":
    main()
