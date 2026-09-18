from __future__ import annotations

import argparse

from lm_cl.cli._common import cli_entry, print_json, write_json
from lm_cl.cli.inspect_probe_results import load_probe_results
from lm_cl.metrics import compute_probe_auc_report


def command() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute Phase 7 AUC from a saved full curve"
    )
    parser.add_argument("results")
    parser.add_argument("--output-report")
    args = parser.parse_args()
    result = load_probe_results(args.results)
    schedule = result["pairing_identity"]["evaluation_schedule"]
    report = compute_probe_auc_report(
        result["curve_records"],
        early_milestones=schedule["early_milestones"],
        policy=result["auc_report"]["policy"],
    )
    if report != result["auc_report"]:
        raise ValueError("Recomputed AUC differs from saved AUC report")
    if args.output_report:
        write_json(args.output_report, report)
    print_json(report)


def main() -> None:
    cli_entry(command)


if __name__ == "__main__":
    main()
