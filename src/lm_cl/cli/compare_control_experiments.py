from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from lm_cl.analysis.controls import build_control_comparison
from lm_cl.cli._common import cli_entry, print_json
from lm_cl.data.storage import atomic_write_json


def command() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and compare the four 5M continual control arms"
    )
    parser.add_argument("--transformer", required=True)
    parser.add_argument("--backbone-matched-k", required=True)
    parser.add_argument("--fastmem-rmt-zero", required=True)
    parser.add_argument("--fastmem-rmt", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()
    report = build_control_comparison(
        transformer=args.transformer,
        backbone_matched_k=args.backbone_matched_k,
        fastmem_rmt_zero=args.fastmem_rmt_zero,
        fastmem_rmt=args.fastmem_rmt,
    )
    output_json = Path(args.output_json).expanduser().resolve()
    output_csv = Path(args.output_csv).expanduser().resolve()
    atomic_write_json(output_json, report)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model",
        "seed",
        "cycles",
        "total_logical_batches",
        "total_slow_updates",
        "total_fast_updates",
        "final_average_forgetting_ce",
        "final_prior_language_forgetting_ce",
        "probe_primary_normalized_auc_by_cycle",
        "probe_primary_final_ce_by_cycle",
        "probe_normalized_auc_by_mode",
        "probe_final_ce_by_mode",
        "reset_carried_max_abs_ce_difference_by_cycle",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for model, metrics in report["metrics"].items():
            row = {"model": model, **metrics}
            for field in (
                "probe_primary_normalized_auc_by_cycle",
                "probe_primary_final_ce_by_cycle",
                "probe_normalized_auc_by_mode",
                "probe_final_ce_by_mode",
                "reset_carried_max_abs_ce_difference_by_cycle",
            ):
                row[field] = json.dumps(row[field], sort_keys=True)
            writer.writerow({field: row[field] for field in fields})
    print_json(
        {
            "status": "valid",
            "output_json": str(output_json),
            "output_csv": str(output_csv),
            "warnings": report["warnings"],
        }
    )


def main() -> None:
    cli_entry(command)


if __name__ == "__main__":
    main()
