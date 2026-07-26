from __future__ import annotations

import argparse
from importlib.util import find_spec

from lm_cl.cli._common import cli_entry, print_json
from lm_cl.launcher.config import load_launcher_config
from lm_cl.launcher.data import resolve_data_contract
from lm_cl.launcher.jobs import expand_job_specs
from lm_cl.launcher.scheduler import (
    LocalJobScheduler,
    allocate_job_slots,
    preflight_launch,
    write_job_configurations,
    write_launcher_summaries,
)


def _csv_strings(value: str | None) -> list[str] | None:
    if value is None:
        return None
    result = [item.strip() for item in value.split(",") if item.strip()]
    if not result:
        raise ValueError("Comma-separated override must not be empty")
    return result


def _csv_ints(value: str | None) -> list[int] | None:
    values = _csv_strings(value)
    return None if values is None else [int(item) for item in values]


def command() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Launch complete Transformer/FastMem continual experiments by model×seed"
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--name", default=None)
    parser.add_argument("--models", default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--cycles", type=int, default=None)
    parser.add_argument("--tokens-per-task", type=int, default=None)
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--jobs-per-gpu", type=int, default=None)
    parser.add_argument("--gpus-per-job", type=int, default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--resume", choices=["never", "auto", "required"])
    probe_group = parser.add_mutually_exclusive_group()
    probe_group.add_argument("--probe", dest="probe_enabled", action="store_true")
    probe_group.add_argument(
        "--no-probe", dest="probe_enabled", action="store_false"
    )
    parser.set_defaults(probe_enabled=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--manifest-only-preflight",
        action="store_true",
        help="skip packed shard rereads; intended only after a recorded full validation",
    )
    args = parser.parse_args()
    overrides = {
        "name": args.name,
        "models": _csv_strings(args.models),
        "seeds": _csv_ints(args.seeds),
        "cycles": args.cycles,
        "tokens_per_task": args.tokens_per_task,
        "gpu_ids": _csv_ints(args.gpus),
        "jobs_per_gpu": args.jobs_per_gpu,
        "gpus_per_job": args.gpus_per_job,
        "output_root": args.output_root,
        "precision": args.precision,
        "resume": args.resume,
        "probe_enabled": args.probe_enabled,
    }
    config = load_launcher_config(args.config, overrides=overrides)
    if config.tracking.tensorboard and find_spec("tensorboard") is None:
        raise ImportError(
            "tracking.tensorboard=true requires: python -m pip install 'lm-cl[tracking]'"
        )
    data_contract = resolve_data_contract(
        config,
        full_checksum_validation=not args.manifest_only_preflight,
    )
    jobs = expand_job_specs(config, data_contract)
    assignments = allocate_job_slots(config, jobs)
    preflight = preflight_launch(config, jobs, assignments)
    public_preflight = {
        key: value for key, value in preflight.items() if key != "old_resolved"
    }
    if args.dry_run:
        print_json(
            {
                "status": "dry_run",
                "child_processes_started": 0,
                "preflight": public_preflight,
            }
        )
        return
    write_job_configurations(jobs, preflight)
    scheduler = LocalJobScheduler(config, jobs, assignments)
    results = scheduler.run()
    summary_path, csv_path = write_launcher_summaries(config, results)
    failed = [item for item in results if item.get("status") != "complete"]
    if failed:
        raise RuntimeError(f"{len(failed)} launcher jobs failed")
    print_json(
        {
            "status": "complete",
            "job_count": len(results),
            "summary_json": str(summary_path),
            "summary_csv": str(csv_path),
        }
    )


def main() -> None:
    cli_entry(command)


if __name__ == "__main__":
    main()
