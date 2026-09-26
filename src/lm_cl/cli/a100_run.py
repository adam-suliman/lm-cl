"""Validated entry point for fresh 5M/12M A100 experiment pairs.

Streaming preparation publishes verified packed blocks while production training
runs. Legacy fully prepared datasets remain available with --data-mode packed.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import os
from pathlib import Path
from typing import Sequence

from lm_cl.cli._common import cli_entry, print_json
from lm_cl.launcher.config import load_launcher_config
from lm_cl.launcher.schema import PUBLIC_MODEL_VARIANTS, resolve_token_budget


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-size", choices=["5m", "12m"], required=True)
    p.add_argument("--action", choices=["plan", "preflight", "prepare", "calibrate", "run"], default="plan")
    p.add_argument("--dry-run", action="store_true", help="read-only preflight; requires a frozen streaming recipe or completed legacy data")
    p.add_argument("--a100-memory-gb", type=int, choices=[40, 80], default=40)
    p.add_argument("--data-mode", choices=["streaming", "packed"], default="streaming")
    p.add_argument("--streaming-settings", help="JSON mapping of validated streaming resource settings")
    p.add_argument("--streaming-schedule", choices=["independent", "alternating"], default="independent",
                   help="independent trajectories or bounded alternating model/seed turns")
    p.add_argument("--streaming-chunk-batches", type=int, default=None,
                   help="maximum global logical batches per alternating turn (default 2048)")
    p.add_argument("--checkpoint-every-batches", type=int, default=0)
    p.add_argument("--checkpoint-retention", choices=["all", "cycle"], default="all",
                   help="retain all checkpoints or only durable cycle/probe sources plus the current recovery point")
    p.add_argument("--data-root", default=os.environ.get("LM_CL_DATA_ROOT"))
    p.add_argument("--output-root", default=os.environ.get("LM_CL_OUTPUT_ROOT"))
    p.add_argument("--cache-root", default=None)
    p.add_argument("--name", default=None)
    p.add_argument("--gpus", default="0")
    p.add_argument("--gpus-per-job", type=int, choices=[1, 2], default=1)
    p.add_argument("--models", default="transformer,fastmem_rmt")
    p.add_argument("--seeds", default="81010")
    p.add_argument("--cycles", type=int, default=5)
    p.add_argument("--tokens-per-task", type=int, default=1_000_000_000)
    p.add_argument("--probe-tokens", type=int, default=1_000_000_000)
    p.add_argument("--probe-training-manifest", help="optional existing completed VI pool; consume only the configured probe prefix")
    p.add_argument("--physical-microbatch-sequences", type=int, default=None)
    p.add_argument("--resume", choices=["never", "auto", "required"], default="never")
    p.add_argument("--parallel-languages", type=int, default=1)
    p.add_argument("--calibration-config", help="validated production continual config using verified packed blocks or shards")
    p.add_argument("--calibration-output", help="new dedicated directory for bounded calibration")
    p.add_argument("--calibration-batches", type=int, default=8)
    p.add_argument("--calibration-max-seconds", type=int, default=1800)
    return p


def _csv(value: str, *, integer: bool = False) -> list:
    values = [part.strip() for part in value.split(",")]
    if not values or any(not part for part in values):
        raise ValueError("Comma-separated selections must not contain empty entries")
    return [int(part) for part in values] if integer else values


def build_config(args):
    if not args.data_root or not args.output_root:
        raise ValueError("Set --data-root/--output-root or LM_CL_DATA_ROOT/LM_CL_OUTPUT_ROOT")
    data_root = Path(args.data_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    cache = Path(args.cache_root).expanduser().resolve() if args.cache_root else data_root / "hf-cache"
    forbidden = {Path("/"), Path.home().resolve()}
    if data_root in forbidden or cache in forbidden or output_root in forbidden:
        raise ValueError("Data/cache/output roots must be dedicated directories")
    gpu_ids = _csv(args.gpus, integer=True)
    if len(gpu_ids) not in {1, 2} or len(gpu_ids) != len(set(gpu_ids)) or min(gpu_ids) < 0:
        raise ValueError("Select one or two distinct nonnegative GPU IDs")
    if len(gpu_ids) % args.gpus_per_job:
        raise ValueError("GPU count must be divisible by GPUs per job")
    if not 1 <= args.parallel_languages <= 8:
        raise ValueError("Parallel language count must be in 1..8")
    models = _csv(args.models)
    if not set(models).issubset(PUBLIC_MODEL_VARIANTS):
        raise ValueError("Unknown variant; consult the public configuration schema")
    microbatch = args.physical_microbatch_sequences
    if microbatch is None:
        microbatch = 4 if args.a100_memory_gb == 40 else 8
    repository = Path(__file__).resolve().parents[3]
    template = repository / "configs/experiments" / f"zyphra_pair_a100_{args.model_size}_5cycle_1b.yaml"
    # Expansion is confined to this process, with original values restored.
    previous = {key: os.environ.get(key) for key in ("LM_CL_DATA_ROOT", "LM_CL_OUTPUT_ROOT")}
    try:
        os.environ.update(LM_CL_DATA_ROOT=str(data_root), LM_CL_OUTPUT_ROOT=str(output_root))
        config = load_launcher_config(template, overrides={
            "name": args.name or f"a100-{args.a100_memory_gb}gb-{args.model_size}-{args.cycles}cycle-{args.tokens_per_task}t-w{args.gpus_per_job}-mb{microbatch}",
            "models": models, "seeds": _csv(args.seeds, integer=True), "cycles": args.cycles,
            "tokens_per_task": args.tokens_per_task, "gpu_ids": gpu_ids,
            "gpus_per_job": args.gpus_per_job, "physical_microbatch_sequences": microbatch,
            "resume": args.resume,
        })
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    probe_budget = resolve_token_budget(args.probe_tokens, config.experiment.sequence_length,
                                       policy=config.experiment.token_budget_policy)
    probe_manifest = (Path(args.probe_training_manifest).expanduser().resolve()
                      if args.probe_training_manifest else
                      Path(config.data.generated_root) / "stages" /
                      f"zyphra-vi-probe-train-{probe_budget.effective_input_tokens}" / "manifest.json")
    from lm_cl.launcher.streaming import StreamingSettings, AlternatingSettings
    import json
    streaming = None
    if args.data_mode == "streaming":
        if args.probe_training_manifest:
            raise ValueError("Existing probe pools require --data-mode packed; streaming freezes a new paired study")
        if args.parallel_languages != 1:
            raise ValueError("Streaming uses one ordered producer; --parallel-languages must be 1")
        values = json.loads(Path(args.streaming_settings).read_text()) if args.streaming_settings else {}
        if "checkpoint_retention" in values:
            raise ValueError("Select checkpoint retention with --checkpoint-retention, not --streaming-settings")
        if "schedule" in values and values.pop("schedule") != args.streaming_schedule:
            raise ValueError("Streaming JSON schedule differs from CLI selection")
        if args.streaming_schedule == "alternating":
            if args.streaming_chunk_batches is not None:
                values["chunk_batches"] = args.streaming_chunk_batches
            streaming = {**asdict(AlternatingSettings(**values)), "schedule": "alternating"}
            if args.checkpoint_retention == "cycle":
                streaming["checkpoint_retention"] = "cycle_end_v1"
        else:
            if args.checkpoint_retention != "all":
                raise ValueError("Cycle checkpoint retention requires alternating streaming")
            if args.streaming_chunk_batches is not None:
                raise ValueError("Chunk size requires --streaming-schedule alternating")
            streaming = asdict(StreamingSettings(**values))
    elif args.streaming_settings:
        raise ValueError("--streaming-settings requires streaming mode")
    elif args.checkpoint_retention != "all":
        raise ValueError("Cycle checkpoint retention requires alternating streaming")
    elif args.streaming_schedule != "independent" or args.streaming_chunk_batches is not None:
        raise ValueError("Alternating schedule requires streaming data mode")
    config = replace(config, data=replace(config.data, mode=args.data_mode, streaming=streaming, dataset_cache_root=str(cache),
                     probe_training_manifest=str(probe_manifest),
                     prepare_if_missing=args.action in {"prepare", "run", "calibrate"} and not args.dry_run),
                     training=replace(config.training, checkpoint_frequency=args.checkpoint_every_batches),
                     probe=replace(config.probe, training_tokens=args.probe_tokens),
                     launcher=replace(config.launcher, max_parallel_jobs=len(gpu_ids) // args.gpus_per_job))
    config.validate()
    if config.experiment.model_size != args.model_size:
        raise ValueError("Preset model size differs from requested size")
    if cache == output_root or output_root in cache.parents:
        raise ValueError("Output root cannot contain the download cache")
    return config


def calibration_contract(measured, data_mode):
    """Use a standalone calibration's own data; never demand unrelated pools."""
    expected = "streaming_packed" if data_mode == "streaming" else "packed_shards"
    if any(task.train_source.kind != expected for task in measured.tasks):
        raise ValueError("Calibration source kind differs from --data-mode")
    if data_mode == "packed":
        return {"mode":"packed"}
    from lm_cl.data.streaming import source_from_pipeline
    roots = {str(source_from_pipeline(task.train_source.packed).root) for task in measured.tasks}
    if len(roots) != 1:
        raise ValueError("Calibration requires one streaming study")
    return {"mode":"streaming", "streaming_root":roots.pop()}


def command(argv: Sequence[str] | None = None) -> None:
    args = parser().parse_args(argv)
    config = build_config(args)
    action = "preflight" if args.dry_run else args.action
    from lm_cl.launcher.streaming import prepare_streaming, producer_service
    if config.data.mode == "streaming" and action in {"prepare", "run", "calibrate"}:
        prepare_streaming(config)
    if action == "calibrate":
        if not args.calibration_output:
            raise ValueError("Calibration requires --calibration-output in a new directory")
        if len(config.launcher.gpu_ids) != config.launcher.gpus_per_job:
            raise ValueError("Calibrate one job at a time; select exactly --gpus-per-job devices")
        from lm_cl.config import load_continual_config, save_continual_config
        from lm_cl.cli.calibrate_continual import calibrate
        calibration_source = args.calibration_config
        if calibration_source is None:
            if len(config.experiment.models) != 1 or len(config.experiment.seeds) != 1:
                raise ValueError("Generated calibration config requires one --models variant and one seed")
            from lm_cl.launcher.data import resolve_data_contract
            from lm_cl.launcher.jobs import expand_job_specs, build_continual_job_config
            data = resolve_data_contract(config, full_checksum_validation=True)
            job = expand_job_specs(config, data)[0]
            measured = build_continual_job_config(config, job)
            calibration_source = Path(args.calibration_output).expanduser().resolve().with_suffix(".source.yaml")
            if calibration_source.exists() or Path(args.calibration_output).expanduser().exists():
                raise FileExistsError("Calibration source/output already exists; choose a new name")
            calibration_source.parent.mkdir(parents=True, exist_ok=True)
            save_continual_config(measured, calibration_source)
        else:
            measured = load_continual_config(calibration_source)
        if measured.model.name != f"zyphra_{args.model_size}":
            raise ValueError("Calibration model differs from the selected entry script")
        if (measured.optimization.precision != config.training.precision or
                measured.optimization.physical_microbatch_sequences != config.training.physical_microbatch_sequences):
            raise ValueError("Calibration config precision/microbatch differs from the selected launch settings")
        calibration_data = calibration_contract(measured, config.data.mode)
        with producer_service(calibration_data):
            result = calibrate(calibration_source, args.calibration_output,
                             gpus=list(config.launcher.gpu_ids), batches=args.calibration_batches,
                             max_seconds=args.calibration_max_seconds,
                             minimum_gpu_memory_bytes=int(args.a100_memory_gb * .95 * 1024**3))
        print_json(result)
        return
    budget = resolve_token_budget(config.experiment.tokens_per_task, config.experiment.sequence_length,
                                  policy=config.experiment.token_budget_policy)
    if action == "plan":
        print_json({"status": "configuration_validated_only", "data_or_gpu_validation_performed": False,
                    "streaming_schedule": args.streaming_schedule,
                    "production_data_route": config.data.mode,
                    "incremental_block_production_launch_supported": True,
                    "a100_memory_gb_requested": args.a100_memory_gb,
                    "microbatch_is_unmeasured_starting_value": True,
                    "task_token_budget": budget.to_dict(), "resolved_config": config.to_dict()})
        return
    if action == "prepare":
        if not os.environ.get("HF_HOME"):
            raise ValueError("Preparation requires an explicit authenticated HF_HOME")
        from lm_cl.launcher.data import prepare_or_validate_data
        result = prepare_or_validate_data(config, full_checksum_validation=True,
                                          parallel_languages=args.parallel_languages)
        print_json({"status": "streaming_recipe_ready" if config.data.mode == "streaming" else "data_ready", "data_contract": result})
        return
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise ValueError("Unset CUDA_VISIBLE_DEVICES; --gpus selects physical device IDs")
    import torch
    for index in config.launcher.gpu_ids:
        if index >= torch.cuda.device_count() or "A100" not in torch.cuda.get_device_name(index):
            raise ValueError("A100 preflight/run requires the selected A100 devices; use --action plan offline")
        capacity = torch.cuda.get_device_properties(index).total_memory / 1024**3
        if capacity < args.a100_memory_gb * .95:
            raise ValueError("GPU memory is below the selected A100 memory class")
    from lm_cl.launcher.data import resolve_data_contract
    from lm_cl.launcher.jobs import expand_job_specs
    from lm_cl.launcher.scheduler import (allocate_job_slots, preflight_launch,
        write_job_configurations, LocalJobScheduler, write_launcher_summaries)
    data = resolve_data_contract(config, full_checksum_validation=True)
    jobs = expand_job_specs(config, data)
    assignments = allocate_job_slots(config, jobs)
    preflight = preflight_launch(config, jobs, assignments)
    if action == "preflight":
        print_json({"status": "preflight_passed", "child_processes_started": 0,
                    "preflight": {k: v for k, v in preflight.items() if k != "old_resolved"}})
        return
    write_job_configurations(jobs, preflight)
    with producer_service(data):
        results = LocalJobScheduler(config, jobs, assignments).run()
    summary, csv_path = write_launcher_summaries(config, results)
    if any(item.get("status") != "complete" for item in results):
        raise RuntimeError("One or more experiment jobs did not complete; inspect launcher evidence")
    print_json({"status": "complete", "summary_json": str(summary), "summary_csv": str(csv_path)})


def main():
    cli_entry(command)


if __name__ == "__main__":
    main()
