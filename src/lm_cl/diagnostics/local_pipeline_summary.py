"""Summarize completed owned benchmarks without treating screens as steady state."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from lm_cl.data.incremental import file_hash, immutable_write, json_bytes


def stats(values):
    values = sorted(values)
    if not values:
        return None
    return {"n": len(values), "mean": statistics.mean(values), "median": statistics.median(values),
            "min": values[0], "max": values[-1],
            "sample_stdev": statistics.stdev(values) if len(values)>1 else None}


def summarize_job(config_path: Path):
    cfg = json.loads(config_path.read_text())
    root = Path(cfg["runtime"]["output_dir"])
    metrics_path = root/cfg["runtime"]["metrics_jsonl"]
    if not metrics_path.exists():
        return {"name":cfg["run_name"], "status":"not_started", "config":str(config_path)}
    records = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    train = [r for r in records if r["event"]=="timing_train"]
    measured = [r for r in train if r["measured"]]
    optim = [r for r in records if r["event"]=="timing_optimizer"]
    measured_optim = [r for r in optim if r["measured"]]
    windows=[]
    for update in measured_optim:
        end=update["global_logical_batches"]
        previous=max([r["global_logical_batches"] for r in optim if r["global_logical_batches"]<end], default=0)
        batches=[r for r in measured if previous<r["global_logical_batches"]<=end]
        if len(batches)==end-previous:
            seconds=sum(r["seconds"] for r in batches)+update["seconds"]
            tokens=sum(r["input_tokens"] for r in batches)
            windows.append({"start_batch":previous+1,"end_batch":end,"seconds":seconds,
                            "input_tokens":tokens,"input_tokens_per_second":tokens/seconds})
    evaluations=[]
    for r in records:
        if r["event"]!="timing_evaluation":
            continue
        result=r["result"]
        modes=[result[k] for k in ("reset","carried")] if "reset" in result else [result]
        evaluations.append({"seconds":r["seconds"],"mode_count":len(modes),
                            "input_tokens_per_mode":modes[0]["input_token_count"],
                            "definition":"continual evaluation; duplicated validation on each DDP rank",
                            "seconds_per_mode_sequence":r["seconds"]/(sum(m["input_token_count"] for m in modes)/cfg["tasks"][0]["train_source"]["sequence_length"])})
    complete_path=root/"complete.json"
    complete=json.loads(complete_path.read_text()) if complete_path.exists() else None
    errors=[r for r in records if r["event"]=="run_error"]
    elapsed=sum(r["seconds"] for r in measured)+sum(r["seconds"] for r in measured_optim)
    tokens=sum(r["input_tokens"] for r in measured)
    targets=sum(r["valid_targets"] for r in measured)
    checkpoints=[r for r in records if r["event"]=="timing_checkpoint"]
    checkpoint_bytes=[]
    for p in (root/"checkpoints").glob("*.demo.pt"):
        checkpoint_bytes.append({"path":str(p),"bytes":p.stat().st_size})
    final_state=None if complete is None else complete["state"]
    return {"name":cfg["run_name"],"status":"complete" if complete and complete["status"]=="complete" else "failed" if errors else "incomplete",
        "config":str(config_path),"config_sha256":file_hash(config_path),"metrics":str(metrics_path),"metrics_sha256":file_hash(metrics_path),
        "source_checkpoint_sha256":cfg["source_checkpoint_sha256"],"source_recipe_sha256":cfg["tasks"][0]["train_source"]["recipe_sha256"],
        "variant":cfg["variant"]["name"],"precision":cfg["optimization"]["precision"],
        "model":cfg["model"]["name"],"parameter_count":next((r["parameter_count"] for r in records if r["event"]=="run_start"),None),
        "world_size":next((r.get("world_size",1) for r in records if r["event"]=="run_start"),1),
        "global_batch":cfg["optimization"]["global_sequences_per_logical_batch"],
        "physical_microbatch":cfg["optimization"]["physical_microbatch_sequences"],
        "slow_k":cfg["variant"]["slow_update_period_k"],"warmup_batches":cfg["warmup_logical_batches"],
        "measured_batches":len(measured),"complete_measured_windows":windows,
        "training_input_tokens_per_second":tokens/elapsed if elapsed else None,
        "valid_targets_per_second":targets/elapsed if elapsed else None,
        "logical_batch_seconds":stats([r["seconds"] for r in measured]),
        "optimizer_seconds":stats([r["seconds"] for r in measured_optim]),
        "window_rate_stats":stats([w["input_tokens_per_second"] for w in windows]),
        "physical_cycle_seconds":stats([s for r in measured for s in r.get("physical_cycle_seconds",[])]),
        "physical_cycle_definition":"successive forward starts, includes backward/communication/host feed; final physical microbatch omitted",
        "forward_device_milliseconds":stats([s for r in measured for s in r.get("forward_device_milliseconds",[])]),
        "evaluations":evaluations,"checkpoint_timings":checkpoints,"checkpoint_files":checkpoint_bytes,
        "max_cuda_allocated_bytes":max([r["max_cuda_allocated_bytes"] for r in train],default=0),
        "loss_scales":sorted(set(r["loss_scale"] for r in train)),
        "all_batch_seconds":[r["seconds"] for r in train],
        "first_batch_start_unix":train[0]["unix_time"]-train[0]["seconds"] if train else None,
        "first_slow_update_unix":optim[0]["unix_time"] if optim else None,
        "final_state":final_state,"errors":[{"type":r["error_type"],"message":r["error_message"]} for r in errors],
        "purpose":"timing_only; repeated fixed token bank; no scientific CE/AUC claim"}


def summarize(report: Path):
    layouts=[]
    for result_path in sorted((report/"raw").glob("*/result.json")):
        result=json.loads(result_path.read_text())
        jobs=[summarize_job(Path(j["config"])) for j in result["jobs"]]
        for index,job in enumerate(jobs):
            launch_path=result_path.parent/f"job-{index}-launch.json"
            if launch_path.exists():
                launch=json.loads(launch_path.read_text())
                job["launch_unix"]=launch["unix_time"]
                if job.get("first_slow_update_unix") is not None:
                    job["time_to_first_slow_update_seconds"]=job["first_slow_update_unix"]-launch["unix_time"]
                if job.get("first_batch_start_unix") is not None:
                    job["training_startup_seconds"]=job["first_batch_start_unix"]-launch["unix_time"]
        resource_path=result_path.parent/"resources.jsonl"
        samples=[json.loads(l) for l in resource_path.read_text().splitlines()] if resource_path.exists() else []
        gpu_values={}
        for sample in samples:
            for line in sample["gpu_csv"].strip().splitlines():
                i,mem,util,memutil,power=[v.strip() for v in line.split(",")]
                gpu_values.setdefault(i,[]).append([float(mem),float(util),float(memutil),float(power)])
        gpus={i:{"peak_used_MiB":max(x[0] for x in v),"utilization_percent":stats([x[1] for x in v]),
                 "power_watts":stats([x[3] for x in v])} for i,v in gpu_values.items()}
        layouts.append({**result,"jobs":jobs,"resource_summary":{"gpus":gpus,
            "peak_owned_rss_bytes":max([sum(p["rss"] for p in s["processes"]) for s in samples],default=0)},
            "aggregate_measured_input_tokens_per_second":sum(j["training_input_tokens_per_second"] or 0 for j in jobs),
            "aggregate_rate_caveat":"Sum of actual concurrent job windows; use end-to-end layout wall time for short-job makespan",
            "result_path":str(result_path),"result_sha256":file_hash(result_path)})
    probe_evaluations=[]
    for path in sorted((report/"verification").glob("probe-eval-*.json")):
        value=json.loads(path.read_text())
        if value.get("status")!="complete" or not value.get("state_unchanged"):
            continue
        probe_evaluations.append({**value,"artifact_path":str(path),"artifact_sha256":file_hash(path)})
    return {"schema_version":1,"layouts":layouts,"probe_evaluations":probe_evaluations,"planning_only":True}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report",required=True)
    p.add_argument("--output",required=True)
    args=p.parse_args()
    report=Path(args.report).resolve(); output=Path(args.output).resolve()
    if not output.is_relative_to(report):
        raise ValueError("Summary output must be inside its report root")
    result=summarize(report)
    immutable_write(output,json_bytes(result))
    print(json.dumps({"output":str(output),"layouts":len(result["layouts"])}))


if __name__=="__main__":
    main()
