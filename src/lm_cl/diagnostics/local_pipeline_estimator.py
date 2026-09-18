"""Configuration-derived runtime/storage forecasts with explicit measurement gaps."""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import statistics
from pathlib import Path

from lm_cl.data.incremental import file_hash, immutable_write, json_bytes
from lm_cl.launcher.config import load_launcher_config
from lm_cl.launcher.schema import resolve_token_budget


def workload_counts(requested, *, sequence_length, global_batch, k, interval, milestones):
    budget=resolve_token_budget(requested,sequence_length)
    n=budget.effective_complete_sequences
    batches=math.ceil(n/global_batch)
    points=sorted({0,batches,*[x for x in milestones if x<=batches],*range(interval,batches+1,interval)})
    return {**budget.to_dict(),"logical_batches":batches,"full_logical_batches":n//global_batch,
            "last_batch_sequences":n%global_batch or global_batch,"slow_updates":math.ceil(batches/k),
            "full_k_windows":batches//k,"tail_window_batches":batches%k,
            "evaluation_steps":points,"evaluation_points":len(points)}


def retention_passes(languages, cycles, memory):
    seen=set(); passes=0
    for _ in range(cycles):
        for language in languages:
            seen.add(language)
            passes+=len(seen)+int(memory)  # previous languages reset; current also carried
    return passes


def simulate_overlap(tokens, *, producer_rate, consumer_rate, block_tokens,
                     startup_seconds, lookahead_tokens):
    """Deterministic bounded block timeline, assuming a measured constant producer rate.

    Each service period is a block; this models disk publication, not a scientific
    task boundary. No rate is inferred from authentication or metadata requests.
    """
    if min(tokens,producer_rate,consumer_rate,block_tokens,lookahead_tokens)<=0 or lookahead_tokens<block_tokens:
        raise ValueError("Invalid overlap simulation inputs")
    ready=[]; producer_time=startup_seconds; consumer_time=0.; consumed=0; produced=0; stalls=0.; timeline=[]
    while produced<tokens:
        size=min(block_tokens,tokens-produced)
        # A producer cannot exceed the chosen ready-token lookahead. Release one
        # completed consumer block before publishing another beyond that bound.
        while ready and produced-consumed+size>lookahead_tokens:
            available,n=ready.pop(0)
            begin=max(consumer_time,available)
            stalls+=max(0.,available-consumer_time)
            consumer_time=begin+n/consumer_rate; consumed+=n
            timeline.append({"seconds":consumer_time,"event":"consumed","tokens":consumed})
            producer_time=max(producer_time,consumer_time)
        producer_time+=size/producer_rate; produced+=size
        ready.append((producer_time,size))
        timeline.append({"seconds":producer_time,"event":"published","tokens":produced})
    for available,n in ready:
        begin=max(consumer_time,available)
        stalls+=max(0.,available-consumer_time)
        consumer_time=begin+n/consumer_rate; consumed+=n
        timeline.append({"seconds":consumer_time,"event":"consumed","tokens":consumed})
    return {"elapsed_seconds":consumer_time,"producer_done_seconds":producer_time,
            "data_wait_seconds":stalls,"first_block_seconds":startup_seconds+min(tokens,block_tokens)/producer_rate,
            "timeline":sorted(timeline,key=lambda r:r["seconds"])}


def choose_job(measurements, variant, world=3, model="zyphra_5m", concurrent=None):
    candidates=[]
    for layout in measurements["layouts"]:
        if layout["status"]!="complete":
            continue
        for job in layout["jobs"]:
            if (job["status"]=="complete" and job["variant"]==variant and job["world_size"]==world
                and job["model"]==model and job["precision"]=="fp32" and job["physical_microbatch"]==1
                and len(job["complete_measured_windows"])>=3 and not job["name"].startswith("heldout")):
                if concurrent is None and len(layout["jobs"])!=1:
                    continue
                if concurrent is not None and not layout["label"].startswith(concurrent):
                    continue
                candidates.append((job,layout))
    if len(candidates)!=1:
        raise ValueError(f"Expected one completed measurement for {variant}/{world}/{model}/{concurrent}; got {len(candidates)}")
    return candidates[0]


def probe_seconds_per_mode_sequence(measurements,variant,model,world):
    candidates=[p for p in measurements.get("probe_evaluations",[]) if p["variant"]==variant and p["model"]==model and p["world_size"]==world]
    if not candidates:return None
    if len(candidates)!=1:raise ValueError("Ambiguous probe evaluation measurement")
    value=candidates[0]; records=[r for r in value["records"] if r["measured"]]
    if len(records)<2:raise ValueError("Probe evaluation requires repeated measured points")
    rate=statistics.mean((r["seconds"]+2*r["state_digest_seconds"])/(value["validation_sequences"]*len(r["modes"])) for r in records)
    return rate,value["artifact_path"]


def forecast(measurements, source_config, assumptions):
    cfg=load_launcher_config(source_config)
    if assumptions["schema_version"]!=1 or assumptions["long_run_slowdown_factor"]<1:
        raise ValueError("Invalid forecast assumptions")
    if assumptions["source_preparation_tokens_per_second"] is not None and assumptions["source_preparation_tokens_per_second"]<=0:
        raise ValueError("Preparation rate must be positive or explicitly unavailable")
    if assumptions["source_preparation_tokens_per_second"] is not None:
        raise ValueError("Live measured preparation integration is pending; do not silently ignore a provided rate")
    if not 1<=assumptions["packed_metadata_factor"]<=4 or assumptions["source_cache_reserve_bytes"]<0:
        raise ValueError("Invalid storage planning factors")
    if len(set(assumptions["matrix_seeds"]))!=len(assumptions["matrix_seeds"]) or not assumptions["matrix_seeds"]:
        raise ValueError("Matrix requires distinct explicit seeds")
    variants=list(assumptions["variants"])
    variants.append({"name":"full_ag_12m","measurement_variant":"fastmem_rmt","model":"zyphra_12m",
                     "memory":True,"k":cfg.fastmem.slow_accumulation_k,"proxy":False,
                     "comment":"Targeted measured12M AG; same validated horizon/count schedule, checkpoint size extrapolated"})
    if len({v["name"] for v in variants})!=len(variants):
        raise ValueError("Duplicate forecast variants")
    sequence=cfg.experiment.sequence_length; batch=cfg.training.global_batch_sequences
    probe_cfg=cfg.probe; eval_sequences=cfg.forgetting.validation_sequences_per_language
    rows=[]; measurement_index={}
    for v in variants:
        model=v.get("model","zyphra_5m")
        job,layout=choose_job(measurements,v["measurement_variant"],model=model)
        measurement_cfg=json.loads(Path(job["config"]).read_text())
        if file_hash(Path(job["config"]))!=job["config_sha256"]:
            raise ValueError("Measured training configuration changed")
        if job["global_batch"]!=batch or measurement_cfg["tasks"][0]["train_source"]["sequence_length"]!=sequence:
            raise ValueError("Forecast dimensions differ from measured global batch/sequence")
        if v["memory"] and measurement_cfg["variant"]["segment_length"]!=cfg.fastmem.segment_length:
            raise ValueError("Forecast AG segment length differs from measurement")
        memory=v["memory"]; k=v["k"]
        if type(memory) is not bool or k not in {1,2}:
            raise ValueError("Unsupported forecast variant contract")
        measurement_index[v["name"]]={"name":job["name"],"proxy":v["proxy"],"comment":v["comment"]}
        evalsec=statistics.mean(e["seconds_per_mode_sequence"] for e in job["evaluations"])
        # Continual validation is measured on every rank. Until an actual probe
        # benchmark exists, retain this conservative rate rather than dividing
        # by GPU count and claiming a measured DDP probe speedup.
        probe_measurement=probe_seconds_per_mode_sequence(measurements,v["measurement_variant"],model,3)
        probe_eval=evalsec if probe_measurement is None else probe_measurement[0]
        checkpoint_jobs=[j for l in measurements["layouts"] for j in l["jobs"]
                         if j.get("status")=="complete" and j.get("model")==job["model"] and j.get("checkpoint_timings")]
        checkpoint_scale=1.
        if not checkpoint_jobs:
            checkpoint_jobs=[j for l in measurements["layouts"] for j in l["jobs"]
                             if j.get("status")=="complete" and j.get("model")=="zyphra_5m" and j.get("checkpoint_timings")]
            checkpoint_scale=job["parameter_count"]/checkpoint_jobs[0]["parameter_count"]
        ckptsec=statistics.median(r["seconds"] for j in checkpoint_jobs for r in j["checkpoint_timings"])*checkpoint_scale
        ckptbytes=math.ceil(max(p["bytes"] for j in checkpoint_jobs for p in j["checkpoint_files"])*checkpoint_scale)
        batchsec=job["logical_batch_seconds"]["mean"]
        optsec=job["optimizer_seconds"]["mean"]
        startup=max(0.,layout["elapsed_seconds"]-sum(job["all_batch_seconds"])-sum(e["seconds"] for e in job["evaluations"])
                    -sum(r["seconds"] for r in job["checkpoint_timings"]))

        def counts(tokens):
            return workload_counts(tokens,sequence_length=sequence,global_batch=batch,k=k,
                interval=probe_cfg.evaluation_interval,milestones=probe_cfg.evaluation_milestones)

        def traincost(c):
            # Tail compute scales with physical sequence count; add a full slow
            # transition for the final partial K window, as the trainer does.
            return c["effective_complete_sequences"]/batch*batchsec+c["slow_updates"]*optsec

        def add(name,stage_tokens,cycles,probe_tokens,kind):
            c=counts(stage_tokens) if stage_tokens else None
            p=counts(probe_tokens) if probe_tokens else None
            languages=len(cfg.experiment.languages) if cycles else 1
            tasks=languages*cycles if cycles else int(c is not None)
            probe_count=cycles if cycles and p else int(p is not None)
            stage_train=traincost(c)*tasks if c else 0.
            probe_train=traincost(p)*probe_count if p else 0.
            passes=retention_passes(cfg.experiment.languages,cycles,memory) if cycles else (1+int(memory) if c else 0)
            retention=passes*eval_sequences*evalsec
            probe_passes=p["evaluation_points"]*(1+int(memory))*probe_count if p else 0
            probe_evaluation=probe_passes*probe_cfg.validation_sequences*probe_eval
            # Launcher retains task boundaries, an augmented cycle checkpoint,
            # and each completed probe. retain_last is not automatic eviction.
            checkpoints=tasks+cycles+probe_count
            if cfg.training.checkpoint_frequency:
                checkpoints+=((c["logical_batches"]-1)//cfg.training.checkpoint_frequency)*tasks if c else 0
                checkpoints+=((p["logical_batches"]-1)//cfg.training.checkpoint_frequency)*probe_count if p else 0
            exposure=(c["effective_input_tokens"]*tasks if c else 0)+(p["effective_input_tokens"]*probe_count if p else 0)
            unique=(c["effective_input_tokens"]*tasks if c else 0)+(p["effective_input_tokens"] if p else 0)
            val_unique=sequence*((languages*eval_sequences if c else 0)+(probe_cfg.validation_sequences if p else 0))
            total=stage_train+probe_train+retention+probe_evaluation+checkpoints*ckptsec+(1+probe_count)*startup
            overhead=retention+probe_evaluation+checkpoints*ckptsec+(1+probe_count)*startup
            low_rate=max(w["input_tokens_per_second"] for w in job["complete_measured_windows"])
            high_rate=min(w["input_tokens_per_second"] for w in job["complete_measured_windows"])
            low=exposure/low_rate+overhead
            high=(exposure/high_rate+overhead)*assumptions["long_run_slowdown_factor"]
            raw=4*(unique+val_unique)
            rows.append({"scenario":name,"kind":kind,"variant":v["name"],"model":model,"measurement":job["name"],
                "proxy":v["proxy"],"stage_counts":c,"probe_counts":p,"cycles":cycles,"tasks":tasks,
                "probe_count":probe_count,"training_exposure_tokens":exposure,"unique_training_tokens":unique,
                "unique_validation_tokens":val_unique,"retention_mode_passes":passes,"probe_mode_passes":probe_passes,
                "stage_training_seconds":stage_train,"probe_training_seconds":probe_train,
                "retention_evaluation_seconds":retention,"probe_evaluation_seconds":probe_evaluation,
                "probe_evaluation_basis":"Conservative measured continual-eval proxy; actual partitioned probe overhead not yet measured" if probe_measurement is None else
                    "Actual partitioned probe timing plus two no-optimizer state digests; post-update optimizer hashing remains a planning limitation: "+probe_measurement[1],
                "checkpoint_count":checkpoints,"checkpoint_seconds":checkpoints*ckptsec,
                "startup_seconds":(1+probe_count)*startup,"data_ready_seconds":total,
                "planning_low_seconds":min(low,total),"planning_high_seconds":max(high,total),
                "end_to_end_with_download_seconds":None,
                "end_to_end_status":"UNVALIDATED: live source authorization blocked; no measured producer rate",
                "packed_uint32_payload_bytes":raw,"checkpoint_payload_bytes":checkpoints*ckptbytes,
                "minimum_payload_bytes":raw+checkpoints*ckptbytes,
                "storage_planning_bytes":math.ceil(raw*assumptions["packed_metadata_factor"])+checkpoints*ckptbytes+assumptions["source_cache_reserve_bytes"],
                "retry_one_late_stage_upper_seconds":(traincost(c) if c else traincost(p))+ckptsec+startup,
                "checkpoint_size_basis":"measured full demo checkpoint; launcher/probe serialization may differ" if checkpoint_scale==1 else "5M checkpoint scaled by measured total parameter count;12M checkpoint write not measured"})

        for tokens in assumptions["probe_token_budgets"]:
            add(f"vi-{tokens}",0,0,tokens,"probe")
        for tokens in assumptions["stage_token_budgets"]:
            add(f"language-{tokens}",tokens,0,0,"stage")
            add(f"cycle-{tokens}-per-language",tokens,1,0,"cycle")
        add("five-cycles-1B-plus-five-1B-probes",cfg.experiment.tokens_per_task,cfg.experiment.cycles,probe_cfg.training_tokens,"study")
        for study in assumptions["reduced_studies"]:
            if not 1<=study["cycles"]<=cfg.experiment.cycles:
                raise ValueError("Invalid reduced-study cycle count")
            add(study["name"],study["tokens_per_task"],study["cycles"],study["probe_tokens"],"reduced")

    grouped=[]
    for budget in assumptions["probe_token_budgets"]:
        selected=[r for r in rows if r["scenario"]==f"vi-{budget}" and r["variant"] in {"full_ag","fast_off"}]
        grouped.append({"scenario":f"fast-on-off-pair-vi-{budget}","conditions":2,
            "sequential_three_gpu_seconds":sum(r["data_ready_seconds"] for r in selected),
            "shared_unique_training_tokens":max(r["unique_training_tokens"] for r in selected),
            "shared_packed_payload_bytes":max(r["packed_uint32_payload_bytes"] for r in selected),
            "checkpoint_payload_bytes":sum(r["checkpoint_payload_bytes"] for r in selected)})
    for scenario in ["five-cycles-1B-plus-five-1B-probes",*[s["name"] for s in assumptions["reduced_studies"]]]:
        selected=[r for r in rows if r["scenario"]==scenario and r["model"]=="zyphra_5m"]
        seeds=assumptions["matrix_seeds"]
        grouped.append({"scenario":f"matrix-{scenario}","conditions":len(selected)*len(seeds),"seeds":seeds,
            "sequential_three_gpu_seconds":sum(r["data_ready_seconds"] for r in selected)*len(seeds),
            "training_exposure_tokens":sum(r["training_exposure_tokens"] for r in selected)*len(seeds),
            "shared_unique_training_tokens":max(r["unique_training_tokens"] for r in selected),
            "shared_packed_payload_bytes":max(r["packed_uint32_payload_bytes"] for r in selected),
            "checkpoint_payload_bytes":sum(r["checkpoint_payload_bytes"] for r in selected)*len(seeds),
            "note":"Shared immutable data prepared once; proxy baseline rows are not measured scientific variants"})
    by_name={v["name"]:v for v in variants}
    ag2,_=choose_job(measurements,"fastmem_rmt",2,concurrent="measure-E-")
    off1,_=choose_job(measurements,"fastmem_rmt_zero",1,concurrent="measure-E-")
    ag3,_=choose_job(measurements,"fastmem_rmt")
    off3,_=choose_job(measurements,"fastmem_rmt_zero")
    e_two_ratio=ag3["training_input_tokens_per_second"]/ag2["training_input_tokens_per_second"]
    e_one_ratio=off3["training_input_tokens_per_second"]/off1["training_input_tokens_per_second"]
    for group in grouped:
        if group["scenario"].startswith("matrix-"):
            name=group["scenario"][len("matrix-"):]
            jobs=[r for r in rows if r["scenario"]==name and r["model"]=="zyphra_5m"]*len(group["seeds"])
        else:
            budget=int(group["scenario"].rsplit("-",1)[1])
            jobs=[r for r in rows if r["scenario"]==f"vi-{budget}" and r["variant"] in {"full_ag","fast_off"}]
        durations=[]
        for row in jobs:
            v=by_name[row["variant"]]
            d,_=choose_job(measurements,v["measurement_variant"],1,concurrent="measure-D-")
            c,_=choose_job(measurements,v["measurement_variant"])
            ratio=c["training_input_tokens_per_second"]/d["training_input_tokens_per_second"]
            training=row["stage_training_seconds"]+row["probe_training_seconds"]
            overhead=row["data_ready_seconds"]-training
            one=probe_seconds_per_mode_sequence(measurements,v["measurement_variant"],"zyphra_5m",1)
            if one is None:
                ag_one=probe_seconds_per_mode_sequence(measurements,"fastmem_rmt","zyphra_5m",1)
                ag_three=probe_seconds_per_mode_sequence(measurements,"fastmem_rmt","zyphra_5m",3)
                variant_three=probe_seconds_per_mode_sequence(measurements,v["measurement_variant"],"zyphra_5m",3)
                if ag_one and ag_three and variant_three:
                    one=(ag_one[0]*variant_three[0]/ag_three[0],"AG one-GPU ratio proxy")
            d_overhead=overhead
            if one:
                d_overhead+=row["probe_mode_passes"]*probe_cfg.validation_sequences*one[0]-row["probe_evaluation_seconds"]
            durations.append((training*ratio+d_overhead,training,overhead,d_overhead,row["variant"]))
        lanes=[0.,0.,0.]
        for duration,_,_,_,_ in sorted(durations,reverse=True):
            i=min(range(3),key=lanes.__getitem__);lanes[i]+=duration
        e_lanes=[0.,0.]
        for _,training,overhead,d_overhead,_ in sorted(durations,reverse=True):
            choices=[e_lanes[0]+training*e_two_ratio+max(overhead,d_overhead),e_lanes[1]+training*e_one_ratio+d_overhead]
            i=min(range(2),key=choices.__getitem__);e_lanes[i]=choices[i]
        group["three_independent_gpu_seconds"]=max(lanes)
        group["three_independent_lane_seconds"]=lanes
        group["two_plus_one_gpu_seconds"]=max(e_lanes)
        group["two_plus_one_lane_seconds"]=e_lanes
        group["concurrent_schedule_basis"]="Greedy longest-job scheduling using measured contended training rates. One-GPU probe cost measured for AG, variant-scaled proxy otherwise; E conservatively uses one-GPU probe cost on its two-GPU lane. Retention/checkpoint overhead kept from C. E applies AG2/off1 training ratios to other variants as a proxy; finishing-lane speedups not assumed. Data startup remains unknown."
    baseline,_=choose_job(measurements,"fastmem_rmt")
    sensitivities=[]
    for rate in assumptions["illustrative_producer_rates"]:
        sim=simulate_overlap(assumptions["overlap_example_tokens"],producer_rate=rate,
            consumer_rate=baseline["training_input_tokens_per_second"],block_tokens=assumptions["ready_block_tokens"],
            startup_seconds=assumptions["illustrative_cold_start_seconds"],lookahead_tokens=assumptions["lookahead_tokens"])
        sensitivities.append({"producer_tokens_per_second":rate,"basis":"ILLUSTRATIVE, NOT MEASURED",**sim})
    heldouts=[]
    for layout in measurements["layouts"]:
        for j in layout["jobs"]:
            if j["status"]!="complete" or not j["name"].startswith("heldout") or j["physical_microbatch"]!=1:
                continue
            measured=sum(w["seconds"] for w in j["complete_measured_windows"])
            tokens=sum(w["input_tokens"] for w in j["complete_measured_windows"])
            prediction=tokens/baseline["training_input_tokens_per_second"]
            heldouts.append({"job":j["name"],"measured_seconds":measured,"predicted_seconds":prediction,
                "relative_error":(prediction-measured)/measured,"scope":"Steady training incl. slow updates, excludes data startup/eval/checkpoint"})
    return {"schema_version":1,"source_config":str(source_config),"source_config_sha256":file_hash(Path(source_config)),
        "resolved_scientific_config":cfg.to_dict(),"assumptions":assumptions,"measurement_index":measurement_index,
        "local_execution_overrides":{"precision":baseline["precision"],"physical_microbatch_sequences":baseline["physical_microbatch"],
                                     "global_batch_sequences":baseline["global_batch"],"note":"Counts from historical config; costs from validated local FP32 configs, not H100 BF16 rates"},
        "forecasts":rows,"groups":grouped,"overlap_sensitivities":sensitivities,"heldout_validation":heldouts,
        "warnings":["All end-to-end download ETAs remain unvalidated without source access.",
        "Ranges are engineering planning envelopes, not statistical confidence intervals.",
        "Short repeated-token workloads establish cost/finite updates, not long-run precision or scientific equivalence.",
        "The baseline full five-variant matrix is an estimate; base RMT is not currently a supported public launcher variant."]}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ["measurements","source-config","assumptions","output-dir"]:
        p.add_argument("--"+name,required=True)
    args=p.parse_args()
    output=Path(args.output_dir); output.mkdir(parents=True,exist_ok=False)
    result=forecast(json.loads(Path(args.measurements).read_text()),Path(args.source_config),json.loads(Path(args.assumptions).read_text()))
    result["measurements_sha256"]=file_hash(Path(args.measurements))
    immutable_write(output/"estimates.json",json_bytes(result))
    columns=["scenario","variant","proxy","training_exposure_tokens","unique_training_tokens","retention_mode_passes","probe_mode_passes",
             "data_ready_seconds","planning_low_seconds","planning_high_seconds","packed_uint32_payload_bytes","checkpoint_payload_bytes","storage_planning_bytes"]
    stream=io.StringIO(); writer=csv.DictWriter(stream,fieldnames=columns,extrasaction="ignore"); writer.writeheader(); writer.writerows(result["forecasts"])
    immutable_write(output/"estimates.csv",stream.getvalue().encode())
    print(json.dumps({"output":str(output),"forecasts":len(result["forecasts"]),"heldout_checks":len(result["heldout_validation"])}))


if __name__=="__main__":
    main()
