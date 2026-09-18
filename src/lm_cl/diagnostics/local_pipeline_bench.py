"""Owned subprocess controller and resource sampler for GPU layout timings."""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import psutil

from lm_cl.data.incremental import immutable_write, json_bytes, file_hash
from lm_cl.diagnostics.local_pipeline_resources import Limits


def run_layout(plan_path: Path, limits: Limits):
    plan = json.loads(plan_path.read_text())
    if set(plan) != {"schema_version", "label", "jobs", "max_seconds", "sample_interval_seconds"} or plan["schema_version"] != 1:
        raise ValueError("Unknown layout plan")
    if not 0 < plan["max_seconds"] <= limits.v["max_train_process_seconds"] or not 1 <= plan["sample_interval_seconds"] <= 60:
        raise ValueError("Invalid layout time limits")
    assigned = []
    for job in plan["jobs"]:
        if set(job) != {"config", "gpus"} or not job["gpus"]:
            raise ValueError("Invalid layout job")
        assigned.extend(job["gpus"])
        config = json.loads(Path(job["config"]).read_text())
        limits.owned(config["runtime"]["output_dir"])
        distributed = config.get("distributed")
        if (len(job["gpus"]) > 1) != bool(distributed):
            raise ValueError("GPU layout disagrees with job config")
    if len(assigned) != len(set(assigned)) or not set(assigned).issubset({0,1,2}):
        raise ValueError("Overlapping or unknown assigned GPUs")
    inventory = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
    for line in inventory.strip().splitlines():
        i, used, utilization = [int(x.strip()) for x in line.split(",")]
        if i in assigned and (used > 256 or utilization > 10):
            raise RuntimeError(f"Assigned GPU {i} is already busy; refusing interference")
    root = limits.report/"raw"/plan["label"]
    root.mkdir(exist_ok=False)
    processes, handles = [], []
    started = time.monotonic()
    with limits.process("layout-controller", plan["label"]):
        try:
            for index, job in enumerate(plan["jobs"]):
                env = dict(os.environ)
                env.update(CUDA_VISIBLE_DEVICES=",".join(map(str,job["gpus"])), OMP_NUM_THREADS="1",
                           MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1",
                           CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONPATH=str(Path(__file__).resolve().parents[2]))
                command = [sys.executable]
                if len(job["gpus"]) > 1:
                    command += ["-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={len(job['gpus'])}"]
                command += ["-m", "lm_cl.cli.local_pipeline_demo", "--limits",str(limits.path),"train","--config",job["config"]]
                handle = (root/f"job-{index}.log").open("x")
                handles.append(handle)
                p = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, env=env, start_new_session=True)
                processes.append(p)
                immutable_write(root/f"job-{index}-launch.json",json_bytes({"pid":p.pid,
                    "create_time":psutil.Process(p.pid).create_time(), "command":command,
                    "gpus":job["gpus"], "config_sha256":file_hash(Path(job["config"])), "unix_time":time.time()}))
            with (root/"resources.jsonl").open("x") as sample:
                while any(p.poll() is None for p in processes):
                    if time.monotonic()-started > plan["max_seconds"]:
                        raise TimeoutError("Layout controller deadline reached")
                    disk = limits.check()
                    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu,utilization.memory,power.draw", "--format=csv,noheader,nounits"], text=True)
                    proc = []
                    for parent in processes:
                        try:
                            owned = [psutil.Process(parent.pid)]
                            owned += owned[0].children(recursive=True)
                        except psutil.NoSuchProcess:
                            continue
                        for p in owned:
                            try:
                                proc.append({"pid":p.pid,"rss":p.memory_info().rss,"cpu_times":dict(p.cpu_times()._asdict()),
                                             "io":dict(p.io_counters()._asdict())})
                            except (psutil.NoSuchProcess, psutil.AccessDenied):
                                pass
                    sample.write(json.dumps({"unix_time":time.time(),"elapsed":time.monotonic()-started,"gpu_csv":gpu,
                                            "processes":proc,"disk":disk,"available_ram":psutil.virtual_memory().available})+"\n")
                    sample.flush()
                    time.sleep(plan["sample_interval_seconds"])
            result = {"label":plan["label"],"elapsed_seconds":time.monotonic()-started,
                      "exit_codes":[p.returncode for p in processes],"jobs":plan["jobs"],
                      "status":"complete" if all(p.returncode==0 for p in processes) else "failed"}
            immutable_write(root/"result.json",json_bytes(result))
            print(json.dumps(result),flush=True)
        except BaseException:
            # Only process groups created above are eligible for termination.
            for p in processes:
                if p.poll() is None:
                    os.killpg(p.pid,signal.SIGTERM)
            for p in processes:
                try:
                    p.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid,signal.SIGKILL); p.wait()
            raise
        finally:
            for handle in handles:
                handle.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--limits",required=True)
    p.add_argument("--plan",required=True)
    args=p.parse_args()
    run_layout(Path(args.plan),Limits(args.limits))


if __name__=="__main__":
    main()
