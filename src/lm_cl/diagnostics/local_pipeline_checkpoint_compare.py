"""Compare owned demo checkpoints: exact resume or measured cross-layout drift."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import numpy as np
import torch
from lm_cl.data.incremental import file_hash, immutable_write, json_bytes
from lm_cl.diagnostics.local_pipeline_resources import Limits
from lm_cl.diagnostics.local_pipeline_training import load_demo_checkpoint
from lm_cl.training.distributed import state_digest


def tensor_differences(left,right,*,rtol,atol,path=""):
    out=[]
    if isinstance(left,torch.Tensor):
        if not isinstance(right,torch.Tensor) or left.shape!=right.shape or left.dtype!=right.dtype:
            raise ValueError(f"Tensor identity differs: {path}")
        if torch.equal(left,right):return []
        if left.is_floating_point():
            difference=(left.double()-right.double()).abs()
            finite=bool(torch.isfinite(left).all() and torch.isfinite(right).all())
            out.append({"path":path,"elements":left.numel(),"max_absolute":float(difference.max()),
                        "rms_absolute":float(difference.square().mean().sqrt()),"finite":finite,
                        "within_tolerance":bool(torch.allclose(left,right,rtol=rtol,atol=atol))})
        else:out.append({"path":path,"elements":left.numel(),"integer_tensor_differs":True,"within_tolerance":False})
    elif isinstance(left,dict):
        if set(left)!=set(right):raise ValueError(f"Mapping keys differ: {path}")
        for key in left:out.extend(tensor_differences(left[key],right[key],rtol=rtol,atol=atol,path=f"{path}.{key}"))
    elif isinstance(left,(list,tuple)):
        if len(left)!=len(right):raise ValueError(f"Sequence lengths differ: {path}")
        for i,(a,b) in enumerate(zip(left,right)):out.extend(tensor_differences(a,b,rtol=rtol,atol=atol,path=f"{path}[{i}]"))
    elif isinstance(left,np.ndarray):
        if not np.array_equal(left,right):out.append({"path":path,"numpy_state_differs":True,"within_tolerance":False})
    elif isinstance(left,float):
        if left!=right:out.append({"path":path,"reference":left,"candidate":right,"max_absolute":abs(left-right),
                                  "within_tolerance":math.isclose(left,right,rel_tol=rtol,abs_tol=atol)})
    elif left!=right:out.append({"path":path,"reference":left,"candidate":right,"within_tolerance":False})
    return out


def compare(settings_path,limits):
    settings=json.loads(Path(settings_path).read_text())
    if set(settings)!={"schema_version","reference","candidate","mode","rtol","atol","output","data_equality_record"} or settings["schema_version"]!=1:
        raise ValueError("Unknown checkpoint comparison settings")
    if settings["mode"] not in {"exact_same_layout","cross_layout"} or not 0<=settings["rtol"]<=.01 or not 0<=settings["atol"]<=.01:
        raise ValueError("Invalid comparison mode/tolerances")
    left_path=limits.owned(settings["reference"]);right_path=limits.owned(settings["candidate"])
    left=load_demo_checkpoint(left_path);right=load_demo_checkpoint(right_path)
    a,b=left["resolved_config"],right["resolved_config"]
    for key in ["model","variant","optimization"]:
        if a[key]!=b[key]:raise ValueError(f"Scientific training configuration differs: {key}")
    for key in ["seed","deterministic_algorithms"]:
        if a["runtime"][key]!=b["runtime"][key]:raise ValueError(f"Scientific execution setting differs: {key}")
    if len(a["tasks"])!=len(b["tasks"]):raise ValueError("Task schedule lengths differ")
    for x,y in zip(a["tasks"],b["tasks"]):
        for key in ["language","cycle_index","task_index","input_token_budget","train_sequence_prefix_count","train_sequence_offset_count","validation_logical_batches"]:
            if x[key]!=y[key]:raise ValueError(f"Task schedule differs: {key}")
        if x["validation_source"]!=y["validation_source"]:raise ValueError("Validation identities differ")
        if x["train_source"]!=y["train_source"]:
            proof_path=settings["data_equality_record"]
            if not proof_path:raise ValueError("Different source layouts require a verified data equality record")
            proof=json.loads(Path(proof_path).read_text())
            if proof["status"]!="exact_logical_data_match" or {s["root"] for s in proof["streams"]}!={x["train_source"]["root"],y["train_source"]["root"]}:
                raise ValueError("Data equality record does not match checkpoint sources")
            for stream in proof["streams"]:
                if file_hash(Path(stream["root"])/"complete.json")!=stream["completion_sha256"]:
                    raise ValueError("Data equality record references changed completion identity")
    same=settings["mode"]=="exact_same_layout"
    if same and a["distributed"]!=b["distributed"]:raise ValueError("Exact resume requires the same distributed configuration")
    if same and left["distributed_state"]["world_size"]!=right["distributed_state"]["world_size"]:
        raise ValueError("Exact resume requires the same world size")
    fields=["model_state","optimizer_state","scheduler_state","gradients","scaler_state","trainer_state","memory_state"]
    if same:fields.append("rng_state")
    digests={key:{"reference":state_digest(left[key]),"candidate":state_digest(right[key])} for key in fields}
    if same:
        key="rank_rng_states"
        digests[key]={"reference":state_digest(left["distributed_state"][key]),"candidate":state_digest(right["distributed_state"][key])}
    exact=all(d["reference"]==d["candidate"] for d in digests.values())
    differences=[]
    for key in fields:
        if digests[key]["reference"]!=digests[key]["candidate"]:
            differences.extend(tensor_differences(left[key],right[key],rtol=settings["rtol"],atol=settings["atol"],path=key))
    record={"status":"exact_match" if exact else "differences_observed","mode":settings["mode"],
            "settings_sha256":file_hash(Path(settings_path)),"reference_sha256":file_hash(left_path),
            "candidate_sha256":file_hash(right_path),"digests":digests,"differences":differences,
            "rtol":settings["rtol"],"atol":settings["atol"],
            "all_compared_differences_within_tolerance":all(d["within_tolerance"] for d in differences)}
    immutable_write(limits.owned(settings["output"]),json_bytes(record))
    print(json.dumps({"status":record["status"],"output":settings["output"],"differences":len(differences)}))
    if same and not exact:raise RuntimeError("Exact same-layout interrupted/reference state comparison failed; evidence retained")


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--limits",required=True);p.add_argument("--settings",required=True)
    args=p.parse_args();torch.set_num_threads(1);compare(args.settings,Limits(args.limits))

if __name__=="__main__":main()
