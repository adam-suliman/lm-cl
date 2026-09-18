"""Generate validated engineering configurations from frozen local identities."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from lm_cl.config.continual_schema import ContinualOptimizationConfig, ContinualRuntimeConfig, DistributedConfig
from lm_cl.config.schema import ModelConfig, VariantConfig
from lm_cl.data.incremental import IncrementalSource, immutable_write, json_bytes
from lm_cl.diagnostics.local_pipeline_resources import Limits
from lm_cl.diagnostics.local_pipeline_training import BlockInput, DemoTask, DemoTrainingConfig


def make_config(limits, *, name, model_size="5m", variant="ag", precision="fp32", microbatch=1,
                world_size=3, batches=8, save_checkpoint=False, train_root=None, source_record=None, validation_root=None):
    if model_size != "5m" and source_record is None:
        raise ValueError("12M requires an explicit verified source record")
    if source_record is None:
        inventory=json.loads((limits.report/"provenance/5m-derived-sources.json").read_text())
        key="backbone_clean" if variant=="transformer" else "fastmem_rmt"
        source_record=next(s for s in inventory["sources"] if s["variant"]==key)
    model=ModelConfig(**source_record["model"])
    if variant=="transformer":
        model_variant=VariantConfig("backbone_clean",False,False,0.,0,1,None)
    elif variant in {"ag","fast_off"}:
        model_variant=VariantConfig("fastmem_rmt" if variant=="ag" else "fastmem_rmt_zero",
            True,True,.005 if variant=="ag" else 0.,8,2,1.,1024,"task_boundary_from_m0_stopgrad","reset_and_carried")
    else:
        raise ValueError("Unknown timing variant")
    root=Path(train_root) if train_root else limits.work/"data/vi-cached-timing-v1"
    train=IncrementalSource(root,wait_seconds=0)
    val=IncrementalSource(Path(validation_root) if validation_root else limits.work/"data/vi-cached-validation-v1",wait_seconds=0)
    task=DemoTask("vi",0,0,BlockInput(str(root),train.recipe.sha256,2048,60),
                  batches*256*2048,batches*256,
                  BlockInput(str(val.root),val.recipe.sha256,2048,0),1)
    distributed=None if world_size==1 else DistributedConfig(True,"nccl",600,
        "contiguous_floor_v1","ddp_average_world_scaled_global_sum_v1",
        "sum_unscale_normalize_clip_rank0_broadcast_v1",False,False,False,False)
    cfg=DemoTrainingConfig(101,name,model,model_variant,
        ContinualOptimizationConfig("adamw",.9,.95,1e-8,.1,.05,256,microbatch,None,precision,2,0),
        ContinualRuntimeConfig(91301,"cuda",True,str(limits.work/"jobs"/name),"metrics.jsonl",
                              diagnostic_norm_interval_steps=1000000),
        [task],distributed,source_record["path"],source_record["sha256"],2,save_checkpoint)
    cfg.validate()
    path=limits.report/"configs"/f"train-{name}.json"
    immutable_write(path,json_bytes(asdict(cfg)))
    return path


def make_layout(limits, label, jobs, *, max_seconds=7200):
    plan={"schema_version":1,"label":label,"jobs":[{"config":str(path),"gpus":gpus} for path,gpus in jobs],
          "max_seconds":max_seconds,"sample_interval_seconds":5}
    path=limits.report/"configs"/f"layout-{label}.json"
    immutable_write(path,json_bytes(plan))
    return path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--limits",required=True)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--screens",action="store_true")
    mode.add_argument("--matrix",action="store_true")
    p.add_argument("--microbatch",type=int,choices=[1,2],default=2)
    p.add_argument("--attempt-suffix",default="")
    args=p.parse_args(); limits=Limits(args.limits)
    if args.attempt_suffix and not args.attempt_suffix.replace("-", "").isalnum():
        raise ValueError("Attempt suffix must be alphanumeric or hyphenated")
    suffix=("-"+args.attempt_suffix) if args.attempt_suffix else ""
    mb=args.microbatch
    outputs=[]
    if args.screens:
        for precision,microbatch in [("fp16",1),("fp32",1),("fp32",2)]:
            name=f"screen-5m-ag-{precision}-3gpu-mb{microbatch}"
            cfg=make_config(limits,name=name,precision=precision,microbatch=microbatch,batches=2)
            outputs.append(str(make_layout(limits,name,[(cfg,[0,1,2])],max_seconds=1800)))
    else:
        root=limits.work/"data/vi-cached-timing-v2"
        for world in [3,1,2]:
            name=f"measure-5m-ag-{world}gpu-fp32-mb{mb}{suffix}"
            cfg=make_config(limits,name=name,microbatch=mb,world_size=world,train_root=root,save_checkpoint=world in {1,3})
            outputs.append(str(make_layout(limits,name,[(cfg,list(range(world)))])))
        for variant in ["fast_off","transformer"]:
            name=f"measure-5m-{variant}-3gpu-fp32-mb{mb}{suffix}"
            cfg=make_config(limits,name=name,variant=variant,microbatch=mb,train_root=root)
            outputs.append(str(make_layout(limits,name,[(cfg,[0,1,2])])) )
        jobs=[]
        for gpu,variant in enumerate(["ag","fast_off","transformer"]):
            name=f"measure-D-5m-{variant}-1gpu-fp32-mb{mb}{suffix}"
            jobs.append((make_config(limits,name=name,variant=variant,microbatch=mb,world_size=1,train_root=root),[gpu]))
        outputs.append(str(make_layout(limits,"measure-D-three-independent"+suffix,jobs)))
        jobs=[]
        for variant,gpus in [("ag",[0,1]),("fast_off",[2])]:
            name=f"measure-E-5m-{variant}-{len(gpus)}gpu-fp32-mb{mb}{suffix}"
            jobs.append((make_config(limits,name=name,variant=variant,microbatch=mb,world_size=len(gpus),train_root=root),gpus))
        outputs.append(str(make_layout(limits,"measure-E-two-plus-one"+suffix,jobs)))
        source12=json.loads((limits.report/"provenance/12m-source.json").read_text())
        name=f"measure-12m-ag-3gpu-fp32-mb{mb}{suffix}"
        cfg=make_config(limits,name=name,model_size="12m",microbatch=mb,source_record=source12,train_root=root)
        outputs.append(str(make_layout(limits,name,[(cfg,[0,1,2])])) )
        name=f"heldout-5m-ag-3gpu-fp32-mb{mb}{suffix}"
        cfg=make_config(limits,name=name,microbatch=mb,batches=12,train_root=root)
        outputs.append(str(make_layout(limits,name,[(cfg,[0,1,2])])) )
        immutable_write(limits.report/"configs"/f"measurement-layout-order{suffix}.json",json_bytes(outputs))
    print(json.dumps(outputs,indent=2))


if __name__=="__main__":
    main()
