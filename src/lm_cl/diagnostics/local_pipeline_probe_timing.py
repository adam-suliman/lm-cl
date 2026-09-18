"""Time the actual production probe-evaluation method on a declared replay bank.

No training or scientific evaluation is performed. The 32-sequence fixed bank
is repeated in RAM solely to measure full 1,280-sequence evaluation cost, with
the actual distributed partition and reset/carried isolation methods.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from lm_cl.data.incremental import IncrementalSource, file_hash, immutable_write, json_bytes
from lm_cl.data.sources import ArrayTokenSource
from lm_cl.diagnostics.local_pipeline_resources import Limits
from lm_cl.diagnostics.local_pipeline_training import DemoTrainer, DistributedDemoTrainer, load_demo_training_config
from lm_cl.training.checkpoint import capture_rng_state, restore_rng_state
from lm_cl.training.distributed import DistributedContext
from lm_cl.training.probe import ProbeTrainer


class ProbeTimingMixin:
    _evaluation_root_for_probe=ProbeTrainer._evaluation_root_for_probe
    _evaluation_modes=ProbeTrainer._evaluation_modes
    _evaluate_one_mode=ProbeTrainer._evaluate_one_mode
    _evaluation_state_digest=ProbeTrainer._evaluation_state_digest


class SingleProbeTiming(ProbeTimingMixin,DemoTrainer):
    pass


class DistributedProbeTiming(ProbeTimingMixin,DistributedDemoTrainer):
    pass


def run(config_path, limits):
    plan=json.loads(Path(config_path).read_text())
    if set(plan)!={"schema_version","training_config","validation_sequences","repetitions","warmup_repetitions","output"} or plan["schema_version"]!=1:
        raise ValueError("Unknown probe timing plan")
    cfg=load_demo_training_config(plan["training_config"])
    length=cfg.tasks[0].train_source.sequence_length
    batch=cfg.optimization.global_sequences_per_logical_batch
    if plan["validation_sequences"]%batch or not 1<=plan["validation_sequences"]<=1280:
        raise ValueError("Timing evaluation must use complete global batches, at most1280 sequences")
    if not 2<=plan["repetitions"]<=4 or plan["warmup_repetitions"]!=1:
        raise ValueError("Timing evaluation requires one warmup and bounded repeats")
    output=limits.owned(plan["output"])
    if output.exists():
        raise FileExistsError("Probe timing output already exists")
    torch.set_num_threads(limits.v["cpu_threads_per_trainer"])
    os.environ["LM_CL_DEMO_LIMITS"]=str(limits.path)
    context=None
    with limits.process("probe-validation-timing",cfg.run_name):
        try:
            if cfg.distributed:
                context=DistributedContext.initialize(cfg.distributed,runtime_device=cfg.runtime.device)
                trainer=DistributedProbeTiming(cfg,context)
            else:
                trainer=SingleProbeTiming(cfg)
            source_cfg=cfg.tasks[0].validation_source
            source=IncrementalSource(source_cfg.root,recipe_sha256=source_cfg.recipe_sha256,wait_seconds=0)
            original=source.read_tokens(source.token_count)[0]
            count=plan["validation_sequences"]*length
            if count%len(original):
                raise ValueError("Timing bank must repeat a whole immutable validation fixture")
            repeated=np.tile(original,count//len(original))
            trainer.validation_source=ArrayTokenSource(repeated)
            trainer.validation_ignore_index=-100
            trainer.validation_source_identity={"purpose":"prohibited_scientific_use_timing_only",
                "recipe_sha256":source.recipe.sha256,"unique_input_tokens":len(original),
                "repeated_exposure_tokens":count,"repeats":count//len(original)}
            trainer.training_source_identity={"purpose":"no_training_performed"}
            trainer.source_checkpoint_hash_before=cfg.source_checkpoint_sha256
            trainer.probe_config=SimpleNamespace(variant=cfg.variant,optimization=cfg.optimization,
                sequence_length=length,validation_sequences=plan["validation_sequences"],probe_mode="timing_only")
            if cfg.variant.memory_enabled:
                trainer.active_memory=trainer._rmt_model().initial_memory.detach().clone().requires_grad_(True)
            digest=trainer._evaluation_state_digest()
            records=[]
            for index in range(plan["repetitions"]+plan["warmup_repetitions"]):
                limits.check(); rng=capture_rng_state(); trainer._sync()
                start=time.perf_counter()
                modes=[trainer._evaluate_one_mode(mode) for mode in trainer._evaluation_modes()]
                trainer._sync(); elapsed=time.perf_counter()-start
                restore_rng_state(rng)
                tick=time.perf_counter(); after=trainer._evaluation_state_digest()
                if after!=digest:
                    raise RuntimeError("Probe timing changed model/memory/RNG/trainer state")
                records.append({"index":index,"measured":index>=plan["warmup_repetitions"],"seconds":elapsed,
                    "state_digest_seconds":time.perf_counter()-tick,"modes":modes,
                    "max_cuda_allocated_bytes":torch.cuda.max_memory_allocated(trainer.device) if trainer.device.type=="cuda" else 0})
            if cfg.source_checkpoint and file_hash(Path(cfg.source_checkpoint))!=cfg.source_checkpoint_sha256:
                raise RuntimeError("Preserved source checkpoint changed")
            if context is None or context.is_primary:
                result={"schema_version":1,"status":"complete","plan":str(config_path),"plan_sha256":file_hash(Path(config_path)),
                    "variant":cfg.variant.name,"model":cfg.model.name,"world_size":1 if context is None else context.world_size,
                    "physical_microbatch":cfg.optimization.physical_microbatch_sequences,"global_batch":batch,
                    "validation_sequences":plan["validation_sequences"],"records":records,
                    "input_identity":trainer.validation_source_identity,"state_unchanged":True,
                    "method":"ProbeTrainer._evaluate_one_mode, actual partition/collectives; fixed M0 for both roots; no optimizer state",
                    "limitation":"Repeated fixture is only a timing workload. Post-update optimizer state hashing is not measured."}
                immutable_write(output,json_bytes(result))
                print(json.dumps({"status":"complete","output":str(output),"seconds":[r["seconds"] for r in records]}),flush=True)
        finally:
            if context:
                context.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--limits",required=True);p.add_argument("--config",required=True)
    args=p.parse_args(); limits=Limits(args.limits)
    def deadline(*_):
        raise TimeoutError("Owned validation benchmark deadline reached")
    signal.signal(signal.SIGALRM,deadline);signal.alarm(limits.v["max_train_process_seconds"])
    run(args.config,limits)


if __name__=="__main__":
    main()
