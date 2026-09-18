"""Freeze the supervised Vietnamese live/reference demonstration after validation."""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
from lm_cl.data.incremental import IncrementalSource, Recipe, file_hash, immutable_write, json_bytes
from lm_cl.diagnostics.local_pipeline_resources import Limits


def freeze_vi(settings_path, limits):
    from lm_cl.cli.local_pipeline_demo import _initialize
    from lm_cl.diagnostics.local_pipeline_plan import make_config, make_layout
    from types import SimpleNamespace
    settings=json.loads(Path(settings_path).read_text())
    expected={"schema_version","validation_root","tokens","block_tokens","reference_block_tokens","document_order_seed",
              "global_batch","sequence_length","world_size","physical_microbatch","precision","run_prefix"}
    if set(settings)!=expected or settings["schema_version"]!=1:
        raise ValueError("Unknown live planning settings")
    if (settings["global_batch"],settings["sequence_length"],settings["world_size"],settings["physical_microbatch"],settings["precision"])!=(256,2048,3,1,"fp32"):
        raise ValueError("Live demo settings must match validated local production shape")
    if settings["tokens"]%(256*2048) or not 4<=settings["tokens"]//(256*2048)<=32:
        raise ValueError("Live demo requires4–32 complete logical batches")
    if not settings["run_prefix"].replace("-","").isalnum():
        raise ValueError("Invalid run prefix")
    validation=IncrementalSource(settings["validation_root"],wait_seconds=0)
    if not (validation.root/"complete.json").exists() or validation.recipe.purpose!="validation":
        raise ValueError("Live validation must be complete before freezing training")
    if validation.recipe.source_identity.get("kind")!="pinned_culturax_parquet_prefix_v1":
        raise ValueError("Live validation must be real pinned source, not a timing replay")
    predecessor={"path":str(validation.root),"recipe_sha256":validation.recipe.sha256,
                 "completion_sha256":file_hash(validation.root/"complete.json")}
    recipe=replace(validation.recipe,purpose="train",output_tokens=settings["tokens"],
                   block_tokens=settings["block_tokens"],document_order_seed=settings["document_order_seed"],
                   predecessor_streams=[predecessor])
    outputs=[]
    for role,block in [("incremental",settings["block_tokens"]),("reference",settings["reference_block_tokens"])]:
        r=replace(recipe,block_tokens=block);r.validate()
        name=f"{settings['run_prefix']}-{role}"
        recipe_path=limits.report/"configs"/f"recipe-{name}.json"
        immutable_write(recipe_path,json_bytes(asdict(r)))
        root=limits.work/"data"/name
        _initialize(SimpleNamespace(recipe=str(recipe_path),root=str(root)),limits)
        cfg=make_config(limits,name=name,world_size=settings["world_size"],microbatch=settings["physical_microbatch"],
                        precision=settings["precision"],batches=settings["tokens"]//(256*2048),train_root=root,
                        validation_root=validation.root,save_checkpoint=True)
        layout=make_layout(limits,name,[(cfg,list(range(settings["world_size"])))])
        outputs.append({"role":role,"root":str(root),"recipe":str(recipe_path),"training_config":str(cfg),"layout":str(layout)})
    immutable_write(limits.report/"configs"/f"{settings['run_prefix']}-frozen.json",json_bytes(outputs))
    print(json.dumps(outputs,indent=2))


def compare(left,right,output,limits):
    import hashlib
    sources=[IncrementalSource(p,wait_seconds=0) for p in [left,right]]
    results=[]
    for s in sources:
        if not (s.root/"complete.json").exists():
            raise ValueError("Comparison requires two completed streams")
        h=hashlib.sha256()
        for array in s.arrays:h.update(array.tobytes())
        boundaries=[b for r in s.records for b in r["boundaries"]]
        results.append({"root":str(s.root),"tokens":s.token_count,"data_sha256":h.hexdigest(),
                        "boundaries_sha256":hashlib.sha256(json_bytes(boundaries)).hexdigest(),
                        "completion_sha256":file_hash(s.root/"complete.json")})
    recipes=[asdict(s.recipe) for s in sources]
    for r in recipes:r.pop("block_tokens")
    if recipes[0]!=recipes[1]:raise ValueError("Scientific preparation recipes differ beyond block size")
    for key in ["tokens","data_sha256","boundaries_sha256"]:
        if results[0][key]!=results[1][key]:raise ValueError(f"Reference comparison failed: {key}")
    immutable_write(limits.owned(output),json_bytes({"status":"exact_logical_data_match","streams":results}))
    print(json.dumps({"status":"exact_logical_data_match","output":str(output)}))


def freeze_sample(settings_path,language,limits):
    from lm_cl.cli.local_pipeline_demo import _initialize
    from types import SimpleNamespace
    settings=json.loads(Path(settings_path).read_text())
    if set(settings)!={"schema_version","languages","validation_root","source_inputs","output_tokens","block_tokens","document_order_seed"} or settings["schema_version"]!=1:
        raise ValueError("Unknown continual-language sampling settings")
    if settings["languages"]!=["en","zh_written","fr","ja","es","de","pt","ru"] or language not in settings["languages"]:
        raise ValueError("Sampling language/order differs from the frozen continual route")
    if set(settings["source_inputs"])!=set(settings["languages"]):
        raise ValueError("Every sampled language requires an explicit source metadata path")
    validation=IncrementalSource(settings["validation_root"],wait_seconds=0)
    predecessors=[]
    if validation.recipe.purpose!="validation" or validation.recipe.source_identity.get("kind")!="pinned_culturax_parquet_prefix_v1":
        raise ValueError("Continual samples require real pinned validation, not a timing replay")
    roots=[validation.root,*[limits.work/"data"/f"sample-{prior}-v1" for prior in settings["languages"][:settings["languages"].index(language)]]]
    for root in roots:
        source=IncrementalSource(root,wait_seconds=0)
        if not (source.root/"complete.json").exists():
            raise ValueError("Every prior ownership stream must be complete before sampling the next")
        predecessors.append({"path":str(source.root),"recipe_sha256":source.recipe.sha256,"completion_sha256":file_hash(source.root/"complete.json")})
    identity=json.loads(Path(settings["source_inputs"][language]).read_text())
    expected="zh" if language=="zh_written" else language
    if identity.get("language_config")!=expected or identity.get("revision")!=validation.recipe.source_identity.get("revision"):
        raise ValueError("Sample source revision/language mapping differs")
    recipe=replace(validation.recipe,source_identity=identity,language=language,purpose="train",
                   output_tokens=settings["output_tokens"],block_tokens=settings["block_tokens"],
                   document_order_seed=settings["document_order_seed"],predecessor_streams=predecessors)
    recipe.validate();name=f"sample-{language}-v1"
    path=limits.report/"configs"/f"recipe-{name}.json";immutable_write(path,json_bytes(asdict(recipe)))
    _initialize(SimpleNamespace(recipe=str(path),root=str(limits.work/"data"/name)),limits)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--limits",required=True)
    sub=p.add_subparsers(dest="command",required=True)
    q=sub.add_parser("freeze-vi");q.add_argument("--settings",required=True)
    q=sub.add_parser("freeze-sample");q.add_argument("--settings",required=True);q.add_argument("--language",required=True)
    q=sub.add_parser("compare");q.add_argument("--left",required=True);q.add_argument("--right",required=True);q.add_argument("--output",required=True)
    args=p.parse_args();limits=Limits(args.limits)
    if args.command=="freeze-vi":freeze_vi(args.settings,limits)
    elif args.command=="freeze-sample":freeze_sample(args.settings,args.language,limits)
    else:compare(args.left,args.right,args.output,limits)

if __name__=="__main__":
    main()
