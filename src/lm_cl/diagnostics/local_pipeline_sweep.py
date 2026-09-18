"""Run a frozen list of bounded GPU layouts sequentially, stopping on failure."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from lm_cl.diagnostics.local_pipeline_bench import run_layout
from lm_cl.diagnostics.local_pipeline_resources import Limits

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--limits",required=True)
    p.add_argument("--plans",required=True)
    p.add_argument("--start-index",type=int,default=0)
    args=p.parse_args(); limits=Limits(args.limits)
    plans=json.loads(Path(args.plans).read_text())
    if not isinstance(plans,list) or not 0<=args.start_index<len(plans):
        raise ValueError("Invalid frozen plan list or start index")
    for index,path in enumerate(plans):
        plan=json.loads(Path(path).read_text())
        result=limits.report/"raw"/plan["label"]/"result.json"
        if index<args.start_index:
            if not result.exists() or json.loads(result.read_text())["status"]!="complete":
                raise ValueError("Cannot skip a layout without a successful result")
            continue
        print(json.dumps({"event":"layout_start","index":index,"plan":path}),flush=True)
        run_layout(Path(path),limits)
        if json.loads(result.read_text())["status"]!="complete":
            raise RuntimeError("Sweep stopped at failed layout; preserve evidence and review")

if __name__=="__main__":
    main()
