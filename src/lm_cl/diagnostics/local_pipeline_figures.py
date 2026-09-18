"""Standalone Matplotlib figures; speculative preparation curves are labelled."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--measurements",required=True);p.add_argument("--estimates",required=True);p.add_argument("--output-dir",required=True)
    args=p.parse_args();out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=False)
    measurements=json.loads(Path(args.measurements).read_text());estimates=json.loads(Path(args.estimates).read_text())
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"savefig.dpi":150})
    layouts=[l for l in measurements["layouts"] if l["status"]=="complete" and "v3" in l["label"] and "heldout" not in l["label"]]
    labels=[];rates=[]
    for l in layouts:
        labels.append(l["label"].replace("measure-","").replace("-fp32-mb1-v3","").replace("-v3",""))
        rates.append(l["aggregate_measured_input_tokens_per_second"]/1000)
    fig,ax=plt.subplots(figsize=(10,5)); y=np.arange(len(labels)); ax.barh(y,rates,color="#245e87")
    ax.set(yticks=y,yticklabels=labels,xlabel="Measured aggregate training input tokens/s (thousands)",
           title="Local GPU layouts: full logical batches and slow updates")
    ax.invert_yaxis()
    for i,value in enumerate(rates):ax.text(value+.15,i,f"{value:.1f}",va="center")
    fig.text(.01,.01,"FP32, global batch256, sequence2048, physical microbatch1. Cached repeated tokens; training cost only.",fontsize=8)
    fig.tight_layout(rect=(0,.04,1,1));fig.savefig(out/"gpu-layout-throughput.png");plt.close(fig)
    row=[r for r in estimates["forecasts"] if r["variant"]=="full_ag" and r["kind"] in {"probe","study","reduced"}]
    fig,ax=plt.subplots(figsize=(10,5));values=np.array([r["data_ready_seconds"]/3600 for r in row]);y=np.arange(len(row))
    low=values-np.array([r["planning_low_seconds"]/3600 for r in row]);high=np.array([r["planning_high_seconds"]/3600 for r in row])-values
    ax.barh(y,values,xerr=np.stack([low,high]),color="#347b60",capsize=3)
    ax.set(yticks=y,yticklabels=[r["scenario"] for r in row],xscale="log",xlabel="Hours, assuming data are ready (log scale)",
           title="5M AG experiment durations on three GPUs")
    ax.invert_yaxis();fig.text(.01,.01,"Engineering planning ranges, not confidence intervals. Download/startup unvalidated; probe evaluation basis in estimates.json.",fontsize=8)
    fig.tight_layout(rect=(0,.04,1,1));fig.savefig(out/"experiment-duration-estimates.png");plt.close(fig)
    scenarios=estimates["overlap_sensitivities"]
    fig,axes=plt.subplots(2,2,figsize=(10,7),sharey=True)
    fig2,ax2=plt.subplots(figsize=(9,4))
    for ax,s in zip(axes.flat,scenarios):
        total={"published":0,"consumed":0};times=[0];buffer=[0]
        for event in ["published","consumed"]:
            selected=[r for r in s["timeline"] if r["event"]==event]
            ax.step([0]+[r["seconds"]/60 for r in selected],[0]+[r["tokens"]/1e6 for r in selected],where="post",label=event)
        for r in s["timeline"]:
            total[r["event"]]=r["tokens"];times.append(r["seconds"]/60);buffer.append((total["published"]-total["consumed"])/1e6)
        ax.set(title=f"Assumed producer: {s['producer_tokens_per_second']:,} tokens/s",xlabel="Minutes",ylabel="Million input tokens")
        ax.legend(fontsize=8)
        ax2.step(times,buffer,where="post",label=f"{s['producer_tokens_per_second']:,} tok/s")
    fig.suptitle("Illustrative preparation/consumption timeline — no live preparation measurement")
    fig.tight_layout();fig.savefig(out/"illustrative-preparation-timeline.png");plt.close(fig)
    ax2.set(xlabel="Minutes",ylabel="Published but unconsumed tokens (millions)",title="Illustrative buffer occupancy — producer rates are assumptions")
    ax2.legend(title="Assumed producer rate",fontsize=8);fig2.tight_layout();fig2.savefig(out/"illustrative-buffer-occupancy.png");plt.close(fig2)
    print(json.dumps({"output":str(out),"figures":len(list(out.glob('*.png')))}))

if __name__=="__main__":
    main()
