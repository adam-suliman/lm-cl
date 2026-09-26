"""Barrier between bounded turns, using the ordinary job runner and GPU slots."""
from dataclasses import replace
import json
from pathlib import Path

from lm_cl.data.alternating import acknowledge, advance, control, recover_release
from lm_cl.data.streaming import load_plan
from lm_cl.launcher.scheduler import LocalJobScheduler
from lm_cl.launcher.checkpoint_retention import retire_completed_turn_checkpoints


class TurnScheduler(LocalJobScheduler):
    """One wave of model/seed turns; retains ordinary child supervision/retries."""


def run_alternating(config, jobs, assignments):
    root = Path(jobs[0].resolved_experiment["data_contract"]["streaming_root"])
    plan = load_plan(root)
    expected = set(plan["alternation"]["consumers"])
    if {job.job_id for job in jobs} != expected:
        raise ValueError("Alternating consumer set changed")
    recover_release(root, plan)
    retire_completed_turn_checkpoints(root, plan, jobs)
    while True:
        state = control(root, plan)
        index = state["turn"]
        if index == len(plan["alternation"]["turns"]):
            advance(root)  # Finish an interrupted release idempotently.
            break
        turn = plan["alternation"]["turns"][index]
        pending = []
        for assignment in assignments:
            ack = state["consumers"].get(assignment.job_id)
            if ack is None or ack["end_tokens"] < turn["end_tokens"]:
                pending.append(replace(assignment, command=[*assignment.command,
                    "--alternating-turn", str(index), "--retry-resume"]))
        if pending:
            results = TurnScheduler(config, jobs, pending).run()
            by_identity = {(job.public_model, job.seed): job.job_id for job in jobs}
            for result in results:
                if result.get("status") not in {"yielded", "complete"}:
                    raise RuntimeError("Alternating consumer failed; queue preserved")
                consumer = by_identity[(result["model"], result["seed"])]
                acknowledge(root, consumer, result["final_checkpoint_path"], result["final_checkpoint_sha256"])
        if not advance(root):
            raise RuntimeError("Alternating turn lacks verified consumer checkpoints")
        retire_completed_turn_checkpoints(root, plan, jobs)
    summaries = [json.loads((Path(job.output_dir) / "summary.json").read_text()) for job in jobs]
    if any(value.get("status") != "complete" for value in summaries):
        raise ValueError("Queue completion without complete experiment summaries")
    return summaries
