"""Tiny offline tests exercise production storage, trainers and job orchestration."""
from contextlib import contextmanager
from dataclasses import replace
import json
from pathlib import Path
import threading

import pytest
import torch

from lm_cl.data.alternating import AlternatingProducer, acknowledge, advance, control
from lm_cl.data.incremental import file_hash
from lm_cl.data.streaming import StreamingPackedSource, block_path, connect, load_plan, receipt
from lm_cl.launcher import jobs as launcher_jobs
from lm_cl.launcher.streaming import resolve_streaming_contract
from lm_cl.training import ContinualTrainer, ProbeTrainer
from lm_cl.training.checkpoint import load_checkpoint
from test_production_streaming import study, Rows, Tokenizer


def alternating(tmp_path, monkeypatch, *, chunks=1, cycles=2):
    cfg, legacy = study(tmp_path, monkeypatch, cycles=cycles)
    cfg = replace(cfg, data=replace(cfg.data, streaming={**cfg.data.streaming,
        "block_tokens": 8, "token_cache_bytes": 256, "schedule": "alternating", "chunk_batches": chunks}))
    cfg.validate()
    contract = resolve_streaming_contract(cfg, prepare=True)
    return cfg, contract, legacy


@contextmanager
def producer(root):
    stop = threading.Event(); errors = []
    def work():
        try:
            engine = AlternatingProducer(root, source_factory=Rows, tokenizer=Tokenizer())
            while not stop.is_set():
                for path in sorted((Path(root) / "requests").glob("*.json")):
                    engine.ensure(json.loads(path.read_text())["ordinal"])
                    path.unlink(missing_ok=True)
                stop.wait(.002)
        except BaseException as exc: errors.append(exc)
    thread = threading.Thread(target=work, daemon=True); thread.start()
    try: yield
    finally:
        stop.set(); thread.join(10)
        assert not thread.is_alive()
        if errors: raise errors[0]


def test_compact_receipts_sparse_states_and_consumer_barrier(tmp_path, monkeypatch):
    cfg, contract, _ = alternating(tmp_path, monkeypatch)
    root = Path(contract["streaming_root"]); plan = load_plan(root)
    engine = AlternatingProducer(root, source_factory=Rows, tokenizer=Tokenizer())
    engine.ensure(1)
    assert not advance(root)
    assert len(list((root / "cache").glob("*.bin"))) == 2
    engine.ensure(2)  # Future turn is not admitted.
    assert receipt(root, 2) is None
    with connect(root) as db:
        assert db.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0] == 1
        assert db.execute("SELECT COUNT(*) FROM heads").fetchone()[0] == 1
        assert db.execute("SELECT COUNT(*) FROM receipts WHERE state != ''").fetchone()[0] == 0
        assert db.execute("SELECT length(hash) FROM content_hashes LIMIT 1").fetchone()[0] == 32
    for i in range(2):
        record, _ = receipt(root, i)
        assert "producer_state" not in record and "boundaries" not in record
    jobs = launcher_jobs.expand_job_specs(cfg, contract)
    for index, job in enumerate(jobs):
        with producer(root):
            result = ContinualTrainer(launcher_jobs.build_continual_job_config(cfg, job)).run(stop_after_global_logical_batches=1)
        if index == 1:
            assert load_checkpoint(result.checkpoint_path)["trainer_state"]["window_logical_batches"] == 1
        with pytest.raises(ValueError, match="checksum"):
            acknowledge(root, job.job_id, result.checkpoint_path, "0" * 64)
        with pytest.raises(ValueError, match="another job/study"):
            acknowledge(root, jobs[1 - index].job_id, result.checkpoint_path, result.checkpoint_sha256)
        acknowledge(root, job.job_id, result.checkpoint_path, result.checkpoint_sha256)
        assert advance(root) == (index == 1)
        if index == 0: assert len(list((root / "cache").glob("*.bin"))) == 2
    assert not list((root / "cache").glob("*.bin"))
    assert control(root, plan)["turn"] == 1
    engine.ensure(3)
    assert receipt(root, 3) is not None
    expected = receipt(root, 0)[0]["data_sha256"]
    engine.generate(0)  # Replay verifies history, but cannot republish released queue bytes.
    assert receipt(root, 0)[0]["data_sha256"] == expected
    assert not block_path(root, plan, 0).exists()


def test_compact_reconstruction_after_commit_before_publication(tmp_path, monkeypatch):
    import lm_cl.data.alternating as module
    _, contract, _ = alternating(tmp_path, monkeypatch)
    root = Path(contract["streaming_root"])
    original = module.publish
    def crash(*args): raise InterruptedError("after receipt commit")
    monkeypatch.setattr(module, "publish", crash)
    with pytest.raises(InterruptedError):
        AlternatingProducer(root, source_factory=Rows, tokenizer=Tokenizer()).ensure(0)
    identity = receipt(root, 0)[1]
    monkeypatch.setattr(module, "publish", original)
    AlternatingProducer(root, source_factory=Rows, tokenizer=Tokenizer()).ensure(0)
    assert receipt(root, 0)[1] == identity
    assert block_path(root, load_plan(root), 0).stat().st_size == 32


def test_release_recovery_and_pinned_corruption(tmp_path, monkeypatch):
    import lm_cl.data.alternating as module
    cfg, contract, _ = alternating(tmp_path, monkeypatch, chunks=3, cycles=1)
    root = Path(contract["streaming_root"]); plan = load_plan(root)
    jobs = launcher_jobs.expand_job_specs(cfg, contract)
    for job in jobs:
        with producer(root):
            result = ContinualTrainer(launcher_jobs.build_continual_job_config(cfg, job)).run(stop_after_task_boundaries=1)
        acknowledge(root, job.job_id, result.checkpoint_path, result.checkpoint_sha256)
    pinned = {p.name: file_hash(p) for p in (root / "pinned").glob("*.bin")}
    assert pinned
    original = module._release_files
    def crash(*args): raise InterruptedError("after release watermark commit")
    monkeypatch.setattr(module, "_release_files", crash)
    with pytest.raises(InterruptedError): advance(root)
    assert control(root, plan)["released_tokens"] == 48
    assert list((root / "cache").glob("*.bin"))
    monkeypatch.setattr(module, "_release_files", original)
    module.recover_release(root)
    assert not list((root / "cache").glob("*.bin"))
    assert {p.name: file_hash(p) for p in (root / "pinned").glob("*.bin")} == pinned
    source = StreamingPackedSource(root, "validation-en")
    path = block_path(root, plan, source.blocks[0]["ordinal"])
    path.write_bytes(b"0" * path.stat().st_size)
    with pytest.raises(ValueError, match="Corrupt"):
        source.read_tokens(8)


def test_reconstruction_from_selected_point_and_corrupt_snapshot(tmp_path, monkeypatch):
    _, contract, _ = alternating(tmp_path, monkeypatch, chunks=1)
    root = Path(contract["streaming_root"]); plan = load_plan(root)
    engine = AlternatingProducer(root, source_factory=Rows, tokenizer=Tokenizer())
    for i in range(6): engine.generate(i)
    before = {i: receipt(root, i)[1] for i in range(6)}
    # The head covers block 5; block 3 must replay from selected state 1.
    path = block_path(root, plan, 3); path.unlink()
    engine.generate(3)
    assert {i: receipt(root, i)[1] for i in range(6)} == before
    with connect(root) as db:
        db.execute("UPDATE snapshots SET state=? WHERE stream='train-en' AND ordinal=1", (b"corrupt",))
    import zlib
    with pytest.raises(zlib.error): engine.generate(3)


def test_two_rank_partial_k_resume_after_acknowledged_release(tmp_path, monkeypatch):
    import socket
    import torch.multiprocessing as mp
    from lm_cl.config import DistributedConfig, save_continual_config
    from lm_cl.training.distributed import state_digest
    from test_production_streaming import _production_ddp_worker, worker
    cfg, _, _ = alternating(tmp_path, monkeypatch)
    cfg = replace(cfg, experiment=replace(cfg.experiment, models=["fastmem_rmt"]))
    contract = resolve_streaming_contract(cfg, prepare=True)
    job = launcher_jobs.expand_job_specs(cfg, contract)[0]
    distributed = DistributedConfig(True, "gloo", 30, "contiguous_floor_v1",
        "ddp_average_world_scaled_global_sum_v1", "sum_unscale_normalize_clip_rank0_broadcast_v1",
        False, False, True, False)
    continual = replace(launcher_jobs.build_continual_job_config(cfg, job), distributed=distributed)
    root = Path(contract["streaming_root"])
    def run(config, tag, context, resume=None, stop=2):
        path = tmp_path / f"{tag}.yaml"; save_continual_config(config, path)
        output = tmp_path / f"{tag}.json"
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0)); port = sock.getsockname()[1]
        with context:
            mp.spawn(_production_ddp_worker, args=(port, str(path), str(output), resume, stop), nprocs=2, join=True)
        return json.loads(output.read_text())["checkpoint"]
    partial = run(continual, "ddp-first-turn", producer(root), stop=1)
    acknowledge(root, job.job_id, partial, file_hash(Path(partial)))
    assert advance(root) and not list((root / "cache").glob("*.bin"))
    resumed = run(continual, "ddp-second-turn", producer(root), resume=partial)
    reference = replace(cfg, experiment=replace(cfg.experiment, name="ddp-reference"),
        data=replace(cfg.data, streaming={k: v for k, v in cfg.data.streaming.items() if k not in {"schedule", "chunk_batches"}}))
    ref_contract = resolve_streaming_contract(reference, prepare=True)
    ref_job = launcher_jobs.expand_job_specs(reference, ref_contract)[0]
    ref_config = replace(launcher_jobs.build_continual_job_config(reference, ref_job), distributed=distributed)
    full = run(ref_config, "ddp-uninterrupted", worker(Path(ref_contract["streaming_root"])))
    left, right = load_checkpoint(resumed), load_checkpoint(full)
    for key in ("model_state", "optimizer_state", "gradients", "trainer_state", "memory_state", "rng_state"):
        assert state_digest(left[key]) == state_digest(right[key]), key
    assert left["distributed_state"]["world_size"] == 2


def test_real_supervisor_and_subprocess_turns_on_one_slot(tmp_path, monkeypatch):
    import os
    import subprocess
    import sys
    from lm_cl.launcher.alternating import TurnScheduler
    from lm_cl.launcher.scheduler import allocate_job_slots
    from lm_cl.launcher.streaming import producer_service
    cfg, contract, _ = alternating(tmp_path, monkeypatch, cycles=1)
    cfg = replace(cfg, launcher=replace(cfg.launcher, max_parallel_jobs=1))
    jobs = launcher_jobs.expand_job_specs(cfg, contract)
    for job in jobs: launcher_jobs.write_resolved_job(job)
    assignments = [replace(a, command=[*a.command, "--alternating-turn", "0", "--retry-resume"])
                   for a in allocate_job_slots(cfg, jobs)]
    assert {a.slot_index for a in assignments} == {0}
    original = subprocess.Popen
    repository = Path(__file__).resolve().parents[1]
    def offline_process(command, **kwargs):
        if isinstance(command, list) and command[:2] == [sys.executable, "-m"]:
            env = dict(kwargs.get("env", os.environ), PYTHONPATH=f"{repository}/src:{repository}/tests",
                       OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="")
            kwargs["env"] = env
            if command[2] == "lm_cl.data.streaming_producer":
                script = ("import sys; from test_production_streaming import Rows, Tokenizer; "
                          "from lm_cl.data.alternating import AlternatingProducer; "
                          "AlternatingProducer(sys.argv[1], source_factory=Rows, tokenizer=Tokenizer()).serve()")
                command = [sys.executable, "-c", script, *command[3:]]
            elif command[2] == "lm_cl.cli.run_experiment_job":
                script = ("from pathlib import Path; import torch; torch.set_num_threads(1); "
                          "from test_calibration import tiny_config; "
                          "import lm_cl.launcher.jobs as jobs; "
                          "jobs._model_config=lambda cfg: tiny_config(Path('/unused-offline-config'), 1).model; "
                          "from lm_cl.cli.run_experiment_job import main; main()")
                command = [sys.executable, "-c", script, *command[3:]]
        return original(command, **kwargs)
    monkeypatch.setattr(subprocess, "Popen", offline_process)
    root = Path(contract["streaming_root"])
    with producer_service(contract):
        results = TurnScheduler(cfg, jobs, assignments).run()
    assert [result["status"] for result in results] == ["yielded", "yielded"]
    for result, job in zip(results, jobs):
        acknowledge(root, job.job_id, result["final_checkpoint_path"], result["final_checkpoint_sha256"])
    assert advance(root)
    assert not list((root / "cache").glob("*.bin"))


def test_cli_selection_and_queue_capacity(tmp_path):
    from lm_cl.cli.a100_run import parser, build_config
    from lm_cl.launcher.scheduler import _checkpoint_estimate
    args = ["--model-size", "5m", "--data-root", str(tmp_path / "data"), "--output-root", str(tmp_path / "out")]
    legacy = build_config(parser().parse_args(args))
    assert "schedule" not in legacy.data.streaming
    current = build_config(parser().parse_args([*args, "--streaming-schedule", "alternating"]))
    assert current.data.streaming["schedule"] == "alternating"
    assert _checkpoint_estimate(current, 2)["estimated_alternating_turn_checkpoints_per_job"] == 0
    small = build_config(parser().parse_args([*args, "--streaming-schedule", "alternating", "--streaming-chunk-batches", "512"]))
    assert _checkpoint_estimate(small, 2)["estimated_alternating_turn_checkpoints_per_job"] == 120
    with pytest.raises(ValueError, match="Chunk size"):
        build_config(parser().parse_args([*args, "--streaming-chunk-batches", "2"]))
    with pytest.raises(ValueError, match="queue must fit"):
        replace(current, data=replace(current.data, streaming={**current.data.streaming, "token_cache_bytes": 128 * 1024**2})).validate()


def test_cycle_checkpoint_retention_contract_and_estimate(tmp_path):
    from lm_cl.cli.a100_run import parser, build_config
    from lm_cl.launcher.scheduler import _checkpoint_estimate
    args = ["--model-size", "5m", "--data-root", str(tmp_path / "data"),
            "--output-root", str(tmp_path / "out")]
    base = build_config(parser().parse_args([*args, "--streaming-schedule", "alternating"]))
    assert "checkpoint_retention" not in base.data.streaming
    retained = build_config(parser().parse_args([*args, "--streaming-schedule", "alternating",
                                                 "--checkpoint-retention", "cycle"]))
    assert retained.data.streaming["checkpoint_retention"] == "cycle_end_v1"
    assert _checkpoint_estimate(retained, 2)["estimated_checkpoints_per_job"] == 18
    for additions, message in [
        (["--checkpoint-retention", "cycle"], "requires alternating"),
        (["--streaming-schedule", "alternating", "--checkpoint-retention", "cycle",
          "--streaming-chunk-batches", "512"], "one turn per language"),
        (["--streaming-schedule", "alternating", "--checkpoint-retention", "cycle",
          "--checkpoint-every-batches", "10"], "no periodic saves"),
    ]:
        with pytest.raises(ValueError, match=message):
            build_config(parser().parse_args([*args, *additions]))


def test_preflight_counts_pinned_data_separately(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import lm_cl.launcher.scheduler as module
    cfg, contract, _ = alternating(tmp_path, monkeypatch)
    jobs = launcher_jobs.expand_job_specs(cfg, contract)
    monkeypatch.setattr(module, "inspect_environment", lambda: {})
    monkeypatch.setattr(module.shutil, "disk_usage", lambda path: SimpleNamespace(free=10**15))
    result = module.preflight_launch(cfg, jobs, module.allocate_job_slots(cfg, jobs))
    disk = result["disk"]
    stream = disk["streaming"]
    limits = load_plan(contract["streaming_root"])["limits"]
    assert stream["pinned_token_bytes"] == (32 + 9 * 16) * 4
    assert stream["cache_and_metadata_reserve_bytes"] == limits["token_cache_bytes"] + limits["metadata_bytes"]
    assert stream["total_streaming_reserve_bytes"] == stream["cache_and_metadata_reserve_bytes"] + stream["pinned_token_bytes"]
    assert disk["required_bytes"] == (disk["estimated_total_checkpoint_bytes"] + stream["total_streaming_reserve_bytes"]
                                      + max(cfg.launcher.disk_free_floor_bytes, limits["minimum_free_bytes"]))


def test_alternating_calibration_rejects_crossing_a_turn(tmp_path, monkeypatch):
    from lm_cl.cli.calibrate_continual import validate_window
    cfg, _, _ = alternating(tmp_path, monkeypatch)
    cfg = replace(cfg, experiment=replace(cfg.experiment, tokens_per_task=128))
    contract = resolve_streaming_contract(cfg, prepare=True)
    job = launcher_jobs.expand_job_specs(cfg, contract)[0]
    continual = launcher_jobs.build_continual_job_config(cfg, job)
    with pytest.raises(ValueError, match="initial alternating turn"):
        validate_window(continual, warmup=2, batches=8, world_size=1)


def tiny_probe_vocabulary(monkeypatch):
    from lm_cl.config.probe_schema import ProbeExperimentConfig
    original = ProbeExperimentConfig._validate_source
    def validate(self, source, **kwargs):
        if source.kind == "streaming_packed":
            from lm_cl.data.streaming import source_from_pipeline
            assert source_from_pipeline(source.packed).plan["tokenizer_reference"]["model_embedding_vocab_size"] == self.model.vocab_size == 16
        else: original(self, source, **kwargs)
    monkeypatch.setattr(ProbeExperimentConfig, "_validate_source", validate)


def inline_stage(controller, command, *, environment):
    """Execute the real CLI/trainer in-process so the test vocabulary stays tiny."""
    import importlib
    import sys
    module = command[command.index("-m") + 1]
    assert module in {"lm_cl.cli.train_continual", "lm_cl.cli.resume_continual",
                      "lm_cl.cli.run_probe", "lm_cl.cli.resume_probe"}
    if module == "lm_cl.cli.resume_continual":
        payload = load_checkpoint(command[command.index("-m") + 3])
        identity = payload["source_identity"]
        if identity.get("kind") == "streaming_packed":
            from lm_cl.data.streaming import ALTERNATING_FORMAT
            plan = load_plan(identity["root"])
            if plan["format"] == ALTERNATING_FORMAT:
                index = control(identity["root"], plan)["turn"]
                expected = 0 if index == 0 else plan["alternation"]["turns"][index - 1]["end_batches"]
                assert payload["trainer_state"]["global_logical_batches"] == expected
    old = sys.argv
    try:
        sys.argv = [module, *command[command.index("-m") + 2:]]
        importlib.import_module(module).command()
    finally: sys.argv = old


def test_real_job_runner_pair_two_cycles_matches_independent_with_pinned_data(tmp_path, monkeypatch):
    from lm_cl.launcher.runner import run_resolved_job, StageProcessController
    from lm_cl.launcher.scheduler import allocate_job_slots
    import lm_cl.launcher.alternating as scheduling
    from lm_cl.training.distributed import state_digest
    from test_production_streaming import worker
    cfg, contract, _ = alternating(tmp_path, monkeypatch, chunks=2)
    tiny_probe_vocabulary(monkeypatch)
    monkeypatch.setattr(StageProcessController, "run", inline_stage)
    jobs = launcher_jobs.expand_job_specs(cfg, contract)
    for job in jobs: launcher_jobs.write_resolved_job(job)
    root = Path(contract["streaming_root"]); plan = load_plan(root)
    seen = []
    def run_turn(scheduler):
        values = []
        for assignment in scheduler.assignments:
            index = int(assignment.command[assignment.command.index("--alternating-turn") + 1])
            seen.append((index, assignment.job_id))
            values.append(run_resolved_job(Path(assignment.output_dir) / "resolved_experiment.yaml",
                          rendezvous_port=assignment.rendezvous_port, retry_resume=True, alternating_turn=index))
            assert sum(p.stat().st_size for p in (root / "cache").iterdir()) <= plan["limits"]["token_cache_bytes"]
        return values
    monkeypatch.setattr(scheduling.TurnScheduler, "run", run_turn)
    original_ack = scheduling.acknowledge
    injected = False
    def fail_after_ack(*args):
        nonlocal injected
        original_ack(*args)
        if not injected:
            injected = True
            raise InterruptedError("crash after first durable consumer ack")
    monkeypatch.setattr(scheduling, "acknowledge", fail_after_ack)
    with producer(root), pytest.raises(InterruptedError):
        scheduling.run_alternating(cfg, jobs, allocate_job_slots(cfg, jobs))
    assert control(root, plan)["released_tokens"] == 0
    assert list((root / "cache").glob("*.bin"))
    monkeypatch.setattr(scheduling, "acknowledge", original_ack)
    with producer(root):
        summaries = scheduling.run_alternating(cfg, jobs, allocate_job_slots(cfg, jobs))
    assert not list((root / "cache").glob("*.bin"))
    pinned = {p.name: file_hash(p) for p in (root / "pinned").glob("*.bin")}
    assert sum(p.stat().st_size for p in (root / "pinned").glob("*.bin")) == (32 + 9 * 16) * 4
    assert all(s["status"] == "complete" and s["completed_language_tasks"] == 16 for s in summaries)
    assert [index for index, _ in seen] == sorted(index for index, _ in seen)
    # Identical input algorithm, conventional complete trajectories as reference.
    settings = {k: v for k, v in cfg.data.streaming.items() if k not in {"schedule", "chunk_batches"}}
    reference = replace(cfg, experiment=replace(cfg.experiment, name="independent-reference"),
                        data=replace(cfg.data, streaming=settings))
    reference_contract = resolve_streaming_contract(reference, prepare=True)
    reference_root = Path(reference_contract["streaming_root"])
    reference_jobs = launcher_jobs.expand_job_specs(reference, reference_contract)
    for job in reference_jobs: launcher_jobs.write_resolved_job(job)
    with worker(reference_root):
        expected = [run_resolved_job(Path(job.output_dir) / "resolved_experiment.yaml", rendezvous_port=29500)
                    for job in reference_jobs]
    for actual, wanted in zip(summaries, expected):
        left, right = load_checkpoint(actual["final_checkpoint_path"]), load_checkpoint(wanted["final_checkpoint_path"])
        for key in ("model_state", "optimizer_state", "scheduler_state", "gradients", "rng_state", "memory_state", "trainer_state"):
            assert state_digest(left[key]) == state_digest(right[key]), key
        for a, b in zip(actual["per_cycle_probe_auc"], wanted["per_cycle_probe_auc"]):
            assert a["normalized_token_auc"] == b["normalized_token_auc"]
    assert {p.name: file_hash(p) for p in (root / "pinned").glob("*.bin")} == pinned
    for i in range(len(plan["blocks"])):
        assert receipt(root, i)[0]["data_sha256"] == receipt(reference_root, i)[0]["data_sha256"]
    with connect(root) as db:
        assert db.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0] < len(plan["blocks"]) // 2
        compact_bytes = db.execute("SELECT SUM(length(record)+length(state)) FROM receipts").fetchone()[0]
    with connect(reference_root) as db:
        old_bytes = db.execute("SELECT SUM(length(record)+length(state)) FROM receipts").fetchone()[0]
    assert compact_bytes < old_bytes / 4  # Detailed state no longer grows per block.


def test_cycle_retention_keeps_probe_sources_and_recovers_after_ledger_crash(tmp_path, monkeypatch):
    from lm_cl.launcher.runner import run_resolved_job, StageProcessController
    from lm_cl.launcher.scheduler import allocate_job_slots, _checkpoint_estimate
    import lm_cl.launcher.alternating as scheduling
    import lm_cl.launcher.checkpoint_retention as retention

    cfg, _, _ = alternating(tmp_path, monkeypatch, chunks=3, cycles=2)
    cfg = replace(cfg, data=replace(cfg.data, streaming={**cfg.data.streaming,
        "checkpoint_retention": "cycle_end_v1"}))
    cfg.validate()
    contract = resolve_streaming_contract(cfg, prepare=True)
    root = Path(contract["streaming_root"])
    plan = load_plan(root)
    assert plan["checkpoint_retention"] == "cycle_end_v1"
    assert _checkpoint_estimate(cfg, 2)["estimated_checkpoints_per_job"] == 9
    tiny_probe_vocabulary(monkeypatch)
    monkeypatch.setattr(StageProcessController, "run", inline_stage)
    jobs = launcher_jobs.expand_job_specs(cfg, contract)
    for job in jobs:
        launcher_jobs.write_resolved_job(job)

    def run_turn(scheduler):
        return [run_resolved_job(Path(a.output_dir) / "resolved_experiment.yaml",
                                rendezvous_port=a.rendezvous_port, retry_resume=True,
                                alternating_turn=int(a.command[a.command.index("--alternating-turn") + 1]))
                for a in scheduler.assignments]
    monkeypatch.setattr(scheduling.TurnScheduler, "run", run_turn)
    original = retention.atomic_write_json
    crashed = False
    def crash_after_retirement_record(path, value):
        nonlocal crashed
        original(path, value)
        if Path(path).name == "checkpoint_retention.json" and not crashed:
            crashed = True
            raise InterruptedError("after durable retirement record")
    monkeypatch.setattr(retention, "atomic_write_json", crash_after_retirement_record)
    with producer(root), pytest.raises(InterruptedError, match="retirement record"):
        scheduling.run_alternating(cfg, jobs, allocate_job_slots(cfg, jobs))
    assert crashed and control(root, plan)["turn"] == 2
    assert (Path(jobs[0].output_dir) / "checkpoints/task-0000-en-boundary.pt").exists()
    monkeypatch.setattr(retention, "atomic_write_json", original)
    with producer(root):
        summaries = scheduling.run_alternating(cfg, jobs, allocate_job_slots(cfg, jobs))
    assert all(s["status"] == "complete" for s in summaries)
    for job, summary in zip(jobs, summaries):
        directory = Path(job.output_dir)
        names = {p.name for p in (directory / "checkpoints").glob("*.pt")}
        assert names == {"task-0007-ru-boundary.pt", "cycle-0001-complete.pt",
                         "task-0015-ru-boundary.pt", "cycle-0002-complete.pt"}
        assert summary["final_checkpoint_path"] == str(directory / "checkpoints/cycle-0002-complete.pt")
        for cycle, probe in enumerate(summary["per_cycle_probe_auc"], 1):
            source = probe["source_checkpoint"]["path"]
            assert Path(source).name == f"task-{cycle * 8 - 1:04d}-ru-boundary.pt"
            assert Path(source).is_file()
        assert len(list((directory / "probes").rglob("probe-complete-step-*.pt"))) == 2
        ledger = json.loads((directory / "checkpoint_retention.json").read_text())
        assert len(ledger["retired"]) == 14
        assert all(value["sha256"] for value in ledger["retired"].values())
