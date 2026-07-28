from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lm_cl.launcher.config import load_launcher_config
from lm_cl.launcher.data import resolve_data_contract
from lm_cl.launcher import data as launcher_data
from lm_cl.launcher.jobs import (
    build_continual_job_config,
    build_probe_job_config,
    expand_job_specs,
)
from lm_cl.launcher.runner import StageProcessController
from lm_cl.launcher.scheduler import (
    JobAssignment,
    LocalJobScheduler,
    allocate_job_slots,
)
from lm_cl.launcher.schema import (
    PUBLIC_LANGUAGE_ORDER,
    PUBLIC_MODEL_VARIANTS,
    resolve_token_budget,
)
from lm_cl.launcher.state import (
    migrate_checkpoint_horizon,
    pointer_for_checkpoint,
    validate_horizon_extension,
    validate_latest_pointer,
    write_latest_pointer,
)
from lm_cl.metrics import JsonlMetricLogger, TensorBoardTracker
from lm_cl.metrics import update_forgetting_metrics
from lm_cl.config import (
    DataConfig,
    TokenizerReference,
    TrainSourceConfig,
    load_model_config,
)
from lm_cl.models import RMTZyphraTransformer
from lm_cl.training import ContinualTrainer, ProbeTrainer
from lm_cl.training import continual as continual_training
from lm_cl.training.checkpoint import load_checkpoint, sha256_file
from lm_cl.training.distributed import state_digest
from lm_cl.data.sources import ArrayTokenSource


ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "configs/experiments/zyphra_fastmem_two_cycle_smoke.yaml"


def _config(
    tmp_path: Path,
    *,
    name: str = "test-public",
    models: list[str] | None = None,
    cycles: int = 2,
    seeds: list[int] | None = None,
    tensorboard: bool = False,
):
    config = load_launcher_config(
        SMOKE,
        overrides={
            "name": name,
            "models": models or ["transformer", "fastmem_rmt"],
            "seeds": seeds or [101],
            "cycles": cycles,
            "output_root": str(tmp_path),
            "resume": "never",
        },
    )
    return replace(
        config,
        tracking=replace(config.tracking, tensorboard=tensorboard),
    )


def _jobs(config):
    data = resolve_data_contract(config)
    return data, expand_job_specs(config, data)


def test_public_model_names_map_to_approved_variants():
    assert PUBLIC_MODEL_VARIANTS == {
        "transformer": "backbone_clean",
        "fastmem_rmt": "fastmem_rmt",
    }


def test_release_rejects_internal_variant_names(tmp_path):
    config = _config(tmp_path)
    bad = replace(
        config,
        experiment=replace(config.experiment, models=["backbone_matched_k"]),
    )
    with pytest.raises(ValueError, match="Only transformer and fastmem_rmt"):
        bad.validate()


def test_models_times_seeds_expand_to_complete_jobs(tmp_path):
    config = _config(tmp_path, seeds=[11, 12])
    _, jobs = _jobs(config)
    assert [job.job_id for job in jobs] == [
        "transformer-seed-11",
        "fastmem_rmt-seed-11",
        "transformer-seed-12",
        "fastmem_rmt-seed-12",
    ]


def test_inactive_new_launcher_fields_preserve_legacy_serialization(tmp_path):
    config = _config(
        tmp_path,
        name="legacy-launcher-identity",
        models=["transformer"],
        cycles=1,
        tensorboard=False,
    )
    serialized = config.to_dict()
    assert "forgetting" not in serialized
    assert "language_validation_manifest_template" not in serialized["data"]
    assert "window_source_tokens_per_task" not in serialized["data"]
    assert "ddp_debug_assert_synced" not in serialized["launcher"]


def test_zero_sequence_offsets_preserve_legacy_continual_serialization(tmp_path):
    config = _config(
        tmp_path,
        name="legacy-continual-identity",
        models=["transformer"],
        cycles=1,
        tensorboard=False,
    )
    _, jobs = _jobs(config)
    continual = build_continual_job_config(config, jobs[0])
    assert all(
        "train_sequence_offset_count" not in task
        for task in continual.to_dict()["tasks"]
    )
    assert "diagnostic_norm_interval_steps" not in continual.to_dict()[
        "runtime"
    ]


def test_h100_configs_disable_per_step_ddp_digests(monkeypatch, tmp_path):
    monkeypatch.setenv("LM_CL_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("LM_CL_OUTPUT_ROOT", str(tmp_path / "output"))
    for name in (
        "zyphra_fastmem_h100_5m_5cycle_1b.yaml",
        "zyphra_fastmem_h100_12m_5cycle_1b.yaml",
    ):
        config = load_launcher_config(ROOT / "configs/experiments" / name)
        assert config.launcher.ddp_debug_assert_synced is False
        assert config.tracking.log_every_batches == 100


def test_rmt_attention_mask_is_vectorized_and_cached():
    config = load_model_config(ROOT / "configs/models/tiny_test.yaml")
    model = RMTZyphraTransformer(
        config,
        memory_tokens=2,
        segment_length=4,
    )
    mask = model._attention_mask(
        4,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    again = model._attention_mask(
        4,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert mask.data_ptr() == again.data_ptr()
    assert tuple(mask.shape) == (8, 8)
    assert torch.equal(mask[:2, :2], torch.zeros(2, 2))
    assert bool(torch.isneginf(mask[:2, 2:]).all())
    assert torch.equal(mask[2:6, :2], torch.zeros(4, 2))
    expected_causal = torch.triu(
        torch.full((4, 4), float("-inf")),
        diagonal=1,
    )
    assert torch.equal(mask[2:6, 2:6], expected_causal)
    assert bool(torch.isneginf(mask[2:6, 6:]).all())
    assert torch.equal(mask[6:, :], torch.zeros(2, 8))


def test_one_job_contains_every_task_and_cycle(tmp_path):
    config = _config(tmp_path, models=["transformer"])
    _, jobs = _jobs(config)
    continual = build_continual_job_config(config, jobs[0])
    assert len(continual.tasks) == 16
    assert [task.language for task in continual.tasks] == list(
        PUBLIC_LANGUAGE_ORDER
    ) * 2


def test_token_budget_is_explicit_complete_sequence_floor():
    budget = resolve_token_budget(5_000_000_000, 2048)
    assert budget.effective_complete_sequences == 2_441_406
    assert budget.effective_input_tokens == 4_999_999_488
    assert budget.effective_valid_targets == 4_997_558_082
    assert budget.discarded_remainder_tokens == 512


def test_five_one_billion_windows_exactly_fit_one_five_billion_source():
    task = resolve_token_budget(1_000_000_000, 2048)
    source = resolve_token_budget(5_000_000_000, 2048)
    windows = [
        (cycle * task.effective_complete_sequences, task.effective_complete_sequences)
        for cycle in range(5)
    ]
    assert windows[-1][0] + windows[-1][1] <= source.effective_complete_sequences
    assert source.effective_complete_sequences - sum(
        count for _, count in windows
    ) == 1
    assert all(
        left_start + left_count == right_start
        for (left_start, left_count), (right_start, _) in zip(
            windows, windows[1:]
        )
    )


def test_two_cycle_config_can_reuse_a_frozen_five_billion_source(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("LM_CL_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("LM_CL_OUTPUT_ROOT", str(tmp_path / "output"))
    config = load_launcher_config(
        ROOT
        / "configs/experiments/zyphra_fastmem_h100_5m_2cycle_1b_8gpu.yaml"
    )
    task = resolve_token_budget(config.experiment.tokens_per_task, 2048)
    source = resolve_token_budget(
        config.data.window_source_tokens_per_task,
        2048,
    )
    assert config.experiment.cycles == 2
    assert launcher_data._source_task_budget(config) == source
    assert source.effective_complete_sequences >= (
        2 * task.effective_complete_sequences
    )


def test_sequence_window_reader_starts_at_offset_and_stops_at_exclusive_end():
    source = ArrayTokenSource(np.arange(40, dtype=np.uint32))
    start = source.position_for_global_sequence(2, sequence_length=4)
    batches = list(
        source.iter_batches(
            sequence_length=4,
            global_sequences_per_batch=2,
            start=start,
            sequence_prefix_count=4,
        )
    )
    assert len(batches) == 1
    assert batches[0].input_ids.tolist() == [
        [8, 9, 10, 11],
        [12, 13, 14, 15],
    ]


def test_windowed_data_contract_records_distinct_nonoverlapping_views(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("LM_CL_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("LM_CL_OUTPUT_ROOT", str(tmp_path / "output"))
    config = load_launcher_config(
        ROOT
        / "configs/experiments/zyphra_fastmem_h100_5m_5cycle_1b.yaml"
    )

    tokenizer_manifest = {
        "manifest_content_sha256": "1" * 64,
        "repo_id": "Qwen/Qwen3-0.6B-Base",
        "revision": "a" * 40,
    }
    monkeypatch.setattr(
        launcher_data,
        "_tokenizer_reference",
        lambda config: (
            TokenizerReference(
                repo_id=tokenizer_manifest["repo_id"],
                revision=tokenizer_manifest["revision"],
                manifest_path=str(
                    tmp_path / "data/generated/tokenizers/qwen3/manifest.json"
                ),
                base_vocab_size=151_643,
                effective_vocab_size=151_669,
                maximum_emitted_token_id=151_668,
                model_embedding_vocab_size=151_680,
                expected_eos_token_id=151_643,
                expected_pad_token_id=151_643,
            ),
            tokenizer_manifest,
            "2" * 64,
        ),
    )

    identity_calls = []

    def identity(path, **kwargs):
        identity_calls.append(str(path))
        budget = kwargs["budget"]
        language = kwargs["expected_language"]
        purpose = (
            "language_validation"
            if "retention-validation" in str(path)
            else (
                "vietnamese_validation"
                if "vi-validation" in str(path)
                else (
                    "vietnamese_train"
                    if "vi-probe-train" in str(path)
                    else "continual_train"
                )
            )
        )
        complete = (
            2_441_406
            if purpose in {"continual_train", "vietnamese_train"}
            else 1_280
        )
        sequence_length = 2048
        return {
            "manifest_path": str(path),
            "manifest_file_sha256": "3" * 64,
            "manifest_content_sha256": "4" * 64,
            "ordered_data_sha256": (language or "vi").encode().hex().ljust(64, "0")[:64],
            "tokenizer_manifest_path": "",
            "tokenizer_manifest_file_sha256": "2" * 64,
            "tokenizer_manifest_content_sha256": "1" * 64,
            "tokenizer_repo_id": tokenizer_manifest["repo_id"],
            "tokenizer_revision": tokenizer_manifest["revision"],
            "stage_id": Path(path).parent.name,
            "purpose": purpose,
            "language": language,
            "cycle_index": kwargs["expected_cycle"],
            "task_index": kwargs["expected_task_index"],
            "preparation_lane_namespace": None,
            "packed_token_count": complete * sequence_length,
            "packed_target_token_count": complete * (sequence_length - 1),
            "packed_complete_sequence_count": complete,
            "requested_input_tokens": None if budget is None else budget.requested_input_tokens,
            "effective_complete_sequences": complete if budget is None else budget.effective_complete_sequences,
            "effective_input_tokens": complete * sequence_length if budget is None else budget.effective_input_tokens,
            "effective_valid_targets": complete * (sequence_length - 1) if budget is None else budget.effective_valid_targets,
            "sequence_length": sequence_length,
        }

    monkeypatch.setattr(launcher_data, "_manifest_identity", identity)
    contract = resolve_data_contract(config, full_checksum_validation=False)
    en_windows = [
        cycle["en"]["sequence_window"]
        for cycle in contract["data_manifests"]
    ]
    assert [item["sequence_start"] for item in en_windows] == [
        0,
        488_281,
        976_562,
        1_464_843,
        1_953_124,
    ]
    assert len({item["view_ordered_data_sha256"] for item in en_windows}) == 5
    assert set(contract["language_validation_manifests"]) == set(
        PUBLIC_LANGUAGE_ORDER
    )
    assert len(identity_calls) == 18
    assert sum("zyphra-cycle-0000" in path for path in identity_calls) == 8


def test_trainer_reuses_validated_packed_source_and_detects_mutation(
    tmp_path, monkeypatch
):
    config = _config(
        tmp_path,
        name="packed-source-cache",
        models=["transformer"],
        cycles=1,
        tensorboard=False,
    )
    _, jobs = _jobs(config)
    trainer = ContinualTrainer(build_continual_job_config(config, jobs[0]))

    stage_dir = tmp_path / "packed-stage"
    stage_dir.mkdir()
    (stage_dir / "manifest.json").write_text("{}", encoding="utf-8")
    shard_path = stage_dir / "tokens-00000.bin"
    shard_path.write_bytes(b"1234")
    source = SimpleNamespace(
        stage_dir=stage_dir,
        manifest={
            "shards": [{"filename": shard_path.name}],
            "boundaries": None,
        },
    )
    source_config = SimpleNamespace(
        packed=SimpleNamespace(to_dict=lambda: {"source": "same"})
    )
    calls = []

    def fake_build_source(*args, **kwargs):
        calls.append((args, kwargs))
        return source, {"manifest_content_sha256": "a" * 64}, -100

    monkeypatch.setattr(continual_training, "build_source", fake_build_source)
    first = trainer._open_source(source_config)
    second = trainer._open_source(source_config)
    assert first[0] is second[0]
    assert len(calls) == 1

    shard_path.write_bytes(b"changed-size")
    with pytest.raises(ValueError, match="changed after in-process validation"):
        trainer._open_source(source_config)


def test_lower_is_better_forgetting_is_measured_from_best_history():
    first = {}
    best = {}
    _, first, best = update_forgetting_metrics(
        {"en": 2.0}, first_mean_ce=first, best_mean_ce=best
    )
    _, first, best = update_forgetting_metrics(
        {"en": 1.5, "zh_written": 3.0},
        first_mean_ce=first,
        best_mean_ce=best,
    )
    report, _, _ = update_forgetting_metrics(
        {"en": 1.8, "zh_written": 3.2},
        first_mean_ce=first,
        best_mean_ce=best,
    )
    assert report["languages"]["en"]["forgetting_from_best_ce"] == pytest.approx(0.3)
    assert report["languages"]["en"]["ce_change_from_first"] == pytest.approx(-0.2)
    assert report["languages"]["zh_written"]["forgetting_from_best_ce"] == pytest.approx(0.2)


def test_too_small_token_budget_fails():
    with pytest.raises(ValueError, match="at least one complete sequence"):
        resolve_token_budget(15, 16)


def test_missing_packed_contract_fails_before_launch(tmp_path, monkeypatch):
    monkeypatch.setenv("LM_CL_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("LM_CL_OUTPUT_ROOT", str(tmp_path / "output"))
    config = load_launcher_config(
        ROOT / "configs/experiments/zyphra_fastmem_a100.yaml"
    )
    with pytest.raises(FileNotFoundError):
        resolve_data_contract(config)


def test_cycle_manifest_order_is_exact(tmp_path):
    config = _config(tmp_path)
    data = resolve_data_contract(config)
    assert [list(cycle) for cycle in data["data_manifests"]] == [
        list(PUBLIC_LANGUAGE_ORDER),
        list(PUBLIC_LANGUAGE_ORDER),
    ]


def _cuda_launcher(config, *, gpu_ids, jobs_per_gpu=1, gpus_per_job=1):
    result = replace(
        config,
        training=replace(config.training, device="cuda", precision="fp32"),
        launcher=replace(
            config.launcher,
            gpu_ids=gpu_ids,
            jobs_per_gpu=jobs_per_gpu,
            gpus_per_job=gpus_per_job,
            max_parallel_jobs=max(1, len(gpu_ids) * jobs_per_gpu),
        ),
    )
    result.validate()
    return result


def test_gpu_slot_allocation_is_deterministic(tmp_path):
    config = _cuda_launcher(_config(tmp_path), gpu_ids=[2, 4])
    _, jobs = _jobs(config)
    first = allocate_job_slots(config, jobs)
    second = allocate_job_slots(config, jobs)
    assert first == second
    assert [item.gpu_ids for item in first] == [[2], [4]]


def test_jobs_per_gpu_creates_queue_slots(tmp_path):
    config = _cuda_launcher(
        _config(tmp_path, seeds=[1, 2]),
        gpu_ids=[0],
        jobs_per_gpu=2,
    )
    _, jobs = _jobs(config)
    assignments = allocate_job_slots(config, jobs)
    assert [item.slot_index for item in assignments] == [0, 1, 0, 1]
    assert all(item.gpu_ids == [0] for item in assignments)


def test_multi_gpu_groups_are_disjoint(tmp_path):
    config = _cuda_launcher(
        _config(tmp_path), gpu_ids=[0, 1, 2, 3], gpus_per_job=2
    )
    _, jobs = _jobs(config)
    assignments = allocate_job_slots(config, jobs)
    assert assignments[0].gpu_ids == [0, 1]
    assert assignments[1].gpu_ids == [2, 3]
    assert set(assignments[0].gpu_ids).isdisjoint(assignments[1].gpu_ids)


def test_impossible_gpu_group_fails(tmp_path):
    config = _config(tmp_path)
    bad = replace(
        config,
        training=replace(config.training, device="cuda"),
        launcher=replace(
            config.launcher, gpu_ids=[0, 1, 2], gpus_per_job=2
        ),
    )
    with pytest.raises(ValueError, match="divisible"):
        bad.validate()


def test_dry_run_starts_no_child(tmp_path, monkeypatch, capsys):
    from lm_cl.cli import launch_experiments

    output = tmp_path / "dry-output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "launch_experiments",
            "--config",
            str(SMOKE),
            "--name",
            "dry-test",
            "--output-root",
            str(output),
            "--dry-run",
        ],
    )
    monkeypatch.setattr(
        LocalJobScheduler,
        "run",
        lambda self: (_ for _ in ()).throw(AssertionError("child scheduler ran")),
    )
    launch_experiments.command()
    report = json.loads(capsys.readouterr().out)
    assert report["child_processes_started"] == 0
    assert not output.exists()


def test_child_failure_propagates(tmp_path):
    config = _config(tmp_path, models=["transformer"])
    output = tmp_path / "failing-job"
    output.mkdir()
    assignment = JobAssignment(
        job_index=0,
        job_id="transformer-seed-101",
        gpu_ids=[],
        slot_index=0,
        rendezvous_port=32000,
        output_dir=str(output),
        command=[sys.executable, "-c", "raise SystemExit(3)"],
        launch_state="fresh",
    )
    with pytest.raises(RuntimeError, match="Job failed"):
        LocalJobScheduler(config, [], [assignment]).run()


def test_sigint_is_forwarded_to_active_stage():
    class Child:
        def __init__(self):
            self.signals = []

        def poll(self):
            return None

        def send_signal(self, value):
            self.signals.append(value)

    controller = StageProcessController()
    child = Child()
    controller.child = child  # type: ignore[assignment]
    controller.handler(2, None)
    assert controller.received_signal == 2
    assert child.signals == [2]


def test_output_directories_do_not_collide(tmp_path):
    config = _config(tmp_path, seeds=[1, 2])
    _, jobs = _jobs(config)
    assert len({job.output_dir for job in jobs}) == len(jobs)


def test_jsonl_preserves_required_fields(tmp_path):
    path = tmp_path / "metrics.jsonl"
    logger = JsonlMetricLogger(path)
    logger.log(
        {
            "event": "logical_batch",
            "global_logical_batches": 1,
            "global_input_tokens": 16,
            "global_valid_targets": 15,
            "mean_loss": 2.0,
        }
    )
    logger.close()
    record = json.loads(path.read_text())
    assert set(record) >= {
        "event",
        "global_logical_batches",
        "global_input_tokens",
        "global_valid_targets",
        "mean_loss",
    }


def test_tensorboard_contains_required_tags(tmp_path):
    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator,
    )

    logger = JsonlMetricLogger(
        tmp_path / "metrics.jsonl",
        tensorboard_dir=tmp_path / "tensorboard",
        tensorboard_flush_seconds=1,
    )
    logger.log(
        {
            "event": "logical_batch",
            "global_logical_batches": 1,
            "global_input_tokens": 16,
            "global_valid_targets": 15,
            "mean_loss": 2.0,
            "learning_rate": 0.001,
            "parameter_norm": 3.0,
            "throughput_input_tokens_per_second": 4.0,
            "cycle_index": 0,
            "task_index": 0,
            "fast_update_count": 1,
            "active_memory_norm": 1.0,
            "active_memory_gradient_norm_before_clip": 0.5,
            "m0_norm": 0.8,
        }
    )
    logger.log(
        {
            "event": "optimizer_step",
            "global_logical_batches": 1,
            "gradient_norm": 0.7,
            "learning_rate": 0.001,
        }
    )
    logger.log(
        {
            "event": "probe_evaluation",
            "global_logical_batches": 0,
            "cumulative_input_tokens": 0,
            "memory_evaluation_mode": "carried",
            "mean_validation_ce": 2.5,
        }
    )
    logger.close()
    accumulator = EventAccumulator(str(tmp_path / "tensorboard")).Reload()
    tags = set(accumulator.Tags()["scalars"])
    assert {
        "train/loss",
        "train/learning_rate",
        "train/slow_gradient_norm",
        "train/parameter_norm",
        "train/input_tokens",
        "train/valid_targets",
        "train/tokens_per_second",
        "continual/cycle_index",
        "continual/task_index",
        "fastmem/active_memory_norm",
        "fastmem/fast_gradient_norm",
        "fastmem/fast_updates",
        "fastmem/m0_norm",
        "probe/validation_ce",
    } <= tags


def test_tensorboard_resume_deduplicates_steps(tmp_path):
    log_dir = tmp_path / "tb"
    first = TensorBoardTracker(log_dir, flush_seconds=1)
    assert first.scalar("train/loss", 1.0, 1)
    first.close()
    resumed = TensorBoardTracker(log_dir, flush_seconds=1)
    assert not resumed.scalar("train/loss", 9.0, 1)
    assert resumed.scalar("train/loss", 0.5, 2)
    resumed.close()
    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator,
    )

    events = EventAccumulator(str(log_dir)).Reload().Scalars("train/loss")
    assert [(item.step, item.value) for item in events] == [(1, 1.0), (2, 0.5)]


@pytest.fixture(scope="module")
def tiny_runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("public-tiny-runs")
    results = {}
    for public_model in ("transformer", "fastmem_rmt"):
        config = _config(
            root,
            name=f"tiny-{public_model}",
            models=[public_model],
            cycles=1,
            tensorboard=False,
        )
        data, jobs = _jobs(config)
        continual = build_continual_job_config(config, jobs[0])
        result = ContinualTrainer(continual).run()
        results[public_model] = (config, data, jobs[0], continual, result)
    return results


def test_final_checkpoint_is_always_written(tiny_runs):
    for _, _, _, _, result in tiny_runs.values():
        assert result.status == "complete"
        assert Path(result.checkpoint_path).is_file()
        assert sha256_file(result.checkpoint_path) == result.checkpoint_sha256


def test_task_optimizer_restarts_at_every_language(tiny_runs):
    config, _, spec, _, _ = tiny_runs["transformer"]
    records = [
        json.loads(line)
        for line in (Path(spec.output_dir) / "metrics.jsonl").read_text().splitlines()
    ]
    generations = [
        record["optimizer_generation"]
        for record in records
        if record["event"] == "task_start"
    ]
    assert generations == list(range(1, 9))


def test_fastmem_resets_at_every_language(tiny_runs):
    _, _, _, _, result = tiny_runs["fastmem_rmt"]
    assert result.state["memory_reset_count"] == 8
    assert result.state["global_fast_updates"] == 16


def test_task_boundaries_record_seen_language_forgetting_matrix(tmp_path):
    config = _config(
        tmp_path,
        name="tiny-forgetting",
        models=["transformer"],
        cycles=1,
        tensorboard=False,
    )
    data, jobs = _jobs(config)
    continual = build_continual_job_config(config, jobs[0])
    validation_sources = {}
    for language_index, language in enumerate(PUBLIC_LANGUAGE_ORDER):
        validation_sources[language] = TrainSourceConfig(
            kind="synthetic",
            synthetic=DataConfig(
                backend="synthetic",
                vocab_size=continual.model.vocab_size,
                sequence_length=16,
                num_sequences=1,
                seed=7000 + language_index,
                ignore_index=-100,
                mask_probability=0.0,
            ),
            packed=None,
        )
    continual = replace(
        continual,
        tasks=[
            replace(
                task,
                validation_source=validation_sources[task.language],
                validation_logical_batches=1,
            )
            for task in continual.tasks
        ],
    )
    continual.validate()
    result = ContinualTrainer(continual).run()
    assert result.state["forgetting_evaluation_count"] == 8
    assert set(result.state["forgetting_latest_mean_loss"]) == set(
        PUBLIC_LANGUAGE_ORDER
    )
    records = [
        json.loads(line)
        for line in (Path(jobs[0].output_dir) / "metrics.jsonl")
        .read_text()
        .splitlines()
    ]
    forgetting = [
        record for record in records if record["event"] == "forgetting_evaluation"
    ]
    assert len(forgetting) == 8
    assert list(forgetting[0]["languages"]) == ["en"]
    assert set(forgetting[-1]["languages"]) == set(PUBLIC_LANGUAGE_ORDER)
    assert forgetting[-1]["memory_evaluation_mode"] == "not_applicable"


def _windowed_tiny_continual(tmp_path: Path, *, name: str):
    config = _config(
        tmp_path,
        name=name,
        models=["transformer"],
        cycles=2,
        tensorboard=False,
    )
    _, jobs = _jobs(config)
    continual = build_continual_job_config(config, jobs[0])
    expanded_sources = []
    for task in continual.tasks[: len(PUBLIC_LANGUAGE_ORDER)]:
        assert task.train_source.synthetic is not None
        expanded_sources.append(
            replace(task.train_source.synthetic, num_sequences=4)
        )
    tasks = []
    for task in continual.tasks:
        language_index = task.task_index % len(PUBLIC_LANGUAGE_ORDER)
        tasks.append(
            replace(
                task,
                train_source=TrainSourceConfig(
                    kind="synthetic",
                    synthetic=expanded_sources[language_index],
                    packed=None,
                ),
                train_sequence_prefix_count=2,
                train_sequence_offset_count=2 * task.cycle_index,
            )
        )
    result = replace(continual, tasks=tasks)
    result.validate()
    return result


def test_nonzero_sequence_window_interruption_resumes_exactly(tmp_path):
    full_config = _windowed_tiny_continual(tmp_path / "full", name="window-full")
    full = ContinualTrainer(full_config).run()

    resume_config = _windowed_tiny_continual(
        tmp_path / "resume", name="window-resume"
    )
    interrupted = ContinualTrainer(resume_config).run(
        stop_after_global_logical_batches=17
    )
    assert interrupted.status == "interrupted"
    payload = load_checkpoint(interrupted.checkpoint_path)
    window = payload["source_identity"]["sequence_window"]
    assert window["sequence_start"] == 2
    assert window["sequence_count"] == 2
    assert window["sequence_end_exclusive"] == 4
    assert payload["trainer_state"]["current_task_index"] == 8
    assert payload["trainer_state"]["source_position"] == {
        "shard_index": 0,
        "token_offset": 3,
    }

    resumed = ContinualTrainer(resume_config).run(
        resume_checkpoint=interrupted.checkpoint_path
    )
    assert resumed.status == "complete"
    assert resumed.loss_history == full.loss_history
    full_payload = load_checkpoint(full.checkpoint_path)
    resumed_payload = load_checkpoint(resumed.checkpoint_path)
    assert state_digest(resumed_payload["model_state"]) == state_digest(
        full_payload["model_state"]
    )


def _forgetting_tiny_continual(tmp_path: Path, *, name: str):
    config = _config(
        tmp_path,
        name=name,
        models=["fastmem_rmt"],
        cycles=1,
        tensorboard=False,
    )
    _, jobs = _jobs(config)
    continual = build_continual_job_config(config, jobs[0])
    tasks = []
    for language_index, task in enumerate(continual.tasks):
        validation = TrainSourceConfig(
            kind="synthetic",
            synthetic=DataConfig(
                backend="synthetic",
                vocab_size=continual.model.vocab_size,
                sequence_length=16,
                num_sequences=1,
                seed=8000 + language_index,
                ignore_index=-100,
                mask_probability=0.0,
            ),
            packed=None,
        )
        tasks.append(
            replace(
                task,
                validation_source=validation,
                validation_logical_batches=1,
            )
        )
    result = replace(continual, tasks=tasks)
    result.validate()
    return result


def test_fastmem_forgetting_state_survives_task_boundary_resume(tmp_path):
    full_config = _forgetting_tiny_continual(
        tmp_path / "full", name="forgetting-full"
    )
    full = ContinualTrainer(full_config).run()

    resume_config = _forgetting_tiny_continual(
        tmp_path / "resume", name="forgetting-resume"
    )
    interrupted = ContinualTrainer(resume_config).run(
        stop_after_task_boundaries=4
    )
    interrupted_payload = load_checkpoint(interrupted.checkpoint_path)
    assert interrupted_payload["trainer_state"][
        "forgetting_evaluation_count"
    ] == 4
    assert set(
        interrupted_payload["trainer_state"][
            "forgetting_latest_mean_loss"
        ]
    ) == set(PUBLIC_LANGUAGE_ORDER[:4])

    resumed = ContinualTrainer(resume_config).run(
        resume_checkpoint=interrupted.checkpoint_path
    )
    assert resumed.status == "complete"
    assert resumed.state["forgetting_evaluation_count"] == 8
    assert resumed.state["forgetting_first_mean_loss"] == (
        full.state["forgetting_first_mean_loss"]
    )
    assert resumed.state["forgetting_best_mean_loss"] == (
        full.state["forgetting_best_mean_loss"]
    )
    assert resumed.state["forgetting_latest_mean_loss"] == (
        full.state["forgetting_latest_mean_loss"]
    )
    full_payload = load_checkpoint(full.checkpoint_path)
    resumed_payload = load_checkpoint(resumed.checkpoint_path)
    assert state_digest(resumed_payload["model_state"]) == state_digest(
        full_payload["model_state"]
    )
    assert state_digest(resumed_payload["memory_state"]) == state_digest(
        full_payload["memory_state"]
    )


def test_probe_is_derived_and_source_hash_is_immutable(tiny_runs):
    config, _, spec, _, result = tiny_runs["transformer"]
    source = Path(result.checkpoint_path)
    before = sha256_file(source)
    probe = build_probe_job_config(
        config,
        spec,
        cycle_index=0,
        source_checkpoint=source,
        source_checkpoint_sha256=before,
    )
    assert Path(probe.runtime.output_dir) != Path(spec.output_dir)
    probe_result = ProbeTrainer(probe).run()
    assert probe_result.status == "complete"
    assert sha256_file(source) == before
    results = json.loads(
        (Path(probe.runtime.output_dir) / "probe_results.json").read_text()
    )
    assert results["source_checkpoint_sha256_before"] == before
    assert results["source_checkpoint_sha256_after"] == before


def test_latest_checkpoint_pointer_is_validated(tiny_runs):
    _, _, spec, _, result = tiny_runs["transformer"]
    pointer = pointer_for_checkpoint(
        result.checkpoint_path,
        job_dir=spec.output_dir,
        scientific_sha256=spec.scientific_sha256,
        resolved_experiment_sha256=spec.resolved_sha256,
        requested_horizon_cycles=1,
    )
    path = write_latest_pointer(spec.output_dir, pointer)
    loaded, _, checkpoint = validate_latest_pointer(
        path,
        expected_job_dir=spec.output_dir,
        expected_scientific_sha256=spec.scientific_sha256,
    )
    assert loaded["completed_cycle_count"] == 1
    assert checkpoint == Path(result.checkpoint_path).resolve()


def _resolved_pair(tmp_path):
    one = _config(
        tmp_path,
        name="extension",
        models=["transformer"],
        cycles=1,
    )
    two = replace(
        one,
        experiment=replace(one.experiment, cycles=2, resume="auto"),
    )
    one_data, one_jobs = _jobs(one)
    two_data, two_jobs = _jobs(two)
    return one_data, one_jobs[0].resolved_experiment, two_data, two_jobs[0].resolved_experiment


def test_one_to_two_cycle_extension_is_accepted(tmp_path):
    _, old, _, new = _resolved_pair(tmp_path)
    validate_horizon_extension(old, new, completed_cycles=1)


def test_extension_requires_future_manifests(tmp_path):
    _, old, _, new = _resolved_pair(tmp_path)
    broken = copy.deepcopy(new)
    broken["data_contract"]["data_manifests"] = broken["data_contract"][
        "data_manifests"
    ][:1]
    with pytest.raises(ValueError, match="Future cycle manifests"):
        validate_horizon_extension(old, broken, completed_cycles=1)


def test_decreasing_cycle_horizon_fails(tmp_path):
    _, old, _, new = _resolved_pair(tmp_path)
    with pytest.raises(ValueError, match="Decreasing"):
        validate_horizon_extension(new, old, completed_cycles=1)


def test_changing_completed_manifest_fails(tmp_path):
    _, old, _, new = _resolved_pair(tmp_path)
    broken = copy.deepcopy(new)
    broken["data_contract"]["data_manifests"][0]["en"][
        "ordered_data_sha256"
    ] = "0" * 64
    with pytest.raises(ValueError, match="Manifest identities changed"):
        validate_horizon_extension(old, broken, completed_cycles=1)


def test_changing_scientific_semantics_fails(tmp_path):
    _, old, _, new = _resolved_pair(tmp_path)
    broken = copy.deepcopy(new)
    broken["scientific_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="scientific semantics"):
        validate_horizon_extension(old, broken, completed_cycles=1)


def test_checkpoint_contains_exact_task_order_and_counters(tiny_runs):
    _, _, _, _, result = tiny_runs["transformer"]
    payload = load_checkpoint(result.checkpoint_path)
    assert payload["trainer_state"]["next_task_index"] == 8
    assert [task["language"] for task in payload["resolved_config"]["tasks"]] == list(
        PUBLIC_LANGUAGE_ORDER
    )


def test_horizon_migration_preserves_all_scientific_state(tiny_runs, tmp_path):
    config, _, _, _, result = tiny_runs["transformer"]
    extended_config = replace(
        config,
        experiment=replace(config.experiment, cycles=2, resume="auto"),
    )
    data, jobs = _jobs(extended_config)
    del data
    continual = build_continual_job_config(extended_config, jobs[0])
    output = tmp_path / "extended.pt"
    migrate_checkpoint_horizon(
        result.checkpoint_path,
        new_internal_config=continual.to_dict(),
        output_path=output,
        extension_record={"old_horizon_cycles": 1, "new_horizon_cycles": 2},
    )
    before = load_checkpoint(result.checkpoint_path)
    after = load_checkpoint(output)
    for field in (
        "model_state",
        "optimizer_state",
        "scheduler_state",
        "scaler_state",
        "gradients",
        "trainer_state",
        "rng_state",
        "memory_state",
        "source_identity",
    ):
        assert state_digest(before[field]) == state_digest(after[field])


def test_release_builder_excludes_internal_artifacts(tmp_path):
    output = tmp_path / "release"
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/build_release.py"), "--output", str(output)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    paths = [path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()]
    assert not any("runs/" in path or path.endswith(".pt") for path in paths)
    assert not any("phase8" in path.lower() for path in paths)
    assert sorted(path for path in paths if path.endswith(".md")) == [
        "README.md",
        "docs/CONFIGURATION.md",
        "docs/DATA_AND_RESUME.md",
    ]


def test_gitignore_does_not_hide_source_data_package():
    rules = {
        line.strip()
        for line in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    assert "/data/" in rules
    assert "data/" not in rules
    assert (ROOT / "src/lm_cl/data/__init__.py").is_file()


def test_exported_release_imports_and_cli_help(tmp_path):
    output = tmp_path / "release"
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/build_release.py"), "--output", str(output)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(output / "src")
    for module in (
        "lm_cl.cli.launch_experiments",
        "lm_cl.cli.prepare_experiment_data",
        "lm_cl.cli.inspect_environment",
    ):
        completed = subprocess.run(
            [sys.executable, "-m", module, "--help"],
            cwd=output,
            env=environment,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
