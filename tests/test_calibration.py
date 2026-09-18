from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch

from lm_cl.cli.calibrate_continual import summarize, validate_window
from lm_cl.config import (ContinualExperimentConfig, ContinualOptimizationConfig,
                         ContinualRuntimeConfig, ContinualTaskConfig, DataConfig,
                         ModelConfig, TrainSourceConfig, VariantConfig)
from lm_cl.training import ContinualTrainer


def tiny_config(tmp_path, k):
    hidden, layers, vocab, positions = 8, 1, 16, 16
    nonembedding = layers * (12 * hidden**2 + 13 * hidden) + 2 * hidden
    model = ModelConfig(name="tiny_calibration", layers=layers, hidden_size=hidden,
        attention_heads=2, head_dim=4, mlp_hidden_size=32, vocab_size=vocab,
        max_position_embeddings=positions, expected_non_embedding_parameters=nonembedding,
        expected_total_parameters=nonembedding+(vocab+positions)*hidden,
        learning_rate=.001, dropout=0., initializer_std=.02, layer_norm_epsilon=1e-5,
        use_bias=True, activation="gelu", gelu_approximation="none",
        position_embedding_type="learned_absolute", tie_word_embeddings=True)
    source = TrainSourceConfig("synthetic", DataConfig("synthetic", vocab, 8, 20, 11, -100, 0.), None)
    return ContinualExperimentConfig(1, "calibration-test", model,
        VariantConfig("backbone_clean" if k == 1 else "backbone_matched_k", False, False, 0., 0, k, None),
        ContinualOptimizationConfig("adamw", .9, .95, 1e-8, .1, .05, 2, 1, None, "fp32", 2, 0),
        ContinualRuntimeConfig(123, "cpu", True, str(tmp_path), "metrics.jsonl"),
        [ContinualTaskConfig("en", 0, 0, 10, None, source, None, 0)])


@pytest.mark.parametrize("k", [1, 2])
def test_calibration_uses_complete_production_optimizer_windows(tmp_path, k):
    torch.set_num_threads(1)
    config = tiny_config(tmp_path, k)
    validate_window(config, warmup=2, batches=8, world_size=1)
    result = ContinualTrainer(config).run(stop_after_global_logical_batches=8)
    assert result.status == "interrupted"
    records = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    measured = summarize(records, warmup=2, batches=8)
    assert measured["input_tokens"] == 6 * 2 * 8
    assert len(measured["windows"]) == 6 // k
    assert measured["input_tokens_per_second"] > 0
    assert Path(result.checkpoint_path).is_file()


@pytest.mark.parametrize("warmup,batches,world", [(1, 8, 1), (2, 5, 1), (2, 4, 1), (2, 66, 1), (2, 8, 2), (2, 12, 1)])
def test_invalid_measurement_window_rejected(tmp_path, warmup, batches, world):
    with pytest.raises(ValueError):
        validate_window(tiny_config(tmp_path, 2), warmup=warmup, batches=batches, world_size=world)


def test_missing_or_nonmonotonic_measurements_fail_closed():
    with pytest.raises(ValueError, match="Missing"):
        summarize([], warmup=2, batches=8)
    rows = [{"event": "optimizer_step", "global_logical_batches": step,
             "global_input_tokens": step * 16, "wall_time_seconds": 1.} for step in [2, 8]]
    with pytest.raises(ValueError, match="Nonmonotonic"):
        summarize(rows, warmup=2, batches=8)


def test_resumed_throughput_excludes_tokens_from_previous_process(tmp_path):
    torch.set_num_threads(1)
    config = tiny_config(tmp_path, 2)
    partial = ContinualTrainer(config).run(stop_after_global_logical_batches=3)
    ContinualTrainer(config).run(resume_checkpoint=partial.checkpoint_path,
                                 stop_after_global_logical_batches=8)
    records = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    resume = next(r for r in records if r["event"] == "resume")
    assert resume["global_input_tokens"] == 3 * 16
    assert resume["process_input_tokens"] == 0
    assert resume["throughput_input_tokens_per_second"] == 0
    final = records[-1]
    assert final["global_input_tokens"] == 8 * 16
    assert final["process_input_tokens"] == 5 * 16
    assert final["throughput_input_tokens_per_second"] == pytest.approx(5 * 16 / final["wall_time_seconds"])
