from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from lm_cl.cli.a100_run import build_config, parser


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("size", ["5m", "12m"])
@pytest.mark.parametrize("gpus,per_job", [("0", 1), ("0,1", 1), ("0,1", 2)])
def test_size_script_defaults_to_a_read_only_plan(tmp_path, size, gpus, per_job):
    data, output = tmp_path / "data", tmp_path / "output"
    env = dict(os.environ, LM_CL_PYTHON=sys.executable)
    result = subprocess.run([str(ROOT / "scripts" / f"run_{size}.sh"),
        "--data-root", str(data), "--output-root", str(output),
        "--gpus", gpus, "--gpus-per-job", str(per_job)],
        capture_output=True, text=True, env=env, check=True)
    value = json.loads(result.stdout)
    assert value["status"] == "configuration_validated_only"
    assert value["data_or_gpu_validation_performed"] is False
    cfg = value["resolved_config"]
    assert cfg["experiment"]["model_size"] == size
    assert cfg["experiment"]["models"] == ["transformer", "fastmem_rmt"]
    assert cfg["experiment"]["cycles"] == 5
    assert cfg["experiment"]["resume"] == "never"
    assert cfg["training"]["global_batch_sequences"] == 256
    assert cfg["data"]["prepare_if_missing"] is False
    assert cfg["launcher"]["gpus_per_job"] == per_job
    assert cfg["launcher"]["max_parallel_jobs"] == len(gpus.split(",")) // per_job
    assert not data.exists() and not output.exists()


def test_a100_plan_preserves_existing_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("LM_CL_DATA_ROOT", "original-data")
    monkeypatch.delenv("LM_CL_OUTPUT_ROOT", raising=False)
    args = parser().parse_args(["--model-size", "12m", "--data-root", str(tmp_path / "data"),
                               "--output-root", str(tmp_path / "out"), "--a100-memory-gb", "80"])
    cfg = build_config(args)
    assert cfg.training.physical_microbatch_sequences == 8
    assert cfg.training.precision == "bf16"
    assert os.environ["LM_CL_DATA_ROOT"] == "original-data"
    assert "LM_CL_OUTPUT_ROOT" not in os.environ


@pytest.mark.parametrize("options", [
    ["--gpus", "0,0"], ["--gpus", "0", "--gpus-per-job", "2"],
    ["--gpus", "-1"], ["--models", "unknown"], ["--parallel-languages", "9"],
    ["--physical-microbatch-sequences", "0"], ["--cycles", "0"],
])
def test_invalid_launch_is_rejected_before_any_data_access(tmp_path, options):
    args = parser().parse_args(["--model-size", "5m", "--data-root", str(tmp_path / "data"),
                               "--output-root", str(tmp_path / "out"), *options])
    with pytest.raises(ValueError):
        build_config(args)
    assert list(tmp_path.iterdir()) == []


def test_dry_run_cannot_enable_automatic_download(tmp_path):
    args = parser().parse_args(["--model-size", "5m", "--action", "prepare", "--dry-run",
        "--data-root", str(tmp_path / "data"), "--output-root", str(tmp_path / "out")])
    assert build_config(args).data.prepare_if_missing is False


def test_fresh_probe_pool_name_matches_prepared_budget_and_allows_explicit_reuse(tmp_path):
    base = ["--model-size", "5m", "--data-root", str(tmp_path / "data"),
            "--output-root", str(tmp_path / "out"), "--probe-tokens", "500000000"]
    config = build_config(parser().parse_args(base))
    assert "499998720" in config.data.probe_training_manifest
    existing = tmp_path / "existing-5b-pool/manifest.json"
    config = build_config(parser().parse_args([*base, "--probe-training-manifest", str(existing)]))
    assert config.data.probe_training_manifest == str(existing)
    assert config.probe.training_tokens == 500000000
