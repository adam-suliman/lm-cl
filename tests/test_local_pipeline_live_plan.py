"""Offline gates for the supervised handoff; no live network or GPU work."""
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from lm_cl.data.incremental import initialize
from lm_cl.diagnostics.local_pipeline_live_plan import compare, freeze_vi
from test_incremental_pipeline import prepared, recipe


def test_reference_comparator_checks_recipe_and_logical_content(tmp_path):
    first = prepared(tmp_path / "first")
    second = prepared(tmp_path / "second", replace(recipe(), block_tokens=253))
    limits = SimpleNamespace(owned=lambda p: Path(p))
    output = tmp_path / "match.json"
    compare(first.root, second.root, output, limits)
    record = json.loads(output.read_text())
    assert record["status"] == "exact_logical_data_match"
    assert record["streams"][0]["data_sha256"] == record["streams"][1]["data_sha256"]
    changed = prepared(tmp_path / "changed", replace(recipe(), document_order_seed=12))
    with pytest.raises(ValueError, match="Scientific preparation recipes differ"):
        compare(first.root, changed.root, tmp_path / "invalid.json", limits)
    assert not (tmp_path / "invalid.json").exists()


@pytest.mark.parametrize("complete", [False, True])
def test_live_plan_refuses_unfinished_or_substitute_validation(tmp_path, complete):
    root = tmp_path / "validation"
    r = replace(recipe(), purpose="validation", validation_permyriad=9900)
    if complete:
        prepared(root, r)
        expected = "real pinned source, not a timing replay"
    else:
        initialize(root, r)
        expected = "complete before freezing training"
    settings = dict(schema_version=1, validation_root=str(root), tokens=6291456,
        block_tokens=65536, reference_block_tokens=6291456, document_order_seed=31003,
        global_batch=256, sequence_length=2048, world_size=3, physical_microbatch=1,
        precision="fp32", run_prefix="offline-refusal-test")
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(settings))
    with pytest.raises(ValueError, match=expected):
        freeze_vi(path, SimpleNamespace(report=tmp_path / "report", work=tmp_path / "work"))
    assert not (tmp_path / "report").exists()
    assert not (tmp_path / "work").exists()
