from __future__ import annotations

import copy
import json
import threading
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from lm_cl.data.incremental import (
    FORMAT, IncrementalSource, Producer, Recipe, initialize, file_hash,
)
from lm_cl.data.selection import select_documents
from lm_cl.config.data_schema import SelectionConfig
from lm_cl.data.types import TokenPosition
from lm_cl.training.distributed import plan_logical_batch_partition


class Rows:
    def __init__(self, count=300):
        self.values = [{"text": f"document {i:05d} " + "abc" * (i % 13 + 3),
                        "url": f"https://example.invalid/{i}"} for i in range(count)]
        self.reads = []

    def row(self, index):
        self.reads.append(index)
        return self.values[index] if index < len(self.values) else None


class Tokenizer:
    def encode(self, text, *, add_special_tokens=False):
        assert add_special_tokens is False
        return list(text.encode())


def recipe(**kwargs):
    return Recipe(FORMAT, {"kind": "offline-test", "identity": "fixture-v1"},
                  "vi", "train", {"identity": "utf8-test-v1"}, 300, 2048,
                  67, 16, 31011, 17, 41001, 100, 255, 255,
                  "text", "url", [], **kwargs)


def prepared(root, r=None):
    r = r or recipe()
    initialize(root, r)
    p = Producer(root, Rows(), Tokenizer())
    p.run()
    return IncrementalSource(root, wait_seconds=0)


def tokens(source):
    return source.read_tokens(source.token_count)[0]


def boundaries(source):
    return [b for record in source.records for b in record["boundaries"]]


def test_reference_order_packing_and_storage_layout(tmp_path):
    r = recipe()
    first = prepared(tmp_path / "a", r)
    second = prepared(tmp_path / "b", replace(r, block_tokens=253))
    np.testing.assert_array_equal(tokens(first), tokens(second))
    assert boundaries(first) == boundaries(second)
    cfg = SelectionConfig(max_input_documents=r.max_input_documents,
                          max_output_tokens=r.output_tokens, max_runtime_seconds=60,
                          document_order_seed=r.document_order_seed,
                          split_seed=r.split_seed, validation_permyriad=r.validation_permyriad,
                          shuffle_buffer_documents=r.shuffle_buffer_documents,
                          order_algorithm=r.order_algorithm, split_algorithm=r.split_algorithm,
                          document_hash_algorithm="sha256_utf8", token_hash_algorithm="sha256_uint32_le")
    selected = select_documents(Rows().values, text_field="text", id_field="url",
                                purpose="vietnamese_train", config=cfg, rejection_counts={})
    expected = []
    for doc, split in selected:
        ids = Tokenizer().encode(doc.text)
        expected.extend(ids[:r.output_tokens-len(expected)-1] + [r.eos_token_id])
        if len(expected) == r.output_tokens:
            break
    np.testing.assert_array_equal(tokens(first), expected)
    assert tokens(first)[-1] == r.eos_token_id
    assert boundaries(first)[-1]["truncated"]


@pytest.mark.parametrize("fault", ["after_data", "after_commit"])
def test_producer_exact_resume_nonempty_shuffle_and_packing(tmp_path, fault):
    reference = prepared(tmp_path / "reference")
    root = tmp_path / "resumed"
    initialize(root, recipe())
    producer = Producer(root, Rows(), Tokenizer())
    producer.run(max_blocks=2)
    state = copy.deepcopy(producer.state)
    assert len(state["buffer"]) == recipe().shuffle_buffer_documents
    assert state["pending_offset"] < len(state["pending_tokens"])
    with pytest.raises(InterruptedError):
        producer.publish_one(fault=fault)
    resumed_source = Rows()
    resumed = Producer(root, resumed_source, Tokenizer())
    cursor = resumed.state["source_cursor"]
    resumed.run()
    actual = IncrementalSource(root, wait_seconds=0)
    np.testing.assert_array_equal(tokens(actual), tokens(reference))
    assert boundaries(actual) == boundaries(reference)
    assert file_hash(root / "complete.json") == file_hash(reference.root / "complete.json")
    # No replay of all earlier rows: only retained shuffle references and new rows.
    assert len(resumed_source.reads) < 300
    assert cursor > 0


def test_live_publication_waits_without_short_batches(tmp_path):
    root = tmp_path / "concurrent"
    initialize(root, recipe())
    producer = Producer(root, Rows(), Tokenizer())
    producer.run(max_blocks=1)
    source = IncrementalSource(root, wait_seconds=5, poll_seconds=.005)
    assert source.ready_tokens < source.token_count
    errors = []
    def finish():
        try:
            time.sleep(.05)
            producer.run()
        except BaseException as e:
            errors.append(e)
    worker = threading.Thread(target=finish)
    worker.start()
    batches = list(source.iter_batches(sequence_length=16, global_sequences_per_batch=7))
    worker.join()
    assert not errors
    assert [len(b.input_ids) for b in batches] == [7] * 18 + [2]
    assert sum(b.valid_target_count for b in batches) == 128 * 15
    assert source.waited_seconds > .01
    np.testing.assert_array_equal(np.concatenate([b.input_ids.reshape(-1) for b in batches]), tokens(source))


def test_uncommitted_is_not_eof_or_consumable(tmp_path):
    root = tmp_path / "pending"
    initialize(root, recipe())
    producer = Producer(root, Rows(), Tokenizer())
    with pytest.raises(InterruptedError):
        producer.publish_one(fault="after_data")
    source = IncrementalSource(root, wait_seconds=0)
    assert source.ready_tokens == 0
    with pytest.raises(TimeoutError, match="not EOF"):
        next(source.iter_batches(sequence_length=16, global_sequences_per_batch=1))
    assert not (root / "complete.json").exists()


@pytest.mark.parametrize("damage", ["bytes", "missing-data", "missing-commit", "duplicate", "reorder", "completion"])
def test_fail_closed(tmp_path, damage):
    source = prepared(tmp_path / damage)
    root = source.root
    path = root / "commits/00000001.json"
    if damage == "bytes":
        with (root / "blocks/00000000.bin").open("r+b") as f:
            f.write(b"xxxx")
    elif damage == "missing-data":
        (root / "blocks/00000001.bin").unlink()
    elif damage == "missing-commit":
        path.unlink()
    elif damage == "completion":
        (root / "complete.json").write_text("{}")
    else:
        value = json.loads(path.read_text())
        value["index" if damage == "duplicate" else "previous"] = 0 if damage == "duplicate" else "bad"
        path.write_text(json.dumps(value))
    with pytest.raises((ValueError, FileNotFoundError)):
        IncrementalSource(root, wait_seconds=0)


def test_prefix_proof_survives_future_publication(tmp_path):
    root = tmp_path / "prefix"
    initialize(root, recipe())
    p = Producer(root, Rows(), Tokenizer())
    p.run(max_blocks=2)
    s = IncrementalSource(root, wait_seconds=0)
    proof = s.proof(64)
    p.run()
    resumed = IncrementalSource(root, wait_seconds=0)
    resumed.validate_proof(proof)
    bad = {**proof, "chain_sha256": "0" * 64}
    with pytest.raises(ValueError):
        resumed.validate_proof(bad)


def test_uneven_partition_crosses_blocks_and_sequences(tmp_path):
    source = prepared(tmp_path / "partition")
    global_batches = list(source.iter_batches(sequence_length=16, global_sequences_per_batch=7))
    for batch in global_batches:
        spans = [plan_logical_batch_partition(len(batch.input_ids), rank=r, world_size=3) for r in range(3)]
        pieces = [batch.input_ids[p.start:p.end] for p in spans]
        np.testing.assert_array_equal(np.concatenate(pieces), batch.input_ids)
        assert sum(len(p)*(16-1) for p in pieces) == batch.valid_target_count


def test_frozen_predecessor_duplicate_ownership(tmp_path):
    first = prepared(tmp_path / "first")
    r = replace(recipe(), predecessor_streams=[{"path": str(first.root),
        "recipe_sha256": first.recipe.sha256, "completion_sha256": file_hash(first.root / "complete.json")}])
    second = prepared(tmp_path / "second", r)
    a = {b["content_sha256"] for b in boundaries(first)}
    b = {b["content_sha256"] for b in boundaries(second)}
    assert a.isdisjoint(b)
    assert second.records[-1]["producer_state"]["rejections"]["duplicate_content_or_tokens"] > 0


class ModelTokenizer:
    def encode(self, text, *, add_special_tokens=False):
        return [i % 30 for i in text.encode()]


def training_config(tmp_path, roots, output="train"):
    from lm_cl.config.yaml import load_model_config
    from lm_cl.config.schema import VariantConfig
    from lm_cl.config.continual_schema import ContinualOptimizationConfig, ContinualRuntimeConfig
    from lm_cl.diagnostics.local_pipeline_training import BlockInput, DemoTask, DemoTrainingConfig
    tasks = []
    for i, root in enumerate(roots):
        source = IncrementalSource(root, wait_seconds=0)
        tasks.append(DemoTask(source.recipe.language, i, 0,
            BlockInput(str(root), source.recipe.sha256, 16, 5),
            source.token_count, source.token_count//16))
    cfg = DemoTrainingConfig(101, "offline-incremental-proof",
        load_model_config(Path(__file__).resolve().parents[1]/"configs/models/tiny_test.yaml"),
        VariantConfig("fastmem_rmt", True, True, .005, 8, 2, 1., 8,
                      "task_boundary_from_m0_stopgrad", "reset_and_carried"),
        ContinualOptimizationConfig("adamw", .9, .95, 1e-8, .1, .05, 4, 1, None, "fp32", 2, 0),
        ContinualRuntimeConfig(123, "cpu", True, str(tmp_path/output), "metrics.jsonl"), tasks)
    cfg.validate()
    return cfg


def assert_training_equal(left, right):
    import torch
    assert left["trainer_state"] == right["trainer_state"]
    assert left["scheduler_state"] == right["scheduler_state"]
    for name in left["model_state"]:
        torch.testing.assert_close(left["model_state"][name], right["model_state"][name], atol=0, rtol=0)
    for k in ["active_memory", "initial_memory"]:
        torch.testing.assert_close(left["memory_state"][k], right["memory_state"][k], atol=0, rtol=0)
    for param, state in left["optimizer_state"]["state"].items():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                torch.testing.assert_close(v, right["optimizer_state"]["state"][param][k], atol=0, rtol=0)
            else:
                assert v == right["optimizer_state"]["state"][param][k]


@pytest.mark.parametrize("stop", ["partial-k", "task-boundary"])
def test_existing_training_updates_exact_resume(tmp_path, stop):
    import torch
    from lm_cl.diagnostics.local_pipeline_training import DemoTrainer, load_demo_checkpoint
    torch.set_num_threads(1)
    roots = []
    for i, language in enumerate(["vi", "en"]):
        root = tmp_path/f"data-{i}"
        initialize(root, replace(recipe(), language=language, output_tokens=16*9, eos_token_id=31, maximum_token_id=31))
        Producer(root, Rows(), ModelTokenizer()).run()
        roots.append(root)
    reference_cfg = training_config(tmp_path, roots, "reference")
    ref = DemoTrainer(reference_cfg).run()
    cfg = training_config(tmp_path, roots, "resumed")
    if stop == "partial-k":
        first = DemoTrainer(cfg).run(stop_after_global_logical_batches=1)
        p = load_demo_checkpoint(first.checkpoint_path)
        assert p["trainer_state"]["window_logical_batches"] == 1
        assert any(v is not None for v in p["gradients"].values())
    else:
        first = DemoTrainer(cfg).run(stop_after_task_boundaries=1)
    result = DemoTrainer(cfg).run(resume_checkpoint=first.checkpoint_path)
    assert result.status == "complete"
    assert result.state["global_fast_updates"] == 6
    assert result.state["global_slow_steps"] == 4
    assert result.state["memory_reset_count"] == 2
    assert result.state["global_input_tokens"] == 288
    assert result.state["global_valid_targets"] == 270
    assert_training_equal(load_demo_checkpoint(ref.checkpoint_path), load_demo_checkpoint(result.checkpoint_path))


def test_joint_restart_and_safe_wait_checkpoint(tmp_path):
    import torch
    from lm_cl.diagnostics.local_pipeline_training import DemoTrainer, load_demo_checkpoint
    torch.set_num_threads(1)
    r = replace(recipe(), output_tokens=16*12, block_tokens=64, eos_token_id=31, maximum_token_id=31)
    root = tmp_path/"data"
    initialize(root, r)
    producer = Producer(root, Rows(), ModelTokenizer())
    producer.run(max_blocks=1)
    cfg = training_config(tmp_path, [root], "resumed")
    cfg = replace(cfg, tasks=[replace(cfg.tasks[0], train_source=replace(cfg.tasks[0].train_source, wait_seconds=.01))])
    first = DemoTrainer(cfg).run()
    assert first.status == "waiting_for_data"
    assert first.state["global_logical_batches"] == 1
    assert first.state["window_logical_batches"] == 1
    # New producer resumes from committed shuffle and partial document state.
    Producer(root, Rows(), ModelTokenizer()).run()
    second = DemoTrainer(cfg).run(resume_checkpoint=first.checkpoint_path)
    reference_cfg = replace(cfg, runtime=replace(cfg.runtime, output_dir=str(tmp_path/"reference")))
    reference = DemoTrainer(reference_cfg).run()
    assert_training_equal(load_demo_checkpoint(reference.checkpoint_path), load_demo_checkpoint(second.checkpoint_path))


def test_legacy_materializer_matches_new_protocol(tmp_path):
    from test_data_materialization_performance import _config, _manifest, BatchTokenizer, _rows
    from lm_cl.data.materialize import materialize_stage
    from lm_cl.data.packed import PackedShardSource
    cfg = _config(tmp_path, generated_name="legacy", token_cap=2048)
    rows = _rows()
    materialize_stage(cfg, rows=rows, tokenizer=BatchTokenizer(), tokenizer_manifest=_manifest())
    old_root = Path(cfg.storage.generated_root)/"stages"/cfg.stage.stage_id
    old = PackedShardSource(old_root)
    r = replace(recipe(), language="en", sequence_length=8,
                document_order_seed=cfg.selection.document_order_seed,
                split_seed=cfg.selection.split_seed,
                shuffle_buffer_documents=cfg.selection.shuffle_buffer_documents,
                max_input_documents=cfg.selection.max_input_documents)
    root = tmp_path/"incremental"
    initialize(root, r)
    source = Rows(); source.values = rows
    Producer(root, source, BatchTokenizer()).run()
    new = IncrementalSource(root, wait_seconds=0)
    np.testing.assert_array_equal(old.read_tokens(old.token_count)[0], tokens(new))
    old_boundaries = [json.loads(line) for line in (old_root/"boundaries.jsonl").read_text().splitlines()]
    assert boundaries(new) == old_boundaries


def _ddp_worker(rank, world, port, config_path, result_path, resume=None, stop=None):
    import os
    import torch
    from lm_cl.diagnostics.local_pipeline_training import load_demo_training_config, DistributedDemoTrainer
    from lm_cl.training.distributed import DistributedContext
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(world),
                      LOCAL_WORLD_SIZE=str(world), MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.set_num_threads(1)
    cfg = load_demo_training_config(config_path)
    ctx = DistributedContext.initialize(cfg.distributed, runtime_device="cpu")
    try:
        result = DistributedDemoTrainer(cfg, ctx).run(resume_checkpoint=resume, stop_after_global_logical_batches=stop)
        if rank == 0:
            Path(result_path).write_text(json.dumps({"checkpoint": result.checkpoint_path, "state": result.state}))
    finally:
        ctx.close()


def test_three_rank_gloo_partial_k_resume_and_uneven_tail(tmp_path):
    import socket
    import torch.multiprocessing as mp
    from dataclasses import asdict
    from lm_cl.config.continual_schema import DistributedConfig
    from lm_cl.diagnostics.local_pipeline_training import load_demo_checkpoint
    root = tmp_path/"data"
    initialize(root, replace(recipe(), output_tokens=16*9, block_tokens=67, eos_token_id=31, maximum_token_id=31))
    Producer(root, Rows(), ModelTokenizer()).run()
    cfg = training_config(tmp_path, [root], "ddp-reference")
    cfg = replace(cfg, distributed=DistributedConfig(True, "gloo", 120,
        "contiguous_floor_v1", "ddp_average_world_scaled_global_sum_v1",
        "sum_unscale_normalize_clip_rank0_broadcast_v1", False, False, True, False))
    def run(config, tag, resume=None, stop=None):
        path = tmp_path/f"{tag}.json"; path.write_text(json.dumps(asdict(config)))
        result_path = tmp_path/f"{tag}-result.json"
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]
        mp.spawn(_ddp_worker, args=(3,port,str(path),str(result_path),resume,stop), nprocs=3, join=True)
        return json.loads(result_path.read_text())
    reference = run(cfg, "reference")
    resumed_cfg = replace(cfg, runtime=replace(cfg.runtime, output_dir=str(tmp_path/"ddp-resumed")))
    first = run(resumed_cfg, "partial", stop=1)
    second = run(resumed_cfg, "resumed", resume=first["checkpoint"])
    assert second["state"]["global_input_tokens"] == 144
    assert second["state"]["global_valid_targets"] == 135
    assert second["state"]["global_fast_updates"] == 3
    assert second["state"]["global_slow_steps"] == 2
    assert_training_equal(load_demo_checkpoint(reference["checkpoint"]), load_demo_checkpoint(second["checkpoint"]))
