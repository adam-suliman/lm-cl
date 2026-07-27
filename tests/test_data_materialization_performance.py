from __future__ import annotations

import json
import threading
import types
from pathlib import Path

import pytest

import lm_cl.data.huggingface as huggingface_data
from lm_cl.config import (
    DataPipelineConfig,
    DatasetReference,
    PackingConfig,
    ReaderConfig,
    SelectionConfig,
    StageConfig,
    StorageConfig,
    TokenizerReference,
)
from lm_cl.data.materialize import (
    LEGACY_ORDERED_MATERIALIZATION_SOURCE_SHA256S,
    SimulatedInterruption,
    _config_fingerprint,
    materialize_stage,
)
from lm_cl.data.packed import validate_packed_shards
from lm_cl.data.storage import PeriodicDiskLimitGuard, ensure_owned_root


class BatchTokenizer:
    eos_token_id = 255
    pad_token_id = 255
    vocab_size = 256
    is_fast = True

    def __init__(self) -> None:
        self.batch_calls = 0
        self.scalar_calls = 0

    def __len__(self) -> int:
        return 256

    def get_vocab(self) -> dict[str, int]:
        return {f"token-{token_id}": token_id for token_id in range(256)}

    def encode(
        self, text: str, *, add_special_tokens: bool = False
    ) -> list[int]:
        assert add_special_tokens is False
        self.scalar_calls += 1
        return list(text.encode("utf-8"))

    def __call__(self, texts: list[str], **kwargs):
        assert kwargs["add_special_tokens"] is False
        self.batch_calls += 1
        return {"input_ids": [list(text.encode("utf-8")) for text in texts]}


def _manifest() -> dict[str, object]:
    return {
        "schema_version": 2,
        "kind": "lm-cl-tokenizer-manifest",
        "repo_id": "example/qwen3-tokenizer",
        "revision": "b" * 40,
        "snapshot_path": "/offline/fake",
        "tokenizer_class": "BatchTokenizer",
        "base_vocab_size": 256,
        "effective_vocab_size": 256,
        "maximum_emitted_token_id": 255,
        "model_embedding_vocab_size": 256,
        "registered_token_id_count": 256,
        "trailing_unused_embedding_rows": 0,
        "special_token_ids": {
            "bos_token_id": None,
            "eos_token_id": 255,
            "pad_token_id": 255,
            "unk_token_id": None,
            "sep_token_id": None,
            "cls_token_id": None,
            "mask_token_id": None,
            "additional_special_tokens_ids": [],
        },
        "files": [],
        "tokenizer_content_sha256": "e" * 64,
        "manifest_content_sha256": "c" * 64,
    }


def _rows(count: int = 128) -> list[dict[str, str]]:
    return [
        {
            "url": f"https://example.test/{index}",
            "text": f"ordered unique document {index:04d} " + "x" * 48,
        }
        for index in range(count)
    ]


def _config(
    tmp_path: Path,
    *,
    generated_name: str,
    token_cap: int = 512,
    checkpoint_every: int = 8,
) -> DataPipelineConfig:
    generated = (tmp_path / generated_name).resolve()
    ensure_owned_root(generated, purpose="generated-data")
    tokenizer_manifest = generated / "tokenizers/qwen3/manifest.json"
    tokenizer_manifest.parent.mkdir(parents=True, exist_ok=True)
    tokenizer_manifest.write_text("{}\n", encoding="utf-8")
    languages = {
        "en": "en",
        "zh_written": "zh",
        "fr": "fr",
        "ja": "ja",
        "es": "es",
        "de": "de",
        "pt": "pt",
        "ru": "ru",
        "vi": "vi",
    }
    config = DataPipelineConfig(
        schema_version=1,
        name="ordered-batch-test",
        mode="culturax_stage_materialize",
        run_kind="smoke",
        dataset=DatasetReference(
            repo_id="uonlp/CulturaX",
            revision="a" * 40,
            split="train",
            text_field="text",
            id_field="url",
            source_id_policy="sha256_canonical_json",
            missing_id_policy="content_sha256",
            language_configs=languages,
        ),
        tokenizer=TokenizerReference(
            repo_id="example/qwen3-tokenizer",
            revision="b" * 40,
            manifest_path=str(tokenizer_manifest),
            base_vocab_size=256,
            effective_vocab_size=256,
            maximum_emitted_token_id=255,
            model_embedding_vocab_size=256,
            expected_eos_token_id=255,
            expected_pad_token_id=255,
        ),
        selection=SelectionConfig(
            max_input_documents=128,
            max_output_tokens=token_cap,
            max_runtime_seconds=60,
            document_order_seed=31,
            split_seed=41,
            validation_permyriad=100,
            shuffle_buffer_documents=16,
            order_algorithm="bounded_buffer_python_v1",
            split_algorithm="sha256_permyriad_v1",
            document_hash_algorithm="sha256_utf8",
            token_hash_algorithm="sha256_uint32_le",
        ),
        storage=StorageConfig(
            hf_cache_root=str((tmp_path / f"cache-{generated_name}").resolve()),
            generated_root=str(generated),
            max_cache_bytes=10_000_000,
            max_generated_bytes=10_000_000,
            max_temporary_bytes=2_000_000,
            auto_clean_cache=False,
        ),
        packing=PackingConfig(
            format_version=1,
            dtype="uint32_le",
            max_shard_tokens=128,
            max_shard_bytes=512,
            write_boundaries=True,
            add_bos=False,
            add_chat_template=False,
            add_special_tokens=False,
            eos_between_documents=True,
            eos_after_each_document=True,
            truncate_final_document_to_budget=True,
            mask_document_boundary_loss=False,
            checksum_algorithm="sha256",
        ),
        stage=StageConfig(
            stage_id="ordered-stage",
            purpose="continual_train",
            language="en",
            task_index=0,
            cycle_index=0,
            resume=True,
            delete_temporary_cache_after_success=False,
            require_exact_output_tokens=True,
            checkpoint_every_candidates=checkpoint_every,
        ),
        reader=ReaderConfig(
            sequence_length=8,
            global_sequences_per_batch=2,
            drop_incomplete_sequence=True,
        ),
    )
    config.validate()
    return config


def _run(config, tokenizer, **kwargs):
    return materialize_stage(
        config,
        rows=_rows(),
        tokenizer=tokenizer,
        tokenizer_manifest=_manifest(),
        **kwargs,
    )


def _packed_bytes(config: DataPipelineConfig, manifest: dict) -> bytes:
    stage = Path(config.storage.generated_root) / "stages/ordered-stage"
    return b"".join(
        (stage / shard["filename"]).read_bytes()
        for shard in manifest["shards"]
    )


def test_ordered_batching_preserves_packed_identity(tmp_path, monkeypatch):
    scalar_config = _config(tmp_path, generated_name="scalar")
    monkeypatch.setenv("LM_CL_TOKENIZER_BATCH_DOCUMENTS", "1")
    scalar_tokenizer = BatchTokenizer()
    scalar = _run(scalar_config, scalar_tokenizer)

    batch_config = _config(tmp_path, generated_name="batch")
    monkeypatch.setenv("LM_CL_TOKENIZER_BATCH_DOCUMENTS", "32")
    batch_tokenizer = BatchTokenizer()
    batched = _run(batch_config, batch_tokenizer)

    assert batch_tokenizer.batch_calls > 0
    assert batch_tokenizer.scalar_calls == 0
    assert _packed_bytes(scalar_config, scalar) == _packed_bytes(
        batch_config, batched
    )
    assert scalar["ordered_data_sha256"] == batched["ordered_data_sha256"]
    assert scalar["accepted_document_count"] == batched[
        "accepted_document_count"
    ]
    validate_packed_shards(
        Path(batch_config.storage.generated_root) / "stages/ordered-stage"
    )


def test_progress_reports_checkpoint_throughput(tmp_path, monkeypatch):
    monkeypatch.setenv("LM_CL_TOKENIZER_BATCH_DOCUMENTS", "16")
    events = []
    manifest = _run(
        _config(tmp_path, generated_name="progress", checkpoint_every=4),
        BatchTokenizer(),
        progress_callback=events.append,
    )
    assert events
    assert events[-1]["token_count"] == manifest["token_count"]
    assert events[-1]["invocation_tokens_per_second"] >= 0
    assert all(event["event"] == "materialization_progress" for event in events)


@pytest.mark.parametrize(
    ("legacy_source", "expected_reason"),
    [
        (
            "888f3af9474c74aab93ccdfa39c0a25b0ab223d1def241b70707aca7ea5fc30d",
            "ordered_materialization_engine_v2_performance_upgrade",
        ),
        (
            "5f50e9d27d94968e959c7de71754ebe653a12b4413ec14bd594d61e58bf92223",
            "bounded_contiguous_shard_prefetch_performance_upgrade",
        ),
    ],
)
def test_legacy_incomplete_stage_migrates_and_resumes_exactly(
    tmp_path, monkeypatch, legacy_source, expected_reason
):
    monkeypatch.setenv("LM_CL_TOKENIZER_BATCH_DOCUMENTS", "16")
    config = _config(
        tmp_path,
        generated_name="resume",
        token_cap=768,
        checkpoint_every=4,
    )
    with pytest.raises(SimulatedInterruption):
        _run(config, BatchTokenizer(), _interrupt_after_documents=4)

    incomplete = (
        Path(config.storage.generated_root)
        / "stages/ordered-stage/manifest.incomplete.json"
    )
    state = json.loads(incomplete.read_text(encoding="utf-8"))
    assert legacy_source in LEGACY_ORDERED_MATERIALIZATION_SOURCE_SHA256S
    previous_software = dict(state["software"])
    previous_software.pop("materialization_performance", None)
    previous_software["source_tree_sha256"] = legacy_source
    state["software"] = previous_software
    state["resume_protocol_version"] = 1
    state["config_fingerprint"] = _config_fingerprint(
        config,
        _manifest(),
        source_tree_sha256=legacy_source,
        python_version=previous_software["python"],
        package_versions=previous_software["package_versions"],
    )
    incomplete.write_text(json.dumps(state), encoding="utf-8")

    resumed = _run(config, BatchTokenizer())
    assert resumed["resume_protocol_version"] == 2
    assert resumed["software_history"][-1]["reason"] == expected_reason
    assert (
        resumed["resume_migrations"][-1]["scientific_ordering_changed"]
        is False
    )

    clean_config = _config(
        tmp_path,
        generated_name="clean",
        token_cap=768,
        checkpoint_every=4,
    )
    clean = _run(clean_config, BatchTokenizer())
    assert _packed_bytes(config, resumed) == _packed_bytes(clean_config, clean)
    assert resumed["ordered_data_sha256"] == clean["ordered_data_sha256"]


def test_periodic_disk_guard_amortizes_recursive_checks(monkeypatch, tmp_path):
    import lm_cl.data.storage as storage

    calls = []
    clock = iter([0.0, 0.1, 0.2, 5.1])
    monkeypatch.setattr(storage.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(
        storage,
        "enforce_disk_limit",
        lambda path, maximum, *, label: calls.append((path, maximum, label)) or 7,
    )
    guard = PeriodicDiskLimitGuard(
        tmp_path,
        100,
        label="test",
        interval_seconds=5,
    )
    assert guard.check() == 7
    assert guard.check() == 7
    assert guard.check() == 7
    assert guard.check() == 7
    assert len(calls) == 2


def test_parallel_shard_prefetch_preserves_exact_concatenated_order(
    tmp_path, monkeypatch
):
    barrier = threading.Barrier(3)
    closed = []
    shard_rows = [
        [
            {"url": "shard-0-row-0", "text": "zero-a"},
            {"url": "shard-0-row-1", "text": "zero-b"},
        ],
        [{"url": "shard-1-row-0", "text": "one-a"}],
        [
            {"url": "shard-2-row-0", "text": "two-a"},
            {"url": "shard-2-row-1", "text": "two-b"},
        ],
    ]

    class Shard:
        def __init__(self, index):
            self.index = index

        def __iter__(self):
            barrier.wait(timeout=5)
            yield from shard_rows[self.index]

        def close(self):
            closed.append(f"shard-{self.index}")

    class Stream:
        num_shards = 3

        def shard(self, *, num_shards, index, contiguous):
            assert num_shards == self.num_shards
            assert contiguous is True
            return Shard(index)

        def close(self):
            closed.append("stream")

    monkeypatch.setenv("LM_CL_STREAM_PREFETCH_SHARDS", "3")
    monkeypatch.setenv("LM_CL_STREAM_PREFETCH_ROWS_PER_SHARD", "2")
    monkeypatch.setattr(
        huggingface_data,
        "_datasets_module",
        lambda: types.SimpleNamespace(load_dataset=lambda *args, **kwargs: Stream()),
    )
    config = _config(tmp_path, generated_name="parallel-stream")
    rows = list(huggingface_data.stream_culturax_rows(config))
    assert [row["url"] for row in rows] == [
        "shard-0-row-0",
        "shard-0-row-1",
        "shard-1-row-0",
        "shard-2-row-0",
        "shard-2-row-1",
    ]
    assert set(closed) == {"shard-0", "shard-1", "shard-2", "stream"}


def test_parallel_shard_prefetch_propagates_worker_failure(tmp_path, monkeypatch):
    closed = []

    class Shard:
        def __init__(self, index):
            self.index = index

        def __iter__(self):
            if self.index == 1:
                raise ValueError("broken shard")
            yield {"url": f"row-{self.index}", "text": "valid"}

        def close(self):
            closed.append(self.index)

    class Stream:
        num_shards = 2

        def shard(self, *, num_shards, index, contiguous):
            return Shard(index)

        def close(self):
            closed.append("stream")

    monkeypatch.setenv("LM_CL_STREAM_PREFETCH_SHARDS", "2")
    monkeypatch.setattr(
        huggingface_data,
        "_datasets_module",
        lambda: types.SimpleNamespace(load_dataset=lambda *args, **kwargs: Stream()),
    )
    config = _config(tmp_path, generated_name="failed-stream")
    with pytest.raises(RuntimeError, match="broken shard"):
        list(huggingface_data.stream_culturax_rows(config))
    assert set(closed) == {0, 1, "stream"}
