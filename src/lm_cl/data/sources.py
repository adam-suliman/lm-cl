from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from lm_cl.config import DataConfig, DataPipelineConfig
from lm_cl.data.iteration import close_iterable
from lm_cl.data.selection import select_documents
from lm_cl.data.storage import (
    RuntimeGuard,
    enforce_disk_limit,
    ensure_owned_root,
)
from lm_cl.data.tokenizer import (
    sha256_file,
    validate_tokenizer_manifest_reference,
    validate_tokenizer_reference,
)
from lm_cl.data.synthetic import SyntheticTokenDataset
from lm_cl.data.types import TokenBatch, TokenPosition, normalize_token_ids


class ArrayTokenSource:
    def __init__(self, token_ids: np.ndarray):
        array = np.asarray(token_ids)
        if array.ndim != 1:
            raise ValueError("Array token source must be one-dimensional")
        self.tokens = np.asarray(array, dtype=np.uint32)

    def _offset(self, position: TokenPosition) -> int:
        position.validate()
        if position.shard_index == 0:
            if position.token_offset > len(self.tokens):
                raise ValueError("Token position is past end")
            return position.token_offset
        if position.shard_index == 1 and position.token_offset == 0:
            return len(self.tokens)
        raise ValueError("Invalid array token position")

    def _position(self, offset: int) -> TokenPosition:
        if offset < 0 or offset > len(self.tokens):
            raise ValueError("Token offset is outside array")
        if offset == len(self.tokens):
            return TokenPosition(1, 0)
        return TokenPosition(0, offset)

    def iter_batches(
        self,
        *,
        sequence_length: int,
        global_sequences_per_batch: int,
        start: TokenPosition | None = None,
        sequence_prefix_count: int | None = None,
    ) -> Iterator[TokenBatch]:
        if sequence_length <= 1 or global_sequences_per_batch <= 0:
            raise ValueError("Invalid batch dimensions")
        position = start or TokenPosition(0, 0)
        if sequence_prefix_count is not None and sequence_prefix_count <= 0:
            raise ValueError("sequence_prefix_count must be positive")
        while True:
            offset = self._offset(position)
            if offset % sequence_length:
                raise ValueError("Batch start is not sequence-aligned")
            possible = (len(self.tokens) - offset) // sequence_length
            if sequence_prefix_count is not None:
                possible = min(
                    possible,
                    max(sequence_prefix_count - offset // sequence_length, 0),
                )
            if possible == 0:
                return
            sequences = min(global_sequences_per_batch, possible)
            count = sequences * sequence_length
            inputs = np.asarray(
                self.tokens[offset : offset + count].reshape(
                    sequences, sequence_length
                ),
                dtype=np.int64,
            )
            next_position = self._position(offset + count)
            yield TokenBatch(
                input_ids=inputs,
                labels=inputs.copy(),
                valid_target_count=sequences * (sequence_length - 1),
                start_position=position,
                next_position=next_position,
            )
            position = next_position


class SyntheticBatchSource:
    """Common-interface adapter preserving existing deterministic synthetic data."""

    def __init__(self, config: DataConfig):
        self.config = config
        self.dataset = SyntheticTokenDataset(config)

    def iter_batches(
        self,
        *,
        sequence_length: int,
        global_sequences_per_batch: int,
        start: TokenPosition | None = None,
        sequence_prefix_count: int | None = None,
    ) -> Iterator[TokenBatch]:
        if sequence_length != self.config.sequence_length:
            raise ValueError("Requested sequence length differs from synthetic config")
        position = start or TokenPosition(0, 0)
        if sequence_prefix_count is not None and sequence_prefix_count <= 0:
            raise ValueError("sequence_prefix_count must be positive")
        if position.shard_index not in {0, 1}:
            raise ValueError("Invalid synthetic position")
        index = (
            len(self.dataset)
            if position.shard_index == 1
            else position.token_offset
        )
        while index < len(self.dataset):
            if sequence_prefix_count is not None and index >= sequence_prefix_count:
                return
            count = min(
                global_sequences_per_batch, len(self.dataset) - index
            )
            if sequence_prefix_count is not None:
                count = min(count, sequence_prefix_count - index)
            arrays = [
                self.dataset[item]["input_ids"].numpy()
                for item in range(index, index + count)
            ]
            label_arrays = [
                self.dataset[item]["labels"].numpy()
                for item in range(index, index + count)
            ]
            inputs = np.stack(arrays).astype(np.int64, copy=False)
            labels = np.stack(label_arrays).astype(np.int64, copy=False)
            next_index = index + count
            next_position = (
                TokenPosition(1, 0)
                if next_index == len(self.dataset)
                else TokenPosition(0, next_index)
            )
            yield TokenBatch(
                input_ids=inputs,
                labels=labels,
                valid_target_count=int(
                    np.count_nonzero(
                        labels[:, 1:] != self.config.ignore_index
                    )
                ),
                start_position=position,
                next_position=next_position,
            )
            position = next_position
            index = next_index


class CulturaXStreamBatchSource(ArrayTokenSource):
    def __init__(self, token_ids: np.ndarray, report: dict[str, Any]):
        super().__init__(token_ids)
        self.report = report


def build_bounded_culturax_stream(
    config: DataPipelineConfig,
    *,
    rows: Iterable[Mapping[str, Any]],
    tokenizer: Any,
    tokenizer_manifest: dict[str, Any] | None = None,
) -> CulturaXStreamBatchSource:
    """Build a bounded source and deterministically release the live row stream."""
    try:
        return _build_bounded_culturax_stream(
            config,
            rows=rows,
            tokenizer=tokenizer,
            tokenizer_manifest=tokenizer_manifest,
        )
    finally:
        close_iterable(rows)


def _build_bounded_culturax_stream(
    config: DataPipelineConfig,
    *,
    rows: Iterable[Mapping[str, Any]],
    tokenizer: Any,
    tokenizer_manifest: dict[str, Any] | None = None,
) -> CulturaXStreamBatchSource:
    config.require_access_ready()
    if config.mode != "culturax_stream":
        raise ValueError("Config mode must be culturax_stream")
    validate_tokenizer_reference(tokenizer, config.tokenizer)
    if tokenizer_manifest is not None:
        validate_tokenizer_manifest_reference(
            tokenizer_manifest, config.tokenizer
        )
    eos_id = config.tokenizer.expected_eos_token_id
    cache_root = ensure_owned_root(
        config.storage.hf_cache_root, purpose="hf-cache"
    )
    generated_root = ensure_owned_root(
        config.storage.generated_root, purpose="generated-data"
    )
    enforce_disk_limit(
        generated_root,
        config.storage.max_generated_bytes,
        label="Generated data",
    )
    runtime = RuntimeGuard(config.selection.max_runtime_seconds)
    selection_rejections: dict[str, int] = {}
    processing_rejections: dict[str, int] = {}
    counters: dict[str, int] = {}
    accepted = []
    token_ids: list[int] = []
    seen_content_hashes: set[str] = set()
    seen_token_hashes: set[str] = set()
    assert config.dataset.text_field is not None
    for document, split in select_documents(
        rows,
        text_field=config.dataset.text_field,
        id_field=config.dataset.id_field,
        purpose=config.stage.purpose,
        config=config.selection,
        rejection_counts=selection_rejections,
        counters=counters,
    ):
        runtime.check()
        enforce_disk_limit(
            cache_root,
            config.storage.max_cache_bytes,
            label="Hugging Face cache",
        )
        if document.content_sha256 in seen_content_hashes:
            processing_rejections["duplicate_within_stream"] = (
                processing_rejections.get("duplicate_within_stream", 0) + 1
            )
            continue
        try:
            encoded = normalize_token_ids(
                tokenizer.encode(
                    document.text,
                    add_special_tokens=config.packing.add_special_tokens,
                )
            )
        except Exception:
            processing_rejections["tokenization_error"] = (
                processing_rejections.get("tokenization_error", 0) + 1
            )
            continue
        if not encoded:
            processing_rejections["empty_tokenization"] = (
                processing_rejections.get("empty_tokenization", 0) + 1
            )
            continue
        if (
            min(encoded) < 0
            or max(encoded) > config.tokenizer.maximum_emitted_token_id
            or max(encoded) >= config.tokenizer.model_embedding_vocab_size
        ):
            processing_rejections["token_id_out_of_range"] = (
                processing_rejections.get("token_id_out_of_range", 0) + 1
            )
            continue
        token_ids_sha256 = hashlib.sha256(
            np.asarray(encoded, dtype="<u4").tobytes()
        ).hexdigest()
        if token_ids_sha256 in seen_token_hashes:
            processing_rejections["token_sequence_overlap"] = (
                processing_rejections.get("token_sequence_overlap", 0) + 1
            )
            continue
        remaining = config.selection.max_output_tokens - len(token_ids)
        if remaining <= 1:
            break
        content_count = min(len(encoded), remaining - 1)
        start = len(token_ids)
        token_ids.extend(encoded[:content_count])
        token_ids.append(eos_id)
        accepted.append(
            {
                "document_index": len(accepted),
                "source_id": document.source_id,
                "content_sha256": document.content_sha256,
                "token_ids_sha256": token_ids_sha256,
                "token_start": start,
                "content_token_count": content_count,
                "token_end": len(token_ids),
                "eos_after": config.packing.eos_after_each_document,
                "truncated": content_count < len(encoded),
                "split": split,
            }
        )
        seen_content_hashes.add(document.content_sha256)
        seen_token_hashes.add(token_ids_sha256)
        if len(token_ids) >= config.selection.max_output_tokens:
            break
    rejections = dict(selection_rejections)
    for reason, count in processing_rejections.items():
        rejections[reason] = rejections.get(reason, 0) + count
    report = {
        "mode": "culturax_stream",
        "resolved_data_config": config.to_dict(),
        "dataset_repo_id": config.dataset.repo_id,
        "dataset_revision": config.dataset.revision,
        "dataset_configuration": config.language_config,
        "language": config.stage.language,
        "purpose": config.stage.purpose,
        "tokenizer": {
            "repo_id": config.tokenizer.repo_id,
            "revision": config.tokenizer.revision,
            "base_vocab_size": config.tokenizer.base_vocab_size,
            "effective_vocab_size": config.tokenizer.effective_vocab_size,
            "maximum_emitted_token_id": (
                config.tokenizer.maximum_emitted_token_id
            ),
            "model_embedding_vocab_size": (
                config.tokenizer.model_embedding_vocab_size
            ),
            "trailing_unused_embedding_rows": (
                config.tokenizer.model_embedding_vocab_size
                - config.tokenizer.maximum_emitted_token_id
                - 1
            ),
            "eos_token_id": config.tokenizer.expected_eos_token_id,
            "tokenizer_content_sha256": (
                None
                if tokenizer_manifest is None
                else tokenizer_manifest["tokenizer_content_sha256"]
            ),
            "manifest_content_sha256": (
                None
                if tokenizer_manifest is None
                else tokenizer_manifest["manifest_content_sha256"]
            ),
        },
        "input_document_count": counters.get("input_documents_seen", 0),
        "accepted_document_count": len(accepted),
        "accepted_documents": accepted,
        "rejection_counts": dict(sorted(rejections.items())),
        "token_count": len(token_ids),
        "target_token_count": (
            len(token_ids) // config.reader.sequence_length
        )
        * (config.reader.sequence_length - 1),
        "hit_input_document_cap": (
            counters.get("input_documents_seen", 0)
            >= config.selection.max_input_documents
        ),
        "hit_output_token_cap": (
            len(token_ids) >= config.selection.max_output_tokens
        ),
        "raw_text_retained": False,
    }
    return CulturaXStreamBatchSource(
        np.asarray(token_ids, dtype=np.uint32), report
    )


def open_token_batch_source(
    config: DataConfig | DataPipelineConfig,
    *,
    rows: Iterable[Mapping[str, Any]] | None = None,
    tokenizer: Any | None = None,
    tokenizer_manifest: dict[str, Any] | None = None,
    expected_tokenizer_manifest_sha256: str | None = None,
) -> Any:
    """Open every Phase 3 data mode behind the same batch-source interface."""
    if isinstance(config, DataConfig):
        return SyntheticBatchSource(config)
    config.validate()
    if config.mode == "culturax_stream":
        if rows is None or tokenizer is None:
            raise ValueError("CulturaX stream mode requires rows and tokenizer")
        return build_bounded_culturax_stream(
            config,
            rows=rows,
            tokenizer=tokenizer,
            tokenizer_manifest=tokenizer_manifest,
        )
    if config.mode in {"culturax_stage_materialize", "packed_shards"}:
        from lm_cl.data.packed import PackedShardSource
        from lm_cl.data.tokenizer import load_tokenizer_manifest

        config.require_access_ready()
        if config.mode == "packed_shards":
            config.require_packed_launch_ready()
        if expected_tokenizer_manifest_sha256 is None:
            assert config.tokenizer.manifest_path is not None
            expected_tokenizer_manifest_sha256 = load_tokenizer_manifest(
                config.tokenizer.manifest_path
            )["manifest_content_sha256"]
        stage_dir = (
            Path(config.storage.generated_root)
            / "stages"
            / config.stage.stage_id
        )
        source = PackedShardSource(
            stage_dir,
            drop_incomplete_sequence=config.reader.drop_incomplete_sequence,
            expected_tokenizer_manifest_sha256=(
                expected_tokenizer_manifest_sha256
            ),
        )
        manifest = source.manifest
        packed_identity = config.packed_manifest_identity
        if packed_identity is not None:
            manifest_path = source.stage_dir / "manifest.json"
            if sha256_file(manifest_path) != (
                packed_identity.manifest_file_sha256
            ):
                raise ValueError("Packed manifest file SHA-256 differs")
            if manifest["manifest_content_sha256"] != (
                packed_identity.manifest_content_sha256
            ):
                raise ValueError("Packed manifest content SHA-256 differs")
            if manifest["ordered_data_sha256"] != (
                packed_identity.ordered_data_sha256
            ):
                raise ValueError("Packed ordered-data SHA-256 differs")
            if manifest["token_count"] != packed_identity.expected_token_count:
                raise ValueError("Packed token count differs from frozen identity")
            if manifest["target_token_count"] != (
                packed_identity.expected_target_token_count
            ):
                raise ValueError("Packed target count differs from frozen identity")
        expected = {
            "stage.stage_id": (
                manifest["stage"]["stage_id"],
                config.stage.stage_id,
            ),
            "stage.purpose": (
                manifest["stage"]["purpose"],
                config.stage.purpose,
            ),
            "stage.language": (
                manifest["stage"]["language"],
                config.stage.language,
            ),
            "stage.task_index": (
                manifest["stage"]["task_index"],
                config.stage.task_index,
            ),
            "stage.cycle_index": (
                manifest["stage"]["cycle_index"],
                config.stage.cycle_index,
            ),
            "stage.require_exact_output_tokens": (
                manifest["stage"]["require_exact_output_tokens"],
                config.stage.require_exact_output_tokens,
            ),
            "stage.checkpoint_every_candidates": (
                manifest["stage"]["checkpoint_every_candidates"],
                config.stage.checkpoint_every_candidates,
            ),
            "dataset.repo_id": (
                manifest["dataset"]["repo_id"],
                config.dataset.repo_id,
            ),
            "dataset.revision": (
                manifest["dataset"]["revision"],
                config.dataset.revision,
            ),
            "dataset.configuration": (
                manifest["dataset"]["configuration"],
                config.language_config,
            ),
            "dataset.split": (
                manifest["dataset"]["split"],
                config.dataset.split,
            ),
            "dataset.text_field": (
                manifest["dataset"]["text_field"],
                config.dataset.text_field,
            ),
            "dataset.id_field": (
                manifest["dataset"]["id_field"],
                config.dataset.id_field,
            ),
            "dataset.missing_id_policy": (
                manifest["dataset"]["missing_id_policy"],
                config.dataset.missing_id_policy,
            ),
            "dataset.source_id_policy": (
                manifest["dataset"]["source_id_policy"],
                config.dataset.source_id_policy,
            ),
            "tokenizer.repo_id": (
                manifest["tokenizer"]["repo_id"],
                config.tokenizer.repo_id,
            ),
            "tokenizer.revision": (
                manifest["tokenizer"]["revision"],
                config.tokenizer.revision,
            ),
            "tokenizer.effective_vocab_size": (
                manifest["tokenizer"]["effective_vocab_size"],
                config.tokenizer.effective_vocab_size,
            ),
            "tokenizer.base_vocab_size": (
                manifest["tokenizer"]["base_vocab_size"],
                config.tokenizer.base_vocab_size,
            ),
            "tokenizer.maximum_emitted_token_id": (
                manifest["tokenizer"]["maximum_emitted_token_id"],
                config.tokenizer.maximum_emitted_token_id,
            ),
            "tokenizer.model_embedding_vocab_size": (
                manifest["tokenizer"]["model_embedding_vocab_size"],
                config.tokenizer.model_embedding_vocab_size,
            ),
            "tokenizer.eos_token_id": (
                manifest["tokenizer"]["special_token_ids"]["eos_token_id"],
                config.tokenizer.expected_eos_token_id,
            ),
            "selection.max_input_documents": (
                manifest["selection"]["max_input_documents"],
                config.selection.max_input_documents,
            ),
            "selection.max_output_tokens": (
                manifest["selection"]["max_output_tokens"],
                config.selection.max_output_tokens,
            ),
            "selection.document_order_seed": (
                manifest["selection"]["document_order_seed"],
                config.selection.document_order_seed,
            ),
            "selection.split_seed": (
                manifest["selection"]["split_seed"],
                config.selection.split_seed,
            ),
            "selection.validation_permyriad": (
                manifest["selection"]["validation_permyriad"],
                config.selection.validation_permyriad,
            ),
            "selection.shuffle_buffer_documents": (
                manifest["selection"]["shuffle_buffer_documents"],
                config.selection.shuffle_buffer_documents,
            ),
            "selection.order_algorithm": (
                manifest["selection"]["order_algorithm"],
                config.selection.order_algorithm,
            ),
            "selection.split_algorithm": (
                manifest["selection"]["split_algorithm"],
                config.selection.split_algorithm,
            ),
            "selection.document_hash_algorithm": (
                manifest["selection"]["document_hash_algorithm"],
                config.selection.document_hash_algorithm,
            ),
            "selection.token_hash_algorithm": (
                manifest["selection"]["token_hash_algorithm"],
                config.selection.token_hash_algorithm,
            ),
            "packing.max_shard_tokens": (
                manifest["packing"]["max_shard_tokens"],
                config.packing.max_shard_tokens,
            ),
            "packing.dtype": (
                manifest["packing"]["dtype"],
                config.packing.dtype,
            ),
            "packing.write_boundaries": (
                manifest["packing"]["write_boundaries"],
                config.packing.write_boundaries,
            ),
            "packing.eos_after_each_document": (
                manifest["packing"]["eos_after_each_accepted_document"],
                config.packing.eos_after_each_document,
            ),
            "packing.truncate_final_document_to_budget": (
                manifest["packing"]["truncate_final_document_to_budget"],
                config.packing.truncate_final_document_to_budget,
            ),
            "packing.add_bos": (
                manifest["packing"]["add_bos"],
                config.packing.add_bos,
            ),
            "packing.add_chat_template": (
                manifest["packing"]["add_chat_template"],
                config.packing.add_chat_template,
            ),
            "packing.add_special_tokens": (
                manifest["packing"]["add_special_tokens"],
                config.packing.add_special_tokens,
            ),
            "packing.eos_between_documents": (
                manifest["packing"]["eos_between_documents"],
                config.packing.eos_between_documents,
            ),
            "packing.mask_document_boundary_loss": (
                manifest["packing"]["mask_document_boundary_loss"],
                config.packing.mask_document_boundary_loss,
            ),
            "packing.checksum_algorithm": (
                manifest["packing"]["checksum_algorithm"],
                config.packing.checksum_algorithm,
            ),
            "packing.max_shard_bytes": (
                manifest["packing"]["max_shard_bytes"],
                config.packing.max_shard_bytes,
            ),
            "reader.sequence_length": (
                manifest["reader"]["sequence_length"],
                config.reader.sequence_length,
            ),
            "reader.drop_incomplete_sequence": (
                manifest["reader"]["drop_incomplete_sequence"],
                config.reader.drop_incomplete_sequence,
            ),
        }
        mismatches = [
            f"{name}: manifest={actual!r}, config={configured!r}"
            for name, (actual, configured) in expected.items()
            if actual != configured
        ]
        if mismatches:
            raise ValueError(
                "Packed stage/config identity mismatch: " + "; ".join(mismatches)
            )
        return source
    raise AssertionError(f"Unhandled data mode: {config.mode}")
