from __future__ import annotations

import hashlib
import json
import os
import platform
from collections.abc import Iterable, Iterator, Mapping
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np

from lm_cl.config.data_schema import DataPipelineConfig
from lm_cl.data.iteration import close_iterable
from lm_cl.data.packed import (
    PackedShardWriter,
    manifest_content_hash,
    ordered_data_hash,
    validate_packed_shards,
)
from lm_cl.data.registry import OverlapRegistry
from lm_cl.data.selection import select_documents
from lm_cl.data.storage import (
    FileLock,
    RuntimeGuard,
    atomic_write_json,
    enforce_disk_limit,
    ensure_owned_root,
    estimate_stage_bytes,
    directory_size,
)
from lm_cl.data.tokenizer import (
    load_tokenizer_manifest,
    sha256_file,
    validate_tokenizer_manifest_reference,
    validate_tokenizer_reference,
)
from lm_cl.data.types import normalize_token_ids


class SimulatedInterruption(RuntimeError):
    """Test-only abrupt interruption that may precede the next checkpoint."""


def _source_tree_sha256() -> str:
    repository_root = Path(__file__).resolve().parents[3]
    paths = [repository_root / "pyproject.toml"]
    paths.extend(sorted((repository_root / "src" / "lm_cl").rglob("*.py")))
    digest = hashlib.sha256()
    for path in paths:
        if not path.is_file():
            continue
        relative = path.relative_to(repository_root).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _package_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for package in (
        "numpy",
        "PyYAML",
        "datasets",
        "transformers",
        "huggingface-hub",
    ):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def _config_fingerprint(
    config: DataPipelineConfig, tokenizer_manifest: dict[str, Any]
) -> str:
    value = {
        "config": config.to_dict(),
        "tokenizer_manifest_content_sha256": tokenizer_manifest[
            "manifest_content_sha256"
        ],
        "source_tree_sha256": _source_tree_sha256(),
        "python_version": platform.python_version(),
        "package_versions": _package_versions(),
    }
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _base_incomplete_manifest(
    config: DataPipelineConfig,
    tokenizer_manifest: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "format_version": config.packing.format_version,
        "resume_protocol_version": 1,
        "completion_status": "incomplete",
        "config_fingerprint": _config_fingerprint(config, tokenizer_manifest),
        "dataset": {
            "repo_id": config.dataset.repo_id,
            "revision": config.dataset.revision,
            "configuration": config.language_config,
            "split": config.dataset.split,
            "text_field": config.dataset.text_field,
            "id_field": config.dataset.id_field,
            "source_id_policy": config.dataset.source_id_policy,
            "missing_id_policy": config.dataset.missing_id_policy,
        },
        "stage": {
            "stage_id": config.stage.stage_id,
            "purpose": config.stage.purpose,
            "language": config.stage.language,
            "task_index": config.stage.task_index,
            "cycle_index": config.stage.cycle_index,
            "require_exact_output_tokens": (
                config.stage.require_exact_output_tokens
            ),
            "checkpoint_every_candidates": (
                config.stage.checkpoint_every_candidates
            ),
            "delete_temporary_cache_after_success": (
                config.stage.delete_temporary_cache_after_success
            ),
        },
        "tokenizer": {
            "repo_id": tokenizer_manifest["repo_id"],
            "revision": tokenizer_manifest["revision"],
            "tokenizer_class": tokenizer_manifest["tokenizer_class"],
            "base_vocab_size": tokenizer_manifest["base_vocab_size"],
            "effective_vocab_size": tokenizer_manifest[
                "effective_vocab_size"
            ],
            "maximum_emitted_token_id": tokenizer_manifest[
                "maximum_emitted_token_id"
            ],
            "model_embedding_vocab_size": tokenizer_manifest[
                "model_embedding_vocab_size"
            ],
            "registered_token_id_count": tokenizer_manifest[
                "registered_token_id_count"
            ],
            "trailing_unused_embedding_rows": tokenizer_manifest[
                "trailing_unused_embedding_rows"
            ],
            "special_token_ids": tokenizer_manifest["special_token_ids"],
            "files": tokenizer_manifest["files"],
            "manifest_content_sha256": tokenizer_manifest[
                "manifest_content_sha256"
            ],
            "tokenizer_content_sha256": tokenizer_manifest[
                "tokenizer_content_sha256"
            ],
        },
        "selection": {
            "max_input_documents": config.selection.max_input_documents,
            "max_output_tokens": config.selection.max_output_tokens,
            "max_runtime_seconds": config.selection.max_runtime_seconds,
            "document_order_seed": config.selection.document_order_seed,
            "split_seed": config.selection.split_seed,
            "validation_permyriad": config.selection.validation_permyriad,
            "shuffle_buffer_documents": (
                config.selection.shuffle_buffer_documents
            ),
            "order_algorithm": config.selection.order_algorithm,
            "split_algorithm": config.selection.split_algorithm,
            "document_hash_algorithm": (
                config.selection.document_hash_algorithm
            ),
            "token_hash_algorithm": config.selection.token_hash_algorithm,
        },
        "packing": {
            "dtype": config.packing.dtype,
            "write_boundaries": config.packing.write_boundaries,
            "eos_between_documents": config.packing.eos_between_documents,
            "eos_after_each_accepted_document": (
                config.packing.eos_after_each_document
            ),
            "truncate_final_document_to_budget": (
                config.packing.truncate_final_document_to_budget
            ),
            "add_bos": config.packing.add_bos,
            "add_chat_template": config.packing.add_chat_template,
            "add_special_tokens": config.packing.add_special_tokens,
            "mask_document_boundary_loss": (
                config.packing.mask_document_boundary_loss
            ),
            "max_shard_tokens": config.packing.max_shard_tokens,
            "max_shard_bytes": config.packing.max_shard_bytes,
            "checksum_algorithm": config.packing.checksum_algorithm,
        },
        "storage_limits": {
            "max_cache_bytes": config.storage.max_cache_bytes,
            "max_generated_bytes": config.storage.max_generated_bytes,
            "max_temporary_bytes": config.storage.max_temporary_bytes,
            "auto_clean_cache": config.storage.auto_clean_cache,
        },
        "overlap_registry": {
            "schema_version": 2,
            "relative_path_from_stage": "../../overlap.sqlite3",
        },
        "reader": {
            "sequence_length": config.reader.sequence_length,
            "global_sequences_per_batch": (
                config.reader.global_sequences_per_batch
            ),
            "drop_incomplete_sequence": (
                config.reader.drop_incomplete_sequence
            ),
        },
        "input_document_count": 0,
        "selected_document_count": 0,
        "accepted_document_count": 0,
        "rejection_counts": {},
        "selection_rejection_counts": {},
        "token_count": 0,
        "target_token_count": 0,
        "adjacent_target_token_count": 0,
        "shards": [],
        "boundaries_checkpoint_bytes": 0,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "software": {
            "package": "lm-cl",
            "package_version": "0.1.0",
            "source_tree_sha256": _source_tree_sha256(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "package_versions": _package_versions(),
        },
    }


def _load_or_initialize(
    *,
    config: DataPipelineConfig,
    tokenizer_manifest: dict[str, Any],
    stage_dir: Path,
) -> tuple[dict[str, Any], bool]:
    complete_path = stage_dir / "manifest.json"
    incomplete_path = stage_dir / "manifest.incomplete.json"
    if complete_path.exists():
        if not config.stage.resume:
            raise FileExistsError(
                f"Complete stage exists; refusing overwrite: {stage_dir}"
            )
        validate_packed_shards(complete_path)
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        if complete.get("config_fingerprint") != _config_fingerprint(
            config, tokenizer_manifest
        ):
            raise ValueError(
                "Complete stage configuration/tokenizer fingerprint mismatch"
            )
        if complete.get("boundaries") is None:
            (stage_dir / "boundaries.jsonl").unlink(missing_ok=True)
        incomplete_path.unlink(missing_ok=True)
        return complete, True
    if incomplete_path.exists():
        if not config.stage.resume:
            raise FileExistsError(
                f"Incomplete stage exists and resume=false: {stage_dir}"
            )
        state = json.loads(incomplete_path.read_text(encoding="utf-8"))
        if state.get("completion_status") != "incomplete":
            raise ValueError("Resume checkpoint has invalid completion status")
        if state.get("config_fingerprint") != _config_fingerprint(
            config, tokenizer_manifest
        ):
            raise ValueError("Resume configuration/tokenizer fingerprint mismatch")
        return state, False
    state = _base_incomplete_manifest(config, tokenizer_manifest)
    atomic_write_json(incomplete_path, state)
    return state, False


def _recover_boundaries(
    path: Path, *, checkpoint_bytes: int
) -> Any:
    if checkpoint_bytes:
        if not path.is_file() or path.stat().st_size < checkpoint_bytes:
            raise ValueError("Boundary file is shorter than resume checkpoint")
        with path.open("r+b") as handle:
            handle.truncate(checkpoint_bytes)
            handle.flush()
            os.fsync(handle.fileno())
    elif path.exists():
        path.unlink()
    return path.open("a", encoding="utf-8")


def _boundary_records(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _merge_rejections(
    selection_rejections: dict[str, int],
    processing_rejections: dict[str, int],
) -> dict[str, int]:
    result = dict(selection_rejections)
    for reason, count in processing_rejections.items():
        result[reason] = result.get(reason, 0) + count
    return dict(sorted(result.items()))


def _checkpoint(
    *,
    state: dict[str, Any],
    writer: PackedShardWriter,
    boundary_handle: Any,
    incomplete_path: Path,
    counters: dict[str, int],
    selection_rejections: dict[str, int],
    processing_rejections: dict[str, int],
) -> None:
    writer.flush()
    boundary_handle.flush()
    os.fsync(boundary_handle.fileno())
    state["input_document_count"] = counters.get("input_documents_seen", 0)
    state["selection_rejection_counts"] = dict(
        sorted(selection_rejections.items())
    )
    state["rejection_counts"] = _merge_rejections(
        selection_rejections, processing_rejections
    )
    state["token_count"] = writer.total_tokens
    complete_sequences = (
        writer.total_tokens // state["reader"]["sequence_length"]
    )
    state["target_token_count"] = complete_sequences * (
        state["reader"]["sequence_length"] - 1
    )
    state["adjacent_target_token_count"] = max(writer.total_tokens - 1, 0)
    state["shards"] = writer.checkpoint()
    state["boundaries_checkpoint_bytes"] = boundary_handle.tell()
    state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    temporary_bytes = len(
        json.dumps(state, sort_keys=True, separators=(",", ":")).encode()
    )
    if temporary_bytes > state["storage_limits"]["max_temporary_bytes"]:
        raise ValueError(
            "Incomplete manifest exceeds configured temporary-data cap: "
            f"{temporary_bytes} > "
            f"{state['storage_limits']['max_temporary_bytes']}"
        )
    atomic_write_json(incomplete_path, state)


def materialize_stage(
    config: DataPipelineConfig,
    *,
    rows: Iterable[Mapping[str, Any]],
    tokenizer: Any,
    tokenizer_manifest: dict[str, Any] | None = None,
    _interrupt_after_documents: int | None = None,
) -> dict[str, Any]:
    """Materialize a deterministic stage and release its live row stream."""
    try:
        return _materialize_stage(
            config,
            rows=rows,
            tokenizer=tokenizer,
            tokenizer_manifest=tokenizer_manifest,
            _interrupt_after_documents=_interrupt_after_documents,
        )
    finally:
        close_iterable(rows)


def _materialize_stage(
    config: DataPipelineConfig,
    *,
    rows: Iterable[Mapping[str, Any]],
    tokenizer: Any,
    tokenizer_manifest: dict[str, Any] | None = None,
    _interrupt_after_documents: int | None = None,
) -> dict[str, Any]:
    """Materialize a deterministic stage without retaining raw text.

    `_interrupt_after_documents` exists only for offline resume tests.
    """
    config.require_access_ready()
    if config.mode != "culturax_stage_materialize":
        raise ValueError("materialize_stage requires culturax_stage_materialize mode")
    if tokenizer_manifest is None:
        assert config.tokenizer.manifest_path is not None
        tokenizer_manifest = load_tokenizer_manifest(
            config.tokenizer.manifest_path
        )
    validate_tokenizer_manifest_reference(
        tokenizer_manifest, config.tokenizer
    )
    validate_tokenizer_reference(tokenizer, config.tokenizer)
    eos_token_id = config.tokenizer.expected_eos_token_id

    generated_root = ensure_owned_root(
        config.storage.generated_root, purpose="generated-data"
    )
    cache_root = ensure_owned_root(
        config.storage.hf_cache_root, purpose="hf-cache"
    )
    stage_dir = generated_root / "stages" / config.stage.stage_id
    complete_path = stage_dir / "manifest.json"
    if complete_path.exists():
        if not config.stage.resume:
            raise FileExistsError(
                f"Complete stage exists; refusing overwrite: {stage_dir}"
            )
        validate_packed_shards(complete_path)
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        if complete.get("config_fingerprint") != _config_fingerprint(
            config, tokenizer_manifest
        ):
            raise ValueError(
                "Complete stage exists but its configuration/tokenizer "
                "fingerprint differs"
            )
        if complete.get("boundaries") is None:
            (stage_dir / "boundaries.jsonl").unlink(missing_ok=True)
        (stage_dir / "manifest.incomplete.json").unlink(missing_ok=True)
        return complete
    remaining_token_budget = config.selection.max_output_tokens
    remaining_document_budget = config.selection.max_input_documents
    preview_incomplete_path = stage_dir / "manifest.incomplete.json"
    if preview_incomplete_path.exists():
        if not config.stage.resume:
            raise FileExistsError(
                f"Incomplete stage exists and resume=false: {stage_dir}"
            )
        preview = json.loads(
            preview_incomplete_path.read_text(encoding="utf-8")
        )
        if preview.get("completion_status") != "incomplete":
            raise ValueError("Resume checkpoint has invalid completion status")
        if preview.get("config_fingerprint") != _config_fingerprint(
            config, tokenizer_manifest
        ):
            raise ValueError(
                "Resume configuration/tokenizer fingerprint mismatch"
            )
        remaining_token_budget = max(
            config.selection.max_output_tokens
            - int(preview.get("token_count", 0)),
            0,
        )
        remaining_document_budget = max(
            config.selection.max_input_documents
            - int(preview.get("accepted_document_count", 0)),
            0,
        )
    estimate = estimate_stage_bytes(
        max_output_tokens=remaining_token_budget,
        max_input_documents=remaining_document_budget,
        write_boundaries=config.packing.write_boundaries,
    )
    existing_generated_bytes = directory_size(
        Path(config.storage.generated_root)
    )
    if (
        existing_generated_bytes + estimate["total_upper_bound"]
        > config.storage.max_generated_bytes
    ):
        raise ValueError(
            "Configured generated-data cap is below existing use plus dry-run "
            f"upper bound: {config.storage.max_generated_bytes} < "
            f"{existing_generated_bytes + estimate['total_upper_bound']}"
        )
    if (
        estimate["temporary_bytes_upper_bound"]
        > config.storage.max_temporary_bytes
    ):
        raise ValueError(
            "Configured temporary-data cap is below dry-run upper bound: "
            f"{config.storage.max_temporary_bytes} < "
            f"{estimate['temporary_bytes_upper_bound']}"
        )

    stage_dir.mkdir(parents=True, exist_ok=True)
    incomplete_path = stage_dir / "manifest.incomplete.json"
    boundary_path = stage_dir / "boundaries.jsonl"
    runtime = RuntimeGuard(config.selection.max_runtime_seconds)

    with FileLock(generated_root / ".overlap.lock"):
        state, already_complete = _load_or_initialize(
            config=config,
            tokenizer_manifest=tokenizer_manifest,
            stage_dir=stage_dir,
        )
        if already_complete:
            return state
        writer = PackedShardWriter(
            stage_dir,
            max_shard_tokens=config.packing.max_shard_tokens,
            checkpoint_shards=state["shards"],
        )
        boundary_handle = _recover_boundaries(
            boundary_path,
            checkpoint_bytes=int(state["boundaries_checkpoint_bytes"]),
        )
        processing_rejections = {
            key: value
            for key, value in state["rejection_counts"].items()
            if key not in state.get("selection_rejection_counts", {})
        }
        selection_rejections: dict[str, int] = {}
        counters: dict[str, int] = {}
        skip_selected = int(state["selected_document_count"])
        try:
            with OverlapRegistry(generated_root / "overlap.sqlite3") as registry:
                registered_count = registry.count_for_stage(
                    config.stage.stage_id
                )
                registered_max = registry.max_document_index_for_stage(
                    config.stage.stage_id
                )
                if registered_count > state["accepted_document_count"]:
                    registry.truncate_stage(
                        config.stage.stage_id,
                        keep_document_count=state[
                            "accepted_document_count"
                        ],
                    )
                    registered_count = registry.count_for_stage(
                        config.stage.stage_id
                    )
                    registered_max = registry.max_document_index_for_stage(
                        config.stage.stage_id
                    )
                if registered_count and registered_max != registered_count - 1:
                    raise ValueError(
                        "Current-stage overlap registry indices are not "
                        "contiguous"
                    )
                recovered_count = 0
                for item in _boundary_records(boundary_path):
                    if item["document_index"] >= registered_count:
                        registry.register(
                            content_sha256=item["content_sha256"],
                            token_ids_sha256=item["token_ids_sha256"],
                            stage_id=config.stage.stage_id,
                            purpose=config.stage.purpose,
                            language=config.stage.language,
                            split=item["split"],
                            source_id=item["source_id"],
                            document_index=item["document_index"],
                            token_start=item["token_start"],
                            token_end=item["token_end"],
                        )
                    recovered_count += 1
                if recovered_count != state["accepted_document_count"]:
                    raise ValueError(
                        "Boundary checkpoint count differs from incomplete "
                        "manifest"
                    )
                if (
                    registry.count_for_stage(config.stage.stage_id)
                    != state["accepted_document_count"]
                ):
                    raise ValueError(
                        "Recovered overlap registry count differs from "
                        "incomplete manifest"
                    )
                assert config.dataset.text_field is not None
                selected = select_documents(
                    rows,
                    text_field=config.dataset.text_field,
                    id_field=config.dataset.id_field,
                    purpose=config.stage.purpose,
                    config=config.selection,
                    rejection_counts=selection_rejections,
                    counters=counters,
                )
                last_checkpoint_selected = int(
                    state["selected_document_count"]
                )

                def checkpoint_if_due(*, force: bool = False) -> None:
                    nonlocal last_checkpoint_selected
                    processed = (
                        int(state["selected_document_count"])
                        - last_checkpoint_selected
                    )
                    if (
                        force
                        or processed
                        >= config.stage.checkpoint_every_candidates
                    ):
                        _checkpoint(
                            state=state,
                            writer=writer,
                            boundary_handle=boundary_handle,
                            incomplete_path=incomplete_path,
                            counters=counters,
                            selection_rejections=selection_rejections,
                            processing_rejections=processing_rejections,
                        )
                        last_checkpoint_selected = int(
                            state["selected_document_count"]
                        )

                for selected_index, (document, split) in enumerate(selected):
                    runtime.check()
                    enforce_disk_limit(
                        cache_root,
                        config.storage.max_cache_bytes,
                        label="Hugging Face cache",
                    )
                    if selected_index < skip_selected:
                        continue
                    state["selected_document_count"] += 1
                    existing = registry.lookup(document.content_sha256)
                    if existing is not None:
                        if existing["stage_id"] == config.stage.stage_id:
                            processing_rejections["duplicate_within_stage"] = (
                                processing_rejections.get(
                                    "duplicate_within_stage", 0
                                )
                                + 1
                            )
                        else:
                            processing_rejections["overlap_registry"] = (
                                processing_rejections.get(
                                    "overlap_registry", 0
                                )
                                + 1
                            )
                        checkpoint_if_due()
                        continue
                    try:
                        token_ids = normalize_token_ids(
                            tokenizer.encode(
                                document.text,
                                add_special_tokens=(
                                    config.packing.add_special_tokens
                                ),
                            )
                        )
                    except Exception:
                        processing_rejections["tokenization_error"] = (
                            processing_rejections.get("tokenization_error", 0)
                            + 1
                        )
                        checkpoint_if_due()
                        continue
                    if not token_ids:
                        processing_rejections["empty_tokenization"] = (
                            processing_rejections.get("empty_tokenization", 0) + 1
                        )
                        checkpoint_if_due()
                        continue
                    if (
                        min(token_ids) < 0
                        or max(token_ids)
                        > config.tokenizer.maximum_emitted_token_id
                        or max(token_ids)
                        >= config.tokenizer.model_embedding_vocab_size
                    ):
                        processing_rejections["token_id_out_of_range"] = (
                            processing_rejections.get(
                                "token_id_out_of_range", 0
                            )
                            + 1
                        )
                        checkpoint_if_due()
                        continue
                    token_ids_sha256 = hashlib.sha256(
                        np.asarray(token_ids, dtype="<u4").tobytes()
                    ).hexdigest()
                    token_existing = registry.lookup_token_ids(
                        token_ids_sha256
                    )
                    if token_existing is not None:
                        processing_rejections["token_sequence_overlap"] = (
                            processing_rejections.get(
                                "token_sequence_overlap", 0
                            )
                            + 1
                        )
                        checkpoint_if_due()
                        continue

                    remaining = (
                        config.selection.max_output_tokens - writer.total_tokens
                    )
                    if remaining <= 1:
                        processing_rejections["insufficient_token_budget"] = (
                            processing_rejections.get(
                                "insufficient_token_budget", 0
                            )
                            + 1
                        )
                        break
                    content_count = min(len(token_ids), remaining - 1)
                    truncated = content_count < len(token_ids)
                    packed = token_ids[:content_count] + [eos_token_id]
                    token_start = writer.total_tokens
                    writer.append(packed)
                    token_end = writer.total_tokens
                    document_index = int(state["accepted_document_count"])
                    boundary = {
                        "document_index": document_index,
                        "source_id": document.source_id,
                        "content_sha256": document.content_sha256,
                        "token_ids_sha256": token_ids_sha256,
                        "token_start": token_start,
                        "content_token_count": content_count,
                        "token_end": token_end,
                        "eos_after": config.packing.eos_after_each_document,
                        "truncated": truncated,
                        "split": split,
                    }
                    boundary_handle.write(
                        json.dumps(boundary, sort_keys=True, separators=(",", ":"))
                        + "\n"
                    )
                    state["accepted_document_count"] += 1
                    checkpoint_if_due()
                    registry.register(
                        content_sha256=document.content_sha256,
                        token_ids_sha256=token_ids_sha256,
                        stage_id=config.stage.stage_id,
                        purpose=config.stage.purpose,
                        language=config.stage.language,
                        split=split,
                        source_id=document.source_id,
                        document_index=document_index,
                        token_start=token_start,
                        token_end=token_end,
                    )
                    enforce_disk_limit(
                        generated_root,
                        config.storage.max_generated_bytes,
                        label="Generated data",
                    )
                    if (
                        _interrupt_after_documents is not None
                        and state["accepted_document_count"]
                        >= _interrupt_after_documents
                    ):
                        raise SimulatedInterruption(
                            "Simulated interruption before the next forced checkpoint"
                        )
                    if writer.total_tokens >= config.selection.max_output_tokens:
                        break

                if state["accepted_document_count"] == 0:
                    checkpoint_if_due(force=True)
                    raise ValueError(
                        "No documents were accepted; stage remains incomplete"
                    )
                if (
                    config.stage.require_exact_output_tokens
                    and writer.total_tokens
                    != config.selection.max_output_tokens
                ):
                    checkpoint_if_due(force=True)
                    raise ValueError(
                        "Exact output-token budget was not reached; stage "
                        "remains incomplete: "
                        f"{writer.total_tokens} != "
                        f"{config.selection.max_output_tokens}"
                    )
                checkpoint_if_due(force=True)
                boundary_handle.close()
                final_shards = writer.finalize()
                boundaries = None
                remove_boundary_after_complete = False
                document_records_metadata = {
                    "encoding": "jsonl",
                    "filename": boundary_path.name,
                    "record_count": state["accepted_document_count"],
                    "size_bytes": boundary_path.stat().st_size,
                    "sha256": sha256_file(boundary_path),
                }
                if config.packing.write_boundaries:
                    boundaries = dict(document_records_metadata)
                    state["accepted_documents"] = {
                        "storage": "boundary_sidecar",
                        **document_records_metadata,
                    }
                else:
                    state["accepted_documents"] = list(
                        _boundary_records(boundary_path)
                    )
                    remove_boundary_after_complete = True
                state["shards"] = final_shards
                state["boundaries"] = boundaries
                state.pop("boundaries_checkpoint_bytes", None)
                state.pop("selection_rejection_counts", None)
                state["completion_status"] = "complete"
                state["hit_output_token_cap"] = (
                    state["token_count"]
                    == config.selection.max_output_tokens
                )
                state["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
                state["ordered_data_sha256"] = ordered_data_hash(state)
                state["manifest_content_sha256"] = manifest_content_hash(state)
                complete_path = stage_dir / "manifest.json"
                atomic_write_json(complete_path, state)
                validate_packed_shards(complete_path)
                if remove_boundary_after_complete:
                    boundary_path.unlink(missing_ok=True)
                incomplete_path.unlink()
                return state
        finally:
            if not boundary_handle.closed:
                boundary_handle.close()
            writer.close()


def dry_run_materialization(config: DataPipelineConfig) -> dict[str, Any]:
    config.validate()
    stage_dir = (
        Path(config.storage.generated_root)
        / "stages"
        / config.stage.stage_id
    )
    incomplete_path = stage_dir / "manifest.incomplete.json"
    remaining_tokens = config.selection.max_output_tokens
    remaining_documents = config.selection.max_input_documents
    resume_checkpoint_detected = False
    if incomplete_path.is_file():
        checkpoint = json.loads(incomplete_path.read_text(encoding="utf-8"))
        if checkpoint.get("completion_status") != "incomplete":
            raise ValueError("Existing resume checkpoint status is invalid")
        remaining_tokens = max(
            remaining_tokens - int(checkpoint.get("token_count", 0)), 0
        )
        remaining_documents = max(
            remaining_documents
            - int(checkpoint.get("accepted_document_count", 0)),
            0,
        )
        resume_checkpoint_detected = True
    estimate = estimate_stage_bytes(
        max_output_tokens=remaining_tokens,
        max_input_documents=remaining_documents,
        write_boundaries=config.packing.write_boundaries,
    )
    existing_generated_bytes = directory_size(
        Path(config.storage.generated_root)
    )
    return {
        "stage_id": config.stage.stage_id,
        "mode": config.mode,
        "resume_checkpoint_detected": resume_checkpoint_detected,
        "remaining_output_tokens": remaining_tokens,
        "remaining_document_upper_bound": remaining_documents,
        "estimate": estimate,
        "existing_generated_bytes": existing_generated_bytes,
        "projected_generated_bytes_upper_bound": (
            existing_generated_bytes + estimate["total_upper_bound"]
        ),
        "configured_limits": {
            "max_cache_bytes": config.storage.max_cache_bytes,
            "max_generated_bytes": config.storage.max_generated_bytes,
            "max_temporary_bytes": config.storage.max_temporary_bytes,
        },
        "fits_generated_cap": (
            existing_generated_bytes + estimate["total_upper_bound"]
            <= config.storage.max_generated_bytes
        ),
        "fits_temporary_cap": (
            estimate["temporary_bytes_upper_bound"]
            <= config.storage.max_temporary_bytes
        ),
    }
