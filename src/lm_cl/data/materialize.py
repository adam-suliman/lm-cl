from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from collections.abc import Iterable, Iterator, Mapping
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from itertools import islice
from pathlib import Path
from typing import Any, Callable

import numpy as np

from lm_cl.config.data_schema import DataPipelineConfig
from lm_cl.data.huggingface import streaming_performance_settings
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
    directory_size,
    enforce_disk_limit,
    ensure_owned_root,
    estimate_stage_bytes,
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


MATERIALIZATION_ENGINE_VERSION = 2
LEGACY_ORDERED_MATERIALIZATION_MIGRATIONS = {
    # Public release commit 56c2f08. Engine v2 changes only execution: ordered
    # batches, transaction cadence, and cap-check cadence. The selection,
    # tokenization, overlap, packing, and hash contracts remain unchanged.
    "888f3af9474c74aab93ccdfa39c0a25b0ab223d1def241b70707aca7ea5fc30d": (
        "ordered_materialization_engine_v2"
    ),
    # Commit a318fe3 introduced engine v2 before bounded concurrent shard
    # prefetch. Prefetch preserves the documented contiguous shard order.
    "5f50e9d27d94968e959c7de71754ebe653a12b4413ec14bd594d61e58bf92223": (
        "bounded_contiguous_shard_prefetch"
    ),
    # Commit 82009c6 adds exact-order shard prefetch. Increasing only the
    # durable checkpoint interval changes crash replay cost, never accepted
    # documents, packing, hashes, or the final manifest identity.
    "3f4083e2c8d41fa449cea97290963eb319aba807bb8052ef06a2ea551456bdf6": (
        "amortized_materialization_checkpoint"
    ),
}
LEGACY_ORDERED_MATERIALIZATION_SOURCE_SHA256S = frozenset(
    LEGACY_ORDERED_MATERIALIZATION_MIGRATIONS
)


def materialization_performance_settings() -> dict[str, Any]:
    raw_batch_size = os.environ.get("LM_CL_TOKENIZER_BATCH_DOCUMENTS")
    if raw_batch_size is None:
        batch_size = min(4096, max(64, (os.cpu_count() or 1) * 16))
    else:
        try:
            batch_size = int(raw_batch_size)
        except ValueError as exc:
            raise ValueError(
                "LM_CL_TOKENIZER_BATCH_DOCUMENTS must be an integer"
            ) from exc
        if batch_size <= 0 or batch_size > 16_384:
            raise ValueError(
                "LM_CL_TOKENIZER_BATCH_DOCUMENTS must be in [1, 16384]"
            )
    checkpoint_raw = os.environ.get(
        "LM_CL_MATERIALIZATION_CHECKPOINT_CANDIDATES"
    )
    checkpoint_candidates = None
    if checkpoint_raw is not None:
        try:
            checkpoint_candidates = int(checkpoint_raw)
        except ValueError as exc:
            raise ValueError(
                "LM_CL_MATERIALIZATION_CHECKPOINT_CANDIDATES must be an integer"
            ) from exc
        if checkpoint_candidates <= 0 or checkpoint_candidates > 1_000_000:
            raise ValueError(
                "LM_CL_MATERIALIZATION_CHECKPOINT_CANDIDATES must be in "
                "[1, 1000000]"
            )
    settings = {
        "engine_version": MATERIALIZATION_ENGINE_VERSION,
        "tokenizer_batch_documents": batch_size,
        "checkpoint_candidates_override": checkpoint_candidates,
        "tokenizers_parallelism": os.environ.get(
            "TOKENIZERS_PARALLELISM", "unset"
        ),
        "rayon_num_threads": os.environ.get("RAYON_NUM_THREADS", "automatic"),
        "registry_cache_mib": os.environ.get(
            "LM_CL_REGISTRY_CACHE_MIB", "512"
        ),
        "registry_mmap_mib": os.environ.get(
            "LM_CL_REGISTRY_MMAP_MIB", "4096"
        ),
        "overlap_commit_policy": "stage_checkpoint_v1",
        "generated_cap_policy": "preflight_bound_and_final_check_v1",
        "cache_cap_check_interval_seconds": 5,
    }
    settings.update(streaming_performance_settings())
    return settings


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
    config: DataPipelineConfig,
    tokenizer_manifest: dict[str, Any],
    *,
    source_tree_sha256: str | None = None,
    python_version: str | None = None,
    package_versions: dict[str, str | None] | None = None,
) -> str:
    value = {
        "config": config.to_dict(),
        "tokenizer_manifest_content_sha256": tokenizer_manifest[
            "manifest_content_sha256"
        ],
        "source_tree_sha256": source_tree_sha256 or _source_tree_sha256(),
        "python_version": python_version or platform.python_version(),
        "package_versions": (
            _package_versions() if package_versions is None else package_versions
        ),
    }
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _software_record() -> dict[str, Any]:
    return {
        "package": "lm-cl",
        "package_version": "0.1.0",
        "source_tree_sha256": _source_tree_sha256(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "package_versions": _package_versions(),
        "materialization_performance": materialization_performance_settings(),
    }


def _load_compatible_incomplete_manifest(
    path: Path,
    *,
    config: DataPipelineConfig,
    tokenizer_manifest: dict[str, Any],
) -> dict[str, Any]:
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("completion_status") != "incomplete":
        raise ValueError("Resume checkpoint has invalid completion status")
    current_fingerprint = _config_fingerprint(config, tokenizer_manifest)
    if state.get("config_fingerprint") == current_fingerprint:
        return state

    previous_software = state.get("software")
    if not isinstance(previous_software, dict):
        raise ValueError("Resume configuration/tokenizer fingerprint mismatch")
    previous_source = previous_software.get("source_tree_sha256")
    previous_python = previous_software.get("python")
    previous_packages = previous_software.get("package_versions")
    if (
        previous_source not in LEGACY_ORDERED_MATERIALIZATION_SOURCE_SHA256S
        or not isinstance(previous_python, str)
        or not isinstance(previous_packages, dict)
    ):
        raise ValueError("Resume configuration/tokenizer fingerprint mismatch")
    legacy_fingerprint = _config_fingerprint(
        config,
        tokenizer_manifest,
        source_tree_sha256=previous_source,
        python_version=previous_python,
        package_versions=previous_packages,
    )
    if state.get("config_fingerprint") != legacy_fingerprint:
        raise ValueError("Resume configuration/tokenizer fingerprint mismatch")

    migrated_at = datetime.now(timezone.utc).isoformat()
    migration_kind = LEGACY_ORDERED_MATERIALIZATION_MIGRATIONS[
        previous_source
    ]
    migration_reason = f"{migration_kind}_performance_upgrade"
    history = state.setdefault("software_history", [])
    if not isinstance(history, list):
        raise ValueError("Resume software history is invalid")
    history.append(
        {
            "reason": migration_reason,
            "replaced_at_utc": migrated_at,
            "config_fingerprint": state["config_fingerprint"],
            "software": previous_software,
        }
    )
    state["software"] = _software_record()
    state["config_fingerprint"] = current_fingerprint
    state["resume_protocol_version"] = 2
    migrations = state.setdefault("resume_migrations", [])
    if not isinstance(migrations, list):
        raise ValueError("Resume migration history is invalid")
    migrations.append(
        {
            "kind": migration_kind,
            "at_utc": migrated_at,
            "from_source_tree_sha256": previous_source,
            "to_source_tree_sha256": state["software"]["source_tree_sha256"],
            "scientific_ordering_changed": False,
            "packed_token_semantics_changed": False,
        }
    )
    state["updated_at_utc"] = migrated_at
    atomic_write_json(path, state)
    return state


def _base_incomplete_manifest(
    config: DataPipelineConfig,
    tokenizer_manifest: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "format_version": config.packing.format_version,
        "resume_protocol_version": 2,
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
        "software": _software_record(),
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
        state = _load_compatible_incomplete_manifest(
            incomplete_path,
            config=config,
            tokenizer_manifest=tokenizer_manifest,
        )
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


def _ordered_batches(
    values: Iterator[Any], batch_size: int
) -> Iterator[list[Any]]:
    while True:
        batch = list(islice(values, batch_size))
        if not batch:
            return
        yield batch


def _encode_document_batch(
    tokenizer: Any,
    texts: list[str],
    *,
    add_special_tokens: bool,
) -> list[list[int] | None]:
    """Encode in an ordered fast-tokenizer batch with exact scalar fallback."""
    if not texts:
        return []
    if (
        len(texts) > 1
        and callable(tokenizer)
        and getattr(tokenizer, "is_fast", False)
    ):
        try:
            encoded = tokenizer(
                texts,
                add_special_tokens=add_special_tokens,
                padding=False,
                truncation=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
            values = encoded["input_ids"]
            if len(values) != len(texts):
                raise ValueError("Tokenizer batch output length mismatch")
            return [normalize_token_ids(item) for item in values]
        except Exception:
            # Preserve the original per-document rejection behavior if a batch
            # contains an input the tokenizer cannot encode.
            pass
    result: list[list[int] | None] = []
    for text in texts:
        try:
            result.append(
                normalize_token_ids(
                    tokenizer.encode(
                        text,
                        add_special_tokens=add_special_tokens,
                    )
                )
            )
        except Exception:
            result.append(None)
    return result


def materialize_stage(
    config: DataPipelineConfig,
    *,
    rows: Iterable[Mapping[str, Any]],
    tokenizer: Any,
    tokenizer_manifest: dict[str, Any] | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    _interrupt_after_documents: int | None = None,
) -> dict[str, Any]:
    """Materialize a deterministic stage and release its live row stream."""
    try:
        return _materialize_stage(
            config,
            rows=rows,
            tokenizer=tokenizer,
            tokenizer_manifest=tokenizer_manifest,
            progress_callback=progress_callback,
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
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
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
        preview = _load_compatible_incomplete_manifest(
            preview_incomplete_path,
            config=config,
            tokenizer_manifest=tokenizer_manifest,
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
                        keep_document_count=state["accepted_document_count"],
                    )
                    registered_count = registry.count_for_stage(
                        config.stage.stage_id
                    )
                    registered_max = registry.max_document_index_for_stage(
                        config.stage.stage_id
                    )
                if registered_count and registered_max != registered_count - 1:
                    raise ValueError(
                        "Current-stage overlap registry indices are not contiguous"
                    )
                recovered_count = 0
                for item in _boundary_records(boundary_path):
                    if item["document_index"] >= registered_count:
                        registry.insert_prechecked(
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
                registry.commit()
                if recovered_count != state["accepted_document_count"]:
                    raise ValueError(
                        "Boundary checkpoint count differs from incomplete manifest"
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
                selected = iter(
                    select_documents(
                        rows,
                        text_field=config.dataset.text_field,
                        id_field=config.dataset.id_field,
                        purpose=config.stage.purpose,
                        config=config.selection,
                        rejection_counts=selection_rejections,
                        counters=counters,
                    )
                )
                performance_settings = materialization_performance_settings()
                batch_size = int(
                    performance_settings["tokenizer_batch_documents"]
                )
                checkpoint_override = performance_settings[
                    "checkpoint_candidates_override"
                ]
                checkpoint_candidates = int(
                    checkpoint_override
                    if checkpoint_override is not None
                    else config.stage.checkpoint_every_candidates
                )
                last_checkpoint_selected = int(state["selected_document_count"])
                invocation_started = time.monotonic()
                invocation_start_tokens = writer.total_tokens
                pending_token_arrays: list[np.ndarray] = []
                pending_token_count = 0

                def current_token_count() -> int:
                    return writer.total_tokens + pending_token_count

                def flush_pending_tokens() -> None:
                    nonlocal pending_token_arrays, pending_token_count
                    if not pending_token_arrays:
                        return
                    writer.append(np.concatenate(pending_token_arrays))
                    pending_token_arrays = []
                    pending_token_count = 0

                def checkpoint_if_due(*, force: bool = False) -> None:
                    nonlocal last_checkpoint_selected
                    processed = (
                        int(state["selected_document_count"])
                        - last_checkpoint_selected
                    )
                    if (
                        not force
                        and processed < checkpoint_candidates
                    ):
                        return
                    flush_pending_tokens()
                    # A registry commit ahead of the atomic manifest is safe:
                    # resume truncates any rows beyond the manifest checkpoint.
                    registry.commit()
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
                    if progress_callback is not None:
                        elapsed = max(
                            time.monotonic() - invocation_started, 1e-12
                        )
                        added = (
                            int(state["token_count"])
                            - invocation_start_tokens
                        )
                        progress_callback(
                            {
                                "event": "materialization_progress",
                                "stage_id": config.stage.stage_id,
                                "token_count": int(state["token_count"]),
                                "target_token_count": (
                                    config.selection.max_output_tokens
                                ),
                                "accepted_document_count": int(
                                    state["accepted_document_count"]
                                ),
                                "selected_document_count": int(
                                    state["selected_document_count"]
                                ),
                                "input_document_count": int(
                                    state["input_document_count"]
                                ),
                                "invocation_elapsed_seconds": elapsed,
                                "invocation_added_tokens": added,
                                "invocation_tokens_per_second": added / elapsed,
                            }
                        )

                selected_cursor = 0
                reached_target = False
                for selected_batch in _ordered_batches(selected, batch_size):
                    runtime.check()
                    preexisting: list[dict[str, Any] | None] = []
                    candidate_positions: list[int] = []
                    candidate_texts: list[str] = []
                    for offset, (document, _) in enumerate(selected_batch):
                        selected_index = selected_cursor + offset
                        if selected_index < skip_selected:
                            preexisting.append(None)
                            continue
                        existing = registry.lookup(document.content_sha256)
                        preexisting.append(existing)
                        if existing is None:
                            candidate_positions.append(offset)
                            candidate_texts.append(document.text)
                    selected_cursor += len(selected_batch)
                    encoded_candidates = _encode_document_batch(
                        tokenizer,
                        candidate_texts,
                        add_special_tokens=config.packing.add_special_tokens,
                    )
                    encoded_by_position: dict[int, list[int] | None] = dict(
                        zip(candidate_positions, encoded_candidates, strict=True)
                    )

                    for offset, (document, split) in enumerate(selected_batch):
                        selected_index = (
                            selected_cursor - len(selected_batch) + offset
                        )
                        if selected_index < skip_selected:
                            continue
                        runtime.check()
                        state["selected_document_count"] += 1
                        existing = preexisting[offset]
                        if existing is None:
                            # Catch a duplicate accepted earlier in this same batch.
                            existing = registry.lookup(document.content_sha256)
                        if existing is not None:
                            reason = (
                                "duplicate_within_stage"
                                if existing["stage_id"] == config.stage.stage_id
                                else "overlap_registry"
                            )
                            processing_rejections[reason] = (
                                processing_rejections.get(reason, 0) + 1
                            )
                            checkpoint_if_due()
                            continue

                        token_ids = encoded_by_position[offset]
                        if token_ids is None:
                            processing_rejections["tokenization_error"] = (
                                processing_rejections.get("tokenization_error", 0)
                                + 1
                            )
                            checkpoint_if_due()
                            continue
                        if not token_ids:
                            processing_rejections["empty_tokenization"] = (
                                processing_rejections.get("empty_tokenization", 0)
                                + 1
                            )
                            checkpoint_if_due()
                            continue
                        token_array = np.asarray(token_ids, dtype=np.int64)
                        minimum_id = int(token_array.min())
                        maximum_id = int(token_array.max())
                        if (
                            minimum_id < 0
                            or maximum_id
                            > config.tokenizer.maximum_emitted_token_id
                            or maximum_id
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
                        token_u32 = np.asarray(token_array, dtype="<u4")
                        token_ids_sha256 = hashlib.sha256(
                            token_u32.tobytes()
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
                            config.selection.max_output_tokens
                            - current_token_count()
                        )
                        if remaining <= 1:
                            processing_rejections["insufficient_token_budget"] = (
                                processing_rejections.get(
                                    "insufficient_token_budget", 0
                                )
                                + 1
                            )
                            reached_target = True
                            break
                        content_count = min(len(token_u32), remaining - 1)
                        truncated = content_count < len(token_u32)
                        packed = np.empty(content_count + 1, dtype="<u4")
                        packed[:content_count] = token_u32[:content_count]
                        packed[content_count] = eos_token_id
                        token_start = current_token_count()
                        pending_token_arrays.append(packed)
                        pending_token_count += len(packed)
                        token_end = current_token_count()
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
                            json.dumps(
                                boundary,
                                sort_keys=True,
                                separators=(",", ":"),
                            )
                            + "\n"
                        )
                        state["accepted_document_count"] += 1
                        registry.insert_prechecked(
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
                        checkpoint_if_due()
                        if (
                            _interrupt_after_documents is not None
                            and state["accepted_document_count"]
                            >= _interrupt_after_documents
                        ):
                            raise SimulatedInterruption(
                                "Simulated interruption before the next "
                                "forced checkpoint"
                            )
                        if (
                            current_token_count()
                            >= config.selection.max_output_tokens
                        ):
                            reached_target = True
                            break
                    if reached_target:
                        break

                checkpoint_if_due(force=True)
                enforce_disk_limit(
                    cache_root,
                    config.storage.max_cache_bytes,
                    label="Hugging Face cache",
                )
                enforce_disk_limit(
                    generated_root,
                    config.storage.max_generated_bytes,
                    label="Generated data",
                )

                if state["accepted_document_count"] == 0:
                    raise ValueError(
                        "No documents were accepted; stage remains incomplete"
                    )
                if (
                    config.stage.require_exact_output_tokens
                    and writer.total_tokens
                    != config.selection.max_output_tokens
                ):
                    raise ValueError(
                        "Exact output-token budget was not reached; stage "
                        "remains incomplete: "
                        f"{writer.total_tokens} != "
                        f"{config.selection.max_output_tokens}"
                    )
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
