from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lm_cl.config.data_schema import (
    DATA_SCHEMA_VERSION,
    DataPipelineConfig,
    DatasetReference,
    PackedManifestIdentity,
    PackingConfig,
    ReaderConfig,
    SelectionConfig,
    StageConfig,
    StorageConfig,
    TokenizerReference,
)
from lm_cl.config.data_yaml import save_data_pipeline_config
from lm_cl.data.packed import load_packed_manifest, validate_packed_shards
from lm_cl.data.registry import OverlapRegistry
from lm_cl.data.storage import atomic_write_json
from lm_cl.data.tokenizer import (
    load_tokenizer_manifest,
    sha256_file,
)
from lm_cl.launcher.schema import (
    CYCLE_MANIFEST_POLICY,
    PUBLIC_LANGUAGE_ORDER,
    LauncherConfig,
    TokenBudget,
    resolve_token_budget,
)
from lm_cl.training.checkpoint import canonical_sha256


CULTURAX_REPO_ID = "uonlp/CulturaX"
CULTURAX_REVISION = "6a8734bc69fefcbb7735f4f9250f43e4cd7a442e"
TOKENIZER_REPO_ID = "Qwen/Qwen3-0.6B-Base"
TOKENIZER_REVISION = "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"
LANGUAGE_CONFIGS = {
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


def _lane_namespace(language: str) -> str:
    return f"language-{language.replace('_', '-')}"


def _lane_registry_path(generated_root: Path, language: str) -> Path:
    return (
        generated_root
        / "preparation-lanes"
        / f"overlap-{_lane_namespace(language)}.sqlite3"
    )


def _checkpoint_sqlite(path: Path) -> None:
    if not path.is_file():
        return
    connection = sqlite3.connect(path)
    try:
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        connection.close()


def _rows_differ_clause(left: str, right: str) -> str:
    columns = (
        "content_sha256",
        "token_ids_sha256",
        "stage_id",
        "purpose",
        "language",
        "split_name",
        "source_id",
        "document_index",
        "token_start",
        "token_end",
    )
    return " OR ".join(
        f"{left}.{column} IS NOT {right}.{column}" for column in columns
    )


def _parallel_manifest_records(
    manifest_paths: list[Path],
) -> list[dict[str, Any]]:
    records = []
    for path in manifest_paths:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        overlap = manifest.get("overlap_registry", {})
        namespace = overlap.get("preparation_lane_namespace")
        if namespace is None:
            continue
        if overlap.get("requires_global_merge") is not True:
            raise ValueError(f"Lane manifest lacks global-merge gate: {path}")
        records.append(
            {
                "path": str(path),
                "manifest_file_sha256": sha256_file(path),
                "manifest_content_sha256": manifest[
                    "manifest_content_sha256"
                ],
                "ordered_data_sha256": manifest["ordered_data_sha256"],
                "stage_id": manifest["stage"]["stage_id"],
                "language": manifest["stage"]["language"],
                "lane_namespace": namespace,
                "accepted_document_count": int(
                    manifest["accepted_document_count"]
                ),
            }
        )
    return records


def _parallel_audit_is_current(
    audit_path: Path,
    records: list[dict[str, Any]],
    global_registry: Path,
) -> bool:
    if not audit_path.is_file() or not global_registry.is_file():
        return False
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    expected = {
        item["path"]: item["manifest_file_sha256"] for item in records
    }
    actual = {
        item["path"]: item["manifest_file_sha256"]
        for item in audit.get("manifests", [])
    }
    return (
        audit.get("status") == "complete"
        and audit.get("cross_lane_conflicts") == []
        and actual == expected
        and audit.get("global_registry", {}).get("sha256")
        == sha256_file(global_registry)
    )


def _merge_parallel_overlap_registries(
    config: LauncherConfig,
    manifest_paths: list[Path],
) -> dict[str, Any] | None:
    records = _parallel_manifest_records(manifest_paths)
    if not records:
        return None
    generated_root = Path(config.data.generated_root).resolve()
    lane_root = generated_root / "preparation-lanes"
    lane_root.mkdir(parents=True, exist_ok=True)
    global_registry = generated_root / "overlap.sqlite3"
    audit_path = generated_root / "parallel-preparation-audit.json"
    if _parallel_audit_is_current(audit_path, records, global_registry):
        return json.loads(audit_path.read_text(encoding="utf-8"))

    languages = sorted({item["language"] for item in records})
    lane_paths = {
        language: _lane_registry_path(generated_root, language)
        for language in languages
    }
    missing_lanes = [str(path) for path in lane_paths.values() if not path.is_file()]
    if missing_lanes:
        raise FileNotFoundError(
            "Parallel preparation lane registries are missing: "
            + ", ".join(missing_lanes)
        )
    for path in lane_paths.values():
        _checkpoint_sqlite(path)
    _checkpoint_sqlite(global_registry)

    temporary = lane_root / f"global-merge-{os.getpid()}.sqlite3"
    if temporary.exists():
        raise FileExistsError(f"Refusing stale merge target: {temporary}")
    conflict: dict[str, Any] | None = None
    with OverlapRegistry(temporary) as merged:
        connection = merged.connection
        if global_registry.is_file():
            connection.execute("ATTACH DATABASE ? AS baseline", (str(global_registry),))
            connection.execute(
                "INSERT INTO documents SELECT * FROM baseline.documents"
            )
            connection.commit()
            connection.execute("DETACH DATABASE baseline")
        for lane_index, (language, lane_path) in enumerate(lane_paths.items()):
            alias = f"lane_{lane_index}"
            connection.execute(
                f"ATTACH DATABASE ? AS {alias}", (str(lane_path),)
            )
            content_row = connection.execute(
                f"""
                SELECT lane.content_sha256, lane.stage_id, main.stage_id
                FROM {alias}.documents AS lane
                JOIN documents AS main
                  ON main.content_sha256=lane.content_sha256
                WHERE {_rows_differ_clause('lane', 'main')}
                LIMIT 1
                """
            ).fetchone()
            token_row = connection.execute(
                f"""
                SELECT lane.token_ids_sha256, lane.stage_id, main.stage_id
                FROM {alias}.documents AS lane
                JOIN documents AS main
                  ON main.token_ids_sha256=lane.token_ids_sha256
                WHERE {_rows_differ_clause('lane', 'main')}
                LIMIT 1
                """
            ).fetchone()
            if content_row is not None or token_row is not None:
                conflict = {
                    "language": language,
                    "content_conflict": content_row,
                    "token_conflict": token_row,
                }
                connection.execute(f"DETACH DATABASE {alias}")
                break
            connection.execute(
                f"INSERT OR IGNORE INTO documents SELECT * FROM {alias}.documents"
            )
            connection.commit()
            connection.execute(f"DETACH DATABASE {alias}")
        if conflict is not None:
            raise ValueError(
                "Cross-lane overlap prevents deterministic global merge: "
                f"{conflict}"
            )
        for record in records:
            count = merged.count_for_stage(record["stage_id"])
            if count != record["accepted_document_count"]:
                raise ValueError(
                    "Merged registry count mismatch for stage "
                    f"{record['stage_id']}: {count} != "
                    f"{record['accepted_document_count']}"
                )
        merged.commit()
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        merged_count = merged.count()

    backup = None
    if global_registry.is_file():
        old_sha = sha256_file(global_registry)
        backup = lane_root / f"global-before-merge-{old_sha[:16]}.sqlite3"
        if backup.exists():
            if sha256_file(backup) != old_sha:
                raise ValueError(f"Existing global-registry backup differs: {backup}")
        else:
            shutil.copy2(global_registry, backup)
    os.replace(temporary, global_registry)
    directory_fd = os.open(generated_root, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)

    audit = {
        "schema_version": 1,
        "status": "complete",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "parallel_language_lanes_then_checked_global_merge_v1",
        "cross_lane_conflicts": [],
        "manifests": sorted(records, key=lambda item: item["path"]),
        "lane_registries": [
            {
                "language": language,
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for language, path in lane_paths.items()
        ],
        "global_registry": {
            "path": str(global_registry),
            "size_bytes": global_registry.stat().st_size,
            "sha256": sha256_file(global_registry),
            "document_count": merged_count,
            "pre_merge_backup": None if backup is None else str(backup),
        },
    }
    atomic_write_json(audit_path, audit)
    return audit


def _tokenizer_reference(
    config: LauncherConfig,
) -> tuple[TokenizerReference, dict[str, Any], str]:
    manifest_path = Path(config.data.tokenizer_manifest).resolve()
    manifest = load_tokenizer_manifest(manifest_path)
    if manifest.get("repo_id") != TOKENIZER_REPO_ID:
        raise ValueError("Tokenizer repository differs from the frozen contract")
    if manifest.get("revision") != TOKENIZER_REVISION:
        raise ValueError("Tokenizer revision differs from the frozen contract")
    expected = {
        "base_vocab_size": 151_643,
        "effective_vocab_size": 151_669,
        "maximum_emitted_token_id": 151_668,
        "model_embedding_vocab_size": 151_680,
    }
    mismatches = {
        key: {"expected": value, "actual": manifest.get(key)}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    special = manifest.get("special_token_ids", {})
    if special.get("eos_token_id") != 151_643:
        mismatches["eos_token_id"] = {
            "expected": 151_643,
            "actual": special.get("eos_token_id"),
        }
    if special.get("pad_token_id") != 151_643:
        mismatches["pad_token_id"] = {
            "expected": 151_643,
            "actual": special.get("pad_token_id"),
        }
    if mismatches:
        raise ValueError(f"Tokenizer contract mismatch: {mismatches}")
    reference = TokenizerReference(
        repo_id=manifest["repo_id"],
        revision=manifest["revision"],
        manifest_path=str(manifest_path),
        base_vocab_size=manifest["base_vocab_size"],
        effective_vocab_size=manifest["effective_vocab_size"],
        maximum_emitted_token_id=manifest["maximum_emitted_token_id"],
        model_embedding_vocab_size=manifest["model_embedding_vocab_size"],
        expected_eos_token_id=special["eos_token_id"],
        expected_pad_token_id=special["pad_token_id"],
    )
    reference.validate()
    return reference, manifest, sha256_file(manifest_path)


def render_manifest_path(
    config: LauncherConfig,
    *,
    cycle: int,
    language: str,
    task_index: int,
    budget: TokenBudget,
) -> Path:
    relative = config.data.manifest_template.format(
        cycle=cycle,
        language=language,
        task_index=task_index,
        effective_tokens=budget.effective_input_tokens,
    )
    root = Path(config.data.manifest_root).resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("Rendered manifest escapes data.manifest_root") from exc
    return path


def _manifest_identity(
    manifest_path: Path,
    *,
    expected_language: str | None,
    expected_cycle: int | None,
    expected_task_index: int | None,
    budget: TokenBudget | None,
    tokenizer_manifest: dict[str, Any],
    tokenizer_manifest_file_sha256: str,
    full_checksum_validation: bool,
) -> dict[str, Any]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Packed manifest not found: {manifest_path}")
    if full_checksum_validation:
        validate_packed_shards(
            manifest_path,
            expected_tokenizer_manifest_sha256=(
                tokenizer_manifest["manifest_content_sha256"]
            ),
        )
    _, manifest = load_packed_manifest(manifest_path)
    if manifest["tokenizer"]["manifest_content_sha256"] != (
        tokenizer_manifest["manifest_content_sha256"]
    ):
        raise ValueError(f"Tokenizer identity mismatch: {manifest_path}")
    stage = manifest["stage"]
    overlap_registry = manifest.get("overlap_registry", {})
    expected = {
        "language": expected_language,
        "cycle_index": expected_cycle,
        "task_index": expected_task_index,
    }
    stage_mismatches = {
        key: {"expected": value, "actual": stage.get(key)}
        for key, value in expected.items()
        if value is not None and stage.get(key) != value
    }
    if stage_mismatches:
        raise ValueError(
            f"Packed stage identity mismatch for {manifest_path}: {stage_mismatches}"
        )
    sequence_length = int(manifest["reader"]["sequence_length"])
    complete_sequences = int(manifest["token_count"]) // sequence_length
    if budget is not None:
        if sequence_length != budget.sequence_length:
            raise ValueError(
                f"Packed sequence length mismatch: {manifest_path}"
            )
        if complete_sequences < budget.effective_complete_sequences:
            raise ValueError(
                f"Packed manifest has {complete_sequences} complete sequences; "
                f"requires {budget.effective_complete_sequences}: {manifest_path}"
            )
    return {
        "manifest_path": str(manifest_path),
        "manifest_file_sha256": sha256_file(manifest_path),
        "manifest_content_sha256": manifest["manifest_content_sha256"],
        "ordered_data_sha256": manifest["ordered_data_sha256"],
        "tokenizer_manifest_path": "",
        "tokenizer_manifest_file_sha256": tokenizer_manifest_file_sha256,
        "tokenizer_manifest_content_sha256": tokenizer_manifest[
            "manifest_content_sha256"
        ],
        "tokenizer_repo_id": tokenizer_manifest["repo_id"],
        "tokenizer_revision": tokenizer_manifest["revision"],
        "stage_id": stage["stage_id"],
        "purpose": stage["purpose"],
        "language": stage["language"],
        "cycle_index": stage["cycle_index"],
        "task_index": stage["task_index"],
        "preparation_lane_namespace": overlap_registry.get(
            "preparation_lane_namespace"
        ),
        "packed_token_count": int(manifest["token_count"]),
        "packed_target_token_count": int(manifest["target_token_count"]),
        "packed_complete_sequence_count": complete_sequences,
        "requested_input_tokens": (
            None if budget is None else budget.requested_input_tokens
        ),
        "effective_complete_sequences": (
            complete_sequences
            if budget is None
            else budget.effective_complete_sequences
        ),
        "effective_input_tokens": (
            complete_sequences * sequence_length
            if budget is None
            else budget.effective_input_tokens
        ),
        "effective_valid_targets": (
            complete_sequences * (sequence_length - 1)
            if budget is None
            else budget.effective_valid_targets
        ),
        "sequence_length": sequence_length,
    }


def resolve_data_contract(
    config: LauncherConfig,
    *,
    full_checksum_validation: bool = True,
) -> dict[str, Any]:
    task_budget = resolve_token_budget(
        config.experiment.tokens_per_task,
        config.experiment.sequence_length,
        policy=config.experiment.token_budget_policy,
    )
    probe_budget = resolve_token_budget(
        config.probe.training_tokens,
        config.experiment.sequence_length,
        policy=config.experiment.token_budget_policy,
    )
    if config.data.mode == "synthetic":
        matrix: list[dict[str, dict[str, Any]]] = []
        for cycle in range(config.experiment.cycles):
            cycle_items: dict[str, dict[str, Any]] = {}
            for language_index, language in enumerate(PUBLIC_LANGUAGE_ORDER):
                task_index = cycle * len(PUBLIC_LANGUAGE_ORDER) + language_index
                descriptor = {
                    "kind": "synthetic",
                    "cycle_index": cycle,
                    "task_index": task_index,
                    "language": language,
                    **task_budget.to_dict(),
                }
                descriptor["ordered_data_sha256"] = canonical_sha256(descriptor)
                cycle_items[language] = descriptor
            matrix.append(cycle_items)
        return {
            "mode": "synthetic",
            "cycle_manifest_policy": CYCLE_MANIFEST_POLICY,
            "task_token_budget": task_budget.to_dict(),
            "probe_token_budget": probe_budget.to_dict(),
            "data_manifests": matrix,
            "probe_training_manifest": None,
            "probe_validation_manifest": None,
            "tokenizer": None,
        }

    tokenizer_ref, tokenizer_manifest, tokenizer_file_sha = _tokenizer_reference(
        config
    )
    matrix = []
    paths: set[str] = set()
    ordered_hashes: set[str] = set()
    for cycle in range(config.experiment.cycles):
        cycle_items = {}
        for language_index, language in enumerate(PUBLIC_LANGUAGE_ORDER):
            task_index = cycle * len(PUBLIC_LANGUAGE_ORDER) + language_index
            manifest_path = render_manifest_path(
                config,
                cycle=cycle,
                language=language,
                task_index=task_index,
                budget=task_budget,
            )
            identity = _manifest_identity(
                manifest_path,
                expected_language=language,
                expected_cycle=cycle,
                expected_task_index=task_index,
                budget=task_budget,
                tokenizer_manifest=tokenizer_manifest,
                tokenizer_manifest_file_sha256=tokenizer_file_sha,
                full_checksum_validation=full_checksum_validation,
            )
            identity["tokenizer_manifest_path"] = config.data.tokenizer_manifest
            if identity["manifest_path"] in paths:
                raise ValueError(
                    "fresh_disjoint_v1 prohibits manifest replay across appearances"
                )
            if identity["ordered_data_sha256"] in ordered_hashes:
                raise ValueError(
                    "fresh_disjoint_v1 requires distinct ordered-data identities"
                )
            paths.add(identity["manifest_path"])
            ordered_hashes.add(identity["ordered_data_sha256"])
            cycle_items[language] = identity
        matrix.append(cycle_items)

    probe_training = _manifest_identity(
        Path(config.data.probe_training_manifest or "").resolve(),
        expected_language="vi",
        expected_cycle=None,
        expected_task_index=None,
        budget=probe_budget,
        tokenizer_manifest=tokenizer_manifest,
        tokenizer_manifest_file_sha256=tokenizer_file_sha,
        full_checksum_validation=full_checksum_validation,
    )
    probe_training["tokenizer_manifest_path"] = config.data.tokenizer_manifest
    validation_path = Path(config.data.probe_validation_manifest or "").resolve()
    probe_validation = _manifest_identity(
        validation_path,
        expected_language="vi",
        expected_cycle=None,
        expected_task_index=None,
        budget=None,
        tokenizer_manifest=tokenizer_manifest,
        tokenizer_manifest_file_sha256=tokenizer_file_sha,
        full_checksum_validation=full_checksum_validation,
    )
    probe_validation["tokenizer_manifest_path"] = config.data.tokenizer_manifest
    if probe_validation["packed_complete_sequence_count"] < (
        config.probe.validation_sequences
    ):
        raise ValueError(
            "Vietnamese validation manifest has too few complete sequences"
        )
    identities = [
        identity
        for cycle_items in matrix
        for identity in cycle_items.values()
    ] + [probe_training, probe_validation]
    lane_paths = [
        Path(identity["manifest_path"])
        for identity in identities
        if identity.get("preparation_lane_namespace") is not None
    ]
    parallel_audit = None
    if lane_paths:
        records = _parallel_manifest_records(lane_paths)
        generated_root = Path(config.data.generated_root).resolve()
        global_registry = generated_root / "overlap.sqlite3"
        audit_path = generated_root / "parallel-preparation-audit.json"
        if not _parallel_audit_is_current(
            audit_path, records, global_registry
        ):
            raise ValueError(
                "Parallel language-lane manifests require a current, "
                "conflict-free global overlap merge audit"
            )
        parallel_audit = json.loads(audit_path.read_text(encoding="utf-8"))
    return {
        "mode": "packed",
        "cycle_manifest_policy": CYCLE_MANIFEST_POLICY,
        "task_token_budget": task_budget.to_dict(),
        "probe_token_budget": probe_budget.to_dict(),
        "data_manifests": matrix,
        "probe_training_manifest": probe_training,
        "probe_validation_manifest": probe_validation,
        "parallel_preparation_audit": parallel_audit,
        "tokenizer": {
            **asdict(tokenizer_ref),
            "manifest_file_sha256": tokenizer_file_sha,
            "manifest_content_sha256": tokenizer_manifest[
                "manifest_content_sha256"
            ],
        },
    }


def data_pipeline_from_identity(
    config: LauncherConfig,
    identity: dict[str, Any],
    *,
    purpose: str,
    global_sequences_per_batch: int,
) -> DataPipelineConfig:
    if config.data.mode != "packed":
        raise ValueError("Packed data pipeline requested for synthetic mode")
    manifest_path = Path(identity["manifest_path"]).resolve()
    _, manifest = load_packed_manifest(manifest_path)
    tokenizer_reference, _, _ = _tokenizer_reference(config)
    generated_root = Path(config.data.generated_root).resolve()
    expected_path = generated_root / "stages" / manifest["stage"]["stage_id"] / "manifest.json"
    if manifest_path != expected_path:
        raise ValueError(
            "Manifest path does not match generated_root/stages/stage_id"
        )
    packing = manifest["packing"]
    pipeline = DataPipelineConfig(
        schema_version=DATA_SCHEMA_VERSION,
        name=f"release-packed-{manifest['stage']['stage_id']}",
        mode="packed_shards",
        run_kind=(
            "production"
            if config.experiment.run_kind == "production"
            else "smoke"
        ),
        dataset=DatasetReference(
            repo_id=manifest["dataset"]["repo_id"],
            revision=manifest["dataset"]["revision"],
            split=manifest["dataset"]["split"],
            text_field=manifest["dataset"]["text_field"],
            id_field=manifest["dataset"]["id_field"],
            source_id_policy=manifest["dataset"]["source_id_policy"],
            missing_id_policy=manifest["dataset"]["missing_id_policy"],
            language_configs=dict(LANGUAGE_CONFIGS),
        ),
        tokenizer=tokenizer_reference,
        selection=SelectionConfig(**manifest["selection"]),
        storage=StorageConfig(
            hf_cache_root=config.data.dataset_cache_root,
            generated_root=config.data.generated_root,
            max_cache_bytes=config.data.max_cache_bytes,
            max_generated_bytes=config.data.max_generated_bytes,
            max_temporary_bytes=config.data.max_temporary_bytes,
            auto_clean_cache=False,
        ),
        packing=PackingConfig(
            format_version=int(manifest["format_version"]),
            dtype=packing["dtype"],
            max_shard_tokens=packing["max_shard_tokens"],
            max_shard_bytes=packing["max_shard_bytes"],
            write_boundaries=packing["write_boundaries"],
            add_bos=packing["add_bos"],
            add_chat_template=packing["add_chat_template"],
            add_special_tokens=packing["add_special_tokens"],
            eos_between_documents=packing["eos_between_documents"],
            eos_after_each_document=packing[
                "eos_after_each_accepted_document"
            ],
            truncate_final_document_to_budget=packing[
                "truncate_final_document_to_budget"
            ],
            mask_document_boundary_loss=packing[
                "mask_document_boundary_loss"
            ],
            checksum_algorithm=packing["checksum_algorithm"],
        ),
        stage=StageConfig(
            stage_id=manifest["stage"]["stage_id"],
            purpose=purpose,
            language=manifest["stage"]["language"],
            task_index=manifest["stage"]["task_index"],
            cycle_index=manifest["stage"]["cycle_index"],
            resume=True,
            delete_temporary_cache_after_success=False,
            require_exact_output_tokens=manifest["stage"][
                "require_exact_output_tokens"
            ],
            checkpoint_every_candidates=manifest["stage"][
                "checkpoint_every_candidates"
            ],
        ),
        reader=ReaderConfig(
            sequence_length=manifest["reader"]["sequence_length"],
            global_sequences_per_batch=global_sequences_per_batch,
            drop_incomplete_sequence=manifest["reader"][
                "drop_incomplete_sequence"
            ],
        ),
        packed_manifest_identity=PackedManifestIdentity(
            status="frozen",
            manifest_file_sha256=identity["manifest_file_sha256"],
            manifest_content_sha256=identity["manifest_content_sha256"],
            ordered_data_sha256=identity["ordered_data_sha256"],
            expected_token_count=identity["packed_token_count"],
            expected_target_token_count=identity[
                "packed_target_token_count"
            ],
            expected_complete_sequence_count=identity[
                "packed_complete_sequence_count"
            ],
        ),
    )
    pipeline.validate()
    return pipeline


def materialization_config(
    config: LauncherConfig,
    *,
    cycle: int,
    language: str,
    task_index: int,
    budget: TokenBudget,
    manifest_path_override: Path | None = None,
    purpose: str = "continual_train",
) -> tuple[DataPipelineConfig, Path]:
    tokenizer_reference, _, _ = _tokenizer_reference(config)
    manifest_path = (
        render_manifest_path(
            config,
            cycle=cycle,
            language=language,
            task_index=task_index,
            budget=budget,
        )
        if manifest_path_override is None
        else manifest_path_override.resolve()
    )
    stage_id = manifest_path.parent.name
    expected = Path(config.data.generated_root) / "stages" / stage_id / "manifest.json"
    if manifest_path != expected.resolve():
        raise ValueError(
            "Materialization manifest template must resolve to generated_root/stages/<stage_id>/manifest.json"
        )
    pipeline = DataPipelineConfig(
        schema_version=DATA_SCHEMA_VERSION,
        name=f"release-materialize-{stage_id}",
        mode="culturax_stage_materialize",
        run_kind=(
            "production"
            if config.experiment.run_kind == "production"
            else "smoke"
        ),
        dataset=DatasetReference(
            repo_id=CULTURAX_REPO_ID,
            revision=CULTURAX_REVISION,
            split="train",
            text_field="text",
            id_field="url",
            source_id_policy="sha256_canonical_json",
            missing_id_policy="content_sha256",
            language_configs=dict(LANGUAGE_CONFIGS),
        ),
        tokenizer=tokenizer_reference,
        selection=SelectionConfig(
            max_input_documents=config.data.max_input_documents,
            max_output_tokens=budget.effective_input_tokens,
            max_runtime_seconds=config.data.max_runtime_seconds,
            document_order_seed=(
                config.data.document_order_seed + task_index
            ),
            split_seed=config.data.split_seed,
            validation_permyriad=config.data.validation_permyriad,
            shuffle_buffer_documents=config.data.shuffle_buffer_documents,
            order_algorithm="bounded_buffer_python_v1",
            split_algorithm="sha256_permyriad_v1",
            document_hash_algorithm="sha256_utf8",
            token_hash_algorithm="sha256_uint32_le",
        ),
        storage=StorageConfig(
            hf_cache_root=config.data.dataset_cache_root,
            generated_root=config.data.generated_root,
            max_cache_bytes=config.data.max_cache_bytes,
            max_generated_bytes=config.data.max_generated_bytes,
            max_temporary_bytes=config.data.max_temporary_bytes,
            auto_clean_cache=False,
        ),
        packing=PackingConfig(
            format_version=1,
            dtype="uint32_le",
            max_shard_tokens=config.data.max_shard_tokens,
            max_shard_bytes=config.data.max_shard_tokens * 4,
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
            stage_id=stage_id,
            purpose=purpose,
            language=language,
            task_index=task_index,
            cycle_index=cycle,
            resume=True,
            delete_temporary_cache_after_success=False,
            require_exact_output_tokens=True,
            checkpoint_every_candidates=1000,
        ),
        reader=ReaderConfig(
            sequence_length=config.experiment.sequence_length,
            global_sequences_per_batch=(
                config.training.global_batch_sequences
            ),
            drop_incomplete_sequence=True,
        ),
    )
    pipeline.require_access_ready()
    return pipeline, manifest_path


def prepare_or_validate_data(
    config: LauncherConfig,
    *,
    execute_missing: bool | None = None,
    full_checksum_validation: bool = True,
    parallel_languages: int = 1,
) -> dict[str, Any]:
    if parallel_languages <= 0 or parallel_languages > len(LANGUAGE_CONFIGS):
        raise ValueError("parallel_languages must be in [1, 9]")
    if config.data.mode == "synthetic":
        return resolve_data_contract(
            config, full_checksum_validation=full_checksum_validation
        )
    execute = (
        config.data.prepare_if_missing
        if execute_missing is None
        else execute_missing
    )
    budget = resolve_token_budget(
        config.experiment.tokens_per_task,
        config.experiment.sequence_length,
        policy=config.experiment.token_budget_policy,
    )
    missing: list[tuple[DataPipelineConfig, Path, Path]] = []
    expected_manifest_paths: list[Path] = []
    config_root = (
        Path(config.data.generated_root)
        / "preparation-configs"
        / config.experiment.name
    )
    for cycle in range(config.experiment.cycles):
        for language_index, language in enumerate(PUBLIC_LANGUAGE_ORDER):
            task_index = cycle * len(PUBLIC_LANGUAGE_ORDER) + language_index
            pipeline, manifest_path = materialization_config(
                config,
                cycle=cycle,
                language=language,
                task_index=task_index,
                budget=budget,
            )
            expected_manifest_paths.append(manifest_path)
            if not manifest_path.is_file():
                config_path = config_root / f"cycle-{cycle:04d}-{language}.yaml"
                missing.append((pipeline, manifest_path, config_path))
    probe_budget = resolve_token_budget(
        config.probe.training_tokens,
        config.experiment.sequence_length,
        policy=config.experiment.token_budget_policy,
    )
    validation_budget = resolve_token_budget(
        config.probe.validation_sequences * config.experiment.sequence_length,
        config.experiment.sequence_length,
        policy=config.experiment.token_budget_policy,
    )
    probe_specs = (
        (
            Path(config.data.probe_training_manifest or "").resolve(),
            probe_budget,
            "vietnamese_train",
            900_000,
            "vietnamese-probe-training.yaml",
        ),
        (
            Path(config.data.probe_validation_manifest or "").resolve(),
            validation_budget,
            "vietnamese_validation",
            900_001,
            "vietnamese-probe-validation.yaml",
        ),
    )
    for manifest_path, stage_budget, purpose, seed_index, filename in probe_specs:
        expected_manifest_paths.append(manifest_path)
        if manifest_path.is_file():
            continue
        pipeline, _ = materialization_config(
            config,
            cycle=0,
            language="vi",
            task_index=seed_index,
            budget=stage_budget,
            manifest_path_override=manifest_path,
            purpose=purpose,
        )
        missing.append((pipeline, manifest_path, config_root / filename))
    if missing and not execute:
        paths = [str(item[1]) for item in missing]
        raise FileNotFoundError(
            "Required packed stages are missing and prepare_if_missing=false: "
            + ", ".join(paths)
        )
    for pipeline, _, config_path in missing:
        save_data_pipeline_config(pipeline, config_path)
    grouped: dict[str, list[tuple[DataPipelineConfig, Path, Path]]] = {}
    for item in missing:
        grouped.setdefault(item[0].stage.language, []).append(item)

    def run_lane(
        language: str,
        items: list[tuple[DataPipelineConfig, Path, Path]],
    ) -> int:
        environment = dict(os.environ)
        environment.setdefault("HF_HOME", config.data.dataset_cache_root)
        environment["TOKENIZERS_PARALLELISM"] = "true"
        if parallel_languages > 1:
            environment["LM_CL_OVERLAP_REGISTRY_NAMESPACE"] = (
                _lane_namespace(language)
            )
            environment["RAYON_NUM_THREADS"] = str(
                max(1, (os.cpu_count() or 1) // parallel_languages)
            )
            environment["LM_CL_TOKENIZER_BATCH_DOCUMENTS"] = "2048"
            environment["LM_CL_STREAM_RESHARD_ROW_GROUPS"] = "false"
            environment["LM_CL_STREAM_PREFETCH_SHARDS"] = "4"
            environment["LM_CL_STREAM_PREFETCH_ROWS_PER_SHARD"] = "256"
            environment[
                "LM_CL_MATERIALIZATION_CHECKPOINT_CANDIDATES"
            ] = "100000"
        completed_count = 0
        for _, _, config_path in items:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lm_cl.cli.materialize_stage",
                    str(config_path),
                    "--execute",
                ],
                env=environment,
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    "Materialization failed with status "
                    f"{completed.returncode}: {config_path}"
                )
            completed_count += 1
        return completed_count

    if parallel_languages == 1:
        for language, items in grouped.items():
            run_lane(language, items)
    elif grouped:
        with ThreadPoolExecutor(
            max_workers=min(parallel_languages, len(grouped)),
            thread_name_prefix="language-lane",
        ) as executor:
            futures = {
                executor.submit(run_lane, language, items): language
                for language, items in grouped.items()
            }
            for future in as_completed(futures):
                language = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    for pending in futures:
                        pending.cancel()
                    raise RuntimeError(
                        f"Parallel materialization lane failed: {language}"
                    ) from exc
    parallel_audit = None
    if parallel_languages > 1:
        parallel_audit = _merge_parallel_overlap_registries(
            config, expected_manifest_paths
        )
    result = resolve_data_contract(
        config, full_checksum_validation=full_checksum_validation
    )
    result["materialized_stage_count"] = len(missing)
    result["validated_stage_count"] = (
        config.experiment.cycles * len(PUBLIC_LANGUAGE_ORDER)
    )
    result["preparation_config_root"] = str(config_root)
    result["parallel_languages"] = parallel_languages
    result["parallel_preparation_audit"] = parallel_audit
    return result
