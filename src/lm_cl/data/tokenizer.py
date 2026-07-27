from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from lm_cl.config.data_schema import IMMUTABLE_REVISION_RE, TokenizerReference
from lm_cl.data.storage import atomic_write_json


TOKENIZER_ALLOW_PATTERNS = (
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "config.json",
)
TOKENIZER_MANIFEST_SCHEMA_VERSION = 2


def _optional_module(name: str, package: str) -> Any:
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"Tokenizer access requires `{package}`. Install the project "
            "data extra with: python -m pip install 'lm-cl[data]'"
        ) from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _special_token_ids(tokenizer: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in (
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
        "unk_token_id",
        "sep_token_id",
        "cls_token_id",
        "mask_token_id",
    ):
        result[name] = getattr(tokenizer, name, None)
    result["additional_special_tokens_ids"] = list(
        getattr(tokenizer, "additional_special_tokens_ids", []) or []
    )
    return result


def tokenizer_vocabulary_contract(
    tokenizer: Any,
    *,
    model_embedding_vocab_size: int,
) -> dict[str, int]:
    """Describe and validate the tokenizer ID space against model storage.

    The complete vocabulary registry, including added tokens and special-token
    IDs, is the finite set from which the tokenizer can emit IDs. Runtime
    encoding paths are checked separately as a defense in depth.
    """
    if model_embedding_vocab_size <= 0:
        raise ValueError("Model embedding vocabulary size must be positive")
    vocabulary = tokenizer.get_vocab()
    if not isinstance(vocabulary, dict) or not vocabulary:
        raise ValueError(
            "Tokenizer vocabulary registry must be a non-empty mapping"
        )
    registered_ids: list[int] = []
    for token, token_id in vocabulary.items():
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise ValueError(
                f"Tokenizer ID for token {token!r} is not an integer"
            )
        registered_ids.append(token_id)
    special_ids = _special_token_ids(tokenizer)
    for name, value in special_ids.items():
        values = (
            value if name == "additional_special_tokens_ids" else [value]
        )
        for token_id in values:
            if token_id is None:
                continue
            if isinstance(token_id, bool) or not isinstance(token_id, int):
                raise ValueError(
                    f"Tokenizer special ID {name} is not an integer"
                )
            registered_ids.append(token_id)
    minimum_token_id = min(registered_ids)
    maximum_token_id = max(registered_ids)
    if minimum_token_id < 0:
        raise ValueError("Tokenizer can emit a negative token ID")
    if maximum_token_id >= model_embedding_vocab_size:
        raise ValueError(
            "Tokenizer ID space does not fit the configured model embedding "
            f"vocabulary: maximum emitted token ID {maximum_token_id} >= "
            f"{model_embedding_vocab_size}"
        )
    base_vocab_size = int(tokenizer.vocab_size)
    effective_vocab_size = len(tokenizer)
    if base_vocab_size <= 0 or effective_vocab_size <= 0:
        raise ValueError("Tokenizer vocabulary sizes must be positive")
    if base_vocab_size > effective_vocab_size:
        raise ValueError(
            "Tokenizer base vocabulary cannot exceed effective tokenizer length"
        )
    return {
        "base_vocab_size": base_vocab_size,
        "effective_vocab_size": effective_vocab_size,
        "maximum_emitted_token_id": maximum_token_id,
        "model_embedding_vocab_size": model_embedding_vocab_size,
        "registered_token_id_count": len(set(registered_ids)),
        "trailing_unused_embedding_rows": (
            model_embedding_vocab_size - maximum_token_id - 1
        ),
    }


def validate_tokenizer_reference(
    tokenizer: Any,
    reference: TokenizerReference,
) -> dict[str, int]:
    contract = tokenizer_vocabulary_contract(
        tokenizer,
        model_embedding_vocab_size=reference.model_embedding_vocab_size,
    )
    expected = {
        "base_vocab_size": reference.base_vocab_size,
        "effective_vocab_size": reference.effective_vocab_size,
        "maximum_emitted_token_id": reference.maximum_emitted_token_id,
        "model_embedding_vocab_size": reference.model_embedding_vocab_size,
    }
    for name, expected_value in expected.items():
        if expected_value is None:
            raise ValueError(f"tokenizer.{name} is unresolved")
        if contract[name] != expected_value:
            raise ValueError(
                f"Loaded tokenizer {name} differs from configured value: "
                f"{contract[name]} != {expected_value}"
            )
    if getattr(tokenizer, "eos_token_id", None) != reference.expected_eos_token_id:
        raise ValueError("Loaded tokenizer EOS ID differs from configured EOS ID")
    if getattr(tokenizer, "pad_token_id", None) != reference.expected_pad_token_id:
        raise ValueError("Loaded tokenizer pad ID differs from configured pad ID")
    return contract


def validate_tokenizer_manifest_reference(
    manifest: dict[str, Any],
    reference: TokenizerReference,
) -> None:
    if manifest["repo_id"] != reference.repo_id:
        raise ValueError("Tokenizer repo differs from configured repo")
    if manifest["revision"] != reference.revision:
        raise ValueError("Tokenizer revision differs from configured revision")
    expected = {
        "base_vocab_size": reference.base_vocab_size,
        "effective_vocab_size": reference.effective_vocab_size,
        "maximum_emitted_token_id": reference.maximum_emitted_token_id,
        "model_embedding_vocab_size": reference.model_embedding_vocab_size,
    }
    for name, expected_value in expected.items():
        if manifest[name] != expected_value:
            raise ValueError(
                f"Tokenizer manifest {name} differs from configured value"
            )
    eos_id = manifest["special_token_ids"]["eos_token_id"]
    if eos_id != reference.expected_eos_token_id:
        raise ValueError("Tokenizer EOS ID differs from configured EOS ID")
    pad_id = manifest["special_token_ids"]["pad_token_id"]
    if pad_id != reference.expected_pad_token_id:
        raise ValueError("Tokenizer pad ID differs from configured pad ID")


def inspect_tokenizer(
    *,
    repo_id: str,
    revision: str,
    cache_dir: Path,
    output_manifest: Path,
    model_embedding_vocab_size: int = 151_680,
) -> dict[str, Any]:
    """Download only tokenizer metadata/files and emit a hash-pinned manifest."""
    if not repo_id:
        raise ValueError("repo_id must not be empty")
    if not IMMUTABLE_REVISION_RE.fullmatch(revision):
        raise ValueError("revision must be an immutable 40-hex commit")
    output_manifest = output_manifest.expanduser().resolve()
    if output_manifest.exists():
        existing = load_tokenizer_manifest(output_manifest)
        if (
            existing["repo_id"] != repo_id
            or existing["revision"] != revision
            or existing["model_embedding_vocab_size"]
            != model_embedding_vocab_size
        ):
            raise ValueError(
                "Existing tokenizer manifest differs from requested identity"
            )
        return existing
    hub = _optional_module("huggingface_hub", "huggingface-hub")
    transformers = _optional_module("transformers", "transformers")
    try:
        snapshot = Path(
            hub.snapshot_download(
                repo_id=repo_id,
                revision=revision,
                cache_dir=str(cache_dir),
                allow_patterns=list(TOKENIZER_ALLOW_PATTERNS),
            )
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            snapshot,
            local_files_only=True,
            trust_remote_code=False,
        )
    except Exception as exc:
        raise RuntimeError(
            "Tokenizer inspection failed. Check the exact artifact, immutable "
            "revision, standard Hugging Face authentication/access, and cache "
            f"space. Original error: {type(exc).__name__}: {exc}"
        ) from exc

    contract = tokenizer_vocabulary_contract(
        tokenizer,
        model_embedding_vocab_size=model_embedding_vocab_size,
    )
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    frozen_snapshot = output_manifest.parent / "files"
    source_files = {
        str(path.relative_to(snapshot)): sha256_file(path)
        for path in sorted(snapshot.rglob("*"))
        if path.is_file()
    }
    if frozen_snapshot.exists():
        frozen_files = {
            str(path.relative_to(frozen_snapshot)): sha256_file(path)
            for path in sorted(frozen_snapshot.rglob("*"))
            if path.is_file()
        }
        if frozen_files != source_files:
            raise ValueError(
                "Incomplete frozen tokenizer files differ from the pinned "
                "download; refusing overwrite"
            )
    else:
        temporary_snapshot = Path(
            tempfile.mkdtemp(
                prefix=".tokenizer-files-",
                dir=output_manifest.parent,
            )
        )
        try:
            for source in sorted(snapshot.rglob("*")):
                if not source.is_file():
                    continue
                relative = source.relative_to(snapshot)
                destination = temporary_snapshot / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
            os.replace(temporary_snapshot, frozen_snapshot)
        except BaseException:
            if temporary_snapshot.exists():
                shutil.rmtree(temporary_snapshot)
            raise

    files = []
    for path in sorted(frozen_snapshot.rglob("*")):
        if path.is_file():
            files.append(
                {
                    "path": str(path.relative_to(frozen_snapshot)),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    added_vocab = tokenizer.get_added_vocab()
    manifest = {
        "schema_version": TOKENIZER_MANIFEST_SCHEMA_VERSION,
        "kind": "lm-cl-tokenizer-manifest",
        "repo_id": repo_id,
        "revision": revision,
        "snapshot_path": str(frozen_snapshot),
        "tokenizer_class": type(tokenizer).__name__,
        **contract,
        "added_vocabulary_size": len(added_vocab),
        "added_vocabulary": {
            token: int(token_id)
            for token, token_id in sorted(
                added_vocab.items(), key=lambda item: (item[1], item[0])
            )
        },
        "special_token_ids": _special_token_ids(tokenizer),
        "model_max_length": int(tokenizer.model_max_length),
        "files": files,
    }
    portable_content = {
        key: value for key, value in manifest.items() if key != "snapshot_path"
    }
    manifest["tokenizer_content_sha256"] = hashlib.sha256(
        json.dumps(
            portable_content, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    manifest["manifest_content_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    atomic_write_json(output_manifest, manifest)
    return manifest


def load_tokenizer_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "kind",
        "repo_id",
        "revision",
        "snapshot_path",
        "base_vocab_size",
        "effective_vocab_size",
        "maximum_emitted_token_id",
        "model_embedding_vocab_size",
        "registered_token_id_count",
        "trailing_unused_embedding_rows",
        "special_token_ids",
        "files",
        "tokenizer_content_sha256",
        "manifest_content_sha256",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise ValueError(f"Tokenizer manifest missing fields: {missing}")
    if manifest["schema_version"] != TOKENIZER_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            "Unknown tokenizer manifest schema version: "
            f"{manifest['schema_version']}"
        )
    for name in (
        "base_vocab_size",
        "effective_vocab_size",
        "model_embedding_vocab_size",
        "registered_token_id_count",
    ):
        if not isinstance(manifest[name], int) or manifest[name] <= 0:
            raise ValueError(f"Tokenizer manifest {name} must be positive")
    maximum_token_id = manifest["maximum_emitted_token_id"]
    model_vocab_size = manifest["model_embedding_vocab_size"]
    if (
        not isinstance(maximum_token_id, int)
        or maximum_token_id < 0
        or maximum_token_id >= model_vocab_size
    ):
        raise ValueError(
            "Tokenizer manifest maximum emitted token ID does not fit the "
            "model embedding vocabulary"
        )
    if manifest["trailing_unused_embedding_rows"] != (
        model_vocab_size - maximum_token_id - 1
    ):
        raise ValueError(
            "Tokenizer manifest trailing unused embedding-row count mismatch"
        )
    content_hash = manifest.pop("manifest_content_sha256")
    actual = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest["manifest_content_sha256"] = content_hash
    if actual != content_hash:
        raise ValueError("Tokenizer manifest content checksum mismatch")
    portable = {
        key: value
        for key, value in manifest.items()
        if key
        not in {
            "snapshot_path",
            "tokenizer_content_sha256",
            "manifest_content_sha256",
        }
    }
    portable_hash = hashlib.sha256(
        json.dumps(portable, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if portable_hash != manifest["tokenizer_content_sha256"]:
        raise ValueError("Tokenizer portable-content checksum mismatch")
    snapshot = Path(manifest["snapshot_path"]).expanduser().resolve()
    try:
        snapshot.relative_to(manifest_path.parent)
    except ValueError as exc:
        raise ValueError(
            "Frozen tokenizer snapshot must be beside its manifest"
        ) from exc
    if not snapshot.is_dir():
        raise ValueError("Frozen tokenizer snapshot directory is missing")
    if not manifest["files"]:
        raise ValueError("Tokenizer manifest has no frozen files")
    seen_paths: set[str] = set()
    for entry in manifest["files"]:
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Tokenizer manifest contains an unsafe file path")
        if entry["path"] in seen_paths:
            raise ValueError("Tokenizer manifest contains a duplicate file")
        seen_paths.add(entry["path"])
        file_path = (snapshot / relative).resolve()
        try:
            file_path.relative_to(snapshot)
        except ValueError as exc:
            raise ValueError("Tokenizer file escapes frozen snapshot") from exc
        if (
            not file_path.is_file()
            or file_path.stat().st_size != entry["size_bytes"]
            or sha256_file(file_path) != entry["sha256"]
        ):
            raise ValueError(
                f"Tokenizer file missing or checksum mismatch: {entry['path']}"
            )
    return manifest


def load_verified_tokenizer(
    reference: TokenizerReference,
) -> tuple[Any, dict[str, Any]]:
    if reference.manifest_path is None:
        raise ValueError("tokenizer.manifest_path is unresolved")
    manifest = load_tokenizer_manifest(reference.manifest_path)
    validate_tokenizer_manifest_reference(manifest, reference)
    transformers = _optional_module("transformers", "transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        manifest["snapshot_path"],
        local_files_only=True,
        trust_remote_code=False,
    )
    validate_tokenizer_reference(tokenizer, reference)
    return tokenizer, manifest
