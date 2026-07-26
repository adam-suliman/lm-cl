from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DATA_SCHEMA_VERSION = 1
PACKED_FORMAT_VERSION = 1
MAX_INLINE_DOCUMENT_RECORDS = 10_000
MAX_IN_MEMORY_STREAM_TOKENS = 16_777_216
MAX_IN_MEMORY_STREAM_DOCUMENTS = 100_000
IMMUTABLE_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
REQUIRED_LANGUAGE_KEYS = ("en", "zh_written", "fr", "ja", "es", "de", "pt", "ru", "vi")


@dataclass(frozen=True)
class DatasetReference:
    repo_id: str
    revision: str | None
    split: str
    text_field: str | None
    id_field: str | None
    source_id_policy: str
    missing_id_policy: str
    language_configs: dict[str, str | None]

    def validate(self) -> None:
        if not self.repo_id:
            raise ValueError("dataset.repo_id must not be empty")
        if not self.split:
            raise ValueError("dataset.split must not be empty")
        if set(self.language_configs) != set(REQUIRED_LANGUAGE_KEYS):
            raise ValueError(
                "dataset.language_configs must contain exactly "
                f"{list(REQUIRED_LANGUAGE_KEYS)}"
            )
        for language, configuration in self.language_configs.items():
            if configuration is not None and not configuration:
                raise ValueError(
                    f"dataset.language_configs.{language} must be null or non-empty"
                )
        for name, value in (
            ("dataset.text_field", self.text_field),
            ("dataset.id_field", self.id_field),
        ):
            if value is not None and not value:
                raise ValueError(f"{name} must be null or non-empty")
        if self.id_field is not None and self.id_field == self.text_field:
            raise ValueError(
                "dataset.id_field cannot equal dataset.text_field because "
                "raw text must not be retained as an identifier"
            )
        if self.missing_id_policy != "content_sha256":
            raise ValueError(
                "dataset.missing_id_policy must be content_sha256"
            )
        if self.source_id_policy != "sha256_canonical_json":
            raise ValueError(
                "dataset.source_id_policy must be sha256_canonical_json"
            )
        if self.revision is not None and not IMMUTABLE_REVISION_RE.fullmatch(
            self.revision
        ):
            raise ValueError("dataset.revision must be null or a 40-hex commit")


@dataclass(frozen=True)
class TokenizerReference:
    repo_id: str | None
    revision: str | None
    manifest_path: str | None
    base_vocab_size: int | None
    effective_vocab_size: int | None
    maximum_emitted_token_id: int | None
    model_embedding_vocab_size: int
    expected_eos_token_id: int | None
    expected_pad_token_id: int | None = None

    def validate(self) -> None:
        if self.repo_id is not None and not self.repo_id:
            raise ValueError("tokenizer.repo_id must be null or non-empty")
        if self.revision is not None and not IMMUTABLE_REVISION_RE.fullmatch(
            self.revision
        ):
            raise ValueError("tokenizer.revision must be null or a 40-hex commit")
        if self.manifest_path is not None and not self.manifest_path:
            raise ValueError("tokenizer.manifest_path must be null or non-empty")
        if self.manifest_path is not None and not Path(
            self.manifest_path
        ).expanduser().is_absolute():
            raise ValueError("tokenizer.manifest_path must be absolute")
        for name, value in (
            ("base_vocab_size", self.base_vocab_size),
            ("effective_vocab_size", self.effective_vocab_size),
        ):
            if value is not None and value <= 0:
                raise ValueError(f"tokenizer.{name} must be null or positive")
        if (
            self.base_vocab_size is not None
            and self.effective_vocab_size is not None
            and self.base_vocab_size > self.effective_vocab_size
        ):
            raise ValueError(
                "tokenizer.base_vocab_size cannot exceed "
                "tokenizer.effective_vocab_size"
            )
        if self.maximum_emitted_token_id is not None:
            if self.maximum_emitted_token_id < 0:
                raise ValueError(
                    "tokenizer.maximum_emitted_token_id must be null or "
                    "non-negative"
                )
        if self.model_embedding_vocab_size <= 0:
            raise ValueError(
                "tokenizer.model_embedding_vocab_size must be positive"
            )
        if (
            self.maximum_emitted_token_id is not None
            and self.maximum_emitted_token_id
            >= self.model_embedding_vocab_size
        ):
            raise ValueError(
                "tokenizer.maximum_emitted_token_id must be smaller than "
                "tokenizer.model_embedding_vocab_size"
            )
        if (
            self.expected_eos_token_id is not None
            and self.expected_eos_token_id < 0
        ):
            raise ValueError(
                "tokenizer.expected_eos_token_id must be non-negative"
            )
        if (
            self.expected_pad_token_id is not None
            and self.expected_pad_token_id < 0
        ):
            raise ValueError(
                "tokenizer.expected_pad_token_id must be non-negative"
            )


@dataclass(frozen=True)
class SelectionConfig:
    max_input_documents: int
    max_output_tokens: int
    max_runtime_seconds: int
    document_order_seed: int
    split_seed: int
    validation_permyriad: int
    shuffle_buffer_documents: int
    order_algorithm: str
    split_algorithm: str
    document_hash_algorithm: str
    token_hash_algorithm: str

    def validate(self, *, bounded_mode: bool) -> None:
        if bounded_mode and (
            self.max_input_documents <= 0 or self.max_output_tokens <= 0
        ):
            raise ValueError(
                "bounded data modes require explicit positive "
                "max_input_documents and max_output_tokens"
            )
        if self.max_runtime_seconds <= 0:
            raise ValueError("selection.max_runtime_seconds must be positive")
        if self.document_order_seed < 0 or self.split_seed < 0:
            raise ValueError("selection seeds must be non-negative")
        if not 0 < self.validation_permyriad < 10_000:
            raise ValueError("selection.validation_permyriad must be in (0, 10000)")
        if self.shuffle_buffer_documents <= 0:
            raise ValueError("selection.shuffle_buffer_documents must be positive")
        expected = {
            "order_algorithm": "bounded_buffer_python_v1",
            "split_algorithm": "sha256_permyriad_v1",
            "document_hash_algorithm": "sha256_utf8",
            "token_hash_algorithm": "sha256_uint32_le",
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(f"selection.{name} must be {value}")


@dataclass(frozen=True)
class StorageConfig:
    hf_cache_root: str
    generated_root: str
    max_cache_bytes: int
    max_generated_bytes: int
    max_temporary_bytes: int
    auto_clean_cache: bool

    def validate(self) -> None:
        resolved: dict[str, Path] = {}
        for name, value in (
            ("hf_cache_root", self.hf_cache_root),
            ("generated_root", self.generated_root),
        ):
            path = Path(value).expanduser().resolve()
            resolved[name] = path
            if not path.is_absolute():
                raise ValueError(f"storage.{name} must be an absolute path")
            if path == Path("/"):
                raise ValueError(f"storage.{name} cannot be the filesystem root")
        cache = resolved["hf_cache_root"]
        generated = resolved["generated_root"]
        if (
            cache == generated
            or cache in generated.parents
            or generated in cache.parents
        ):
            raise ValueError(
                "storage cache and generated roots must be distinct and "
                "non-nested"
            )
        for name, value in (
            ("max_cache_bytes", self.max_cache_bytes),
            ("max_generated_bytes", self.max_generated_bytes),
            ("max_temporary_bytes", self.max_temporary_bytes),
        ):
            if value <= 0:
                raise ValueError(f"storage.{name} must be positive")


@dataclass(frozen=True)
class PackingConfig:
    format_version: int
    dtype: str
    max_shard_tokens: int
    max_shard_bytes: int
    write_boundaries: bool
    add_bos: bool
    add_chat_template: bool
    add_special_tokens: bool
    eos_between_documents: bool
    eos_after_each_document: bool
    truncate_final_document_to_budget: bool
    mask_document_boundary_loss: bool
    checksum_algorithm: str

    def validate(self) -> None:
        if self.format_version != PACKED_FORMAT_VERSION:
            raise ValueError(
                f"packing.format_version must be {PACKED_FORMAT_VERSION}"
            )
        if self.dtype != "uint32_le":
            raise ValueError("packing.dtype must be uint32_le")
        if self.max_shard_tokens <= 0:
            raise ValueError("packing.max_shard_tokens must be positive")
        if self.max_shard_bytes != self.max_shard_tokens * 4:
            raise ValueError(
                "packing.max_shard_bytes must equal max_shard_tokens * 4 "
                "for uint32 storage"
            )
        if self.add_bos or self.add_chat_template or self.add_special_tokens:
            raise ValueError("Phase 3 faithful packing prohibits BOS/chat templates")
        if not self.eos_between_documents:
            raise ValueError("Phase 3 faithful packing requires EOS between documents")
        if not self.eos_after_each_document:
            raise ValueError(
                "Phase 3 requires eos_after_each_document=true"
            )
        if not self.truncate_final_document_to_budget:
            raise ValueError(
                "Phase 3 requires truncate_final_document_to_budget=true"
            )
        if self.mask_document_boundary_loss:
            raise ValueError(
                "Phase 3 faithful packing does not mask document-boundary loss"
            )
        if self.checksum_algorithm != "sha256":
            raise ValueError("packing.checksum_algorithm must be sha256")


@dataclass(frozen=True)
class StageConfig:
    stage_id: str
    purpose: str
    language: str
    task_index: int
    cycle_index: int
    resume: bool
    delete_temporary_cache_after_success: bool
    require_exact_output_tokens: bool
    checkpoint_every_candidates: int

    def validate(self) -> None:
        valid_purposes = {
            "inspection",
            "continual_train",
            "language_validation",
            "vietnamese_train",
            "vietnamese_validation",
        }
        if not self.stage_id:
            raise ValueError("stage.stage_id must not be empty")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", self.stage_id):
            raise ValueError(
                "stage.stage_id must be a safe single path component"
            )
        if self.purpose not in valid_purposes:
            raise ValueError(f"Unknown stage purpose: {self.purpose}")
        if self.language not in REQUIRED_LANGUAGE_KEYS:
            raise ValueError(f"Unknown language key: {self.language}")
        if self.task_index < 0 or self.cycle_index < 0:
            raise ValueError("stage task/cycle indices must be non-negative")
        if self.checkpoint_every_candidates <= 0:
            raise ValueError(
                "stage.checkpoint_every_candidates must be positive"
            )
        if self.language == "vi" and self.purpose == "continual_train":
            raise ValueError("Vietnamese is prohibited from continual-training shards")
        if self.purpose.startswith("vietnamese_") and self.language != "vi":
            raise ValueError("Vietnamese purposes require stage.language=vi")


@dataclass(frozen=True)
class ReaderConfig:
    sequence_length: int
    global_sequences_per_batch: int
    drop_incomplete_sequence: bool

    def validate(self) -> None:
        if self.sequence_length <= 1:
            raise ValueError("reader.sequence_length must be greater than one")
        if self.global_sequences_per_batch <= 0:
            raise ValueError(
                "reader.global_sequences_per_batch must be positive"
            )
        if not self.drop_incomplete_sequence:
            raise ValueError(
                "Phase 3 supports only drop_incomplete_sequence=true"
            )


@dataclass(frozen=True)
class PackedManifestIdentity:
    status: str
    manifest_file_sha256: str | None
    manifest_content_sha256: str | None
    ordered_data_sha256: str | None
    expected_token_count: int
    expected_target_token_count: int
    expected_complete_sequence_count: int

    def validate(self, *, sequence_length: int) -> None:
        if self.status not in {"pending", "frozen"}:
            raise ValueError(
                "packed_manifest_identity.status must be pending or frozen"
            )
        if self.expected_token_count <= 0:
            raise ValueError("Expected packed token count must be positive")
        if self.expected_target_token_count < 0:
            raise ValueError("Expected packed target count must be non-negative")
        if self.expected_complete_sequence_count <= 0:
            raise ValueError("Expected complete-sequence count must be positive")
        if (
            self.expected_complete_sequence_count
            != self.expected_token_count // sequence_length
        ):
            raise ValueError("Expected packed sequence count is inconsistent")
        if (
            self.expected_target_token_count
            != self.expected_complete_sequence_count * (sequence_length - 1)
        ):
            raise ValueError("Expected packed target count is inconsistent")
        hashes = (
            self.manifest_file_sha256,
            self.manifest_content_sha256,
            self.ordered_data_sha256,
        )
        if self.status == "pending":
            if any(value is not None for value in hashes):
                raise ValueError("Pending packed identity must not contain hashes")
            return
        if any(
            not isinstance(value, str)
            or not re.fullmatch(r"[0-9a-f]{64}", value)
            for value in hashes
        ):
            raise ValueError("Frozen packed identity requires three SHA-256 values")


@dataclass(frozen=True)
class DataPipelineConfig:
    schema_version: int
    name: str
    mode: str
    run_kind: str
    dataset: DatasetReference
    tokenizer: TokenizerReference
    selection: SelectionConfig
    storage: StorageConfig
    packing: PackingConfig
    stage: StageConfig
    reader: ReaderConfig
    packed_manifest_identity: PackedManifestIdentity | None = None

    def validate(self) -> None:
        if self.schema_version != DATA_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {DATA_SCHEMA_VERSION}")
        if not self.name:
            raise ValueError("name must not be empty")
        valid_modes = {
            "culturax_stream",
            "culturax_stage_materialize",
            "packed_shards",
        }
        if self.mode not in valid_modes:
            raise ValueError(f"Unknown data mode: {self.mode}")
        if self.run_kind not in {"inspection", "smoke", "production"}:
            raise ValueError("run_kind must be inspection, smoke, or production")
        self.dataset.validate()
        self.tokenizer.validate()
        self.selection.validate(
            bounded_mode=self.mode in {
                "culturax_stream",
                "culturax_stage_materialize",
            }
        )
        self.storage.validate()
        self.packing.validate()
        self.stage.validate()
        self.reader.validate()
        if self.packed_manifest_identity is not None:
            if self.mode != "packed_shards":
                raise ValueError(
                    "packed_manifest_identity is valid only for packed_shards"
                )
            self.packed_manifest_identity.validate(
                sequence_length=self.reader.sequence_length
            )
        if (
            not self.packing.write_boundaries
            and self.selection.max_input_documents
            > MAX_INLINE_DOCUMENT_RECORDS
        ):
            raise ValueError(
                "Packing without a boundary sidecar is limited to "
                f"{MAX_INLINE_DOCUMENT_RECORDS} input documents"
            )
        if self.mode == "culturax_stream" and self.run_kind == "production":
            raise ValueError(
                "Production runs must use materialized packed shards, not a "
                "live CulturaX stream"
            )
        if (
            self.mode == "culturax_stream"
            and self.selection.max_output_tokens > MAX_IN_MEMORY_STREAM_TOKENS
        ):
            raise ValueError(
                "culturax_stream max_output_tokens exceeds the in-memory "
                f"safety limit {MAX_IN_MEMORY_STREAM_TOKENS}; materialize "
                "packed shards instead"
            )
        if (
            self.mode == "culturax_stream"
            and self.selection.max_input_documents
            > MAX_IN_MEMORY_STREAM_DOCUMENTS
        ):
            raise ValueError(
                "culturax_stream max_input_documents exceeds the in-memory "
                f"safety limit {MAX_IN_MEMORY_STREAM_DOCUMENTS}; materialize "
                "packed shards instead"
            )

    @property
    def language_config(self) -> str | None:
        return self.dataset.language_configs[self.stage.language]

    def require_access_ready(self) -> None:
        self.validate()
        missing: list[str] = []
        if self.dataset.revision is None:
            missing.append("dataset.revision")
        if self.dataset.text_field is None:
            missing.append("dataset.text_field")
        if self.language_config is None:
            missing.append(
                f"dataset.language_configs.{self.stage.language}"
            )
        if self.tokenizer.repo_id is None:
            missing.append("tokenizer.repo_id")
        if self.tokenizer.revision is None:
            missing.append("tokenizer.revision")
        if self.tokenizer.manifest_path is None:
            missing.append("tokenizer.manifest_path")
        if self.tokenizer.base_vocab_size is None:
            missing.append("tokenizer.base_vocab_size")
        if self.tokenizer.effective_vocab_size is None:
            missing.append("tokenizer.effective_vocab_size")
        if self.tokenizer.maximum_emitted_token_id is None:
            missing.append("tokenizer.maximum_emitted_token_id")
        if self.tokenizer.expected_eos_token_id is None:
            missing.append("tokenizer.expected_eos_token_id")
        if self.tokenizer.expected_pad_token_id is None:
            missing.append("tokenizer.expected_pad_token_id")
        if self.tokenizer.manifest_path is not None:
            manifest_path = Path(
                self.tokenizer.manifest_path
            ).expanduser().resolve()
            generated_root = Path(
                self.storage.generated_root
            ).expanduser().resolve()
            if not manifest_path.is_file():
                missing.append("existing tokenizer.manifest_path file")
            try:
                manifest_path.relative_to(generated_root)
            except ValueError:
                missing.append(
                    "tokenizer.manifest_path inside storage.generated_root"
                )
        if missing:
            raise ValueError(
                "Data access is not frozen; resolve: " + ", ".join(missing)
            )

    def require_packed_launch_ready(self) -> None:
        self.require_access_ready()
        if self.mode != "packed_shards":
            raise ValueError("Packed launch readiness requires packed_shards")
        identity = self.packed_manifest_identity
        if identity is not None and identity.status != "frozen":
            raise ValueError(
                "Packed manifest identity is pending; materialize and freeze it "
                "before launch"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
