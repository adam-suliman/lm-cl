from __future__ import annotations

import importlib
import time
from pathlib import Path
from typing import Any, Iterable

from lm_cl.config.data_schema import DataPipelineConfig
from lm_cl.data.iteration import close_iterable
from lm_cl.data.storage import (
    PeriodicDiskLimitGuard,
    enforce_disk_limit,
    ensure_owned_root,
)


DATA_INSTALL_COMMAND = (
    "python -m pip install 'lm-cl[data]'"
)


class MissingDataDependencyError(RuntimeError):
    pass


def _datasets_module() -> Any:
    try:
        return importlib.import_module("datasets")
    except ModuleNotFoundError as exc:
        raise MissingDataDependencyError(
            "CulturaX access requires the optional `datasets` package. "
            f"Install only after approval with: {DATA_INSTALL_COMMAND}"
        ) from exc


def stream_culturax_rows(config: DataPipelineConfig) -> Iterable[dict[str, Any]]:
    """Open a pinned bounded-compatible stream using standard HF authentication."""
    config.require_access_ready()
    cache_root = ensure_owned_root(
        config.storage.hf_cache_root, purpose="hf-cache"
    )
    generated_root = ensure_owned_root(
        config.storage.generated_root, purpose="generated-data"
    )
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

    def bounded_rows() -> Iterable[dict[str, Any]]:
        started = time.monotonic()
        cache_limit = PeriodicDiskLimitGuard(
            cache_root,
            config.storage.max_cache_bytes,
            label="Hugging Face cache",
        )
        datasets = _datasets_module()
        stream = None
        iterator = None
        try:
            try:
                stream = datasets.load_dataset(
                    config.dataset.repo_id,
                    config.language_config,
                    split=config.dataset.split,
                    streaming=True,
                    revision=config.dataset.revision,
                    cache_dir=str(cache_root),
                )
            except Exception as exc:
                raise RuntimeError(
                    "Unable to open pinned CulturaX stream. Check dataset access, "
                    "standard Hugging Face authentication, revision, "
                    "configuration, and streaming support. Original error: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            if (
                time.monotonic() - started
                > config.selection.max_runtime_seconds
            ):
                raise TimeoutError(
                    "Runtime cap exceeded while opening CulturaX stream"
                )
            iterator = iter(stream)
            for _ in range(config.selection.max_input_documents):
                if (
                    time.monotonic() - started
                    > config.selection.max_runtime_seconds
                ):
                    raise TimeoutError(
                        "Runtime cap exceeded while reading CulturaX stream"
                    )
                try:
                    row = next(iterator)
                except StopIteration:
                    return
                if (
                    time.monotonic() - started
                    > config.selection.max_runtime_seconds
                ):
                    raise TimeoutError(
                        "Runtime cap exceeded while reading CulturaX stream"
                    )
                cache_limit.check()
                yield row
        finally:
            cache_limit.check(force=True)
            close_iterable(stream, iterator=iterator)

    return bounded_rows()


def inspect_culturax(
    *,
    repo_id: str,
    revision: str,
    cache_dir: Path,
    split: str,
    probe_configs: dict[str, str] | None = None,
) -> dict[str, Any]:
    from lm_cl.config.data_schema import IMMUTABLE_REVISION_RE

    if not IMMUTABLE_REVISION_RE.fullmatch(revision):
        raise ValueError("revision must be an immutable 40-hex commit")
    if not split:
        raise ValueError("split must not be empty")
    datasets = _datasets_module()
    try:
        try:
            hub = importlib.import_module("huggingface_hub")
        except ModuleNotFoundError as exc:
            raise MissingDataDependencyError(
                "CulturaX inspection requires `huggingface-hub`. After "
                "approval install the project data extra with: "
                "python -m pip install 'lm-cl[data]'"
            ) from exc
        info = hub.HfApi().dataset_info(repo_id, revision=revision)
        config_names = datasets.get_dataset_config_names(
            repo_id,
            revision=revision,
            cache_dir=str(cache_dir),
        )
    except Exception as exc:
        raise RuntimeError(
            "CulturaX inspection failed. Check standard Hugging Face "
            "authentication/access and the pinned revision. Original error: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    normalized = {name.casefold(): name for name in config_names}
    conventional = {
        key: normalized.get(key)
        for key in ("en", "fr", "ja", "es", "de", "pt", "ru", "vi")
    }
    chinese_candidates = [
        name
        for name in config_names
        if name.casefold() in {"zh", "zh-hans", "zh_cn", "zho"}
        or "chinese" in name.casefold()
    ]
    probes: dict[str, Any] = {}
    for language, config_name in sorted((probe_configs or {}).items()):
        if config_name not in config_names:
            probes[language] = {
                "configuration": config_name,
                "status": "not_in_discovered_configurations",
            }
            continue
        try:
            stream = datasets.load_dataset(
                repo_id,
                config_name,
                split=split,
                streaming=True,
                revision=revision,
                cache_dir=str(cache_dir),
            )
            iterator = iter(stream)
            try:
                row = next(iterator)
            finally:
                close_iterable(stream, iterator=iterator)
            probes[language] = {
                "configuration": config_name,
                "status": "stream_opened",
                "row_fields": sorted(row) if isinstance(row, dict) else [],
                "row_type": type(row).__name__,
                "raw_values_recorded": False,
            }
        except StopIteration:
            probes[language] = {
                "configuration": config_name,
                "status": "stream_opened_but_empty",
            }
        except Exception as exc:
            probes[language] = {
                "configuration": config_name,
                "status": "stream_failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
    return {
        "schema_version": 1,
        "repo_id": repo_id,
        "revision": revision,
        "split": split,
        "gated": getattr(info, "gated", None),
        "private": getattr(info, "private", None),
        "config_names": sorted(config_names),
        "conventional_exact_matches": conventional,
        "written_chinese_candidates": sorted(chinese_candidates),
        "written_chinese_requires_explicit_selection": True,
        "streaming_probes": probes,
        "streaming_probe_performed": bool(probe_configs),
        "note": (
            "Configuration discovery does not guess written Chinese. Run a "
            "bounded stream probe for each explicitly selected mapping."
        ),
    }
