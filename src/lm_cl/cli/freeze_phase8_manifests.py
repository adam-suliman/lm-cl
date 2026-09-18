from __future__ import annotations

import argparse
import os
from dataclasses import replace
from pathlib import Path

import yaml

from lm_cl.cli._common import cli_entry, print_json
from lm_cl.config import (
    PackedManifestIdentity,
    load_data_pipeline_config,
)
from lm_cl.data.packed import PackedShardSource
from lm_cl.data.sources import open_token_batch_source
from lm_cl.data.tokenizer import sha256_file


def _frozen_view(path: Path):
    config = load_data_pipeline_config(path)
    identity = config.packed_manifest_identity
    if config.mode != "packed_shards" or identity is None:
        raise ValueError(f"Phase 8 view lacks packed identity: {path}")
    if identity.status != "pending":
        raise ValueError(f"Phase 8 view is not pending: {path}")
    stage_dir = (
        Path(config.storage.generated_root)
        / "stages"
        / config.stage.stage_id
    )
    source = PackedShardSource(
        stage_dir,
        drop_incomplete_sequence=config.reader.drop_incomplete_sequence,
    )
    manifest = source.manifest
    actual_sequences = manifest["token_count"] // config.reader.sequence_length
    actual = {
        "expected_token_count": manifest["token_count"],
        "expected_target_token_count": manifest["target_token_count"],
        "expected_complete_sequence_count": actual_sequences,
    }
    expected = {
        "expected_token_count": identity.expected_token_count,
        "expected_target_token_count": identity.expected_target_token_count,
        "expected_complete_sequence_count": (
            identity.expected_complete_sequence_count
        ),
    }
    if actual != expected:
        raise ValueError(
            f"Phase 8 packed counts differ for {path}: "
            f"expected={expected}, actual={actual}"
        )
    frozen_identity = PackedManifestIdentity(
        status="frozen",
        manifest_file_sha256=sha256_file(stage_dir / "manifest.json"),
        manifest_content_sha256=manifest["manifest_content_sha256"],
        ordered_data_sha256=manifest["ordered_data_sha256"],
        **expected,
    )
    frozen = replace(config, packed_manifest_identity=frozen_identity)
    frozen.require_packed_launch_ready()
    # Exercise the ordinary packed launch path before rewriting the view.  It
    # compares the full stage, dataset, tokenizer, packing, reader, and frozen
    # manifest identities rather than validating shard integrity alone.
    open_token_batch_source(frozen)
    return frozen, frozen_identity


def _atomic_write(path: Path, config) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"Temporary output already exists: {temporary}")
    with temporary.open("x", encoding="utf-8") as handle:
        yaml.safe_dump(config.to_dict(), handle, sort_keys=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def command() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Offline-validate materialized Phase 8 stages and atomically freeze "
            "their manifest identities in resolved packed views"
        )
    )
    parser.add_argument("views", nargs="+")
    args = parser.parse_args()
    resolved = [Path(value).expanduser().resolve() for value in args.views]
    prepared = [(path, *_frozen_view(path)) for path in resolved]
    for path, config, _ in prepared:
        _atomic_write(path, config)
    print_json(
        {
            "status": "frozen",
            "view_count": len(prepared),
            "views": [
                {
                    "path": str(path),
                    "packed_manifest_identity": identity.__dict__,
                }
                for path, _, identity in prepared
            ],
        }
    )


def main() -> None:
    cli_entry(command)


if __name__ == "__main__":
    main()
