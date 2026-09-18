from __future__ import annotations

import argparse
import os
from dataclasses import replace
from pathlib import Path

import yaml

from lm_cl.cli._common import cli_entry, print_json
from lm_cl.config import load_probe_config
from lm_cl.training.checkpoint import sha256_file
from lm_cl.training.probe import validate_probe_source_checkpoint


def _require_final_russian_boundary(identity: dict) -> None:
    boundary = identity["continual_boundary"]
    expected = {
        "phase": "task_boundary",
        "task_index": 7,
        "next_task_index": 8,
        "cycle_index": 0,
        "language": "ru",
    }
    actual = {name: boundary.get(name) for name in expected}
    if actual != expected:
        raise ValueError(
            "Phase 8 probe source is not the final Russian boundary: "
            f"expected={expected}, actual={actual}"
        )


def _atomic_write(path: Path, config) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
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
            "Validate final Russian continual boundaries and atomically freeze "
            "their SHA-256 identities into Phase 8 probe configs"
        )
    )
    parser.add_argument("configs", nargs="+")
    args = parser.parse_args()
    prepared = []
    for value in args.configs:
        path = Path(value).expanduser().resolve()
        config = load_probe_config(path)
        if config.source_checkpoint_status != "pending":
            raise ValueError(f"Probe config is not pending: {path}")
        checksum = sha256_file(config.source_checkpoint)
        frozen = replace(
            config,
            source_checkpoint_status="frozen",
            source_checkpoint_sha256=checksum,
        )
        frozen.validate()
        _, identity = validate_probe_source_checkpoint(frozen)
        _require_final_russian_boundary(identity)
        prepared.append((path, frozen, identity))
    for path, frozen, _ in prepared:
        _atomic_write(path, frozen)
    print_json(
        {
            "status": "frozen",
            "probe_config_count": len(prepared),
            "sources": [
                {
                    "config": str(path),
                    "source_checkpoint": identity,
                }
                for path, _, identity in prepared
            ],
        }
    )


def main() -> None:
    cli_entry(command)


if __name__ == "__main__":
    main()
