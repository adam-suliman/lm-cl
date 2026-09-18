from __future__ import annotations

import argparse
from pathlib import Path

from lm_cl.cli._common import cli_entry, print_json, write_json
from lm_cl.config import REQUIRED_LANGUAGE_KEYS
from lm_cl.data.huggingface import inspect_culturax
from lm_cl.data.storage import enforce_disk_limit, ensure_owned_root


def _mapping(values: list[str]) -> dict[str, str]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--probe-config values must use LANGUAGE=CONFIG")
        language, configuration = value.split("=", 1)
        if (
            language not in REQUIRED_LANGUAGE_KEYS
            or not configuration
            or language in result
        ):
            raise ValueError(f"Invalid or duplicate probe mapping: {value}")
        result[language] = configuration
    return result


def run() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Discover pinned CulturaX configurations and optionally open a "
            "single streamed row per explicit mapping. Written Chinese is "
            "never selected automatically."
        )
    )
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", required=True, help="40-hex commit")
    parser.add_argument("--split", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--generated-root", required=True)
    parser.add_argument("--max-cache-bytes", required=True, type=int)
    parser.add_argument("--max-generated-bytes", required=True, type=int)
    parser.add_argument("--output")
    parser.add_argument(
        "--probe-config",
        action="append",
        default=[],
        metavar="LANGUAGE=CONFIG",
    )
    args = parser.parse_args()
    cache = ensure_owned_root(args.cache_root, purpose="hf-cache")
    generated = ensure_owned_root(
        args.generated_root, purpose="generated-data"
    )
    enforce_disk_limit(cache, args.max_cache_bytes, label="Hugging Face cache")
    enforce_disk_limit(
        generated, args.max_generated_bytes, label="Generated data"
    )
    output = None
    if args.output:
        output = Path(args.output).expanduser().resolve()
        try:
            output.relative_to(generated)
        except ValueError as exc:
            raise ValueError(
                "--output must be inside --generated-root"
            ) from exc
        if output.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing report: {output}"
            )
    report = inspect_culturax(
        repo_id=args.repo_id,
        revision=args.revision,
        cache_dir=cache,
        split=args.split,
        probe_configs=_mapping(args.probe_config),
    )
    enforce_disk_limit(cache, args.max_cache_bytes, label="Hugging Face cache")
    enforce_disk_limit(
        generated, args.max_generated_bytes, label="Generated data"
    )
    if args.output:
        assert output is not None
        write_json(args.output, report)
        enforce_disk_limit(
            generated, args.max_generated_bytes, label="Generated data"
        )
    print_json(report)


def main() -> None:
    cli_entry(run)


if __name__ == "__main__":
    main()
