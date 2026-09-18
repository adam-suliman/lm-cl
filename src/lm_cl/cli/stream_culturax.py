from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from lm_cl.cli._common import (
    native_streaming_cli_entry,
    print_json,
    write_json,
)
from lm_cl.config import REQUIRED_LANGUAGE_KEYS, load_data_pipeline_config
from lm_cl.data.huggingface import stream_culturax_rows
from lm_cl.data.sources import build_bounded_culturax_stream
from lm_cl.data.storage import enforce_disk_limit
from lm_cl.data.tokenizer import load_verified_tokenizer


def run() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Open a strictly bounded pinned CulturaX stream. Uses standard "
            "Hugging Face authentication and never prints raw documents."
        )
    )
    parser.add_argument("config")
    parser.add_argument("--report")
    parser.add_argument(
        "--all-required-languages",
        action="store_true",
        help=(
            "run the same strict bounds independently for en, written Chinese, "
            "fr, ja, es, de, pt, ru, and vi"
        ),
    )
    args = parser.parse_args()
    config = load_data_pipeline_config(args.config)
    if config.mode != "culturax_stream":
        raise ValueError("Config mode must be culturax_stream")
    config.require_access_ready()
    output = None
    generated_root = Path(
        config.storage.generated_root
    ).expanduser().resolve()
    if args.report:
        output = Path(args.report).expanduser().resolve()
        try:
            output.relative_to(generated_root)
        except ValueError as exc:
            raise ValueError(
                "--report must be inside storage.generated_root"
            ) from exc
        if output.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing report: {output}"
            )
    tokenizer, tokenizer_manifest = load_verified_tokenizer(config.tokenizer)
    if args.all_required_languages:
        reports = {}
        for language in REQUIRED_LANGUAGE_KEYS:
            language_config = replace(
                config,
                stage=replace(
                    config.stage,
                    stage_id=f"inspect-{language}-bounded",
                    purpose="inspection",
                    language=language,
                ),
            )
            source = build_bounded_culturax_stream(
                language_config,
                rows=stream_culturax_rows(language_config),
                tokenizer=tokenizer,
                tokenizer_manifest=tokenizer_manifest,
            )
            reports[language] = source.report
        report = {
            "status": "complete",
            "all_required_languages": True,
            "language_reports": reports,
        }
    else:
        source = build_bounded_culturax_stream(
            config,
            rows=stream_culturax_rows(config),
            tokenizer=tokenizer,
            tokenizer_manifest=tokenizer_manifest,
        )
        report = source.report
    if args.report:
        assert output is not None
        write_json(output, report)
        enforce_disk_limit(
            generated_root,
            config.storage.max_generated_bytes,
            label="Generated data",
        )
    print_json(report)


def main() -> None:
    native_streaming_cli_entry(run)


if __name__ == "__main__":
    main()
