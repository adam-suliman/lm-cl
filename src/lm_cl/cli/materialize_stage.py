from __future__ import annotations

import argparse
import json
import sys

from lm_cl.cli._common import native_streaming_cli_entry, print_json
from lm_cl.config import load_data_pipeline_config
from lm_cl.data.huggingface import stream_culturax_rows
from lm_cl.data.materialize import (
    dry_run_materialization,
    materialize_stage,
)
from lm_cl.data.tokenizer import load_verified_tokenizer
from lm_cl.data.storage import clean_owned_root


def run() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Dry-run or materialize one independently resumable pinned "
            "CulturaX stage. Network execution requires --execute."
        )
    )
    parser.add_argument("config")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="perform bounded streaming and write owned generated data",
    )
    args = parser.parse_args()
    config = load_data_pipeline_config(args.config)
    if config.mode != "culturax_stage_materialize":
        raise ValueError("Config mode must be culturax_stage_materialize")
    estimate = dry_run_materialization(config)
    if not args.execute:
        print_json({"status": "dry_run_only", **estimate})
        return
    if not estimate["fits_generated_cap"]:
        raise ValueError("Dry-run estimate does not fit generated-data cap")
    if not estimate["fits_temporary_cap"]:
        raise ValueError("Dry-run estimate does not fit temporary-data cap")
    print(
        json.dumps(
            {"event": "materialization_preflight", **estimate},
            sort_keys=True,
        ),
        file=sys.stderr,
        flush=True,
    )
    config.require_access_ready()
    tokenizer, tokenizer_manifest = load_verified_tokenizer(config.tokenizer)
    manifest = materialize_stage(
        config,
        rows=stream_culturax_rows(config),
        tokenizer=tokenizer,
        tokenizer_manifest=tokenizer_manifest,
    )
    cleanup = None
    if (
        config.storage.auto_clean_cache
        and config.stage.delete_temporary_cache_after_success
    ):
        cleanup = clean_owned_root(
            config.storage.hf_cache_root, purpose="hf-cache"
        )
    print_json(
        {
            "status": "complete",
            "stage_id": manifest["stage"]["stage_id"],
            "token_count": manifest["token_count"],
            "target_token_count": manifest["target_token_count"],
            "accepted_document_count": manifest[
                "accepted_document_count"
            ],
            "manifest_content_sha256": manifest[
                "manifest_content_sha256"
            ],
            "ordered_data_sha256": manifest["ordered_data_sha256"],
            "owned_cache_cleanup": cleanup,
        }
    )


def main() -> None:
    native_streaming_cli_entry(run)


if __name__ == "__main__":
    main()
