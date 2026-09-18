from __future__ import annotations

import argparse

from lm_cl.cli._common import cli_entry, print_json
from lm_cl.config import load_data_pipeline_config
from lm_cl.data.storage import clean_owned_root, directory_size, require_owned_root


def run() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect or clean only the explicitly configured lm-cl-owned HF "
            "cache root. Generated shards and unrelated caches are never touched."
        )
    )
    parser.add_argument("config")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="delete contents below the validated owned cache marker",
    )
    args = parser.parse_args()
    config = load_data_pipeline_config(args.config)
    root = require_owned_root(config.storage.hf_cache_root, purpose="hf-cache")
    if not args.execute:
        print_json(
            {
                "status": "inspection_only",
                "root": str(root),
                "bytes": directory_size(root),
            }
        )
        return
    print_json(
        {
            "status": "cleaned",
            "root": str(root),
            **clean_owned_root(root, purpose="hf-cache"),
        }
    )


def main() -> None:
    cli_entry(run)


if __name__ == "__main__":
    main()

