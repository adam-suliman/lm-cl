from __future__ import annotations

import argparse

from lm_cl.cli._common import cli_entry, print_json
from lm_cl.data.packed import load_packed_manifest


def run() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect a complete packed-stage manifest without raw text"
    )
    parser.add_argument("path")
    args = parser.parse_args()
    _, manifest = load_packed_manifest(args.path)
    print_json(
        {
            "completion_status": manifest["completion_status"],
            "stage": manifest["stage"],
            "dataset": manifest["dataset"],
            "tokenizer": {
                "repo_id": manifest["tokenizer"]["repo_id"],
                "revision": manifest["tokenizer"]["revision"],
                "base_vocab_size": manifest["tokenizer"][
                    "base_vocab_size"
                ],
                "effective_vocab_size": manifest["tokenizer"][
                    "effective_vocab_size"
                ],
                "maximum_emitted_token_id": manifest["tokenizer"][
                    "maximum_emitted_token_id"
                ],
                "model_embedding_vocab_size": manifest["tokenizer"][
                    "model_embedding_vocab_size"
                ],
                "trailing_unused_embedding_rows": manifest["tokenizer"][
                    "trailing_unused_embedding_rows"
                ],
                "manifest_content_sha256": manifest["tokenizer"][
                    "manifest_content_sha256"
                ],
                "tokenizer_content_sha256": manifest["tokenizer"][
                    "tokenizer_content_sha256"
                ],
            },
            "token_count": manifest["token_count"],
            "target_token_count": manifest["target_token_count"],
            "accepted_document_count": manifest[
                "accepted_document_count"
            ],
            "shards": manifest["shards"],
            "manifest_content_sha256": manifest[
                "manifest_content_sha256"
            ],
            "ordered_data_sha256": manifest["ordered_data_sha256"],
        }
    )


def main() -> None:
    cli_entry(run)


if __name__ == "__main__":
    main()
