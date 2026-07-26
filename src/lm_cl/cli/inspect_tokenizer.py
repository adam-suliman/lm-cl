from __future__ import annotations

import argparse
from pathlib import Path

from lm_cl.cli._common import cli_entry, print_json
from lm_cl.data.storage import enforce_disk_limit, ensure_owned_root
from lm_cl.data.tokenizer import inspect_tokenizer


def run() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect and hash a pinned tokenizer using standard Hugging Face "
            "authentication. No credential argument is accepted."
        )
    )
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", required=True, help="40-hex commit")
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--generated-root", required=True)
    parser.add_argument("--max-cache-bytes", required=True, type=int)
    parser.add_argument("--max-generated-bytes", required=True, type=int)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument(
        "--model-embedding-vocab-size", type=int, required=True
    )
    args = parser.parse_args()
    if args.model_embedding_vocab_size != 151_680:
        raise ValueError(
            "Faithful tokenizer inspection requires "
            "--model-embedding-vocab-size 151680"
        )
    cache = ensure_owned_root(args.cache_root, purpose="hf-cache")
    generated = ensure_owned_root(
        args.generated_root, purpose="generated-data"
    )
    enforce_disk_limit(cache, args.max_cache_bytes, label="Hugging Face cache")
    enforce_disk_limit(
        generated, args.max_generated_bytes, label="Generated data"
    )
    output = Path(args.output_manifest).expanduser().resolve()
    try:
        output.relative_to(generated)
    except ValueError as exc:
        raise ValueError(
            "--output-manifest must be inside --generated-root"
        ) from exc
    manifest = inspect_tokenizer(
        repo_id=args.repo_id,
        revision=args.revision,
        cache_dir=cache,
        output_manifest=output,
        model_embedding_vocab_size=args.model_embedding_vocab_size,
    )
    enforce_disk_limit(cache, args.max_cache_bytes, label="Hugging Face cache")
    enforce_disk_limit(
        generated, args.max_generated_bytes, label="Generated data"
    )
    print_json(manifest)


def main() -> None:
    cli_entry(run)


if __name__ == "__main__":
    main()
