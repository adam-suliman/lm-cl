from __future__ import annotations

import argparse

from lm_cl.cli._common import cli_entry, print_json
from lm_cl.data.packed import PackedShardSource, validate_packed_shards


def run() -> None:
    parser = argparse.ArgumentParser(
        description="Validate packed manifest, counts, boundaries, and SHA-256 files"
    )
    parser.add_argument("path")
    parser.add_argument("--expected-tokenizer-manifest-sha256")
    parser.add_argument(
        "--read-batches",
        type=int,
        default=0,
        help="also mmap and reread this many batches from the start",
    )
    parser.add_argument("--sequence-length", type=int)
    parser.add_argument("--global-sequences-per-batch", type=int)
    args = parser.parse_args()
    report = validate_packed_shards(
        args.path,
        expected_tokenizer_manifest_sha256=(
            args.expected_tokenizer_manifest_sha256
        ),
    )
    if args.read_batches < 0:
        raise ValueError("--read-batches must be non-negative")
    if args.read_batches:
        source = PackedShardSource(args.path)
        sequence_length = (
            args.sequence_length
            or int(source.manifest["reader"]["sequence_length"])
        )
        batch_size = (
            args.global_sequences_per_batch
            or int(
                source.manifest["reader"]["global_sequences_per_batch"]
            )
        )
        batches = source.iter_batches(
            sequence_length=sequence_length,
            global_sequences_per_batch=batch_size,
        )
        read = 0
        targets = 0
        next_position = None
        for _ in range(args.read_batches):
            try:
                batch = next(batches)
            except StopIteration:
                break
            read += 1
            targets += batch.valid_target_count
            next_position = {
                "shard_index": batch.next_position.shard_index,
                "token_offset": batch.next_position.token_offset,
            }
        report["reread"] = {
            "requested_batches": args.read_batches,
            "read_batches": read,
            "valid_target_count": targets,
            "next_position": next_position,
        }
    print_json(report)


def main() -> None:
    cli_entry(run)


if __name__ == "__main__":
    main()
