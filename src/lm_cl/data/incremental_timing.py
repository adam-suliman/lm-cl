"""Explicit cached-data timing fallback. This is not a new scientific dataset.

Repeated exposure is declared, never hidden behind synthetic document IDs.
Legacy inputs are checked read-only; only new owned blocks are written.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from lm_cl.data.incremental import FORMAT, Recipe, IncrementalSource, file_hash, initialize, immutable_write, json_bytes
from lm_cl.data.packed import load_packed_manifest, validate_packed_shards


def import_timing_replay(*, packed_root: Path, output_root: Path, repeats: int,
                         block_tokens: int, limits, validation: bool = False) -> dict:
    limits.owned(output_root)
    if not 1 <= repeats <= 256 or block_tokens <= 0:
        raise ValueError("Invalid bounded timing replay")
    report = validate_packed_shards(packed_root)
    stage, manifest = load_packed_manifest(packed_root)
    expected_purpose = "vietnamese_validation" if validation else "vietnamese_train"
    if manifest["stage"]["purpose"] != expected_purpose or manifest["stage"]["language"] != "vi":
        raise ValueError("Timing fallback requires the declared Vietnamese packed role")
    if manifest["token_count"] > 1_048_576:
        raise ValueError("Timing fallback only accepts a bounded existing packed fixture")
    length = manifest["reader"]["sequence_length"]
    if manifest["token_count"] % length:
        raise ValueError("Timing fallback requires a complete-sequence source")
    source_tokens = np.concatenate([np.fromfile(stage/s["filename"], dtype="<u4") for s in manifest["shards"]])
    token_count = len(source_tokens)*repeats
    if token_count > limits.v["max_unique_prepared_tokens"]:
        raise ValueError("Timing token cap exceeded")
    boundaries_meta = manifest["boundaries"]
    original = [json.loads(line) for line in (stage/boundaries_meta["filename"]).read_text().splitlines()]
    if validation and repeats != 1:
        raise ValueError("Fixed timing validation is never repeated")
    identity = {"kind": "completed_packed_timing_replay_v1", "manifest_path": str((stage/"manifest.json").resolve()),
                "manifest_file_sha256": file_hash(stage/"manifest.json"),
                "manifest_content_sha256": manifest["manifest_content_sha256"],
                "ordered_data_sha256": manifest["ordered_data_sha256"], "repeats": repeats,
                "unique_input_tokens": len(source_tokens), "scientific_use": "prohibited_timing_only",
                "source_verification": report}
    selection = manifest["selection"]
    t = manifest["tokenizer"]
    eos = t["special_token_ids"]["eos_token_id"]
    recipe = Recipe(FORMAT, identity, manifest["stage"]["language"], "validation" if validation else "timing_only",
                    t, selection["max_input_documents"], token_count, block_tokens, length,
                    selection["document_order_seed"], selection["shuffle_buffer_documents"], selection["split_seed"],
                    selection["validation_permyriad"], eos, t["maximum_emitted_token_id"], "text", "url", [])
    recipe.validate()
    initialize(output_root, recipe)
    previous = recipe.sha256
    total_blocks = (token_count+block_tokens-1)//block_tokens
    by_block = [[] for _ in range(total_blocks)]
    index = 0
    for repeat in range(repeats):
        for b in original:
            record = dict(b)
            record.update(document_index=index, token_start=b["token_start"]+repeat*len(source_tokens),
                          token_end=b["token_end"]+repeat*len(source_tokens))
            by_block[record["token_start"]//block_tokens].append(record)
            index += 1
    for i in range(total_blocks):
        start = i*block_tokens
        count = min(block_tokens, token_count-start)
        indices = np.arange(start, start+count) % len(source_tokens)
        content = source_tokens[indices].astype("<u4").tobytes()
        record = {"format": FORMAT, "recipe_sha256": recipe.sha256, "index": i, "previous": previous,
                  "start": start, "count": count, "data_sha256": hashlib.sha256(content).hexdigest(),
                  "boundaries": by_block[i], "producer_state": {"kind": "timing_replay_import", "next_token": start+count}}
        with limits.allocation(len(content)+1024**2):
            immutable_write(output_root/"blocks"/f"{i:08d}.bin", content)
            path = output_root/"commits"/f"{i:08d}.json"
            immutable_write(path, json_bytes(record))
            previous = file_hash(path)
    immutable_write(output_root/"complete.json", json_bytes({"format":FORMAT,"recipe_sha256":recipe.sha256,
        "blocks":total_blocks,"tokens":token_count,"chain_sha256":previous}))
    source = IncrementalSource(output_root, wait_seconds=0)
    return {"root": str(output_root), "recipe_sha256": recipe.sha256, "purpose": recipe.purpose,
            "unique_input_tokens": len(source_tokens), "exposure_tokens": source.token_count, "repeats": repeats}


def verify_timing_replay(source: IncrementalSource) -> dict:
    """Accept repeats only by matching every byte and boundary to frozen inputs."""
    identity=source.recipe.source_identity
    if identity.get("kind")!="completed_packed_timing_replay_v1" or identity.get("scientific_use")!="prohibited_timing_only":
        raise ValueError("Not an explicitly declared timing replay")
    path=Path(identity["manifest_path"])
    if file_hash(path)!=identity["manifest_file_sha256"]:
        raise ValueError("Original timing fixture manifest changed")
    validate_packed_shards(path.parent)
    stage,manifest=load_packed_manifest(path.parent)
    if manifest["manifest_content_sha256"]!=identity["manifest_content_sha256"] or manifest["ordered_data_sha256"]!=identity["ordered_data_sha256"]:
        raise ValueError("Original timing fixture identity changed")
    original=np.concatenate([np.fromfile(stage/s["filename"],dtype="<u4") for s in manifest["shards"]])
    repeats=identity["repeats"]
    if type(repeats) is not int or not 1<=repeats<=256 or source.token_count!=len(original)*repeats or identity["unique_input_tokens"]!=len(original):
        raise ValueError("Declared timing exposure/unique counts differ")
    offset=0
    for array in source.arrays:
        expected=original[np.arange(offset,offset+len(array))%len(original)]
        if not np.array_equal(array,expected):
            raise ValueError("Timing replay token bytes differ from frozen fixture")
        offset+=len(array)
    boundaries=[json.loads(l) for l in (stage/manifest["boundaries"]["filename"]).read_text().splitlines()]
    actual=[b for record in source.records for b in record["boundaries"]]
    if len(actual)!=len(boundaries)*repeats:
        raise ValueError("Timing replay document count differs")
    for index,b in enumerate(actual):
        repeat,within=divmod(index,len(boundaries))
        expected=dict(boundaries[within])
        expected.update(document_index=index,token_start=expected["token_start"]+repeat*len(original),
                        token_end=expected["token_end"]+repeat*len(original))
        if b!=expected:
            raise ValueError("Timing replay canonical document differs")
    return {"unique_input_tokens":len(original),"exposure_tokens":source.token_count,"repeats":repeats,
            "scientific_use":"prohibited_timing_only","original_manifest_sha256":identity["manifest_file_sha256"]}
