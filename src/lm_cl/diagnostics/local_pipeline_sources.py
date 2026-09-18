"""Read immutable checkpoint archives into compact, owned timing sources."""
from __future__ import annotations

import gc
import hashlib
import io
import os
import tarfile
from pathlib import Path

import torch

from lm_cl.data.incremental import file_hash, immutable_write, json_bytes


def derive_archive_sources(archive: Path, expected_sha256: str, limits) -> dict:
    before = file_hash(archive)
    if before != expected_sha256:
        raise ValueError("Preserved archive hash differs")
    root = limits.work/"sources"
    root.mkdir(exist_ok=True)
    records = []
    # Stream the archive once. Never extract a full production checkpoint to disk.
    with tarfile.open(archive, mode="r|gz") as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith(".pt"):
                continue
            if member.size > 2*1024**3 or len(records) >= 2:
                raise ValueError("Archive exceeds frozen two-checkpoint bounded source contract")
            f = tar.extractfile(member)
            assert f is not None
            content = f.read(member.size+1)
            if len(content) != member.size:
                raise ValueError("Incomplete archive member")
            member_hash = hashlib.sha256(content).hexdigest()
            p = torch.load(io.BytesIO(content), map_location="cpu", weights_only=False)
            del content
            state = p["trainer_state"]
            if state["phase"] != "task_boundary" or state["window_logical_batches"] or state["window_valid_targets"]:
                raise ValueError("Archive member is not a stable boundary")
            if any(v is not None for v in p["gradients"].values()):
                raise ValueError("Archive boundary contains partial slow gradients")
            variant = p["resolved_config"]["variant"]["name"]
            compact = {"checkpoint_kind": "lm-cl-derived-timing-source-v1",
                       "model_state": p["model_state"], "resolved_config": {"model": p["resolved_config"]["model"],
                       "variant": p["resolved_config"]["variant"]},
                       "trainer_state": {"phase":"task_boundary", "window_logical_batches":0},
                       "origin": {"archive_path":str(archive.resolve()), "archive_sha256":before,
                                  "member":member.name, "member_sha256":member_hash,
                                  "original_trainer_state":state, "original_source_identity":p["source_identity"]},
                       "excluded": ["optimizer", "scheduler", "scaler", "rng", "active_memory", "partial_gradients", "source_position"],
                       "use": "immutable slow-weight initializer for fresh local timing only"}
            path = root/f"5m-{variant}-cycle5-timing-source.pt"
            if path.exists():
                raise FileExistsError(path)
            reserve = sum(t.numel()*t.element_size() for t in p["model_state"].values())+1024**2
            with limits.allocation(reserve):
                temporary = path.with_name(path.name+".partial")
                with temporary.open("xb") as output:
                    torch.save(compact, output); output.flush(); os.fsync(output.fileno())
                check = torch.load(temporary, map_location="cpu", weights_only=False)
                for name, value in compact["model_state"].items():
                    if not torch.equal(value, check["model_state"][name]):
                        raise ValueError("Derived source weights differ")
                os.link(temporary, path)
                records.append({"path": str(path), "bytes":path.stat().st_size, "sha256":file_hash(path),
                                "variant":variant, "member":member.name, "member_sha256":member_hash,
                                "model":compact["resolved_config"]["model"]})
            del p, compact, check
            gc.collect()
    after = file_hash(archive)
    if before != after or len(records) != 2:
        raise ValueError("Source archive changed or lacks the two expected models")
    result = {"archive":str(archive.resolve()), "sha256_before":before, "sha256_after":after, "sources":records}
    immutable_write(limits.report/"provenance/5m-derived-sources.json",json_bytes(result))
    return result
