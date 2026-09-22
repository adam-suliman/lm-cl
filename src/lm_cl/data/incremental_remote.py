"""Bounded, pinned Parquet range reads with an immutable local range cache.

Uses installed PyArrow and standard Hugging Face authentication. No secrets or
redirect URLs are logged. This source is only available to the producer.
"""
from __future__ import annotations

import io
import json
import os
import time
from collections import OrderedDict
from pathlib import Path
from urllib.parse import quote

from lm_cl.data.incremental import digest, file_hash, immutable_write, json_bytes


REVISION = "6a8734bc69fefcbb7735f4f9250f43e4cd7a442e"
MAPPINGS = {"en": "en", "zh_written": "zh", "fr": "fr", "ja": "ja", "es": "es", "de": "de", "pt": "pt", "ru": "ru", "vi": "vi"}


class RemoteAccess:
    def __init__(self, limits):
        import requests
        from huggingface_hub import get_token
        if not os.environ.get("HF_HOME"):
            raise ValueError("Explicit HF_HOME required")
        self.limits = limits
        self.session = requests.Session()
        token = get_token()
        if token:
            self.session.headers["Authorization"] = "Bearer " + token
        self.network_bytes = 0
        self.cache_bytes = 0
        self.requests = 0
        self.log = limits.report/"raw/network.jsonl"

    def fetch(self, url, *, maximum_bytes, byte_range=None):
        import requests
        failures = []
        for attempt in range(self.limits.v["max_network_attempts_per_request"]):
            request_number = self.limits.network_attempt()
            self.limits.check(reserve_bytes=maximum_bytes)
            start = time.monotonic()
            try:
                headers = {} if byte_range is None else {"Range": f"bytes={byte_range[0]}-{byte_range[1]-1}"}
                with self.session.get(url, headers=headers, stream=True,
                          timeout=self.limits.v["network_timeout_seconds"]) as response:
                    status = response.status_code
                    if status in {401, 403}:
                        raise PermissionError("Pinned dataset access denied; existing HF authentication required")
                    if byte_range is not None and status != 206:
                        raise ValueError(f"Server ignored range request (HTTP {status}); full download refused")
                    if status != 200 and status != 206:
                        raise RuntimeError(f"HTTP status {status}")
                    if int(response.headers.get("Content-Length", "0")) > maximum_bytes:
                        raise ValueError("Remote response exceeds byte cap")
                    parts, size = [], 0
                    for chunk in response.iter_content(65536):
                        size += len(chunk)
                        self.network_bytes += len(chunk)
                        if size > maximum_bytes:
                            raise ValueError("Remote response exceeds byte cap")
                        parts.append(chunk)
                    data = b"".join(parts)
                    if byte_range is not None:
                        a,b = byte_range
                        if len(data) != b-a or not response.headers.get("Content-Range", "").startswith(f"bytes {a}-{b-1}/"):
                            raise ValueError("Invalid HTTP range response")
                self.last_links = dict(response.links)
                self.requests += 1
                with self.log.open("a") as f:
                    f.write(json.dumps({"request": request_number, "attempt": attempt+1, "status": status,
                        "range": byte_range, "bytes": len(data), "seconds": time.monotonic()-start,
                        "resource_sha256": digest(url), "unix_time": time.time()})+"\n")
                return data
            except (PermissionError, ValueError):
                raise
            except (requests.RequestException, RuntimeError) as e:
                failures.append(type(e).__name__)
                with self.log.open("a") as f:
                    f.write(json.dumps({"request": request_number, "attempt": attempt+1,
                        "error_type": type(e).__name__, "seconds": time.monotonic()-start,
                        "resource_sha256": digest(url), "unix_time": time.time()})+"\n")
        raise RuntimeError(f"Bounded remote request exhausted attempts: {failures}")

    def discover(self, language, max_files):
        if language not in MAPPINGS or not 1 <= max_files <= 8:
            raise ValueError("Invalid language/file cap")
        config = MAPPINGS[language]
        url = f"https://huggingface.co/api/datasets/uonlp/CulturaX/tree/{REVISION}/{config}?recursive=false&expand=false&limit=1000"
        items = json.loads(self.fetch(url, maximum_bytes=4*1024**2))
        files = sorted([item for item in items if item.get("type")=="file" and item["path"].endswith(".parquet")], key=lambda x:x["path"])
        if not files:
            raise ValueError("Pinned language directory has no Parquet files")
        # Bounded file prefix is explicit. Never silently extend the frozen source.
        selected = [{"path": f["path"], "size": f["size"], "oid": f.get("oid"), "lfs": f.get("lfs")} for f in files[:max_files]]
        return {"kind": "pinned_culturax_parquet_prefix_v1", "repo_id": "uonlp/CulturaX",
                "revision": REVISION, "language_config": config, "split": "train",
                "file_order": "lexicographic_repo_path_v1", "files": selected,
                "columns": ["text", "url"], "range_cache": "immutable_exact_ranges_v1"}


class RangeFile(io.RawIOBase):
    def __init__(self, remote, identity, entry):
        self.remote, self.identity, self.entry = remote, identity, entry
        self.position = 0
        self.size = entry["size"]
        self.url = f"https://huggingface.co/datasets/uonlp/CulturaX/resolve/{identity['revision']}/{quote(entry['path'])}"
        self.cache = remote.limits.work/"source-cache"/digest(self.url)
        self.cache.mkdir(exist_ok=True)

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.position

    def seek(self, offset, whence=0):
        value = offset if whence==0 else self.position+offset if whence==1 else self.size+offset
        if value < 0:
            raise ValueError("Negative file offset")
        self.position = value
        return value

    def read(self, size=-1):
        if self.closed:
            raise ValueError("Closed range file")
        end = self.size if size < 0 else min(self.size, self.position+size)
        size = max(0, end-self.position)
        if not size:
            return b""
        if size > self.remote.limits.v["max_row_group_bytes"]:
            raise ValueError("Parquet range exceeds bounded fragment size")
        path = self.cache/f"{self.position:012d}-{end:012d}.bin"
        meta = path.with_suffix(".json")
        if path.exists() and meta.exists():
            record = json.loads(meta.read_text())
            if file_hash(path) != record["sha256"] or path.stat().st_size != size:
                raise ValueError("Corrupt immutable source range")
            data = path.read_bytes()
            self.remote.cache_bytes += len(data)
        else:
            data = self.remote.fetch(self.url, maximum_bytes=size, byte_range=(self.position,end))
            with self.remote.limits.allocation(len(data)+8192):
                from lm_cl.diagnostics.local_pipeline_resources import allocated_bytes
                if allocated_bytes(self.remote.limits.work/"source-cache")+len(data)+8192 > self.remote.limits.v["max_source_cache_bytes"]:
                    raise RuntimeError("Source cache reservation exceeds cap")
                immutable_write(path, data)
                immutable_write(meta, json_bytes({"sha256": file_hash(path), "size": len(data),
                    "path": self.entry["path"], "revision": self.identity["revision"], "start": self.position, "end": end}))
        self.position = end
        return data

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)


class ParquetRows:
    def __init__(self, identity, remote):
        if identity.get("revision") != REVISION or identity.get("repo_id") != "uonlp/CulturaX":
            raise ValueError("Unapproved or unpinned data identity")
        if identity.get("language_config") not in MAPPINGS.values():
            raise ValueError("Unapproved language mapping")
        if identity.get("columns") != ["text", "url"] or not identity.get("files"):
            raise ValueError("Invalid source columns/files")
        self.identity, self.remote = identity, remote
        self.files, self.groups = [], []
        self.indexed_rows = 0
        self.loaded_groups = OrderedDict()
        self.loaded_bytes = 0
        self.read_seconds = 0.

    def _open_next(self):
        import pyarrow.parquet as pq
        i = len(self.files)
        if i == len(self.identity["files"]):
            return False
        file = pq.ParquetFile(RangeFile(self.remote, self.identity, self.identity["files"][i]), pre_buffer=False)
        if not {"text", "url"}.issubset(file.schema.names):
            raise ValueError("Pinned Parquet fields differ")
        self.files.append(file)
        for g in range(file.num_row_groups):
            group = file.metadata.row_group(g)
            count = group.num_rows
            self.groups.append((self.indexed_rows, self.indexed_rows+count, i, g))
            self.indexed_rows += count
        return True

    def row(self, index):
        start = time.monotonic()
        while index >= self.indexed_rows:
            if not self._open_next():
                return None
        for a,b,i,g in self.groups:
            if a <= index < b:
                key = (i,g)
                if key not in self.loaded_groups:
                    table = self.files[i].read_row_group(g, columns=["text", "url"], use_threads=False)
                    if table.nbytes > self.remote.limits.v["max_row_group_bytes"]*8:
                        raise ValueError("Decoded row group exceeds memory cap")
                    estimate = max(table.nbytes*4, 1)
                    while self.loaded_groups and self.loaded_bytes+estimate > 512*1024**2:
                        _, (_, used) = self.loaded_groups.popitem(last=False)
                        self.loaded_bytes -= used
                    self.loaded_groups[key] = (table.to_pylist(), estimate)
                    self.loaded_bytes += estimate
                self.loaded_groups.move_to_end(key)
                self.read_seconds += time.monotonic()-start
                return self.loaded_groups[key][0][index-a]
        raise RuntimeError("Source cursor has no row group")
