"""Run with unittest in the installed producer environment (PyArrow required)."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from lm_cl.data.incremental import Producer, Recipe, FORMAT, initialize, IncrementalSource
from lm_cl.data.incremental_remote import ParquetRows, RangeFile, REVISION


class LimitsFixture:
    def __init__(self, root):
        self.work = root
        (root/"source-cache").mkdir()
        self.v = {"max_row_group_bytes": 1024**2, "max_source_cache_bytes": 10*1024**2}

    def allocation(self, _):
        return contextlib.nullcontext()


class LocalRanges:
    def __init__(self, path, root):
        self.path = path
        self.limits = LimitsFixture(root)
        self.cache_bytes = 0
        self.network_bytes = 0
        self.calls = []

    def fetch(self, url, *, maximum_bytes, byte_range):
        start, end = byte_range
        with self.path.open("rb") as f:
            f.seek(start)
            result = f.read(end-start)
        assert len(result) <= maximum_bytes
        self.calls.append(byte_range)
        self.network_bytes += len(result)
        return result


class Tokenizer:
    def encode(self, text, *, add_special_tokens=False):
        return list(text.encode())


class ParquetRangeTests(unittest.TestCase):
    def test_seek_row_groups_cache_and_exact_producer_resume(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            rows = [{"text": f"document {i} " + "xyz"*(i%7+1), "url": str(i)} for i in range(300)]
            local = root/"fixture.parquet"
            pq.write_table(pa.Table.from_pylist(rows), local, row_group_size=23)
            identity = {"repo_id": "uonlp/CulturaX", "revision": REVISION, "language_config": "vi",
                        "columns": ["text", "url"], "files": [{"path": "vi/fixture.parquet", "size": local.stat().st_size}]}
            remote = LocalRanges(local, root)
            source = ParquetRows(identity, remote)
            self.assertEqual([source.row(i) for i in [0, 24, 299, 11, 87]], [rows[i] for i in [0, 24, 299, 11, 87]])
            recipe = Recipe(FORMAT, identity, "vi", "train", {"fixture": "utf8"}, 300, 2048, 67, 16, 31, 51, 41, 100, 255, 255, "text", "url", [])
            initialize(root/"reference", recipe)
            Producer(root/"reference", ParquetRows(identity, remote), Tokenizer()).run()
            initialize(root/"resumed", recipe)
            Producer(root/"resumed", ParquetRows(identity, remote), Tokenizer()).run(max_blocks=2)
            calls_before = len(remote.calls)
            Producer(root/"resumed", ParquetRows(identity, remote), Tokenizer()).run()
            self.assertEqual(calls_before, len(remote.calls), "Warm resume should reuse verified ranges")
            a = IncrementalSource(root/"reference", wait_seconds=0)
            b = IncrementalSource(root/"resumed", wait_seconds=0)
            self.assertEqual(a.read_tokens(2048)[0].tobytes(), b.read_tokens(2048)[0].tobytes())
            self.assertEqual((a.root/"complete.json").read_bytes(), (b.root/"complete.json").read_bytes())
            self.assertGreater(remote.network_bytes, 0)
            self.assertGreater(remote.cache_bytes, 0)
            entry = identity["files"][0]
            f = RangeFile(remote, identity, entry)
            f.seek(10); first = f.read(20); f.seek(10)
            self.assertEqual(first, f.read(20))
            cached = f.cache/"000000000010-000000000030.bin"
            with cached.open("r+b") as corrupt:
                corrupt.write(b"x")
            f.seek(10)
            with self.assertRaisesRegex(ValueError, "Corrupt"):
                f.read(20)

    def test_unknown_revision_and_large_range_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); source=root/"fixture"; source.write_bytes(b"x"*32)
            remote=LocalRanges(source, root)
            with self.assertRaises(ValueError):
                ParquetRows({"revision":"main", "repo_id":"uonlp/CulturaX"}, remote)
            f=RangeFile(remote, {"revision":REVISION}, {"path":"vi/fixture", "size":2*1024**2})
            with self.assertRaisesRegex(ValueError, "bounded"):
                f.read(2*1024**2)
            self.assertEqual(remote.calls, [])


if __name__ == "__main__":
    unittest.main()
