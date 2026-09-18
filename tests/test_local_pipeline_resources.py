import json
import os
from pathlib import Path
from types import SimpleNamespace

from lm_cl.diagnostics.local_pipeline_resources import Limits, allocated_bytes


def test_allocation_scan_tolerates_atomic_temporary_rename(tmp_path,monkeypatch):
    temporary=tmp_path/".record.tmp";final=tmp_path/"record.json"
    temporary.write_bytes(b"record")
    original=Path.stat
    def racing_stat(path,*args,**kwargs):
        if path==temporary and kwargs.get("follow_symlinks",True):
            temporary.replace(final)
            raise FileNotFoundError(str(temporary))
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,"stat",racing_stat)
    allocated_bytes(tmp_path)  # The transient scan must not fail a trainer.
    assert final.exists()
    assert allocated_bytes(tmp_path)==final.stat().st_blocks*512


def test_dead_pid_observation_stops_growing_elapsed_without_touching_original(tmp_path,monkeypatch):
    import psutil
    import lm_cl.diagnostics.local_pipeline_resources as module
    limits=object.__new__(Limits);limits.work=tmp_path/"work";limits.report=tmp_path/"report"
    (limits.work/"processes").mkdir(parents=True);limits.report.mkdir()
    limits.v={"minimum_free_bytes":100*1024**3,"max_additional_allocated_bytes":50*1024**3,
              "max_hf_cache_bytes":20*1024**3,"max_source_cache_bytes":5*1024**3,
              "max_benchmark_elapsed_seconds":43200}
    monkeypatch.setattr(module.shutil,"disk_usage",lambda _:SimpleNamespace(free=500*1024**3))
    record={"pid":os.getpid(),"create_time":psutil.Process().create_time()-100,
            "role":"test","label":"prior-process-identity","started_unix":100.,"status":"running"}
    path=limits.work/"processes"/"record.json";content=json.dumps(record);path.write_text(content)
    monkeypatch.setattr(module.time,"time",lambda:200.)
    first=limits.check();assert first["benchmark_elapsed_seconds"]==100.
    monkeypatch.setattr(module.time,"time",lambda:300.)
    second=limits.check();assert second["benchmark_elapsed_seconds"]==100.
    assert path.read_text()==content
    assert psutil.Process(os.getpid()).is_running()  # Never signal the reused PID.
