"""Tiny offline production-path tests: no HF access, CUDA, or full models."""
from contextlib import contextmanager
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import threading
import time

import numpy as np
import pytest
import torch

from lm_cl.config import TokenizerReference
from lm_cl.data.incremental import digest, file_hash
from lm_cl.data.streaming import StreamingPackedSource, cache_path, receipt, connect, prefix_proof, validate_prefix
from lm_cl.data.streaming_producer import ProductionProducer
from lm_cl.launcher import data as launcher_data, jobs as launcher_jobs
from lm_cl.launcher.schema import ForgettingSettings
from lm_cl.launcher.streaming import resolve_streaming_contract, StreamingSettings
from lm_cl.training import ContinualTrainer, ProbeTrainer
from lm_cl.training.checkpoint import load_checkpoint
from lm_cl.training.probe_checkpoint import load_probe_checkpoint
from test_calibration import tiny_config
from test_public_release import _config


class Tokenizer:
    def encode(self, text, **kwargs):
        return [int(x) % 14 for x in hashlib.sha256(text.encode()).digest()[:9]]


class Rows:
    def __init__(self, language): self.language = language
    def row(self, index):
        return {"text":f"{self.language} document {index}", "url":f"fixture://{self.language}/{index}"}


def study(tmp_path, monkeypatch, *, cycles=2, vocab=16):
    torch.set_num_threads(1)
    cfg = _config(tmp_path / "out", cycles=cycles)
    tokenpath = tmp_path / 'generated/tokenizer.json'; tokenpath.parent.mkdir(); tokenpath.write_text('{}')
    ref = TokenizerReference(launcher_data.TOKENIZER_REPO_ID, launcher_data.TOKENIZER_REVISION,
        str(tokenpath), vocab, vocab, vocab-1, vocab, vocab-1, vocab-1)
    monkeypatch.setattr(launcher_data, '_tokenizer_reference', lambda c:(ref, {"manifest_content_sha256":digest({})}, file_hash(tokenpath)))
    cfg = replace(cfg, experiment=replace(cfg.experiment, sequence_length=8, tokens_per_task=48),
        data=replace(cfg.data, mode='streaming', cycle_manifest_policy='disjoint_sequence_windows_v1',
            manifest_root=str(tmp_path/'generated/stages'), manifest_template='unused-{language}/manifest.json',
            generated_root=str(tmp_path/'generated'), dataset_cache_root=str(tmp_path/'cache'), tokenizer_manifest=str(tokenpath),
            probe_training_manifest=str(tmp_path/'unused/manifest.json'), probe_validation_manifest=str(tmp_path/'unused2/manifest.json'),
            language_validation_manifest_template='unused-validation-{language}/manifest.json',
            max_input_documents=10000, shuffle_buffer_documents=3, validation_permyriad=5000,
            streaming=asdict(StreamingSettings(block_tokens=16, prefetch_blocks=1, token_cache_bytes=192,
                metadata_bytes=10*1024**2, minimum_free_bytes=0, wait_seconds=5, max_block_seconds=3))),
        training=replace(cfg.training, global_batch_sequences=2, physical_microbatch_sequences=1),
        fastmem=replace(cfg.fastmem, segment_length=4),
        probe=replace(cfg.probe, training_tokens=32, validation_sequences=2),
        forgetting=ForgettingSettings(True, 'after_each_task_boundary', 2, 'reset', 'mean_validation_ce_from_best_v1'))
    cfg.validate()
    contract = resolve_streaming_contract(cfg, prepare=True)
    model = tiny_config(tmp_path/'unused',1).model
    if vocab != 16:
        model = replace(model, vocab_size=vocab, expected_total_parameters=model.expected_total_parameters+(vocab-16)*model.hidden_size)
    monkeypatch.setattr(launcher_jobs, '_model_config', lambda c:model)
    return cfg, contract


@contextmanager
def worker(root):
    stop = threading.Event(); errors = []
    def run():
        try:
            engine = ProductionProducer(root, source_factory=Rows, tokenizer=Tokenizer())
            while not stop.is_set():
                for path in sorted((Path(root)/'requests').glob('*.json')):
                    engine.ensure(json.loads(path.read_text())['ordinal'])
                    path.unlink(missing_ok=True)
                stop.wait(.005)
        except BaseException as exc:
            errors.append(exc)
    thread=threading.Thread(target=run,daemon=True);thread.start()
    try: yield
    finally:
        stop.set();thread.join(timeout=5)
        assert not thread.is_alive()
        if errors: raise errors[0]


def test_plan_requires_no_tokens_and_partial_k_resume_survives_eviction(tmp_path, monkeypatch):
    cfg, contract = study(tmp_path, monkeypatch)
    root = Path(contract['streaming_root'])
    assert list((root/'cache').iterdir()) == []
    jobs = launcher_jobs.expand_job_specs(cfg, contract)
    trainer_cfg = launcher_jobs.build_continual_job_config(cfg, jobs[1])
    # One language, three logical batches; interrupt inside K=2.
    trainer_cfg = replace(trainer_cfg, tasks=trainer_cfg.tasks[:1])
    with worker(root):
        partial = ContinualTrainer(trainer_cfg).run(stop_after_global_logical_batches=1)
    payload = load_checkpoint(partial.checkpoint_path)
    assert payload['trainer_state']['window_logical_batches'] == 1
    assert len(payload['streaming_prefix_proof']['covering_receipts']) == 1
    with connect(root) as db:
        assert db.execute('SELECT COUNT(*) FROM receipts').fetchone()[0] < len(StreamingPackedSource(root,'train-en').plan['blocks'])
    # Reversible cache eviction only in pytest's disposable root.
    for path in (root/'cache').glob('*.bin'): path.unlink()
    with worker(root):
        resumed = ContinualTrainer(trainer_cfg).run(resume_checkpoint=partial.checkpoint_path)
        fresh_cfg = replace(trainer_cfg, runtime=replace(trainer_cfg.runtime, output_dir=str(tmp_path/'fresh')))
        fresh = ContinualTrainer(fresh_cfg).run()
    left, right = load_checkpoint(resumed.checkpoint_path), load_checkpoint(fresh.checkpoint_path)
    for key in left['model_state']: assert torch.equal(left['model_state'][key],right['model_state'][key])
    assert left['trainer_state'] == right['trainer_state']
    assert left['memory_state']['active_update_count'] == right['memory_state']['active_update_count'] if 'active_update_count' in left['memory_state'] else True
    assert sum(p.stat().st_size for p in (root/'cache').glob('*.bin')) <= 192


def test_reconstruction_including_pending_document_and_prior_ownership(tmp_path, monkeypatch):
    cfg, contract = study(tmp_path, monkeypatch)
    root = Path(contract['streaming_root'])
    engine = ProductionProducer(root, source_factory=Rows, tokenizer=Tokenizer())
    for ordinal in range(10): engine.generate(ordinal)
    hashes = [receipt(root,i)[1] for i in range(10)]
    for ordinal in [0, 7, 1, 9, 3]:
        ProductionProducer(root,source_factory=Rows,tokenizer=Tokenizer()).generate(ordinal)
        assert receipt(root,ordinal)[1] == hashes[ordinal]
    source = StreamingPackedSource(root,'train-en')
    proof = prefix_proof(source.identity, 48)
    validate_prefix(source.identity, proof)
    with connect(root) as db:
        db.execute("UPDATE receipts SET record='{}' WHERE ordinal=0")
    with pytest.raises(ValueError,match='checksum'): validate_prefix(source.identity,proof)


def test_late_data_is_not_eof_and_cache_corruption_fails_closed(tmp_path, monkeypatch):
    _, contract=study(tmp_path,monkeypatch)
    root=Path(contract['streaming_root']);source=StreamingPackedSource(root,'train-en')
    source.plan['limits']['wait_seconds']=.02
    with pytest.raises(TimeoutError): next(source.iter_batches(sequence_length=8,global_sequences_per_batch=2))
    engine=ProductionProducer(root,source_factory=Rows,tokenizer=Tokenizer());engine.generate(0)
    cache_path(root,0).write_bytes(b'0'*64)
    with pytest.raises(ValueError,match='Corrupt'): next(source.iter_batches(sequence_length=8,global_sequences_per_batch=2))


def test_production_pair_with_two_cycles_retention_and_probes(tmp_path, monkeypatch):
    cfg, contract=study(tmp_path,monkeypatch)
    # The production tokenizer schema is checked separately below; this fixture
    # uses a 16-token test vocabulary to keep every actual training step tiny.
    from lm_cl.config.probe_schema import ProbeExperimentConfig
    original = ProbeExperimentConfig._validate_source
    def test_vocabulary(self, source, **kwargs):
        if source.kind == 'streaming_packed':
            from lm_cl.data.streaming import source_from_pipeline
            opened=source_from_pipeline(source.packed)
            assert opened.plan['tokenizer_reference']['model_embedding_vocab_size'] == self.model.vocab_size == 16
        else: original(self,source,**kwargs)
    monkeypatch.setattr(ProbeExperimentConfig,'_validate_source',test_vocabulary)
    root=Path(contract['streaming_root'])
    for spec in launcher_jobs.expand_job_specs(cfg,contract):
        continual=launcher_jobs.build_continual_job_config(cfg,spec)
        checkpoint=None
        with worker(root):
            for cycle in range(2):
                trained=ContinualTrainer(continual).run(resume_checkpoint=checkpoint,stop_after_task_boundaries=(cycle+1)*8)
                checkpoint=trained.checkpoint_path
                source_hash=file_hash(Path(checkpoint))
                probe=launcher_jobs.build_probe_job_config(cfg,spec,cycle_index=cycle,source_checkpoint=Path(checkpoint),source_checkpoint_sha256=source_hash)
                partial=ProbeTrainer(probe).run(stop_after_global_logical_batches=1)
                for p in (root/'cache').glob('*.bin'):p.unlink()
                result=ProbeTrainer(probe).run(resume_checkpoint=partial.checkpoint_path)
                assert result.status == 'complete'
                assert file_hash(Path(checkpoint)) == source_hash
                assert json.loads((Path(probe.runtime.output_dir)/'probe_results.json').read_text())['curve_records']
        assert trained.status == 'complete'
    assert sum(p.stat().st_size for p in (root/'cache').glob('*.bin')) <= 192


def test_production_schema_freezes_real_tokenizer_without_generating_data(tmp_path, monkeypatch):
    cfg, contract=study(tmp_path,monkeypatch,vocab=151680)
    # Replace the test tokenizer descriptor with the exact production values.
    # No model is constructed here.
    ref = TokenizerReference(launcher_data.TOKENIZER_REPO_ID, launcher_data.TOKENIZER_REVISION,
        cfg.data.tokenizer_manifest,151643,151669,151668,151680,151643,151643)
    monkeypatch.setattr(launcher_data,'_tokenizer_reference',lambda c:(ref,{'manifest_content_sha256':digest({})},file_hash(Path(ref.manifest_path))))
    contract=resolve_streaming_contract(cfg,prepare=True)
    spec=launcher_jobs.expand_job_specs(cfg,contract)[0]
    continual=launcher_jobs.build_continual_job_config(cfg,spec)
    probe=launcher_jobs.build_probe_job_config(cfg,spec,cycle_index=0,
        source_checkpoint=tmp_path/'not-yet-produced.pt', source_checkpoint_sha256='1'*64)
    assert continual.tasks[-1].train_source.kind == probe.train_source.kind == 'streaming_packed'
    assert continual.tasks[8].train_sequence_offset_count == 6
    assert not list(Path(contract['streaming_root']).glob('cache/*.bin'))
    assert probe.validation_source.packed.tokenizer.maximum_emitted_token_id == 151668


def _production_ddp_worker(rank, port, config_path, result_path, resume, stop):
    import os
    from lm_cl.config import load_continual_config
    from lm_cl.training.distributed import DistributedContext
    from lm_cl.training.distributed_continual import DistributedContinualTrainer
    os.environ.update(RANK=str(rank),LOCAL_RANK=str(rank),WORLD_SIZE='2',LOCAL_WORLD_SIZE='2',
                      MASTER_ADDR='127.0.0.1',MASTER_PORT=str(port))
    torch.set_num_threads(1)
    cfg=load_continual_config(config_path)
    ctx=DistributedContext.initialize(cfg.distributed,runtime_device='cpu')
    try:
        result=DistributedContinualTrainer(cfg,ctx).run(resume_checkpoint=resume,stop_after_global_logical_batches=stop)
        if rank == 0: Path(result_path).write_text(json.dumps({'checkpoint':result.checkpoint_path}))
    finally:ctx.close()


def test_two_rank_production_resume_after_cache_eviction(tmp_path,monkeypatch):
    import socket
    import torch.multiprocessing as mp
    from lm_cl.config import DistributedConfig,save_continual_config
    cfg,contract=study(tmp_path,monkeypatch)
    spec=launcher_jobs.expand_job_specs(cfg,contract)[1]
    continual=launcher_jobs.build_continual_job_config(cfg,spec)
    continual=replace(continual,tasks=continual.tasks[:1],distributed=DistributedConfig(True,'gloo',30,
        'contiguous_floor_v1','ddp_average_world_scaled_global_sum_v1',
        'sum_unscale_normalize_clip_rank0_broadcast_v1',False,False,True,False))
    root=Path(contract['streaming_root'])
    def run(config,tag,resume=None,stop=None):
        path=tmp_path/f'{tag}.yaml';save_continual_config(config,path)
        result=tmp_path/f'{tag}.json'
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        with worker(root):
            mp.spawn(_production_ddp_worker,args=(port,str(path),str(result),resume,stop),nprocs=2,join=True)
        return json.loads(result.read_text())['checkpoint']
    first=run(continual,'ddp-partial',stop=1)
    for p in (root/'cache').glob('*.bin'):p.unlink()
    resumed=run(continual,'ddp-resumed',resume=first)
    reference=replace(continual,runtime=replace(continual.runtime,output_dir=str(tmp_path/'ddp-fresh')))
    full=run(reference,'ddp-fresh')
    left,right=load_checkpoint(resumed),load_checkpoint(full)
    for k in left['model_state']:assert torch.equal(left['model_state'][k],right['model_state'][k])
    assert left['trainer_state'] == right['trainer_state']
    assert left['distributed_state']['world_size'] == 2


def test_range_backed_parquet_reads_only_columns_and_no_disk_cache(tmp_path):
    pa=pytest.importorskip('pyarrow')
    import pyarrow.parquet as pq
    from lm_cl.data.streaming_producer import Rows as ParquetRows
    from lm_cl.data.incremental_remote import REVISION
    from test_incremental_remote import LocalRanges
    import os
    rows=[{'text':f'document {i}', 'url':str(i),'unused':os.urandom(65536)} for i in range(16)]
    path=tmp_path/'source.parquet';pq.write_table(pa.Table.from_pylist(rows),path,row_group_size=4)
    identity={'repo_id':'uonlp/CulturaX','revision':REVISION,'language_config':'en','columns':['text','url'],
              'files':[{'path':'en/test.parquet','size':path.stat().st_size}]}
    remote=LocalRanges(path,tmp_path)
    source=ParquetRows(identity,remote)
    for i in [0,10,2,15]:assert source.row(i)=={'text':rows[i]['text'],'url':rows[i]['url']}
    assert remote.network_bytes < path.stat().st_size//2
    assert list((tmp_path/'source-cache').iterdir()) == []


def test_owned_cache_rejects_symlinks_and_collects_only_its_orphan_temporary(tmp_path,monkeypatch):
    _,contract=study(tmp_path,monkeypatch)
    root=Path(contract['streaming_root']);engine=ProductionProducer(root,source_factory=Rows,tokenizer=Tokenizer())
    outside=tmp_path/'user-file';outside.write_bytes(b'important')
    cache_path(root,0).symlink_to(outside)
    with pytest.raises(ValueError,match='cache entry'):engine.generate(0)
    assert outside.read_bytes()==b'important'
    cache_path(root,0).unlink()
    (root/'cache/00000001.tmp').write_bytes(b'x'*192)
    engine.generate(0)
    assert not (root/'cache/00000001.tmp').exists()
    assert sum(p.stat().st_size for p in (root/'cache').iterdir()) <= 192


def test_crash_between_receipt_commit_and_cache_publication_recovers(tmp_path,monkeypatch):
    import lm_cl.data.streaming_producer as module
    _,contract=study(tmp_path,monkeypatch)
    root=Path(contract['streaming_root'])
    original=module.publish_cache
    def crash(*args):raise InterruptedError('injected after commit')
    monkeypatch.setattr(module,'publish_cache',crash)
    with pytest.raises(InterruptedError):ProductionProducer(root,source_factory=Rows,tokenizer=Tokenizer()).generate(0)
    committed=receipt(root,0)[1]
    assert not cache_path(root,0).exists()
    monkeypatch.setattr(module,'publish_cache',original)
    ProductionProducer(root,source_factory=Rows,tokenizer=Tokenizer()).ensure(0)
    assert receipt(root,0)[1] == committed
    assert cache_path(root,0).stat().st_size == 64


def test_one_token_final_remainder_is_explicit_eos_truncation(tmp_path,monkeypatch):
    cfg,_=study(tmp_path,monkeypatch,cycles=1)
    cfg=replace(cfg,experiment=replace(cfg.experiment,tokens_per_task=8))
    contract=resolve_streaming_contract(cfg,prepare=True)
    class SixTokens(Tokenizer):
        def encode(self,text,**kwargs):return super().encode(text,**kwargs)[:6]
    root=Path(contract['streaming_root'])
    ProductionProducer(root,source_factory=Rows,tokenizer=SixTokens()).generate(0)
    record,_=receipt(root,0)
    assert record['boundaries'][-1]['content_token_count'] == 0
    assert record['boundaries'][-1]['truncated']
    assert np.frombuffer(cache_path(root,0).read_bytes(),dtype='<u4').tolist()[-2:] == [15,15]


def test_launcher_supervises_only_its_producer_and_restart_recovers(tmp_path,monkeypatch):
    import os
    import subprocess
    import sys
    from lm_cl.launcher.streaming import producer_service
    _,contract=study(tmp_path,monkeypatch)
    root=Path(contract['streaming_root'])
    original=subprocess.Popen
    repository=Path(__file__).resolve().parents[1]
    def offline_producer(command,**kwargs):
        assert command[1:3] == ['-m','lm_cl.data.streaming_producer']
        script=('import sys; from test_production_streaming import Rows, Tokenizer; '
                'from lm_cl.data.streaming_producer import ProductionProducer; '
                'ProductionProducer(sys.argv[1],source_factory=Rows,tokenizer=Tokenizer()).serve()')
        environment=dict(os.environ,PYTHONPATH=f'{repository}/src:{repository}/tests')
        return original([sys.executable,'-c',script,command[3]],env=environment,**kwargs)
    monkeypatch.setattr(subprocess,'Popen',offline_producer)
    with producer_service(contract):
        first=next(StreamingPackedSource(root,'train-en').iter_batches(sequence_length=8,global_sequences_per_batch=2))
        with pytest.raises(BlockingIOError):
            with producer_service(contract):pass
    for path in (root/'cache').glob('*.bin'):path.unlink()
    with pytest.raises(RuntimeError,match='producer exited'):
        next(StreamingPackedSource(root,'train-en').iter_batches(sequence_length=8,global_sequences_per_batch=2))
    with producer_service(contract):
        recovered=next(StreamingPackedSource(root,'train-en').iter_batches(sequence_length=8,global_sequences_per_batch=2))
    np.testing.assert_array_equal(first.input_ids,recovered.input_ids)


def test_calibration_uses_its_own_recipe_and_legacy_needs_no_other_pools(tmp_path,monkeypatch):
    from lm_cl.cli.a100_run import calibration_contract
    from types import SimpleNamespace
    cfg,contract=study(tmp_path,monkeypatch)
    spec=launcher_jobs.expand_job_specs(cfg,contract)[0]
    measured=launcher_jobs.build_continual_job_config(cfg,spec)
    assert calibration_contract(measured,'streaming')['streaming_root']==contract['streaming_root']
    with pytest.raises(ValueError,match='source kind'):calibration_contract(measured,'packed')
    legacy=SimpleNamespace(tasks=[SimpleNamespace(train_source=SimpleNamespace(kind='packed_shards'))])
    assert calibration_contract(legacy,'packed') == {'mode':'packed'}


def test_unknown_policy_and_noncontiguous_plan_fail_before_publication(tmp_path,monkeypatch):
    import copy
    from lm_cl.launcher.streaming import _plan
    from lm_cl.data.streaming import initialize
    cfg,_=study(tmp_path,monkeypatch)
    plan,_=_plan(cfg)
    bad=copy.deepcopy(plan);bad['policy']='unknown'
    with pytest.raises(ValueError,match='policy'):initialize(tmp_path/'bad-policy',bad)
    assert not (tmp_path/'bad-policy').exists()
    bad=copy.deepcopy(plan);bad['format']='future-version'
    with pytest.raises(ValueError,match='policy/version'):initialize(tmp_path/'bad-format',bad)
    assert not (tmp_path/'bad-format').exists()
    bad=copy.deepcopy(plan);bad['blocks'][1]['start']+=8
    with pytest.raises(ValueError,match='coverage'):initialize(tmp_path/'bad-blocks',bad)
    assert not (tmp_path/'bad-blocks').exists()
