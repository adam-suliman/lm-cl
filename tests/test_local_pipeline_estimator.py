from lm_cl.diagnostics.local_pipeline_estimator import workload_counts, retention_passes, simulate_overlap

def test_production_budget_and_evaluation_schedule():
    c=workload_counts(1_000_000_000,sequence_length=2048,global_batch=256,k=2,
                     interval=95,milestones=[1,2,4,8,16,32,64,95])
    assert c["effective_input_tokens"]==999_999_488
    assert c["effective_valid_targets"]==999_511_207
    assert c["logical_batches"]==1908
    assert c["last_batch_sequences"]==89
    assert c["slow_updates"]==954
    assert c["evaluation_points"]==29
    assert c["evaluation_steps"][-2:]==[1900,1908]

def test_retention_reuses_current_reset_and_adds_current_carried_only():
    languages=list("abcdefgh")
    assert retention_passes(languages,1,False)==36
    assert retention_passes(languages,5,False)==292
    assert retention_passes(languages,5,True)==332

def test_slow_producer_and_fast_producer_have_analytical_completion_times():
    slow=simulate_overlap(100,producer_rate=1,consumer_rate=10,block_tokens=10,
                          startup_seconds=2,lookahead_tokens=100)
    assert slow["elapsed_seconds"]==103
    fast=simulate_overlap(100,producer_rate=10,consumer_rate=1,block_tokens=10,
                          startup_seconds=2,lookahead_tokens=100)
    assert fast["elapsed_seconds"]==103
    assert fast["data_wait_seconds"]==3

def test_bounded_lookahead_keeps_complete_consumption_timeline():
    result=simulate_overlap(100,producer_rate=10,consumer_rate=1,block_tokens=10,
                            startup_seconds=2,lookahead_tokens=20)
    consumed=[r for r in result["timeline"] if r["event"]=="consumed"]
    assert [r["tokens"] for r in consumed]==list(range(10,101,10))
    assert result["elapsed_seconds"]==103
    assert result["producer_done_seconds"]>12

def test_targeted_saved_example_reader_does_not_confuse_other_sections(tmp_path):
    import json
    from lm_cl.diagnostics.local_pipeline_tokenizer_timing import saved_examples
    path=tmp_path/"manifest.json"
    path.write_text(json.dumps({"detector":{"en":{"calibration":[{"token_ids":[99]}]}},
        "languages":{l:{"calibration":[{"token_ids":[i],"sources":[{"nested":True}]} for i in range(4)],
                        "primary":[{"token_ids":[99]}]} for l in ["en","vi"]},
        "later":{"en":{}}},indent=2)+"\n")
    values=saved_examples(path,["en","vi"],2)
    assert [[r["token_ids"] for r in values[l]] for l in ["en","vi"]]==[[[0],[1]],[[0],[1]]]

def test_shared_producer_uses_maximum_matching_cursor_not_global_counter(tmp_path):
    import json
    from types import SimpleNamespace
    from pathlib import Path
    from lm_cl.cli.local_pipeline_demo import shared_consumed_prefix
    root=tmp_path/"stream"
    def row(offset,identity="recipe",source=root):
        return {"source_root":str(source),"source_recipe_sha256":identity,
                "source_position":{"shard_index":0,"token_offset":offset},"global_input_tokens":999999}
    a=tmp_path/"a.jsonl";b=tmp_path/"b.jsonl"
    a.write_text(json.dumps(row(30))+"\n"+json.dumps(row(999,source=tmp_path/"other"))+"\n")
    b.write_text(json.dumps(row(20))+"\n"+json.dumps(row(999,identity="wrong"))+"\n"+'{"unfinished":')
    limits=SimpleNamespace(owned=lambda p:Path(p))
    assert shared_consumed_prefix([a,b],root,"recipe",limits)==30
