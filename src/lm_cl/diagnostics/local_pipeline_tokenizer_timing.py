"""Nine-language tokenizer-only timings from bounded, detokenized saved examples.

This cannot measure original document selection, downloads or sustained block
publication. It provides an explicitly separate local instrument comparison.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import time

from lm_cl.data.incremental import file_hash, immutable_write, json_bytes
from lm_cl.diagnostics.local_pipeline_resources import Limits


def saved_examples(path, languages, count):
    result={language:[] for language in languages}
    inside=False;language=None;section=None;record=None
    with path.open() as stream:
        for line in stream:
            if line=='  "languages": {\n':inside=True;continue
            if not inside:continue
            if re.match(r'^  "[^\"]+":',line):break
            match=re.match(r'^    "([^\"]+)": \{',line)
            if match:language=match[1];section=None
            match=re.match(r'^      "([^\"]+)": \[',line)
            if match:section=match[1]
            if language not in result or section!="calibration" or len(result[language])>=count:continue
            if line=='        {\n':record=[]
            if record is not None:
                record.append(line)
                if line.rstrip() in {'        },','        }'}:
                    value=json.loads(''.join(record).rstrip().rstrip(','));record=None
                    result[language].append(value)
                    if all(len(v)==count for v in result.values()):break
    if any(len(v)!=count for v in result.values()):raise ValueError("Saved diagnostic source lacks requested bounded examples")
    return result


def run(settings_path,limits):
    from lm_cl.cli.local_pipeline_demo import _tokenizer_reference
    from lm_cl.data.tokenizer import load_verified_tokenizer
    settings=json.loads(Path(settings_path).read_text())
    expected={"schema_version","dataset_manifest","dataset_file_sha256","tokenizer_manifest","languages",
              "examples_per_language","repetitions","warmup_repetitions","output"}
    if set(settings)!=expected or settings["schema_version"]!=1:raise ValueError("Unknown tokenizer timing settings")
    if not 1<=settings["examples_per_language"]<=256 or not 2<=settings["repetitions"]<=8 or settings["warmup_repetitions"]!=1:
        raise ValueError("Invalid bounded tokenizer timing repetitions")
    if set(settings["languages"])!={"en","zh","fr","ja","es","de","pt","ru","vi"}:
        raise ValueError("Tokenizer timing requires explicit nine-language identity; zh maps to written Chinese")
    path=Path(settings["dataset_manifest"]);output=limits.owned(settings["output"])
    if output.exists():raise FileExistsError("Timing output already exists")
    with limits.process("tokenizer-timing","saved-nine-language-examples"):
        before=file_hash(path)
        if before!=settings["dataset_file_sha256"]:raise ValueError("Diagnostic source hash differs")
        examples=saved_examples(path,settings["languages"],settings["examples_per_language"])
        tokenizer,manifest=load_verified_tokenizer(_tokenizer_reference(settings["tokenizer_manifest"]))
        records=[]
        for language,items in examples.items():
            strings=[tokenizer.decode(item["token_ids"],skip_special_tokens=False,clean_up_tokenization_spaces=False) for item in items]
            expected_ids=None
            for mode in ["individual_document","batch_documents"]:
                for repetition in range(settings["warmup_repetitions"]+settings["repetitions"]):
                    limits.check();tick=time.perf_counter()
                    if mode=="individual_document":ids=[tokenizer.encode(s,add_special_tokens=False) for s in strings]
                    else:ids=tokenizer(strings,add_special_tokens=False,padding=False,truncation=False)["input_ids"]
                    elapsed=time.perf_counter()-tick
                    if expected_ids is None:expected_ids=ids
                    if ids!=expected_ids:raise RuntimeError("Batched tokenizer IDs differ from ordered individual encoding")
                    records.append({"language":language,"mode":mode,"repetition":repetition,
                        "measured":repetition>=settings["warmup_repetitions"],"seconds":elapsed,
                        "documents":len(strings),"input_utf8_bytes":sum(len(s.encode()) for s in strings),
                        "output_tokens":sum(len(v) for v in ids),"tokens_per_second":sum(len(v) for v in ids)/elapsed,
                        "ids_sha256":hashlib.sha256(json_bytes(ids)).hexdigest()})
        if file_hash(path)!=before:raise RuntimeError("Diagnostic source changed during tokenizer benchmark")
        result={"schema_version":1,"status":"complete","records":records,
                "source_path":str(path),"source_file_sha256":before,"settings_sha256":file_hash(Path(settings_path)),
                "tokenizer_manifest_sha256":manifest["manifest_content_sha256"],
                "interpretation":"Tokenizer-only, decoded diagnostic snippets, repeated cache-warm strings. Not raw document preprocessing or download throughput.",
                "batch_individual_ids_equal":True,"original_source_unchanged":True}
        immutable_write(output,json_bytes(result))
        print(json.dumps({"status":"complete","output":str(output),"records":len(records)}))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--limits",required=True);p.add_argument("--settings",required=True)
    args=p.parse_args();limits=Limits(args.limits)
    from lm_cl.cli.local_pipeline_demo import _deadline
    _deadline(limits.v["max_prepare_process_seconds"])
    os.environ["RAYON_NUM_THREADS"]=str(limits.v["cpu_threads_producer"])
    os.environ["TOKENIZERS_PARALLELISM"]="true"
    run(args.settings,limits)

if __name__=="__main__":main()
