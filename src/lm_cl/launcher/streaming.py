"""Freeze a production study before tokens exist; prepare only requested blocks."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from lm_cl.data.incremental import Recipe, FORMAT as RECIPE_FORMAT, digest
from lm_cl.data.incremental_remote import REVISION, MAPPINGS
from lm_cl.data.storage import atomic_write_json, ensure_owned_root, enforce_disk_limit
from lm_cl.data.streaming import FORMAT, initialize, load_plan, lock
from lm_cl.launcher.schema import PUBLIC_LANGUAGE_ORDER, resolve_token_budget


@dataclass(frozen=True)
class StreamingSettings:
    block_tokens: int = 1_048_576
    prefetch_blocks: int = 2
    token_cache_bytes: int = 4 * 1024**3
    metadata_bytes: int = 64 * 1024**3
    minimum_free_bytes: int = 20 * 1024**3
    wait_seconds: int = 3600
    max_block_seconds: int = 1800
    max_row_group_bytes: int = 256 * 1024**2
    max_source_files: int = 10000
    max_document_bytes: int = 4 * 1024**2
    max_network_requests: int = 10_000_000
    max_network_attempts_per_request: int = 3
    network_timeout_seconds: int = 60

    def validate(self, sequence_length):
        for key, value in asdict(self).items():
            if type(value) is not int or value < (0 if key in {"minimum_free_bytes", "prefetch_blocks"} else 1):
                raise ValueError(f"Invalid streaming.{key}")
        if self.block_tokens % sequence_length:
            raise ValueError("Streaming blocks must contain complete sequences")
        if self.token_cache_bytes < self.block_tokens * 4 * (self.prefetch_blocks + 2):
            raise ValueError("Token cache must fit prefetch plus two reader blocks")
        if self.wait_seconds <= self.max_block_seconds:
            raise ValueError("Consumer wait must exceed one block preparation deadline")


@dataclass(frozen=True)
class AlternatingSettings(StreamingSettings):
    metadata_bytes: int = 8 * 1024**3
    chunk_batches: int = 2048


def settings_from_mapping(mapping):
    values = dict(mapping or {})
    schedule = values.pop("schedule", "independent")
    retention = values.pop("checkpoint_retention", None)
    if schedule not in {"independent", "alternating"}:
        raise ValueError("Unknown streaming schedule")
    if retention not in {None, "cycle_end_v1"} or (retention and schedule != "alternating"):
        raise ValueError("Invalid streaming checkpoint-retention policy")
    return (AlternatingSettings if schedule == "alternating" else StreamingSettings)(**values)


def settings(config):
    value = settings_from_mapping(config.data.streaming)
    value.validate(config.experiment.sequence_length)
    return value


def _plan(config):
    from lm_cl.launcher.data import _tokenizer_reference
    ref, manifest, file_sha = _tokenizer_reference(config)
    opts = settings(config)
    length = config.experiment.sequence_length
    budget = lambda n: resolve_token_budget(n, length, policy=config.experiment.token_budget_policy)
    task, probe = budget(config.experiment.tokens_per_task), budget(config.probe.training_tokens)
    source = {"kind":"pinned_culturax_parquet_v1", "repo_id":"uonlp/CulturaX", "revision":REVISION,
              "mappings":MAPPINGS, "split":"train", "file_order":"lexicographic_repo_path_v1"}
    tokenizer = {**asdict(ref), "manifest_file_sha256":file_sha,
                 "manifest_content_sha256":manifest["manifest_content_sha256"]}
    streams, blocks, purposes = {}, [], {}
    def add(name, language, purpose, count, seed):
        recipe = Recipe(RECIPE_FORMAT, source, language, "validation" if "validation" in purpose else "train",
            tokenizer, config.data.max_input_documents, count, opts.block_tokens, length, seed,
            config.data.shuffle_buffer_documents, config.data.split_seed, config.data.validation_permyriad,
            ref.expected_eos_token_id, ref.maximum_emitted_token_id, "text", "url", [])
        recipe.validate()
        streams[name] = asdict(recipe); purposes[name] = purpose
    for i, language in enumerate(PUBLIC_LANGUAGE_ORDER):
        add(f"train-{language}", language, "continual_train", task.effective_input_tokens*config.experiment.cycles,
            config.data.document_order_seed+i)
        if config.forgetting and config.forgetting.enabled:
            add(f"validation-{language}", language, "language_validation",
                config.forgetting.validation_sequences_per_language*length, config.data.document_order_seed+i)
    if config.probe.enabled:
        add("validation-vi", "vi", "vietnamese_validation", config.probe.validation_sequences*length, config.data.document_order_seed+8)
        add("train-vi", "vi", "vietnamese_train", probe.effective_input_tokens, config.data.document_order_seed+8)
    def window(name, start, count):
        for offset in range(start, start+count, opts.block_tokens):
            blocks.append({"stream":name, "start":offset, "count":min(opts.block_tokens, start+count-offset)})
    for cycle in range(config.experiment.cycles):
        for language in PUBLIC_LANGUAGE_ORDER:
            window(f"train-{language}", cycle*task.effective_input_tokens, task.effective_input_tokens)
            if cycle == 0 and f"validation-{language}" in streams:
                window(f"validation-{language}", 0, streams[f"validation-{language}"]["output_tokens"])
        if cycle == 0 and config.probe.enabled:
            window("validation-vi", 0, streams["validation-vi"]["output_tokens"])
            window("train-vi", 0, streams["train-vi"]["output_tokens"])
    plan = {"format":FORMAT, "final_document_remainder":"eos_only_if_one_token_v1", "policy":"serial_interleaved_global_dedup_v1", "streams":streams,
        "purposes":purposes, "blocks":blocks, "limits":asdict(opts), "tokenizer_reference":asdict(ref),
        "tokenizer_manifest":ref.manifest_path, "task_budget":task.to_dict(), "probe_budget":probe.to_dict()}
    if config.data.streaming.get("checkpoint_retention") == "cycle_end_v1":
        plan["checkpoint_retention"] = "cycle_end_v1"
    if isinstance(opts, AlternatingSettings):
        from lm_cl.data.alternating import configure_plan
        configure_plan(plan, config)
    return plan, tokenizer


def resolve_streaming_contract(config, *, prepare=False):
    from lm_cl.launcher.data import materialization_config
    if config.data.cycle_manifest_policy != "disjoint_sequence_windows_v1":
        raise ValueError("Production streaming requires disjoint_sequence_windows_v1")
    plan, tokenizer = _plan(config)
    root = Path(config.data.generated_root).resolve() / "streaming" / digest(plan)
    if prepare:
        root.parent.mkdir(parents=True, exist_ok=True)
        with lock(root.parent / "initialize.lock"):
            plan = initialize(root, plan)
    else:
        expected = {**plan, "sha256":digest(plan)}
        plan = load_plan(root)
        if plan != expected: raise ValueError("Streaming recipe changed")
    identities = {}
    for name, spec in plan["streams"].items():
        stage_id = f"stream-{plan['sha256']}-{name}"
        stage = Path(config.data.generated_root).resolve() / "stages" / stage_id
        reference = {"root":str(root), "stream":name, "recipe_sha256":plan["sha256"]}
        path = stage / "stream.json"
        if prepare and not path.exists():
            stage.mkdir(parents=True, exist_ok=True)
            atomic_write_json(path, reference)
        if json.loads(path.read_text()) != reference: raise ValueError("Streaming stage reference changed")
        budget = resolve_token_budget(spec["output_tokens"], spec["sequence_length"], policy=config.experiment.token_budget_policy)
        pipeline, _ = materialization_config(config, cycle=0, language=spec["language"],
            task_index=(list(PUBLIC_LANGUAGE_ORDER).index(spec["language"]) if spec["language"] in PUBLIC_LANGUAGE_ORDER else 8), budget=budget, manifest_path_override=stage / "manifest.json", purpose=plan["purposes"][name])
        pipeline = replace(pipeline, mode="packed_shards", selection=replace(pipeline.selection,
            document_order_seed=spec["document_order_seed"]))
        # No fabricated completed legacy manifest: the source kind and stage ID
        # bind the full recipe hash; each actual block has a completed receipt.
        identities[name] = {"kind":"streaming_packed", "root":str(root), "stream":name,
            "recipe_sha256":plan["sha256"], "purpose":plan["purposes"][name], "pipeline":pipeline.to_dict()}
    matrix = []
    sequences = plan["task_budget"]["effective_complete_sequences"]
    for cycle in range(config.experiment.cycles):
        matrix.append({language:{**identities[f"train-{language}"], "sequence_window":{
            "sequence_start":cycle*sequences, "sequence_count":sequences,
            "sequence_end_exclusive":(cycle+1)*sequences}} for language in PUBLIC_LANGUAGE_ORDER})
    return {"mode":"streaming", "streaming_root":str(root), "streaming_recipe_sha256":plan["sha256"],
        "cycle_manifest_policy":config.data.cycle_manifest_policy,
        "task_token_budget":plan["task_budget"], "probe_token_budget":plan["probe_budget"],
        "data_manifests":matrix, "probe_training_manifest":identities.get("train-vi"),
        "probe_validation_manifest":identities.get("validation-vi"),
        "language_validation_manifests":{lang:identities[f"validation-{lang}"] for lang in PUBLIC_LANGUAGE_ORDER
                                         if f"validation-{lang}" in identities}, "tokenizer":tokenizer}


def prepare_streaming(config):
    """Inspect the small pinned tokenizer, freeze a recipe; download no corpus here."""
    from lm_cl.data.tokenizer import inspect_tokenizer
    from lm_cl.launcher.data import TOKENIZER_REPO_ID, TOKENIZER_REVISION
    if not os.environ.get("HF_HOME"): raise ValueError("Set an explicit authenticated HF_HOME")
    generated = ensure_owned_root(config.data.generated_root, purpose="generated-data")
    cache = ensure_owned_root(config.data.dataset_cache_root, purpose="hf-cache")
    enforce_disk_limit(cache, config.data.max_cache_bytes, label="Tokenizer cache")
    enforce_disk_limit(generated, config.data.max_generated_bytes, label="Generated data")
    path = Path(config.data.tokenizer_manifest)
    if not path.exists():
        path.resolve().relative_to(generated)
        inspect_tokenizer(repo_id=TOKENIZER_REPO_ID, revision=TOKENIZER_REVISION, cache_dir=cache,
                          output_manifest=path, model_embedding_vocab_size=151680)
    return resolve_streaming_contract(config, prepare=True)


@contextmanager
def producer_service(contract):
    """One supervised producer for all variants/seeds/ranks in this launcher."""
    if contract.get("mode") != "streaming":
        yield
        return
    root = Path(contract["streaming_root"])
    # No second launcher may evict this launcher's cache behind its supervisor.
    # Multiple GPUs/jobs within the launcher share this one service.
    with lock(root / "supervisor.lock", nonblocking=True):
        log_path = root / f"producer-{time.time_ns()}.log"
        with log_path.open("a") as log:
            process = subprocess.Popen([sys.executable, "-m", "lm_cl.data.streaming_producer", str(root)],
                                       stdout=log, stderr=log, start_new_session=True)
            try:
                deadline = time.monotonic()+15
                while True:
                    if process.poll() is not None: raise RuntimeError(f"Producer startup failed; inspect {log_path}")
                    ready = root / "producer-ready.json"
                    if ready.exists() and json.loads(ready.read_text()).get("pid") == process.pid: break
                    if time.monotonic() > deadline: raise TimeoutError("Producer startup timed out")
                    time.sleep(.05)
                yield
                if process.poll() is not None: raise RuntimeError(f"Producer exited; inspect {log_path}")
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try: process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL); process.wait(timeout=5)
