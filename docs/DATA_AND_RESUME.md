# Data preparation and resume

## Immutable inputs

The release pins CulturaX and Qwen3 revisions in source and validates them
against every packed manifest. The written-Chinese key `zh_written` maps to
CulturaX `zh`; no alternative is selected implicitly. Every stage uses `text`
and stable `url`, falling back to content SHA-256 only when `url` is empty.

For each cycle, prepare eight independent stages in fixed language order. A
two-cycle 5B experiment therefore needs sixteen continual stages, one reusable
5B Vietnamese probe-training stage, and one fixed 1,280-sequence Vietnamese
validation stage. Token payload alone is about 20 GB per 5B stage before
boundary metadata, manifests, cache, registry, checkpoints, and filesystem
reserve. The two-cycle preset needs roughly 340 GB of packed token payload;
five cycles need roughly 820 GB. The preset therefore caps owned generated
data at 1 TiB. Provision that capacity plus checkpoint/output and filesystem
reserve before preparing a five-cycle horizon.

The preparation CLI runs one isolated materialization subprocess per missing
stage. It uses a global overlap registry and distinct deterministic order seeds
to enforce fresh appearances. Existing stages are checksum-validated and are
never fetched again. Once all stages exist, the launcher operates offline.

Materialization engine v2 keeps that stage order but removes hot-loop storage
serialization. It tokenizes an ordered document batch through the fast
tokenizer, buffers packed writes until the existing resumable checkpoint,
commits the overlap registry once per checkpoint window, and amortizes
recursive cache-cap scans. SQLite remains `synchronous=FULL`; the transaction
is merely enlarged. A registry commit may be ahead of the atomic incomplete
manifest after a crash, which is safe because resume already truncates registry
rows, packed bytes, and boundary bytes back to the manifest checkpoint.

The generated-data limit remains strictly admitted before a stage: the
existing generated size plus the conservative maximum for all remaining token,
boundary, registry, manifest, and temporary bytes must fit. It is checked again
at finalization. The less predictable Hugging Face cache is checked during the
stream at a bounded five-second cadence and forced at stream close.

These execution settings do not enter packed scientific identity and may be
tuned without changing order or bytes:

```bash
export TOKENIZERS_PARALLELISM=true
export RAYON_NUM_THREADS="$(nproc)"
export LM_CL_TOKENIZER_BATCH_DOCUMENTS=2048
export LM_CL_REGISTRY_CACHE_MIB=4096
export LM_CL_REGISTRY_MMAP_MIB=65536
export LM_CL_STREAM_PREFETCH_SHARDS=16
export LM_CL_STREAM_PREFETCH_ROWS_PER_SHARD=256
```

Without an explicit batch size, the engine uses 16 documents per visible CPU,
bounded to 64–4,096. An explicit value must be in 1–16,384. Larger batches use
more transient RAM. Progress is emitted after every `checkpoint_every_candidates`
window with cumulative tokens, documents, elapsed time, and invocation
tokens/second.

The registry defaults to a 512 MiB SQLite page cache and a 4 GiB mmap window.
The larger values above are appropriate only on a high-memory machine; they do
not preallocate or persist additional project data and do not alter registry
contents.

When the installed `datasets` streaming object exposes multiple data-source
shards, the engine opens a bounded number concurrently and fills a bounded
per-shard row queue. Consumers still drain shard 0 completely before shard 1,
and so on. Hugging Face specifies that concatenating these contiguous shards
reconstructs the original dataset order; tests exercise both true concurrent
startup and exact concatenated output. Defaults are at most eight concurrent
shards and 64 queued rows per shard. The larger values above are intended for
this high-bandwidth, high-memory machine. A stream without the required shard
API uses the original serial iterator.

An incomplete stage produced by public release commit `56c2f08` is eligible
for one explicit engine-v2 resume migration. The old configuration fingerprint
is recomputed and must match before migration; no arbitrary source mismatch is
accepted. The final manifest preserves the prior software record and records
that ordering and packing semantics were unchanged. Complete manifests are
never rewritten. Incomplete engine-v2 stages from commit `a318fe3` are likewise
eligible for the bounded contiguous-shard-prefetch migration.

Use `--manifest-only` or `--manifest-only-preflight` only after a full shard
checksum validation has been recorded for those immutable files. These modes
validate manifest identities but deliberately avoid rereading every token bin.

## Resume pointer

`latest_checkpoint.json` is atomically replaced and contains a safe relative
path, checkpoint SHA-256, schema/kind, config SHA-256, completed task/cycle
counters, scientific identity, and requested horizon. Resume loads and checks
the payload; filename sorting is not trusted. If an interrupted child wrote a
newer exact-config checkpoint before updating the pointer, recovery compares
validated state counters and rejects ambiguous equal-progress payloads.

Cycle-end probes are derived runs. They hash the stable Russian boundary,
create fresh optimizer/scheduler/scaler/RNG/counters and active-from-M0 state,
write to `probes/cycle-NNNN`, verify the source hash afterward, and then return
to the untouched continual checkpoint. FastMem primary AUC uses the carried
full-system curve; reset remains the slow/backbone-comparability curve.

At each Russian boundary the launcher writes an augmented stable checkpoint
containing manifest identities, completed probe identities/summaries,
launcher/job/scientific identities, and the current horizon. Core model,
optimizer, scheduler, scaler, gradients, RNG, source position, M0, and active
state remain in the validated continual checkpoint schema.

## Horizon extension

An N→N+M extension is accepted only at a complete cycle boundary. Existing
task configs and manifest identities must be an exact prefix; future manifests
must already validate. The launcher migrates only checkpoint configuration
metadata to the longer horizon and preserves all scientific state bitwise.
The next English task creates a fresh task optimizer/scheduler and resets
FastMem active memory from current learned M0.

Lower horizons, changed completed manifests, model/seed changes, optimizer or
batch changes, precision changes, FastMem changes, probe changes, missing data,
checksum errors, and ambiguous pointers fail before training.

## Limitations

- Single-machine launch and existing single-node DDP only; no FSDP or
  multi-node rendezvous.
- No matched-K, Base RMT, FastMem-zero, or two-pass CIFAR public methods.
- Checkpoints are not automatically deleted. Retention requires a separate
  evidence-preserving proposal and explicit approval.
- The smoke configuration is synthetic and tiny. Only a correctly pinned,
  fully materialized run at the requested budgets can support a reproduction
  claim, and performance still depends on the documented paper-unstated
  assumptions and selected seeds.
