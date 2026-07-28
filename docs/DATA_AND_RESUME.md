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

The scaled five-cycle 1B presets are different: they reuse five disjoint
sequence windows inside each already materialized cycle-0 5B language stage.
They create no additional continual-training token files. Only eight fixed
1,280-sequence held-out language-validation stages are added (80 MiB of packed
uint32 tokens total), plus manifests and boundary metadata. The Vietnamese
probe reuses a 1B prefix of the existing frozen 5B probe-training stage.

The preparation CLI normally runs one isolated materialization subprocess per
missing stage. With `--parallel-languages N`, it instead runs up to `N`
independent language lanes. Cycles remain sequential inside each lane, so later
appearances reject earlier documents exactly as before. Each lane has an
independent overlap registry and lock. After every stage completes, preparation
merges the lanes into the global registry, verifies exact per-stage row counts,
and fails on any non-identical content-hash or token-hash collision. A
hash-bound `parallel-preparation-audit.json` is mandatory before lane-produced
manifests can enter a resolved experiment. Existing stages are never fetched
again, and the launcher remains offline after validation.

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
export LM_CL_STREAM_RESHARD_ROW_GROUPS=false
export LM_CL_STREAM_RESHARD_FILE_PREFIX=32
export LM_CL_STREAM_PREFETCH_SHARDS=4
export LM_CL_STREAM_PREFETCH_ROWS_PER_SHARD=256
export LM_CL_MATERIALIZATION_CHECKPOINT_CANDIDATES=100000
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

With `LM_CL_STREAM_RESHARD_ROW_GROUPS=true`, supported parquet streams are
deterministically expanded from file shards to their contiguous row groups
before prefetch. This uses the installed `datasets` public `reshard()` contract:
concatenating the expanded shards reconstructs the original row order. A large
per-shard row queue lets later row groups finish remote reads while the current
group is consumed. `250000` rows across 16 workers can retain tens of GiB of raw
text and Python objects, so that setting is intended only for the documented
1.5 TiB preparation host. The first 32 original parquet files are intended to
cover the configured 30,000,000-document ceiling and avoid opening metadata for
the entire multi-terabyte language configuration; failure to reach the exact
token target still fails the stage. The CulturaX remote backend did not scale
under concurrent row-group range reads on the documented host, so parallel
language preparation explicitly disables this option. Defaults remain
row-group resharding off and 64 queued rows per shard.

`LM_CL_MATERIALIZATION_CHECKPOINT_CANDIDATES` optionally increases the number
of selected candidates between durable materialization checkpoints. The
configured stage value remains the default. A larger override amortizes costly
`fsync` operations on shared filesystems at the cost of replaying more source
rows after an interruption. Pending registry rows, packed tokens, and boundary
records become durable together at the next checkpoint; accepted order, final
packed bytes, hashes, and manifest identity are unchanged. The value is bounded
to one million candidates.

An incomplete stage produced by public release commit `56c2f08` is eligible
for one explicit engine-v2 resume migration. The old configuration fingerprint
is recomputed and must match before migration; no arbitrary source mismatch is
accepted. The final manifest preserves the prior software record and records
that ordering and packing semantics were unchanged. Complete manifests are
never rewritten. Incomplete engine-v2 stages from commit `a318fe3` are likewise
eligible for the bounded contiguous-shard-prefetch migration. Incomplete stages
from commit `82009c6` can migrate to the amortized-checkpoint engine under the
same exact-identity requirement. Incomplete stages from commit `85adf1a` can
migrate to contiguous parquet row-group prefetch. An increasing
`max_input_documents` execution ceiling is accepted during these migrations
only when every other frozen configuration value matches; the old and new caps
are recorded in the migration evidence. Incomplete stages from commit `9f3a94a`
can migrate into a named language-lane registry; their checkpointed boundaries
are replayed into that lane before new documents are accepted.

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
