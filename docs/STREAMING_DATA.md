# Production streaming data contract (v1)

This document describes the default independent schedule. The optional
`--streaming-schedule alternating` uses a separate versioned storage contract,
compact receipts, pinned reusable data, and checkpoint-acknowledged release;
see [ALTERNATING_STREAMING.md](ALTERNATING_STREAMING.md). Existing v1 recipes and
their reconstruction behavior remain supported without migration.

The `streaming` launcher mode is a new production route. The network-facing CPU
producer emits completed, checksum-verified uint32 blocks; continual/probe
trainers consume only those blocks through `streaming_packed` sources. Neither
trainer directly iterates a network dataset. Legacy `packed_shards` manifests,
validation, checkpoints and the earlier diagnostic/demo checkpoint kind retain
their original meanings.

## Frozen inputs and scientific identity

`study.json` binds the CulturaX commit
`6a8734bc69fefcbb7735f4f9250f43e4cd7a442e`, explicit language mappings (written
Chinese is `zh`), the inspected Qwen tokenizer and its manifest hash, document
and split seeds, shuffle buffer, sequence length, block size, all task/probe/
validation budgets, and bounded resource settings. The study's SHA-256 appears
in source references, resolved configurations, scientific job identity,
checkpoints and block receipts. `stream.json` is a recipe reference, **not** a
fabricated completed legacy manifest. Its stage name contains the full recipe
hash. Launch verifies these bindings; each actual consumed block has a completed
receipt plus verified bytes. Pending legacy manifests remain invalid.

Publication order is deterministic: each language's current training window,
then its first fixed retention-validation set; Vietnamese validation and the
training prefix follow cycle one's Russian window. Later cycles reuse the same
validation and Vietnamese data. Each continual language has one source stream
with disjoint sequence windows and persistent shuffle/pending-document state.
There is no invented EOS at block/window boundaries. At the final end of a
stream, documents are truncated to the exact budget with EOS. A one-token
remainder records an EOS-only truncated final document explicitly.

Global document and token-sequence hashes reject duplicates. Ownership follows
this **interleaved** order. This differs from historical all-pools-first packing
and is explicitly a new data identity, although corpus, language order,
selection/splitting algorithms, tokenizer and training budgets are matched.
Compare fresh variants/seeds sharing this study. Do not pool their results with
historical H100 data as if the exact examples were identical. A second model
size with identical data settings can reuse the same recipe sequentially.

Model sizes, tied embeddings, global-batch target weighting, partial-K windows,
manual active-memory updates, task resets, DDP reductions, and derived-probe
initialization/CE-AUC semantics are not changed by the adapter.

## Download, storage and publication

The producer discovers a lexicographically ordered Parquet file inventory at
the immutable revision, using bounded/paginated metadata requests. It saves the
inventory and its hash. HTTP byte-range reads fetch `text`/`url` column chunks;
no persistent raw Parquet cache or whole-dataset download is required. PyArrow
holds bounded decoded row groups in memory. A row group may still be much
larger than one training block, and library decompression/Python object overhead
can exceed the compressed range size. The shuffle buffer must initially fill.
Thus first-batch startup is small relative to whole-corpus packing, **not** a
guarantee of instantaneous startup or low RAM on every source file.

One locked producer serves all jobs/ranks in one launcher. Requests are atomic;
readers receive only complete batches (except the legitimate final task tail).
Prefetch stays within the current task window. A file lock protects byte-copying
and LRU eviction. The cache cap includes owned orphan temporary block files;
symlinks/unrecognized entries are refused. Existing packed datasets, external
repositories, checkpoints, and historical outputs are never eviction targets.

A SQLite transaction stores each immutable receipt, source cursor, shuffle RNG
and index buffer, pending-document tokens/offset, boundaries, and dedup hashes.
Receipts form a hash chain. A block is usable only when its receipt and atomic
cache publication both exist and the byte count/SHA match. A crash between the
transaction and cache publication is recoverable through regeneration.
Per-stream completion records and the final data `complete.json` record that
all required receipts exist; cached bytes may subsequently be evicted. These
records do not certify training completion.

Metadata and checkpoint growth are separate from the token-cache bound.
`receipts.sqlite`, source indices and completion records must be retained.
The metadata budget, per-document/range sizes, file/request counts, free-space
floor, request retry/timeout, block deadline and reader wait are explicit
resource limits. Reaching a limit fails the attempt; it never silently reduces
a task/probe budget, skips an oversized document, or treats unavailable data as
EOF. The deadline also covers source scanning/tokenization inside a block.
Choose caps before creating the study; editing a frozen plan invalidates it.

## Resume, recovery, and operating limits

Use the same launcher identity with `--resume auto`. Only the launcher's own
producer subprocess is supervised/stopped. A second supervisor for the same
study is rejected; independent model jobs belong in the same launcher. Producer
failure records are retained under unique names on restart. SIGKILL or power
loss can lose an unfinished block and unsaved GPU work, but not authorize
consumption of partial data.

To reconstruct an evicted block, restore the preceding state in its language/
role stream and re-read immutable source ranges. The dedup query ignores entries
created at that block's ordinal or later, recreating the original ownership
state. The complete regenerated receipt (including source inventory identity,
boundaries and producer state) and token SHA must match the original. A mismatch
fails; the old receipt is not replaced. Temporary cache writes are atomic.

Continual and probe checkpoints bind every receipt covering the consumed input
prefix; probes additionally bind the fixed validation prefix. Resume validates
these proofs, source/configuration identity and the existing world-size/partition
rules. The producer can have prepared more than the GPU consumed; this does not
advance the checkpoint's position. All old checkpoint/model/memory invariants
remain enforced. Evicted data are reconstructible only while the pinned remote
revision and local tokenizer snapshot remain available. This mode is not an
archive that supports guaranteed offline reruns.

Retention evaluation and repeated Vietnamese probes may revisit old blocks.
Concurrent variants with different speeds may also require reconstruction.
A larger cache reduces this re-download/re-tokenization cost; the bounded cache
does not guarantee that a slow network/CPU can saturate two A100s. Producer
status records per-block time, whether reconstruction occurred, and downloaded
bytes. GPU metrics/calibration include data waits in wall time. Benchmark the
actual layout; do not extrapolate across GPUs solely by card count.

The full horizon is frozen when the study is created. This version supports
exact resume **within that horizon**, not appending cycles to the same recipe
or migrating historical packed runs to streaming. Changing resource/scientific
settings creates a new recipe; it is not an in-place resume amendment.

## Verification scope

`tests/test_production_streaming.py` exercises the actual production
continual/probe trainers using tiny offline fixtures: early start before future
blocks exist, pending-document reconstruction, bounded eviction, checksum and
ownership failures, matched interrupted/uninterrupted AG-RMT weights, two
cycles for Transformer/AG-RMT with retention and derived probes, and two-rank
CPU/Gloo resume. A local Parquet fixture tests column/range reads without a raw
disk cache. Real-tokenizer schema checks construct configuration only. Tiny
probe training substitutes a 16-token test vocabulary for the fixed production
vocabulary; it does not train 5M/12M models. No A100/CUDA throughput or live
CulturaX availability is established by these offline tests.
