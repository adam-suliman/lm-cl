# Alternating streaming: implementation and operating contract

User-requested optional launch mode; the existing reconstructible streaming
mode remains the default and its recipe format stays unchanged. No model,
optimizer, active-memory, task-reset, probe or evaluation semantics change.
Verification uses tiny offline CPU/Gloo fixtures, not lengthy experiments.

The new versioned recipe binds every model/seed consumer and a maximum number
of global logical batches per turn. Each turn stops at that limit or the next
language boundary, whichever comes first. All consumers complete a turn before
the next turn opens. GPU slots may execute consumers concurrently; on one GPU
they alternate. A mid-task yield is an exact checkpoint, including partial-K
gradients and active memory; it is not a language reset. A language-end turn
finishes the tail update and evaluation before saving. Russian boundaries also
finish the original derived Vietnamese probe and cycle summary.

The default maximum is 2048 logical batches, so the default 1B-token task fits
in one turn. This avoids extra mid-task checkpoints at that budget. Smaller
turns are configurable and require additional retained checkpoints; their disk
cost must appear in preflight. No checkpoint is automatically deleted.

Vietnamese training and all fixed validation blocks are permanently pinned when
first produced. Continual blocks occupy a separately bounded queue. The producer
cannot advance beyond the open turn. Blocks leave that queue only after every
configured consumer acknowledges a checksum-valid checkpoint past them. A crash
before acknowledgment keeps data; a crash after acknowledgment resumes from the
acknowledged checkpoint. Pinned data, receipts and checkpoint files are never
queue eviction targets.

Block receipts retain token checksums, chain identity, boundary/state digests and
counts, not full per-document boundary lists or duplicated shuffle state. Exact
document/token deduplication uses binary SHA-256 keys with publication ordinals.
One compressed current producer state per stream supports crash recovery;
compressed historical states are saved at turn recovery points. Missing data
can be reconstructed from the preceding selected state by replaying immutable
source rows and checking every compact receipt. Full metadata remains bounded
by an explicit cap; no claim that 1 GiB is sufficient without measurement.

This is a new storage/recipe identity with the same source selection and token
ordering algorithm as the existing streaming route. Tests must compare token
bytes and model/probe outcomes, not merely successful exits. Old studies are not
migrated, pruned, or edited. The recipe, consumer set and resource limits remain
immutable for exact resume.

Required evidence: byte agreement with existing streaming; pinned retention;
queue bound and slow/failed-consumer protection; checkpoint-validated release;
crashes around receipt/cache/ack publication; sparse-state reconstruction;
partial-K and two-rank exact resume; two complete tiny paired cycles including
all retention evaluations and Vietnamese probes; legacy regression; CLI plans,
disk estimates and operator documentation. No live corpus or A100 timing is
claimed by these tests.

The implementation passed the requirement-level offline checks recorded in
[ALTERNATING_VERIFICATION.md](ALTERNATING_VERIFICATION.md). Select it through
either A100 entry script using `--streaming-schedule alternating`; the full
operator path and configuration reference are in [A100_CONTROLS.md](A100_CONTROLS.md).

Resume is coordinated for the same consumer group at its latest durable
positions. An older checkpoint remains usable as scientific evidence and a
probe source; the launcher does not rewind an acknowledged group to an older
turn. Sparse preparation states can verify/reconstruct history, but obsolete
requests cannot republish released queue blocks behind the group's watermark.
The consumer output paths are part of this new recipe identity. Do not reuse
its queue for another study or convert an existing independent recipe in place.

The 8 GiB alternating metadata default is a resource ceiling, not a measured
production requirement. The full receipt history now contains compact digests;
deduplication hashes still grow with accepted document count. A current recovery
head is overwritten per stream; historical detailed states are retained only at
the selected turn endpoints and pinned-stream endpoints. Reaching a cap fails
the attempt rather than removing provenance or silently changing data selection.
