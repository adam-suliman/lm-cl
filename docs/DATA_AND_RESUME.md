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
