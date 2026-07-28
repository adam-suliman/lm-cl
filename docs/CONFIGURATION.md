# Configuration contract

The launcher rejects unknown keys. CLI overrides are applied before validation
and recorded in each resolved job configuration and SHA-256.

## Experiment

`models` accepts only `transformer` and `fastmem_rmt`. `model_size` is `5m` or
`12m` for production; `tiny` is restricted to functional synthetic smoke
tests. `languages` and `probe_schedule: cycle_end_after_ru` are fixed.
`resume` is `never`, `auto`, or `required`.

The production architecture files preserve the approved counts:

| Name | Layers | Width | Heads | Non-embedding | Total with tied vocabulary |
|---|---:|---:|---:|---:|---:|
| 5m | 4 | 320 | 5 | 4,932,480 | 54,125,440 |
| 12m | 5 | 448 | 7 | 12,072,256 | 80,942,400 |

The apparent total-count difference is the fixed 151,680-row tied embedding
table and 2,048-row absolute-position table. Do not change the vocabulary to
match tokenizer length.

## Exact budget

The only public policy is `floor_complete_sequences_v1`:

```text
complete_sequences = floor(requested_input_tokens / sequence_length)
effective_input_tokens = complete_sequences * sequence_length
effective_valid_targets = complete_sequences * (sequence_length - 1)
```

For five billion requested tokens at length 2,048, the job consumes exactly
2,441,406 complete sequences, 4,999,999,488 input tokens, and 4,997,558,082
valid targets; the 512-token remainder is recorded and never cropped from an
arbitrary position. A partial final logical batch is allowed and globally
target-normalized.

## Data

`cycle_manifest_policy` is either `fresh_disjoint_v1` or the explicitly scaled
`disjoint_sequence_windows_v1`. `manifest_template` may use `{cycle}`,
`{language}`, `{task_index}`, `{source_task_index}`, and `{effective_tokens}`. Each
rendered path, manifest-file SHA-256, manifest-content SHA-256, ordered-data
SHA-256, tokenizer identity, and requested/effective count is frozen in the
resolved cycle×language matrix.

The windowed policy materializes or opens one cycle-0 source per language whose
requested budget defaults to `cycles * tokens_per_task`. An explicit
`window_source_tokens_per_task` may retain a larger already frozen source when
running a shorter logical horizon; it must contain every configured disjoint
window. Logical cycle `c` reads the
half-open complete-sequence interval
`[c * task_sequences, (c + 1) * task_sequences)`. The resolved contract records
the source hashes, exact range, and a distinct view SHA-256. Overlapping or
out-of-range windows fail before launch.

Enabled forgetting requires a `language_validation_manifest_template` and a
positive validation count divisible by the global logical batch. Validation
documents come from the stable held-out split. At every task boundary the model
is evaluated on all languages seen so far. Lower CE is better; per-language
forgetting is current CE minus the best CE previously observed for that
language. FastMem uses reset M0 for this metric.

The production `max_input_documents` value is 30,000,000. This is a fail-safe
execution ceiling, not a target: materialization stops as soon as the exact
token budget is reached. The larger ceiling is required because observed
English documents average fewer than 1,000 tokenizer tokens, making the former
5,000,000-document ceiling unable to reach the frozen five-billion-token
budget. Later cycles must additionally pass documents already registered by
earlier cycles; 30,000,000 leaves headroom for all three appearances.
Increasing an incomplete stage's ceiling is resume-compatible only through the
explicitly audited migration; decreasing it or changing any other selection
field remains an error.

The tokenizer contract is fixed at base vocabulary 151,643, effective length
151,669, maximum emitted ID 151,668, model embedding rows 151,680, and EOS/pad
ID 151,643. The final eleven embedding rows are unused padding.

## Training and FastMem

Production uses explicit FP32, FP16, or BF16. FP16 uses conditioning multiplier
2. The release never infers a different precision. AdamW is fixed to betas
(0.9, 0.95), epsilon 1e-8, weight decay 0.1, and five-percent task-local
warm-up. The Transformer steps every logical batch. FastMem averages slow
gradients over K=2, flushes a normalized tail, and advances its scheduler only
when AdamW steps.

FastMem memory vectors are direct hidden states. They receive no token,
absolute-position, type, or role embeddings. Text retains positions 0…2047.
The persistent active root is manual task-local state outside AdamW; learned M0
is a slow parameter. The explicit update is one-pass, target-normalized,
active-only clipped, detached, and synchronized once per global logical batch.
An explicit non-default positive `fastmem.fast_lr` is recorded as an ablation;
the release preset is 0.005.

## Launcher and tracking

`jobs_per_gpu` creates concurrent slots on each GPU. `gpus_per_job > 1` creates
disjoint groups and invokes the existing DDP implementation. Impossible or
overlapping assignments fail. Disk admission estimates every task-boundary,
cycle, and probe checkpoint because this package never silently deletes one.
`ddp_debug_assert_synced` performs full state hashing after every update and is
intended for smoke/debug gates; H100 production presets disable it while
retaining mandatory checkpoint-boundary state-digest verification. Expensive
full-model norm diagnostics and TensorBoard optimizer scalars follow
`tracking.log_every_batches`; JSONL loss and update records remain complete.

Training scalar steps are cumulative logical batches. Probe-curve steps are
probe cumulative input tokens in per-cycle TensorBoard subdirectories. Cycle
AUC summaries use one-based cycle numbers. The atomic TensorBoard sidecar
prevents duplicate scalar steps on resume.
