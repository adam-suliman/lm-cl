# 5M matched-update Transformer control

## Question and limits

Does AG-RMT's observed 5M advantage remain when the Transformer has the same
**slow-parameter update cadence**? The existing clean Transformer updates AdamW
after every logical batch (`K=1`); AG-RMT accumulates two logical batches per
slow update (`K=2`). The `backbone_matched_k` control is the same memory-free
Transformer architecture with `K=2`. It has no active memory or fast update.
AG-RMT still updates its task-local active memory after each logical batch;
that state is not an AdamW parameter. Thus this control removes the slow-update
cadence difference. A remaining AG-RMT advantage would not, by itself, prove
that the explicit memory update caused it.

The primary **paired** study runs four public variants together: clean
Transformer (`K=1`), matched-K Transformer (`K=2`), persistent-memory AG-RMT
with `fast_lr=0`, and full AG-RMT with `fast_lr=0.005`. The first contrast
tests slow-update cadence; the second asks whether AG-RMT still differs from
a cadence-matched memory-free Transformer; the third tests the nonzero fast
update within the same AG-RMT architecture. The separate `base_rmt` arm in
the five-variant preregistration in the workspace at
`results/multiseed-mechanism-validation/PREREGISTRATION.md` is not exposed by
this public A100 launcher, so the four-arm study does not
separate RMT recurrence from persistence of the memory root. A memory-specific
causal claim requires that remaining control and/or direct memory interventions.

The historical 5M Transformer/AG-RMT pair, seed `81010`, is useful context:
it was run with packed data, four H100 ranks per model and a physical
microbatch of 16 per rank. Its original packed shards are not present in this
workspace. The new A100 stream freezes a new source/tokenizer/data recipe, so
the old results cannot serve as identity-matched baselines for this study.
The `matched-only` option saves compute for an exploratory comparison against
them; the paired mode is the one to use for the stated scientific question.

## Frozen design

The script fixes 5M, one 80 GB A100 (GPU 0), BF16, physical microbatch 8,
global logical batch 256, sequence length 2048, the ordered eight continual
languages, five cycles, 1B requested input tokens per language per cycle,
one 1B-token Vietnamese probe at each cycle end, and the existing peak learning
rate/schedule. It defaults to seed `81010`; `--seeds` can select a larger
prespecified paired set. All arms share the same streaming recipe, fixed
validation pools, language order, source selection and seed within a paired
launch. The alternating schedule limits the continual token queue and pins
Vietnamese training and all fixed validation blocks. A turn ends at a language
boundary under the default 2048-batch limit.

The primary cycle-5 outcomes are (1) mean prior-language forgetting after
Russian, measured in reset-validation CE above each language's best previous
CE, and (2) normalized trapezoidal Vietnamese probe validation-CE AUC against
cumulative probe input tokens. Lower is better. Also inspect every cycle's
values, all individual-language CE trajectories, and the AG-RMT reset and
carried probe modes. The memory-free controls have a single
`not_applicable` memory-evaluation mode. Report absolute values and paired
differences, not a memory-causality claim from the system contrast alone.

The historical seed-81010 cycle-5 context is Transformer versus AG-RMT mean
prior-language forgetting **4.581 versus 3.547 CE** and primary Vietnamese
AUC **5.058 versus 3.543 CE**. Those are old-data/H100 values, not the new
A100 study's preregistered baselines. The workspace's `docs/LOCAL_HANDOFF.md`
and `results/multiseed-mechanism-validation/REPORT.md` record their provenance
and one-seed limit.

## Run on an authenticated A100 host

Set up the repository, CUDA environment and interactive Hugging Face login
using the [A100 runbook](A100_CONTROLS.md). Use dedicated storage locations;
keep the same roots, study name, model set, seed set, GPU layout and recipe
on resume. These commands **prepare and check** the study before starting it:

```bash
export HF_HOME=/workspace/lm-cl-auth
export LM_CL_DATA_ROOT=/workspace/lm-cl-data-matched-update
export LM_CL_OUTPUT_ROOT=/workspace/lm-cl-results
mkdir -p "$HF_HOME" "$LM_CL_DATA_ROOT" "$LM_CL_OUTPUT_ROOT"

./scripts/run_5m_matched_update_study.sh --action plan
./scripts/run_5m_matched_update_study.sh --action prepare
./scripts/run_5m_matched_update_study.sh --action preflight
```

After reviewing the plan and preflight, launch inside `tmux` or a scheduler
allocation:

```bash
./scripts/run_5m_matched_update_study.sh --action run
```

For the three preregistered seeds, pass `--seeds 81010,81011,81012` to **every**
action. For a cheaper, explicitly exploratory historical comparison, pass
`--arms matched-only` to every action; its default study name is separate.
After the original process has stopped, resume with the original arguments
plus `--resume auto --action run`.

The script now defaults to opt-in `--checkpoint-retention cycle`. On one
filesystem, its admission estimate is **112.90 GiB** for four arms at one seed,
**55.09 GiB** for one matched-K arm, or **267.08 GiB** for four arms across
three seeds. The all-checkpoints policy would reserve **249.95 GiB** for four
arms at one seed. These estimates include checkpoint files, a 4 GiB continual
queue, an 8 GiB metadata cap, about 3.813 GiB of pinned tokens, and a 20 GiB
free-space floor. Installation, tokenizer cache, logs and failed attempts need
additional space. Cycle retention records and retires older within-cycle
language checkpoints after all consumers have advanced. Russian probe sources,
augmented cycle checkpoints, completed probe checkpoints and the current
recovery point remain. To keep all checkpoints, pass `--checkpoint-retention
all` from the first `prepare` onward; it cannot change on resume.

## Verify and compare after completion

Require the launcher summary and each arm's complete per-job summary, 40
completed language tasks, 40 forgetting evaluations, five Vietnamese probes,
and valid completion/checkpoint hashes. The existing comparison tool verifies
paired identities and update counts for all four arms:

```bash
study="$LM_CL_OUTPUT_ROOT/5m-matched-update-paired-s81010-v1"
python -m lm_cl.cli.compare_control_experiments \
  --transformer "$study/transformer/seed-81010/summary.json" \
  --backbone-matched-k "$study/backbone_matched_k/seed-81010/summary.json" \
  --fastmem-rmt-zero "$study/fastmem_rmt_zero/seed-81010/summary.json" \
  --fastmem-rmt "$study/fastmem_rmt/seed-81010/summary.json" \
  --output-json "$study/matched-update-comparison.json" \
  --output-csv "$study/matched-update-comparison.csv"
```

Repeat the comparison once per seed if several seeds were run; report paired
seed effects rather than treating probes or languages as independent seeds.
An identity mismatch or incomplete arm invalidates the paired comparison.
Do not combine the output with the old H100 result as though it were another
paired seed. No full production run is part of preparing this script.
