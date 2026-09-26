# 5M A100 commands: missing seeds and matched-K control

These commands target one 80 GB A100 at GPU index 0. They require a checkout
containing the `--checkpoint-retention cycle` implementation, an installed
CUDA-enabled environment, and an authenticated explicit `HF_HOME` as described
in [A100_CONTROLS.md](A100_CONTROLS.md). Run the two studies sequentially on
that GPU. The default checkpoint policy is `all`; the commands below opt into
cycle retention from the first preparation step.

```bash
cd /workspace/lm-cl
git switch main
git pull --ff-only origin main
source .venv/bin/activate
export HF_HOME=/workspace/lm-cl-auth
export LM_CL_DATA_ROOT=/workspace/lm-cl-streaming-data
export LM_CL_OUTPUT_ROOT=/workspace/lm-cl-results
mkdir -p "$HF_HOME" "$LM_CL_DATA_ROOT" "$LM_CL_OUTPUT_ROOT"
```

## Missing Transformer and AG-RMT seeds: 81011 and 81012

Use **one** launcher so the four trajectories share one frozen token recipe.
`plan` is read-only, `prepare` freezes the recipe and tokenizer, `preflight`
checks GPU/disk/identities, and `run` starts the full five-cycle experiment.

```bash
PAIR=(
  --name 5m-pair-s81011-81012-cycle-v1
  --data-mode streaming --streaming-schedule alternating
  --checkpoint-retention cycle --checkpoint-every-batches 0
  --a100-memory-gb 80 --gpus 0 --gpus-per-job 1
  --models transformer,fastmem_rmt --seeds 81011,81012
  --cycles 5 --tokens-per-task 1000000000 --probe-tokens 1000000000
)
./scripts/run_5m.sh --action plan "${PAIR[@]}"
./scripts/run_5m.sh --action prepare "${PAIR[@]}"
./scripts/run_5m.sh --action preflight "${PAIR[@]}"
./scripts/run_5m.sh --action run "${PAIR[@]}"
```

After an interruption, wait until the original launcher is no longer active,
then resume with the same array and `--resume auto`:

```bash
./scripts/run_5m.sh --action run "${PAIR[@]}" --resume auto
```

The admission estimate is about **112.90 GiB free** on one filesystem,
including the 20 GiB free floor. Other datasets, tokenizer downloads, logs
and failed attempts consume additional space.

## Matched-K Transformer separately

This is the memory-free Transformer with `K=2`, matching AG-RMT's **slow
parameter** update frequency. It is a separate streaming recipe; use the
same seed list as the pair above to compare their trained trajectories.
For one seed, replace `81011,81012` with `81011` in **every** command. A
different seed list creates a distinct study name and recipe.

```bash
MATCHED=(--arms matched-only --seeds 81011,81012)
./scripts/run_5m_matched_update_study.sh --action plan "${MATCHED[@]}"
./scripts/run_5m_matched_update_study.sh --action prepare "${MATCHED[@]}"
./scripts/run_5m_matched_update_study.sh --action preflight "${MATCHED[@]}"
./scripts/run_5m_matched_update_study.sh --action run "${MATCHED[@]}"
```

Resume only after the original process has exited:

```bash
./scripts/run_5m_matched_update_study.sh --action run "${MATCHED[@]}" --resume auto
```

The separate two-seed matched-K study reserves about **74.36 GiB free**;
one seed reserves about **55.09 GiB**. Existing pair outputs still occupy
disk when the matched-K preflight checks available space. Keeping both finished
studies on one disk implies roughly **150.41 GiB initially free** under these
checkpoint/cap estimates, plus installation, source caches, logs and failed
attempts.

Separate recipes can select identical tokens, but their distinct recipe
identities do **not** prove byte identity. Compare their immutable per-block
`data_sha256` receipts, tokenizer identity, budgets, and validation pools
before calling the resulting model contrasts paired. If neither study has
started yet, the scientifically cleaner and more efficient design is a
single three-arm launcher with
`--models transformer,backbone_matched_k,fastmem_rmt --seeds 81011,81012`
in the `PAIR` array; all arms then share one recipe by construction. The old
seed-81010 H100/packed-data run remains historical context, not an identical
third seed for this new streaming study.

With cycle retention, the current within-cycle recovery checkpoint remains
until its successor is verified and all consumers advance. At each cycle end,
the raw Russian checkpoint, augmented cycle checkpoint, and completed
Vietnamese probe checkpoint remain. The runner records retired within-cycle
checkpoint hashes in `checkpoint_retention.json`; it does not remove
checkpoints from any pre-existing study.
