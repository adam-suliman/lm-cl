# A100 5M control experiments

This runbook launches the two immediate five-cycle, one-billion-token controls:

- `backbone_matched_k`: the no-memory Transformer backbone with the same K=2
  slow-update cadence as AG-RMT;
- `fastmem_rmt_zero`: the exact persistent FastMem-RMT code path with
  `fast_lr=0`.

Together with the existing `transformer` and `fastmem_rmt` summaries, these
separate the no-memory cadence effect, the RMT/persistent-state architecture at
zero fast LR, and the contribution of the positive explicit fast update. One
seed is a controlled descriptive ablation; use paired seeds 81011 and 81012
before making an uncertainty or general-mechanism claim.

## 1. Install and choose owned roots

Run from a fresh clone of `main`:

```bash
git pull --ff-only origin main
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[data,tracking,test]'

export LM_CL_REPO="$(pwd -P)"
export LM_CL_DATA_ROOT=/workspace/lm-cl-data
export LM_CL_OUTPUT_ROOT=/workspace/lm-cl-results
export HF_HOME=/workspace/lm-cl-hf-home
export TMPDIR=/workspace/lm-cl-tmp
mkdir -p "$LM_CL_DATA_ROOT" "$LM_CL_OUTPUT_ROOT" "$HF_HOME" "$TMPDIR"
```

Use paths owned by your account and verify their mounts before any download:

```bash
df -h "$LM_CL_DATA_ROOT" "$LM_CL_OUTPUT_ROOT" "$HF_HOME" "$TMPDIR"
python -m lm_cl.cli.inspect_environment
```

The repository does not contain packed CulturaX data. Copy the existing
`generated/` tree into `$LM_CL_DATA_ROOT/generated`, or authenticate through
the standard Hugging Face lookup under the explicit `HF_HOME` and prepare it.
Never place a credential in a command, YAML, log, or manifest.

## 2. Select the A100 preset

For an 80 GB A100:

```bash
export CONTROL_CONFIG=configs/experiments/zyphra_controls_a100_80gb_5m_5cycle_1b.yaml
export CONTROL_EXPERIMENT=zyphra-controls-a100-80gb-5m-5cycle-1b
```

For a 40 GB A100:

```bash
export CONTROL_CONFIG=configs/experiments/zyphra_controls_a100_40gb_5m_5cycle_1b.yaml
export CONTROL_EXPERIMENT=zyphra-controls-a100-40gb-5m-5cycle-1b
```

The presets start at 12 sequences per rank on 80 GB and 4 per rank on 40 GB.
These are conservative starting values inferred from the completed 80 GB H100
memory trace; they are not a substitute for observing the first A100 batches.
Physical microbatching changes the execution partition, not the global logical
batch or target normalization, but it is included in the run identity. Never
change it while resuming an existing output directory. If the first fresh run
OOMs, choose a new experiment name and reduce it, for example:

```bash
--name zyphra-controls-a100-80gb-5m-5cycle-1b-mb8 \
--physical-microbatch-sequences 8
```

## 3. Validate or prepare the exact data

The control presets use exactly the same tokenizer, five disjoint 1B windows
per continual language, fixed retention-validation sets, and Vietnamese
training/validation manifests as the completed 5M experiment.

If the data have already been copied, perform a full checksum preflight:

```bash
python -m lm_cl.cli.launch_experiments \
  --config "$CONTROL_CONFIG" \
  --models backbone_matched_k,fastmem_rmt_zero \
  --seeds 81010 \
  --dry-run
```

If anything is missing, prepare it once (this is the authenticated network
step) and then repeat the full preflight:

```bash
python -m lm_cl.cli.prepare_experiment_data \
  --config "$CONTROL_CONFIG" \
  --parallel-languages 8

python -m lm_cl.cli.launch_experiments \
  --config "$CONTROL_CONFIG" \
  --models backbone_matched_k,fastmem_rmt_zero \
  --seeds 81010 \
  --dry-run
```

Do not use `--manifest-only-preflight` until this full shard reread succeeds
and its output is saved. Repeated launches may then use the faster manifest
preflight while still checking all recorded identities.

## 4A. One GPU per complete model job

With two A100s, this runs both controls concurrently, one complete trajectory
per GPU:

```bash
set -o pipefail
mkdir -p "$LM_CL_OUTPUT_ROOT/launcher-logs"
run_log="$LM_CL_OUTPUT_ROOT/launcher-logs/a100-controls-one-gpu-per-job-$(date -u +%Y%m%dT%H%M%SZ).log"

python -m lm_cl.cli.launch_experiments \
  --config "$CONTROL_CONFIG" \
  --models backbone_matched_k,fastmem_rmt_zero \
  --seeds 81010 \
  --gpus 0,1 \
  --gpus-per-job 1 \
  --jobs-per-gpu 1 \
  --resume auto \
  --manifest-only-preflight \
  2>&1 | tee "$run_log"

launcher_status=${PIPESTATUS[0]}
printf 'launcher status: %s\nlog: %s\n' "$launcher_status" "$run_log"
test "$launcher_status" -eq 0
```

With only one A100, change `--gpus 0`; the two complete jobs run serially.
`jobs_per_gpu=2` would colocate both models on one card and is not recommended.

## 4B. Multiple GPUs for each complete model job

With two A100s, both GPUs cooperate on one model through DDP and the second
model waits for the same two-GPU slot:

```bash
set -o pipefail
mkdir -p "$LM_CL_OUTPUT_ROOT/launcher-logs"
run_log="$LM_CL_OUTPUT_ROOT/launcher-logs/a100-controls-two-gpus-per-job-$(date -u +%Y%m%dT%H%M%SZ).log"

python -m lm_cl.cli.launch_experiments \
  --config "$CONTROL_CONFIG" \
  --models backbone_matched_k,fastmem_rmt_zero \
  --seeds 81010 \
  --gpus 0,1 \
  --gpus-per-job 2 \
  --jobs-per-gpu 1 \
  --resume auto \
  --manifest-only-preflight \
  2>&1 | tee "$run_log"

launcher_status=${PIPESTATUS[0]}
printf 'launcher status: %s\nlog: %s\n' "$launcher_status" "$run_log"
test "$launcher_status" -eq 0
```

With four A100s, use `--gpus 0,1,2,3 --gpus-per-job 2`; the launcher creates
two disjoint two-GPU groups and runs both controls concurrently. Exact active
task or probe resume requires the same GPU count per job and partition policy.

## 5. Monitor and verify completion

```bash
watch -n 5 nvidia-smi
pgrep -af 'launch_experiments|run_experiment_job|run_continual|resume_continual|run_probe|resume_probe|torch.distributed.run'
find "$LM_CL_OUTPUT_ROOT" -path '*/seed-81010/metrics.jsonl' -printf '%TY-%Tm-%Td %TH:%TM:%TS %p\n' | sort
```

A valid five-cycle 1B completion is rejected unless the summary contains these
derived counts:

| Public model | Logical batches | Slow updates | Fast-update transitions | Probes | Forgetting evaluations |
|---|---:|---:|---:|---:|---:|
| `backbone_matched_k` | 76,320 | 38,160 | 0 | 5 | 40 |
| `fastmem_rmt_zero` | 76,320 | 38,160 | 76,320 at `fast_lr=0` | 5 | 40 |

`fastmem_rmt_zero` retains both reset and carried probe curves. They can differ:
the persistent active root remains the probe-start root at zero fast LR while
slow AdamW updates can change learned M0 used by reset evaluation. The summary
records the maximum absolute CE separation; it does not impose false equality.

## 6. Compare with the completed original arms

Set the four job-level summary paths and write a hash-bound report:

```bash
export TRANSFORMER_SUMMARY=/path/to/original/transformer/seed-81010/summary.json
export MATCHED_K_SUMMARY="$LM_CL_OUTPUT_ROOT/$CONTROL_EXPERIMENT/backbone_matched_k/seed-81010/summary.json"
export ZERO_SUMMARY="$LM_CL_OUTPUT_ROOT/$CONTROL_EXPERIMENT/fastmem_rmt_zero/seed-81010/summary.json"
export FASTMEM_SUMMARY=/path/to/original/fastmem_rmt/seed-81010/summary.json
export COMPARISON_ROOT="$LM_CL_OUTPUT_ROOT/control-comparison-seed-81010"
mkdir -p "$COMPARISON_ROOT"

python -m lm_cl.cli.compare_control_experiments \
  --transformer "$TRANSFORMER_SUMMARY" \
  --backbone-matched-k "$MATCHED_K_SUMMARY" \
  --fastmem-rmt-zero "$ZERO_SUMMARY" \
  --fastmem-rmt "$FASTMEM_SUMMARY" \
  --output-json "$COMPARISON_ROOT/report.json" \
  --output-csv "$COMPARISON_ROOT/metrics.csv"
```

The command fails on incomplete counters, mismatched data, seed, horizon, or
non-variant scientific settings. It reports rather than hides hardware,
physical-microbatch, source-tree, legacy-summary, and single-seed limitations.

For the strongest causal comparison, rerun all four arms on the same A100
layout by adding
`--models transformer,backbone_matched_k,fastmem_rmt_zero,fastmem_rmt` and a
new experiment name. For paired uncertainty, repeat all compared arms with
`--seeds 81011,81012`; never pool 5M and 12M as independent seeds.
