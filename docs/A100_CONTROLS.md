# A100 5M control experiments

For a **fresh Transformer/AG-RMT pair at either 5M or 12M**, use the paired
entry points described at the end of this document. The original control-only
instructions below remain applicable to the matched-K and zero-fast-LR study.

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

## Fresh 5M and 12M paired runs

The executable `scripts/run_5m.sh` and `scripts/run_12m.sh` default to a
read-only configuration plan. Their defaults are a Transformer/AG-RMT pair,
seed 81010, five cycles of eight languages, 1B requested tokens per task and
per cycle-end Vietnamese probe, global batch 256, sequence length 2048, and
BF16. Flooring to complete sequences yields 999,999,488 input tokens per
task. This is the scaled-budget study, not the paper's 5B-per-task protocol.

Use the existing environment if it satisfies the declared dependencies. On a
new machine, the installation command is `python -m pip install -e
'.[data,tracking,pipeline,test]'`; installation is an operator action, not
something the launch scripts perform. Set `LM_CL_PYTHON` when the interpreter
is not `python3`. Keep standard HF authentication under an explicit `HF_HOME`.
Never pass a token value in a command or put it into a config.

```bash
export LM_CL_PYTHON="$PWD/.venv/bin/python"
export LM_CL_DATA_ROOT=/workspace/lm-cl-data
export LM_CL_OUTPUT_ROOT=/workspace/lm-cl-results
export HF_HOME=/workspace/lm-cl-hf-home
unset CUDA_VISIBLE_DEVICES

./scripts/run_5m.sh --action plan --gpus 0,1 --gpus-per-job 1
./scripts/run_12m.sh --action plan --gpus 0,1 --gpus-per-job 2
```

GPU IDs are physical IDs in an unmasked environment. With `--gpus 0,1`,
`--gpus-per-job 1` runs independent jobs; `--gpus-per-job 2` assigns both GPUs
to one DDP job at a time. With `--gpus 0 --gpus-per-job 1`, the pair runs
sequentially. Other controls and paired seeds are selectable with `--models`
and `--seeds`. A100 memory class is selected with `--a100-memory-gb 40` or
`80`. Starting physical microbatches are respectively 4 and 8, and are
**unmeasured A100 starting values**, not sustained-fit guarantees.

### Freeze a tokenizer and prepare data

For a new data root, inspect the pinned tokenizer before the first download.
Use this only when the output manifest does not already exist:

```bash
"$LM_CL_PYTHON" -m lm_cl.cli.inspect_tokenizer \
  --repo-id Qwen/Qwen3-0.6B-Base \
  --revision da87bfb608c14b7cf20ba1ce41287e8de496c0cd \
  --cache-root "$LM_CL_DATA_ROOT/hf-cache" \
  --generated-root "$LM_CL_DATA_ROOT/generated" \
  --max-cache-bytes 21474836480 --max-generated-bytes 1099511627776 \
  --model-embedding-vocab-size 151680 \
  --output-manifest "$LM_CL_DATA_ROOT/generated/tokenizers/qwen3/manifest.json"

./scripts/run_5m.sh --action prepare --parallel-languages 1
./scripts/run_5m.sh --action preflight --gpus 0,1 --gpus-per-job 1
```

Data are shared between the model sizes and variants. The fresh paired presets
prepare a 1B Vietnamese pool and reuse its fixed prefix for all five probes.
To reuse an existing larger validated pool, supply
`--probe-training-manifest /absolute/path/to/manifest.json` consistently to
every invocation. Pool filenames do not establish their token counts; the
manifest is authoritative.

The native materializer streams the source dataset into completed packed
shards. The production launcher currently validates the whole study's manifest
matrix before launch. It does **not** yet train the full study from partially
published incremental-demo blocks. The separate `local_pipeline_demo` has a
distinct checkpoint kind and cannot be used as a production resume.

Existing packed data require their recorded tokenizer identity. Tokenizer
manifests include a snapshot path as well as a portable content hash. Merely
copying a generated tree to a different path, or editing only its tokenizer
manifest, does not establish compatibility. Preserve the required path mapping
or perform a separately verified relocation; do not bypass identity checks.

### Calibrate before scheduling a full run

With completed data, these commands construct a separate calibration config,
run the real production trainer for eight logical batches, discard the first
two from the timing estimate, and retain an interrupted checkpoint:

```bash
./scripts/run_5m.sh --action calibrate --models transformer \
  --gpus 0 --gpus-per-job 1 --a100-memory-gb 40 \
  --calibration-output "$LM_CL_OUTPUT_ROOT/calibrations/5m-transformer-1gpu-v1"

./scripts/run_5m.sh --action calibrate --models fastmem_rmt \
  --gpus 0,1 --gpus-per-job 2 --a100-memory-gb 40 \
  --calibration-output "$LM_CL_OUTPUT_ROOT/calibrations/5m-ag-2gpu-v1"
```

Repeat with `run_12m.sh` and fresh calibration output names for 12M. Calibrate
both variants and each layout being considered. A validated standalone
continual config using a bounded completed packed stage may instead be passed
through `--calibration-config`; its model, precision, microbatch, and DDP layout
must match the requested launch. That permits calibration without a completed
five-cycle corpus when such a bounded config is available.

Calibration defaults to a 30-minute deadline, a 10 GiB output cap, and 20 GiB
remaining free-space floor. It refuses occupied GPUs, existing output roots,
synthetic data, and mismatched devices. `calibration.json` records throughput
over complete slow-update windows, including communication, data wait, and
optimizer work. Initialization, final checkpoint writes, and probes are
separate overheads. It counts global input tokens, not only valid targets.

If a trial fails or OOMs, retain it and choose a new output name. Change the
physical microbatch only in a new calibration/fresh-run identity. Do not reuse
an interrupted scientific output with a different microbatch or GPU layout.

### Launch, monitor, and resume

```bash
./scripts/run_5m.sh --action run --name pair-5m-1b-w1-mb4 \
  --gpus 0,1 --gpus-per-job 1 --physical-microbatch-sequences 4

./scripts/run_12m.sh --action run --name pair-12m-1b-w2-mb4 \
  --gpus 0,1 --gpus-per-job 2 --physical-microbatch-sequences 4
```

Those are alternative allocations, not commands to run simultaneously on the
same GPUs. Use `tmux` or the site's scheduler for a long run. Inspect
`nvidia-smi`, the launcher logs, each job's `metrics.jsonl`, and
`latest_checkpoint.json`. Checkpoint filenames or directory existence alone
do not prove completion. Require launcher and job summaries reporting complete,
40 completed language tasks, five completed probes, and matching final payload
hashes. The saved probe curves and unsmoothed CE-AUC are the plasticity endpoint.

Resume the same command and same identity with `--resume required` after
checking that its original processes have exited. `--resume never` is the
fresh-run default. `--dry-run` aliases the full read-only preflight and does
not download missing data. The launcher propagates failed jobs and refuses
incompatible checkpoint identities.

The production trainer writes at configured checkpoint and task boundaries;
it does not promise an emergency checkpoint on an arbitrary SIGINT/SIGTERM.
For a planned interruption of a direct continual invocation, use
`--stop-after-global-logical-batches` or `--stop-after-task-boundaries` with
`lm_cl.cli.train_continual`, and resume using `lm_cl.cli.resume_continual`.
For the full launcher, allow the current boundary to finish before ending its
owned session and expect replay from the most recent verified boundary.
Do not terminate by broad process name or assume a partly written checkpoint
is resumable. Preserve failed attempts.

### Historical 12M continuation and smaller local models

The preserved cycle-five 12M Transformer and AG-RMT payloads are full continual
checkpoints, but their execution identity is four-rank NCCL/BF16. Exact resume
on one or two A100s is not supported by the current contract. Horizon extension
also requires the historical packed identities and future disjoint windows.
Changing a checkpoint's world size or task list is not a valid migration.
An explicitly defined new branch from the boundary weights is a different
experiment; a source hash alone cannot recover the missing data provenance.

`configs/models/exploratory_1m_nonembedding.yaml` is a timing candidate with
991,616 non-embedding parameters and 20,668,800 total Transformer parameters
(20,669,824 with learned M0). It preserves the vocabulary and is not a 1M-total
model. On the measured GTX 1080 Ti FP32/microbatch-1 layout, eight-batch timing
gave about 27.6k input tokens/s for the three-GPU Transformer and 25.4k for
AG-RMT. These short measurements do not establish long-run stability or a tuned
learning rate, and do not justify proportional scaling from 5M/12M.
