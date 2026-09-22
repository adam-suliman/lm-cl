# 5M / 12M A100 runbook — streaming preparation

Use the checkout containing the production streaming integration. The older
`e30f8da` release only had the demonstration; it cannot execute this runbook's
streaming route. These commands are for the operator; no full run is required
to install or inspect the configuration.

## 1. Repository, environment, authentication

```bash
git clone git@github.com:adam-suliman/lm-cl.git
cd lm-cl
git switch main
git pull --ff-only
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[data,pipeline,tracking,test]'

export HF_HOME=/workspace/lm-cl-auth
export LM_CL_DATA_ROOT=/workspace/lm-cl-streaming-data
export LM_CL_OUTPUT_ROOT=/workspace/lm-cl-results
mkdir -p "$HF_HOME" "$LM_CL_DATA_ROOT" "$LM_CL_OUTPUT_ROOT"
python -c "from huggingface_hub import login; login()"
python -m lm_cl.cli.inspect_environment
python -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda)'
```

Use Python 3.10+ and a CUDA-enabled PyTorch compatible with the machine's NVIDIA
driver. If the installed PyTorch is CPU-only, replace it using the appropriate
CUDA wheel before launching. Accept CulturaX's access conditions using the same
HF account. Enter the credential interactively; never put it in a command,
configuration, or log. Use new, dedicated data/cache directories.

## 2. Inspect the plan; optionally prepare the small tokenizer

```bash
./scripts/run_5m.sh --action plan --gpus 0,1 --gpus-per-job 1
./scripts/run_5m.sh --action prepare
```

The plan must report `production_data_route: streaming` and
`incremental_block_production_launch_supported: true`. `prepare` downloads and
inspects only the pinned tokenizer if missing, then freezes the data recipe.
It does **not** download/tokenize all cycles. `run` includes this step
automatically. After `prepare`, `--action preflight` checks recipes, GPUs,
resume identities, and disk admission without starting training or fetching
corpus blocks. Future data cannot be checksum-validated until it is produced;
every consumed block is checked before use.

## 3. Run the full experiment

Run **one** of these allocations at a time, inside `tmux` or a scheduler job.
The defaults are five cycles, 1B requested input tokens per language per cycle,
1B per Vietnamese probe, sequence length 2048, seed 81010, Transformer + AG-RMT,
BF16, global batch 256, and physical microbatch 4 on a 40 GB A100.

```bash
# One A100: the two complete model trajectories run sequentially.
./scripts/run_5m.sh --action run --name stream-5m-w1 --gpus 0

# Two A100s: one model per GPU, concurrently, sharing the producer/data recipe.
./scripts/run_5m.sh --action run --name stream-5m-pair --gpus 0,1 --gpus-per-job 1

# Two A100s cooperating through DDP: model trajectories run sequentially.
./scripts/run_12m.sh --action run --name stream-12m-ddp --gpus 0,1 --gpus-per-job 2
```

Use either script with any listed allocation. For 80 GB cards add
`--a100-memory-gb 80` (starting microbatch 8). These microbatches are starting
values, not measured A100 guarantees. Select a microbatch before a fresh run;
changing it or the number of ranks invalidates exact resume.

For all four controls in a **fresh paired study**, add:

```text
--models transformer,backbone_matched_k,fastmem_rmt_zero,fastmem_rmt
```

For three paired seeds, add `--seeds 81010,81011,81012`. More jobs are queued on
the selected slots. Do not run two independent launchers against one streaming
recipe simultaneously; the supervisor lock rejects this. One launcher supports
multiple models, seeds and GPU ranks. Sequential 5M/12M runs can reuse the recipe.

## 4. Time, disk, and optional bounded A100 calibration

No A100 throughput is claimed from the offline integration tests. To measure a
selected variant/layout on the target machine without starting the full run:

```bash
./scripts/run_5m.sh --action calibrate --models transformer --gpus 0 \
  --calibration-batches 8 --calibration-max-seconds 1800 \
  --calibration-output "$LM_CL_OUTPUT_ROOT/calibration-5m-transformer-w1"
```

Repeat with `--models fastmem_rmt`, the 12M script, or
`--gpus 0,1 --gpus-per-job 2`, using a distinct output directory each time.
Eight logical batches are an operator-requested timing run, **not** an install
step. Streaming prepares only the needed initial blocks. Calibration refuses
occupied GPUs and has a deadline; preserve failed measurements.

A trajectory trains on 39,999,979,520 continual input tokens plus 4,999,997,440
probe input tokens: **44,999,976,960** total. For an observed end-to-end rate
`r` input tokens/s, `44,999,976,960 / r / 86400` is a training-only day estimate.
Add validation, checkpoint I/O, initialization, and cold-data/reconstruction
waits. One GPU runs the two trajectory times in series; two independent GPUs
can approach their maximum if the producer keeps up. For DDP use its measured
rate; do not divide single-GPU time by two. An eight-batch EN sample is a rough
estimate, not evidence of sustained multilingual or probe throughput.

The token cache defaults to **4 GiB**, not the approximately 153 GiB needed to
retain all 41B unique uint32 training/probe tokens. Metadata has a separate
**64 GiB cap** (a ceiling, not predicted use); tokenizer cache and checkpoints
are additional. Preflight includes cache/metadata reservation and checkpoint
estimates and keeps a 20 GiB free-space floor. Logs and failed attempts also
consume space. No checkpoints are automatically deleted.

For the default two-model, one-seed, five-cycle study, the conservative current
admission estimates on one filesystem are:

| Size | Checkpoints for both models | Including 68 GiB data/metadata reservation + 20 GiB free floor |
|---|---:|---:|
| 5M | 107.07 GiB | 195.07 GiB |
| 12M | 157.02 GiB | 245.02 GiB |

These are reservations, not measured final usage; extra seeds and periodic
checkpoints increase them. Streaming reduces corpus storage, not checkpoint size.

By default checkpoints are saved at task/cycle/probe boundaries. Optional
`--checkpoint-every-batches N` improves recovery granularity but adds many large
files; preflight counts them. `retain_last_checkpoints` is not automatic cleanup.
Resource settings are validated and frozen in the recipe. To change defaults
before a fresh study, supply a JSON mapping with `--streaming-settings FILE`,
for example `{"token_cache_bytes":8589934592}`; preserve that file for resume.

## 5. Monitor, resume, and recognize completion

```bash
watch -n 5 nvidia-smi
# Resume the same name, roots, budgets, model/seed selection, and layout:
./scripts/run_5m.sh --action run --name stream-5m-pair --gpus 0,1 \
  --gpus-per-job 1 --resume auto
```

Resume only after the original launcher/processes exit. `auto` can recover a
verified checkpoint even before the first completed-cycle pointer exists;
`required` additionally requires that pointer. The producer restarts under the
same recipe, recovers its transactionally committed position, and reconstructs
missing cache blocks. GPU training resumes from the latest valid checkpoint;
unsaved computation is replayed. Arbitrary SIGKILL cannot create an emergency
checkpoint. Do not terminate experiments by broad process-name matching.

Inspect `$LM_CL_OUTPUT_ROOT/<name>/summary.json`, per-model/seed summaries and
`metrics.jsonl`, plus
`$LM_CL_DATA_ROOT/generated/streaming/<recipe-sha>/producer-status.json`,
`producer-*.log`, and any `producer-error*.json`. `producer-status.json` describes
the last published block; check the PID in `producer-ready.json` to establish
whether the service is still alive. A data wait/error is never
silently treated as EOF. A successful run needs complete launcher/job summaries,
40 completed tasks, five verified probes, full retention evaluation, and matching
checkpoint hashes. The data `complete.json` alone is **not** experiment completion.

Keep `study.json`, `receipts.sqlite`, `source-index/`, `completed/`, stage
`stream.json` references, tokenizer snapshot/manifest, configs, and checkpoints.
Only the specifically owned `streaming/<sha>/cache/` files are disposable.
Eviction can require re-downloading ranges; reliable network access remains
necessary after eviction. [Protocol and limitations](STREAMING_DATA.md).

## Historical runs

`--data-mode packed` selects the original full packed-manifest route, including
`--probe-training-manifest PATH` when reusing a historical VI pool. The older
`zyphra_controls_a100_*` configurations also retain that route. Use historical
identical data for direct comparisons with historical results; new streaming
controls belong with fresh streaming baselines.

The saved historical 12M checkpoints used four-rank NCCL/BF16. This integration
does not make them exactly resumable on one/two A100s or replace their original
data identity. A new branch from their weights needs a separately specified
experiment. The new streaming recipe fixes its full horizon at creation;
changing `--cycles`, budgets, selection or resource settings creates another
recipe and is not an exact continuation of an existing streaming run.
