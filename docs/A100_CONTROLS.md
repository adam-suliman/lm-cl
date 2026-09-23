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

Override the per-GPU execution microbatch explicitly, for example:

```bash
./scripts/run_12m.sh --action plan --gpus 0 --physical-microbatch-sequences 2
```

Use the same override with `prepare`, `preflight`, `calibrate`, `run`, and resume.
Smaller microbatches reduce GPU memory demand; they do not reduce the global
logical batch of 256 sequences or materially reduce disk usage. `plan` performs
no training; change to `run` after the target-machine checks.

For all four controls in a **fresh paired study**, add:

```text
--models transformer,backbone_matched_k,fastmem_rmt_zero,fastmem_rmt
```

For three paired seeds, add `--seeds 81010,81011,81012`. More jobs are queued on
the selected slots. Do not run two independent launchers against one streaming
recipe simultaneously; the supervisor lock rejects this. One launcher supports
multiple models, seeds and GPU ranks. Sequential 5M/12M runs can reuse the recipe.

### Optional alternating schedule

The preceding commands default to `--streaming-schedule independent`, retaining
the original streaming behavior. For a fresh study with compact preparation
history and bounded model/seed turns:

```bash
./scripts/run_5m.sh --action plan --name alt-5m --gpus 0 \
  --streaming-schedule alternating
./scripts/run_5m.sh --action run --name alt-5m --gpus 0 \
  --streaming-schedule alternating
```

The 12M script accepts the same flags. One GPU alternates models/seeds; two GPUs
can process two consumers in a turn, or one consumer with two-rank DDP. Every
consumer finishes the turn before the queue advances. The default maximum is
2048 global logical batches per turn, capped at the next language boundary.
At the default 1B-token task budget, this means one language task per turn and
no additional mid-task checkpoints.

`--streaming-chunk-batches 512` requests smaller turns (up to 1 GiB of uint32
training tokens at the default batch/sequence sizes). Each extra mid-task turn
saves a retained resume checkpoint, including partial-K gradients and memory.
For five cycles this setting adds 120 checkpoints per model/seed; it can make
total disk usage larger despite the smaller data queue. Preflight counts these
files. No checkpoint is automatically deleted.

The alternating defaults use a 4 GiB continual-data queue, an 8 GiB metadata
cap, and separately pinned Vietnamese/fixed validation data (about 3.813 GiB
at the default budgets). The metadata cap is unmeasured at production scale.
Keep the same schedule, chunk size, resource file, name, paths, model/seed set,
and GPU layout for `--resume auto`. This mode binds the consumer output paths;
use a separate recipe for a different study/model size. Existing independent
studies are not converted. [Storage and recovery details](ALTERNATING_STREAMING.md).

For the default two-model, one-seed, five-cycle study, the alternating admission
estimates are **142.880 GiB (5M)** and **192.830 GiB (12M)** on one filesystem.
They include the unchanged checkpoint estimate, 4 GiB queue, 8 GiB metadata cap,
3.813 GiB pinned tokens and 20 GiB free floor. These are conservative reservations,
not measured production usage; software installation, logs and failed attempts
need additional room. Smaller chunks add checkpoint files as described above.

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
With the alternating schedule, calibration must fit inside its initial turn;
the validator rejects a larger calibration window before training.

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

### CLI and streaming resource reference

Both entry scripts accept the same options; `./scripts/run_5m.sh --help` lists
them. The model size is selected by the script.

| Option | Meaning / default |
|---|---|
| `--action` | `plan` (default), `prepare`, `preflight`, `calibrate`, or `run` |
| `--dry-run` | Alias for read-only preflight; requires a prepared recipe |
| `--data-mode` | `streaming` (default) or legacy `packed` |
| `--streaming-schedule` | `independent` (default) or `alternating` |
| `--streaming-chunk-batches` | Alternating maximum logical batches per turn; 2048 |
| `--streaming-settings` | JSON file overriding the resource settings below |
| `--physical-microbatch-sequences` | Sequences processed together per GPU; 4 for 40 GB, 8 for 80 GB |
| `--a100-memory-gb` | GPU memory class, 40 (default) or 80; not an allocation limit |
| `--gpus` / `--gpus-per-job` | Physical GPU IDs (`0` by default); 1 or 2 ranks per job |
| `--models` / `--seeds` | Comma-separated variants / seeds; Transformer + AG-RMT, seed 81010 |
| `--cycles` / `--tokens-per-task` / `--probe-tokens` | Fresh-study budgets: 5 / 1B / 1B |
| `--name` | Unique experiment name; otherwise derived from model/layout/budget |
| `--resume` | `never` (default), `auto`, or `required` |
| `--checkpoint-every-batches` | Additional periodic saves; 0 disables only periodic saves |
| `--data-root` / `--output-root` | Override `LM_CL_DATA_ROOT` / `LM_CL_OUTPUT_ROOT` |
| `--cache-root` | Tokenizer/HF cache path; defaults inside the data root |
| `--parallel-languages` | Legacy preparation parallelism; streaming requires 1 |
| `--probe-training-manifest` | Historical VI pool override; requires packed mode |
| `--calibration-config` | Optional pre-existing validated continual config for calibration |
| `--calibration-output` | New calibration output directory, required for calibration |
| `--calibration-batches` / `--calibration-max-seconds` | Calibration limit: 8 batches / 1800 seconds |

Values in a resource JSON file are integers in bytes, tokens, counts or seconds:

| JSON key | Default | Meaning |
|---|---:|---|
| `token_cache_bytes` | 4294967296 | Independent LRU cache or alternating continual queue |
| `metadata_bytes` | 68719476736 independent; 8589934592 alternating | Preparation records cap, excluding token data |
| `minimum_free_bytes` | 21474836480 | Free-space floor |
| `block_tokens` | 1048576 | Packed block size; must divide into complete sequences |
| `prefetch_blocks` | 2 | Lookahead within the task and, for alternating, the open turn |
| `chunk_batches` | 2048, alternating only | Same maximum as the CLI chunk override |
| `wait_seconds` | 3600 | Consumer wait limit; must exceed block deadline |
| `max_block_seconds` | 1800 | Producer block/request deadline |
| `max_row_group_bytes` | 268435456 | Remote range / decoded row-group admission limit |
| `max_document_bytes` | 4194304 | Maximum accepted UTF-8 document size |
| `max_source_files` | 10000 | Maximum files in a pinned language inventory |
| `max_network_requests` | 10000000 | Request-attempt limit per producer process |
| `max_network_attempts_per_request` | 3 | Bounded request retries |
| `network_timeout_seconds` | 60 | HTTP request timeout |

Select the schedule with its CLI flag. Lowering a cap does not shrink the
records produced. Caps are frozen into the recipe; exceeding one fails the
attempt, and changing it does not constitute exact resume. The alternating
queue must fit a complete turn plus block-boundary allowance. Do not decrease
it independently of the chunk size.

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
