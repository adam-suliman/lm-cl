# FastMem-RMT continual language learning

This package runs one portable Zyphra-style continual-language experiment with
exactly two public model names:

- `transformer`, mapped to the approved `backbone_clean` baseline (AdamW after
  every logical batch, K=1);
- `fastmem_rmt`, the approved one-pass FastMem-RMT realization with eight
  memory vectors, two 1,024-token segments, K=2, fast LR 0.005 by default,
  active-memory-only clipping at 1.0, task-boundary reset from learned M0, and
  no graph across logical batches.

The language order is always `en → zh_written → fr → ja → es → de → pt → ru`.
The full-budget preset uses a distinct packed CulturaX stage for every repeated
appearance. The explicitly labelled scaled-budget preset instead uses five
non-overlapping 1B sequence windows from each frozen cycle-0 5B stage. The primary
cycle-end Vietnamese probe is `system`: the Transformer reports its ordinary
curve; FastMem reports carried-memory full-system plasticity while retaining
the reset curve as a diagnostic.

## Install

Python 3.10+ and PyTorch 2.2+ are required. On the A100 machine, install the
package and its data, tracking, and test extras:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[data,tracking,test]'
python -m lm_cl.cli.inspect_environment
```

No CLI accepts a Hugging Face credential. Authenticate through the standard
Hugging Face mechanism under an explicit `HF_HOME`.

The source repository supplied for this release contains no license file.
This export therefore does not invent or imply a license; obtain the necessary
permission before redistributing or using it beyond the rights you already
hold.

## Prepare packed data

Choose machine-owned roots with ample storage:

```bash
export LM_CL_DATA_ROOT=/workspace/lm-cl-data
export LM_CL_OUTPUT_ROOT=/workspace/results
export HF_HOME=/workspace/lm-cl-hf-home
```

Inspect the immutable tokenizer once:

```bash
python -m lm_cl.cli.inspect_tokenizer \
  --repo-id Qwen/Qwen3-0.6B-Base \
  --revision da87bfb608c14b7cf20ba1ce41287e8de496c0cd \
  --cache-root "$LM_CL_DATA_ROOT/hf-cache" \
  --generated-root "$LM_CL_DATA_ROOT/generated" \
  --max-cache-bytes 21474836480 \
  --max-generated-bytes 1099511627776 \
  --output-manifest "$LM_CL_DATA_ROOT/generated/tokenizers/qwen3/manifest.json" \
  --model-embedding-vocab-size 151680

python -m lm_cl.cli.prepare_experiment_data \
  --config configs/experiments/zyphra_fastmem_a100.yaml \
  --parallel-languages 8
```

The materializer automatically uses ordered fast-tokenizer batches sized from
the visible CPU count. On large CPU machines, set the non-scientific execution
controls explicitly before preparation:

```bash
export TOKENIZERS_PARALLELISM=true
export RAYON_NUM_THREADS=128
export LM_CL_TOKENIZER_BATCH_DOCUMENTS=2048
export LM_CL_REGISTRY_CACHE_MIB=4096
export LM_CL_REGISTRY_MMAP_MIB=65536
export LM_CL_STREAM_RESHARD_ROW_GROUPS=false
export LM_CL_STREAM_RESHARD_FILE_PREFIX=32
export LM_CL_STREAM_PREFETCH_SHARDS=4
export LM_CL_STREAM_PREFETCH_ROWS_PER_SHARD=256
export LM_CL_MATERIALIZATION_CHECKPOINT_CANDIDATES=100000
```

Batching changes only execution. Documents are still accepted and registered
in the exact deterministic order, and tests require scalar and batched runs to
produce identical packed bytes and `ordered_data_sha256`. The CLI emits a JSON
progress event at every resumable stage checkpoint, including invocation token
throughput. The optional checkpoint override amortizes durable filesystem
synchronization; these controls change neither final data nor manifest identity.
`--parallel-languages 8` assigns one CPU/tokenizer lane to each continual
language. Cycles remain sequential inside a lane. Training stays gated until
the lane registries merge into the global registry with no cross-language
content or token conflict and a hash-bound audit is written.

Preparation uses pinned `uonlp/CulturaX` revision
`6a8734bc69fefcbb7735f4f9250f43e4cd7a442e`, the approved language mapping,
`text`, stable `url` IDs with content-SHA-256 fallback, strict caps, and a
global overlap registry. Training is offline after all manifests validate.
See [DATA_AND_RESUME.md](docs/DATA_AND_RESUME.md) before allocating storage.

## Launch two complete experiments

```bash
python -m lm_cl.cli.inspect_environment \
  --config configs/experiments/zyphra_fastmem_a100.yaml

python -m lm_cl.cli.launch_experiments \
  --config configs/experiments/zyphra_fastmem_a100.yaml \
  --models transformer,fastmem_rmt \
  --cycles 2 \
  --tokens-per-task 5000000000 \
  --gpus 0,1,2,3 \
  --jobs-per-gpu 1 \
  --output-root /workspace/results \
  --resume auto
```

One scheduled job is the entire trajectory for one model and seed, including
all languages, cycles, and cycle-end probes. To allow two independent jobs on
each GPU, use `--jobs-per-gpu 2`; this intentionally shares each GPU and must
fit memory. For multi-GPU jobs, set `--gpus-per-job 2`; GPU groups are disjoint
and each stage uses the existing DDP path through `torchrun`.

Monitor all jobs with:

```bash
tensorboard --logdir /workspace/results/zyphra-fastmem-a100
```

## Urgent five-cycle 1B scaled experiment

The H100 presets reuse the eight existing cycle-0 5B training stages without
copying token files. Each appearance consumes 488,281 complete sequences, or
999,999,488 input tokens. Five windows consume 2,441,405 of the 2,441,406
sequences in each source, leaving one sequence unused. The same window is never
replayed in another cycle.

Forgetting is evaluated after every language boundary on eight fixed held-out
language-validation stages. These are selected by the existing SHA-256 document
split and never overlap the training stages. Each contains 1,280 sequences; all
eight add only 80 MiB of packed uint32 tokens. Transformer forgetting uses
ordinary validation CE. FastMem forgetting always uses reset memory, making it
a slow/backbone-retention measurement; carried current-task validation remains
a separate full-system diagnostic.

Prepare only the missing held-out validation stages and validate all existing
inputs once:

```bash
python -m lm_cl.cli.prepare_experiment_data \
  --config configs/experiments/zyphra_fastmem_h100_5m_5cycle_1b.yaml \
  --cycles 5 \
  --parallel-languages 8

python -m lm_cl.cli.launch_experiments \
  --config configs/experiments/zyphra_fastmem_h100_5m_5cycle_1b.yaml \
  --dry-run

python -m lm_cl.cli.launch_experiments \
  --config configs/experiments/zyphra_fastmem_h100_12m_5cycle_1b.yaml \
  --dry-run
```

The presets contain three paired seeds. Run the 5M launcher on GPUs 2–4 and the
12M launcher on GPUs 5–7 in separate terminals. Each GPU executes one job at a
time; six jobs per model size are queued across three GPUs:

```bash
python -m lm_cl.cli.launch_experiments \
  --config configs/experiments/zyphra_fastmem_h100_5m_5cycle_1b.yaml \
  --gpus 2,3,4 \
  --jobs-per-gpu 1 \
  --resume auto

python -m lm_cl.cli.launch_experiments \
  --config configs/experiments/zyphra_fastmem_h100_12m_5cycle_1b.yaml \
  --gpus 5,6,7 \
  --jobs-per-gpu 1 \
  --resume auto
```

This is a `scaled_budget` experiment, not an exact 5B-per-appearance Zyphra
reproduction. Its Vietnamese probes are likewise 1B prefixes of the frozen 5B
probe-training shard. Full task-boundary forgetting matrices are in each job's
`metrics.jsonl`; the final per-language and average values are copied to
`summary.json`.

## Resume or extend a completed horizon

`resume: auto` validates `latest_checkpoint.json` and resumes interruption
state when present. To extend a completed two-cycle job to five cycles, first
prepare the three future cycle×language stages, then run:

```bash
python -m lm_cl.cli.prepare_experiment_data \
  --config configs/experiments/zyphra_fastmem_a100.yaml \
  --cycles 5

python -m lm_cl.cli.launch_experiments \
  --config configs/experiments/zyphra_fastmem_a100.yaml \
  --models transformer,fastmem_rmt \
  --cycles 5 \
  --gpus 0,1,2,3 \
  --jobs-per-gpu 1 \
  --resume required
```

Only the horizon, future manifest additions, and non-scientific logging fields
may change. Model, seed, completed manifests, tokenizer, optimizer, batches,
precision, FastMem, and probe protocol are immutable.

## Outputs and smoke tests

Each job writes `resolved_experiment.yaml`, `job_metadata.json`,
`metrics.jsonl`, `stdout.log`, `stderr.log`, `tensorboard/`, `checkpoints/`,
`probes/`, `latest_checkpoint.json`, and `summary.json`. The experiment root
also contains `jobs.jsonl`, `summary.json`, and `summary.csv`.

The bundled two-cycle smoke uses a tiny CPU model and synthetic tokens while
exercising both public variants, all eight language names twice, two probes,
checkpointing, TensorBoard, and summaries:

```bash
python -m lm_cl.cli.launch_experiments \
  --config configs/experiments/zyphra_fastmem_two_cycle_smoke.yaml
```

This is a functional test only. It is not a paper-performance result or a
scientific CulturaX run.
