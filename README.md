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
Every repeated appearance uses a distinct packed CulturaX stage. The primary
cycle-end Vietnamese probe is `system`: the Transformer reports its ordinary
curve; FastMem reports carried-memory full-system plasticity while retaining
the reset curve as a diagnostic.

## Install

Python 3.10+ and PyTorch 2.2+ are required. On the A100 machine, install the
package and its data, tracking, and test extras:

```bash
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
  --config configs/experiments/zyphra_fastmem_a100.yaml
```

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
