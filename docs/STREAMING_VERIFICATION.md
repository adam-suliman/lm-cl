# Streaming integration verification — 2026-09-22

Implemented for the real production continual/probe trainers and the 5M/12M
A100 entry scripts. This is a fresh streaming data identity, with the legacy
packed-data route retained. See [runbook](A100_CONTROLS.md) and
[protocol](STREAMING_DATA.md).

Only tiny CPU/offline tests were run for this integration. No 5M/12M training,
A100 benchmark, live CulturaX access, dependency installation, or historical data
cleanup was performed. The earlier large acceptance matrix was
not repeated, following the user's explicit restriction on lengthy experiments.

| Check | Result |
|---|---|
| Focused regression suite: production streaming, A100 scripts, calibration, public launcher, prior incremental pipeline and Parquet tests | 121 passed; 95.73 s |
| Streaming-specific suite after recovery/validation fixes | 13 passed; 48.44 s |
| Final stricter unknown-version guard | Targeted test rerun recorded separately |
| Release check, 2026-09-23: same six test modules from an isolated base-commit export containing only the 28 integration files | 126 passed; 96.19 s |
| Both shell scripts: syntax and offline plan execution | Passed; streaming selected, no data/output directories created |
| `git diff --check` | Passed |
| Original dirty/untracked files compared to the initial snapshot | All 104 preserved byte-for-byte |

The suites overlap; their counts must not be added as independent tests.
The release check excludes the pre-existing diagnostic modules and edits to
`pyproject.toml`, `config/__init__.py`, and `models/transformer.py`. Both entry
scripts also passed offline plan execution from that isolated source, with
explicit data/output roots and no directories created.

The streaming tests cover:

- Starting production training before future blocks exist.
- Completed-block checksum/size checks; unavailable data raises instead of EOF.
- Exact pending-document/shuffle-state reconstruction after eviction.
- Partial-K AG-RMT resume matching uninterrupted weights and counters exactly.
- Two complete tiny cycles for Transformer and AG-RMT, including fixed retention
  validation and derived Vietnamese probes; source checkpoints remain unchanged.
- Two-rank CPU/Gloo AG-RMT resume after eviction, matching uninterrupted weights.
- A real local Parquet fixture: selected-column range reads, no raw disk cache.
- Owned-cache symlink rejection and bounded orphan temporary-file handling.
- Recovery after a crash between receipt commit and token-cache publication.
- Exact EOS handling for a final one-token remainder.
- Real producer subprocess supervision, duplicate-supervisor rejection, process
  exit detection, and recovery under the same study.
- Production tokenizer configuration without constructing a production model.
- Standalone packed calibration compatibility and streaming calibration binding.
- Rejection of unknown preparation policies/versions and noncontiguous blocks.

The tiny probe tests replace only the fixed production tokenizer-schema check
with a 16-token fixture check to avoid full-vocabulary computation. All model,
optimizer, active-memory, data-reader, probe and checkpoint execution uses the
production trainers. The actual production tokenizer descriptor is validated
separately without training. CPU/Gloo agreement does not establish bitwise
agreement or throughput on CUDA/NCCL.

Reproduce using an environment containing the declared test/data/pipeline extras:

```bash
python -m pytest -q tests/test_production_streaming.py
python -m pytest -q tests/test_a100_run.py tests/test_calibration.py \
  tests/test_public_release.py tests/test_incremental_pipeline.py \
  tests/test_incremental_remote.py
```

Local evidence is preserved in
`/data/home/admin/cbp/nlp-cl/lm-cl/reports/streaming-production-2026-09-21/`:
initial file/diff snapshots; focused and final test logs; both script plans;
user-change preservation check; default disk-admission estimates; and the
transferable integration patch/file manifest. That patch records the initial
implementation against base `e30f8da`; use the published integration commit for
deployment, including its release documentation. Pre-existing user changes are
excluded from the integration commit.

Remaining operational checks belong on the selected A100 machine: authenticated
source access, its actual Parquet/RAM behavior, sustained producer throughput,
A100 microbatch capacity, and one-/two-GPU timing including probes and data waits.
The runbook supplies bounded calibration and supervised full-run commands.
