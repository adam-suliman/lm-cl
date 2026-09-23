# Alternating streaming verification — 2026-09-23

The optional alternating schedule was checked from an isolated export of base
commit `2b07a92a97b1d2c12532202eb991caa3aeb6119a` plus only the integration files.
Pre-existing diagnostic code and the user's unrelated edits to `pyproject.toml`,
`config/__init__.py`, and `models/transformer.py` were excluded. No full-size model
training, live CulturaX download, dependency installation, or A100 timing run
was performed. The user's tiny-test restriction replaces a fresh long acceptance
run for this implementation; the earlier real-data acceptance record is unchanged.

## Results

**135 tests passed in 274.65 seconds**, including 9 new alternating tests and the
existing production-streaming, A100-entry, calibration, public-launcher,
incremental-pipeline and remote-Parquet fixture suites. All computation used tiny
CPU models; the distributed checks used two CPU/Gloo ranks. Tests use a 16-token
fixture vocabulary in place of the production tokenizer-schema check. The model,
optimizer, manual active-memory, checkpoint, probe and data-reader code paths
remain the production implementations.

A final disk-report clarification separates the pinned-data allowance from
cache/metadata allowance in preflight output. Its targeted checks passed
**4 tests in 2.11 seconds**, including one additional pinned-space accounting
test (136 distinct tests covered overall; three targeted checks overlap the
135-test run). No training/storage algorithm changed after that full run.
The two-rank checkpoint artifacts were also inspected directly: scheduler state,
as well as all fields compared in the distributed test, matched exactly.

| Requirement | Evidence |
|---|---|
| Separate optional mode; existing route remains available | Both 5M/12M shell scripts executed offline plans in both schedules; legacy regression suite passes; legacy recipe limits/format remain unchanged |
| Alternate bounded turns across consumers | Actual subprocess job runners execute two consumers serially on one slot; two-cycle paired runner uses the production scheduler's turn orchestration |
| Permanently retain Vietnamese and all fixed validation data | Two complete tiny cycles with all nine validation streams and reused VI training; pinned byte count and hashes verified; release cannot target the pinned directory |
| Bound continual queue | Configuration rejects undersized queues; publication refuses to evict unacknowledged blocks; queue size checked after each consumer turn |
| Release only after every consumer has durable recovery | One consumer acknowledgment cannot release data; wrong checkpoint hash and wrong consumer identity fail; final acknowledgment permits release |
| Recover interrupted publication and release | Injected crash after receipt commit, crash after first durable acknowledgment, and crash after release watermark publication; verified restart completes the operation |
| Compact verification/deduplication | Empty historical state column, compact state/boundary digests, binary 32-byte SHA-256 dedup keys; receipt payloads less than one quarter of the fixture's legacy size |
| Selected detailed recovery points | Snapshot count below half the fixture block count, one mutable head per stream, reconstruction from selected state; corrupt snapshot fails |
| Preserve exact scientific execution | Two-cycle Transformer/AG-RMT comparison matches all token-block hashes, final weights, optimizer, scheduler, gradients, RNG, active memory, trainer state, and both cycle probe AUCs |
| Preserve partial-K and DDP resume | Two-rank AG-RMT resumes after acknowledged queue release; weights, optimizer, gradients, counters, RNG and active memory match uninterrupted execution |
| Preserve validation integrity | Corrupted pinned validation block is rejected before consumption |
| Reject incompatible calibration | Calibration crossing the initial alternating turn is rejected before training |

The first clean run exposed an existing cycle-backfill problem: a checkpoint
inside the first language of a later cycle could be replaced by the preceding
cycle-boundary checkpoint. Independent reconstruction allowed that work to be
replayed. The runner now promotes the cycle checkpoint only when the current
checkpoint itself is at the boundary. The final test asserts that each turn
starts at its acknowledged predecessor, preventing that silent rollback.

The test suites overlap with earlier implementation checks; counts from those
earlier runs must not be added to the coverage count. Tiny fixture compression/storage ratios
are not estimates of production metadata growth. CPU/Gloo agreement does not
establish CUDA/NCCL bitwise equivalence or throughput.

## Reproduction and evidence

With the declared test/data/pipeline extras already installed:

```bash
python -m pytest -q tests/test_alternating_streaming.py \
  tests/test_production_streaming.py tests/test_a100_run.py \
  tests/test_calibration.py tests/test_public_release.py \
  tests/test_incremental_pipeline.py tests/test_incremental_remote.py
```

Local evidence is under
`/data/home/admin/cbp/nlp-cl/lm-cl/reports/alternating-streaming-2026-09-23/`:
initial user-file hashes; clean-source manifests; all test attempts including
the failed clean run; `clean-tests-v2.txt`; four offline plan outputs; disk
estimates; and `completion-audit.json` / `alternating-integration-final.patch`.
All 104 initial user-owned
dirty/untracked files are preserved byte-for-byte. No existing experiment or
reference repository was modified, and no historical checkpoint was deleted.

The new mode requires its own frozen recipe and consumer group. Use the
[runbook](A100_CONTROLS.md) for setup, microbatch selection, disk admission,
bounded target-machine calibration and full launch. Authenticated source access,
actual A100 capacity/throughput, and production metadata growth remain operational
measurements to perform on the intended machine; none are claimed by these tests.
