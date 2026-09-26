# Cycle checkpoint retention verification — 2026-09-25

The opt-in `--checkpoint-retention cycle` mode was verified without touching
historical checkpoints or accessing a live CulturaX stream. The default
`all` mode retains the original recipe representation and behavior. Cycle
retention is accepted only for fresh alternating recipes with one turn per
language task and no periodic saves.

The focused two-cycle CPU fixture ran the production trainer, probe, job
runner and alternating scheduler for Transformer and AG-RMT. It injected a
crash after a retirement decision was durably recorded but before the file
was removed. Resume completed both cycles. Each job ended with exactly the
two raw Russian probe-source checkpoints and two augmented cycle checkpoints;
both completed Vietnamese probe checkpoints remained in their probe folders.
The job ledger recorded the 14 retired non-Russian task-boundary files by
path and SHA-256. A separate configuration test rejected incompatible
schedule, turn length and periodic-checkpoint settings and verified the
preflight checkpoint allowance.

The relevant regression command passed **107 tests in 316.48 seconds**:

```bash
python -m pytest -q tests/test_alternating_streaming.py \
  tests/test_production_streaming.py tests/test_a100_run.py \
  tests/test_public_release.py
```

Both intended 5M launch plans were validated offline: the `81011,81012`
Transformer/AG-RMT pair and a separate matched-K Transformer study using the
same seed list. These were plans only; no production training, GPU benchmark,
network corpus access or dependency installation occurred. The 80 GB A100
microbatch and production-scale metadata growth still need observation on
the target host. The exact terminal commands and disk estimates are in
[5M_CYCLE_RETENTION_COMMANDS.md](5M_CYCLE_RETENTION_COMMANDS.md).

## Publication verification — 2026-09-26

The same 107-test regression command passed in **315.47 seconds** from a clean
export of the staged publication files, excluding all unrelated local diagnostic
changes. This verifies that the release does not depend on those local files.
The executable matched-K wrapper also produced validated offline launch plans
for seed `81010` and seeds `81010,81011,81012`; the two-seed Transformer/AG-RMT
plan passed the same check. Each selected alternating streaming, five cycles,
and `cycle_end_v1` retention; the matched-K configuration retained `K=2`.
Shell syntax and staged whitespace checks passed. Publication checks performed
no live corpus access, dependency installation, or production experiment.
