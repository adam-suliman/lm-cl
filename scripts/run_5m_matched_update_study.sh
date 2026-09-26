#!/usr/bin/env bash
set -euo pipefail

# The default paired study supplies its own contemporary baselines. The
# matched-only mode is useful for exploration, but cannot be paired with the
# historical H100/packed-data runs as an identity-matched control.
usage() {
  cat <<'EOF'
Usage: run_5m_matched_update_study.sh [options]

  --action plan|prepare|preflight|run   Default: plan (no data or training)
  --arms paired|matched-only           Default: paired
  --seeds LIST                         Default: 81010 (e.g. 81010,81011,81012)
  --name NAME                          Default: derived from arms and seeds
  --resume never|auto|required         Default: never
  --checkpoint-retention all|cycle     Default: cycle
  --help

Requires LM_CL_DATA_ROOT and LM_CL_OUTPUT_ROOT. Preparation and running also
require an authenticated, explicitly set HF_HOME. The paired study runs the
clean Transformer, K=2 Transformer, zero-fast-update AG-RMT, and full AG-RMT
on one 80 GB A100. The matched-only mode runs just the K=2 Transformer.
EOF
}

action=plan
arms=paired
seeds=81010
study_name=
resume=never
checkpoint_retention=cycle
while (($#)); do
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --action|--arms|--seeds|--name|--resume|--checkpoint-retention)
      option=$1
      shift
      if (($# == 0)) || [[ "$1" == --* ]]; then
        printf 'Missing value for %s\n' "$option" >&2
        exit 2
      fi
      case "$option" in
        --action) action=$1 ;;
        --arms) arms=$1 ;;
        --seeds) seeds=$1 ;;
        --name) study_name=$1 ;;
        --resume) resume=$1 ;;
        --checkpoint-retention) checkpoint_retention=$1 ;;
      esac
      ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

case "$action" in plan|prepare|preflight|run) ;; *) printf 'Invalid action: %s\n' "$action" >&2; exit 2 ;; esac
case "$arms" in
  paired) models=transformer,backbone_matched_k,fastmem_rmt_zero,fastmem_rmt ;;
  matched-only) models=backbone_matched_k ;;
  *) printf 'Invalid arms selection: %s\n' "$arms" >&2; exit 2 ;;
esac
case "$resume" in never|auto|required) ;; *) printf 'Invalid resume mode: %s\n' "$resume" >&2; exit 2 ;; esac
case "$checkpoint_retention" in all|cycle) ;; *) printf 'Invalid checkpoint retention: %s\n' "$checkpoint_retention" >&2; exit 2 ;; esac
if [[ ! "$seeds" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  printf 'Seeds must be comma-separated nonnegative integers\n' >&2
  exit 2
fi
if [[ -z "$study_name" ]]; then
  study_name="5m-matched-update-${arms}-s${seeds//,/-}-v1"
fi
: "${LM_CL_DATA_ROOT:?Set LM_CL_DATA_ROOT to a dedicated data directory}"
: "${LM_CL_OUTPUT_ROOT:?Set LM_CL_OUTPUT_ROOT to a dedicated output directory}"
if [[ "$action" == prepare || "$action" == run ]]; then
  : "${HF_HOME:?Set HF_HOME to the authenticated Hugging Face home}"
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
exec "$script_dir/run_5m.sh" \
  --action "$action" \
  --name "$study_name" \
  --resume "$resume" \
  --data-mode streaming \
  --streaming-schedule alternating \
  --streaming-chunk-batches 2048 \
  --a100-memory-gb 80 \
  --gpus 0 \
  --gpus-per-job 1 \
  --physical-microbatch-sequences 8 \
  --models "$models" \
  --seeds "$seeds" \
  --cycles 5 \
  --tokens-per-task 1000000000 \
  --probe-tokens 1000000000 \
  --checkpoint-every-batches 0 \
  --checkpoint-retention "$checkpoint_retention"
