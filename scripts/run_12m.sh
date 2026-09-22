#!/usr/bin/env bash
set -euo pipefail
# Defaults to a read-only plan; --action run includes streaming preparation.
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
export PYTHONPATH="$repo_dir/src${PYTHONPATH:+:$PYTHONPATH}"
exec "${LM_CL_PYTHON:-python3}" -m lm_cl.cli.a100_run --model-size 12m "$@"
