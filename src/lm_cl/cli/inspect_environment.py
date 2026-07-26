from __future__ import annotations

import argparse
import json
from importlib.util import find_spec

from lm_cl.environment import inspect_environment
from lm_cl.launcher.config import load_launcher_config
from lm_cl.launcher.data import resolve_data_contract
from lm_cl.launcher.jobs import expand_job_specs
from lm_cl.launcher.scheduler import allocate_job_slots, preflight_launch


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect PyTorch/GPU precision support and optional launch readiness"
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()
    environment = inspect_environment()
    if args.config is None:
        print(json.dumps(environment, indent=2, sort_keys=True))
        return
    result = {"environment": environment}
    if args.config is not None:
        config = load_launcher_config(args.config)
        if config.tracking.tensorboard and find_spec("tensorboard") is None:
            raise ImportError(
                "tracking.tensorboard=true requires: "
                "python -m pip install 'lm-cl[tracking]'"
            )
        data_contract = resolve_data_contract(
            config,
            full_checksum_validation=not args.manifest_only,
        )
        jobs = expand_job_specs(config, data_contract)
        assignments = allocate_job_slots(config, jobs)
        preflight = preflight_launch(config, jobs, assignments)
        preflight.pop("old_resolved", None)
        result["launch_preflight"] = preflight
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
