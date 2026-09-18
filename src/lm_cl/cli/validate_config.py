from __future__ import annotations

import argparse
import json

from lm_cl.config import load_experiment_config, save_resolved_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate an lm-cl experiment YAML")
    parser.add_argument("config")
    parser.add_argument("--output", help="Write fully resolved YAML")
    args = parser.parse_args()
    config = load_experiment_config(args.config)
    if args.output:
        save_resolved_config(config, args.output)
    print(
        json.dumps(
            {
                "status": "valid",
                "run_name": config.run_name,
                "model": config.model.name,
                "variant": config.variant.name,
                "planned_slow_steps": config.training.planned_slow_steps(
                    config.variant.slow_update_period_k
                ),
                "warmup_slow_steps": config.training.warmup_slow_steps(
                    config.variant.slow_update_period_k
                ),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
