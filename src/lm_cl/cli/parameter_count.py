from __future__ import annotations

import argparse
import json

from lm_cl.config import load_model_config
from lm_cl.models import ZyphraTransformer


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect model parameter counts")
    parser.add_argument("config", help="Model YAML file")
    args = parser.parse_args()
    config = load_model_config(args.config)
    model = ZyphraTransformer(config)
    breakdown = model.parameter_breakdown()
    result = {
        "model": config.name,
        "token_embeddings": breakdown.token_embeddings,
        "positional_embeddings": breakdown.positional_embeddings,
        "non_embedding": breakdown.non_embedding,
        "total": breakdown.total,
        "matches_expected": (
            breakdown.non_embedding == config.expected_non_embedding_parameters
            and breakdown.total == config.expected_total_parameters
        ),
    }
    print(json.dumps(result, sort_keys=True))
    if not result["matches_expected"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
