from __future__ import annotations

import argparse
import json

import torch

from lm_cl.config import load_experiment_config
from lm_cl.data import synthetic_batch
from lm_cl.models import ZyphraTransformer
from lm_cl.training import set_deterministic_seed


def _device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(requested)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a synthetic forward/backward smoke")
    parser.add_argument("config")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive")

    config = load_experiment_config(args.config)
    if config.variant.memory_enabled:
        raise NotImplementedError("Memory variants are not implemented before Phase 5")
    set_deterministic_seed(
        config.runtime.seed,
        deterministic_algorithms=config.runtime.deterministic_algorithms,
    )
    device = _device(config.runtime.device)
    model = ZyphraTransformer(config.model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.model.learning_rate,
        betas=(config.training.adam_beta1, config.training.adam_beta2),
        eps=config.training.adam_epsilon,
        weight_decay=config.training.weight_decay,
    )
    last = None
    for step in range(args.steps):
        batch = synthetic_batch(
            config.data,
            args.batch_size,
            start_index=step * args.batch_size,
        )
        optimizer.zero_grad(set_to_none=True)
        output = model(
            batch["input_ids"].to(device),
            batch["labels"].to(device),
            ignore_index=config.data.ignore_index,
        )
        if output.mean_loss is None:
            raise RuntimeError("Synthetic smoke expected a loss")
        output.mean_loss.backward()
        optimizer.step()
        last = {
            "step": step + 1,
            "loss_sum": float(output.loss_sum.detach().cpu()),
            "mean_loss": float(output.mean_loss.detach().cpu()),
            "target_count": int(output.target_count.detach().cpu()),
        }
    print(json.dumps({"device": str(device), **(last or {})}, sort_keys=True))


if __name__ == "__main__":
    main()
