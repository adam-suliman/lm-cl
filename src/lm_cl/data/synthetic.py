from __future__ import annotations

import torch
from torch.utils.data import Dataset

from lm_cl.config import DataConfig


class SyntheticTokenDataset(Dataset[dict[str, torch.Tensor]]):
    """Index-deterministic synthetic next-token sequences."""

    def __init__(self, config: DataConfig):
        config.validate()
        self.config = config

    def __len__(self) -> int:
        return self.config.num_sequences

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.config.seed + index)
        input_ids = torch.randint(
            0,
            self.config.vocab_size,
            (self.config.sequence_length,),
            generator=generator,
            dtype=torch.long,
        )
        labels = input_ids.clone()
        if self.config.mask_probability:
            mask = torch.rand(
                self.config.sequence_length,
                generator=generator,
            ) < self.config.mask_probability
            labels[mask] = self.config.ignore_index
        return {"input_ids": input_ids, "labels": labels}


def synthetic_batch(
    config: DataConfig,
    batch_size: int,
    start_index: int = 0,
) -> dict[str, torch.Tensor]:
    dataset = SyntheticTokenDataset(config)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if start_index < 0 or start_index + batch_size > len(dataset):
        raise ValueError("Requested batch lies outside synthetic dataset")
    samples = [dataset[index] for index in range(start_index, start_index + batch_size)]
    return {
        "input_ids": torch.stack([sample["input_ids"] for sample in samples]),
        "labels": torch.stack([sample["labels"] for sample in samples]),
    }
