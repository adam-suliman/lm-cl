from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_deterministic_seed(seed: int, deterministic_algorithms: bool = True) -> None:
    """Seed Python, NumPy, Torch CPU, and every visible CUDA device."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic_algorithms)
    if deterministic_algorithms:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
