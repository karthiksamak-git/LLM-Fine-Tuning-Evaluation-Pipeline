"""Deterministic seeding for python, numpy and torch."""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int) -> int:
    """Seed every RNG we use and return the seed (for logging)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(False)  # CPU kernels with deterministic mode are slow
    except ImportError:  # pragma: no cover - torch is a hard dependency in practice
        pass
    return seed


def seeded_generator(seed: int) -> np.random.Generator:
    """A numpy Generator isolated from the global RNG."""
    return np.random.default_rng(seed)
