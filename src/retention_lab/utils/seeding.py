"""Deterministic seeding.

One shared experiment seed fans out into named streams so that adding a new
consumer of randomness never perturbs the draws of existing consumers. Data
order, model initialization, and dropout each read their own stream.
"""

from __future__ import annotations

import hashlib
import random

import numpy as np
import torch


def stream_seed(seed: int, name: str) -> int:
    """Derive a stable 63-bit seed for a named stream from the shared seed."""
    digest = hashlib.sha256(f"{seed}:{name}".encode()).digest()
    return int.from_bytes(digest[:8], "big") >> 1


def numpy_rng(seed: int, name: str) -> np.random.Generator:
    return np.random.default_rng(stream_seed(seed, name))


def torch_generator(seed: int, name: str) -> torch.Generator:
    gen = torch.Generator()
    gen.manual_seed(stream_seed(seed, name))
    return gen


def set_global_determinism(seed: int) -> None:
    """Seed every global RNG and force deterministic kernel selection."""
    random.seed(stream_seed(seed, "python"))
    np.random.seed(stream_seed(seed, "numpy") % (2**32))
    torch.manual_seed(stream_seed(seed, "torch"))
    torch.use_deterministic_algorithms(True)
