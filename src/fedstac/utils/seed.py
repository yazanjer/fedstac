"""Determinism helpers. Call set_seed before any tensor is created."""
from __future__ import annotations

import os
import random

import numpy as np


def configure_env(cpu_threads: int | None = None) -> None:
    """Environment flags that must be set before CUDA initialises."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if cpu_threads is not None:
        os.environ["OMP_NUM_THREADS"] = str(cpu_threads)
        os.environ["MKL_NUM_THREADS"] = str(cpu_threads)


def set_seed(seed: int, cpu_threads: int | None = None) -> None:
    configure_env(cpu_threads)
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=False)
    if cpu_threads is not None:
        torch.set_num_threads(cpu_threads)


def rng_state() -> dict:
    import torch

    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict) -> None:
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
