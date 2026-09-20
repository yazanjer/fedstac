"""Client-side standardisation. Statistics always come from training splits only."""
from __future__ import annotations

import numpy as np

EPS = 1e-6


def sufficient_stats(x: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    x64 = x.astype(np.float64)
    return len(x), x64.sum(0), (x64 ** 2).sum(0)


def stats_to_scaler(n: int, s: np.ndarray, ss: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = s / max(n, 1)
    var = np.maximum(ss / max(n, 1) - mu ** 2, 0.0)
    sd = np.sqrt(var)
    sd = np.where(sd < EPS, 1.0, sd)  # constant on this training split: centre only (as sklearn)
    return mu.astype(np.float32), sd.astype(np.float32)


def local_scalers(train_x: list[np.ndarray]) -> list[tuple[np.ndarray, np.ndarray]]:
    return [stats_to_scaler(*sufficient_stats(x)) for x in train_x]


def global_scaler(train_x: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """SFS: server sums the clients' (n, sum, sum of squares) uploads."""
    n, s, ss = 0, 0.0, 0.0
    for x in train_x:
        nk, sk, ssk = sufficient_stats(x)
        n, s, ss = n + nk, s + sk, ss + ssk
    return stats_to_scaler(n, s, ss)
