"""Dirichlet label-skew partitioning and per-client splits."""
from __future__ import annotations

import numpy as np


def dirichlet_partition(y: np.ndarray, n_clients: int, alpha: float, seed: int,
                        min_size: int = 50, max_tries: int = 1000) -> list[np.ndarray]:
    """Standard Dir(alpha) label-skew partition (Hsu et al. 2019; Li et al. NIID-Bench).

    For every class, client proportions are drawn from Dir(alpha * 1_K); resampled until
    each client holds at least ``min_size`` samples.
    """
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    n = len(y)
    for _ in range(max_tries):
        buckets: list[list[np.ndarray]] = [[] for _ in range(n_clients)]
        for c in classes:
            idx = np.flatnonzero(y == c)
            rng.shuffle(idx)
            p = rng.dirichlet(np.full(n_clients, alpha))
            # balance heuristic from NIID-Bench: stop feeding clients already above n/K
            sizes = np.array([sum(len(b) for b in bl) for bl in buckets])
            p = p * (sizes < n / n_clients)
            if p.sum() == 0:
                p = np.full(n_clients, 1.0 / n_clients)
            p = p / p.sum()
            cuts = (np.cumsum(p) * len(idx)).astype(int)[:-1]
            for k, part in enumerate(np.split(idx, cuts)):
                buckets[k].append(part)
        parts = [np.sort(np.concatenate(b)) for b in buckets]
        if min(len(p) for p in parts) >= min_size:
            return parts
    raise RuntimeError("Could not satisfy min_size; lower min_size or raise alpha")


def split_client(idx: np.ndarray, y: np.ndarray, seed: int, val: float = 0.1, test: float = 0.2) -> dict:
    """Per-class train/val/test split inside one client (floor for val/test, rest train)."""
    rng = np.random.default_rng(seed)
    out = {"train": [], "val": [], "test": []}
    for c in np.unique(y[idx]):
        ci = idx[y[idx] == c].copy()
        rng.shuffle(ci)
        nt, nv = int(np.floor(test * len(ci))), int(np.floor(val * len(ci)))
        out["test"].append(ci[:nt])
        out["val"].append(ci[nt:nt + nv])
        out["train"].append(ci[nt + nv:])
    return {k: np.sort(np.concatenate(v)) if v else np.array([], dtype=np.int64) for k, v in out.items()}


def build_federation(y_fine: np.ndarray, n_clients: int, alpha: float, seed: int,
                     min_size: int = 50, val: float = 0.1, test: float = 0.2) -> list[dict]:
    if n_clients == 1:
        parts = [np.arange(len(y_fine))]
    else:
        parts = dirichlet_partition(y_fine, n_clients, alpha, seed, min_size=min_size)
    return [split_client(p, y_fine, seed * 1000 + k, val, test) for k, p in enumerate(parts)]
