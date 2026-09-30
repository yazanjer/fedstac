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


# ---------------------------------------------------------------------------------------------
# v0.5.0 — Arm B: fixed samples per client (confounder control for the client-count study)
# ---------------------------------------------------------------------------------------------

def nested_stratified_subsample(y: np.ndarray, m_total: int, seed: int) -> np.ndarray:
    """Class-stratified subsample of size ``m_total`` that is nested in ``m_total`` for a fixed seed.

    Every class is shuffled once with a seed that does not depend on ``m_total``; class c contributes
    its first m_c samples, with m_c from largest-remainder rounding of m_total * N_c / N. Because m_c is
    non-decreasing in m_total, the subsample for a smaller federation is contained in the subsample
    for a larger one (same seed), so Arm B federations of different K share their data where they can.
    """
    n = len(y)
    if m_total >= n:
        return np.arange(n)
    classes, counts = np.unique(y, return_counts=True)
    quota = m_total * counts / n
    m = np.floor(quota).astype(int)
    rem = m_total - m.sum()
    order = np.lexsort((classes, -(quota - m)))            # largest remainder, ties by class id
    m[order[:rem]] += 1
    out = []
    for c, mc in zip(classes, m):
        idx = np.flatnonzero(y == c)
        np.random.default_rng([seed, 5150, int(c)]).shuffle(idx)
        out.append(idx[:mc])
    return np.sort(np.concatenate(out))


def _ipf_equal_rows(A: np.ndarray, row: np.ndarray, col: np.ndarray, iters: int = 500, tol: float = 1e-9) -> np.ndarray:
    """Iterative proportional fitting of A (K x C) to row sums ``row`` and column sums ``col``."""
    A = A.astype(np.float64).copy()
    for _ in range(iters):
        A *= (row / np.maximum(A.sum(1), 1e-300))[:, None]
        A *= (col / np.maximum(A.sum(0), 1e-300))[None, :]
        if np.abs(A.sum(1) - row).max() < tol * max(1.0, row.max()):
            break
    return A


def _integerise(A: np.ndarray, row: np.ndarray, col: np.ndarray) -> np.ndarray:
    """Integer matrix with exact margins, close to A (floor, then largest fractional parts)."""
    F = np.floor(A).astype(np.int64)
    r = row - F.sum(1)
    c = col - F.sum(0)
    frac = A - F
    order = np.argsort(-frac, axis=None, kind="stable")
    for flat in order:
        k, j = divmod(int(flat), A.shape[1])
        if r[k] > 0 and c[j] > 0:
            F[k, j] += 1; r[k] -= 1; c[j] -= 1
        if r.sum() == 0:
            break
    # any residual (only when all positive-fraction cells are exhausted): fill greedily
    for k in np.flatnonzero(r > 0):
        for j in np.flatnonzero(c > 0):
            t = min(r[k], c[j])
            F[k, j] += t; r[k] -= t; c[j] -= t
            if r[k] == 0:
                break
    assert (F.sum(1) == row).all() and (F.sum(0) == col).all() and (F >= 0).all()
    return F


def equal_size_dirichlet_partition(y: np.ndarray, pool: np.ndarray, n_clients: int, n_per_client: int,
                                   alpha: float, seed: int) -> list[np.ndarray]:
    """Arm B partition: ``n_clients`` clients of exactly ``n_per_client`` samples, Dir(alpha) label skew.

    Class proportions across clients are drawn exactly as in the Arm A partition (for each class c,
    p_c ~ Dir(alpha 1_K)); the resulting K x C count matrix is then rescaled by IPF so that every row
    sums to n_per_client while column sums stay equal to the class counts of the pool, and rounded to
    integers with exact margins. The skew pattern of the Dirichlet draw is preserved up to the row
    rescaling.
    """
    ypool = y[pool]
    classes, col = np.unique(ypool, return_counts=True)
    K = n_clients
    rng = np.random.default_rng([seed, 7070, K])
    A = np.stack([rng.dirichlet(np.full(K, alpha)) * mc for mc in col], axis=1)   # K x C
    A += 1e-9 * col[None, :] / K                                                    # keeps IPF feasible
    row = np.full(K, n_per_client, dtype=np.int64)
    if row.sum() != col.sum():
        raise ValueError("pool size must equal n_clients * n_per_client")
    F = _integerise(_ipf_equal_rows(A, row.astype(np.float64), col.astype(np.float64)), row, col)
    parts = [[] for _ in range(K)]
    for j, c in enumerate(classes):
        idx = pool[ypool == c].copy()
        rng.shuffle(idx)
        cuts = np.cumsum(F[:, j])[:-1]
        for k, part in enumerate(np.split(idx, cuts)):
            parts[k].append(part)
    return [np.sort(np.concatenate(p)) for p in parts]


def build_federation_armb(y_fine: np.ndarray, n_clients: int, n_per_client: int, alpha: float, seed: int,
                          val: float = 0.1, test: float = 0.2) -> list[dict]:
    pool = nested_stratified_subsample(y_fine, n_clients * n_per_client, seed)
    parts = equal_size_dirichlet_partition(y_fine, pool, n_clients, n_per_client, alpha, seed)
    return [split_client(p, y_fine, seed * 1000 + k, val, test) for k, p in enumerate(parts)]


def label_skew_stats(splits: list[dict], y: np.ndarray, n_classes: int) -> dict:
    """Heterogeneity summary of a federation (training splits): mean total-variation distance of the
    client label distributions to the pooled one, mean number of classes present, size dispersion."""
    cnt = np.stack([np.bincount(y[s["train"]], minlength=n_classes) for s in splits]).astype(np.float64)
    n = cnt.sum(1)
    p = cnt / np.maximum(n[:, None], 1)
    g = cnt.sum(0) / cnt.sum()
    tv = 0.5 * np.abs(p - g[None, :]).sum(1)
    return {"mean_tv_to_global": float(tv.mean()), "mean_classes_present": float((cnt > 0).sum(1).mean()),
            "train_size_min": int(n.min()), "train_size_median": float(np.median(n)), "train_size_max": int(n.max()),
            "train_size_cv": float(n.std() / n.mean())}
