"""Paired significance tests, effect sizes and Friedman/Nemenyi ranking."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

# Nemenyi q_{0.05} (Demšar 2006), index = number of methods k
Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164,
       11: 3.219, 12: 3.268, 13: 3.313, 14: 3.354, 15: 3.391}


def holm(p: list[float]) -> list[float]:
    p = np.asarray(p, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for i, idx in enumerate(order):
        run = max(run, (m - i) * p[idx])
        adj[idx] = min(1.0, run)
    return adj.tolist()


def bootstrap_ci(d: np.ndarray, n: int = 10000, seed: int = 0, level: float = 0.95) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = rng.choice(d, size=(n, len(d)), replace=True).mean(1)
    lo, hi = np.percentile(means, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(lo), float(hi)


def rank_biserial(d: np.ndarray) -> float:
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = stats.rankdata(np.abs(d))
    wp, wm = r[d > 0].sum(), r[d < 0].sum()
    return float((wp - wm) / (wp + wm))


def paired_vs_reference(df: pd.DataFrame, metric: str, ref: str, block: list[str]) -> pd.DataFrame:
    """Reference method vs every other method, paired on `block` (e.g. dataset, labels, seed)."""
    piv = df.pivot_table(index=block, columns="method", values=metric)
    rows = []
    for m in piv.columns:
        if m == ref:
            continue
        pair = piv[[ref, m]].dropna()
        d = (pair[ref] - pair[m]).to_numpy()
        if len(d) < 2:
            continue
        w = stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided") if np.any(d != 0) else None
        t = stats.ttest_rel(pair[ref], pair[m])
        lo, hi = bootstrap_ci(d)
        rows.append({"baseline": m, "n_pairs": len(d), "mean_diff": d.mean(), "ci_low": lo, "ci_high": hi,
                     "wins": int((d > 0).sum()), "ties": int((d == 0).sum()), "losses": int((d < 0).sum()),
                     "wilcoxon_W": float(w.statistic) if w else np.nan, "p_wilcoxon": float(w.pvalue) if w else 1.0,
                     "rank_biserial": rank_biserial(d), "p_ttest": float(t.pvalue),
                     "cohen_dz": float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else np.nan})
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = holm(out["p_wilcoxon"].tolist())
    return out


def friedman_nemenyi(df: pd.DataFrame, metric: str, block: list[str]) -> dict:
    piv = df.pivot_table(index=block, columns="method", values=metric).dropna()
    k, n = piv.shape[1], piv.shape[0]
    ranks = piv.rank(axis=1, ascending=False).mean(0).sort_values()
    chi2, p = stats.friedmanchisquare(*[piv[c] for c in piv.columns])
    cd = Q05.get(k, np.nan) * np.sqrt(k * (k + 1) / (6.0 * n))
    return {"avg_ranks": ranks.to_dict(), "chi2": float(chi2), "p": float(p), "cd": float(cd), "n_blocks": n, "k": k}
