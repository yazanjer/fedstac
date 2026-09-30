"""Pre-registered analysis of the v0.5.0 scaling study (report/technical_report.md, Section 3).

Reads per-run results.json files listed in index/scaling_<stage>.json and writes
<dst>/scaling_analysis.json plus CSV tables. Usage:
  python scripts/analyze_scaling.py --out outputs --stage full --dst results/v050
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

BOOT, BOOT_SEED = 10_000, 20260930


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for i, j in enumerate(order):
        running = max(running, min(1.0, (len(p) - i) * p[j]))
        adj[j] = running
    return adj


def boot_ci(x, seed=BOOT_SEED):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    m = rng.choice(x, size=(BOOT, len(x)), replace=True).mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def wil(x):
    x = np.asarray(x, float)
    if np.allclose(x, 0):
        return 1.0
    return float(wilcoxon(x, alternative="two-sided", method="exact" if len(x) <= 25 else "auto").pvalue)


def slope(xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    return float(np.polyfit(xs, ys, 1)[0])


def load(out: Path, stage: str) -> pd.DataFrame:
    idx = json.loads((out / "index" / f"scaling_{stage}.json").read_text())
    rows = []
    for e in idx:
        if e.get("block") == "tuning":
            continue
        f = out / "runs" / e["run_id"] / "results.json"
        if not f.exists():
            continue
        r = json.loads(f.read_text())
        d = [x for x in r.get("diagnostics", []) if x["round"] >= 10]
        mean = lambda k: float(np.mean([x[k] for x in d if k in x])) if any(k in x for x in d) else np.nan
        rows.append({**{k: e.get(k) for k in ("block", "dataset", "arm", "K", "method", "mode", "seed")},
                     "E": e.get("fl.local_epochs", 1), "alpha": e.get("federation.alpha", 0.1),
                     "rounds": e.get("fl.rounds", 50), "run_id": e["run_id"],
                     "f1": 100 * r["test"]["macro_f1"], "sizes": tuple(r["client_train_sizes"]),
                     "steps": float(np.mean(r.get("local_steps_per_round") or [np.nan])),
                     "corr_err": mean("corr_rel_err_mean"), "noise": mean("ideal_noise_rel_mean"),
                     "stale": mean("staleness_mean"), "cos": mean("corr_cos_mean"),
                     "dissim": mean("grad_dissimilarity")})
    return pd.DataFrame(rows)


def gaps(df, block="main", a="fedstap", b="scaffold", extra=("E",)):
    key = ["dataset", "arm", "mode", "K", "seed", *extra]
    A = df[(df.block == block) & (df.method == a)].set_index(key)
    B = df[(df.block == block) & (df.method == b)].set_index(key)
    j = A.join(B, lsuffix="_a", rsuffix="_b", how="inner")
    assert (j.sizes_a == j.sizes_b).all(), "paired runs do not share a partition"
    j["gap"] = j.f1_a - j.f1_b
    return j.reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--stage", default="full")
    ap.add_argument("--dst", default="results/v050")
    a = ap.parse_args()
    out, dst = Path(a.out), Path(a.dst); dst.mkdir(parents=True, exist_ok=True)
    df = load(out, a.stage)
    df.drop(columns=["sizes"]).to_csv(dst / "scaling_runs.csv", index=False)
    res = {"n_runs": int(len(df))}

    # cell means for every method (descriptive)
    cm = df[df.block == "main"].groupby(["dataset", "arm", "mode", "K", "method"]).f1.agg(["mean", "std", "count"]).reset_index()
    cm.to_csv(dst / "cell_means.csv", index=False)

    g = gaps(df)
    g[["dataset", "arm", "mode", "K", "seed", "gap"]].to_csv(dst / "gaps_main.csv", index=False)

    # 1. per-seed slopes; 2. per-K tests; 3. crossover
    fam_rows, perk_rows, cross_rows, slopes = [], [], [], {}
    for (ds, arm, mode), sub in g.groupby(["dataset", "arm", "mode"]):
        per_seed = {}
        for s, ss in sub.groupby("seed"):
            ss = ss.sort_values("K")
            if len(ss) < 3:
                continue
            per_seed[s] = slope(np.log2(ss.K), ss.gap)
            x, y = np.log2(ss.K.values), ss.gap.values
            if y[0] >= 0:
                cross = "<=10"
            elif (y < 0).all():
                cross = ">200"
            else:
                i = int(np.argmax(y >= 0))
                t = y[i - 1] / (y[i - 1] - y[i])
                cross = float(2 ** (x[i - 1] + t * (x[i] - x[i - 1])))
            cross_rows.append({"dataset": ds, "arm": arm, "mode": mode, "seed": s, "crossover_K": cross})
        b = np.array(list(per_seed.values()))
        slopes[(ds, arm, mode)] = per_seed
        lo, hi = boot_ci(b)
        fam_rows.append({"dataset": ds, "arm": arm, "mode": mode, "n_seeds": len(b), "slope_mean_pp_per_doubling": b.mean(),
                         "ci_lo": lo, "ci_hi": hi, "p": wil(b)})
        ks = []
        for K, kk in sub.groupby("K"):
            lo2, hi2 = boot_ci(kk.gap.values)
            ks.append({"dataset": ds, "arm": arm, "mode": mode, "K": K, "n": len(kk), "gap_mean": kk.gap.mean(),
                       "gap_sd": kk.gap.std(ddof=1), "ci_lo": lo2, "ci_hi": hi2, "p": wil(kk.gap.values),
                       "wins_fedstap": int((kk.gap > 0).sum())})
        for r, ph in zip(ks, holm([r["p"] for r in ks])):
            r["p_holm"] = ph
        perk_rows += ks
    fam = pd.DataFrame(fam_rows)
    if len(fam):
        fam["p_holm"] = holm(fam.p)
        fam["supported"] = (fam.ci_lo > 0) & (fam.p_holm < 0.05)
    fam.to_csv(dst / "test1_slopes.csv", index=False)
    pd.DataFrame(perk_rows).to_csv(dst / "test2_perK.csv", index=False)
    cr = pd.DataFrame(cross_rows); cr.to_csv(dst / "test3_crossover_per_seed.csv", index=False)
    summ = []
    for (ds, arm, mode), c in cr.groupby(["dataset", "arm", "mode"]) if len(cr) else []:
        num = [v for v in c.crossover_K if isinstance(v, float)]
        summ.append({"dataset": ds, "arm": arm, "mode": mode, "median": float(np.median(num)) if num else None,
                     "min": min(num) if num else None, "max": max(num) if num else None,
                     "n_le10": int((c.crossover_K == "<=10").sum()), "n_gt200": int((c.crossover_K == ">200").sum())})
    pd.DataFrame(summ).to_csv(dst / "test3_crossover.csv", index=False)

    # 4. confounder (A - B) and 5. staleness (cohort - rho, Arm B)
    def diff_test(pairs, name):
        rows = []
        for lbl, (k1, k2) in pairs.items():
            s1, s2 = slopes.get(k1, {}), slopes.get(k2, {})
            common = sorted(set(s1) & set(s2))
            if len(common) < 5:
                continue
            d = np.array([s1[s] - s2[s] for s in common])
            lo, hi = boot_ci(d)
            rows.append({"contrast": lbl, "n": len(d), "mean_diff": d.mean(), "ci_lo": lo, "ci_hi": hi, "p": wil(d)})
        t = pd.DataFrame(rows)
        if len(t):
            t["p_holm"] = holm(t.p)
        t.to_csv(dst / f"{name}.csv", index=False)
        return t
    dss = sorted(g.dataset.unique())
    diff_test({f"{ds}|{m}|A-B": ((ds, "A", m), (ds, "B", m)) for ds in dss for m in ("rho", "cohort")}, "test4_confounder")
    diff_test({f"{ds}|B|cohort-rho": ((ds, "B", "cohort"), (ds, "B", "rho")) for ds in dss}, "test5_staleness")

    # 6. Arm C: gap vs log2 E at K in {20, 200}
    rows = []
    if (df.block == "arm_c").any():
        main_b = df[(df.block == "main") & (df.arm == "B") & (df["mode"] == "rho") & df.K.isin([20, 200])]
        both = pd.concat([df[df.block == "arm_c"], main_b])
        both = both.assign(block="c")
        gc = gaps(both, block="c")
        for (ds, K), sub in gc.groupby(["dataset", "K"]):
            b = np.array([slope(np.log2(ss.E), ss.gap) for _, ss in sub.groupby("seed") if ss.E.nunique() >= 3])
            if len(b) < 5:
                continue
            lo, hi = boot_ci(b)
            rows.append({"dataset": ds, "K": K, "n": len(b), "slope_mean_pp_per_doubling_E": b.mean(), "ci_lo": lo, "ci_hi": hi, "p": wil(b)})
    t6 = pd.DataFrame(rows)
    if len(t6):
        t6["p_holm"] = holm(t6.p)
    t6.to_csv(dst / "test6_local_steps.csv", index=False)

    # 7. mechanism association (SCAFFOLD runs of the main block)
    rows = []
    sc = g.rename(columns={"corr_err_b": "err", "steps_b": "steps", "stale_b": "stale", "noise_b": "noise"})
    for ds, sub in sc.groupby("dataset"):
        for xn, yn in (("err", "steps"), ("err", "stale"), ("gap", "err")):
            rho = spearmanr(sub[xn], sub[yn]).statistic
            rng = np.random.default_rng(BOOT_SEED)
            seeds = sub.seed.unique()
            bs = []
            for _ in range(2000):
                pick = rng.choice(seeds, size=len(seeds), replace=True)
                bb = pd.concat([sub[sub.seed == s] for s in pick])
                bs.append(spearmanr(bb[xn], bb[yn]).statistic)
            rows.append({"dataset": ds, "x": xn, "y": yn, "spearman": rho, "ci_lo": float(np.nanpercentile(bs, 2.5)),
                         "ci_hi": float(np.nanpercentile(bs, 97.5)), "n_runs": len(sub),
                         "noise_floor_mean": float(sub.noise.mean())})
    pd.DataFrame(rows).to_csv(dst / "test7_mechanism.csv", index=False)
    res["families"] = fam.to_dict(orient="records") if len(fam) else []
    (dst / "scaling_analysis.json").write_text(json.dumps(res, indent=1, default=str))
    print(fam.to_string(index=False) if len(fam) else "no complete families yet")


if __name__ == "__main__":
    main()
