"""Independent re-derivation of the v0.5.0 scaling numbers.

Deliberately imports nothing from `fedstac` or `scripts/`: it reads the per-run results.json files and the
run index, re-implements the pre-registered statistics from first principles (own Holm, own OLS slope,
own crossover interpolation; SciPy only for the exact Wilcoxon distribution) and compares them with the
tables written by scripts/analyze_scaling.py. Usage:
  python evidence/rederive_scaling.py --runs <dir with run_id/results.json or run_id.json> \
      --index <scaling_full.json> --tables <dir with analyze_scaling outputs> [--tol 1e-6]
"""
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from scipy.stats import wilcoxon


def read_run(root: Path, rid: str):
    for p in (root / rid / "results.json", root / f"{rid}.json"):
        if p.exists():
            return json.loads(p.read_text())
    return None


def ols_slope(x, y):
    n = len(x); mx = sum(x) / n; my = sum(y) / n
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / sum((a - mx) ** 2 for a in x)


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    out, run = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i])); out[i] = run
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True); ap.add_argument("--index", required=True)
    ap.add_argument("--tables", required=True); ap.add_argument("--tol", type=float, default=1e-6)
    a = ap.parse_args()
    runs, tables = Path(a.runs), Path(a.tables)
    idx = [e for e in json.loads(Path(a.index).read_text()) if e.get("block") == "main"]
    f1 = {}
    for e in idx:
        r = read_run(runs, e["run_id"])
        if r is None:
            continue
        f1[(e["dataset"], e["arm"], e["mode"], int(e["K"]), e["method"], int(e["seed"]))] = (100 * r["test"]["macro_f1"], tuple(r["client_train_sizes"]))
    gap = defaultdict(dict)
    for (ds, arm, mode, K, m, s), (v, sz) in f1.items():
        if m != "fedstap":
            continue
        o = f1.get((ds, arm, mode, K, "scaffold", s))
        if o is None:
            continue
        assert o[1] == sz, "pairing violated"
        gap[(ds, arm, mode)][(K, s)] = v - o[0]
    checks, fails = [], 0

    def check(name, mine, theirs):
        nonlocal fails
        ok = theirs is not None and abs(mine - theirs) <= a.tol * max(1.0, abs(mine))
        fails += (not ok)
        checks.append((name, mine, theirs, "OK" if ok else "FAIL"))

    t1 = {(r["dataset"], r["arm"], r["mode"]): r for r in csv.DictReader(open(tables / "test1_slopes.csv"))}
    t2 = {(r["dataset"], r["arm"], r["mode"], int(r["K"])): r for r in csv.DictReader(open(tables / "test2_perK.csv"))}
    fam_p = {}
    for fam, g in sorted(gap.items()):
        seeds = sorted({s for _, s in g})
        Ks = sorted({K for K, _ in g})
        b = []
        for s in seeds:
            pts = [(math.log2(K), g[(K, s)]) for K in Ks if (K, s) in g]
            if len(pts) >= 3:
                b.append(ols_slope([p[0] for p in pts], [p[1] for p in pts]))
        if not b:
            continue
        mean_b = sum(b) / len(b)
        p = 1.0 if all(abs(x) < 1e-12 for x in b) else float(wilcoxon(b, method="exact").pvalue)
        fam_p[fam] = p
        check(f"slope {fam}", mean_b, float(t1[fam]["slope_mean_pp_per_doubling"]) if fam in t1 else None)
        check(f"slope p {fam}", p, float(t1[fam]["p"]) if fam in t1 else None)
        ks_p = []
        for K in Ks:
            v = [g[(K, s)] for s in seeds if (K, s) in g]
            check(f"gap mean {fam} K={K}", sum(v) / len(v), float(t2[(*fam, K)]["gap_mean"]) if (*fam, K) in t2 else None)
            ks_p.append(1.0 if all(abs(x) < 1e-12 for x in v) else float(wilcoxon(v, method="exact").pvalue))
        for K, ph in zip(Ks, holm(ks_p)):
            check(f"gap p_holm {fam} K={K}", ph, float(t2[(*fam, K)]["p_holm"]) if (*fam, K) in t2 else None)
    fams = sorted(fam_p)
    for fam, ph in zip(fams, holm([fam_p[f] for f in fams])):
        check(f"slope p_holm {fam}", ph, float(t1[fam]["p_holm"]) if fam in t1 else None)
    out = Path(a.tables) / "rederivation_audit.csv"
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["quantity", "rederived", "analysis_table", "status"]); w.writerows(checks)
    print(f"{len(checks)} checks, {fails} failures -> {out}")
    raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    main()
