"""Collect results and emit booktabs tables, statistics and figures into outputs/paper/.
Usage: python scripts/analyze.py --out outputs
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from fedstac.evaluation import plotting as P  # noqa: E402
from fedstac.evaluation.stats import friedman_nemenyi, paired_vs_reference  # noqa: E402

BASE = ["centralised", "fedavg", "fedprox", "scaffold", "fedbn", "fedlc", "fedrs", "statavg", "fedstac"]
SFS_VARIANTS = ["fedprox_sfs", "scaffold_sfs", "fedbn_sfs", "fedlc_sfs", "fedrs_sfs"]
MAIN = BASE[:-1] + SFS_VARIANTS + ["fedstac"]
PRETTY = {"centralised": "Centralised", "fedavg": "FedAvg", "fedprox": "FedProx", "scaffold": "SCAFFOLD",
          "fedbn": "FedBN", "fedlc": "FedLC", "fedrs": "FedRS", "statavg": "StatAvg", "fedstac": "FedSTAC (ours)",
          "fedprox_sfs": "FedProx + SFS", "scaffold_sfs": "SCAFFOLD + SFS", "fedbn_sfs": "FedBN + SFS",
          "fedlc_sfs": "FedLC + SFS", "fedrs_sfs": "FedRS + SFS"}
DS_NAMES = {"ciciot2023": "CICIoT2023", "edgeiiot": "Edge-IIoTset", "synthetic": "Synthetic"}
DS = dict(DS_NAMES)
LAB = {"fine": "fine", "grouped": "grouped"}
METRICS = {"macro_f1": "Macro-F1", "accuracy": "Accuracy", "balanced_accuracy": "Bal. acc.",
           "tail_recall": "Tail recall", "worst_client_f1_p10": "Worst-client F1 (P10)"}


def load(out: Path, suite: str) -> pd.DataFrame:
    p = out / "index" / f"{suite}.json"
    if not p.exists():
        return pd.DataFrame()
    rows = []
    for e in json.loads(p.read_text()):
        r = out / "runs" / e["run_id"] / "results.json"
        if not r.exists():
            continue
        res = json.loads(r.read_text())
        row = dict(e)
        row.update({k: v for k, v in res["test"].items() if isinstance(v, (int, float))})
        row.update({f"prof_{k}": v for k, v in res["profile"].items()})
        row["best_round"] = res["best_round"]
        row["_res"] = res
        rows.append(row)
    return pd.DataFrame(rows)


def fmt(m, s, bold=False, pct=True):
    k = 100 if pct else 1
    txt = f"{m*k:.2f} $\\pm$ {s*k:.2f}"
    return f"\\textbf{{{txt}}}" if bold else txt


def write_tex(df: pd.DataFrame, path: Path, caption: str, label: str, colfmt: str | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    body = df.to_latex(index=False, escape=False, column_format=colfmt or "l" + "c" * (df.shape[1] - 1))
    body = body.replace("\\begin{tabular}", "\\begin{tabular}")
    tex = ("\\begin{table*}[t]\n\\centering\n\\caption{" + caption + "}\n\\label{" + label + "}\n\\small\n" + body + "\\end{table*}\n")
    path.write_text(tex)
    df.to_csv(path.with_suffix(".csv"), index=False)


def main_table(df, outp):
    cols = [(d, l) for d in DS for l in ("fine", "grouped")]
    rows = []
    agg = df.groupby(["method", "dataset", "labels"])["macro_f1"].agg(["mean", "std", "count"])
    best = {}
    for c in cols:
        fl = [m for m in MAIN if m != "centralised" and (m, *c) in agg.index]
        if fl:
            best[c] = max(fl, key=lambda m: agg.loc[(m, *c), "mean"])
    for m in MAIN:
        r = {"Method": PRETTY[m]}
        for c in cols:
            if (m, *c) in agg.index:
                a = agg.loc[(m, *c)]
                r[f"{DS[c[0]]} ({c[1]})"] = fmt(a["mean"], a["std"], best.get(c) == m)
            else:
                r[f"{DS[c[0]]} ({c[1]})"] = "--"
        rows.append(r)
    write_tex(pd.DataFrame(rows), outp / "tables/main_macro_f1.tex",
              "Test macro-F1 (\\%, mean $\\pm$ std over five seeds; Dirichlet $\\alpha=0.1$, $K=20$, 50\\% participation). "
              "Best federated result in bold; the centralised model is a pooled-data reference.", "tab:main")


def secondary_table(df, outp):
    rows = []
    for d in DS:
        for m in MAIN:
            s = df[(df.method == m) & (df.dataset == d) & (df.labels == "fine")]
            if s.empty:
                continue
            r = {"Dataset": DS[d], "Method": PRETTY[m]}
            for k, name in METRICS.items():
                if k == "macro_f1":
                    continue
                r[name] = fmt(s[k].mean(), s[k].std())
            r["Upload (MB)"] = f"{s['prof_comm_up_MB'].mean():.1f}"
            rows.append(r)
    write_tex(pd.DataFrame(rows), outp / "tables/secondary_fine.tex",
              "Secondary metrics on the fine-grained tasks (\\%, mean $\\pm$ std over five seeds) and total client upload.", "tab:secondary")


def _stats_tex(show, path, caption, label):
    tex = pd.DataFrame({
        "Baseline": show.baseline.map(PRETTY),
        "$\\Delta$ macro-F1 (pp)": [f"{a*100:+.2f} [{b*100:+.2f}, {c*100:+.2f}]" for a, b, c in zip(show.mean_diff, show.ci_low, show.ci_high)],
        "W/T/L": [f"{w}/{t}/{l}" for w, t, l in zip(show.wins, show.ties, show.losses)],
        "$p$ (Wilcoxon)": [f"{p:.2e}" for p in show.p_wilcoxon],
        "$p_{\\text{Holm}}$": [f"{p:.2e}" for p in show.p_holm],
        "$r_{rb}$": [f"{r:.2f}" for r in show.rank_biserial],
    })
    write_tex(tex, path, caption, label)


def stats_tables(df, outp):
    """Primary: per label granularity, paired over (dataset, seed) blocks — independent partitions.
    Secondary: pooled over both granularities (fine/grouped share partitions, so p is optimistic)."""
    res = {}
    fed = df[df.method.isin([m for m in MAIN if m != "centralised"])]
    rows = []
    for lab in ("fine", "grouped"):
        g = fed[fed.labels == lab]
        if g.empty:
            continue
        for metric in ["macro_f1", "tail_recall", "worst_client_f1_p10"]:
            t = paired_vs_reference(g, metric, "fedstac", ["dataset", "seed"])
            t.insert(0, "metric", metric); t.insert(0, "labels", lab)
            rows.append(t)
        show = rows[-3]
        _stats_tex(show, outp / f"tables/stats_fedstac_vs_baselines_{lab}.tex",
                   f"FedSTAC versus each baseline on test macro-F1 ({lab} labels), paired over {int(show.n_pairs.max())} "
                   "(dataset, seed) blocks: mean difference with bootstrap 95\\% CI, win/tie/loss count, two-sided Wilcoxon "
                   "signed-rank test, Holm-adjusted $p$ across baselines, and rank-biserial effect size.", f"tab:stats_{lab}")
        fr = friedman_nemenyi(g, "macro_f1", ["dataset", "seed"])
        res[f"friedman_{lab}"] = fr
        cd_diagram(fr, outp / f"figures/cd_diagram_{lab}")
    pd.concat(rows).to_csv(outp / "tables/stats_by_granularity.csv", index=False)
    pooled = paired_vs_reference(fed, "macro_f1", "fedstac", ["dataset", "labels", "seed"])
    pooled.to_csv(outp / "tables/stats_pooled_all.csv", index=False)
    per = []
    for (d, l), g in fed.groupby(["dataset", "labels"]):
        t = paired_vs_reference(g, "macro_f1", "fedstac", ["seed"])
        t.insert(0, "labels", l); t.insert(0, "dataset", d)
        per.append(t)
    pd.concat(per).to_csv(outp / "tables/stats_per_setting.csv", index=False)
    (outp / "tables/friedman.json").write_text(json.dumps(res, indent=1))
    return res


def cd_diagram(fr, path):
    """Demšar-style critical-difference diagram (rank 1 on the right)."""
    P.setup()
    ranks = fr["avg_ranks"]
    names = sorted(ranks, key=ranks.get)
    k, cd = len(names), fr["cd"]
    half = (k + 1) // 2
    fig, ax = P.plt.subplots(figsize=(P.SINGLE, 1.0 + 0.16 * half))
    lo, hi = 1, k
    ax.set_xlim(hi + 2.6, lo - 2.6)
    ax.axis("off")
    ax.hlines(0.8, lo, hi, color="k", lw=0.6)
    for i in range(lo, hi + 1):
        ax.vlines(i, 0.78, 0.82, color="k", lw=0.6)
        ax.text(i, 0.85, str(i), ha="center", va="bottom", fontsize=6)
    rs = [ranks[n] for n in names]
    cliques = []
    for i in range(k):
        j = i
        while j + 1 < k and rs[j + 1] - rs[i] < cd:
            j += 1
        if j > i and not any(a <= i and j <= b for a, b in cliques):
            cliques.append((i, j))
    y0 = 0.72 - 0.06 * len(cliques) - 0.08
    ax.set_ylim(y0 - 0.15 * (half - 1) - 0.1, 1.12)
    for j, n in enumerate(names):
        r = ranks[n]
        best_side = j < half
        row = j if best_side else k - 1 - j
        y = y0 - 0.15 * row
        xt = lo - 0.4 if best_side else hi + 0.4
        ax.plot([r, r, xt], [0.8, y, y], color="k", lw=0.5)
        ax.text(xt - 0.05 if best_side else xt + 0.05, y, f"{PRETTY.get(n, n)} ({r:.2f})",
                ha="left" if best_side else "right", va="center", fontsize=6)
    for c, (i, j) in enumerate(cliques):
        yy = 0.72 - 0.06 * c
        ax.hlines(yy, rs[i] - 0.03, rs[j] + 0.03, color="#922b21", lw=1.6)
    ax.hlines(1.0, lo, lo + cd, color="k", lw=1.0)
    ax.vlines([lo, lo + cd], 0.98, 1.02, color="k", lw=0.8)
    ax.text(lo + cd / 2, 1.03, f"CD = {cd:.2f}", ha="center", va="bottom", fontsize=6)
    P.save(fig, path)


def ablation_table(df, outp):
    order = ["abl_s0a0p0", "abl_s1a0p0", "abl_s0a1p0", "abl_s0a0p1", "abl_s1a1p0", "abl_s1a0p1", "abl_s0a1p1", "abl_s1a1p1"]
    cols = [(d, l) for d in DS for l in ("fine", "grouped")]
    agg = df.groupby(["method", "dataset", "labels"])["macro_f1"].agg(["mean", "std"])
    rows = []
    for m in order:
        s, a, p = (m[5] == "1"), (m[7] == "1"), (m[9] == "1")
        r = {"SFS": "\\checkmark" if s else "", "CAA": "\\checkmark" if a else "", "PCL": "\\checkmark" if p else ""}
        for c in cols:
            r[f"{DS[c[0]]} ({c[1]})"] = fmt(*agg.loc[(m, *c)]) if (m, *c) in agg.index else "--"
        rows.append(r)
    write_tex(pd.DataFrame(rows), outp / "tables/ablation.tex",
              "Ablation of FedSTAC components (test macro-F1, \\%, mean $\\pm$ std over five seeds). "
              "No component = FedAvg; SFS only = StatAvg; all three = FedSTAC.", "tab:ablation", "ccc" + "c" * len(cols))
    # component main effects (paired over dataset, labels, seed, other components)
    eff = []
    for comp, pos in (("SFS", 5), ("CAA", 7), ("PCL", 9)):
        on = df[df.method.str[pos] == "1"].copy(); off = df[df.method.str[pos] == "0"].copy()
        key = lambda x: x.method.str.slice(4).str.replace(r"(?<=[sap])\d", "", regex=True)
        on["pair"] = on.method.apply(lambda s: s[:pos] + "x" + s[pos + 1:]); off["pair"] = off.method.apply(lambda s: s[:pos] + "x" + s[pos + 1:])
        mg = on.merge(off, on=["pair", "dataset", "labels", "seed"], suffixes=("_on", "_off"))
        d = mg.macro_f1_on - mg.macro_f1_off
        from scipy import stats
        w = stats.wilcoxon(d) if (d != 0).any() else None
        eff.append({"component": comp, "n_pairs": len(d), "mean_gain_pp": d.mean() * 100, "median_gain_pp": d.median() * 100,
                    "p_wilcoxon": w.pvalue if w else 1.0})
    pd.DataFrame(eff).to_csv(outp / "tables/ablation_main_effects.csv", index=False)


def cost_table(df, outp, timing=None):
    if timing is not None and not timing.empty:
        df = timing
    rows = []
    for m in BASE:
        s = df[(df.method == m) & (df.labels == "fine")]
        if s.empty:
            continue
        rows.append({"Method": PRETTY[m], "Params": f"{int(s.prof_params.iloc[0]):,}",
                     "FLOPs/sample": f"{int(s.prof_flops_per_sample.iloc[0]):,}",
                     "Upload (MB)": f"{s.prof_comm_up_MB.mean():.1f}",
                     "Train time (s)": f"{s.prof_train_time_s.mean():.0f}" + (f" $\\pm$ {s.prof_train_time_s.std():.0f}" if len(s) > 1 else ""),
                     "Latency b=1 (ms)": f"{s.prof_latency_ms_b1.mean():.3f}",
                     "Peak VRAM (MB)": f"{s.prof_peak_vram_mb.mean():.0f}" if "prof_peak_vram_mb" in s else "--",
                     "Peak RSS (MB)": f"{s.prof_peak_rss_mb.mean():.0f}"})
    write_tex(pd.DataFrame(rows), outp / "tables/cost.tex",
              "Computational and communication cost (fine labels, mean over datasets and seeds; single GPU, see provenance).", "tab:cost")


def convergence_fig(df, outp):
    P.setup()
    fig, axes = P.plt.subplots(1, len(DS), figsize=(P.DOUBLE if len(DS) > 1 else P.SINGLE, 2.2), squeeze=False); axes = axes[0]
    for ax, d in zip(axes, DS):
        for m in [x for x in BASE if x != "centralised"]:
            s = df[(df.method == m) & (df.dataset == d) & (df.labels == "fine")]
            if s.empty:
                continue
            curves = pd.concat([pd.DataFrame(r["curve"]).set_index("round")["val_macro_f1"] for r in s._res], axis=1)
            mu, sd = curves.mean(axis=1), curves.std(axis=1)
            c = P.color(m, MAIN)
            ax.plot(mu.index, mu * 100, color=c, lw=1.4 if m == "fedstac" else 0.9, label=PRETTY[m],
                    ls="-" if m in ("fedstac", "statavg", "fedavg") else "--")
            ax.fill_between(mu.index, (mu - sd) * 100, (mu + sd) * 100, color=c, alpha=0.06, lw=0)
        ax.set_title(DS[d]); ax.set_xlabel("Communication round"); ax.set_ylabel("Validation macro-F1 (%)")
    fig.legend(*axes[0].get_legend_handles_labels(), ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    P.save(fig, outp / "figures/convergence_fine")


def sensitivity_figs(df, outp):
    P.setup()
    specs = [("federation.alpha", "Dirichlet $\\alpha$", True), ("federation.n_clients", "Number of clients $K$", True),
             ("method.beta", "Class-aware smoothing $\\beta$", True), ("method.tau", "Prior calibration $\\tau$", True),
             ("fl.participation", "Participation fraction", False)]
    for key, xl, logx in specs:
        if key not in df.columns:
            continue
        s = df[df[key].notna()]
        if s.empty:
            continue
        fig, axes = P.plt.subplots(1, len(DS), figsize=(P.DOUBLE if len(DS) > 1 else P.SINGLE, 2.0), squeeze=False); axes = axes[0]
        for ax, d in zip(axes, DS):
            for m in sorted(s.method.unique(), key=lambda x: MAIN.index(x) if x in MAIN else 99):
                g = s[(s.dataset == d) & (s.method == m)].groupby(key)["macro_f1"].agg(["mean", "std"]).sort_index()
                if g.empty:
                    continue
                x = g.index.to_numpy(float)
                if logx and (x <= 0).any():
                    x = np.where(x <= 0, x[x > 0].min() / 10 if (x > 0).any() else 1e-3, x)
                c = P.color(m, MAIN)
                ax.errorbar(x, g["mean"] * 100, yerr=g["std"] * 100, color=c, marker="o", capsize=2, lw=1, label=PRETTY.get(m, m))
            if logx:
                ax.set_xscale("log")
            ax.set_title(DS[d]); ax.set_xlabel(xl); ax.set_ylabel("Test macro-F1 (%)")
        axes[-1].legend(loc="best")
        P.save(fig, outp / f"figures/sensitivity_{key.split('.')[-1]}")
        s.groupby([key, "dataset", "method"])["macro_f1"].agg(["mean", "std", "count"]).reset_index().to_csv(
            outp / f"tables/sensitivity_{key.split('.')[-1]}.csv", index=False)


def per_class_fig(df, outp):
    P.setup()
    fig, axes = P.plt.subplots(len(DS), 1, figsize=(P.DOUBLE, 2.1 * len(DS)), squeeze=False); axes = axes[:, 0]
    for ax, d in zip(axes, DS):
        sub = df[(df.dataset == d) & (df.labels == "fine")]
        if sub.empty:
            continue
        names = sub._res.iloc[0]["class_names"]
        counts = np.array(sub._res.iloc[0]["train_class_counts"])
        order = np.argsort(counts)
        methods = ["fedavg", "statavg", "fedlc", "fedstac"]
        wdt = 0.8 / len(methods)
        for i, m in enumerate(methods):
            s = sub[sub.method == m]
            if s.empty:
                continue
            rec = np.mean([r["test"]["per_class_recall"] for r in s._res], axis=0)[order]
            ax.bar(np.arange(len(order)) + i * wdt, rec * 100, width=wdt, color=P.color(m, MAIN), label=PRETTY[m])
        ax.set_xticks(np.arange(len(order)) + 0.4)
        ax.set_xticklabels([names[j] for j in order], rotation=70, ha="right", fontsize=5)
        ax.set_ylabel("Recall (%)"); ax.set_title(f"{DS[d]} (classes sorted by training frequency)")
    axes[0].legend(ncol=4, loc="upper left")
    P.save(fig, outp / "figures/per_class_recall")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args()
    out = Path(a.out)
    outp = out / "paper"
    (outp / "tables").mkdir(parents=True, exist_ok=True)
    main_df = load(out, "main")
    global DS
    present = set()
    for suite in ("main", "ablation", "sensitivity"):
        d = load(out, suite)
        if not d.empty:
            present |= set(d.dataset)
    DS = {k: v for k, v in DS_NAMES.items() if k in present}
    summary = {}
    if not main_df.empty:
        main_table(main_df, outp); secondary_table(main_df, outp); cost_table(main_df, outp, load(out, "timing"))
        summary.update(stats_tables(main_df, outp)); convergence_fig(main_df, outp); per_class_fig(main_df, outp)
        summary["n_main_runs"] = len(main_df)
    abl = load(out, "ablation")
    if not abl.empty:
        ablation_table(abl, outp); summary["n_ablation_runs"] = len(abl)
    sens = load(out, "sensitivity")
    if not sens.empty:
        sensitivity_figs(sens, outp); summary["n_sensitivity_runs"] = len(sens)
    (outp / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
