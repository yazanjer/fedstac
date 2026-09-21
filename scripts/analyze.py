"""Collect results and emit booktabs tables, statistics and figures into outputs/paper/.
Usage: python scripts/analyze.py --out outputs
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from fedstac.evaluation import plotting as P  # noqa: E402
from fedstac.evaluation.stats import friedman_nemenyi, holm, paired_vs_reference  # noqa: E402

PROPOSED = "fedstap"
# Scope of the claim: methods that deliver one global model. FedBN keeps BatchNorm layers local, so it has
# no single global model; it is reported as a personalised reference and excluded from ranks and tests.
GLOBAL = ["fedavg", "fedprox", "scaffold", "fedlc", "fedrs", "statavg",
          "fedprox_sfs", "scaffold_sfs", "fedlc_sfs", "fedrs_sfs", PROPOSED]
PERSONALISED = ["fedbn", "fedbn_sfs"]
REFERENCE = ["centralised"]
MAIN = GLOBAL + PERSONALISED + REFERENCE
BASE = ["centralised", "fedavg", "fedprox", "scaffold", "fedbn", "fedlc", "fedrs", "statavg", PROPOSED]
PRETTY = {"centralised": "Centralised", "fedavg": "FedAvg", "fedprox": "FedProx", "scaffold": "SCAFFOLD",
          "fedbn": "FedBN", "fedlc": "FedLC", "fedrs": "FedRS", "statavg": "StatAvg", "fedstap": "FedStaP (ours)",
          "fedstac": "SFS + CAA + PCL",
          "fedprox_sfs": "FedProx + SFS", "scaffold_sfs": "SCAFFOLD + SFS", "fedbn_sfs": "FedBN + SFS",
          "fedlc_sfs": "FedLC + SFS", "fedrs_sfs": "FedRS + SFS",
          "scaffold_pcl": "SCAFFOLD + PCL", "scaffold_sfs_pcl": "SCAFFOLD + SFS + PCL"}
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
    # a block-heading row spans the table: drop the empty cells pandas emits after its \multicolumn
    body = re.sub(r"(\\multicolumn\{\d+\}\{l\}\{.*?\}\})(?:\s*&\s*)+(\\\\)", r"\\midrule\n\1 \2", body)
    body = body.replace("\\midrule\n\\midrule", "\\midrule")
    tex = ("\\begin{table*}[t]\n\\centering\n\\caption{" + caption + "}\n\\label{" + label + "}\n\\small\n" + body + "\\end{table*}\n")
    path.write_text(tex)
    df.to_csv(path.with_suffix(".csv"), index=False)


def main_table(df, outp):
    cols = [(d, l) for d in DS for l in ("fine", "grouped")]
    rows = []
    agg = df.groupby(["method", "dataset", "labels"])["macro_f1"].agg(["mean", "std", "count"])
    best = {}
    for c in cols:
        fl = [m for m in GLOBAL if (m, *c) in agg.index]
        if fl:
            best[c] = max(fl, key=lambda m: agg.loc[(m, *c), "mean"])
    blocks = [("Global model", GLOBAL), ("Personalised reference (local BatchNorm)", PERSONALISED),
              ("Pooled-data reference", REFERENCE)]
    for title, members in blocks:
        present = [m for m in members if (df.method == m).any()]
        if not present:
            continue
        rows.append({"Method": f"\\multicolumn{{{len(cols) + 1}}}{{l}}{{\\textit{{{title}}}}}"})
        for m in present:
            rows.append(_main_row(m, cols, agg, best))
    out = pd.DataFrame(rows).fillna("")
    write_tex(out, outp / "tables/main_macro_f1.tex",
              "Test macro-F1 (\\%, mean $\\pm$ std over five seeds; Dirichlet $\\alpha=0.1$, $K=20$, 50\\% participation). "
              "Best global-model result in bold. FedBN keeps BatchNorm statistics and affine parameters on each client and is "
              "evaluated with them, so it yields no single global model; it is a personalised reference outside the comparison. "
              "The centralised model is trained on pooled data.", "tab:main")


def _main_row(m, cols, agg, best):
    r = {"Method": PRETTY[m]}
    for c in cols:
        if (m, *c) in agg.index:
            a = agg.loc[(m, *c)]
            r[f"{DS[c[0]]} ({c[1]})"] = fmt(a["mean"], a["std"], best.get(c) == m)
        else:
            r[f"{DS[c[0]]} ({c[1]})"] = "--"
    return r


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
    fed = df[df.method.isin(GLOBAL)]
    rows = []
    for lab in ("fine", "grouped"):
        g = fed[fed.labels == lab]
        if g.empty:
            continue
        for metric in ["macro_f1", "tail_recall", "worst_client_f1_p10"]:
            t = paired_vs_reference(g, metric, PROPOSED, ["dataset", "seed"])
            t.insert(0, "metric", metric); t.insert(0, "labels", lab)
            rows.append(t)
        show = rows[-3]
        _stats_tex(show, outp / f"tables/stats_fedstap_vs_baselines_{lab}.tex",
                   f"FedStaP versus each global-model baseline on test macro-F1 ({lab} labels), paired over {int(show.n_pairs.max())} "
                   "(dataset, seed) blocks: mean difference with bootstrap 95\\% CI, win/tie/loss count, two-sided Wilcoxon "
                   "signed-rank test, Holm-adjusted $p$ across baselines, and rank-biserial effect size.", f"tab:stats_{lab}")
        fr = friedman_nemenyi(g, "macro_f1", ["dataset", "seed"])
        res[f"friedman_{lab}"] = fr
        cd_diagram(fr, outp / f"figures/cd_diagram_{lab}")
    pd.concat(rows).to_csv(outp / "tables/stats_by_granularity.csv", index=False)
    pooled = paired_vs_reference(fed, "macro_f1", PROPOSED, ["dataset", "labels", "seed"])
    pooled.to_csv(outp / "tables/stats_pooled_all.csv", index=False)
    per = []
    for (d, l), g in fed.groupby(["dataset", "labels"]):
        t = paired_vs_reference(g, "macro_f1", PROPOSED, ["seed"])
        t.insert(0, "labels", l); t.insert(0, "dataset", d)
        per.append(t)
    pd.concat(per).to_csv(outp / "tables/stats_per_setting.csv", index=False)
    # gap to the personalised reference, reported for transparency (not part of the Holm family)
    ref = df[df.method.isin([PROPOSED] + PERSONALISED)]
    gaps = []
    for lab in ("fine", "grouped"):
        g = ref[ref.labels == lab]
        if g.method.nunique() > 1:
            t = paired_vs_reference(g, "macro_f1", PROPOSED, ["dataset", "seed"])
            t.insert(0, "labels", lab); gaps.append(t)
    if gaps:
        pd.concat(gaps).to_csv(outp / "tables/gap_to_personalised_reference.csv", index=False)
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
              "Component ablation (test macro-F1, \\%, mean $\\pm$ std over five seeds). SFS: shared feature statistics; "
              "CAA: class-aware aggregation of the classifier head; PCL: prior-calibrated local loss. No component = FedAvg; "
              "SFS only = StatAvg; SFS + PCL = FedStaP (proposed). CAA is retained as an evaluated design alternative; "
              "its main effect is negative (Table~\\ref{tab:ablation_effects}).", "tab:ablation", "ccc" + "c" * len(cols))
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
    eff = pd.DataFrame(eff)
    eff.to_csv(outp / "tables/ablation_main_effects.csv", index=False)
    write_tex(pd.DataFrame({"Component": eff.component, "Pairs": eff.n_pairs,
                            "Mean gain (pp)": [f"{v:+.2f}" for v in eff.mean_gain_pp],
                            "Median gain (pp)": [f"{v:+.2f}" for v in eff.median_gain_pp],
                            "$p$ (Wilcoxon)": [f"{v:.1e}" for v in eff.p_wilcoxon]}),
              outp / "tables/ablation_effects.tex",
              "Main effect of each component: macro-F1 with the component minus without it, paired over dataset, label "
              "granularity, seed and the state of the other two components.", "tab:ablation_effects")


PLUGIN = ["scaffold", "scaffold_sfs", "scaffold_pcl", "scaffold_sfs_pcl"]


def plugin_table(df, outp):
    """FedStaP's components added to SCAFFOLD: absolute scores and paired gains over plain SCAFFOLD."""
    cols = [(d, l) for d in DS for l in ("fine", "grouped")]
    agg = df.groupby(["method", "dataset", "labels"])["macro_f1"].agg(["mean", "std"])
    rows = []
    for m in PLUGIN:
        r = {"Method": PRETTY[m]}
        for c in cols:
            r[f"{DS[c[0]]} ({c[1]})"] = fmt(*agg.loc[(m, *c)]) if (m, *c) in agg.index else "--"
        rows.append(r)
    write_tex(pd.DataFrame(rows), outp / "tables/plugin_scaffold.tex",
              "FedStaP components as a plug-in to SCAFFOLD (test macro-F1, \\%, mean $\\pm$ std over five seeds). "
              "PCL uses FedStaP's tuned $\\tau$; the learning rate is SCAFFOLD's tuned value.", "tab:plugin")
    tests = []
    for lab in ("fine", "grouped"):
        g = df[df.labels == lab]
        part = []
        for m in PLUGIN[1:]:
            sub = g[g.method.isin([m, "scaffold"])]
            if sub.method.nunique() < 2:
                continue
            t = paired_vs_reference(sub, "macro_f1", m, ["dataset", "seed"])
            t.insert(0, "variant", m); t.insert(0, "labels", lab)
            part.append(t)
        if part:
            part = pd.concat(part)
            part["p_holm"] = holm(part["p_wilcoxon"].tolist())
            tests.append(part)
    if tests:
        pd.concat(tests).to_csv(outp / "tables/plugin_scaffold_tests.csv", index=False)


def diag_table(df, outp):
    """Learning-rate sweep for SCAFFOLD with and without SFS on the tuning seed (validation macro-F1)."""
    if "fl.lr" not in df.columns:
        return
    df = df.assign(val=[r["best_val_macro_f1"] for r in df._res],
                   curve_last=[pd.DataFrame(r["curve"])["val_macro_f1"].iloc[-1] for r in df._res])
    t = df.groupby(["dataset", "method", "fl.lr"])[["val", "curve_last", "best_round"]].mean().reset_index()
    t.to_csv(outp / "tables/diag_scaffold_sfs_lr.csv", index=False)


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
        for m in [x for x in BASE if x in GLOBAL]:
            s = df[(df.method == m) & (df.dataset == d) & (df.labels == "fine")]
            if s.empty:
                continue
            curves = pd.concat([pd.DataFrame(r["curve"]).set_index("round")["val_macro_f1"] for r in s._res], axis=1)
            mu, sd = curves.mean(axis=1), curves.std(axis=1)
            c = P.color(m, MAIN)
            ax.plot(mu.index, mu * 100, color=c, lw=1.4 if m == PROPOSED else 0.9, label=PRETTY[m],
                    ls="-" if m in (PROPOSED, "statavg", "fedavg") else "--")
            ax.fill_between(mu.index, (mu - sd) * 100, (mu + sd) * 100, color=c, alpha=0.06, lw=0)
        ax.set_title(DS[d]); ax.set_xlabel("Communication round"); ax.set_ylabel("Validation macro-F1 (%)")
    fig.legend(*axes[0].get_legend_handles_labels(), ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    P.save(fig, outp / "figures/convergence_fine")


def sensitivity_figs(df, outp):
    P.setup()
    specs = [("federation.alpha", "Dirichlet $\\alpha$", True), ("federation.n_clients", "Number of clients $K$", True),
             ("method.tau", "Prior calibration $\\tau$", True),
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
        methods = ["fedavg", "statavg", "fedlc", PROPOSED]
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
    for suite in ("main", "ablation", "sensitivity", "plugin"):
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
    plug = load(out, "plugin")
    if not plug.empty:
        plugin_table(plug, outp); summary["n_plugin_runs"] = len(plug)
    diag = load(out, "diag_scaffold_sfs")
    if not diag.empty:
        diag_table(diag, outp); summary["n_diag_runs"] = len(diag)
    sens = load(out, "sensitivity")
    if not sens.empty:
        sensitivity_figs(sens, outp); summary["n_sensitivity_runs"] = len(sens)
    (outp / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
