"""FedStaP v0.5.0 scaling study: expand configs/scaling/plan.yaml and execute it with a worker pool.

Stages
  smoke   two 5-round runs (Arm A K=10, Arm B K=200) that exercise the whole pipeline
  sanity  two v0.4.0 GPU runs re-executed on CPU; the gate fails if |delta| > 2 pp
  pilot   tuning for K in {10, 200} plus seed 0 of the main block at K in {10, 200}
  full    every block of the plan

Per-cell learning-rate tuning (dataset, arm, K, method) on seed 100, rho mode, with automatic grid
extension while an edge value wins; dependent runs are released as soon as their cell is tuned.
Resumable: runs with results.json are skipped (hash identity), interrupted runs resume from their
round checkpoint, tuning decisions are recomputed from stored results.

Usage: python scripts/run_scaling.py --stage pilot --workers 32 --out /workspace/outputs --data data/prepared
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from run_suite import identity_of, tuned_overrides  # noqa: E402

PLAN_PATH = Path(os.environ.get("FEDSTAP_PLAN", REPO / "configs/scaling/plan.yaml"))
PLAN = yaml.safe_load(PLAN_PATH.read_text())
N_TRAIN_FRAC = 0.7
POOL = {"ciciot2023": 1005585, "edgeiiot": 624105}


# ------------------------------------------------------------------ run specification
def cell_overrides(ds, arm, K, method, mode, seed, lr, extra=None):
    ov = {"dataset": ds, "task.labels": PLAN["labels"], "method": method, "seed": seed,
          "federation.n_clients": K, "federation.alpha": PLAN["alpha"], "federation.arm": arm,
          "fl.participation": PLAN["participation"]["rho"]}
    if arm == "B":
        ov["federation.n_per_client"] = PLAN["n_per_client"][ds]
    if mode == "cohort":
        ov["fl.cohort_size"] = PLAN["participation"]["cohort"]
    ov.update({k: v for k, v in tuned_overrides(method).items() if k != "fl.lr"})
    ov["fl.lr"] = lr
    ov.update(extra or {})
    return ov


def est_cost(ov) -> float:
    """Relative CPU cost: samples processed per run (sampled clients x samples x epochs x rounds)."""
    K = int(ov["federation.n_clients"])
    n = ov.get("federation.n_per_client") or POOL[ov["dataset"]] / K
    m = min(K, int(ov["fl.cohort_size"])) if ov.get("fl.cohort_size") else max(1, round(float(ov["fl.participation"]) * K))
    rounds = int(ov.get("fl.rounds", 50))
    E = int(ov.get("fl.local_epochs", 1))
    fscale = 1.0 if ov["dataset"] == "ciciot2023" else 1.3
    return (m * n * N_TRAIN_FRAC * E * rounds + 5e5 * K / 10) * fscale


class Job:
    def __init__(self, ov, label, common, needs=None):
        self.ov, self.label, self.needs = ov, label, needs      # needs: cell key whose lr must be known
        self.common = common
        self.run_id = None
        self.tries = 0

    def overrides(self):
        return [f"{k}={v}" for k, v in self.ov.items()] + self.common


# ------------------------------------------------------------------ plan expansion
def expand_blocks(stage, seeds_override=None):
    """Yield (block, ds, arm, K, method, mode, seed, extra, lr_from) for the requested stage."""
    Ks_all, methods_all = PLAN["K"], PLAN["methods"]
    out = []
    for bname, b in PLAN["blocks"].items():
        if stage == "pilot" and bname != "main":
            continue
        Ks = Ks_all if b["K"] == "all" else b["K"]
        methods = methods_all if b["methods"] == "all" else b["methods"]
        seeds = PLAN["claim_seeds"] if b["seeds"] == "claim" else PLAN["direction_seeds"]
        if stage == "pilot":
            Ks, seeds = [k for k in Ks if k in (10, 200)], [0]
        if seeds_override is not None:
            seeds = seeds_override
        sweep = b.get("sweep") or {}
        combos = [dict()]
        for key, vals in sweep.items():
            combos = [dict(c, **{key: v}) for c in combos for v in vals]
        for ds in PLAN["datasets"]:
            for arm in b["arms"]:
                for K in Ks:
                    for m in methods:
                        for mode in b["modes"]:
                            for s in seeds:
                                for c in combos:
                                    extra = dict(b.get("fixed") or {}, **c)
                                    out.append((bname, ds, arm, K, m, mode, s, extra, b.get("lr_from", m)))
    return out


def tuning_cells(stage):
    cells = set()
    for (_, ds, arm, K, m, _, _, _, lr_from) in expand_blocks(stage):
        cells.add((ds, arm, K, lr_from))
    return sorted(cells)


# ------------------------------------------------------------------ execution
def launch(job, out, env):
    logs = out / "logs"; logs.mkdir(parents=True, exist_ok=True)
    fh = open(logs / f"{job.run_id}.log", "a")
    p = subprocess.Popen([sys.executable, "-m", "fedstac.run", *job.overrides()], cwd=REPO, env=env,
                         stdout=fh, stderr=subprocess.STDOUT)
    return p, fh


def result_of(out, run_id):
    f = out / "runs" / run_id / "results.json"
    return json.loads(f.read_text()) if f.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["smoke", "sanity", "pilot", "full"])
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--data", default="data/prepared")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--extra", nargs="*", default=[], help="extra Hydra overrides for every run (tests only)")
    a = ap.parse_args()
    out = Path(a.out); (out / "index").mkdir(parents=True, exist_ok=True)
    common = [f"paths.out_root={out}", f"paths.data_root={a.data}", "device=cpu", "cpu_threads=1",
              f"suite=scaling_{a.stage}", "wandb.mode=disabled", *a.extra]
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")

    jobs: list[Job] = []
    tune_state = {}          # cell -> {"grid": [...], "lr": None}
    tcfg = PLAN["tuning"]

    if a.stage == "smoke":
        for ds, arm, K, m in (("ciciot2023", "A", 10, "fedavg"), ("ciciot2023", "B", 200, "scaffold")):
            ov = cell_overrides(ds, arm, K, m, "rho", 0, 0.05, {"fl.rounds": 5, "fl.eval_every": 5})
            jobs.append(Job(ov, {"block": "smoke", "dataset": ds, "arm": arm, "K": K, "method": m}, common))
    elif a.stage == "sanity":
        for sc in PLAN["sanity"]:
            ov = dict(x.split("=", 1) for x in sc["overrides"])
            jobs.append(Job(ov, {"block": "sanity", "expect_run_id": sc["run_id"], "gpu_macro_f1": sc["gpu_macro_f1"]}, common))
    else:
        for cell in tuning_cells(a.stage):
            tune_state[cell] = {"grid": list(tcfg["grid"]), "lr": None, "low": list(tcfg["extend_low"]),
                                "high": list(tcfg["extend_high"])}
        for (bname, ds, arm, K, m, mode, s, extra, lr_from) in expand_blocks(a.stage):
            label = {"block": bname, "dataset": ds, "arm": arm, "K": K, "method": m, "mode": mode, "seed": s, **extra}
            jobs.append(Job({"_pending": (ds, arm, K, m, mode, s, extra)}, label, common, needs=(ds, arm, K, lr_from)))

    def tuning_jobs(cell, lrs):
        ds, arm, K, m = cell
        js = []
        for lr in lrs:
            ov = cell_overrides(ds, arm, K, m, tcfg["mode"], tcfg["seed"], lr)
            j = Job(ov, {"block": "tuning", "dataset": ds, "arm": arm, "K": K, "method": m, "mode": tcfg["mode"],
                         "seed": tcfg["seed"], "fl.lr": lr}, common)
            j.run_id = identity_of(j.overrides())
            js.append(j)
        return js

    tjobs = {cell: tuning_jobs(cell, st["grid"]) for cell, st in tune_state.items()}
    for j in jobs:
        if j.needs is None:
            j.run_id = identity_of(j.overrides())

    index_path = out / "index" / f"scaling_{a.stage}.json"
    tuned_path = out / "index" / "scaling_tuned.json"

    def try_select(cell):
        """Return lr if the cell's tuning is complete and interior, else None (and maybe extend)."""
        st = tune_state[cell]
        if st["lr"] is not None:
            return st["lr"]
        res = {j.label["fl.lr"]: result_of(out, j.run_id) for j in tjobs[cell]}
        if any(r is None for r in res.values()):
            return None
        scores = {lr: r[tcfg["criterion"]] for lr, r in res.items()}
        best = max(sorted(scores), key=lambda lr: scores[lr])     # ties -> smaller lr
        lo, hi = min(scores), max(scores)
        if best == lo and st["low"]:
            new = st["low"].pop(0); st["grid"].append(new); tjobs[cell] += tuning_jobs(cell, [new]); return None
        if best == hi and st["high"]:
            new = st["high"].pop(0); st["grid"].append(new); tjobs[cell] += tuning_jobs(cell, [new]); return None
        st["lr"], st["scores"] = best, scores
        st["edge_unresolved"] = (best == lo and not st["low"]) or (best == hi and not st["high"])
        return best

    def materialise(j):
        ds, arm, K, m, mode, s, extra = j.ov.pop("_pending")
        j.ov = cell_overrides(ds, arm, K, m, mode, s, tune_state[j.needs]["lr"], extra)
        j.label["fl.lr"] = tune_state[j.needs]["lr"]
        j.run_id = identity_of(j.overrides())

    def write_index():
        idx = [{"run_id": j.run_id, **j.label} for j in jobs if j.run_id]
        idx += [{"run_id": j.run_id, **j.label} for js in tjobs.values() for j in js]
        index_path.write_text(json.dumps(idx, indent=1))
        tuned = {"|".join(map(str, c)): {k: v for k, v in st.items() if k not in ("low", "high")} for c, st in tune_state.items()}
        if tuned:
            prev = json.loads(tuned_path.read_text()) if tuned_path.exists() else {}
            prev.update({k: v for k, v in tuned.items() if v.get("lr") is not None or k not in prev})
            tuned_path.write_text(json.dumps(prev, indent=1))

    if a.dry:
        n_t = sum(len(v) for v in tjobs.values())
        print(f"[{a.stage}] tuning cells {len(tune_state)}, initial tuning runs {n_t}, dependent runs {len(jobs)}")
        from collections import Counter
        print(Counter(j.label["block"] for j in jobs))
        return

    active, done, failed, t0 = [], set(), [], time.time()
    launched = set()
    while True:
        # release dependent jobs whose cell is tuned
        for cell in tune_state:
            try_select(cell)
        ready = []
        for js in tjobs.values():
            ready += [j for j in js if j.run_id not in launched]
        n_tune_ready = len(ready)
        for j in jobs:
            if j.needs is not None and j.run_id is None and tune_state[j.needs]["lr"] is not None:
                materialise(j)
            if j.run_id and j.run_id not in launched:
                ready.append(j)
        # skip completed, dedupe
        fresh, seen = [], set()
        for j in ready:
            if j.run_id in seen:
                continue
            seen.add(j.run_id)
            if result_of(out, j.run_id) is not None:
                launched.add(j.run_id); done.add(j.run_id); continue
            fresh.append(j)
        fresh.sort(key=lambda j: (j.label["block"] != "tuning", -est_cost(j.ov)))
        while fresh and len(active) < a.workers:
            j = fresh.pop(0)
            launched.add(j.run_id)
            p, fh = launch(j, out, env)
            active.append((p, j, fh))
        pending_dep = sum(1 for j in jobs if j.run_id is None)
        if not active and not fresh and pending_dep == 0:
            break
        if not active and not fresh and pending_dep:
            # tuning finished but some cell could not be selected -> stop with a clear message
            stuck = sorted({j.needs for j in jobs if j.run_id is None})
            print(f"[stuck] cells without a tuned lr: {stuck}", flush=True)
            break
        time.sleep(3)
        for item in list(active):
            p, j, fh = item
            if p.poll() is None:
                continue
            fh.close(); active.remove(item)
            ok = result_of(out, j.run_id) is not None
            if ok:
                done.add(j.run_id)
            else:
                j.tries += 1
                if j.tries < 2:
                    launched.discard(j.run_id)       # retry once
                else:
                    failed.append(j.run_id)
                    if j.label["block"] == "tuning":
                        cell = (j.label["dataset"], j.label["arm"], j.label["K"], j.label["method"])
                        print(f"[fatal] tuning run failed twice: {cell} lr={j.label['fl.lr']}", flush=True)
            el = (time.time() - t0) / 60
            print(f"[{len(done)} done | {len(active)} active | {len(fresh)} queued | {pending_dep} awaiting lr] "
                  f"{'ok ' if ok else 'FAIL'} {j.run_id} {j.label} | {el:.1f} min", flush=True)
            write_index()
    write_index()
    if a.stage == "sanity":
        rows = []
        for j in jobs:
            r = result_of(out, j.run_id)
            d = None if r is None else 100 * (r["test"]["macro_f1"] - j.label["gpu_macro_f1"])
            rows.append({"run_id": j.run_id, "expected_run_id": j.label["expect_run_id"],
                         "id_match": j.run_id == j.label["expect_run_id"], "cpu_macro_f1": None if r is None else r["test"]["macro_f1"],
                         "gpu_macro_f1": j.label["gpu_macro_f1"], "delta_pp": d, "pass": d is not None and abs(d) <= 2.0 and j.run_id == j.label["expect_run_id"]})
        (out / "index" / "sanity.json").write_text(json.dumps(rows, indent=1))
        print(json.dumps(rows, indent=1))
        if not all(r["pass"] for r in rows):
            sys.exit(3)
    print(f"[{a.stage}] finished in {(time.time()-t0)/60:.1f} min; failed: {failed}", flush=True)
    if failed:
        sys.exit(2)


if __name__ == "__main__":
    main()
