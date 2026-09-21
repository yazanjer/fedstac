"""Expand a suite YAML into runs and execute pending ones with N parallel worker processes.

Resumable: a run whose results.json exists is skipped; an interrupted run resumes from its
round checkpoint. Identical configurations reached from different suites share one run_id.
Usage: python scripts/run_suite.py --suite main --workers 4 [--out outputs] [--data data/prepared]
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def tuned_overrides(method: str) -> dict:
    t = yaml.safe_load((REPO / "configs/tuned.yaml").read_text()) or {}
    o = dict(t.get("all") or {})
    ms = t.get("methods") or {}
    base = method
    for suffix in ("_sfs_pcl", "_pcl", "_sfs"):
        if method.endswith(suffix):
            base = method[: -len(suffix)]
            break
    o.update(ms.get(base) or {})
    if method.endswith("_pcl") and "method.tau" in (ms.get("fedstap") or {}):
        o["method.tau"] = ms["fedstap"]["method.tau"]   # PCL used as a plug-in keeps FedStaP's tuned tau
    if method.startswith("abl_"):
        # CAA's beta comes from the tuned SFS + CAA + PCL design; PCL's tau from the proposed FedStaP
        # (falling back to the joint tuning when FedStaP has not been tuned), so abl_s1a0p1 == FedStaP.
        fs, fp = ms.get("fedstac") or {}, ms.get("fedstap") or {}
        if "a1" in method and "method.beta" in fs:
            o["method.beta"] = fs["method.beta"]
        tau = fp.get("method.tau", fs.get("method.tau"))
        if "p1" in method and tau is not None:
            o["method.tau"] = tau
    return o


def expand(suite: dict, common: list[str]) -> list[dict]:
    runs = []
    for g in suite["groups"]:
        sweep = g.get("sweep") or {}
        keys = list(sweep)
        for ds, lab, m, s in itertools.product(g["datasets"], g["labels"], g["methods"], g["seeds"]):
            for vals in itertools.product(*[sweep[k] for k in keys]) if keys else [()]:
                ov = {"dataset": ds, "task.labels": lab, "method": m, "seed": s, "suite": suite["suite"]}
                ov.update(tuned_overrides(m))
                ov.update(g.get("fixed") or {})
                ov.update(dict(zip(keys, vals)))
                runs.append({"overrides": [f"{k}={v}" for k, v in ov.items()] + common,
                             "label": {"dataset": ds, "labels": lab, "method": m, "seed": s, **dict(zip(keys, vals))}})
    return runs


def identity_of(overrides: list[str]) -> str:
    from hydra import compose, initialize_config_dir
    from fedstac.run import run_identity
    from fedstac.utils.io import stable_hash
    with initialize_config_dir(str(REPO / "configs"), version_base="1.3"):
        cfg = compose("main", overrides=overrides)
    return stable_hash(run_identity(cfg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--data", default="data/prepared")
    ap.add_argument("--extra", nargs="*", default=[], help="extra Hydra overrides for every run")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--datasets", nargs="*", default=None, help="replace the suite's dataset list (smoke tests)")
    ap.add_argument("--seeds", nargs="*", type=int, default=None, help="replace the suite's seed list (smoke tests)")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    suite = yaml.safe_load((REPO / "configs/suites" / f"{a.suite}.yaml").read_text())
    for g in suite["groups"]:
        if a.datasets:
            g["datasets"] = a.datasets
        if a.seeds is not None:
            g["seeds"] = a.seeds
    common = [f"paths.out_root={a.out}", f"paths.data_root={a.data}", *a.extra]
    runs = expand(suite, common)
    out = Path(a.out)
    index = []
    for r in runs:
        r["run_id"] = identity_of(r["overrides"])
        index.append({"suite": suite["suite"], "run_id": r["run_id"], **r["label"]})
    (out / "index").mkdir(parents=True, exist_ok=True)
    (out / "index" / f"{suite['suite']}.json").write_text(json.dumps(index, indent=1))
    uniq, seen = [], set()
    for r in runs:
        if r["run_id"] not in seen:
            seen.add(r["run_id"]); uniq.append(r)
    pending = [r for r in uniq if not (out / "runs" / r["run_id"] / "results.json").exists()]
    if a.limit:
        pending = pending[: a.limit]
    print(f"[suite {suite['suite']}] {len(runs)} entries, {len(uniq)} unique, {len(pending)} pending, workers={a.workers}", flush=True)
    if a.dry:
        return
    logs = out / "logs"; logs.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
    active, done, failed, t0 = [], 0, [], time.time()
    queue = list(pending)
    while queue or active:
        while queue and len(active) < a.workers:
            r = queue.pop(0)
            fh = open(logs / f"{r['run_id']}.log", "a")
            p = subprocess.Popen([sys.executable, "-m", "fedstac.run", *r["overrides"]], cwd=REPO, env=env,
                                 stdout=fh, stderr=subprocess.STDOUT)
            active.append((p, r, fh))
        time.sleep(2)
        for item in list(active):
            p, r, fh = item
            if p.poll() is not None:
                fh.close(); active.remove(item); done += 1
                ok = (out / "runs" / r["run_id"] / "results.json").exists()
                if not ok:
                    failed.append(r["run_id"])
                el = time.time() - t0
                eta = el / done * (len(pending) - done)
                print(f"[{done}/{len(pending)}] {'ok ' if ok else 'FAIL'} {r['run_id']} {r['label']} | "
                      f"elapsed {el/60:.1f} min, eta {eta/60:.1f} min", flush=True)
    print(f"[suite {suite['suite']}] finished; failed: {failed}")


if __name__ == "__main__":
    main()
