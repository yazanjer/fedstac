"""Pick hyperparameters by mean validation macro-F1 on tuning seed 100 and write configs/tuned.yaml."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--suite", required=True, choices=["tune_lr", "tune_methods"])
ap.add_argument("--out", default="outputs")
a = ap.parse_args()
out = Path(a.out)
index = json.loads((out / "index" / f"{a.suite}.json").read_text())
score = defaultdict(list)
for e in index:
    r = out / "runs" / e["run_id"] / "results.json"
    if not r.exists():
        raise SystemExit(f"missing result {e['run_id']}; finish the suite first")
    v = json.loads(r.read_text())["best_val_macro_f1"]
    keys = tuple(sorted((k, e[k]) for k in e if "." in k))
    score[(e["method"], keys)].append(v)
tuned = yaml.safe_load((REPO / "configs/tuned.yaml").read_text()) or {}
tuned.setdefault("all", {}); tuned.setdefault("methods", {})
best = {}
for (m, keys), vs in score.items():
    mean = sum(vs) / len(vs)
    if m not in best or mean > best[m][0]:
        best[m] = (mean, keys)
report = {}
for m, (mean, keys) in best.items():
    report[m] = {"val_macro_f1_mean": round(mean, 4), **dict(keys)}
    if a.suite == "tune_lr":
        tuned["all"].update(dict(keys))
    else:
        tuned["methods"][m] = dict(keys)
(REPO / "configs/tuned.yaml").write_text("# Written by scripts/select_hparams.py (validation macro-F1, tuning seed 100)\n" + yaml.safe_dump(tuned, sort_keys=True))
(out / "index" / f"{a.suite}_selection.json").write_text(json.dumps({"all_scores": {f"{m}|{dict(k)}": sum(v)/len(v) for (m, k), v in score.items()}, "selected": report}, indent=1))
print(json.dumps(report, indent=1))
