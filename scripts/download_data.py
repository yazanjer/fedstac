"""Download a dataset with the Kaggle API (credentials from env / Colab Secrets) and record provenance.
Usage: python scripts/download_data.py --dataset ciciot2023 --raw /content/raw/ciciot2023
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True)
ap.add_argument("--raw", required=True)
ap.add_argument("--slug", default=None, help="override the Kaggle slug in the dataset config")
a = ap.parse_args()
spec = yaml.safe_load((REPO / "configs/dataset" / f"{a.dataset}.yaml").read_text())
slug = a.slug or spec["kaggle_slug"]
raw = Path(a.raw); raw.mkdir(parents=True, exist_ok=True)
meta_dir = raw / "_kaggle_meta"; meta_dir.mkdir(exist_ok=True)
subprocess.run(["kaggle", "datasets", "metadata", "-d", slug, "-p", str(meta_dir)], check=True)
t0 = time.time()
subprocess.run(["kaggle", "datasets", "download", "-d", slug, "-p", str(raw), "--unzip"], check=True)
files = sorted(p for p in raw.rglob("*") if p.is_file() and "_kaggle_meta" not in str(p))
rec = {"slug": slug, "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "seconds": round(time.time() - t0),
       "files": [{"path": str(p.relative_to(raw)), "bytes": p.stat().st_size} for p in files]}
meta_file = next(meta_dir.glob("*.json"), None)
if meta_file:
    rec["kaggle_metadata"] = json.loads(meta_file.read_text())
(raw / "download_provenance.json").write_text(json.dumps(rec, indent=1))
print(json.dumps({"slug": slug, "n_files": len(files), "GB": round(sum(f['bytes'] for f in rec['files']) / 1e9, 2)}, indent=1))
