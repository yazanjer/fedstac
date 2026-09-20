"""Prepare a dataset: python scripts/prepare_data.py --dataset ciciot2023 --raw data/raw/ciciot2023"""
import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fedstac.data.prepare import prepare_dataset  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True)
ap.add_argument("--raw", required=True)
ap.add_argument("--out", default="data/prepared")
ap.add_argument("--max-files", type=int, default=None)
ap.add_argument("--no-hash", action="store_true")
a = ap.parse_args()
spec = yaml.safe_load(open(Path(__file__).resolve().parents[1] / "configs/dataset" / f"{a.dataset}.yaml"))
meta = prepare_dataset(Path(a.raw), Path(a.out) / spec["name"], spec, max_files=a.max_files, hash_files=not a.no_hash)
print(json.dumps({k: meta[k] for k in ("n_samples", "n_features", "class_counts_prepared", "unknown_labels_dropped",
                                        "conflicting_label_feature_vectors", "prepared_sha256")}, indent=1))
