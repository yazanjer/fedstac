"""Raw CSV -> prepared array store.

Only fit-free, row-wise transforms are applied here (column dropping, hash encoding of
categorical fields, NaN/inf replacement, signed log1p). Nothing is estimated from the data,
so no statistic can leak across the later train/validation/test splits. Class-capped uniform
sampling (bottom-k on a seeded random key) and exact-duplicate removal are label-conditional
but split-independent.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from fedstac.utils.io import atomic_write_json, sha256_file


def _hash_bucket(values: pd.Series, buckets: int) -> np.ndarray:
    uniq = pd.unique(values.astype(str))
    table = {u: int(hashlib.md5(u.encode()).hexdigest(), 16) % buckets for u in uniq}
    return values.astype(str).map(table).to_numpy()


def transform_frame(frame: pd.DataFrame, spec: dict) -> tuple[np.ndarray, list[str]]:
    """Fit-free feature transform shared by every dataset."""
    drop = [c for c in spec.get("drop_columns", []) if c in frame.columns]
    frame = frame.drop(columns=drop)
    cats = [c for c in spec.get("categorical_columns", []) if c in frame.columns]
    buckets = int(spec.get("hash_buckets", 8))
    blocks, names = [], []
    num = frame.drop(columns=cats)
    num = num.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    arr = num.to_numpy(dtype=np.float64)
    arr = np.sign(arr) * np.log1p(np.abs(arr))
    blocks.append(arr.astype(np.float32))
    names += list(num.columns)
    for c in cats:
        idx = _hash_bucket(frame[c], buckets)
        onehot = np.zeros((len(frame), buckets), dtype=np.float32)
        onehot[np.arange(len(frame)), idx] = 1.0
        blocks.append(onehot)
        names += [f"{c}#h{b}" for b in range(buckets)]
    return np.concatenate(blocks, axis=1), names


def _row_hash(x: np.ndarray) -> np.ndarray:
    return pd.util.hash_pandas_object(pd.DataFrame(x), index=False).to_numpy()


class ClassCappedSampler:
    """Uniform per-class sample without replacement (bottom-k on random keys), with dedup."""

    def __init__(self, n_classes: int, cap: int | None):
        self.cap = cap
        self.buf = {c: [] for c in range(n_classes)}
        self.seen = np.zeros(n_classes, dtype=np.int64)

    def add(self, x: np.ndarray, y: np.ndarray, key: np.ndarray) -> None:
        h = _row_hash(x)
        for c in np.unique(y):
            m = y == c
            self.seen[c] += int(m.sum())
            self.buf[c].append((x[m], key[m], h[m]))
            self._trim(c)

    def _trim(self, c: int) -> None:
        parts = self.buf[c]
        x = np.concatenate([p[0] for p in parts])
        k = np.concatenate([p[1] for p in parts])
        h = np.concatenate([p[2] for p in parts])
        order = np.argsort(k, kind="stable")
        x, k, h = x[order], k[order], h[order]
        _, first = np.unique(h, return_index=True)
        keep = np.sort(first)
        x, k, h = x[keep], k[keep], h[keep]
        if self.cap is not None and len(k) > self.cap:
            x, k, h = x[: self.cap], k[: self.cap], h[: self.cap]
        self.buf[c] = [(x, k, h)]

    def result(self) -> tuple[np.ndarray, np.ndarray]:
        xs, ys = [], []
        for c, parts in self.buf.items():
            if parts:
                xs.append(parts[0][0])
                ys.append(np.full(len(parts[0][0]), c, dtype=np.int16))
        return np.concatenate(xs), np.concatenate(ys)


def prepare_dataset(raw_dir: Path, out_dir: Path, spec: dict, seed: int = 20260920,
                    chunksize: int = 500_000, max_files: int | None = None, hash_files: bool = True) -> dict:
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    files = sorted(raw_dir.rglob(spec["file_glob"]))
    files = [f for f in files if not any(s in str(f) for s in spec.get("exclude_paths", []))]
    if max_files:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"No files matching {spec['file_glob']} under {raw_dir}")
    classes = list(spec["classes"])
    cmap = {c: i for i, c in enumerate(classes)}
    aliases = spec.get("label_aliases", {}) or {}
    sampler = ClassCappedSampler(len(classes), spec.get("class_cap"))
    feature_names = None
    unknown: dict[str, int] = {}
    n_rows = 0
    t0 = time.time()
    for fi, f in enumerate(files):
        reader = pd.read_csv(f, chunksize=chunksize, low_memory=False,
                             dtype={c: str for c in spec.get("categorical_columns", [])})
        for ci, chunk in enumerate(reader):
            chunk.columns = [c.strip() for c in chunk.columns]
            label_col = next(c for c in spec["label_columns"] if c in chunk.columns)
            labels = chunk[label_col].astype(str).str.strip().map(lambda s: aliases.get(s, s))
            bad = ~labels.isin(cmap)
            for lab, cnt in labels[bad].value_counts().items():
                unknown[lab] = unknown.get(lab, 0) + int(cnt)
            chunk, labels = chunk[~bad], labels[~bad]
            feats = chunk.drop(columns=[c for c in spec["label_columns"] + spec.get("other_label_columns", []) if c in chunk.columns])
            x, names = transform_frame(feats, spec)
            if feature_names is None:
                feature_names = names
            elif names != feature_names:
                raise ValueError(f"Feature schema drift in {f}")
            y = labels.map(cmap).to_numpy(dtype=np.int64)
            key = np.random.default_rng([seed, fi, ci]).random(len(y))
            sampler.add(x, y, key)
            n_rows += len(y)
        print(f"[prepare] {fi+1}/{len(files)} files, {n_rows:,} rows, {time.time()-t0:.0f}s", flush=True)
    X, y = sampler.result()
    order = np.random.default_rng(seed).permutation(len(y))
    X, y = X[order], y[order]
    # rows whose features are identical but labels differ are ambiguous; report, keep
    h = _row_hash(X)
    s = pd.DataFrame({"h": h, "y": y})
    conflicting = int((s.groupby("h")["y"].nunique() > 1).sum())
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "prepared.npz", X=X, y=y)
    meta = {
        "dataset": spec["name"],
        "classes": classes,
        "groups": spec["groups"],
        "group_of_class": [spec["group_of_class"][c] for c in classes],
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "n_samples": int(len(y)),
        "class_counts_prepared": np.bincount(y, minlength=len(classes)).tolist(),
        "class_counts_raw": sampler.seen.tolist(),
        "rows_read": n_rows,
        "unknown_labels_dropped": unknown,
        "conflicting_label_feature_vectors": conflicting,
        "class_cap": spec.get("class_cap"),
        "sampling_seed": seed,
        "source_files": [str(f.relative_to(raw_dir)) for f in files],
        "source_sha256": {str(f.relative_to(raw_dir)): sha256_file(f) for f in files} if hash_files else None,
        "prepared_sha256": sha256_file(out_dir / "prepared.npz"),
        "spec": spec,
    }
    atomic_write_json(out_dir / "meta.json", meta)
    return meta


def load_prepared(out_dir: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    out_dir = Path(out_dir)
    meta = json.loads((out_dir / "meta.json").read_text())
    with np.load(out_dir / "prepared.npz") as z:
        X, y = z["X"], z["y"].astype(np.int64)
    return X, y, meta
