"""Test-set metrics computed once on the model selected by validation."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_recall_fscore_support


def compute_metrics(y: np.ndarray, p: np.ndarray, client: np.ndarray, n_classes: int,
                    train_counts: np.ndarray, tail_quantile: float = 0.25) -> dict:
    labels = np.arange(n_classes)
    prec, rec, f1c, sup = precision_recall_fscore_support(y, p, labels=labels, zero_division=0)
    present = sup > 0
    order = np.argsort(train_counts, kind="stable")
    n_tail = max(1, int(np.ceil(tail_quantile * n_classes)))
    tail = [c for c in order[:n_tail] if present[c]]
    per_client = []
    for k in np.unique(client):
        mk = client == k
        lk = np.unique(y[mk])
        per_client.append(f1_score(y[mk], p[mk], labels=lk, average="macro", zero_division=0))
    per_client = np.array(per_client)
    return {
        "macro_f1": float(f1_score(y, p, labels=labels[present], average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y, p)),
        "balanced_accuracy": float(balanced_accuracy_score(y, p)),
        "weighted_f1": float(f1_score(y, p, labels=labels, average="weighted", zero_division=0)),
        "tail_recall": float(np.mean(rec[tail])) if tail else float("nan"),
        "tail_classes": [int(c) for c in tail],
        "worst_client_f1_p10": float(np.percentile(per_client, 10)),
        "mean_client_f1": float(per_client.mean()),
        "per_class_recall": rec.tolist(),
        "per_class_f1": f1c.tolist(),
        "per_class_support": sup.tolist(),
        "per_client_f1": per_client.tolist(),
        "n_test": int(len(y)),
    }
