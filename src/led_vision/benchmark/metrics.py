"""Binary ON/OFF classification metrics. 'ON' is the positive class."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BinaryMetrics:
    n: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    false_on: int   # predicted ON, truly OFF
    false_off: int  # predicted OFF, truly ON
    confusion: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def binary_metrics(truth: Sequence[str], pred: Sequence[str]) -> BinaryMetrics:
    if len(truth) != len(pred):
        raise ValueError(f"Length mismatch: {len(truth)} labels vs {len(pred)} predictions")
    if not truth:
        raise ValueError("Cannot compute metrics on an empty set")
    tp = sum(t == "ON" and p == "ON" for t, p in zip(truth, pred))
    tn = sum(t == "OFF" and p == "OFF" for t, p in zip(truth, pred))
    fp = sum(t == "OFF" and p == "ON" for t, p in zip(truth, pred))
    fn = sum(t == "ON" and p == "OFF" for t, p in zip(truth, pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return BinaryMetrics(
        n=len(truth),
        accuracy=round((tp + tn) / len(truth), 4),
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        false_on=fp,
        false_off=fn,
        confusion={"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    )


def latency_summary(runtimes_ms: Sequence[float]) -> dict[str, float]:
    if not runtimes_ms:
        raise ValueError("No runtimes recorded")
    arr = np.asarray(runtimes_ms, dtype=np.float64)
    return {
        "mean_ms": round(float(arr.mean()), 3),
        "p50_ms": round(float(np.percentile(arr, 50)), 3),
        "p95_ms": round(float(np.percentile(arr, 95)), 3),
        "max_ms": round(float(arr.max()), 3),
    }
