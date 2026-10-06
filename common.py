"""Shared paths, time windows and metric helpers."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "transactions_synthetic.csv"
SAMPLE = ROOT / "data" / "fraud_detection_sample_transactions.csv"
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"
FIGS = REPORTS / "figures"

# Time-based split. January only serves as feature history: behavioural features need
# warm-up (every device/merchant looks "new" on day 1; 30-day windows need 30 days).
WARMUP_END = "2026-02-01"
TRAIN = ("2026-02-01", "2026-05-01")   # fit
VALID = ("2026-05-01", "2026-06-01")   # early stopping + threshold selection
TEST = ("2026-06-01", "2026-07-01")    # untouched hold-out = baseline for monitoring
REFERENCE = ("2026-02-01", "2026-07-01")  # "training period" distribution for drift
LIVE_MONTHS = {"2026-07": ("2026-07-01", "2026-08-01"),
               "2026-08": ("2026-08-01", "2026-09-01"),
               "2026-09": ("2026-09-01", "2026-10-01")}


def window(df: pd.DataFrame, w: tuple[str, str]) -> pd.DataFrame:
    return df[(df.request_time >= w[0]) & (df.request_time < w[1])]


def file_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def classification_metrics(y, score, thr, amount=None) -> dict:
    y = np.asarray(y); score = np.asarray(score)
    pred = (score >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out = dict(
        n=int(len(y)), n_fraud=int(y.sum()), fraud_rate=float(y.mean()),
        threshold=float(thr),
        precision=float(precision_score(y, pred, zero_division=0)),
        recall=float(recall_score(y, pred, zero_division=0)),
        f1=float(f1_score(y, pred, zero_division=0)),
        pr_auc=float(average_precision_score(y, score)) if y.sum() else float("nan"),
        roc_auc=float(roc_auc_score(y, score)) if 0 < y.sum() < len(y) else float("nan"),
        confusion_matrix=dict(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp)),
    )
    if amount is not None:
        amount = np.asarray(amount, dtype=float)
        fa = amount[y == 1].sum()
        out["fraud_amount_recall"] = float(amount[(y == 1) & (pred == 1)].sum() / fa) if fa else float("nan")
    return out


def threshold_max_f1(y, score) -> float:
    p, r, t = precision_recall_curve(y, score)
    f1 = 2 * p * r / np.clip(p + r, 1e-12, None)
    return float(t[np.nanargmax(f1[:-1])])


def threshold_for_recall(y, score, target_recall: float) -> float:
    """Highest threshold that still achieves the target recall."""
    p, r, t = precision_recall_curve(y, score)
    ok = np.where(r[:-1] >= target_recall)[0]
    return float(t[ok[-1]]) if len(ok) else float(t[0])


def dump_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
