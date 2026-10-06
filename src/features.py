"""
Feature engineering.

Every behavioural feature is *point-in-time*: for a transaction at time t it only
uses transactions strictly earlier in the event stream (same ordering a live
feature store would see). Labels of past transactions are never used, because
fraud labels arrive weeks later (chargebacks) and would leak in training.

Two implementations are provided:
  * build_features(df)        - vectorised batch version used for training/scoring
  * OnlineFeatureState        - row-by-row version, mimicking a serving-time
                                feature store; used to test training/serving parity
"""
from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd

RAW_COLUMNS = {
    "request_id": "string", "request_time": "datetime", "service_type": "string",
    "device_id": "string", "merchant_id": "string", "merchant_state": "string",
    "merchant_city": "string", "merchant_type": "string", "mcc_code": "int",
    "mcc_title": "string", "issuer_bank": "string", "currency_code": "string",
    "amount": "float", "request_status": "string",
}
TARGET = "fraud_label"

# Raw fields deliberately NOT fed to the model (see README "Fields not used directly")
EXCLUDED = ["request_id", "device_id", "merchant_id", "merchant_city", "mcc_title",
            "currency_code", "request_status", "request_time"]

CATEGORICAL = ["service_type", "merchant_type", "mcc_code", "issuer_bank", "merchant_state"]
NUMERIC = [
    # time
    "hour", "day_of_week", "is_night", "is_weekend",
    # amount
    "log_amount", "is_round_1000", "ends_in_999",
    # device behaviour
    "device_missing", "dev_is_new", "dev_age_days_cap30",
    "dev_secs_since_last", "dev_cnt_10m", "dev_cnt_1h", "dev_cnt_24h", "dev_cnt_30d",
    "dev_amt_sum_1h", "dev_fail_cnt_24h", "dev_amt_ratio_prior_mean",
    "dev_merchant_seen_30d",
    # merchant behaviour
    "mer_age_days_cap30", "mer_cnt_1h", "mer_cnt_7d", "mer_amt_ratio_prior_mean",
]
# Note: lifetime counters / raw ages grow with calendar time (non-stationary) and drift by
# construction, so we use rolling windows and ages capped at 30 days instead.
FEATURES = NUMERIC + CATEGORICAL

FAIL_STATUSES = ("FAILED", "DECLINED")
H, D = 3600, 86400
AGE_CAP_DAYS = 30


def prepare_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Parse types and impose the canonical event order (time, request_id)."""
    out = df.copy()
    out["request_time"] = pd.to_datetime(out["request_time"])
    out["mcc_code"] = out["mcc_code"].astype("Int64").astype(str).replace("<NA>", np.nan)
    out = out.sort_values(["request_time", "request_id"], kind="stable").reset_index(drop=True)
    return out


def _window_count_sum(t: np.ndarray, w: int, v: np.ndarray | None = None):
    """For a sorted time array t (seconds), number (and sum of v) of *earlier*
    events with time > t_i - w. Ties: earlier position counts as earlier."""
    pos = np.arange(len(t))
    left = np.searchsorted(t, t - w, side="right")
    cnt = np.maximum(pos - left, 0)
    if v is None:
        return cnt
    cs = np.concatenate([[0.0], np.cumsum(v)])
    s = cs[pos] - cs[np.minimum(left, pos)]
    return cnt, s


def _group_features(g: pd.DataFrame, prefix: str) -> pd.DataFrame:
    t = g["_ts"].to_numpy()
    amt = g["amount"].to_numpy(dtype=float)
    fail = g["request_status"].isin(FAIL_STATUSES).to_numpy(dtype=float)
    n = np.arange(len(g))
    prior_sum = np.concatenate([[0.0], np.cumsum(amt)[:-1]])
    prior_mean = np.where(n > 0, prior_sum / np.maximum(n, 1), np.nan)
    out = pd.DataFrame(index=g.index)
    out[f"{prefix}_prior_txns"] = n
    out[f"{prefix}_ratio"] = amt / prior_mean
    out[f"{prefix}_age_days_cap30"] = np.minimum((t - t[0]) / D, AGE_CAP_DAYS)
    out[f"{prefix}_cnt_1h"] = _window_count_sum(t, H)
    if prefix == "mer":
        out["mer_cnt_7d"] = _window_count_sum(t, 7 * D)
    if prefix == "dev":
        out["dev_cnt_30d"] = _window_count_sum(t, 30 * D)
        out["dev_cnt_10m"] = _window_count_sum(t, 600)
        c24, _ = _window_count_sum(t, D, amt)
        _, s1h = _window_count_sum(t, H, amt)
        _, f24 = _window_count_sum(t, D, fail)
        out["dev_cnt_24h"] = c24
        out["dev_amt_sum_1h"] = s1h
        out["dev_fail_cnt_24h"] = f24
        out["dev_secs_since_last"] = np.r_[np.nan, np.diff(t)].astype(float)
    return out


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Batch feature computation. `raw` must contain all history needed
    (behavioural features look back in time)."""
    df = prepare_raw(raw)
    # resolution-safe epoch seconds (pandas may store datetimes as s/ms/us/ns)
    df["_ts"] = ((df["request_time"] - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)).astype(np.int64)

    f = pd.DataFrame(index=df.index)
    f["request_id"] = df["request_id"]
    f["request_time"] = df["request_time"]
    f["amount"] = df["amount"].astype(float)

    # time
    f["hour"] = df.request_time.dt.hour
    f["day_of_week"] = df.request_time.dt.dayofweek
    f["is_night"] = (f.hour < 6).astype(int)
    f["is_weekend"] = (f.day_of_week >= 5).astype(int)

    # amount
    f["log_amount"] = np.log1p(f.amount.clip(lower=0))
    f["is_round_1000"] = ((f.amount % 1000 == 0) & (f.amount > 0)).astype(int)
    f["ends_in_999"] = (np.round(f.amount) % 1000 == 999).astype(int)

    # device behaviour (only where device_id is present)
    f["device_missing"] = df.device_id.isna().astype(int)
    has_dev = df.device_id.notna()
    dev = (df[has_dev].groupby("device_id", group_keys=False, sort=False)
           .apply(lambda g: _group_features(g, "dev")))
    f.loc[dev.index, "dev_is_new"] = (dev["dev_prior_txns"] == 0).astype(float)
    for c in ["dev_age_days_cap30", "dev_secs_since_last", "dev_cnt_10m", "dev_cnt_1h",
              "dev_cnt_24h", "dev_cnt_30d", "dev_amt_sum_1h", "dev_fail_cnt_24h"]:
        f.loc[dev.index, c] = dev[c]
    f.loc[dev.index, "dev_amt_ratio_prior_mean"] = dev["dev_ratio"]
    # "any earlier txn of this device at this merchant in 30d" == "the latest earlier one is within 30d"
    dd = df[has_dev]
    prev = dd.groupby(["device_id", "merchant_id"], sort=False)["_ts"].shift(1)
    f.loc[dd.index, "dev_merchant_seen_30d"] = ((dd["_ts"] - prev) < 30 * D).astype(float)

    # merchant behaviour
    mer = (df.groupby("merchant_id", group_keys=False, sort=False)
           .apply(lambda g: _group_features(g, "mer")))
    f["mer_age_days_cap30"] = mer["mer_age_days_cap30"]
    f["mer_cnt_1h"] = mer["mer_cnt_1h"]
    f["mer_cnt_7d"] = mer["mer_cnt_7d"]
    f["mer_amt_ratio_prior_mean"] = mer["mer_ratio"]

    for c in CATEGORICAL:
        f[c] = df[c].astype("object")
    if TARGET in df:
        f[TARGET] = df[TARGET].astype(int)
    f[NUMERIC] = f[NUMERIC].astype(float)
    return f


class OnlineFeatureState:
    """Serving-style incremental feature computation (one transaction at a time).
    Written independently of build_features() to test training/serving parity."""

    def __init__(self):
        self.dev = {}    # device_id -> state
        self.mer = {}    # merchant_id -> state
        self.pair = {}   # (device_id, merchant_id) -> deque of times (30d)

    @staticmethod
    def _expire(win, t, w, key=lambda e: e):
        while win and key(win[0]) <= t - w:
            win.popleft()

    @staticmethod
    def _count_recent(win, after):
        c = 0
        for x in reversed(win):  # deque is time-ordered
            if x <= after:
                break
            c += 1
        return c

    def score_features(self, tx: dict) -> dict:
        t_dt = pd.Timestamp(tx["request_time"])
        t = int((t_dt - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1))
        amt = float(tx["amount"])
        fail = tx["request_status"] in FAIL_STATUSES
        out = {
            "hour": t_dt.hour, "day_of_week": t_dt.dayofweek,
            "is_night": int(t_dt.hour < 6), "is_weekend": int(t_dt.dayofweek >= 5),
            "log_amount": float(np.log1p(max(amt, 0))),
            "is_round_1000": int(amt % 1000 == 0 and amt > 0),
            "ends_in_999": int(round(amt) % 1000 == 999),
        }
        dev_id, mid = tx.get("device_id"), tx["merchant_id"]
        dev_ok = isinstance(dev_id, str)
        out["device_missing"] = int(not dev_ok)
        keys = ["dev_is_new", "dev_age_days_cap30", "dev_secs_since_last", "dev_cnt_10m",
                "dev_cnt_1h", "dev_cnt_24h", "dev_cnt_30d", "dev_amt_sum_1h", "dev_fail_cnt_24h",
                "dev_amt_ratio_prior_mean", "dev_merchant_seen_30d"]
        out.update({k: np.nan for k in keys})
        if dev_ok:
            s = self.dev.setdefault(dev_id, dict(n=0, sum=0.0, first=t, last=None, win=deque()))
            self._expire(s["win"], t, 30 * D, key=lambda e: e[0])
            win = s["win"]
            day = [e for e in win if e[0] > t - D]
            hour = [e for e in day if e[0] > t - H]
            p = self.pair.setdefault((dev_id, mid), deque())
            self._expire(p, t, 30 * D)
            out.update(
                dev_is_new=float(s["n"] == 0),
                dev_age_days_cap30=min((t - s["first"]) / D, AGE_CAP_DAYS),
                dev_secs_since_last=np.nan if s["last"] is None else t - s["last"],
                dev_cnt_10m=sum(1 for e in hour if e[0] > t - 600),
                dev_cnt_1h=len(hour), dev_cnt_24h=len(day), dev_cnt_30d=len(win),
                dev_amt_sum_1h=sum(e[1] for e in hour),
                dev_fail_cnt_24h=sum(e[2] for e in day),
                dev_amt_ratio_prior_mean=amt / (s["sum"] / s["n"]) if s["n"] else np.nan,
                dev_merchant_seen_30d=float(len(p) > 0),
            )
            s["n"] += 1; s["sum"] += amt; s["last"] = t
            win.append((t, amt, float(fail))); p.append(t)

        m = self.mer.setdefault(mid, dict(n=0, sum=0.0, first=t, win=deque()))
        self._expire(m["win"], t, 7 * D)
        out.update(
            mer_age_days_cap30=min((t - m["first"]) / D, AGE_CAP_DAYS),
            mer_cnt_1h=self._count_recent(m["win"], t - H),
            mer_cnt_7d=len(m["win"]),
            mer_amt_ratio_prior_mean=amt / (m["sum"] / m["n"]) if m["n"] else np.nan,
        )
        m["n"] += 1; m["sum"] += amt; m["win"].append(t)

        for c in CATEGORICAL:
            v = tx.get(c)
            out[c] = None if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)
        return out

