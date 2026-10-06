"""Leakage, parity and validation tests.  Run: pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import DATA, SAMPLE  # noqa: E402
from features import (CATEGORICAL, NUMERIC, OnlineFeatureState, build_features,  # noqa: E402
                      prepare_raw)
from validate import validate_schema  # noqa: E402


@pytest.fixture(scope="module")
def raw():
    df = pd.read_csv(DATA if DATA.exists() else SAMPLE)
    return df.iloc[:20000].copy()


def test_no_future_leakage(raw):
    """Features of a row must not change when all later rows are removed."""
    full = build_features(raw)
    cut = full.request_time.iloc[len(full) // 2]
    past = build_features(raw[pd.to_datetime(raw.request_time) <= cut])
    a = full.set_index("request_id").loc[past.request_id, NUMERIC].to_numpy(float)
    b = past.set_index("request_id")[NUMERIC].to_numpy(float)
    assert np.allclose(a, b, equal_nan=True)


def test_labels_not_used_as_features(raw):
    flipped = raw.assign(fraud_label=1 - raw.fraud_label)
    a = build_features(raw)[NUMERIC].to_numpy(float)
    b = build_features(flipped)[NUMERIC].to_numpy(float)
    assert np.allclose(a, b, equal_nan=True)


def test_training_serving_parity(raw):
    batch = build_features(raw)
    state = OnlineFeatureState()
    online = pd.DataFrame([state.score_features(x) for x in prepare_raw(raw).to_dict("records")])
    for c in NUMERIC:
        assert np.allclose(batch[c].to_numpy(float), online[c].to_numpy(float), equal_nan=True), c
    for c in CATEGORICAL:
        assert (batch[c].fillna("NA").astype(str).values == online[c].fillna("NA").astype(str).values).all(), c


def test_burst_feature_on_sample():
    """DEV1003 makes 3 txns in ~2 minutes in the provided sample; the 3rd must see 2 prior."""
    f = build_features(pd.read_csv(SAMPLE)).set_index("request_id")
    assert f.loc["TXN100005", "dev_cnt_10m"] == 2
    assert f.loc["TXN100003", "dev_is_new"] == 1


def test_schema_validation_catches_bad_input():
    df = pd.read_csv(SAMPLE)
    assert all(r["status"] == "PASS" for r in validate_schema(df))
    bad = df.copy()
    bad.loc[0, "amount"] = -5
    bad.loc[1, "service_type"] = "crypto"
    bad.loc[2, "request_id"] = bad.loc[3, "request_id"]
    fails = {r["check"] for r in validate_schema(bad) if r["status"] == "FAIL"}
    assert {"amount > 0", "service_type in allowed set", "request_id unique"} <= fails
    assert any(r["status"] == "FAIL" for r in validate_schema(df.drop(columns=["amount"])))
