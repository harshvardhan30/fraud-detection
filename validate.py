"""
Pre-release checks (and reusable input validation for live batches).

  1. Input schema & data-type validation
  2. Missing-value & feature-range checks against the training reference
  3. Training/serving feature consistency (batch vs online implementation)
  4. Model version, reproducible evaluation, release gate, rollback plan

Run:  python src/validate.py      -> reports/release_checks.md
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from common import (ARTIFACTS, DATA, REPORTS, TEST, classification_metrics, file_sha256, window)
from features import (CATEGORICAL, FEATURES, NUMERIC, RAW_COLUMNS, OnlineFeatureState,
                      build_features, prepare_raw)

ALLOWED = {
    "service_type": {"upi", "wallet", "imps", "netbanking"},
    "merchant_type": {"low", "medium", "high"},
    "request_status": {"SUCCESS", "FAILED", "DECLINED"},
    "currency_code": {"INR"},
}
CRITICAL_NOT_NULL = ["request_id", "request_time", "service_type", "merchant_id",
                     "merchant_type", "mcc_code", "amount", "currency_code"]


def validate_schema(raw: pd.DataFrame, mcc_title_map: dict | None = None) -> list[dict]:
    res = []
    add = lambda check, ok, detail="": res.append(dict(check=check, status="PASS" if ok else "FAIL", detail=detail))
    missing = [c for c in RAW_COLUMNS if c not in raw.columns]
    add("required columns present", not missing, f"missing={missing}" if missing else f"{len(RAW_COLUMNS)} columns")
    if missing:
        return res
    t = pd.to_datetime(raw.request_time, errors="coerce")
    add("request_time parses as datetime", t.notna().all() or raw.request_time.isna().equals(t.isna()),
        f"{int((t.isna() & raw.request_time.notna()).sum())} unparseable")
    amt = pd.to_numeric(raw.amount, errors="coerce")
    add("amount numeric", amt.notna().sum() == raw.amount.notna().sum(),
        f"{int(amt.isna().sum() - raw.amount.isna().sum())} non-numeric")
    add("amount > 0", (amt.dropna() > 0).all(), f"{int((amt <= 0).sum())} non-positive")
    mcc = pd.to_numeric(raw.mcc_code, errors="coerce")
    add("mcc_code is 4-digit integer", mcc.dropna().between(1000, 9999).all() and (mcc.dropna() % 1 == 0).all())
    nulls = {c: int(raw[c].isna().sum()) for c in CRITICAL_NOT_NULL if raw[c].isna().any()}
    add("no nulls in critical fields", not nulls, str(nulls) if nulls else "")
    add("request_id unique", raw.request_id.is_unique, f"{int(raw.request_id.duplicated().sum())} duplicates")
    for c, allowed in ALLOWED.items():
        bad = set(raw[c].dropna().unique()) - allowed
        add(f"{c} in allowed set", not bad, f"unexpected={sorted(bad)}" if bad else "")
    if "fraud_label" in raw:
        add("fraud_label in {0,1}", raw.fraud_label.dropna().isin([0, 1]).all())
    if mcc_title_map:
        mm = raw.dropna(subset=["mcc_code", "mcc_title"])
        exp = mm.mcc_code.astype(int).astype(str).map(mcc_title_map)
        bad = int(((exp.notna()) & (exp != mm.mcc_title)).sum())
        add("mcc_code -> mcc_title consistent", bad == 0, f"{bad} mismatches")
    return res


def validate_ranges(feats: pd.DataFrame, reference: dict) -> tuple[list[dict], pd.Series]:
    """Returns check results and a boolean mask of rows to quarantine."""
    res = []
    add = lambda check, status, detail="": res.append(dict(check=check, status=status, detail=detail))
    quarantine = pd.Series(False, index=feats.index)
    a = reference["numeric"]["amount"]
    hard_max = a["max"] * 3  # far outside anything seen in training
    out_amt = feats.amount > hard_max
    quarantine |= out_amt
    add("amount within plausible range", "PASS" if not out_amt.any() else "FAIL",
        f"{int(out_amt.sum())} rows > {hard_max:,.0f} (3x training max); by service: "
        f"{feats.loc[out_amt, 'service_type'].value_counts().to_dict()}")
    # unit/scale errors (e.g. paise sent instead of rupees) shift a whole segment, not single rows:
    # compare the daily median amount per payment method with the training median.
    med_ref = reference["amount_median_by_service"]
    day = feats.request_time.dt.date
    daily = feats.groupby([day, "service_type"]).amount.median().rename("median").reset_index()
    daily["ratio"] = daily["median"] / daily.service_type.map(med_ref)
    bad_seg = daily[(daily.ratio > 3) | (daily.ratio < 1 / 3)]
    if len(bad_seg):
        key = set(zip(bad_seg.request_time, bad_seg.service_type))
        seg_mask = pd.Series([k in key for k in zip(day, feats.service_type)], index=feats.index)
        quarantine |= seg_mask
        first = bad_seg.request_time.min()
        add("daily median amount per service within 3x of training", "FAIL",
            f"{len(bad_seg)} service-days flagged ({bad_seg.service_type.value_counts().to_dict()}), first on {first}, "
            f"median ratio up to {bad_seg.ratio.max():.0f}x -> likely unit error; {int(seg_mask.sum()):,} rows quarantined")
    else:
        add("daily median amount per service within 3x of training", "PASS",
            f"max ratio {daily.ratio.max():.2f}x")
    above_p99 = (feats.amount > a["p99"]).mean()
    add("share of amount above training p99", "PASS" if above_p99 < 0.03 else "WARN",
        f"{above_p99:.2%} (expected ~1%)")
    worst = []
    for c in NUMERIC:
        ref_m = reference["numeric"][c]["missing_rate"]
        cur_m = feats[c].isna().mean()
        if cur_m > ref_m * 2 + 0.01:
            worst.append(f"{c}: {ref_m:.2%}->{cur_m:.2%}")
    add("missing rate vs training (<= 2x + 1pp)", "PASS" if not worst else "FAIL", "; ".join(worst))
    for c in ["service_type", "merchant_type", "issuer_bank", "merchant_state", "mcc_code"]:
        ref_m = feats[c].isna().mean()
        if ref_m > 0.02:
            add(f"{c} missing rate", "WARN", f"{ref_m:.2%}")
    unseen = {}
    for c in CATEGORICAL:
        known = set(reference["categorical"][c])
        share = (~feats[c].dropna().astype(str).isin(known)).mean()
        if share > 0:
            unseen[c] = f"{share:.2%}"
    add("unseen categories (mapped to 'infrequent')", "PASS" if not unseen else "WARN", str(unseen) if unseen else "")
    return res, quarantine


def training_serving_parity(raw: pd.DataFrame, upto: str, compare_window: tuple[str, str]) -> list[dict]:
    """Stream all events < `upto` through the online feature state and compare to batch features."""
    r = prepare_raw(raw)
    r = r[r.request_time < upto]
    batch = build_features(r)
    state = OnlineFeatureState()
    online = pd.DataFrame([state.score_features(x) for x in r.to_dict("records")], index=r.index)
    m = (batch.request_time >= compare_window[0]) & (batch.request_time < compare_window[1])
    bad = {}
    for c in NUMERIC:
        a, b = batch.loc[m, c].to_numpy(float), online.loc[m, c].to_numpy(float)
        n = int((~np.isclose(a, b, rtol=1e-9, atol=1e-6, equal_nan=True)).sum())
        if n:
            bad[c] = n
    for c in CATEGORICAL:
        n = int((batch.loc[m, c].fillna("NA").astype(str) != online.loc[m, c].fillna("NA").astype(str)).sum())
        if n:
            bad[c] = n
    return [dict(check="batch (training) vs online (serving) features identical",
                 status="PASS" if not bad else "FAIL",
                 detail=f"{int(m.sum()):,} rows x {len(FEATURES)} features compared; mismatches={bad or 0}")]


def release_gate(card: dict, feats: pd.DataFrame, model) -> list[dict]:
    res = []
    add = lambda check, ok, detail="": res.append(dict(check=check, status="PASS" if ok else "FAIL", detail=detail))
    add("data hash matches model card", file_sha256(DATA) == card["data_sha256"], card["data_sha256"])
    te = window(feats, TEST)
    s = model.predict_proba(te[FEATURES])[:, 1]
    m = classification_metrics(te.fraud_label, s, card["thresholds"]["block"], te.amount)
    ref = card["metrics"]["xgboost"]["test"]
    add("evaluation reproducible from saved artifact",
        abs(m["pr_auc"] - ref["pr_auc"]) < 1e-9 and m["confusion_matrix"] == ref["confusion_matrix"],
        f"PR-AUC {m['pr_auc']:.4f}")
    g = card["release_gate"]
    add(f"test PR-AUC >= {g['min_test_pr_auc']}", m["pr_auc"] >= g["min_test_pr_auc"], f"{m['pr_auc']:.3f}")
    add(f"test recall >= {g['min_test_recall']}", m["recall"] >= g["min_test_recall"], f"{m['recall']:.3f}")
    add(f"test precision >= {g['min_test_precision']}", m["precision"] >= g["min_test_precision"], f"{m['precision']:.3f}")
    base = card["metrics"]["logreg"]["test"]["pr_auc"]
    add("beats logistic-regression baseline (PR-AUC)", m["pr_auc"] > base, f"{m['pr_auc']:.3f} vs {base:.3f}")
    add("model version recorded", bool(card.get("model_version")), card.get("model_version", ""))
    add("rollback target defined", bool(card.get("rollback_to")), card.get("rollback_to", ""))
    return res


def to_md(title, rows):
    out = [f"## {title}\n", "| check | status | detail |", "|---|---|---|"]
    out += [f"| {r['check']} | {r['status']} | {r['detail']} |" for r in rows]
    return "\n".join(out) + "\n"


def main():
    card = json.loads((ARTIFACTS / "model_card.json").read_text())
    reference = json.loads((ARTIFACTS / "reference_stats.json").read_text())
    model = joblib.load(ARTIFACTS / "model.joblib")
    raw = pd.read_csv(DATA)
    feats = build_features(raw)
    raw_test = raw[(raw.request_time >= TEST[0]) & (raw.request_time < TEST[1])]

    sections = [
        ("1. Input schema & data types (June batch)", validate_schema(raw_test, reference["mcc_title_map"])),
        ("2. Missing values & feature ranges (June batch vs Jan-Jun reference)",
         validate_ranges(window(feats, TEST), reference)[0]),
        ("3. Training-serving feature consistency (June, online state replayed from Jan 1)",
         training_serving_parity(raw, TEST[1], TEST)),
        ("4. Model version, evaluation & release gate", release_gate(card, feats, model)),
    ]
    statuses = [r["status"] for _, rows in sections for r in rows]
    verdict = "FAIL - do not release" if "FAIL" in statuses else "PASS - release to shadow, then canary"
    md = [f"# Release checks for `{card['model_version']}`\n", f"**Verdict: {verdict}**\n"]
    md += [to_md(t, r) for t, r in sections]
    md.append("""## Rollout & rollback plan
1. Register the model + `model_card.json` (version, data hash, windows, thresholds, metrics, library versions).
2. **Shadow** for 1-2 weeks: score live traffic, take no action; compare score distribution, alert volume and
   (as labels arrive) precision/recall against the current production system.
3. **Canary** 5-10% of traffic with the decline threshold; watch approval rate, false declines, review queue, latency.
4. Full rollout. The previous version stays deployed behind a feature flag.
5. **Rollback trigger**: schema/parity failure, approval rate drop > 1pp, review queue > 2x capacity, or any
   error-rate/latency SLO breach -> flip traffic back to the previous version (or the rule engine) in one config change.
""")
    (REPORTS / "release_checks.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
