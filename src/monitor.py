"""
Monitoring the champion (trained/validated/tested on Jan-Jun) on live Jul-Sep traffic.

Per live month:
  0. Input validation (schema / ranges) - data-quality issues are triaged first
  1. Data drift        : PSI (+ KS for amount) on raw inputs and model features vs Jan-Jun reference
  2. Prediction drift  : PSI of fraud scores vs out-of-sample reference scores (May-Jun), alert volumes
  3. Performance drift : precision / recall / F1 / PR-AUC / ROC-AUC at the frozen threshold vs June baseline
  4. Business metrics  : fraud loss, false declines, approval rate, manual-review queue
Then: rule-based alert -> action table, root-cause slice of missed fraud, and a challenger
retrained with newly labelled data, compared with the champion on clean September data.

Label delay: chargeback labels mature over ~30-90 days. Here labels are assumed available for the
demonstration; in production performance drift for the latest month is reported as provisional.

Run:  python src/monitor.py   -> reports/monitoring_report.md, reports/monitoring_metrics.json
"""
from __future__ import annotations

import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from common import (ARTIFACTS, DATA, FIGS, LIVE_MONTHS, REFERENCE, REPORTS, TRAIN,
                    classification_metrics, dump_json, threshold_max_f1, window)
from features import FEATURES, NUMERIC, build_features
from train import fit_xgb
from validate import validate_ranges, validate_schema

PSI_WARN, PSI_ALERT = 0.10, 0.25
EPS = 1e-4


# ---------------------------------------------------------------- drift statistics
def psi_numeric(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    ref, cur = ref[~np.isnan(ref)], cur[~np.isnan(cur)]
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # near-constant feature
        edges = np.unique(np.r_[ref.min(), np.median(ref), ref.max()])
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / max(len(cur), 1)
    r, c = np.clip(r, EPS, None), np.clip(c, EPS, None)
    return float(np.sum((c - r) * np.log(c / r)))


def psi_categorical(ref: pd.Series, cur: pd.Series) -> float:
    r = ref.astype(str).value_counts(normalize=True)
    c = cur.astype(str).value_counts(normalize=True)
    idx = r.index.union(c.index)
    r, c = r.reindex(idx, fill_value=0).clip(lower=EPS), c.reindex(idx, fill_value=0).clip(lower=EPS)
    return float(np.sum((c - r) * np.log(c / r)))


def level(psi):
    return "ALERT" if psi >= PSI_ALERT else ("WARN" if psi >= PSI_WARN else "ok")


# ---------------------------------------------------------------- business metrics
def business(df: pd.DataFrame, score: np.ndarray, thr_block: float, thr_review: float) -> dict:
    y, amt = df.fraud_label.to_numpy(), df.amount.to_numpy()
    decline = score >= thr_block
    review = (score >= thr_review) & ~decline
    approve = ~decline & ~review
    days = df.request_time.dt.date.nunique()
    return dict(
        txns=int(len(df)), approval_rate=float(1 - decline.mean()),
        false_declines=int((decline & (y == 0)).sum()),
        false_decline_value=float(amt[decline & (y == 0)].sum()),
        fraud_value_total=float(amt[y == 1].sum()),
        fraud_loss_auto_approved=float(amt[approve & (y == 1)].sum()),
        fraud_loss_rate_bps=float(amt[approve & (y == 1)].sum() / amt.sum() * 1e4),
        review_queue_per_day=float(review.sum() / days),
        review_hit_rate=float(y[review].mean()) if review.any() else float("nan"),
    )


def main():
    card = json.loads((ARTIFACTS / "model_card.json").read_text())
    reference = json.loads((ARTIFACTS / "reference_stats.json").read_text())
    model = joblib.load(ARTIFACTS / "model.joblib")
    thr_b, thr_r = card["thresholds"]["block"], card["thresholds"]["review"]
    base = card["metrics"]["xgboost"]["test"]

    raw = pd.read_csv(DATA)
    feats = build_features(raw)
    feats["score"] = model.predict_proba(feats[FEATURES])[:, 1]
    ref = window(feats, REFERENCE)
    ref_scores = np.array(reference["score_values"])  # out-of-sample (May-Jun) scores
    june = window(feats, ("2026-06-01", "2026-07-01"))
    periods = {"2026-06 (baseline)": ("2026-06-01", "2026-07-01"), **LIVE_MONTHS}

    drift_raw, drift_feat, pred, perf, biz, valid, quarantine = {}, {}, {}, {}, {}, {}, {}
    for name, w in periods.items():
        cur = window(feats, w)
        raw_cur = raw[(raw.request_time >= w[0]) & (raw.request_time < w[1])]
        # 0. validation
        checks = validate_schema(raw_cur, reference["mcc_title_map"])
        rchecks, q = validate_ranges(cur, reference)
        valid[name] = [c for c in checks + rchecks if c["status"] != "PASS"]
        quarantine[name] = q
        clean = cur[~q.values]
        # 1. data drift (on all rows - that is how bad data gets noticed - and on clean rows)
        drift_raw[name] = {
            "amount (PSI, all rows)": psi_numeric(ref.amount.to_numpy(), cur.amount.to_numpy()),
            "amount (PSI, clean rows)": psi_numeric(ref.amount.to_numpy(), clean.amount.to_numpy()),
            "amount (KS stat, clean)": float(ks_2samp(ref.amount, clean.amount).statistic),
            "service_type": psi_categorical(ref.service_type, cur.service_type),
            "merchant_type": psi_categorical(ref.merchant_type, cur.merchant_type),
            "mcc_code": psi_categorical(ref.mcc_code, cur.mcc_code),
            "hour": psi_numeric(ref.hour.to_numpy(float), cur.hour.to_numpy(float), bins=24),
        }
        drift_feat[name] = {c: psi_numeric(ref[c].to_numpy(float), clean[c].to_numpy(float)) for c in NUMERIC
                            if ref[c].nunique() > 2}
        # 2. prediction drift
        s = cur.score.to_numpy()
        pred[name] = dict(score_psi=psi_numeric(ref_scores, s, bins=10), mean_score=float(s.mean()),
                          p99_score=float(np.quantile(s, .99)), decline_rate=float((s >= thr_b).mean()),
                          review_rate=float(((s >= thr_r) & (s < thr_b)).mean()),
                          actual_fraud_rate=float(cur.fraud_label.mean()))
        # 3./4. performance & business on clean rows (quarantined rows go to the fallback path)
        sc = clean.score.to_numpy()
        perf[name] = classification_metrics(clean.fraud_label, sc, thr_b, clean.amount)
        perf[name]["quarantined_rows"] = int(q.sum())
        biz[name] = business(clean, sc, thr_b, thr_r)

    live = list(LIVE_MONTHS)
    # ------------------------------------------------ alert -> action rules
    actions = []
    for name in live:
        v, d, p, m, b = valid[name], drift_raw[name], pred[name], perf[name], biz[name]
        bb = biz["2026-06 (baseline)"]
        fails = [c for c in v if c["status"] == "FAIL"]
        if fails:
            actions.append((name, "Data quality", "; ".join(c["check"] for c in fails),
                            "Investigate data quality: open incident with the integration owner, quarantine affected rows "
                            "(route to rules/fallback), exclude them from performance reporting and from any retraining set. "
                            "Do NOT retrain or move thresholds because of this drift."))
        drifted = [k for k, val in d.items() if "KS" not in k and "all rows" not in k and val >= PSI_WARN]
        if drifted:
            actions.append((name, "Data drift", ", ".join(f"{k} PSI {d[k]:.2f}" for k in drifted),
                            "Check if explained by known seasonality/product change (festive season, UPI growth); "
                            "if performance is stable, document and keep monitoring; refresh reference window at next retrain."))
        if p["score_psi"] >= PSI_WARN:
            actions.append((name, "Prediction drift", f"score PSI {p['score_psi']:.2f}, decline rate "
                            f"{p['decline_rate']:.2%} vs {pred['2026-06 (baseline)']['decline_rate']:.2%}",
                            "Before labels arrive: verify it is not caused by a data issue; if review queue is near capacity, "
                            "temporarily adjust the review threshold."))
        rec_drop = (base["recall"] - m["recall"]) / base["recall"]
        if rec_drop > 0.10 or base["pr_auc"] - m["pr_auc"] > 0.05:
            actions.append((name, "Performance drift", f"recall {m['recall']:.2f} vs {base['recall']:.2f}, "
                            f"PR-AUC {m['pr_auc']:.2f} vs {base['pr_auc']:.2f}",
                            "Concept drift: slice missed fraud to find the new pattern, retrain a challenger with recent "
                            "labels, shadow-test, promote if it wins; add a stop-gap rule for the new pattern meanwhile."))
        if b["fraud_loss_rate_bps"] > 2 * bb["fraud_loss_rate_bps"]:
            actions.append((name, "Business", f"fraud loss {b['fraud_loss_rate_bps']:.0f} bps of volume "
                            f"(Rs {b['fraud_loss_auto_approved']:,.0f}) vs {bb['fraud_loss_rate_bps']:.0f} bps baseline",
                            "Escalate to risk team; tighten threshold for affected segment until challenger is live."))
        if b["approval_rate"] < bb["approval_rate"] - 0.01 or b["false_declines"] > 2 * max(bb["false_declines"], 5):
            actions.append((name, "Business", f"approval {b['approval_rate']:.2%}, false declines {b['false_declines']}",
                            "False declines hurt customers/merchants: check data quality first, then raise decline threshold "
                            "or roll back if a recent release caused it."))
        if b["review_queue_per_day"] > 2 * max(bb["review_queue_per_day"], 1):
            actions.append((name, "Business", f"review queue {b['review_queue_per_day']:.1f}/day vs "
                            f"{bb['review_queue_per_day']:.1f}", "Adjust review threshold to analyst capacity."))

    # ------------------------------------------------ root cause: where is fraud being missed?
    sep = window(feats, LIVE_MONTHS["2026-09"]).assign(q=quarantine["2026-09"].values)
    fr = sep[sep.fraud_label == 1].copy()
    fr["caught"] = fr.score >= thr_b
    ref_merchants = set(raw.loc[(raw.request_time >= REFERENCE[0]) & (raw.request_time < REFERENCE[1]), "merchant_id"])
    fr["merchant"] = np.where(fr.request_id.map(raw.set_index("request_id").merchant_id).isin(ref_merchants),
                              "known at training", "first seen after go-live")
    fr = fr[~fr.q]
    slice_tab = (fr.groupby(["merchant", "mcc_code", "merchant_type", "service_type"])
                 .agg(frauds=("caught", "size"), recall=("caught", "mean")).query("frauds >= 10")
                 .sort_values("frauds", ascending=False).round(2))

    # ------------------------------------------------ challenger (retrain with matured Jul-Aug labels)
    clean = feats.assign(q=pd.concat(quarantine.values()).reindex(feats.index, fill_value=False))
    tr = clean[(clean.request_time >= TRAIN[0]) & (clean.request_time < "2026-08-16") & ~clean.q]
    va = clean[(clean.request_time >= "2026-08-16") & (clean.request_time < "2026-09-01") & ~clean.q]
    te = clean[(clean.request_time >= "2026-09-01") & (clean.request_time < "2026-10-01") & ~clean.q]
    challenger = fit_xgb(tr[FEATURES], tr.fraud_label, va[FEATURES], va.fraud_label, card["scale_pos_weight"])
    ch_thr = threshold_max_f1(va.fraud_label, challenger.predict_proba(va[FEATURES])[:, 1])
    s_ch = challenger.predict_proba(te[FEATURES])[:, 1]
    # variant: up-weight recent months so the new pattern counts more
    w = np.where(tr.request_time >= "2026-07-01", 3.0, 1.0)
    challenger_w = fit_xgb(tr[FEATURES], tr.fraud_label, va[FEATURES], va.fraud_label, card["scale_pos_weight"], w)
    chw_thr = threshold_max_f1(va.fraud_label, challenger_w.predict_proba(va[FEATURES])[:, 1])
    s_chw = challenger_w.predict_proba(te[FEATURES])[:, 1]

    # operational options (no retrain):
    #  (a) risk ops re-tiers post-go-live merchants in 4829/5816 to "high"
    mids = te.request_id.map(raw.set_index("request_id").merchant_id)
    new_risky = (~mids.isin(ref_merchants)) & te.mcc_code.isin(["4829", "5816"])
    te_rt = te.copy(); te_rt.loc[new_risky, "merchant_type"] = "high"
    s_rt = model.predict_proba(te_rt[FEATURES])[:, 1]
    #  (b) stop-gap rule designed from the August investigation, evaluated out-of-time on September:
    #      post-go-live merchant in MCC 4829/5816 AND amount > 2x merchant's running average -> flag
    rule = (new_risky & (te.mer_amt_ratio_prior_mean > 2)).to_numpy()
    as_score = lambda s, thr, extra: np.where(extra, 1.0, s) if extra is not None else s

    def opt(s, thr, extra=None):
        return classification_metrics(te.fraud_label, as_score(s, thr, extra), thr, te.amount)
    cmp = {"champion v1.0 as-is": opt(te.score.to_numpy(), thr_b),
           "champion + merchant re-tiering": opt(s_rt, thr_b),
           "champion + stop-gap rule": opt(te.score.to_numpy(), thr_b, rule),
           "challenger (Feb 1-Aug 15)": opt(s_ch, ch_thr),
           "challenger, recent months x3 weight": opt(s_chw, chw_thr),
           "challenger + stop-gap rule": opt(s_ch, ch_thr, rule)}
    rule_stats = dict(flagged=int(rule.sum()), per_day=float(rule.sum() / te.request_time.dt.date.nunique()),
                      precision=float(te.fraud_label.to_numpy()[rule].mean()) if rule.any() else float("nan"))
    joblib.dump(challenger, ARTIFACTS / "challenger_model.joblib")

    # ------------------------------------------------ figures
    order = list(periods)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for k in ["precision", "recall", "f1", "pr_auc"]:
        ax[0].plot(order, [perf[n][k] for n in order], marker="o", label=k)
    ax[0].set(title="Performance at frozen threshold", ylim=(0, 1.02)); ax[0].legend(); ax[0].tick_params(axis="x", rotation=20)
    dr = pd.DataFrame(drift_raw).drop(index="amount (KS stat, clean)").T
    dr.plot(kind="bar", ax=ax[1]); ax[1].axhline(PSI_WARN, ls=":", c="orange"); ax[1].axhline(PSI_ALERT, ls="--", c="red")
    ax[1].set(title="Data drift (PSI vs Feb-Jun reference)", ylabel="PSI", ylim=(0, 0.3)); ax[1].tick_params(axis="x", rotation=20); ax[1].legend(fontsize=7)
    bins = np.linspace(0, 1, 41)
    ax[2].hist(ref_scores, bins=bins, density=True, alpha=.5, label="reference (May-Jun)", log=True)
    ax[2].hist(window(feats, LIVE_MONTHS["2026-09"]).score, bins=bins, density=True, alpha=.5, label="2026-09", log=True)
    ax[2].axvline(thr_b, c="red", ls="--", label="decline thr"); ax[2].axvline(thr_r, c="orange", ls=":", label="review thr")
    ax[2].set(title="Prediction drift: score distribution", xlabel="fraud score"); ax[2].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIGS / "monitoring_overview.png", dpi=120); plt.close(fig)

    # ------------------------------------------------ report
    f2 = lambda x: f"{x:.3f}"
    def table(d, cols, fmt=f2, label_fn=None):
        out = ["| metric | " + " | ".join(order) + " |", "|---" * (len(order) + 1) + "|"]
        for c in cols:
            vals = [d[n][c] for n in order]
            out.append(f"| {c} | " + " | ".join((label_fn(c, v) if label_fn else fmt(v)) for v in vals) + " |")
        return "\n".join(out)
    psi_lbl = lambda c, v: f"{v:.3f}" + ("" if "KS" in c else f" ({level(v)})")

    md = [f"# Monitoring report - `{card['model_version']}` on live Jul-Sep 2026\n",
          "Reference = Feb 1-Jun 30 (training period after January feature warm-up). Prediction reference = out-of-sample May-Jun scores. "
          "Performance baseline = June hold-out at the frozen decline threshold "
          f"{thr_b:.3f} (review band from {thr_r:.3f}). PSI: <0.10 ok, 0.10-0.25 WARN, >=0.25 ALERT. "
          "KS statistic is reported instead of the p-value because with ~15k rows per month every "
          "difference is 'significant'.\n",
          "## 0. Input validation (non-passing checks)\n"]
    for n in order:
        md.append(f"* **{n}**: " + ("all checks pass" if not valid[n] else
                  "; ".join(f"{c['status']} {c['check']} - {c['detail']}" for c in valid[n])))
    md += ["\n## 1. Data drift - raw inputs\n", table(drift_raw, list(drift_raw[order[0]]), label_fn=psi_lbl),
           "\n### Model features with PSI >= 0.10 in any live month\n"]
    fd = pd.DataFrame(drift_feat)
    fd = fd[(fd[live] >= PSI_WARN).any(axis=1)].round(3)
    md.append(fd.to_markdown() if len(fd) else "none")
    md += ["\n## 2. Prediction drift\n", table(pred, list(pred[order[0]]),
                                              fmt=lambda v: f"{v:.4f}"),
           "\n## 3. Performance drift (frozen threshold)\n",
           table(perf, ["n", "quarantined_rows", "n_fraud", "fraud_rate", "precision", "recall", "f1", "pr_auc", "roc_auc",
                        "fraud_amount_recall"],
                 fmt=lambda v: f"{v:.3f}" if isinstance(v, float) else f"{v:,}"),
           "\nConfusion matrices: " + "; ".join(f"{n}: {perf[n]['confusion_matrix']}" for n in order) + "\n",
           "## 4. Business metrics\n",
           table(biz, list(biz[order[0]]), fmt=lambda v: f"{v:,}" if isinstance(v, int) else
                 (f"{v:.4f}" if abs(v) < 1 else f"{v:,.1f}")),
           "\n## Alerts and actions\n", "| month | type | evidence | action |", "|---|---|---|---|"]
    md += [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in actions]
    md += ["\n## Root cause: September fraud by segment (recall at decline threshold)\n", slice_tab.to_markdown(),
           "\n## Remediation options, evaluated on clean September data\n",
           "Challengers: trained Feb 1-Aug 15, early stopping + max-F1 threshold on Aug 16-31. "
           "Stop-gap rule (designed from the August slice, so September is out-of-time): post-go-live merchant in MCC "
           f"4829/5816 and amount > 2x merchant running average -> {rule_stats['flagged']} flags "
           f"({rule_stats['per_day']:.1f}/day), rule precision {rule_stats['precision']:.2f}. "
           "PR-AUC/ROC-AUC for '+ rule' rows treat rule hits as score 1.0.\n",
           "| option | threshold | precision | recall | F1 | PR-AUC | ROC-AUC | fraud-amount recall |", "|---|---|---|---|---|---|---|---|"]
    md += [f"| {k} | {m['threshold']:.3f} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['pr_auc']:.3f} | "
           f"{m['roc_auc']:.3f} | {m['fraud_amount_recall']:.3f} |" for k, m in cmp.items()]
    md.append(f"\n![overview](figures/monitoring_overview.png)\n")
    (REPORTS / "monitoring_report.md").write_text("\n".join(md) + "\n")
    dump_json(dict(data_drift=drift_raw, feature_psi=drift_feat, prediction=pred, performance=perf,
                   business=biz, actions=actions, remediation=cmp, stop_gap_rule=rule_stats),
              REPORTS / "monitoring_metrics.json")
    print("\n".join(md))


if __name__ == "__main__":
    main()
