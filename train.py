"""
Train and evaluate fraud models with a time-based split.

  warm-up: 2026-01                   (feature history only)
  train : 2026-02-01 .. 2026-04-30   (fit)
  valid : 2026-05                     (early stopping, imbalance choice, thresholds)
  test  : 2026-06                     (untouched hold-out; becomes the monitoring baseline)

Models:
  * Logistic Regression (baseline)  - one-hot + scaled numerics, class_weight="balanced"
  * XGBoost (champion candidate)    - one-hot, native NaN handling, scale_pos_weight

Outputs: artifacts/model.joblib, artifacts/model_card.json, artifacts/reference_stats.json,
         reports/model_results.md, reports/figures/*.png
"""
from __future__ import annotations

from datetime import datetime, timezone

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common import (ARTIFACTS, DATA, FIGS, REFERENCE, REPORTS, TEST, TRAIN, VALID,
                    classification_metrics, dump_json, file_sha256, threshold_for_recall,
                    threshold_max_f1, window)
from features import CATEGORICAL, FEATURES, NUMERIC, TARGET, build_features

SEED = 7
MODEL_VERSION = "fraud-xgb-v1.0.0"


def categorical_pipe():
    return Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="MISSING")),
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20,
                                 sparse_output=False)),
    ])


def make_lr_preprocessor():
    num = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                    ("clip_log", _SignedLog()), ("scale", StandardScaler())])
    return ColumnTransformer([("num", num, NUMERIC), ("cat", categorical_pipe(), CATEGORICAL)])


def make_tree_preprocessor():
    # trees: no scaling, NaN left in place (XGBoost learns a default split direction)
    return ColumnTransformer([("num", "passthrough", NUMERIC), ("cat", categorical_pipe(), CATEGORICAL)])


class _SignedLog(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    """log1p for heavy-tailed counts/ratios so the linear model isn't dominated by outliers."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.sign(X) * np.log1p(np.abs(X))


def fit_xgb(Xtr, ytr, Xva, yva, scale_pos_weight, sample_weight=None):
    pre = make_tree_preprocessor().fit(Xtr)
    clf = xgb.XGBClassifier(
        n_estimators=2000, learning_rate=0.03, max_depth=5, min_child_weight=3,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight, eval_metric="aucpr",
        early_stopping_rounds=100, random_state=SEED, n_jobs=4)
    clf.fit(pre.transform(Xtr), ytr, sample_weight=sample_weight,
            eval_set=[(pre.transform(Xva), yva)], verbose=False)
    return Pipeline([("pre", pre), ("clf", clf)])


def main():
    REPORTS.mkdir(exist_ok=True); FIGS.mkdir(parents=True, exist_ok=True); ARTIFACTS.mkdir(exist_ok=True)
    raw = pd.read_csv(DATA)
    feats = build_features(raw)
    tr, va, te = window(feats, TRAIN), window(feats, VALID), window(feats, TEST)
    Xtr, ytr = tr[FEATURES], tr[TARGET]
    Xva, yva = va[FEATURES], va[TARGET]
    Xte, yte = te[FEATURES], te[TARGET]
    print(f"train {len(tr):,} ({ytr.mean():.2%})  valid {len(va):,} ({yva.mean():.2%})  "
          f"test {len(te):,} ({yte.mean():.2%})")

    # ---------------- baseline: logistic regression ----------------
    lr = Pipeline([("pre", make_lr_preprocessor()),
                   ("clf", LogisticRegression(class_weight="balanced", C=0.3, max_iter=3000))])
    lr.fit(Xtr, ytr)

    # ---------------- tree model: imbalance strategy chosen on validation ----------------
    neg_pos = float((ytr == 0).sum() / (ytr == 1).sum())
    candidates = {"none (spw=1)": 1.0, "sqrt(neg/pos)": float(np.sqrt(neg_pos)), "neg/pos": neg_pos}
    imb_rows, fitted = [], {}
    for name, spw in candidates.items():
        m = fit_xgb(Xtr, ytr, Xva, yva, spw)
        s = m.predict_proba(Xva)[:, 1]
        fitted[name] = m
        imb_rows.append(dict(strategy=name, scale_pos_weight=round(spw, 1),
                             trees=m[-1].best_iteration + 1,
                             valid_pr_auc=round(average_precision_score(yva, s), 4)))
    imb = pd.DataFrame(imb_rows)
    best_name = imb.sort_values("valid_pr_auc", ascending=False).strategy.iloc[0]
    xgbm = fitted[best_name]
    print(imb.to_string(index=False)); print("chosen:", best_name)

    # ---------------- thresholds from VALIDATION only ----------------
    s_va = {"logreg": lr.predict_proba(Xva)[:, 1], "xgboost": xgbm.predict_proba(Xva)[:, 1]}
    s_te = {"logreg": lr.predict_proba(Xte)[:, 1], "xgboost": xgbm.predict_proba(Xte)[:, 1]}
    thr = {k: threshold_max_f1(yva, v) for k, v in s_va.items()}
    review_thr = threshold_for_recall(yva, s_va["xgboost"], 0.90)
    review_thr = min(review_thr, thr["xgboost"])

    results = {}
    for k in ["logreg", "xgboost"]:
        results[k] = dict(
            valid=classification_metrics(yva, s_va[k], thr[k], va.amount),
            test=classification_metrics(yte, s_te[k], thr[k], te.amount))
    # "block OR review" view: what share of fraud reaches either auto-decline or an analyst
    review_view = classification_metrics(yte, s_te["xgboost"], review_thr, te.amount)
    queue_per_day = review_view["confusion_matrix"]["tp"] + review_view["confusion_matrix"]["fp"]
    queue_per_day -= results["xgboost"]["test"]["confusion_matrix"]["tp"] + results["xgboost"]["test"]["confusion_matrix"]["fp"]
    queue_per_day /= te.request_time.dt.date.nunique()

    # ---------------- bootstrap CIs on test (few hundred frauds -> metrics are noisy) ----------------
    rng = np.random.default_rng(SEED)
    y_np = yte.to_numpy()
    boot = {"xgb_pr_auc": [], "lr_pr_auc": [], "diff_pr_auc": [], "xgb_recall": [], "xgb_precision": []}
    for _ in range(500):
        i = rng.integers(0, len(y_np), len(y_np))
        if y_np[i].sum() == 0:
            continue
        px = average_precision_score(y_np[i], s_te["xgboost"][i]); pl = average_precision_score(y_np[i], s_te["logreg"][i])
        pred_i = s_te["xgboost"][i] >= thr["xgboost"]
        boot["xgb_pr_auc"].append(px); boot["lr_pr_auc"].append(pl); boot["diff_pr_auc"].append(px - pl)
        boot["xgb_recall"].append(pred_i[y_np[i] == 1].mean())
        boot["xgb_precision"].append(y_np[i][pred_i].mean() if pred_i.any() else 0)
    ci = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in boot.items()}

    # ---------------- figures ----------------
    fig, ax = plt.subplots(figsize=(5.5, 4.3))
    for k, c in [("logreg", "#7f8c8d"), ("xgboost", "#c0392b")]:
        p, r, _ = precision_recall_curve(yte, s_te[k])
        ax.plot(r, p, color=c, label=f"{k} (PR-AUC {results[k]['test']['pr_auc']:.3f})")
    ax.axhline(yte.mean(), ls=":", color="k", label=f"random ({yte.mean():.3f})")
    ax.set(xlabel="recall", ylabel="precision", title="Precision-recall, June hold-out"); ax.legend()
    fig.tight_layout(); fig.savefig(FIGS / "pr_curve_test.png", dpi=120); plt.close(fig)

    booster = xgbm[-1].get_booster()
    names = list(xgbm[0].get_feature_names_out())
    gain = booster.get_score(importance_type="gain")
    imp = pd.Series({names[int(k[1:])]: v for k, v in gain.items()}).sort_values(ascending=False)
    imp = (imp / imp.sum()).head(15)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.barh(imp.index[::-1].str.replace("num__", "").str.replace("cat__", ""), imp.values[::-1], color="#c0392b")
    ax.set(title="XGBoost feature importance (gain share)"); fig.tight_layout()
    fig.savefig(FIGS / "feature_importance.png", dpi=120); plt.close(fig)

    # ---------------- artifacts ----------------
    joblib.dump(xgbm, ARTIFACTS / "model.joblib")
    joblib.dump(lr, ARTIFACTS / "baseline_logreg.joblib")
    ref = window(feats, REFERENCE)
    ref_raw = raw.assign(request_time=pd.to_datetime(raw.request_time))
    ref_raw = ref_raw[(ref_raw.request_time >= REFERENCE[0]) & (ref_raw.request_time < REFERENCE[1])]
    ref_scores = np.r_[s_va["xgboost"], s_te["xgboost"]]
    reference = dict(
        period=REFERENCE,
        numeric={c: dict(min=float(ref[c].min()), max=float(ref[c].max()),
                         p01=float(ref[c].quantile(.01)), p99=float(ref[c].quantile(.99)),
                         missing_rate=float(ref[c].isna().mean()))
                 for c in NUMERIC + ["amount"]},
        categorical={c: sorted(map(str, ref[c].dropna().unique())) for c in CATEGORICAL},
        amount_median_by_service=ref.groupby("service_type").amount.median().to_dict(),
        mcc_title_map=ref_raw.groupby("mcc_code").mcc_title.first().astype(str).to_dict(),
        amount_values=ref.amount.sample(min(20000, len(ref)), random_state=SEED).round(2).tolist(),
        score_values=np.round(ref_scores, 6).tolist(),
        category_freq={c: ref[c].astype(str).value_counts(normalize=True).to_dict()
                       for c in ["service_type", "merchant_type", "mcc_code"]},
    )
    dump_json(reference, ARTIFACTS / "reference_stats.json")
    card = dict(
        model_version=MODEL_VERSION, created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        algorithm="XGBoost", imbalance_strategy=best_name,
        scale_pos_weight=candidates[best_name], best_iteration=int(xgbm[-1].best_iteration),
        data_file=str(DATA.name), data_sha256=file_sha256(DATA),
        windows=dict(train=TRAIN, valid=VALID, test=TEST),
        features=FEATURES, thresholds=dict(block=thr["xgboost"], review=review_thr),
        threshold_policy="block = max-F1 on validation; review band = [review, block) reaching 90% validation recall",
        test_bootstrap_95ci=ci,
        metrics=dict(xgboost=results["xgboost"], logreg=results["logreg"],
                     xgboost_test_block_or_review=review_view),
        libs=dict(sklearn=sklearn.__version__, xgboost=xgb.__version__, pandas=pd.__version__),
        release_gate=dict(min_test_pr_auc=0.60, min_test_recall=0.60, min_test_precision=0.50,
                          must_beat_baseline_pr_auc=True),
        rollback_to="fraud-rules-v0 (existing rule engine) or previous model version in registry",
    )
    dump_json(card, ARTIFACTS / "model_card.json")

    # ---------------- report ----------------
    def row(k, split):
        m = results[k][split]; cm = m["confusion_matrix"]
        return (f"| {k} | {split} | {m['threshold']:.3f} | {m['precision']:.3f} | {m['recall']:.3f} | "
                f"{m['f1']:.3f} | {m['pr_auc']:.3f} | {m['roc_auc']:.3f} | {m['fraud_amount_recall']:.3f} | "
                f"TN {cm['tn']:,} / FP {cm['fp']} / FN {cm['fn']} / TP {cm['tp']} |")
    lines = ["# Model results\n",
             f"Split: train {TRAIN[0]}..{TRAIN[1]} ({len(tr):,} rows, {ytr.mean():.2%} fraud), "
             f"valid {VALID[0][:7]} ({len(va):,}), test {TEST[0][:7]} ({len(te):,}, {yte.mean():.2%} fraud).\n",
             "## Imbalance handling (XGBoost, validation PR-AUC)\n", imb.to_markdown(index=False),
             f"\nChosen: **{best_name}**.\n",
             "## Metrics (threshold chosen on validation = max F1, then frozen for test)\n",
             "| model | split | threshold | precision | recall | F1 | PR-AUC | ROC-AUC | fraud-amount recall | confusion matrix |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    lines += [row(k, s) for k in ["logreg", "xgboost"] for s in ["valid", "test"]]
    rv = review_view
    fmt_ci = lambda k: f"[{ci[k][0]:.3f}, {ci[k][1]:.3f}]"
    lines += ["\n## Uncertainty (test, 500 bootstrap resamples, 95% CI)\n",
              f"* XGBoost PR-AUC {fmt_ci('xgb_pr_auc')}, LR PR-AUC {fmt_ci('lr_pr_auc')}",
              f"* Paired difference XGBoost - LR PR-AUC {fmt_ci('diff_pr_auc')}",
              f"* XGBoost recall {fmt_ci('xgb_recall')}, precision {fmt_ci('xgb_precision')}"]
    lines += [f"\n## Two-threshold policy on test (XGBoost)\n",
              f"* score >= {thr['xgboost']:.3f} -> **decline** (precision {results['xgboost']['test']['precision']:.2f}, "
              f"recall {results['xgboost']['test']['recall']:.2f})",
              f"* {review_thr:.3f} <= score < {thr['xgboost']:.3f} -> **manual review**; decline+review together catch "
              f"{rv['recall']:.0%} of fraud ({rv['fraud_amount_recall']:.0%} of fraud value) with ~{queue_per_day:.0f} extra reviews/day",
              "\n## Top features (gain share)\n", imp.round(3).to_frame("gain_share").to_markdown()]
    (REPORTS / "model_results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
