"""
Short EDA on the *training period only* (Jan-Jun). Live months (Jul-Sep) are
not inspected here so that nothing about the future leaks into modelling choices.

Writes reports/eda.md and figures in reports/figures/.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import DATA, FIGS, REPORTS, SAMPLE
from features import build_features, prepare_raw


def rate_table(df, col, min_n=50):
    t = df.groupby(col, dropna=False).fraud_label.agg(txns="size", frauds="sum", fraud_rate="mean")
    t = t[t.txns >= min_n].sort_values("fraud_rate", ascending=False)
    t["fraud_rate"] = (t.fraud_rate * 100).round(2).astype(str) + "%"
    return t


def main():
    sample = pd.read_csv(SAMPLE)
    raw_all = pd.read_csv(DATA)
    feats_all = build_features(raw_all)
    raw = prepare_raw(raw_all)
    keep = raw.request_time < "2026-07-01"
    raw, feats = raw[keep].copy(), feats_all[keep.values].copy()

    raw["hour"] = raw.request_time.dt.hour
    raw["time_band"] = pd.cut(raw.hour, [-1, 5, 11, 17, 23], labels=["00-05 night", "06-11", "12-17", "18-23"])
    raw["amount_band"] = pd.cut(raw.amount, [0, 1000, 5000, 20000, 50000, np.inf],
                                labels=["<1k", "1k-5k", "5k-20k", "20k-50k", ">50k"])
    raw["device_burst_10m"] = np.where(feats.dev_cnt_10m.fillna(0) >= 2, "2+ prior txns in 10 min", "<2")
    raw["device_new"] = np.where(feats.dev_is_new == 1, "first txn of device", "seen before")
    raw["round_amount"] = np.where(feats.is_round_1000 == 1, "multiple of 1000", "other")

    out = []
    out.append("# EDA (training period: 2026-01-01 to 2026-06-30)\n")
    out.append(f"The provided sample has {len(sample)} rows and {sample.fraud_label.sum()} frauds "
               f"({sample.fraud_label.mean():.1%}); it is far too small and fraud-heavy to be representative, "
               "so it is used for the schema and as the first rows of the synthetic dataset.\n")
    out.append(f"Synthetic training-period data: **{len(raw):,} transactions, "
               f"{raw.fraud_label.sum():,} frauds, fraud rate {raw.fraud_label.mean():.2%}.**\n")
    m = raw.groupby(raw.request_time.dt.strftime("%Y-%m")).fraud_label.agg(txns="size", frauds="sum", fraud_rate="mean")
    m["fraud_rate"] = (m.fraud_rate * 100).round(2).astype(str) + "%"
    out.append("## Fraud rate by month\n\n" + m.to_markdown() + "\n")
    miss = raw.drop(columns=["hour"]).isna().mean().loc[lambda s: s > 0]
    out.append("## Missing values\n\n" + (miss * 100).round(2).astype(str).add("%").to_frame("missing").to_markdown()
               + "\n\nPlus `issuer_bank = NOT_APPLICABLE` for all wallet transactions (structural, not missing).\n")
    out.append("## Amount\n\n" + raw.groupby("fraud_label").amount.describe(percentiles=[.5, .9, .99]).round(0).to_markdown() + "\n")

    for col, title in [("service_type", "Payment method"), ("merchant_type", "Merchant risk tier"),
                       ("mcc_title", "MCC"), ("time_band", "Time of day"), ("amount_band", "Amount band"),
                       ("request_status", "Request status (see leakage note)"),
                       ("device_burst_10m", "Repeated device activity"), ("device_new", "Device novelty"),
                       ("round_amount", "Round amounts")]:
        out.append(f"## {title}\n\n" + rate_table(raw, col).to_markdown() + "\n")

    out.append("""## Takeaways
* Fraud is rare (~1.2%), so accuracy is meaningless (all-genuine = 98.8% accurate). Use PR-AUC, precision/recall.
* Strong signals: high-risk tier & MCCs (card load, money transfer, gambling, dating), night hours,
  large and round amounts, brand-new devices, bursts of attempts from one device, wallet/netbanking.
* `request_status` differs strongly by class (fraud has many FAILED/DECLINED), but the current status is only known
  *after* processing and may itself be the output of an existing fraud rule, so it is not used as a model input.
  The *previous* failures of the device (`dev_fail_cnt_24h`) are used instead.
* Missingness is low (<1%) and looks random; handled by imputation + missing indicators.
""")
    (REPORTS / "eda.md").write_text("\n".join(out))

    # figures
    FIGS.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    hr = raw.groupby("hour").fraud_label.mean() * 100
    ax[0].bar(hr.index, hr.values, color="#c0392b")
    ax[0].set(title="Fraud rate by hour of day", xlabel="hour", ylabel="fraud rate %")
    bins = np.linspace(4, 12.5, 50)
    for lbl, c in [(0, "#2c7fb8"), (1, "#c0392b")]:
        ax[1].hist(np.log(raw.loc[raw.fraud_label == lbl, "amount"]), bins=bins, density=True,
                   alpha=.55, color=c, label="fraud" if lbl else "genuine")
    ax[1].set(title="log(amount) by class", xlabel="log(amount)"); ax[1].legend()
    fig.tight_layout(); fig.savefig(FIGS / "eda_hour_amount.png", dpi=120); plt.close(fig)
    print("wrote reports/eda.md")


if __name__ == "__main__":
    main()
