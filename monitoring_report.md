# Monitoring report - `fraud-xgb-v1.0.0` on live Jul-Sep 2026

Reference = Feb 1-Jun 30 (training period after January feature warm-up). Prediction reference = out-of-sample May-Jun scores. Performance baseline = June hold-out at the frozen decline threshold 0.252 (review band from 0.003). PSI: <0.10 ok, 0.10-0.25 WARN, >=0.25 ALERT. KS statistic is reported instead of the p-value because with ~15k rows per month every difference is 'significant'.

## 0. Input validation (non-passing checks)

* **2026-06 (baseline)**: all checks pass
* **2026-07**: all checks pass
* **2026-08**: all checks pass
* **2026-09**: FAIL amount within plausible range - 22 rows > 4,848,000 (3x training max); by service: {'wallet': 22}; FAIL daily median amount per service within 3x of training - 16 service-days flagged ({'wallet': 16}), first on 2026-09-15, median ratio up to 291x -> likely unit error; 963 rows quarantined; WARN share of amount above training p99 - 6.01% (expected ~1%)

## 1. Data drift - raw inputs

| metric | 2026-06 (baseline) | 2026-07 | 2026-08 | 2026-09 |
|---|---|---|---|---|
| amount (PSI, all rows) | 0.000 (ok) | 0.007 (ok) | 0.022 (ok) | 0.087 (ok) |
| amount (PSI, clean rows) | 0.000 (ok) | 0.007 (ok) | 0.022 (ok) | 0.050 (ok) |
| amount (KS stat, clean) | 0.005 | 0.038 | 0.057 | 0.092 |
| service_type | 0.000 (ok) | 0.013 (ok) | 0.049 (ok) | 0.117 (WARN) |
| merchant_type | 0.000 (ok) | 0.000 (ok) | 0.001 (ok) | 0.007 (ok) |
| mcc_code | 0.001 (ok) | 0.002 (ok) | 0.025 (ok) | 0.063 (ok) |
| hour | 0.000 (ok) | 0.001 (ok) | 0.001 (ok) | 0.001 (ok) |

### Model features with PSI >= 0.10 in any live month

none

## 2. Prediction drift

| metric | 2026-06 (baseline) | 2026-07 | 2026-08 | 2026-09 |
|---|---|---|---|---|
| score_psi | 0.0002 | 0.0018 | 0.0019 | 0.0211 |
| mean_score | 0.0093 | 0.0091 | 0.0110 | 0.0116 |
| p99_score | 0.2406 | 0.2293 | 0.5288 | 0.5588 |
| decline_rate | 0.0098 | 0.0099 | 0.0122 | 0.0127 |
| review_rate | 0.0379 | 0.0349 | 0.0393 | 0.0539 |
| actual_fraud_rate | 0.0118 | 0.0117 | 0.0197 | 0.0274 |

## 3. Performance drift (frozen threshold)

| metric | 2026-06 (baseline) | 2026-07 | 2026-08 | 2026-09 |
|---|---|---|---|---|
| n | 15,549 | 15,958 | 17,198 | 17,085 |
| quarantined_rows | 0 | 0 | 0 | 963 |
| n_fraud | 184 | 186 | 339 | 462 |
| fraud_rate | 0.012 | 0.012 | 0.020 | 0.027 |
| precision | 0.889 | 0.854 | 0.847 | 0.882 |
| recall | 0.739 | 0.726 | 0.522 | 0.307 |
| f1 | 0.807 | 0.785 | 0.646 | 0.456 |
| pr_auc | 0.877 | 0.865 | 0.657 | 0.420 |
| roc_auc | 0.996 | 0.993 | 0.949 | 0.826 |
| fraud_amount_recall | 0.791 | 0.833 | 0.683 | 0.584 |

Confusion matrices: 2026-06 (baseline): {'tn': 15348, 'fp': 17, 'fn': 48, 'tp': 136}; 2026-07: {'tn': 15749, 'fp': 23, 'fn': 51, 'tp': 135}; 2026-08: {'tn': 16827, 'fp': 32, 'fn': 162, 'tp': 177}; 2026-09: {'tn': 16604, 'fp': 19, 'fn': 320, 'tp': 142}

## 4. Business metrics

| metric | 2026-06 (baseline) | 2026-07 | 2026-08 | 2026-09 |
|---|---|---|---|---|
| txns | 15,549 | 15,958 | 17,198 | 17,085 |
| approval_rate | 0.9902 | 0.9901 | 0.9878 | 0.9906 |
| false_declines | 17 | 23 | 32 | 19 |
| false_decline_value | 653,271.7 | 586,413.9 | 1,420,321.1 | 1,176,589.2 |
| fraud_value_total | 3,741,144.8 | 4,976,309.6 | 6,324,889.1 | 7,033,553.8 |
| fraud_loss_auto_approved | 118,591.4 | 188,664.9 | 859,839.1 | 2,190,336.1 |
| fraud_loss_rate_bps | 12.9 | 18.6 | 72.8 | 170.4 |
| review_queue_per_day | 19.7 | 18.0 | 21.8 | 20.9 |
| review_hit_rate | 0.0712 | 0.0718 | 0.0962 | 0.0877 |

## Alerts and actions

| month | type | evidence | action |
|---|---|---|---|
| 2026-08 | Performance drift | recall 0.52 vs 0.74, PR-AUC 0.66 vs 0.88 | Concept drift: slice missed fraud to find the new pattern, retrain a challenger with recent labels, shadow-test, promote if it wins; add a stop-gap rule for the new pattern meanwhile. |
| 2026-08 | Business | fraud loss 73 bps of volume (Rs 859,839) vs 13 bps baseline | Escalate to risk team; tighten threshold for affected segment until challenger is live. |
| 2026-09 | Data quality | amount within plausible range; daily median amount per service within 3x of training | Investigate data quality: open incident with the integration owner, quarantine affected rows (route to rules/fallback), exclude them from performance reporting and from any retraining set. Do NOT retrain or move thresholds because of this drift. |
| 2026-09 | Data drift | service_type PSI 0.12 | Check if explained by known seasonality/product change (festive season, UPI growth); if performance is stable, document and keep monitoring; refresh reference window at next retrain. |
| 2026-09 | Performance drift | recall 0.31 vs 0.74, PR-AUC 0.42 vs 0.88 | Concept drift: slice missed fraud to find the new pattern, retrain a challenger with recent labels, shadow-test, promote if it wins; add a stop-gap rule for the new pattern meanwhile. |
| 2026-09 | Business | fraud loss 170 bps of volume (Rs 2,190,336) vs 13 bps baseline | Escalate to risk team; tighten threshold for affected segment until challenger is live. |

## Root cause: September fraud by segment (recall at decline threshold)

|                                                       |   frauds |   recall |
|:------------------------------------------------------|---------:|---------:|
| ('first seen after go-live', '5816', 'medium', 'upi') |       97 |     0    |
| ('first seen after go-live', '5816', 'low', 'upi')    |       97 |     0    |
| ('first seen after go-live', '4829', 'medium', 'upi') |       48 |     0    |
| ('first seen after go-live', '4829', 'low', 'upi')    |       36 |     0    |
| ('known at training', '4829', 'high', 'upi')          |       25 |     0.76 |
| ('known at training', '5732', 'medium', 'upi')        |       23 |     0.7  |
| ('known at training', '6540', 'high', 'upi')          |       13 |     0.54 |
| ('known at training', '7995', 'high', 'upi')          |       13 |     0.92 |

## Remediation options, evaluated on clean September data

Challengers: trained Feb 1-Aug 15, early stopping + max-F1 threshold on Aug 16-31. Stop-gap rule (designed from the August slice, so September is out-of-time): post-go-live merchant in MCC 4829/5816 and amount > 2x merchant running average -> 235 flags (7.8/day), rule precision 0.94. PR-AUC/ROC-AUC for '+ rule' rows treat rule hits as score 1.0.

| option | threshold | precision | recall | F1 | PR-AUC | ROC-AUC | fraud-amount recall |
|---|---|---|---|---|---|---|---|
| champion v1.0 as-is | 0.252 | 0.882 | 0.307 | 0.456 | 0.420 | 0.826 | 0.584 |
| champion + merchant re-tiering | 0.252 | 0.851 | 0.320 | 0.465 | 0.549 | 0.959 | 0.594 |
| champion + stop-gap rule | 0.252 | 0.914 | 0.784 | 0.844 | 0.813 | 0.948 | 0.881 |
| challenger (Feb 1-Aug 15) | 0.240 | 0.857 | 0.351 | 0.498 | 0.527 | 0.943 | 0.622 |
| challenger, recent months x3 weight | 0.115 | 0.768 | 0.372 | 0.501 | 0.530 | 0.942 | 0.644 |
| challenger + stop-gap rule | 0.240 | 0.900 | 0.814 | 0.855 | 0.828 | 0.976 | 0.911 |

![overview](figures/monitoring_overview.png)

