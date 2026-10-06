# Model results

Split: train 2026-02-01..2026-05-01 (43,252 rows, 1.25% fraud), valid 2026-05 (15,189), test 2026-06 (15,549, 1.18% fraud).

## Imbalance handling (XGBoost, validation PR-AUC)

| strategy      |   scale_pos_weight |   trees |   valid_pr_auc |
|:--------------|-------------------:|--------:|---------------:|
| none (spw=1)  |                1   |     283 |         0.7822 |
| sqrt(neg/pos) |                8.9 |     639 |         0.7719 |
| neg/pos       |               79.1 |     782 |         0.7685 |

Chosen: **none (spw=1)**.

## Metrics (threshold chosen on validation = max F1, then frozen for test)

| model | split | threshold | precision | recall | F1 | PR-AUC | ROC-AUC | fraud-amount recall | confusion matrix |
|---|---|---|---|---|---|---|---|---|---|
| logreg | valid | 0.940 | 0.731 | 0.676 | 0.702 | 0.752 | 0.978 | 0.784 | TN 14,958 / FP 46 / FN 60 / TP 125 |
| logreg | test | 0.940 | 0.743 | 0.755 | 0.749 | 0.812 | 0.986 | 0.712 | TN 15,317 / FP 48 / FN 45 / TP 139 |
| xgboost | valid | 0.252 | 0.860 | 0.665 | 0.750 | 0.782 | 0.980 | 0.802 | TN 14,984 / FP 20 / FN 62 / TP 123 |
| xgboost | test | 0.252 | 0.889 | 0.739 | 0.807 | 0.877 | 0.996 | 0.791 | TN 15,348 / FP 17 / FN 48 / TP 136 |

## Uncertainty (test, 500 bootstrap resamples, 95% CI)

* XGBoost PR-AUC [0.836, 0.913], LR PR-AUC [0.755, 0.860]
* Paired difference XGBoost - LR PR-AUC [0.034, 0.100]
* XGBoost recall [0.675, 0.799], precision [0.834, 0.933]

## Two-threshold policy on test (XGBoost)

* score >= 0.252 -> **decline** (precision 0.89, recall 0.74)
* 0.003 <= score < 0.252 -> **manual review**; decline+review together catch 97% of fraud (97% of fraud value) with ~20 extra reviews/day

## Top features (gain share)

|                               |   gain_share |
|:------------------------------|-------------:|
| num__dev_secs_since_last      |        0.155 |
| num__is_night                 |        0.122 |
| cat__merchant_type_low        |        0.092 |
| num__dev_age_days_cap30       |        0.09  |
| num__hour                     |        0.059 |
| cat__merchant_type_high       |        0.041 |
| num__dev_amt_sum_1h           |        0.031 |
| num__log_amount               |        0.024 |
| num__dev_cnt_10m              |        0.022 |
| num__dev_merchant_seen_30d    |        0.019 |
| num__ends_in_999              |        0.019 |
| num__mer_age_days_cap30       |        0.016 |
| cat__mcc_code_5941            |        0.016 |
| num__mer_amt_ratio_prior_mean |        0.016 |
| cat__service_type_upi         |        0.014 |
