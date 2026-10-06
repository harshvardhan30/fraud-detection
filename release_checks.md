# Release checks for `fraud-xgb-v1.0.0`

**Verdict: PASS - release to shadow, then canary**

## 1. Input schema & data types (June batch)

| check | status | detail |
|---|---|---|
| required columns present | PASS | 14 columns |
| request_time parses as datetime | PASS | 0 unparseable |
| amount numeric | PASS | 0 non-numeric |
| amount > 0 | PASS | 0 non-positive |
| mcc_code is 4-digit integer | PASS |  |
| no nulls in critical fields | PASS |  |
| request_id unique | PASS | 0 duplicates |
| service_type in allowed set | PASS |  |
| merchant_type in allowed set | PASS |  |
| request_status in allowed set | PASS |  |
| currency_code in allowed set | PASS |  |
| fraud_label in {0,1} | PASS |  |
| mcc_code -> mcc_title consistent | PASS | 0 mismatches |

## 2. Missing values & feature ranges (June batch vs Jan-Jun reference)

| check | status | detail |
|---|---|---|
| amount within plausible range | PASS | 0 rows > 4,848,000 (3x training max); by service: {} |
| daily median amount per service within 3x of training | PASS | max ratio 1.65x |
| share of amount above training p99 | PASS | 1.04% (expected ~1%) |
| missing rate vs training (<= 2x + 1pp) | PASS |  |
| unseen categories (mapped to 'infrequent') | PASS |  |

## 3. Training-serving feature consistency (June, online state replayed from Jan 1)

| check | status | detail |
|---|---|---|
| batch (training) vs online (serving) features identical | PASS | 15,549 rows x 28 features compared; mismatches=0 |

## 4. Model version, evaluation & release gate

| check | status | detail |
|---|---|---|
| data hash matches model card | PASS | 9f767466f79eeef2 |
| evaluation reproducible from saved artifact | PASS | PR-AUC 0.8768 |
| test PR-AUC >= 0.6 | PASS | 0.877 |
| test recall >= 0.6 | PASS | 0.739 |
| test precision >= 0.5 | PASS | 0.889 |
| beats logistic-regression baseline (PR-AUC) | PASS | 0.877 vs 0.812 |
| model version recorded | PASS | fraud-xgb-v1.0.0 |
| rollback target defined | PASS | fraud-rules-v0 (existing rule engine) or previous model version in registry |

## Rollout & rollback plan
1. Register the model + `model_card.json` (version, data hash, windows, thresholds, metrics, library versions).
2. **Shadow** for 1-2 weeks: score live traffic, take no action; compare score distribution, alert volume and
   (as labels arrive) precision/recall against the current production system.
3. **Canary** 5-10% of traffic with the decline threshold; watch approval rate, false declines, review queue, latency.
4. Full rollout. The previous version stays deployed behind a feature flag.
5. **Rollback trigger**: schema/parity failure, approval rate drop > 1pp, review queue > 2x capacity, or any
   error-rate/latency SLO breach -> flip traffic back to the previous version (or the rule engine) in one config change.
