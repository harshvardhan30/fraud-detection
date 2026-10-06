# EDA (training period: 2026-01-01 to 2026-06-30)

The provided sample has 12 rows and 4 frauds (33.3%); it is far too small and fraud-heavy to be representative, so it is used for the schema and as the first rows of the synthetic dataset.

Synthetic training-period data: **87,531 transactions, 1,063 frauds, fraud rate 1.21%.**

## Fraud rate by month

| request_time   |   txns |   frauds | fraud_rate   |
|:---------------|-------:|---------:|:-------------|
| 2026-01        |  13541 |      154 | 1.14%        |
| 2026-02        |  14041 |      166 | 1.18%        |
| 2026-03        |  14511 |      199 | 1.37%        |
| 2026-04        |  14700 |      175 | 1.19%        |
| 2026-05        |  15189 |      185 | 1.22%        |
| 2026-06        |  15549 |      184 | 1.18%        |

## Missing values

|               | missing   |
|:--------------|:----------|
| device_id     | 0.49%     |
| merchant_city | 1.02%     |
| issuer_bank   | 0.4%      |

Plus `issuer_bank = NOT_APPLICABLE` for all wallet transactions (structural, not missing).

## Amount

|   fraud_label |   count |   mean |   std |   min |   50% |   90% |    99% |            max |
|--------------:|--------:|-------:|------:|------:|------:|------:|-------:|---------------:|
|             0 |   86468 |   5707 | 19248 |    38 |  1624 | 12324 |  65000 |      1.616e+06 |
|             1 |    1063 |  24302 | 25314 |     6 | 18388 | 55132 | 118487 | 170146         |

## Payment method

| service_type   |   txns |   frauds | fraud_rate   |
|:---------------|-------:|---------:|:-------------|
| wallet         |  13055 |      312 | 2.39%        |
| netbanking     |  20322 |      296 | 1.46%        |
| imps           |  10715 |      116 | 1.08%        |
| upi            |  43439 |      339 | 0.78%        |

## Merchant risk tier

| merchant_type   |   txns |   frauds | fraud_rate   |
|:----------------|-------:|---------:|:-------------|
| high            |   5126 |      609 | 11.88%       |
| medium          |  13535 |      283 | 2.09%        |
| low             |  68870 |      171 | 0.25%        |

## MCC

| mcc_title                      |   txns |   frauds | fraud_rate   |
|:-------------------------------|-------:|---------:|:-------------|
| Betting and Gambling           |    783 |      125 | 15.96%       |
| Money Transfer                 |   1635 |      182 | 11.13%       |
| Prepaid/Stored Value Card Load |   1418 |      149 | 10.51%       |
| Dating and Escort Services     |    524 |       43 | 8.21%        |
| Electronic Stores              |   4440 |      220 | 4.95%        |
| Jewelry Stores                 |   1782 |       77 | 4.32%        |
| Digital Goods - Games          |   2528 |       27 | 1.07%        |
| Travel Agencies                |   2814 |       29 | 1.03%        |
| Educational Services           |   3452 |       18 | 0.52%        |
| Sporting Goods Stores          |   3069 |       14 | 0.46%        |
| Department Stores              |  10921 |       34 | 0.31%        |
| Utilities                      |   8264 |       24 | 0.29%        |
| Grocery Stores                 |  19488 |       56 | 0.29%        |
| Eating Places and Restaurants  |  17402 |       43 | 0.25%        |
| Service Stations               |   9011 |       22 | 0.24%        |

## Time of day

| time_band   |   txns |   frauds | fraud_rate   |
|:------------|-------:|---------:|:-------------|
| 00-05 night |   3842 |      686 | 17.86%       |
| 18-23       |  29670 |      156 | 0.53%        |
| 06-11       |  23520 |      113 | 0.48%        |
| 12-17       |  30499 |      108 | 0.35%        |

## Amount band

| amount_band   |   txns |   frauds | fraud_rate   |
|:--------------|-------:|---------:|:-------------|
| >50k          |   1465 |      128 | 8.74%        |
| 20k-50k       |   4286 |      374 | 8.73%        |
| 5k-20k        |  13319 |      282 | 2.12%        |
| 1k-5k         |  38665 |      189 | 0.49%        |
| <1k           |  29796 |       90 | 0.3%         |

## Request status (see leakage note)

| request_status   |   txns |   frauds | fraud_rate   |
|:-----------------|-------:|---------:|:-------------|
| DECLINED         |   2214 |      168 | 7.59%        |
| FAILED           |   4047 |      268 | 6.62%        |
| SUCCESS          |  81270 |      627 | 0.77%        |

## Repeated device activity

| device_burst_10m        |   txns |   frauds | fraud_rate   |
|:------------------------|-------:|---------:|:-------------|
| 2+ prior txns in 10 min |   1610 |      378 | 23.48%       |
| <2                      |  85921 |      685 | 0.8%         |

## Device novelty

| device_new          |   txns |   frauds | fraud_rate   |
|:--------------------|-------:|---------:|:-------------|
| first txn of device |   7186 |      273 | 3.8%         |
| seen before         |  80345 |      790 | 0.98%        |

## Round amounts

| round_amount     |   txns |   frauds | fraud_rate   |
|:-----------------|-------:|---------:|:-------------|
| multiple of 1000 |   6669 |      199 | 2.98%        |
| other            |  80862 |      864 | 1.07%        |

## Takeaways
* Fraud is rare (~1.2%), so accuracy is meaningless (all-genuine = 98.8% accurate). Use PR-AUC, precision/recall.
* Strong signals: high-risk tier & MCCs (card load, money transfer, gambling, dating), night hours,
  large and round amounts, brand-new devices, bursts of attempts from one device, wallet/netbanking.
* `request_status` differs strongly by class (fraud has many FAILED/DECLINED), but the current status is only known
  *after* processing and may itself be the output of an existing fraud rule, so it is not used as a model input.
  The *previous* failures of the device (`dev_fail_cnt_24h`) are used instead.
* Missingness is low (<1%) and looks random; handled by imputation + missing indicators.
