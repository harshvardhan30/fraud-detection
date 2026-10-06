"""
Synthetic payment-gateway transaction generator.

The provided sample (12 rows) only demonstrates the schema, so this script
creates Jan-Sep 2026 transactions with the same columns and plausible fraud
mechanics. The 12 original rows are kept at the top of the file.

Fraud mechanics baked in (so we can check whether the model *finds* them):
  * Pattern A "account takeover / card-load burst" (all months):
      new or compromised device, high-risk merchants / MCCs, late-night skew,
      bursts of several attempts within minutes, large & often round amounts,
      many FAILED/DECLINED attempts before a SUCCESS.
  * Genuine traffic also contains short sessions, retries after failures,
    occasional large purchases and night activity, so no single rule is perfect.

Deliberate changes in the "live" months (Jul-Sep) to exercise monitoring:
  * Festive-season amount inflation (genuine amounts x1.10 / x1.20 / x1.35).
  * Payment-mix shift toward UPI.
  * Pattern B (new, from August): compromised existing devices making
    daytime, mid-size, single UPI payments to newly onboarded "digital goods"
    merchants that were mis-tiered as low/medium risk. The champion model has
    never seen this pattern -> recall should drop.
  * Data-quality bug: from 2026-09-15 one wallet integration sends amounts in
    paise instead of rupees (x100). Monitoring should flag this as a data
    issue, not as a reason to retrain.

Usage:
    python src/generate_data.py --out data/transactions_synthetic.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42

# code, title, risk tier, typical genuine amount (INR), genuine popularity weight
MCC_TABLE = [
    (5411, "Grocery Stores", "low", 900, 20),
    (5311, "Department Stores", "low", 2500, 10),
    (5812, "Eating Places and Restaurants", "low", 700, 14),
    (5541, "Service Stations", "low", 1500, 8),
    (4900, "Utilities", "low", 1800, 8),
    (8299, "Educational Services", "low", 9000, 4),
    (5941, "Sporting Goods Stores", "low", 4000, 3),
    (5732, "Electronic Stores", "medium", 18000, 5),
    (5944, "Jewelry Stores", "medium", 30000, 2),
    (4722, "Travel Agencies", "medium", 12000, 4),
    (5816, "Digital Goods - Games", "medium", 600, 3),
    (4829, "Money Transfer", "high", 6000, 2),
    (6540, "Prepaid/Stored Value Card Load", "high", 3000, 2),
    (7273, "Dating and Escort Services", "high", 1500, 0.5),
    (7995, "Betting and Gambling", "high", 2000, 1),
]
MCC = pd.DataFrame(MCC_TABLE, columns=["mcc_code", "mcc_title", "tier", "typ_amount", "pop"])

STATES = {
    "Maharashtra": ["Mumbai", "Pune", "Nagpur"],
    "Delhi": ["New Delhi"],
    "Karnataka": ["Bengaluru", "Mysuru"],
    "Tamil Nadu": ["Chennai", "Coimbatore"],
    "Odisha": ["Bhubaneswar"],
    "Rajasthan": ["Jaipur", "Udaipur"],
    "Uttar Pradesh": ["Lucknow", "Noida"],
    "West Bengal": ["Kolkata"],
    "Gujarat": ["Ahmedabad", "Surat"],
    "Telangana": ["Hyderabad"],
}
BANKS = ["SBI", "HDFC", "ICICI", "AXIS", "KOTAK", "BOB", "PNB", "INDUSIND", "YES"]
SERVICES = np.array(["upi", "wallet", "imps", "netbanking"])
TIERS = ["low", "medium", "high"]

# genuine hour-of-day profile (daytime/evening heavy, some night activity)
GEN_HOUR_W = np.array([1.2, 0.7, 0.4, 0.3, 0.3, 0.6, 1.5, 3, 4.5, 5.5, 6, 6, 6.5, 6,
                       5.5, 5.5, 5.5, 6, 6.5, 7, 7, 6, 4.5, 2.5])
GEN_HOUR_P = GEN_HOUR_W / GEN_HOUR_W.sum()

MONTHS = range(1, 10)  # Jan..Sep 2026
GENUINE_PER_MONTH = 11000
AMOUNT_INFLATION = {7: 1.10, 8: 1.20, 9: 1.35}
UPI_SHIFT = {7: 0.10, 8: 0.20, 9: 0.30}


def month_bounds(m: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(2026, m, 1)
    return start, start + pd.offsets.MonthBegin(1)


def random_times(rng, m, n, hour_p):
    start, end = month_bounds(m)
    days = (end - start).days
    d = rng.integers(0, days, n)
    h = rng.choice(24, n, p=hour_p)
    s = rng.integers(0, 3600, n)
    return start + pd.to_timedelta(d, "D") + pd.to_timedelta(h, "h") + pd.to_timedelta(s, "s")


def build_merchants(rng, n=700):
    pop = MCC["pop"].to_numpy() / MCC["pop"].sum()
    idx = rng.choice(len(MCC), n, p=pop)
    states = rng.choice(list(STATES), n)
    rows = []
    for i, (k, st) in enumerate(zip(idx, states)):
        mcc = MCC.iloc[k]
        tier = mcc.tier
        if rng.random() < 0.10:  # imperfect risk tiering
            tier = TIERS[int(np.clip(TIERS.index(tier) + rng.choice([-1, 1]), 0, 2))]
        rows.append(dict(
            merchant_id=f"M2{i:05d}", merchant_state=st,
            merchant_city=rng.choice(STATES[st]), merchant_type=tier,
            mcc_code=int(mcc.mcc_code), mcc_title=mcc.mcc_title,
            typ_amount=float(mcc.typ_amount),
            popularity=float(rng.lognormal(0, 1)) * (3 if tier == "low" else 1),
            # most merchants exist from the start; the rest onboard steadily through the year
            onboard_month=1 if rng.random() < 0.8 else int(rng.integers(2, 10)),
        ))
    return pd.DataFrame(rows)


def build_new_digital_merchants(rng, start_idx, n=15):
    """Newly onboarded merchants used by fraud Pattern B (mis-tiered)."""
    rows = []
    for i in range(n):
        st = rng.choice(list(STATES))
        mcc = 5816 if i % 3 else 4829
        title = MCC.set_index("mcc_code").loc[mcc, "mcc_title"]
        rows.append(dict(
            merchant_id=f"M2{start_idx + i:05d}", merchant_state=st,
            merchant_city=rng.choice(STATES[st]),
            merchant_type="low" if i % 2 else "medium",
            mcc_code=mcc, mcc_title=title, typ_amount=900.0,
            popularity=0.5, onboard_month=8,
        ))
    return pd.DataFrame(rows)


def build_devices(rng, merchants, n=9000):
    base = np.array([0.50, 0.15, 0.12, 0.23])
    devs = []
    by_state = merchants.groupby("merchant_state").merchant_id.apply(list).to_dict()
    all_m = merchants.merchant_id.tolist()
    mpop = merchants.popularity.to_numpy() / merchants.popularity.sum()
    for i in range(n):
        home = rng.choice(list(STATES))
        local = by_state.get(home, all_m)
        favs = [rng.choice(local) if rng.random() < 0.7 else rng.choice(all_m, p=mpop)
                for _ in range(rng.integers(2, 7))]
        devs.append(dict(
            device_id=f"DEV2{i:05d}", bank=rng.choice(BANKS),
            pref=rng.dirichlet(base * 8), favs=favs,
            activity=float(rng.lognormal(0, 0.9)),
            # customers keep arriving (new phones / new users) every month
            start_month=1 if rng.random() < 0.55 else int(rng.integers(2, 10)),
        ))
    return devs


def status_draw(rng, n, p_success, p_failed):
    u = rng.random(n)
    return np.where(u < p_success, "SUCCESS", np.where(u < p_success + p_failed, "FAILED", "DECLINED"))


def genuine_month(rng, m, devices, merchants):
    n = int(GENUINE_PER_MONTH * (1 + 0.03 * (m - 1)))
    act = np.array([d["activity"] if d["start_month"] <= m else 0.0 for d in devices]); act /= act.sum()
    m_live = merchants[merchants.onboard_month <= m].reset_index(drop=True)
    mpop = m_live.popularity.to_numpy() / m_live.popularity.sum()
    mlook = merchants.set_index("merchant_id")
    dev_idx = rng.choice(len(devices), n, p=act)
    times = random_times(rng, m, n, GEN_HOUR_P)
    infl = AMOUNT_INFLATION.get(m, 1.0)
    shift = UPI_SHIFT.get(m, 0.0)
    rows = []
    for di, t in zip(dev_idx, times):
        d = devices[di]
        mid = rng.choice(d["favs"]) if rng.random() < 0.85 else m_live.merchant_id.iloc[rng.choice(len(m_live), p=mpop)]
        mr = mlook.loc[mid]
        pref = d["pref"] * (1 - shift) + np.array([1, 0, 0, 0]) * shift
        svc = rng.choice(SERVICES, p=pref / pref.sum())
        amt = float(np.round(rng.lognormal(np.log(mr.typ_amount), 0.75) * infl, 2))
        if rng.random() < 0.01:  # occasional genuine big-ticket purchase
            amt = float(np.round(amt * rng.uniform(5, 15), 0))
        r = rng.random()  # people also pay round sums (rent, transfers, top-ups)
        rnd = 0.10 if svc in ("imps", "netbanking", "wallet") else 0.03
        if r < rnd:
            amt = float(max(np.round(amt * rng.uniform(1, 4), -3), 1000))
        elif r < rnd + 0.25:
            amt = float(max(np.round(amt, -2), 100))
        k = 1 + (rng.random() < 0.12) * rng.integers(1, 3)  # short genuine sessions
        tt = t
        for j in range(k):
            st = status_draw(rng, 1, 0.93, 0.045)[0]
            rows.append((tt, svc, d["device_id"], mid, amt, st, d["bank"], 0))
            if st != "SUCCESS" and rng.random() < 0.5:  # genuine retry
                tt = tt + pd.Timedelta(seconds=int(rng.integers(30, 180)))
                rows.append((tt, svc, d["device_id"], mid, amt, "SUCCESS", d["bank"], 0))
            tt = tt + pd.Timedelta(seconds=int(rng.integers(30, 900)))
            amt = float(np.round(amt * rng.uniform(0.85, 1.15), 2))
    return rows


def fraud_pattern_a(rng, m, devices, merchants, n_attacks):
    m_live = merchants[merchants.onboard_month <= m]
    w = m_live.merchant_type.map({"low": 1, "medium": 3, "high": 12}).to_numpy().astype(float)
    w *= np.where(m_live.mcc_code.isin([6540, 4829, 7995, 7273, 5944, 5732]), 3, 1)
    w /= w.sum()
    hour_p = 0.55 * np.r_[np.ones(6), np.zeros(18)] / 6 + 0.45 * np.ones(24) / 24
    times = random_times(rng, m, n_attacks, hour_p)
    rows = []
    for a, t in enumerate(times):
        if rng.random() < 0.7:  # brand-new device
            dev = f"DEVF{m:02d}{a:04d}"
            bank = rng.choice(BANKS)
            svc = rng.choice(SERVICES, p=[0.30, 0.35, 0.10, 0.25])
        else:  # compromised existing device
            d = devices[rng.integers(len(devices))]
            dev, bank = d["device_id"], d["bank"]
            svc = rng.choice(SERVICES, p=d["pref"])
        mid = m_live.merchant_id.iloc[rng.choice(len(m_live), p=w)]
        burst = int(min(1 + rng.poisson(1.8), 8))
        if rng.random() < 0.35:   # lower-value fraud that blends in with normal spend
            base = float(np.clip(rng.lognormal(np.log(4000), 0.7), 200, 200000))
        else:
            base = float(np.clip(rng.lognormal(np.log(30000), 0.7), 800, 200000))
        tt = t
        if rng.random() < 0.2:  # card-testing: tiny probe before the real attempts
            rows.append((tt, svc, dev, mid, float(rng.integers(1, 300)), "SUCCESS", bank, 1))
            tt = tt + pd.Timedelta(seconds=int(rng.integers(20, 300)))
        for j in range(burst):
            amt = base * rng.uniform(0.8, 1.3)
            r = rng.random()
            amt = np.round(amt, -3) if r < 0.20 else (np.round(amt, -3) - 1 if r < 0.26 else np.round(amt, 2))
            last = j == burst - 1
            st = status_draw(rng, 1, 0.75 if last else 0.40, 0.15 if last else 0.35)[0]
            rows.append((tt, svc, dev, mid, float(max(amt, 100)), st, bank, 1))
            tt = tt + pd.Timedelta(seconds=int(rng.integers(15, 300)))
    return rows


def fraud_pattern_b(rng, m, devices, new_merchants, n_attacks):
    times = random_times(rng, m, n_attacks, GEN_HOUR_P)
    rows = []
    for t in times:
        d = devices[rng.integers(len(devices))]
        mid = new_merchants.merchant_id.iloc[rng.integers(len(new_merchants))]
        for j in range(1 + int(rng.random() < 0.3)):
            amt = float(np.round(rng.lognormal(np.log(8000), 0.4), 2))
            st = status_draw(rng, 1, 0.9, 0.07)[0]
            rows.append((t + pd.Timedelta(seconds=int(j * rng.integers(60, 600))), "upi",
                         d["device_id"], mid, amt, st, d["bank"], 1))
    return rows


def generate(out: Path, sample: Path, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    merchants = build_merchants(rng)
    new_m = build_new_digital_merchants(rng, start_idx=len(merchants))
    merchants = pd.concat([merchants, new_m], ignore_index=True)
    devices = build_devices(rng, merchants[merchants.onboard_month == 1])
    live_devices = lambda m: [d for d in devices if d["start_month"] <= m]

    rows = []
    for m in MONTHS:
        g = genuine_month(rng, m, devices, merchants)
        rows += g
        rows += fraud_pattern_a(rng, m, live_devices(m), merchants, n_attacks=int(len(g) * 0.0042))
        if m >= 8:
            # legit customers also start using the new merchants
            rows += [(t, "upi", live_devices(m)[rng.integers(len(live_devices(m)))]["device_id"],
                      new_m.merchant_id.iloc[rng.integers(len(new_m))],
                      float(np.round(rng.lognormal(np.log(900), 0.8), 2)), "SUCCESS", "SBI", 0)
                     for t in random_times(rng, m, 600 if m == 8 else 900, GEN_HOUR_P)]
            rows += fraud_pattern_b(rng, m, live_devices(m), new_m, n_attacks=90 if m == 8 else 200)

    df = pd.DataFrame(rows, columns=["request_time", "service_type", "device_id", "merchant_id",
                                     "amount", "request_status", "bank", "fraud_label"])
    df = df.merge(merchants.drop(columns=["typ_amount", "popularity", "onboard_month"]),
                  on="merchant_id", how="left")
    df["issuer_bank"] = np.where(df.service_type == "wallet", "NOT_APPLICABLE", df.bank)
    df["currency_code"] = "INR"
    df = df[df.request_time < pd.Timestamp(2026, 10, 1)]

    # realistic missingness
    n = len(df)
    df.loc[rng.random(n) < 0.010, "merchant_city"] = np.nan
    df.loc[rng.random(n) < 0.005, "device_id"] = np.nan
    df.loc[rng.random(n) < 0.004, "issuer_bank"] = np.nan

    # data-quality incident: wallet amounts in paise from 15-Sep
    bug = (df.request_time >= "2026-09-15") & (df.service_type == "wallet")
    df.loc[bug, "amount"] = np.round(df.loc[bug, "amount"] * 100, 2)

    df = df.sort_values("request_time", kind="stable").reset_index(drop=True)
    df["request_id"] = [f"TXN{200001 + i}" for i in range(len(df))]
    df["request_time"] = df.request_time.dt.strftime("%Y-%m-%d %H:%M:%S")

    cols = ["request_id", "request_time", "service_type", "device_id", "merchant_id",
            "merchant_state", "merchant_city", "merchant_type", "mcc_code", "mcc_title",
            "issuer_bank", "currency_code", "amount", "request_status", "fraud_label"]
    orig = pd.read_csv(sample)
    full = pd.concat([orig[cols], df[cols]], ignore_index=True)
    full["_t"] = pd.to_datetime(full.request_time)
    full = full.sort_values("_t", kind="stable").drop(columns="_t").reset_index(drop=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(out, index=False)
    return full


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/transactions_synthetic.csv")
    ap.add_argument("--sample", default="data/fraud_detection_sample_transactions.csv")
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()
    d = generate(Path(a.out), Path(a.sample), a.seed)
    t = pd.to_datetime(d.request_time)
    print(f"wrote {len(d):,} rows -> {a.out}")
    print(d.groupby(t.dt.month).fraud_label.agg(["size", "sum", "mean"]).round(4))
