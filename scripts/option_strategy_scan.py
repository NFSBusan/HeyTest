"""Scan standard option strategies on the tickers with real IV history. Approximate pricing.

Fixed before running (no tuning; every row printed is a trial):
  Tickers   : GME, TSLA, NVDA, MU, PLTR (+ SPY as benchmark). Mid-tier names have no IV history.
  Entry     : every trading day with a DoltHub iv_current reading; hold 21 trading days to expiry.
  Pricing   : Black-Scholes, flat vol = iv_current (30d), r = 4%, no dividends. NO SKEW: real OTM puts
              (and meme-stock OTM calls) cost more than this, so OTM legs are mispriced. Treat as a screen.
  Strikes   : ATM = spot; "1 sd" = S*exp(+-iv*sqrt(T)); "2 sd" wings = S*exp(+-2*iv*sqrt(T)).
  Costs     : entry only (held to expiry, cash-settled at intrinsic): per leg max(4% of premium, $0.02)
              per share + $0.65/contract commission. Early assignment ignored.
  Return on : long premium -> premium paid; defined risk -> max loss; short put -> strike (cash-secured);
              naked short straddle/strangle -> 20% of spot (rough Reg-T margin); covered call -> spot.
  Filters   : all | IV rank >= 0.7 | IV rank <= 0.3 (rank vs own trailing 252 obs, point-in-time)
              | after spike (3x volume spike on the day or the 2 days before).
  Stats     : mean return per trade; win rate; worst trade; Sharpe from non-overlapping 21-day blocks
              (annualized x sqrt 12); t-stat from a bootstrap over calendar months of entry.
  Periods   : full sample, and 2021-24 vs 2025-26 to check stability.

python scripts/option_strategy_scan.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import retail_spike_study as S1  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TICKERS = ["GME", "TSLA", "NVDA", "MU", "PLTR", "SPY"]
N, R = 21, 0.04
T = N / 252


def bs(S, K, sig, call):
    d1 = (np.log(S / K) + (R + 0.5 * sig ** 2) * T) / (sig * np.sqrt(T))
    d2 = d1 - sig * np.sqrt(T)
    if call:
        return S * norm.cdf(d1) - K * np.exp(-R * T) * norm.cdf(d2)
    return K * np.exp(-R * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def cost(prem):
    return np.maximum(0.04 * prem, 0.02) + 0.0065


def trades(t: str, spikes: set) -> pd.DataFrame:
    px = pd.read_csv(ROOT / "data" / "prices" / f"{t}.csv", parse_dates=["date"]).set_index("date")
    iv = pd.read_csv(ROOT / "data" / "vol" / f"{t}.csv", parse_dates=["date"]).set_index("date")["iv_current"]
    d = px[["close"]].join(iv, how="left")
    c, v, idx = d.close.to_numpy(), d.iv_current.to_numpy(), d.index
    # IV rank vs trailing 252 IV observations (point-in-time, includes today)
    ivs = d.iv_current.dropna()
    lo, hi = ivs.rolling(252, min_periods=100).min(), ivs.rolling(252, min_periods=100).max()
    rank = ((ivs - lo) / (hi - lo)).reindex(idx)
    rows = []
    for i in range(len(d) - N):
        sig = v[i]
        if not np.isfinite(sig) or sig <= 0:
            continue
        S, ST = c[i], c[i + N]
        k1u, k1d = S * np.exp(sig * np.sqrt(T)), S * np.exp(-sig * np.sqrt(T))
        k2u, k2d = S * np.exp(2 * sig * np.sqrt(T)), S * np.exp(-2 * sig * np.sqrt(T))
        C, P = (lambda K: bs(S, K, sig, True)), (lambda K: bs(S, K, sig, False))
        cpay, ppay = (lambda K: max(ST - K, 0.0)), (lambda K: max(K - ST, 0.0))
        r = {}
        # long ATM straddle (pay premium + costs)
        prem = C(S) + P(S)
        r["long_straddle"] = (cpay(S) + ppay(S) - prem - cost(C(S)) - cost(P(S))) / (prem + cost(C(S)) + cost(P(S)))
        # short ATM straddle, naked, on 20% margin
        r["short_straddle"] = (prem - cost(C(S)) - cost(P(S)) - cpay(S) - ppay(S)) / (0.2 * S)
        # short 1sd strangle, naked
        cr = C(k1u) + P(k1d) - cost(C(k1u)) - cost(P(k1d))
        r["short_strangle"] = (cr - cpay(k1u) - ppay(k1d)) / (0.2 * S)
        # iron condor: short 1sd strangle, long 2sd wings; return on max loss
        cr_ic = cr - C(k2u) - P(k2d) - cost(C(k2u)) - cost(P(k2d))
        width = max(k2u - k1u, k1d - k2d)
        pnl = cr_ic - cpay(k1u) + cpay(k2u) - ppay(k1d) + ppay(k2d)
        r["iron_condor"] = pnl / max(width - cr_ic, 1e-9)
        # cash-secured short 1sd put
        r["short_put"] = (P(k1d) - cost(P(k1d)) - ppay(k1d)) / k1d
        # covered call (long stock, short 1sd call) vs spot; and plain stock for reference
        r["covered_call"] = (ST - S + C(k1u) - cost(C(k1u)) - cpay(k1u)) / S
        r["stock"] = ST / S - 1
        # short call spread (1sd / 2sd): bet that retail upside demand is overpriced
        cr_cs = C(k1u) - C(k2u) - cost(C(k1u)) - cost(C(k2u))
        r["short_call_spread"] = (cr_cs - cpay(k1u) + cpay(k2u)) / max((k2u - k1u) - cr_cs, 1e-9)
        # long call spread (ATM / 1sd): plain bullish bet
        db = C(S) - C(k1u) + cost(C(S)) + cost(C(k1u))
        r["long_call_spread"] = (cpay(S) - cpay(k1u) - db) / db
        # long put spread (ATM / 1sd): plain bearish bet
        db = P(S) - P(k1d) + cost(P(S)) + cost(P(k1d))
        r["long_put_spread"] = (ppay(S) - ppay(k1d) - db) / db
        rows.append(dict(date=idx[i], ticker=t, ivr=rank.iloc[i],
                         spike=any((t, idx[max(i - k, 0)]) in spikes for k in range(3)), **r))
    return pd.DataFrame(rows).set_index("date")


def stats(g: pd.DataFrame, col: str) -> dict:
    x = g[col].dropna()
    if len(x) < 10:
        return {}
    mon = x.groupby(x.index.to_period("M")).agg(["sum", "count"])
    s, c = mon["sum"].to_numpy(), mon["count"].to_numpy()
    rng = np.random.default_rng(0)
    b = rng.integers(0, len(s), (2000, len(s)))
    boots = s[b].sum(1) / c[b].sum(1)
    blocks = []
    for _, gt in g.groupby("ticker"):   # non-overlapping 21-trading-day entries per ticker
        xt = gt[col].dropna()
        blocks.append(xt.iloc[::N] if len(xt) else xt)
    nb = pd.concat(blocks)
    sh = nb.mean() / nb.std(ddof=1) * np.sqrt(12) if len(nb) > 3 and nb.std() > 0 else np.nan
    return dict(n=len(x), mean_pct=round(100 * x.mean(), 2), t=round(x.mean() / boots.std(ddof=1), 2),
                win_pct=round(100 * (x > 0).mean()), worst_pct=round(100 * x.min()), sharpe=round(sh, 2))


STRATS = ["long_straddle", "short_straddle", "short_strangle", "iron_condor", "short_put",
          "covered_call", "short_call_spread", "long_call_spread", "long_put_spread", "stock"]


def main():
    px, spy = S1.load()
    ev = pd.concat([S1.events_for(t, d, spy) for t, d in px.items()])
    spikes = set(zip(ev[ev.spike].ticker, ev[ev.spike].index))
    df = pd.concat([trades(t, spikes) for t in TICKERS]).sort_index()
    stocks = df[df.ticker != "SPY"]
    filters = {"all": stocks, "ivr>=0.7": stocks[stocks.ivr >= 0.7], "ivr<=0.3": stocks[stocks.ivr <= 0.3],
               "after spike": stocks[stocks.spike]}
    rows = []
    for fname, g in filters.items():
        for s in STRATS:
            st = stats(g, s)
            if st:
                e = g[g.index <= S1.DEV_END][s].mean()
                l = g[g.index > S1.DEV_END][s].mean()
                rows.append(dict(filter=fname, strategy=s, **st,
                                 early_pct=round(100 * e, 2), late_pct=round(100 * l, 2)))
    tab = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("POOLED: GME, TSLA, NVDA, MU, PLTR  (return per trade on the capital defined in the docstring)")
    print("early = 2021-24 mean, late = 2025-26 mean\n")
    print(tab.to_string(index=False))
    tab.to_csv(ROOT / "results" / "option_scan_pooled.csv", index=False)

    per = []
    for t, g in df.groupby("ticker"):
        for s in STRATS:
            st = stats(g, s)
            if st:
                per.append(dict(ticker=t, strategy=s, **{k: st[k] for k in ["n", "mean_pct", "t", "win_pct", "worst_pct", "sharpe"]}))
    per = pd.DataFrame(per)
    print("\nBY TICKER (all entries, no filter); Sharpe from non-overlapping monthly entries")
    print(per.pivot(index="strategy", columns="ticker", values="sharpe").round(2).to_string())
    print("\nmean % per trade")
    print(per.pivot(index="strategy", columns="ticker", values="mean_pct").to_string())
    per.to_csv(ROOT / "results" / "option_scan_by_ticker.csv", index=False)
    print(f"\ntrials in this scan: {len(tab) + len(per)}")


if __name__ == "__main__":
    main()
