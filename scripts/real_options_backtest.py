"""Backtest option strategies on REAL end-of-day option quotes (ThetaData) for the retail-heavy names.

Fixed before running (one pass, no tuning; every printed row counts as a trial):
  Data      : data/options/<T>.csv.gz (EOD bid/ask per contract, monthly expirations, ~Oct 2025 - Oct 2026)
              + data/prices/<T>.csv (IBKR daily close/volume) for spot, settlement and spikes.
  Entry     : every trading day; use the monthly expiration with days-to-expiry closest to 30 within [20, 50].
              Spot S = that day's IBKR close.
  ATM vol   : sigma = average Black-Scholes IV of the ATM call and put mids (r = 4%, no dividends).
  Strikes   : listed strike nearest to S*exp(+-k*sigma*sqrt(T)), k = 1 ("1 sd") and 2 ("2 sd" wings).
  Fills     : SELL at the bid, BUY at the ask (conservative), plus $0.65/contract. A leg is untradeable
              (trade skipped) if bid <= 0 on a sold leg, ask <= 0, or (ask-bid)/mid > 50%.
              A "mid" variant (fill at mid) is reported too, as the optimistic bound.
  Exit      : hold to expiry; cash-settle at intrinsic using the IBKR close on the expiration date.
  Return on : same capital rules as option_strategy_scan.py (premium; max loss; strike; 20% margin; spot).
  Filters   : all | after spike (3x volume spike on the entry day or the 2 days before; Study 1 definition)
              | IV rich (sigma / trailing-20-day realized vol >= 1.3).
  Also      : skew (IV at the 1 sd put and call vs ATM), and VRP = ln(sigma) - ln(realized vol over the
              next 21 trading days), spike vs non-spike.
  Stats     : t-stats from a bootstrap over calendar WEEKS of entry (overlapping trades cluster in time).
  Groups    : small/mid caps = every ticker except GME and NVDA, which are reported separately.

python scripts/real_options_backtest.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import retail_spike_study as S1  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
R = 0.04
COMM = 0.0065
LARGE = {"GME", "NVDA"}


def bs(S, K, T, sig, call):
    d1 = (np.log(S / K) + (R + 0.5 * sig * sig) * T) / (sig * np.sqrt(T))
    d2 = d1 - sig * np.sqrt(T)
    if call:
        return S * norm.cdf(d1) - K * np.exp(-R * T) * norm.cdf(d2)
    return K * np.exp(-R * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def iv(price, S, K, T, call):
    intr = max(S - K * np.exp(-R * T), 0) if call else max(K * np.exp(-R * T) - S, 0)
    if not np.isfinite(price) or price <= intr + 1e-6:
        return np.nan
    try:
        return brentq(lambda s: bs(S, K, T, s, call) - price, 0.01, 8.0)
    except ValueError:
        return np.nan


def ticker_trades(t: str, spikes: set) -> pd.DataFrame:
    op = pd.read_csv(ROOT / "data" / "options" / f"{t}.csv.gz", parse_dates=["date", "expiration"])
    op = op[(op.ask > 0)]
    op["mid"] = (op.bid + op.ask) / 2
    px = pd.read_csv(ROOT / "data" / "prices" / f"{t}.csv", parse_dates=["date"]).set_index("date")
    close = px.close
    r = np.log(close).diff()
    rv20 = r.rolling(20, min_periods=15).std() * np.sqrt(252)
    rv_fwd = r[::-1].rolling(21, min_periods=21).std()[::-1].shift(-1) * np.sqrt(252)
    last_px = close.index.max()
    book = {k: g for k, g in op.groupby(["date", "expiration"])}
    exps_by_day = op.groupby("date").expiration.unique()
    rows = []
    for d, exps in exps_by_day.items():
        if d not in close.index:
            continue
        dte = np.array([(e - d).days for e in exps])
        ok = (dte >= 20) & (dte <= 50)
        if not ok.any():
            continue
        e = exps[ok][np.argmin(np.abs(dte[ok] - 30))]
        if e > last_px:
            continue
        S = close.loc[d]
        T = (e - d).days / 365
        ch = book[(d, e)]
        calls = ch[ch.right == "C"].set_index("strike").sort_index()
        puts = ch[ch.right == "P"].set_index("strike").sort_index()
        common = calls.index.intersection(puts.index)
        if len(common) < 3:
            continue
        katm = common[np.argmin(np.abs(common - S))]
        sig = np.nanmean([iv(calls.loc[katm, "mid"], S, katm, T, True), iv(puts.loc[katm, "mid"], S, katm, T, False)])
        if not np.isfinite(sig):
            continue
        ST = close.loc[:e].iloc[-1]

        def near(book_, target):
            return book_.index[np.argmin(np.abs(book_.index - target))]

        k1u, k2u = near(calls, S * np.exp(sig * np.sqrt(T))), near(calls, S * np.exp(2 * sig * np.sqrt(T)))
        k1d, k2d = near(puts, S * np.exp(-sig * np.sqrt(T))), near(puts, S * np.exp(-2 * sig * np.sqrt(T)))

        def q(right, K):
            b = calls if right == "C" else puts
            x = b.loc[K]
            return float(x.bid), float(x.ask), float(x.mid)

        def legal(bid, ask, mid, sell):
            return ask > 0 and (bid > 0 or not sell) and mid > 0 and (ask - bid) / mid <= 0.5

        def fill(right, K, sell, mode):
            bid, ask, mid = q(right, K)
            if not legal(bid, ask, mid, sell):
                return None
            p = mid if mode == "mid" else (bid if sell else ask)
            return p - COMM if sell else p + COMM

        cp = lambda K: max(ST - K, 0.0)  # noqa: E731
        pp = lambda K: max(K - ST, 0.0)  # noqa: E731
        base = dict(date=d, ticker=t, exp=e, S=S, sig=sig, rv20=rv20.get(d, np.nan), rv_fwd=rv_fwd.get(d, np.nan),
                    spike=any((t, dd) in spikes for dd in close.loc[:d].index[-3:]),
                    iv_put1=iv(puts.loc[k1d, "mid"], S, k1d, T, False),
                    iv_call1=iv(calls.loc[k1u, "mid"], S, k1u, T, True))
        for mode in ["ask_bid", "mid"]:
            f = lambda right, K, sell: fill(right, K, sell, mode)  # noqa: E731
            out = {}
            # long ATM straddle
            c, p = f("C", katm, False), f("P", katm, False)
            if c and p:
                out["long_straddle"] = (cp(katm) + pp(katm) - c - p) / (c + p)
            c, p = f("C", katm, True), f("P", katm, True)
            if c and p:
                out["short_straddle"] = (c + p - cp(katm) - pp(katm)) / (0.2 * S)
            c, p = f("C", k1u, True), f("P", k1d, True)
            if c and p and k1u > katm and k1d < katm:
                out["short_strangle"] = (c + p - cp(k1u) - pp(k1d)) / (0.2 * S)
                wc, wp = f("C", k2u, False), f("P", k2d, False)
                if wc and wp and k2u > k1u and k2d < k1d:
                    cr = c + p - wc - wp
                    width = max(k2u - k1u, k1d - k2d)
                    if width - cr > 0:
                        out["iron_condor"] = (cr - cp(k1u) + cp(k2u) - pp(k1d) + pp(k2d)) / (width - cr)
            p = f("P", k1d, True)
            if p and k1d < katm:
                out["short_put"] = (p - pp(k1d)) / k1d
                w = f("P", k2d, False)
                if w and k2d < k1d and (k1d - k2d) - (p - w) > 0:
                    out["put_credit_spread"] = (p - w - pp(k1d) + pp(k2d)) / ((k1d - k2d) - (p - w))
            c = f("C", k1u, True)
            if c and k1u > katm:
                out["covered_call"] = (ST - S + c - cp(k1u)) / S
                w = f("C", k2u, False)
                if w and k2u > k1u and (k2u - k1u) - (c - w) > 0:
                    out["short_call_spread"] = (c - w - cp(k1u) + cp(k2u)) / ((k2u - k1u) - (c - w))
            a, b = f("C", katm, False), f("C", k1u, True)
            if a and b and k1u > katm and a - b > 0:
                out["long_call_spread"] = (cp(katm) - cp(k1u) - (a - b)) / (a - b)
            a, b = f("P", katm, False), f("P", k1d, True)
            if a and b and k1d < katm and a - b > 0:
                out["long_put_spread"] = (pp(katm) - pp(k1d) - (a - b)) / (a - b)
            out["stock"] = ST / S - 1
            rows.append(dict(base, mode=mode, **out))
    return pd.DataFrame(rows)


STRATS = ["short_put", "put_credit_spread", "covered_call", "stock", "short_call_spread", "iron_condor",
          "short_strangle", "short_straddle", "long_straddle", "long_call_spread", "long_put_spread"]


def stats(g, col):
    x = g.set_index("date")[col].dropna()
    if len(x) < 15:
        return None
    m, t, n = S1.week_boot(x.to_frame(col), col)
    return dict(n=n, tickers=g.loc[g[col].notna(), "ticker"].nunique(), mean_pct=round(100 * m, 2), t=round(t, 2),
                win_pct=round(100 * (x > 0).mean()), worst_pct=round(100 * x.min()))


def main():
    px, spy = S1.load()
    ev = pd.concat([S1.events_for(t, d, spy) for t, d in px.items()])
    spikes = set(zip(ev[ev.spike].ticker, ev[ev.spike].index))
    tick = sorted(p.name.split(".")[0] for p in (ROOT / "data" / "options").glob("*.csv.gz"))
    missing = [t for t in tick if not (ROOT / "data" / "prices" / f"{t}.csv").exists()]
    if missing:
        print("skipped (no IBKR price file):", missing)
    tick = [t for t in tick if t not in missing]
    df = pd.concat([ticker_trades(t, spikes) for t in tick], ignore_index=True)
    df.to_csv(ROOT / "results" / "real_options_trades.csv.gz", index=False, compression="gzip")
    one = df[df["mode"] == "ask_bid"]
    print(f"{len(tick)} tickers, {one.date.nunique()} entry days, {len(one)} ticker-days, "
          f"{one.date.min().date()} .. {one.date.max().date()}")

    # skew and VRP (mode-independent)
    sk = one.assign(put_skew=one.iv_put1 / one.sig - 1, call_skew=one.iv_call1 / one.sig - 1)
    print("\nSKEW (IV of 1 sd OTM option vs ATM IV, median by ticker):")
    print(sk.groupby("ticker")[["sig", "put_skew", "call_skew"]].median().round(2).T.to_string())
    v = one.dropna(subset=["rv_fwd"]).assign(vrp=lambda x: np.log(x.sig / x.rv_fwd))
    small = v[~v.ticker.isin(LARGE)]
    for nm, g in [("small/mid, all days", small[~small.spike]), ("small/mid, after spike", small[small.spike])]:
        m, t, n = S1.week_boot(g.set_index("date")[["vrp"]], "vrp")
        print(f"VRP {nm}: {m:+.3f} (t {t:+.2f}, n {n})")

    rows = []
    for mode in ["ask_bid", "mid"]:
        dm = df[df["mode"] == mode]
        groups = {"small/mid": dm[~dm.ticker.isin(LARGE)], "GME+NVDA": dm[dm.ticker.isin(LARGE)]}
        for gname, g in groups.items():
            for fname, gf in [("all", g), ("after spike", g[g.spike]), ("IV rich", g[g.sig / g.rv20 >= 1.3])]:
                for s in STRATS:
                    st = stats(gf, s)
                    if st:
                        rows.append(dict(fill=mode, group=gname, filter=fname, strategy=s, **st))
    tab = pd.DataFrame(rows)
    tab.to_csv(ROOT / "results" / "real_options_summary.csv", index=False)
    pd.set_option("display.width", 220)
    for mode in ["ask_bid", "mid"]:
        print(f"\n== fills: {'sell at bid / buy at ask (realistic-to-pessimistic)' if mode == 'ask_bid' else 'at mid (optimistic)'} ==")
        print(tab[tab.fill == mode].drop(columns="fill").to_string(index=False))
    per = []
    sm = df[(df["mode"] == "ask_bid")]
    for t, g in sm.groupby("ticker"):
        for s in ["short_put", "put_credit_spread", "short_call_spread", "iron_condor", "stock"]:
            x = g[s].dropna()
            if len(x) >= 15:
                per.append(dict(ticker=t, strategy=s, n=len(x), mean_pct=round(100 * x.mean(), 2)))
    per = pd.DataFrame(per).pivot(index="ticker", columns="strategy", values="mean_pct")
    print("\nBY TICKER, bid/ask fills, mean % per trade:\n" + per.to_string())
    per.to_csv(ROOT / "results" / "real_options_by_ticker.csv")
    print(f"\ntrials: {len(tab)}")


if __name__ == "__main__":
    main()
