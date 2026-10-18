"""Delta-hedged version of the real-quote option strategies: is the profit volatility premium or direction?

Fixed before running (same entries, strikes and bid/ask fills as real_options_backtest.py):
  Structures : short call spread (1 sd / 2 sd), short put (1 sd), short strangle (1 sd), short straddle,
               iron condor (1 sd / 2 sd), long straddle. Entry every day at the ~30-day monthly expiry.
  Hedge      : at each daily close from entry to the day before expiry, hold h = -(position delta) shares.
               Delta = Black-Scholes delta of each leg using that day's ATM IV for that expiry
               (carried forward if missing). Hedge P&L = sum of h_t * (S_{t+1} - S_t).
  Hedge cost : 5 bps of the notional of each share trade (small-cap stock spreads).
  P&L        : entry fill (sell at bid / buy at ask, $0.65/contract) - payoff at expiry + hedge P&L.
               Same capital denominators as before. Unhedged P&L is reported on the same entries.
  Filters    : all | after spike | IV rich (ATM IV / RV20 >= 1.3). Small/mid caps vs GME+NVDA separately.
  Stats      : week-bootstrap t-stats.
If a strategy makes money unhedged but not hedged, its profit was the stocks' direction, not overpricing.

python scripts/hedged_backtest.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import retail_spike_study as S1  # noqa: E402
from real_options_backtest import COMM, LARGE, R, iv  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
HEDGE_BPS = 0.0005


def delta(S, K, T, sig, call):
    if T <= 0:
        return float(S > K) if call else -float(S < K)
    d1 = (np.log(S / K) + (R + 0.5 * sig * sig) * T) / (sig * np.sqrt(T))
    return norm.cdf(d1) if call else norm.cdf(d1) - 1


def run_ticker(t: str, spikes: set) -> list[dict]:
    op = pd.read_csv(ROOT / "data" / "options" / f"{t}.csv.gz", parse_dates=["date", "expiration"])
    op = op[op.ask > 0]
    op["mid"] = (op.bid + op.ask) / 2
    px = pd.read_csv(ROOT / "data" / "prices" / f"{t}.csv", parse_dates=["date"]).set_index("date")
    close = px.close
    r = np.log(close).diff()
    rv20 = r.rolling(20, min_periods=15).std() * np.sqrt(252)
    book = {k: g for k, g in op.groupby(["date", "expiration"])}
    last_px = close.index.max()

    # ATM IV per (day, expiry): used for every leg's delta
    atm = {}
    for (d, e), ch in book.items():
        if d not in close.index or e <= d:
            continue
        S, T = close.loc[d], (e - d).days / 365
        c = ch[ch.right == "C"].set_index("strike")
        p = ch[ch.right == "P"].set_index("strike")
        com = c.index.intersection(p.index)
        if len(com) == 0:
            continue
        k = com[np.argmin(np.abs(com - S))]
        atm[(d, e)] = np.nanmean([iv(c.loc[k, "mid"], S, k, T, True), iv(p.loc[k, "mid"], S, k, T, False)])

    out = []
    days = close.index
    for d, exps in op.groupby("date").expiration.unique().items():
        if d not in close.index:
            continue
        dte = np.array([(e - d).days for e in exps])
        ok = (dte >= 20) & (dte <= 50)
        if not ok.any():
            continue
        e = exps[ok][np.argmin(np.abs(dte[ok] - 30))]
        sig = atm.get((d, e), np.nan)
        if e > last_px or not np.isfinite(sig):
            continue
        S0 = close.loc[d]
        T0 = (e - d).days / 365
        ch = book[(d, e)]
        calls = ch[ch.right == "C"].set_index("strike").sort_index()
        puts = ch[ch.right == "P"].set_index("strike").sort_index()
        com = calls.index.intersection(puts.index)
        katm = com[np.argmin(np.abs(com - S0))]
        near = lambda b, x: b.index[np.argmin(np.abs(b.index - x))]  # noqa: E731
        k1u, k2u = near(calls, S0 * np.exp(sig * np.sqrt(T0))), near(calls, S0 * np.exp(2 * sig * np.sqrt(T0)))
        k1d, k2d = near(puts, S0 * np.exp(-sig * np.sqrt(T0))), near(puts, S0 * np.exp(-2 * sig * np.sqrt(T0)))

        def fill(right, K, sell):
            b = calls if right == "C" else puts
            x = b.loc[K]
            bid, ask, mid = float(x.bid), float(x.ask), float(x.mid)
            if ask <= 0 or mid <= 0 or (ask - bid) / mid > 0.5 or (sell and bid <= 0):
                return None
            return bid - COMM if sell else -(ask + COMM)      # cash received (+) / paid (-)

        # structures: list of (right, strike, qty) with qty -1 = short; plus capital denominator
        cand = {
            "short_call_spread": ([("C", k1u, -1), ("C", k2u, +1)], k2u > k1u > katm),
            "short_put": ([("P", k1d, -1)], k1d < katm),
            "short_strangle": ([("C", k1u, -1), ("P", k1d, -1)], k1u > katm > k1d),
            "short_straddle": ([("C", katm, -1), ("P", katm, -1)], True),
            "iron_condor": ([("C", k1u, -1), ("C", k2u, +1), ("P", k1d, -1), ("P", k2d, +1)],
                            k2u > k1u > katm > k1d > k2d),
            "long_straddle": ([("C", katm, +1), ("P", katm, +1)], True),
        }
        path = days[(days >= d) & (days <= e)]
        ST = close.loc[path[-1]]
        spike = any((t, dd) in spikes for dd in days[days <= d][-3:])
        rich = sig / rv20.get(d, np.nan) >= 1.3
        for name, (legs, valid) in cand.items():
            if not valid:
                continue
            cash = [fill(rt, K, q < 0) for rt, K, q in legs]
            if any(c is None for c in cash):
                continue
            credit = sum(cash)
            payoff = sum(q * (max(ST - K, 0) if rt == "C" else max(K - ST, 0)) for rt, K, q in legs)
            pnl_unh = credit + payoff
            # daily delta hedge
            hedge, h_prev, s_last = 0.0, 0.0, None
            for i, day in enumerate(path[:-1]):
                S = close.loc[day]
                T = (e - day).days / 365
                sg = atm.get((day, e), np.nan)
                sig_t = sg if np.isfinite(sg) else (sig if s_last is None else s_last)
                s_last = sig_t
                pos_delta = sum(q * delta(S, K, T, sig_t, rt == "C") for rt, K, q in legs)
                h = -pos_delta
                hedge -= abs(h - h_prev) * S * HEDGE_BPS
                hedge += h * (close.loc[path[i + 1]] - S)
                h_prev = h
            hedge -= abs(h_prev) * close.loc[path[-1]] * HEDGE_BPS    # unwind
            if name == "short_call_spread":
                cap = (k2u - k1u) - credit
            elif name == "iron_condor":
                cap = max(k2u - k1u, k1d - k2d) - credit
            elif name == "short_put":
                cap = k1d
            elif name == "long_straddle":
                cap = -credit
            else:
                cap = 0.2 * S0
            if cap <= 0:
                continue
            out.append(dict(date=d, ticker=t, strategy=name, spike=spike, rich=bool(rich),
                            unhedged=pnl_unh / cap, hedged=(pnl_unh + hedge) / cap, hedge_pnl=hedge / cap,
                            stock=ST / S0 - 1))
    return out


def main():
    px, spy = S1.load()
    ev = pd.concat([S1.events_for(t, d, spy) for t, d in px.items()])
    spikes = set(zip(ev[ev.spike].ticker, ev[ev.spike].index))
    tick = sorted(p.name.split(".")[0] for p in (ROOT / "data" / "options").glob("*.csv.gz")
                  if (ROOT / "data" / "prices" / f"{p.name.split('.')[0]}.csv").exists())
    rows = []
    for t in tick:
        rows += run_ticker(t, spikes)
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "hedged_trades.csv.gz", index=False, compression="gzip")
    tab = []
    for gname, g in [("small/mid", df[~df.ticker.isin(LARGE)]), ("GME+NVDA", df[df.ticker.isin(LARGE)])]:
        for fname, gf in [("all", g), ("after spike", g[g.spike]), ("IV rich", g[g.rich])]:
            for s, gs in gf.groupby("strategy"):
                if len(gs) < 15:
                    continue
                x = gs.set_index("date")
                mu, tu, n = S1.week_boot(x[["unhedged"]], "unhedged")
                mh, th, _ = S1.week_boot(x[["hedged"]], "hedged")
                tab.append(dict(group=gname, filter=fname, strategy=s, n=n, tickers=gs.ticker.nunique(),
                                unhedged_pct=round(100 * mu, 2), t_unh=round(tu, 2),
                                hedged_pct=round(100 * mh, 2), t_hedged=round(th, 2),
                                hedged_win=round(100 * (gs.hedged > 0).mean()),
                                hedged_worst=round(100 * gs.hedged.min()), stock_pct=round(100 * gs.stock.mean(), 1)))
    tab = pd.DataFrame(tab)
    tab.to_csv(ROOT / "results" / "hedged_summary.csv", index=False)
    pd.set_option("display.width", 220)
    print(tab.to_string(index=False))
    sm = df[~df.ticker.isin(LARGE)]
    per = sm.pivot_table(index="ticker", columns="strategy", values="hedged", aggfunc="mean").mul(100).round(1)
    print("\nBY TICKER, hedged mean % per trade:\n" + per.to_string())
    per.to_csv(ROOT / "results" / "hedged_by_ticker.csv")
    print(f"\ntrials: {len(tab)}")


if __name__ == "__main__":
    main()
