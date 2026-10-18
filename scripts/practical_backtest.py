"""Study 6: the short-volatility trade with realistic margin, practical hedging, and a capped-risk version.

Fixed before running (entries, strikes and bid/ask fills as in Studies 4-5; small/mid caps only):
  Structures
    strangle_hedged_daily  : short 1 sd call + 1 sd put, delta-hedged at every close.
    strangle_hedged_weekly : same, hedge re-set every 5 trading days (practical for a person).
    strangle_unhedged      : no hedge (reference).
    condor_liquid          : short 1 sd call + put; long wings = nearest strike at or beyond 1.5 sd whose
                             quote is tradeable (spread <= 50% of mid). Not hedged. Capped loss.
  Margin (capital the trade ties up)
    Reg-T naked option: call = max(prem + 20%*S - OTM, prem + 10%*S); put = max(prem + 20%*S - OTM,
    prem + 10%*K). Strangle = larger leg requirement + premium of the other leg.
    "High-vol" margin: the same with 40% / 20% (IBKR charges more on volatile small caps).
    Stock hedge margin: 50% of the largest hedge position over the trade's life (Reg-T stock).
    Condor capital = max wing width - credit.
  Filter     : all | IV rich (ATM IV / RV20 >= 1.3)
  Account sim: one entry per ticker per expiry cycle (first eligible day >= 25 days before expiry),
               equal capital per trade, IV-rich filter only. Monthly account return = average trade
               return on capital that month. Reports compounded return and max drawdown.

python scripts/practical_backtest.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import retail_spike_study as S1  # noqa: E402
from hedged_backtest import HEDGE_BPS, delta  # noqa: E402
from real_options_backtest import COMM, LARGE, iv  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def naked_req(prem, S, K, call, pct, floor):
    otm = max(K - S, 0) if call else max(S - K, 0)
    return max(prem + pct * S - otm, prem + floor * (S if call else K))


def run_ticker(t: str) -> list[dict]:
    op = pd.read_csv(ROOT / "data" / "options" / f"{t}.csv.gz", parse_dates=["date", "expiration"])
    op = op[op.ask > 0]
    op["mid"] = (op.bid + op.ask) / 2
    px = pd.read_csv(ROOT / "data" / "prices" / f"{t}.csv", parse_dates=["date"]).set_index("date")
    close = px.close
    rv20 = np.log(close).diff().rolling(20, min_periods=15).std() * np.sqrt(252)
    book = {k: g for k, g in op.groupby(["date", "expiration"])}
    days = close.index
    atm = {}
    for (d, e), ch in book.items():
        if d not in close.index or e <= d:
            continue
        S, T = close.loc[d], (e - d).days / 365
        c, p = ch[ch.right == "C"].set_index("strike"), ch[ch.right == "P"].set_index("strike")
        com = c.index.intersection(p.index)
        if len(com):
            k = com[np.argmin(np.abs(com - S))]
            atm[(d, e)] = np.nanmean([iv(c.loc[k, "mid"], S, k, T, True), iv(p.loc[k, "mid"], S, k, T, False)])
    out = []
    for d, exps in op.groupby("date").expiration.unique().items():
        if d not in close.index:
            continue
        dte = np.array([(e - d).days for e in exps])
        ok = (dte >= 20) & (dte <= 50)
        if not ok.any():
            continue
        e = exps[ok][np.argmin(np.abs(dte[ok] - 30))]
        sig = atm.get((d, e), np.nan)
        if e > days.max() or not np.isfinite(sig):
            continue
        S0, T0 = close.loc[d], (e - d).days / 365
        ch = book[(d, e)]
        calls = ch[ch.right == "C"].set_index("strike").sort_index()
        puts = ch[ch.right == "P"].set_index("strike").sort_index()
        com = calls.index.intersection(puts.index)
        katm = com[np.argmin(np.abs(com - S0))]
        near = lambda b, x: b.index[np.argmin(np.abs(b.index - x))]  # noqa: E731
        k1u, k1d = near(calls, S0 * np.exp(sig * np.sqrt(T0))), near(puts, S0 * np.exp(-sig * np.sqrt(T0)))
        if not (k1u > katm > k1d):
            continue

        def quote(b, K):
            x = b.loc[K]
            bid, ask = float(x.bid), float(x.ask)
            mid = (bid + ask) / 2
            return bid, ask, (mid > 0 and (ask - bid) / mid <= 0.5)

        cb, ca, cok = quote(calls, k1u)
        pb, pa, pok = quote(puts, k1d)
        if not (cok and pok and cb > 0 and pb > 0):
            continue
        cprem, pprem = cb - COMM, pb - COMM
        path = days[(days >= d) & (days <= e)]
        ST = close.loc[path[-1]]
        payoff_short = -(max(ST - k1u, 0) + max(k1d - ST, 0))
        base = dict(date=d, exp=e, ticker=t, rich=bool(sig / rv20.get(d, np.nan) >= 1.3), stock=ST / S0 - 1)
        mreg = max(naked_req(cprem, S0, k1u, True, .2, .1), naked_req(pprem, S0, k1d, False, .2, .1)) + min(cprem, pprem)
        mhv = max(naked_req(cprem, S0, k1u, True, .4, .2), naked_req(pprem, S0, k1d, False, .4, .2)) + min(cprem, pprem)
        legs = [("C", k1u, -1), ("P", k1d, -1)]
        for name, every in [("strangle_hedged_daily", 1), ("strangle_hedged_weekly", 5), ("strangle_unhedged", 0)]:
            hedge, h, hmax, sig_t = 0.0, 0.0, 0.0, sig
            if every:
                for i, day in enumerate(path[:-1]):
                    S = close.loc[day]
                    sg = atm.get((day, e), np.nan)
                    sig_t = sg if np.isfinite(sg) else sig_t
                    if i % every == 0:
                        new = -sum(q * delta(S, K, (e - day).days / 365, sig_t, rt == "C") for rt, K, q in legs)
                        hedge -= abs(new - h) * S * HEDGE_BPS
                        h = new
                        hmax = max(hmax, abs(h) * S)
                    hedge += h * (close.loc[path[i + 1]] - S)
                hedge -= abs(h) * close.loc[path[-1]] * HEDGE_BPS
            pnl = cprem + pprem + payoff_short + hedge
            out.append(dict(base, strategy=name, ret_regt=pnl / (mreg + 0.5 * hmax), ret_highvol=pnl / (mhv + 0.5 * hmax)))
        # liquid-wing condor
        def wing(b, start, up):
            ks = b.index[b.index >= start] if up else b.index[b.index <= start][::-1]
            for K in ks:
                bid, ask, okq = quote(b, K)
                if okq and ask > 0:
                    return K, ask + COMM
            return None, None
        wu, wuc = wing(calls, S0 * np.exp(1.5 * sig * np.sqrt(T0)), True)
        wd, wdc = wing(puts, S0 * np.exp(-1.5 * sig * np.sqrt(T0)), False)
        if wu is not None and wd is not None and wu > k1u and wd < k1d:
            credit = cprem + pprem - wuc - wdc
            width = max(wu - k1u, k1d - wd)
            if width - credit > 0 and credit > 0:
                pnl = credit + payoff_short + max(ST - wu, 0) + max(wd - ST, 0)
                r = pnl / (width - credit)
                out.append(dict(base, strategy="condor_liquid", ret_regt=r, ret_highvol=r))
    return out


def account(df: pd.DataFrame, col: str) -> dict:
    """One entry per ticker per expiry (first day >= 25 days before expiry), IV rich only."""
    g = df[df.rich & ((df.exp - df.date).dt.days >= 25)].sort_values("date").groupby(["ticker", "exp"]).head(1)
    m = g.groupby(g.exp.dt.to_period("M"))[col].mean()
    eq = (1 + m).cumprod()
    dd = 1 - eq / eq.cummax()
    return dict(months=len(m), trades=len(g), avg_month_pct=round(100 * m.mean(), 2),
                total_pct=round(100 * (eq.iloc[-1] - 1), 1), worst_month_pct=round(100 * m.min(), 1),
                max_drawdown_pct=round(100 * dd.max(), 1), losing_months=int((m < 0).sum()))


def main():
    tick = sorted(p.name.split(".")[0] for p in (ROOT / "data" / "options").glob("*.csv.gz")
                  if (ROOT / "data" / "prices" / f"{p.name.split('.')[0]}.csv").exists()
                  and p.name.split(".")[0] not in LARGE)
    df = pd.DataFrame([r for t in tick for r in run_ticker(t)])
    df.to_csv(ROOT / "results" / "practical_trades.csv.gz", index=False, compression="gzip")
    rows = []
    for fname, g in [("all", df), ("IV rich", df[df.rich])]:
        for s, gs in g.groupby("strategy"):
            x = gs.set_index("date")
            for col in ["ret_regt", "ret_highvol"]:
                if s == "condor_liquid" and col == "ret_highvol":
                    continue
                m, t, n = S1.week_boot(x[[col]], col)
                rows.append(dict(filter=fname, strategy=s, margin="Reg-T" if col == "ret_regt" else "high-vol",
                                 n=n, tickers=gs.ticker.nunique(), mean_pct=round(100 * m, 2), t=round(t, 2),
                                 win_pct=round(100 * (gs[col] > 0).mean()), worst_pct=round(100 * gs[col].min())))
    tab = pd.DataFrame(rows)
    tab.to_csv(ROOT / "results" / "practical_summary.csv", index=False)
    pd.set_option("display.width", 220)
    print("PER TRADE (small/mid caps, bid/ask fills, return on the margin the trade ties up)")
    print(tab.to_string(index=False))
    acc = []
    for s in ["strangle_hedged_daily", "strangle_hedged_weekly", "strangle_unhedged", "condor_liquid"]:
        for col in (["ret_regt", "ret_highvol"] if s != "condor_liquid" else ["ret_regt"]):
            acc.append(dict(strategy=s, margin="Reg-T" if col == "ret_regt" else "high-vol",
                            **account(df[df.strategy == s], col)))
    acc = pd.DataFrame(acc)
    acc.to_csv(ROOT / "results" / "practical_account.csv", index=False)
    print("\nACCOUNT SIMULATION (IV rich only, one trade per stock per monthly cycle, equal capital per trade)")
    print(acc.to_string(index=False))
    g = df[df.rich & ((df.exp - df.date).dt.days >= 25)].sort_values("date").groupby(["ticker", "exp", "strategy"]).head(1)
    mm = g.pivot_table(index=g.exp.dt.to_period("M"), columns="strategy", values="ret_regt", aggfunc="mean").mul(100).round(1)
    print("\nMONTH BY MONTH, Reg-T margin, % return on capital:\n" + mm.to_string())
    print(f"\ntrials: {len(tab) + len(acc)}")


if __name__ == "__main__":
    main()
