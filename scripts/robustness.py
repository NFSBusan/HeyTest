"""Study 7: robustness checks for the hedged short-strangle strategy (small/mid caps, real quotes).

Base case = Study 6 / paper-trade rule: 1 sd strangle, DTE closest to 30, IV/RV20 >= 1.3, hedge every 5 days,
sell at the bid, enter at the same close the signal is computed on, Reg-T margin + 50% of max hedge.

Checks (written before running):
  A. Sensitivity, one parameter at a time: strike k in {0.75, 1, 1.25} sd; DTE target {20, 30, 45};
     hedge every {1, 3, 5, 10} days; IV/RV threshold {0 (all), 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.7}.
  B. Lookahead: enter at the NEXT day's close and quotes (signal from day d, trade on d+1).
  C. Fills: sell at bid - 25% of the spread; and only legs whose option traded (volume > 0) that day.
  D. Placebo: 2,000 random draws of the same number of trades from ALL eligible days (same tickers),
     compare with the IV-rich subset.
  E. Time split: entries before / after 2026-04-01.
  F. Concentration: drop the 3 tickers with the largest total P&L.
  G. Path risk: daily mark-to-market (option mids + hedge P&L). Count trades whose interim loss exceeded
     the initial margin (would face a margin call / forced close).
  H. Multiple testing: deflated Sharpe of the per-trade series, n_trials = 160 (all studies so far).

python scripts/robustness.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import retail_spike_study as S1  # noqa: E402
from hedged_backtest import HEDGE_BPS, delta  # noqa: E402
from pmct.stats import deflated_sharpe  # noqa: E402
from practical_backtest import naked_req  # noqa: E402
from real_options_backtest import COMM, LARGE, iv  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BASE = dict(k=1.0, dte=30, every=5, lag=0, slip=0.0, need_vol=False, mtm=False)


class Ticker:
    def __init__(self, t):
        self.t = t
        op = pd.read_csv(ROOT / "data" / "options" / f"{t}.csv.gz", parse_dates=["date", "expiration"])
        op = op[op.ask > 0].copy()
        op["mid"] = (op.bid + op.ask) / 2
        pfile = ROOT / "data" / "prices" / f"{t}.csv"
        if not pfile.exists():
            pfile = ROOT / "data" / "prices_td" / f"{t}.csv"      # ThetaData 1-year prices (universe names)
        px = pd.read_csv(pfile, parse_dates=["date"]).set_index("date")
        self.close = px.close
        self.days = px.index
        self.rv20 = np.log(px.close).diff().rolling(20, min_periods=15).std() * np.sqrt(252)
        self.book = {k: (g[g.right == "C"].set_index("strike").sort_index(), g[g.right == "P"].set_index("strike").sort_index())
                     for k, g in op.groupby(["date", "expiration"])}
        self.exps = op.groupby("date").expiration.unique()
        self.atm = {}
        for (d, e), (c, p) in self.book.items():
            if d in self.close.index and e > d:
                S, T = self.close.loc[d], (e - d).days / 365
                com = c.index.intersection(p.index)
                if len(com):
                    k = com[np.argmin(np.abs(com - S))]
                    self.atm[(d, e)] = np.nanmean([iv(c.loc[k, "mid"], S, k, T, True), iv(p.loc[k, "mid"], S, k, T, False)])

    def run(self, k, dte, every, lag, slip, need_vol, mtm):
        out = []
        dlist = list(self.exps.index)
        for i, d in enumerate(dlist):
            if d not in self.close.index:
                continue
            ex = self.exps[d]
            dd = np.array([(e - d).days for e in ex])
            ok = (dd >= 20) & (dd <= 50 + 15 * (dte > 30))
            if not ok.any():
                continue
            e = ex[ok][np.argmin(np.abs(dd[ok] - dte))]
            sig = self.atm.get((d, e), np.nan)
            ratio = sig / self.rv20.get(d, np.nan)
            if not np.isfinite(sig):
                continue
            # execution day: signal on d, trade on the next quote day if lag = 1
            if lag:
                if i + 1 >= len(dlist):
                    continue
                x = dlist[i + 1]
                if x not in self.close.index or (x, e) not in self.book:
                    continue
            else:
                x = d
            if e > self.days.max():
                continue
            S0, T0 = self.close.loc[x], (e - x).days / 365
            calls, puts = self.book[(x, e)]
            sg = self.atm.get((x, e), np.nan)
            sg = sg if np.isfinite(sg) else sig
            near = lambda b, v: b.index[np.argmin(np.abs(b.index - v))]  # noqa: E731
            com = calls.index.intersection(puts.index)
            if len(com) < 3:
                continue
            katm = com[np.argmin(np.abs(com - S0))]
            kc, kp = near(calls, S0 * np.exp(k * sg * np.sqrt(T0))), near(puts, S0 * np.exp(-k * sg * np.sqrt(T0)))
            if not (kc > katm > kp):
                continue
            c, p = calls.loc[kc], puts.loc[kp]
            good = all(q.bid > 0 and (q.ask - q.bid) / q.mid <= 0.5 for q in (c, p))
            if not good or (need_vol and (c.volume <= 0 or p.volume <= 0)):
                continue
            cprem = c.bid - slip * (c.ask - c.bid) - COMM
            pprem = p.bid - slip * (p.ask - p.bid) - COMM
            path = self.days[(self.days >= x) & (self.days <= e)]
            ST = self.close.loc[path[-1]]
            hedge, h, hmax, s_t, worst_mtm = 0.0, 0.0, 0.0, sg, 0.0
            for j, day in enumerate(path[:-1]):
                S = self.close.loc[day]
                a = self.atm.get((day, e), np.nan)
                s_t = a if np.isfinite(a) else s_t
                if j % every == 0:
                    new = delta(S, kc, (e - day).days / 365, s_t, True) + delta(S, kp, (e - day).days / 365, s_t, False)
                    hedge -= abs(new - h) * S * HEDGE_BPS
                    h = new
                    hmax = max(hmax, abs(h) * S)
                hedge += h * (self.close.loc[path[j + 1]] - S)
                if mtm and (path[j + 1], e) in self.book:
                    cc, pp = self.book[(path[j + 1], e)]
                    if kc in cc.index and kp in pp.index:
                        val = cprem + pprem - cc.loc[kc, "mid"] - pp.loc[kp, "mid"] + hedge
                        worst_mtm = min(worst_mtm, val)
            hedge -= abs(h) * ST * HEDGE_BPS
            pnl = cprem + pprem - max(ST - kc, 0) - max(kp - ST, 0) + hedge
            m0 = max(naked_req(c.bid, S0, kc, True, .2, .1), naked_req(p.bid, S0, kp, False, .2, .1)) + min(c.bid, p.bid)
            cap = m0 + 0.5 * hmax
            out.append(dict(date=d, exp=e, ticker=self.t, ratio=ratio, ret=pnl / cap,
                            mtm_breach=bool(mtm and -worst_mtm > m0), worst_mtm=worst_mtm / cap if mtm else np.nan))
        return out


def summ(df, label):
    if len(df) < 15:
        return dict(case=label, n=len(df))
    m, t, n = S1.week_boot(df.set_index("date")[["ret"]], "ret")
    g = df[(df.exp - df.date).dt.days >= 20].sort_values("date").groupby(["ticker", "exp"]).head(1)
    mon = g.groupby(g.exp.dt.to_period("M")).ret.mean()
    return dict(case=label, n=n, tickers=df.ticker.nunique(), mean_pct=round(100 * m, 2), t=round(t, 2),
                win_pct=round(100 * (df.ret > 0).mean()), worst_pct=round(100 * df.ret.min()),
                acct_avg_month_pct=round(100 * mon.mean(), 2), acct_worst_month_pct=round(100 * mon.min(), 1),
                acct_losing_months=f"{int((mon < 0).sum())}/{len(mon)}")


def main():
    tick = sorted(p.name.split(".")[0] for p in (ROOT / "data" / "options").glob("*.csv.gz")
                  if p.name.split(".")[0] not in LARGE and (ROOT / "data" / "prices" / f"{p.name.split('.')[0]}.csv").exists())
    T = [Ticker(t) for t in tick]
    cache = {}

    def get(**kw):
        cfg = dict(BASE, **kw)
        key = tuple(sorted(cfg.items()))
        if key not in cache:
            cache[key] = pd.DataFrame([r for x in T for r in x.run(**cfg)])
        return cache[key]

    rows = []
    base_all = get(mtm=True)
    base = base_all[base_all.ratio >= 1.3]
    rows.append(summ(base, "BASE: k=1, dte=30, hedge 5d, IV/RV>=1.3"))
    # A. sensitivity
    for thr in [0, 1.0, 1.1, 1.2, 1.4, 1.5, 1.7]:
        rows.append(summ(base_all[base_all.ratio >= thr], f"A thr IV/RV>={thr}"))
    for k in [0.75, 1.25]:
        d = get(k=k)
        rows.append(summ(d[d.ratio >= 1.3], f"A strikes {k} sd"))
    for dte in [20, 45]:
        d = get(dte=dte)
        rows.append(summ(d[d.ratio >= 1.3], f"A dte {dte}"))
    for ev in [1, 3, 10]:
        d = get(every=ev)
        rows.append(summ(d[d.ratio >= 1.3], f"A hedge every {ev}d"))
    # B. next-day entry
    d = get(lag=1)
    rows.append(summ(d[d.ratio >= 1.3], "B enter NEXT day"))
    # C. fills
    d = get(slip=0.25)
    rows.append(summ(d[d.ratio >= 1.3], "C sell at bid - 25% spread"))
    d = get(need_vol=True)
    rows.append(summ(d[d.ratio >= 1.3], "C only options that traded that day"))
    d = get(lag=1, slip=0.25, need_vol=True)
    rows.append(summ(d[d.ratio >= 1.3], "B+C combined (most realistic)"))
    # E. time split
    cut = pd.Timestamp("2026-04-01")
    rows.append(summ(base[base.date < cut], "E first half (Oct-Mar)"))
    rows.append(summ(base[base.date >= cut], "E second half (Apr-Oct)"))
    # F. concentration
    top = base.groupby("ticker").ret.sum().nlargest(3).index.tolist()
    rows.append(summ(base[~base.ticker.isin(top)], f"F drop top-3 tickers {top}"))
    tab = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(tab.to_string(index=False))
    tab.to_csv(ROOT / "results" / "robustness_summary.csv", index=False)

    # D. placebo
    rng = np.random.default_rng(0)
    pool = base_all.ret.to_numpy()
    sims = np.array([rng.choice(pool, len(base), replace=False).mean() for _ in range(2000)])
    print(f"\nD placebo: IV-rich mean {100 * base.ret.mean():.2f}% vs random same-size picks "
          f"mean {100 * sims.mean():.2f}% (95% range {100 * np.percentile(sims, 2.5):.2f}% .. {100 * np.percentile(sims, 97.5):.2f}%); "
          f"share of random picks >= IV-rich: {(sims >= base.ret.mean()).mean():.3f}")
    # G. path risk
    print(f"G path risk (base, IV rich): trades whose interim mark-to-market loss exceeded the initial margin: "
          f"{base.mtm_breach.mean():.1%} ({int(base.mtm_breach.sum())} of {len(base)}); "
          f"median worst interim P&L {100 * base.worst_mtm.median():.1f}% of capital, 5th pct {100 * base.worst_mtm.quantile(.05):.1f}%")
    # H. deflated Sharpe on non-overlapping per-ticker trades
    g = base[(base.exp - base.date).dt.days >= 20].sort_values("date").groupby(["ticker", "exp"]).head(1)
    r = g.ret.to_numpy()
    sr = r.mean() / r.std(ddof=1)
    dsr = deflated_sharpe(sr, len(r), skew(r), kurtosis(r, fisher=False), n_trials=160, sr_var=1 / len(r))
    print(f"H deflated Sharpe: per-trade Sharpe {sr:.2f} over {len(r)} non-overlapping trades, skew {skew(r):.2f}; "
          f"P(true Sharpe > 0 after 160 trials) = {dsr:.3f}")
    per = base.groupby("ticker").ret.agg(["size", "mean"])
    print(f"\nTickers positive: {(per['mean'] > 0).sum()} of {len(per)}")
    print(f"trials added by this study: {len(tab)}")


if __name__ == "__main__":
    main()
