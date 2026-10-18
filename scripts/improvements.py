"""Study 8: can we improve the hedged short-vol strategy without overfitting?

Fixed before running. Every variant is judged on the FIRST half (entries before 2026-04-01) and confirmed on
the SECOND half; a variant only "counts" if it beats the base in BOTH halves.
Realistic execution throughout: signal on day d, trade on the next quote day, sell at bid - 25% of spread,
only legs that traded that day, hedge every 5 days, Reg-T margin + 50% of max hedge (Study 7 "B+C").

  V0  base: 1 sd strangle, ~30 DTE, IV/RV20 >= 1.3
  V1  cross-sectional rank (Goyal & Saretto 2009): each day sell only the top third of names by IV/RV20
  V2  ATM straddle instead of strangle (same filter)
  V3  rich side only: sell just the 1 sd option (call or put) whose IV is higher relative to ATM
  V4  short-dated: target ~14 DTE (window 7-20 days) instead of ~30
  V5  long-short (Goyal & Saretto): V1 short leg + LONG hedged straddles on the bottom third
Account level (separately): interest on idle cash at 3.0%/yr (IBKR pays ~3.1% above $10k), and sizing set by
replaying historical 5-day joint moves of the traded names (2021-2026, skipping each stock's first 65 trading
days), plus an IV +50% shock, on random 8-position books; margin use is set so the worst replayed week of an
unlucky (90th-percentile) book costs <= 20% of the account. (Revised after the first run: the all-names
version understated single-name squeezes; the first stress run also included LUNR's pre-options SPAC week.)

python scripts/improvements.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import retail_spike_study as S1  # noqa: E402
from hedged_backtest import HEDGE_BPS, delta  # noqa: E402
from practical_backtest import naked_req  # noqa: E402
from real_options_backtest import COMM, LARGE, bs, iv  # noqa: E402
from robustness import Ticker  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CUT = pd.Timestamp("2026-04-01")
SLIP = 0.25


class T8(Ticker):
    def run8(self, structure="strangle", lo=20, hi=50, target=30, k=1.0, every=5):
        out = []
        dlist = list(self.exps.index)
        for i, d in enumerate(dlist[:-1]):
            if d not in self.close.index:
                continue
            ex = self.exps[d]
            dd = np.array([(e - d).days for e in ex])
            ok = (dd >= lo) & (dd <= hi)
            if not ok.any():
                continue
            e = ex[ok][np.argmin(np.abs(dd[ok] - target))]
            sig = self.atm.get((d, e), np.nan)
            ratio = sig / self.rv20.get(d, np.nan)
            x = dlist[i + 1]
            if not np.isfinite(sig) or x not in self.close.index or (x, e) not in self.book or e > self.days.max() or e <= x:
                continue
            S0, T0 = self.close.loc[x], (e - x).days / 365
            calls, puts = self.book[(x, e)]
            sg = self.atm.get((x, e), np.nan)
            sg = sg if np.isfinite(sg) else sig
            com = calls.index.intersection(puts.index)
            if len(com) < 3:
                continue
            near = lambda b, v: b.index[np.argmin(np.abs(b.index - v))]  # noqa: E731
            katm = com[np.argmin(np.abs(com - S0))]
            kc, kp = near(calls, S0 * np.exp(k * sg * np.sqrt(T0))), near(puts, S0 * np.exp(-k * sg * np.sqrt(T0)))
            kw = None
            if structure == "protected":
                for K in calls.index[calls.index >= S0 * np.exp(2.5 * sg * np.sqrt(T0))]:
                    q = calls.loc[K]
                    if q.ask > 0 and (q.ask - q.bid) / q.mid <= 0.5:
                        kw = K
                        break
                if kw is None:
                    continue
            if structure in ("straddle", "long_straddle"):
                kc = kp = katm
            elif not (kc > katm > kp):
                continue
            c, p = calls.loc[kc], puts.loc[kp]
            liq = lambda q: q.bid > 0 and (q.ask - q.bid) / q.mid <= 0.5 and q.volume > 0  # noqa: E731
            legs = []      # (right, K, qty, cash)
            if structure == "rich_side":
                ivc = iv(c.mid, S0, kc, T0, True)
                ivp = iv(p.mid, S0, kp, T0, False)
                if not (np.isfinite(ivc) and np.isfinite(ivp)):
                    continue
                q, rt, K = (c, "C", kc) if ivc / sg >= ivp / sg else (p, "P", kp)
                if not liq(q):
                    continue
                legs = [(rt, K, -1, q.bid - SLIP * (q.ask - q.bid) - COMM)]
            elif structure == "long_straddle":
                if not (c.ask > 0 and p.ask > 0 and (c.ask - c.bid) / c.mid <= .5 and (p.ask - p.bid) / p.mid <= .5
                        and c.volume > 0 and p.volume > 0):
                    continue
                legs = [("C", kc, +1, -(c.ask + SLIP * (c.ask - c.bid) + COMM)),
                        ("P", kp, +1, -(p.ask + SLIP * (p.ask - p.bid) + COMM))]
            else:
                if not (liq(c) and liq(p)):
                    continue
                legs = [("C", kc, -1, c.bid - SLIP * (c.ask - c.bid) - COMM), ("P", kp, -1, p.bid - SLIP * (p.ask - p.bid) - COMM)]
                if structure == "protected":
                    w = calls.loc[kw]
                    legs.append(("C", kw, +1, -(w.ask + SLIP * (w.ask - w.bid) + COMM)))
            path = self.days[(self.days >= x) & (self.days <= e)]
            ST = self.close.loc[path[-1]]
            hedge, h, hmax, s_t = 0.0, 0.0, 0.0, sg
            for j, day in enumerate(path[:-1]):
                S = self.close.loc[day]
                a = self.atm.get((day, e), np.nan)
                s_t = a if np.isfinite(a) else s_t
                if j % every == 0:
                    new = -sum(qty * delta(S, K, (e - day).days / 365, s_t, rt == "C") for rt, K, qty, _ in legs)
                    hedge -= abs(new - h) * S * HEDGE_BPS
                    h = new
                    hmax = max(hmax, abs(h) * S)
                hedge += h * (self.close.loc[path[j + 1]] - S)
            hedge -= abs(h) * ST * HEDGE_BPS
            pnl = sum(cash for *_, cash in legs) + sum(qty * (max(ST - K, 0) if rt == "C" else max(K - ST, 0))
                                                        for rt, K, qty, _ in legs) + hedge
            if structure == "long_straddle":
                cap = -sum(cash for *_, cash in legs) + 0.5 * hmax
            else:
                short = [(rt, K, cash) for rt, K, qty, cash in legs if qty < 0]
                reqs = [naked_req(cash, S0, K, rt == "C", .2, .1) for rt, K, cash in short]
                cap = max(reqs) + (min(cash for *_, cash in short) if len(short) == 2 else 0) + 0.5 * hmax
            out.append(dict(date=d, exp=e, ticker=self.t, ratio=ratio, ret=pnl / cap, cap_per_share=cap, S0=S0, sig=sg,
                            kc=kc, kp=kp, kw=kw, T0=T0))
        return out


def halves(df, label):
    r = {"variant": label}
    for nm, g in [("1st half", df[df.date < CUT]), ("2nd half", df[df.date >= CUT]), ("full", df)]:
        if len(g) < 15:
            r[nm] = "n/a"
            continue
        m, t, n = S1.week_boot(g.set_index("date")[["ret"]], "ret")
        r[nm] = f"{100 * m:+.2f}% (t {t:.1f}, n {n})"
    return r


def account(df):
    g = df[(df.exp - df.date).dt.days >= 7].sort_values("date").groupby(["ticker", "exp"]).head(1)
    return g.groupby(g.exp.dt.to_period("M")).ret.mean()


def main():
    tick = sorted(p.name.split(".")[0] for p in (ROOT / "data" / "options").glob("*.csv.gz")
                  if p.name.split(".")[0] not in LARGE and (ROOT / "data" / "prices" / f"{p.name.split('.')[0]}.csv").exists())
    T = [T8(t) for t in tick]
    run = lambda **kw: pd.DataFrame([r for x in T for r in x.run8(**kw)])  # noqa: E731
    strangle = run()
    v0 = strangle[strangle.ratio >= 1.3]
    rank = strangle.groupby("date").ratio.rank(pct=True)
    v1 = strangle[rank >= 2 / 3]
    st = run(structure="straddle")
    v2 = st[st.ratio >= 1.3]
    rs = run(structure="rich_side")
    v3 = rs[rs.ratio >= 1.3]
    sd = run(lo=7, hi=20, target=14)
    v4 = sd[sd.ratio >= 1.3]
    ls = run(structure="long_straddle")
    lrank = ls.groupby("date").ratio.rank(pct=True)
    long_leg = ls[lrank <= 1 / 3]
    pr = run(structure="protected")
    v6 = pr[pr.ratio >= 1.3]
    rows = [halves(v0, "V0 base strangle IV/RV>=1.3"), halves(v6, "V6 strangle + long 2.5 sd call (squeeze insurance)"),
            halves(v0[v0.set_index(['ticker', 'date']).index.isin(v6.set_index(['ticker', 'date']).index)],
                   "    (V0 on the same trades V6 could take)"), halves(strangle, "    (reference: all days, no filter)"),
            halves(v1, "V1 top third by IV/RV each day"), halves(v2, "V2 ATM straddle"),
            halves(v3, "V3 rich side only"), halves(v4, "V4 short-dated ~14 DTE"),
            halves(long_leg, "V5 long leg: hedged long straddle, bottom third")]
    # V5 combined, per day: average of short leg and long leg returns
    a, b = v1.groupby("date").ret.mean(), long_leg.groupby("date").ret.mean()
    comb = pd.concat([a, b], axis=1, keys=["s", "l"]).dropna()
    v5 = pd.DataFrame({"date": comb.index, "ret": (comb.s + comb.l) / 2, "exp": comb.index, "ticker": "LS"})
    rows.append(halves(v5, "V5 long-short (V1 short + bottom-third long), daily avg"))
    tab = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print("PER-TRADE RETURN ON CAPITAL, realistic execution (next-day entry, bid-25% spread, traded legs only)")
    print(tab.to_string(index=False))
    tab.to_csv(ROOT / "results" / "improvements_variants.csv", index=False)

    # ---- account level: historical-stress sizing + interest on idle cash
    px = {t: pd.read_csv(ROOT / "data" / "prices" / f"{t}.csv", parse_dates=["date"]).set_index("date").close for t in tick}
    P = pd.DataFrame(px)
    r5 = np.log(P).diff(5)                       # 5-day joint moves, all names on the same dates
    print("\nACCOUNT LEVEL (base rule V0, one trade per stock per cycle, equal margin per position)")
    rows = []
    for nm, dfv in [("V0 base", v0), ("V1 top third", v1), ("V4 short-dated", v4), ("V6 squeeze-protected", v6)]:
        mon = account(dfv)
        # stress: per $1 of margin, loss of a representative position under each historical 5-day move.
        g = dfv.sort_values("date").groupby(["ticker", "exp"]).head(1)
        losses = []
        for t, gt in g.groupby("ticker"):
            if t not in r5:
                continue
            row = gt.iloc[-1]
            S, sigma, T = row.S0, row.sig, row.T0
            kc, kp = row.kc, row.kp
            kw = row.kw if "kw" in row and pd.notna(row.kw) else None
            hist = r5[t].dropna().iloc[65:]          # skip the first ~3 months after listing (no options yet)
            S1_ = S * np.exp(hist.to_numpy())
            Tn = max(T - 5 / 365, 1 / 365)
            prem0 = bs(S, kc, T, sigma, True) + bs(S, kp, T, sigma, False)
            val1 = bs(S1_, kc, Tn, sigma * 1.5, True) + bs(S1_, kp, Tn, sigma * 1.5, False)
            dlt = delta(S, kc, T, sigma, True) + delta(S, kp, T, sigma, False)
            pnl = prem0 - val1 + dlt * (S1_ - S)
            if kw is not None:
                pnl = pnl - bs(S, kw, T, sigma, True) + bs(S1_, kw, Tn, sigma * 1.5, True) - delta(S, kw, T, sigma, True) * (S1_ - S)
            margin = row.cap_per_share
            losses.append(pd.Series(pnl / margin, index=hist.index, name=t))
        Ldf = pd.concat(losses, axis=1)
        L = Ldf.mean(axis=1)                           # all names at once (reported for reference)
        # realistic book: 8 positions. Worst historical week for 1,000 random 8-name books; plan for the
        # 90th-percentile (unlucky) book.
        rng = np.random.default_rng(1)
        cols = list(Ldf.columns)
        mins = [Ldf[list(rng.choice(cols, min(8, len(cols)), replace=False))].mean(axis=1, skipna=True).min()
                for _ in range(1000)]
        worst = -np.percentile(mins, 10)
        util = min(1.0, 0.20 / worst)                  # margin as share of account so worst week <= 20%
        acct = util * mon + (1 - util) * 0.03 / 12
        eq = (1 + acct).cumprod()
        rows.append(dict(variant=nm, worst_week_8name_book=f"{-worst:.1%}", worst_week_all_names=f"{L.min():.1%}",
                         worst_week_date=str(L.idxmin().date()),
                         p99_week_per_margin=f"{L.quantile(0.01):.1%}",
                         margin_use=f"{util:.0%}", acct_avg_month=f"{acct.mean():.2%}",
                         acct_annualized=f"{(1 + acct.mean()) ** 12 - 1:.1%}", acct_worst_month=f"{acct.min():.2%}",
                         acct_max_dd=f"{(1 - eq / eq.cummax()).max():.1%}",
                         without_interest=f"{(1 + (util * mon).mean()) ** 12 - 1:.1%}"))
    acc = pd.DataFrame(rows)
    print(acc.to_string(index=False))
    acc.to_csv(ROOT / "results" / "improvements_account.csv", index=False)
    print("\ntrials added:", len(tab) + len(acc))


if __name__ == "__main__":
    main()
