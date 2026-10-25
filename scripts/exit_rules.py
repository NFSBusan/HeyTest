"""Study 10: standard exit rules for the hedged short strangle (rule-based 71-stock universe).

Fixed before running (standard practitioner rules, not tuned; a variant counts only if it beats
"hold to expiry" in BOTH halves):
  E0  hold to expiry (current)
  E1  take profit when the strangle can be bought back for <= 50% of the credit received
  E2  stop loss when the loss on the options (incl. hedge P&L) reaches 2x the credit
  E3  E1 + E2
  Sensitivity (reported, not selectable): profit targets 25% / 75%, stop 3x.
Selection: top third by IV/RV20 each day (Study 9's V1), realistic execution (next-day entry, sell at bid - 25% of
spread, buy back at ask + 25% of spread, legs must have traded, hedge every 5 days, Reg-T margin + 50% max hedge).
Marking uses daily option mids. Capital efficiency: return per trade and per calendar day held; "per 30 days" =
return x 30 / days held (assumes freed capital is re-deployed at the same rate; stated as an assumption).
Also: account return vs the allowed stress loss (15/20/25/30% of the account) for the hold-to-expiry version.

python scripts/exit_rules.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import retail_spike_study as S1  # noqa: E402
from hedged_backtest import HEDGE_BPS, delta  # noqa: E402
from improvements import T8  # noqa: E402
from practical_backtest import naked_req  # noqa: E402
from real_options_backtest import COMM  # noqa: E402
from universe_study import ETF_LEAKS, price_file  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CUT = pd.Timestamp("2026-04-01")
SLIP = 0.25
RULES = {"E0 hold to expiry": (None, None), "E1 take profit 50%": (0.5, None), "E2 stop loss 2x": (None, 2.0),
         "E3 TP 50% + SL 2x": (0.5, 2.0), "sens: TP 25%": (0.25, None), "sens: TP 75%": (0.75, None),
         "sens: SL 3x": (None, 3.0)}


class TX(T8):
    def run_exits(self, every=5):
        out = []
        dlist = list(self.exps.index)
        for i, d in enumerate(dlist[:-1]):
            if d not in self.close.index:
                continue
            ex = self.exps[d]
            dd = np.array([(e - d).days for e in ex])
            ok = (dd >= 20) & (dd <= 50)
            if not ok.any():
                continue
            e = ex[ok][np.argmin(np.abs(dd[ok] - 30))]
            sig = self.atm.get((d, e), np.nan)
            x = dlist[i + 1]
            if not np.isfinite(sig) or x not in self.close.index or (x, e) not in self.book or e > self.days.max() or e <= x:
                continue
            ratio = sig / self.rv20.get(d, np.nan)
            S0, T0 = self.close.loc[x], (e - x).days / 365
            calls, puts = self.book[(x, e)]
            sg = self.atm.get((x, e), np.nan)
            sg = sg if np.isfinite(sg) else sig
            com = calls.index.intersection(puts.index)
            if len(com) < 3:
                continue
            near = lambda b, v: b.index[np.argmin(np.abs(b.index - v))]  # noqa: E731
            katm = com[np.argmin(np.abs(com - S0))]
            kc, kp = near(calls, S0 * np.exp(sg * np.sqrt(T0))), near(puts, S0 * np.exp(-sg * np.sqrt(T0)))
            if not (kc > katm > kp):
                continue
            c, p = calls.loc[kc], puts.loc[kp]
            liq = lambda q: q.bid > 0 and (q.ask - q.bid) / q.mid <= 0.5 and q.volume > 0  # noqa: E731
            if not (liq(c) and liq(p)):
                continue
            credit = (c.bid - SLIP * (c.ask - c.bid) - COMM) + (p.bid - SLIP * (p.ask - p.bid) - COMM)
            m0 = max(naked_req(c.bid, S0, kc, True, .2, .1), naked_req(p.bid, S0, kp, False, .2, .1)) + min(c.bid, p.bid)
            path = self.days[(self.days >= x) & (self.days <= e)]
            # simulate the path once, recording daily (hedge P&L, buy-back cost)
            hedge, h, hmax, s_t, marks = 0.0, 0.0, 0.0, sg, []
            for j, day in enumerate(path[:-1]):
                S = self.close.loc[day]
                a = self.atm.get((day, e), np.nan)
                s_t = a if np.isfinite(a) else s_t
                if j % every == 0:
                    new = delta(S, kc, (e - day).days / 365, s_t, True) + delta(S, kp, (e - day).days / 365, s_t, False)
                    hedge -= abs(new - h) * S * HEDGE_BPS
                    h = new
                    hmax = max(hmax, abs(h) * S)
                nxt = path[j + 1]
                hedge += h * (self.close.loc[nxt] - S)
                cost = np.nan
                if (nxt, e) in self.book and nxt != path[-1]:
                    cc, pp = self.book[(nxt, e)]
                    if kc in cc.index and kp in pp.index:
                        qc, qp = cc.loc[kc], pp.loc[kp]
                        if qc.ask > 0 and qp.ask > 0:
                            cost = (qc.ask + SLIP * (qc.ask - qc.bid) + COMM) + (qp.ask + SLIP * (qp.ask - qp.bid) + COMM)
                marks.append((nxt, hedge - abs(h) * self.close.loc[nxt] * HEDGE_BPS, cost, (qc.mid + qp.mid) if np.isfinite(cost) else np.nan))
            ST = self.close.loc[path[-1]]
            final = credit - max(ST - kc, 0) - max(kp - ST, 0) + marks[-1][1] if marks else np.nan
            cap = m0 + 0.5 * hmax
            res = dict(date=d, exp=e, ticker=self.t, ratio=ratio, cap=cap, S0=S0, sig=sg, kc=kc, kp=kp, T0=T0)
            for name, (tp, sl) in RULES.items():
                pnl, held = final, (path[-1] - x).days
                for day, hpnl, cost, mid in marks:
                    if not np.isfinite(cost):
                        continue
                    opt_pnl_mid = credit - mid
                    if tp is not None and mid <= (1 - tp) * credit:
                        pnl, held = credit - cost + hpnl, (day - x).days
                        break
                    if sl is not None and -(opt_pnl_mid + hpnl) >= sl * credit:
                        pnl, held = credit - cost + hpnl, (day - x).days
                        break
                res[name] = pnl / cap
                res[name + " days"] = max(held, 1)
            out.append(res)
        return out


def main():
    uni = pd.read_csv(ROOT / "data" / "universe_2025-09.csv")
    syms = [s for s in uni.symbol if s not in ETF_LEAKS and (ROOT / "data" / "options" / f"{s}.csv.gz").exists()
            and price_file(s).exists()]
    T = []
    for s in syms:
        try:
            T.append(TX(s))
        except Exception as e:  # noqa: BLE001
            print("skip", s, repr(e)[:60])
    df = pd.DataFrame([r for x in T for r in x.run_exits()])
    df = df[df.groupby("date").ratio.rank(pct=True) >= 2 / 3]          # V1: top third each day
    rows = []
    for name in RULES:
        r = {"rule": name, "avg_days_held": round(df[name + " days"].mean(), 1)}
        for nm, g in [("1st half", df[df.date < CUT]), ("2nd half", df[df.date >= CUT]), ("full", df)]:
            m, t, n = S1.week_boot(g.set_index("date")[[name]], name)
            per30 = (g[name] * 30 / g[name + " days"]).mean()
            r[nm] = f"{100 * m:+.2f}% (t {t:.1f})"
            r[nm + " per30d"] = f"{100 * per30:+.2f}%"
        r["worst_trade"] = f"{100 * df[name].min():.0f}%"
        r["win"] = f"{100 * (df[name] > 0).mean():.0f}%"
        rows.append(r)
    tab = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(f"{len(df)} trades, {df.ticker.nunique()} stocks (top third by IV/RV each day)\n")
    print(tab.to_string(index=False))
    tab.to_csv(ROOT / "results" / "exit_rules.csv", index=False)
    df.to_csv(ROOT / "results" / "exit_rules_trades.csv.gz", index=False, compression="gzip")


if __name__ == "__main__":
    main()
