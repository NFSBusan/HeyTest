"""Study 9: the hedged short-strangle strategy on the RULE-BASED universe (80 names chosen from Sept-2025 FINRA data).

Fixed before running:
  * Universe: data/universe_2025-09.csv minus ETFs that slipped through the exclusion list
    (VCIT SPLG IEMG IEFA MUB JPST IAU EMB USHY). 71 stocks.
  * Groups reported: all; "retail-heavy half" = Sept-2025 off-exchange share >= the universe median;
    "new names" = stocks NOT in the original hand-picked 26 (true out-of-sample: they played no part in
    designing the strategy); "original names" = the overlap.
  * Strategy = Study 7/8 realistic version: 1 sd strangle, ~30 DTE, IV/RV20 >= 1.3, next-day entry,
    sell at bid - 25% of spread, legs must have traded, hedge every 5 days, Reg-T margin + 50% of max hedge.
    Also reported: no filter (all days) and the cross-sectional top third (V1).
  * Account: one entry per stock per monthly cycle; at most 3 positions per sector cluster per cycle
    (crypto miners, quantum, nuclear, space, regional banks, China ADRs); equal margin per position.
    Margin use set by a stress test (IV +50% shock): an unlucky (90th pct) random book's worst 5-day move over the
    period all names have data, PLUS one position taking the worst single-name 5-day move in any name's history
    (meme squeeze), must cost <= 20% of the account. (Revised after the first run: averaging over whichever names
    had history on old dates distorted the stress.)
    Idle cash earns 3.0%/yr. A 50% haircut gives the "planning" figure.

python scripts/universe_study.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import retail_spike_study as S1  # noqa: E402
from hedged_backtest import delta  # noqa: E402
from improvements import T8, halves  # noqa: E402
from real_options_backtest import bs  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ETF_LEAKS = {"VCIT", "SPLG", "IEMG", "IEFA", "MUB", "JPST", "IAU", "EMB", "USHY"}
ORIGINAL = set("""ASTS RKLB IREN ONDS APLD WULF IBRX SOFI NBIS CRWV SOUN BBAI QBTS RGTI LUNR OKLO ACHR JOBY CIFR
CLSK MARA RIOT AMC PLUG BMEA NAK INDI""".split())
CLUSTERS = {"crypto": "MARA RIOT IREN CIFR WULF APLD BMNR HIVE BITF CLSK HOOD COIN",
            "quantum": "RGTI QBTS IONQ QUBT", "nuclear": "OKLO SMR NNE", "space": "RKLB ASTS LUNR SATS",
            "regional_banks": "KEY RF HBAN", "china": "NIO JD BIDU PDD"}
CLUSTER_OF = {t: c for c, names in CLUSTERS.items() for t in names.split()}
MAX_PER_CLUSTER = 3


def price_file(t):
    f = ROOT / "data" / "prices" / f"{t}.csv"
    return f if f.exists() else ROOT / "data" / "prices_td" / f"{t}.csv"


def cycles(df):
    """One entry per stock per expiry (earliest eligible day), then the cluster cap within each expiry."""
    g = df[(df.exp - df.date).dt.days >= 20].sort_values("date").groupby(["ticker", "exp"]).head(1).copy()
    g["cluster"] = g.ticker.map(CLUSTER_OF).fillna(g.ticker)
    g["rank_in_cluster"] = g.sort_values("ratio", ascending=False).groupby(["exp", "cluster"]).cumcount()
    return g[g.rank_in_cluster < MAX_PER_CLUSTER]


def stress_losses(g):
    out = []
    for t, gt in g.groupby("ticker"):
        px = pd.read_csv(price_file(t), parse_dates=["date"]).set_index("date").close
        h = np.log(px).diff(5).dropna()
        if len(px) > 300:
            h = h.iloc[65:]                  # skip first ~3 months after listing for long histories
        if len(h) < 60:
            continue
        row = gt.iloc[-1]
        S, sig, T, kc, kp = row.S0, row.sig, row.T0, row.kc, row.kp
        S1_ = S * np.exp(h.to_numpy())
        Tn = max(T - 5 / 365, 1 / 365)
        pnl = (bs(S, kc, T, sig, True) + bs(S, kp, T, sig, False) - bs(S1_, kc, Tn, sig * 1.5, True)
               - bs(S1_, kp, Tn, sig * 1.5, False) + (delta(S, kc, T, sig, True) + delta(S, kp, T, sig, False)) * (S1_ - S))
        out.append(pd.Series(pnl / row.cap_per_share, index=h.index, name=t))
    return pd.concat(out, axis=1)


GLOBAL_SINGLE = None   # worst single-name squeeze across ALL names, applied to every group


def account(dfv, label):
    g = cycles(dfv)
    mon = g.groupby(g.exp.dt.to_period("M")).ret.mean()
    npos = int(g.groupby("exp").size().median())
    L = stress_losses(g)
    # Consistent stress: (a) the book's worst common week over the period where >= 90% of names have data,
    # plus (b) one position suffering the worst single-name week seen in ANY name's full history (a squeeze).
    common = L[L.notna().mean(axis=1) >= 0.9]
    single = max(-np.nanmin(L.to_numpy()), GLOBAL_SINGLE or 0)
    rng = np.random.default_rng(1)
    cols = list(L.columns)
    k = max(1, min(npos, len(cols)))
    books = [-common[list(rng.choice(cols, k, replace=False))].mean(axis=1).min() for _ in range(1000)]
    worst = np.percentile(books, 90) + single / k
    util = min(1.0, 0.20 / worst)
    acct = util * mon + (1 - util) * 0.03 / 12
    ann = (1 + acct.mean()) ** 12 - 1
    eq = (1 + acct).cumprod()
    return dict(group=label, months=len(mon), positions_per_month=npos, avg_return_on_margin_month=f"{mon.mean():.1%}",
                losing_months=f"{int((mon < 0).sum())}/{len(mon)}", unlucky_worst_week=f"{-worst:.0%}",
                single_name_squeeze=f"{-single:.0%}",
                margin_use=f"{util:.0%}", backtest_annual=f"{ann:.1%}", planning_annual_50pct_haircut=f"{ann / 2:.1%}",
                worst_month_account=f"{acct.min():.1%}", max_dd=f"{(1 - eq / eq.cummax()).max():.1%}")


def main():
    uni = pd.read_csv(ROOT / "data" / "universe_2025-09.csv")
    uni = uni[~uni.symbol.isin(ETF_LEAKS)].copy()
    uni["offex_share"] = uni.finra_shares / (uni.avg_total_volume * 21)
    med = uni.offex_share.median()
    heavy = set(uni.loc[uni.offex_share >= med, "symbol"])
    syms = [s for s in uni.symbol if (ROOT / "data" / "options" / f"{s}.csv.gz").exists() and price_file(s).exists()]
    print(f"universe {len(uni)} stocks after removing ETFs; {len(syms)} with data; median off-exchange share {med:.2f}")
    T = []
    for s in syms:
        try:
            T.append(T8(s))
        except Exception as e:  # noqa: BLE001
            print("skip", s, repr(e)[:80])
    allrows = pd.DataFrame([r for x in T for r in x.run8()])
    v0 = allrows[allrows.ratio >= 1.3]
    v1 = allrows[allrows.groupby("date").ratio.rank(pct=True) >= 2 / 3]
    new = set(syms) - ORIGINAL
    rows = []
    for nm, g in [("ALL stocks, rule V0", v0), ("all stocks, no filter", allrows), ("all stocks, top third (V1)", v1),
                  ("retail-heavy half, V0", v0[v0.ticker.isin(heavy)]), ("less retail half, V0", v0[~v0.ticker.isin(heavy)]),
                  ("NEW names only (out-of-sample), V0", v0[v0.ticker.isin(new)]),
                  ("original names, V0", v0[v0.ticker.isin(ORIGINAL)])]:
            rows.append(dict(halves(g, nm), tickers=g.ticker.nunique()))
    tab = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print("\nPER TRADE, realistic execution\n" + tab.to_string(index=False))
    tab.to_csv(ROOT / "results" / "universe_per_trade.csv", index=False)
    per = v0.groupby("ticker").ret.agg(["size", "mean"])
    print(f"\nstocks positive (V0): {(per['mean'] > 0).sum()} of {len(per)}; "
          f"worst: {per['mean'].nsmallest(5).mul(100).round(1).to_dict()}")
    global GLOBAL_SINGLE
    GLOBAL_SINGLE = -np.nanmin(stress_losses(cycles(allrows)).to_numpy())
    acc = pd.DataFrame([account(v0, "ALL, V0"), account(v1, "ALL, top third"),
                        account(v0[v0.ticker.isin(heavy)], "retail-heavy half, V0"),
                        account(v0[v0.ticker.isin(new)], "NEW names only, V0")])
    print("\nACCOUNT (cluster cap 3, stress-based sizing, 3% on idle cash)\n" + acc.to_string(index=False))
    acc.to_csv(ROOT / "results" / "universe_account.csv", index=False)
    allrows.to_csv(ROOT / "results" / "universe_trades.csv.gz", index=False, compression="gzip")


if __name__ == "__main__":
    main()
