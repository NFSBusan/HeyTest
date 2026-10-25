"""Paper-trade list v2: the strategy chosen after Studies 9-10, on the rule-based 71-stock universe.

Rule (frozen):
  * Universe: data/universe_2025-09.csv minus ETF leaks (71 stocks).
  * On the latest quote date, for each stock: monthly expiry with DTE closest to 30 in [20, 50]; ATM IV;
    20-day realized vol. Rank by IV/RV20; keep the TOP THIRD.
  * Tradeable legs only: the ~1 sd call and put each need bid > 0, spread <= 50% of mid, volume > 0 that day.
  * At most 3 positions per sector cluster (highest IV/RV kept).
  * Trade: SELL the 1 sd call + 1 sd put (fill assumed at bid - 25% of spread), delta-hedge with stock every Friday.
  * Size: equal margin per position = 43% of the account / 45 (Study 9: ~45 positions per monthly cycle at 43% margin
    use under a 20% worst-week budget), i.e. ~1% of the account each. Contracts = round(target / margin per contract);
    a stock whose single contract needs more than 2x the target is skipped (too big for this account).
    Not every stock qualifies on the first day: as in the backtest, a stock enters on the FIRST day it is in the top
    third with tradeable legs while the expiry is still >= 20 days away, so re-run daily until then to add names.
    Then a portfolio stress check on the ACTUAL positions: worst common 5-day move over the last year (IV +50%)
    plus the largest position taking the worst single-name squeeze ever seen (-722% of its margin). If that
    exceeds 20% of the account, every position is scaled down.

    python scripts/paper_trade_list2.py [account_size]          (default 100000)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hedged_backtest import delta  # noqa: E402
from practical_backtest import naked_req  # noqa: E402
from real_options_backtest import bs, iv  # noqa: E402
from universe_study import CLUSTER_OF, ETF_LEAKS, MAX_PER_CLUSTER, price_file  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "paper_trades"
TARGET_MARGIN = 0.43
EXPECTED_POS = 45     # positions per cycle in Study 9 (top third)
STRESS_BUDGET = 0.20
SQUEEZE = 7.22          # worst single-name 5-day loss seen, as a multiple of that position's margin (Study 9)
SLIP = 0.25


def scan(t):
    op = pd.read_csv(ROOT / "data" / "options" / f"{t}.csv.gz", parse_dates=["date", "expiration"])
    px = pd.read_csv(price_file(t), parse_dates=["date"]).set_index("date")
    d = op.date.max()
    if d not in px.index or len(px.loc[:d]) < 21:
        return None
    S = float(px.close.loc[d])
    rv20 = np.log(px.close.loc[:d]).diff().tail(20).std() * np.sqrt(252)
    today = op[(op.date == d) & (op.ask > 0)].copy()
    today["mid"] = (today.bid + today.ask) / 2
    exps = today.expiration.unique()
    dte = np.array([(e - d).days for e in exps])
    ok = (dte >= 20) & (dte <= 50)
    if not ok.any():
        return None
    e = exps[ok][np.argmin(np.abs(dte[ok] - 30))]
    T = (e - d).days / 365
    ch = today[today.expiration == e]
    calls, puts = ch[ch.right == "C"].set_index("strike").sort_index(), ch[ch.right == "P"].set_index("strike").sort_index()
    com = calls.index.intersection(puts.index)
    if len(com) < 3:
        return None
    katm = com[np.argmin(np.abs(com - S))]
    sig = np.nanmean([iv(calls.loc[katm, "mid"], S, katm, T, True), iv(puts.loc[katm, "mid"], S, katm, T, False)])
    if not np.isfinite(sig):
        return None
    near = lambda b, x: b.index[np.argmin(np.abs(b.index - x))]  # noqa: E731
    kc, kp = near(calls, S * np.exp(sig * np.sqrt(T))), near(puts, S * np.exp(-sig * np.sqrt(T)))
    c, p = calls.loc[kc], puts.loc[kp]
    liq = lambda q: q.bid > 0 and q.mid > 0 and (q.ask - q.bid) / q.mid <= 0.5 and q.volume > 0  # noqa: E731
    row = dict(ticker=t, asof=d.date(), expiry=pd.Timestamp(e).date(), dte=(e - d).days, spot=round(S, 2),
               atm_iv=round(sig, 3), rv20=round(rv20, 3), iv_rv=round(sig / rv20, 2), cluster=CLUSTER_OF.get(t, "-"),
               tradeable=bool(kc > katm > kp and liq(c) and liq(p)))
    if not row["tradeable"]:
        return row
    cfill = c.bid - SLIP * (c.ask - c.bid)
    pfill = p.bid - SLIP * (p.ask - p.bid)
    margin = max(naked_req(cfill, S, kc, True, .2, .1), naked_req(pfill, S, kp, False, .2, .1)) + min(cfill, pfill)
    pos_delta = -(delta(S, kc, T, sig, True) + delta(S, kp, T, sig, False))
    return dict(row, call_strike=kc, call_bid=float(c.bid), call_ask=float(c.ask), put_strike=kp,
                put_bid=float(p.bid), put_ask=float(p.ask), credit_ps=round(cfill + pfill, 3),
                margin_ps=round(margin, 3), hedge_shares_per_contract=round(-pos_delta * 100, 1), T=T, sig=sig)


def stress_dollars(pos):
    """Portfolio $ P&L under each historical 5-day move over the last year (IV +50%), actual sizes."""
    series = []
    for r in pos.itertuples():
        px = pd.read_csv(price_file(r.ticker), parse_dates=["date"]).set_index("date").close
        h = np.log(px).diff(5).dropna().iloc[-250:]
        S1 = r.spot * np.exp(h.to_numpy())
        Tn = max(r.T - 5 / 365, 1 / 365)
        pnl_ps = (r.credit_ps - bs(S1, r.call_strike, Tn, r.sig * 1.5, True) - bs(S1, r.put_strike, Tn, r.sig * 1.5, False)
                  + (r.hedge_shares_per_contract / 100) * (S1 - r.spot))
        series.append(pd.Series(pnl_ps * 100 * r.contracts, index=h.index, name=r.ticker))
    L = pd.concat(series, axis=1)
    L = L[L.notna().mean(axis=1) >= 0.9].fillna(0)
    return L.sum(axis=1)


def main():
    acct = float(sys.argv[1]) if len(sys.argv) > 1 else 100_000
    uni = pd.read_csv(ROOT / "data" / "universe_2025-09.csv")
    syms = [s for s in uni.symbol if s not in ETF_LEAKS and (ROOT / "data" / "options" / f"{s}.csv.gz").exists()]
    df = pd.DataFrame([r for r in (scan(s) for s in syms) if r])
    df = df.sort_values("iv_rv", ascending=False).reset_index(drop=True)
    cutoff = df.iv_rv.quantile(2 / 3)
    top = df[df.iv_rv >= cutoff]
    pos = top[top.tradeable].copy()
    pos["rank_in_cluster"] = pos.groupby("cluster").cumcount()
    capped = pos[(pos.cluster != "-") & (pos.rank_in_cluster >= MAX_PER_CLUSTER)]
    pos = pos[(pos.cluster == "-") | (pos.rank_in_cluster < MAX_PER_CLUSTER)].copy()
    per_pos = TARGET_MARGIN * acct / EXPECTED_POS
    size = lambda f: np.round(f * per_pos / (pos.margin_ps * 100)).astype(int)  # noqa: E731
    pos["contracts"] = size(1.0)
    too_big = pos[pos.contracts == 0]
    pos = pos[pos.contracts > 0].copy()
    scale = 1.0
    for _ in range(20):
        common_worst = -stress_dollars(pos).min()
        squeeze = SQUEEZE * (pos.margin_ps * 100 * pos.contracts).max()
        total = common_worst + squeeze
        if total <= STRESS_BUDGET * acct:
            break
        scale *= 0.9
        pos["contracts"] = size(scale)
        pos = pos[pos.contracts > 0].copy()
    pos["margin"] = (pos.margin_ps * 100 * pos.contracts).round(0)
    pos["credit"] = (pos.credit_ps * 100 * pos.contracts).round(0)
    pos["hedge_shares"] = (pos.hedge_shares_per_contract * pos.contracts).round(0)
    asof = str(df["asof"].max())
    OUT.mkdir(parents=True, exist_ok=True)
    keep = ["ticker", "cluster", "expiry", "dte", "spot", "iv_rv", "call_strike", "call_bid", "call_ask", "put_strike",
            "put_bid", "put_ask", "contracts", "credit", "margin", "hedge_shares"]
    pos[keep].to_csv(OUT / f"{asof}_list_v2.csv", index=False)
    L = [f"# Paper-trade list v2, prices as of {asof} (paper account ${acct:,.0f})", "",
         "Rule: 71-stock rule-based universe; top third by IV / 20-day realized vol; sell ~1 sd call + put (~30 days); "
         "max 3 per sector; hedge with stock every Friday; ~43% of the account in margin, capped by a 20% stress loss. "
         "Paper only. No real orders.", "",
         "| # | stock | sector | expiry | spot | IV/RV | sell call | sell put | contracts | credit | margin | hedge shares |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(pos.itertuples(), 1):
        L.append(f"| {i} | {r.ticker} | {r.cluster} | {r.expiry} | {r.spot} | {r.iv_rv} | {r.call_strike:g} @ {r.call_bid:.2f}/{r.call_ask:.2f} | "
                 f"{r.put_strike:g} @ {r.put_bid:.2f}/{r.put_ask:.2f} | {r.contracts} | ${r.credit:,.0f} | ${r.margin:,.0f} | {r.hedge_shares:+.0f} |")
    L += ["", f"Positions: {len(pos)}. Credit collected: ${pos.credit.sum():,.0f}. Margin used: ${pos.margin.sum():,.0f} "
          f"({pos.margin.sum() / acct:.0%} of account). Idle cash (earns interest): ${acct - pos.margin.sum():,.0f}.",
          f"Stress check: worst 5-day move of the last year on these exact positions ${-common_worst:,.0f}, plus a "
          f"worst-ever squeeze on the largest position ${-squeeze:,.0f} = ${-(common_worst + squeeze):,.0f} "
          f"({(common_worst + squeeze) / acct:.0%} of the account; budget 20%). Sizes scaled by {scale:.2f}.", "",
          f"Top third cutoff: IV/RV >= {cutoff:.2f}. Skipped in the top third: "
          + ", ".join(f"{r.ticker} (legs not tradeable)" for r in top[~top.tradeable].itertuples())
          + ("; " + ", ".join(f"{r.ticker} (sector cap)" for r in capped.itertuples()) if len(capped) else "")
          + ("; " + ", ".join(f"{r.ticker} (1 contract needs ${r.margin_ps * 100:,.0f} margin, > 2x the "
                                 f"${per_pos:,.0f} target)" for r in too_big.itertuples()) if len(too_big) else ""), "",
          f"Target margin per position ${per_pos:,.0f} (~1% of the account). The backtest averaged ~45 positions per "
          "cycle; names are added on the first day they qualify while the expiry is >= 20 days out, so this list grows "
          "over the coming days. Re-run daily until then.", "",
          "Fill prices shown as bid/ask. Assumed fill = bid - 25% of the spread. Hedge shares: + = hold shares, - = short.",
          "Hold to expiry; re-set hedge shares each Friday close."]
    (OUT / f"{asof}_list_v2.md").write_text("\n".join(L) + "\n")
    led = OUT / "paper_ledger.csv"
    old = pd.read_csv(led) if led.exists() else pd.DataFrame()
    if len(old):
        old.loc[old.status == "open", "status"] = "superseded (never entered; replaced by list v2)"
        if "list" in old:
            old = old[~((old["list"] == "v2") & (old["asof"].astype(str) == asof))]   # re-run same day replaces
    new = pos.assign(asof=asof, status="planned", hedge_pnl=0.0, settled_pnl=np.nan, list="v2")[
        ["list", "asof", "ticker", "expiry", "spot", "call_strike", "call_bid", "put_strike", "put_bid", "contracts",
         "credit", "margin", "hedge_shares", "hedge_pnl", "settled_pnl", "status"]]
    pd.concat([old, new], ignore_index=True).to_csv(led, index=False)
    print("\n".join(L))


if __name__ == "__main__":
    main()
