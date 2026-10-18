"""Build the monthly paper-trading list for the hedged short-strangle strategy (Study 6 rules, unchanged).

Rule (frozen; same as the backtest):
  * Universe: the small/mid caps in data/options with an IBKR price file (GME, NVDA excluded).
  * As of each ticker's latest quote date, use the monthly expiry with DTE closest to 30 in [20, 50].
  * Qualify if ATM IV / realized vol of the last 20 days >= 1.3, and both 1 sd legs are tradeable
    (bid > 0, spread <= 50% of mid).
  * Trade: SELL the 1 sd call and the 1 sd put (sell at the bid). Delta-hedge with stock once a week.
  * Size: equal margin per trade, total margin <= 50% of the paper account, THEN scaled down so that if
    every position gapped 50% the wrong way at once the loss is <= 25% of the account.
    Reg-T naked margin as in practical_backtest.py.

    python scripts/paper_trade_list.py [account_size]       (default 100000)
Writes results/paper_trades/<asof>_list.csv and .md, and appends new entries to paper_ledger.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hedged_backtest import delta  # noqa: E402
from practical_backtest import naked_req  # noqa: E402
from real_options_backtest import LARGE, iv  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "paper_trades"
BIG_JUMP = {"ASTS", "LUNR", "NBIS", "RKLB", "QBTS", "RIOT"}   # lost in Study 5 (hindsight; flagged, not excluded)


def scan_ticker(t):
    op = pd.read_csv(ROOT / "data" / "options" / f"{t}.csv.gz", parse_dates=["date", "expiration"])
    px = pd.read_csv(ROOT / "data" / "prices" / f"{t}.csv", parse_dates=["date"]).set_index("date")
    d = op.date.max()
    if d not in px.index:
        return None
    S = px.close.loc[d]
    rv20 = np.log(px.close.loc[:d]).diff().tail(20).std() * np.sqrt(252)
    today = op[(op.date == d) & (op.ask > 0)].copy()
    today["mid"] = (today.bid + today.ask) / 2
    exps = today.expiration.unique()
    dte = np.array([(e - d).days for e in exps])
    ok = (dte >= 20) & (dte <= 50)
    if not ok.any():
        return dict(ticker=t, asof=d.date(), status="no expiry in 20-50 days")
    e = exps[ok][np.argmin(np.abs(dte[ok] - 30))]
    T = (e - d).days / 365
    ch = today[today.expiration == e]
    calls, puts = ch[ch.right == "C"].set_index("strike").sort_index(), ch[ch.right == "P"].set_index("strike").sort_index()
    com = calls.index.intersection(puts.index)
    if len(com) < 3:
        return dict(ticker=t, asof=d.date(), status="chain too thin")
    katm = com[np.argmin(np.abs(com - S))]
    sig = np.nanmean([iv(calls.loc[katm, "mid"], S, katm, T, True), iv(puts.loc[katm, "mid"], S, katm, T, False)])
    ratio = sig / rv20
    row = dict(ticker=t, asof=d.date(), expiry=pd.Timestamp(e).date(), dte=(e - d).days, spot=round(S, 2),
               atm_iv=round(sig, 3), rv20=round(rv20, 3), iv_rv=round(ratio, 2), big_jump_flag=t in BIG_JUMP)
    if not np.isfinite(ratio) or ratio < 1.3:
        return dict(row, status="IV not rich enough")
    near = lambda b, x: b.index[np.argmin(np.abs(b.index - x))]  # noqa: E731
    kc, kp = near(calls, S * np.exp(sig * np.sqrt(T))), near(puts, S * np.exp(-sig * np.sqrt(T)))
    c, p = calls.loc[kc], puts.loc[kp]

    def tradeable(x):
        m = (x.bid + x.ask) / 2
        return x.bid > 0 and m > 0 and (x.ask - x.bid) / m <= 0.5
    if not (kc > katm > kp and tradeable(c) and tradeable(p)):
        return dict(row, status="legs not tradeable (wide/zero bid)")
    credit = float(c.bid + p.bid)
    margin = max(naked_req(c.bid, S, kc, True, .2, .1), naked_req(p.bid, S, kp, False, .2, .1)) + min(c.bid, p.bid)
    dlt = -(delta(S, kc, T, sig, True) + delta(S, kp, T, sig, False))     # position delta of short strangle, per share
    hedge = -dlt * 100                                                    # shares to hold per 1 strangle (100x)
    return dict(row, status="TRADE", call_strike=kc, call_bid=float(c.bid), call_ask=float(c.ask),
                put_strike=kp, put_bid=float(p.bid), put_ask=float(p.ask), credit_per_share=round(credit, 2),
                credit_per_contract=round(100 * credit, 0), margin_per_contract=round(100 * margin, 0),
                hedge_shares_per_contract=round(hedge, 1),
                breakeven_low=round(kp - credit, 2), breakeven_high=round(kc + credit, 2),
                loss_if_gap_down30=round(100 * (credit - max(kp - 0.7 * S, 0)) + hedge * (-0.3 * S), 0),
                loss_if_gap_up30=round(100 * (credit - max(1.3 * S - kc, 0)) + hedge * (0.3 * S), 0),
                loss_if_gap_down50=round(100 * (credit - max(kp - 0.5 * S, 0)) + hedge * (-0.5 * S), 0),
                loss_if_gap_up50=round(100 * (credit - max(1.5 * S - kc, 0)) + hedge * (0.5 * S), 0))


def main():
    acct = float(sys.argv[1]) if len(sys.argv) > 1 else 100_000
    tick = sorted(p.name.split(".")[0] for p in (ROOT / "data" / "options").glob("*.csv.gz")
                  if p.name.split(".")[0] not in LARGE and (ROOT / "data" / "prices" / f"{p.name.split('.')[0]}.csv").exists())
    rows = [r for r in (scan_ticker(t) for t in tick) if r]
    df = pd.DataFrame(rows)
    trades = df[df.status == "TRADE"].copy()
    if len(trades):
        per_trade = 0.5 * acct / len(trades)
        trades["contracts"] = (per_trade // trades.margin_per_contract).astype(int).clip(lower=0)
        trades["total_margin"] = trades.contracts * trades.margin_per_contract
        trades["total_credit"] = trades.contracts * trades.credit_per_contract
        trades["hedge_shares"] = (trades.contracts * trades.hedge_shares_per_contract).round(0)
        trades["worst_gap_loss_30pct"] = trades.contracts * trades[["loss_if_gap_down30", "loss_if_gap_up30"]].min(axis=1)
        trades["worst_gap_loss_50pct"] = trades.contracts * trades[["loss_if_gap_down50", "loss_if_gap_up50"]].min(axis=1)
        # Risk cap: if every position gapped 50% the wrong way at once, lose at most 25% of the account.
        worst = -trades.worst_gap_loss_50pct.clip(upper=0).sum()
        scale = min(1.0, 0.25 * acct / worst) if worst > 0 else 1.0
        if scale < 1:
            trades["contracts"] = (trades.contracts * scale).astype(int)
            for c, per in [("total_margin", "margin_per_contract"), ("total_credit", "credit_per_contract")]:
                trades[c] = trades.contracts * trades[per]
            trades["hedge_shares"] = (trades.contracts * trades.hedge_shares_per_contract).round(0)
            trades["worst_gap_loss_30pct"] = trades.contracts * trades[["loss_if_gap_down30", "loss_if_gap_up30"]].min(axis=1)
            trades["worst_gap_loss_50pct"] = trades.contracts * trades[["loss_if_gap_down50", "loss_if_gap_up50"]].min(axis=1)
    asof = str(df["asof"].max())
    OUT.mkdir(parents=True, exist_ok=True)
    df.merge(trades[["ticker", "contracts", "total_margin", "total_credit", "hedge_shares", "worst_gap_loss_30pct", "worst_gap_loss_50pct"]],
             on="ticker", how="left").to_csv(OUT / f"{asof}_list.csv", index=False)

    lines = [f"# Paper-trade list, prices as of {asof} (paper account ${acct:,.0f})", "",
             "Rule: sell the ~1 sd call and put (~30 days) when ATM IV / 20-day realized vol >= 1.3; "
             "hedge with stock once a week. Paper only. No real orders.", ""]
    if len(trades):
        lines += ["| ticker | expiry | spot | IV / RV | sell call | sell put | credit/contract | margin/contract | contracts | "
                  "total credit | hedge shares now | P&L if 30% gap | P&L if 50% gap | prices from | flag |", "|" + "---|" * 15]
        for r in trades.itertuples():
            lines.append(f"| {r.ticker} | {r.expiry} | {r.spot} | {r.iv_rv} | {r.call_strike:g} @ {r.call_bid:.2f} | "
                         f"{r.put_strike:g} @ {r.put_bid:.2f} | ${r.credit_per_contract:,.0f} | ${r.margin_per_contract:,.0f} | "
                         f"{r.contracts} | ${r.total_credit:,.0f} | {r.hedge_shares:+.0f} | ${r.worst_gap_loss_30pct:,.0f} | "
                         f"${r.worst_gap_loss_50pct:,.0f} | {r.asof} | "
                         f"{'big-jump name' if r.big_jump_flag else ''} |")
        lines += ["", f"Total margin used: ${trades.total_margin.sum():,.0f} of ${acct:,.0f}. "
                  f"Total credit collected: ${trades.total_credit.sum():,.0f}. "
                  f"If EVERY position gapped the wrong way at once: 30% gap ${trades.worst_gap_loss_30pct.sum():,.0f}, "
                  f"50% gap ${trades.worst_gap_loss_50pct.sum():,.0f}.",
                  "", "Credits are at the bid before commissions ($0.65 per contract per leg, about $1.30 per strangle). "
                  "Rows priced on an earlier date than the latest close must be re-priced before they count.", ""]
    skipped = df[df.status != "TRADE"]
    lines += ["Not traded this month:", ""] + [f"- {r.ticker}: {r.status}" + (f" (IV/RV {r.iv_rv})" if pd.notna(r.get('iv_rv', np.nan)) else "")
                                               for _, r in skipped.iterrows()]
    lines += ["", "Hedge shares: positive = buy/hold that many shares, negative = short that many. "
              "Re-compute and adjust every Friday close. Close nothing early; positions settle at expiry."]
    (OUT / f"{asof}_list.md").write_text("\n".join(lines) + "\n")

    led = ROOT / "results" / "paper_trades" / "paper_ledger.csv"
    if len(trades):
        new = trades.assign(status="open", hedge_pnl=0.0, settled_pnl=np.nan)[
            ["asof", "ticker", "expiry", "spot", "call_strike", "call_bid", "put_strike", "put_bid", "contracts",
             "total_credit", "total_margin", "hedge_shares", "hedge_pnl", "settled_pnl", "status"]]
        old = pd.read_csv(led) if led.exists() else pd.DataFrame(columns=new.columns)
        old = old[~((old["asof"].astype(str) == asof) & old.ticker.isin(new.ticker))]
        pd.concat([old, new], ignore_index=True).to_csv(led, index=False)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
