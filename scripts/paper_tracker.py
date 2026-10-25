"""Automatic paper-trading tracker for the v2 rule (see paper_trade_list2.py). Runs each weekday ~3:40pm ET from
IBKR quotes collected by Claude (read-only market data; no orders are ever placed). Two steps per run:

  1. python scripts/paper_tracker.py plan DATE
       reads data/live/DATE/underlyings.json  {SYM: {"last": .., "prior_close": .., "iv": ..}}   (all 71 stocks)
       saves closes to data/prices_live/, ranks by IV / RV20, and writes data/live/DATE/plan.json: the option legs
       to quote: "fill" (planned entries), "new" (top-third names entering now; pick the listed strike nearest
       the target), "mark" (open positions, Fridays and expiry week).
  2. python scripts/paper_tracker.py apply DATE
       reads data/live/DATE/legs.json  {key: {"call": {"strike","bid","ask","volume"}, "put": {...}}}
       fills entries (sell at bid - 25% of spread, $0.65/contract, legs must have bid > 0, spread <= 50% of mid,
       volume > 0), sizes them (~1% of equity in margin each), checks the WHOLE book against the 20% stress
       budget (new entries scaled down / dropped if needed), resets delta hedges on Fridays, settles positions
       after expiry, and writes results/paper_trades/book.csv, account.csv and status.md.

Rule deviations from the backtest, stated: IV is IBKR's 30-day implied vol of the stock (backtest: ATM IV of the
chosen monthly expiry); entry uses 3:40pm live quotes (backtest: next-day closing quotes); hedge reset on Fridays.
"""
from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hedged_backtest import delta  # noqa: E402
from practical_backtest import naked_req  # noqa: E402
from real_options_backtest import bs  # noqa: E402
from universe_study import CLUSTER_OF, ETF_LEAKS, MAX_PER_CLUSTER  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PT = ROOT / "results" / "paper_trades"
LIVE = ROOT / "data" / "live"
PLIVE = ROOT / "data" / "prices_live"
BOOK = PT / "book.csv"
ACCT0 = 100_000.0
TARGET_MARGIN, EXPECTED_POS, STRESS_BUDGET, SQUEEZE = 0.43, 45, 0.20, 7.22
SLIP, COMM, HEDGE_BPS = 0.25, 0.0065, 0.0005
COLS = ["ticker", "cluster", "expiry", "status", "planned", "entry_date", "spot0", "sig0", "call_strike", "put_strike",
        "contracts", "credit_ps", "margin", "hedge_shares", "hedge_spot", "hedge_pnl", "mark_ps", "mark_date",
        "settle_spot", "pnl", "note"]


def universe():
    u = pd.read_csv(ROOT / "data" / "universe_2025-09.csv")
    return [s for s in u.symbol if s not in ETF_LEAKS]


def closes(t):
    parts = []
    for f in (ROOT / "data" / "prices" / f"{t}.csv", ROOT / "data" / "prices_td" / f"{t}.csv", PLIVE / f"{t}.csv"):
        if f.exists():
            parts.append(pd.read_csv(f, parse_dates=["date"])[["date", "close"]])
    if not parts:
        return pd.Series(dtype=float)
    s = pd.concat(parts).drop_duplicates("date", keep="last").set_index("date").close.sort_index()
    return s


def save_close(t, d, px):
    f = PLIVE / f"{t}.csv"
    df = pd.read_csv(f, parse_dates=["date"]) if f.exists() else pd.DataFrame(columns=["date", "close"])
    df = pd.concat([df[df.date != pd.Timestamp(d)], pd.DataFrame({"date": [pd.Timestamp(d)], "close": [px]})])
    PLIVE.mkdir(parents=True, exist_ok=True)
    df.sort_values("date").to_csv(f, index=False, date_format="%Y-%m-%d")


def monthlies(d):
    out = []
    for k in range(4):
        m = (d.month - 1 + k) % 12 + 1
        y = d.year + (d.month - 1 + k) // 12
        first = date(y, m, 1)
        out.append(first + timedelta(days=(4 - first.weekday()) % 7 + 14))
    return out


def pick_expiry(d):
    ok = [e for e in monthlies(d) if 20 <= (e - d).days <= 50]
    return min(ok, key=lambda e: abs((e - d).days - 30)) if ok else None


def load_book():
    if BOOK.exists():
        return pd.read_csv(BOOK)
    v2 = pd.read_csv(PT / "2026-10-23_list_v2.csv")          # seed: list v2 positions, to be filled on the first run
    b = pd.DataFrame({"ticker": v2.ticker, "cluster": v2.cluster, "expiry": v2.expiry, "status": "planned",
                      "planned": v2.contracts, "call_strike": v2.call_strike, "put_strike": v2.put_strike})
    return b.reindex(columns=COLS)


def plan(d):
    D = LIVE / str(d)
    und = json.loads((D / "underlyings.json").read_text())
    syms = universe()
    last_px = {}
    for t, q in und.items():
        if q.get("prior_close"):                       # yesterday's official close replaces yesterday's 3:40pm price
            save_close(t, pd.Timestamp(d) - pd.offsets.BDay(1), q["prior_close"])
        if q.get("last"):
            save_close(t, d, q["last"])
            last_px[t] = q["last"]
    rows = []
    for t in syms:
        q = und.get(t, {})
        c = closes(t)
        c = c[c.index <= pd.Timestamp(d)]
        if not q.get("iv") or not q.get("last") or len(c) < 21:
            continue
        rv = np.log(c).diff().tail(20).std() * np.sqrt(252)
        rows.append(dict(ticker=t, spot=q["last"], iv=q["iv"], rv20=rv, iv_rv=q["iv"] / rv))
    rk = pd.DataFrame(rows).sort_values("iv_rv", ascending=False)
    cutoff = rk.iv_rv.quantile(2 / 3)
    book = load_book()
    legs = []
    for r in book[book.status == "planned"].itertuples():
        legs.append(dict(key=f"{r.ticker}|{r.expiry}", kind="fill", ticker=r.ticker, expiry=r.expiry,
                         spot=last_px.get(r.ticker), call_strike=r.call_strike, put_strike=r.put_strike))
    e = pick_expiry(d)
    if e is not None:
        held = book[(book.expiry == str(e)) & book.status.isin(["planned", "open"])]
        per_cluster = held.cluster.value_counts().to_dict()
        for r in rk[rk.iv_rv >= cutoff].itertuples():
            if ((book.ticker == r.ticker) & (book.expiry == str(e)) & (book.status != "not filled")).any():
                continue
            cl = CLUSTER_OF.get(r.ticker, "-")
            if cl != "-" and per_cluster.get(cl, 0) >= MAX_PER_CLUSTER:
                continue
            T = (e - d).days / 365
            legs.append(dict(key=f"{r.ticker}|{e}", kind="new", ticker=r.ticker, expiry=str(e), spot=r.spot,
                             iv=round(r.iv, 4), iv_rv=round(r.iv_rv, 2),
                             call_target=round(r.spot * np.exp(r.iv * np.sqrt(T)), 2),
                             put_target=round(r.spot * np.exp(-r.iv * np.sqrt(T)), 2)))
            if cl != "-":
                per_cluster[cl] = per_cluster.get(cl, 0) + 1
    friday = d.weekday() == 4
    for r in book[book.status == "open"].itertuples():
        exp = date.fromisoformat(r.expiry)
        if exp < d:
            continue                                   # settles in apply, no quote needed
        if friday or (exp - d).days <= 0:
            legs.append(dict(key=f"{r.ticker}|{r.expiry}", kind="mark", ticker=r.ticker, expiry=r.expiry,
                             spot=last_px.get(r.ticker), call_strike=r.call_strike, put_strike=r.put_strike))
    out = dict(date=str(d), expiry_for_new=str(e) if e else None, iv_rv_cutoff=round(cutoff, 2), legs=legs)
    (D / "plan.json").write_text(json.dumps(out, indent=1, default=float))
    rk.round(3).to_csv(D / "ranking.csv", index=False)
    print(f"{d}: cutoff IV/RV {cutoff:.2f}; legs to quote: "
          + ", ".join(f"{x['kind']}:{x['ticker']}" for x in legs))


def tradeable(q):
    mid = (q["bid"] + q["ask"]) / 2
    vol_ok = q.get("volume") is None or q["volume"] > 0
    return q["bid"] > 0 and mid > 0 and (q["ask"] - q["bid"]) / mid <= 0.5 and vol_ok


def stress(book, spot, d):
    """$ loss of the open book: worst common 5-day move of the last year (IV +50%) + worst squeeze on the biggest."""
    series = []
    op = book[book.status == "open"]
    for r in op.itertuples():
        c = closes(r.ticker)
        h = np.log(c).diff(5).dropna().iloc[-250:]
        S, k = spot[r.ticker], r.contracts * 100
        T = max((date.fromisoformat(r.expiry) - d).days / 365, 1 / 365)
        Tn = max(T - 5 / 365, 1 / 365)
        S1 = S * np.exp(h.to_numpy())
        v0 = bs(S, r.call_strike, T, r.sig0, True) + bs(S, r.put_strike, T, r.sig0, False)
        v1 = bs(S1, r.call_strike, Tn, r.sig0 * 1.5, True) + bs(S1, r.put_strike, Tn, r.sig0 * 1.5, False)
        series.append(pd.Series((v0 - v1) * k + r.hedge_shares * (S1 - S), index=h.index))
    if not series:
        return 0.0
    L = pd.concat(series, axis=1)
    L = L[L.notna().mean(axis=1) >= 0.9].fillna(0)
    return -L.sum(axis=1).min() + SQUEEZE * op.margin.max()


def apply(d):
    D = LIVE / str(d)
    pl = json.loads((D / "plan.json").read_text())
    legs = json.loads((D / "legs.json").read_text()) if (D / "legs.json").exists() else {}
    und = json.loads((D / "underlyings.json").read_text())
    spot = {t: q["last"] for t, q in und.items() if q.get("last")}
    iv = {t: q["iv"] for t, q in und.items() if q.get("iv")}
    book = load_book()
    for c in ["status", "planned", "note", "entry_date", "mark_date", "expiry", "cluster", "ticker"]:
        book[c] = book[c].astype(object)
    acct = pd.read_csv(PT / "account.csv") if (PT / "account.csv").exists() else pd.DataFrame()
    equity = acct.equity.iloc[-1] if len(acct) else ACCT0
    per_pos = TARGET_MARGIN * equity / EXPECTED_POS
    log = []

    # 1. settle expired positions (close on expiry day)
    for i, r in book[book.status == "open"].iterrows():
        exp = date.fromisoformat(r.expiry)
        if exp < d:
            c = closes(r.ticker)
            ST = float(c[c.index <= pd.Timestamp(exp)].iloc[-1])
            hedge = r.hedge_pnl + r.hedge_shares * (ST - r.hedge_spot) - abs(r.hedge_shares) * ST * HEDGE_BPS
            opt = (r.credit_ps - max(ST - r.call_strike, 0) - max(r.put_strike - ST, 0)) * 100 * r.contracts
            book.loc[i, ["status", "settle_spot", "hedge_pnl", "hedge_shares", "pnl"]] = ["settled", ST, hedge, 0, opt + hedge]
            log.append(f"settled {r.ticker} {r.expiry} at {ST:.2f}: P&L ${opt + hedge:,.0f}")

    # 2. entries (planned fills first, then new names by IV/RV rank)
    entries = []
    for L in pl["legs"]:
        if L["kind"] == "mark":
            continue
        q = legs.get(L["key"])
        t = L["ticker"]
        if q is None or t not in spot:
            log.append(f"no quote for {L['key']}: skipped today")
            continue
        S, kc, kp = spot[t], q["call"]["strike"], q["put"]["strike"]
        ok = kc > S > kp and tradeable(q["call"]) and tradeable(q["put"])
        if not ok:
            if L["kind"] == "fill":
                book.loc[(book.ticker == t) & (book.expiry == L["expiry"]), ["status", "note"]] = ["not filled", f"{d}: legs not tradeable"]
            log.append(f"{t}: legs not tradeable today")
            continue
        cf = q["call"]["bid"] - SLIP * (q["call"]["ask"] - q["call"]["bid"]) - COMM
        pf = q["put"]["bid"] - SLIP * (q["put"]["ask"] - q["put"]["bid"]) - COMM
        m = max(naked_req(cf, S, kc, True, .2, .1), naked_req(pf, S, kp, False, .2, .1)) + min(cf, pf)
        T = (date.fromisoformat(L["expiry"]) - d).days / 365
        sig = iv.get(t, L.get("iv"))
        n = int(round(per_pos / (m * 100)))
        if L["kind"] == "fill":
            n = int(book.loc[(book.ticker == t) & (book.expiry == L["expiry"]), "planned"].iloc[0])
        entries.append(dict(ticker=t, cluster=CLUSTER_OF.get(t, "-"), expiry=L["expiry"], status="open", entry_date=str(d),
                            spot0=S, sig0=sig, call_strike=kc, put_strike=kp, contracts=n, credit_ps=cf + pf, m_ps=m,
                            hd=delta(S, kc, T, sig, True) + delta(S, kp, T, sig, False), kind=L["kind"]))
    scale = 1.0
    for _ in range(30):
        trial = book.copy()
        rows = []
        for x in entries:
            n = x["contracts"] if x["kind"] == "fill" else int(round(scale * x["contracts"]))
            if n <= 0:
                if scale == 1.0:
                    log.append(f"{x['ticker']}: 1 contract needs ${x['m_ps'] * 100:,.0f} margin, too big for the "
                               f"${per_pos:,.0f} per-position target")
                continue
            rows.append({**{k: x[k] for k in ["ticker", "cluster", "expiry", "status", "entry_date", "spot0", "sig0",
                                                "call_strike", "put_strike", "credit_ps"]},
                         "contracts": n, "margin": round(x["m_ps"] * 100 * n, 0), "hedge_shares": round(x["hd"] * 100 * n),
                         "hedge_spot": x["spot0"], "hedge_pnl": -abs(x["hd"] * 100 * n) * x["spot0"] * HEDGE_BPS,
                         "mark_ps": x["credit_ps"], "mark_date": str(d), "note": x["kind"]})
        for rw in rows:
            mask = (trial.ticker == rw["ticker"]) & (trial.expiry == rw["expiry"]) & (trial.status == "planned")
            trial = trial[~mask]
        trial = pd.concat([trial, pd.DataFrame(rows)], ignore_index=True)
        loss = stress(trial, {**spot, **{r.ticker: r.spot0 for r in trial.itertuples() if r.ticker not in spot}}, d)
        if loss <= STRESS_BUDGET * equity or not any(x["kind"] == "new" for x in entries) or scale < 0.05:
            break
        scale *= 0.85
    book = trial
    log = [x for x in log if "too big" not in x] + [x for x in dict.fromkeys(log) if "too big" in x][:0] + \
        list(dict.fromkeys(x for x in log if "too big" in x))
    if scale < 1.0:
        log.append(f"new entries scaled to {scale:.2f}x to stay inside the 20% stress budget")
    for rw in rows:
        log.append(f"entered {rw['ticker']} {rw['expiry']}: {rw['contracts']}x  sell {rw['call_strike']:g}C/{rw['put_strike']:g}P, "
                   f"credit ${rw['credit_ps'] * 100 * rw['contracts']:,.0f}, margin ${rw['margin']:,.0f}, hedge {rw['hedge_shares']:+.0f} sh")

    # 3. Friday / expiry-day marks and hedge resets
    for L in pl["legs"]:
        if L["kind"] != "mark" or L["key"] not in legs:
            continue
        q = legs[L["key"]]
        i = book.index[(book.ticker == L["ticker"]) & (book.expiry == L["expiry"]) & (book.status == "open")]
        if not len(i) or L["ticker"] not in spot:
            continue
        i, S = i[0], spot[L["ticker"]]
        r = book.loc[i]
        mid = lambda x: (x["bid"] + x["ask"]) / 2  # noqa: E731
        T = max((date.fromisoformat(r.expiry) - d).days / 365, 0.5 / 365)
        sig = iv.get(r.ticker, r.sig0)
        new = round((delta(S, r.call_strike, T, sig, True) + delta(S, r.put_strike, T, sig, False)) * 100 * r.contracts)
        hp = r.hedge_pnl + r.hedge_shares * (S - r.hedge_spot) - abs(new - r.hedge_shares) * S * HEDGE_BPS
        book.loc[i, ["mark_ps", "mark_date", "hedge_pnl", "hedge_shares", "hedge_spot"]] = [
            mid(q["call"]) + mid(q["put"]), str(d), hp, new, S]

    # 4. account
    op = book[book.status == "open"]
    unreal = float(((op.credit_ps - op.mark_ps) * 100 * op.contracts + op.hedge_pnl
                    + op.hedge_shares * (op.ticker.map(spot).fillna(op.hedge_spot) - op.hedge_spot)).sum())
    realized = float(book.loc[book.status == "settled", "pnl"].sum())
    eq = ACCT0 + realized + unreal
    margin = float(op.margin.sum())
    hedge_margin = float((0.5 * op.hedge_shares.abs() * op.ticker.map(spot).fillna(op.hedge_spot)).sum())
    loss = stress(book, {**spot, **{r.ticker: r.spot0 for r in op.itertuples() if r.ticker not in spot}}, d)
    row = dict(date=str(d), equity=round(eq, 2), realized=round(realized, 2), unrealized=round(unreal, 2),
               option_margin=round(margin), hedge_margin=round(hedge_margin), open_positions=len(op),
               stress_loss=round(loss), stress_pct=round(loss / eq, 3))
    acct = pd.concat([acct[acct.get("date", pd.Series(dtype=str)) != str(d)] if len(acct) else acct, pd.DataFrame([row])])
    acct.to_csv(PT / "account.csv", index=False)
    book.reindex(columns=COLS).to_csv(BOOK, index=False)
    md = [f"# Paper account status, {d}", "", f"Equity ${eq:,.0f} (start $100,000; {eq / ACCT0 - 1:+.2%}). Realized ${realized:,.0f}, "
          f"open P&L ${unreal:,.0f} (options marked at mid on Fridays).",
          f"Open positions {len(op)}; option margin ${margin:,.0f} ({margin / eq:.0%}), stock-hedge margin ${hedge_margin:,.0f}. "
          f"Stress loss ${loss:,.0f} ({loss / eq:.0%} of equity; budget 20%).", "", "Today: " + ("; ".join(log) or "nothing to do."), "",
          "| stock | expiry | entered | contracts | call/put sold | credit | margin | hedge shares | open P&L |", "|---|---|---|---|---|---|---|---|---|"]
    for r in op.sort_values(["expiry", "ticker"]).itertuples():
        s_now = spot.get(r.ticker, r.hedge_spot)
        p = (r.credit_ps - r.mark_ps) * 100 * r.contracts + r.hedge_pnl + r.hedge_shares * (s_now - r.hedge_spot)
        md.append(f"| {r.ticker} | {r.expiry} | {r.entry_date} | {int(r.contracts)} | {r.call_strike:g}C / {r.put_strike:g}P | "
                  f"${r.credit_ps * 100 * r.contracts:,.0f} | ${r.margin:,.0f} | {r.hedge_shares:+.0f} | ${p:,.0f} |")
    st = book[book.status == "settled"]
    if len(st):
        md += ["", "Settled: " + ", ".join(f"{r.ticker} {r.expiry} ${r.pnl:,.0f}" for r in st.itertuples())]
    md += ["", "Paper only. Read-only IBKR market data; no orders placed."]
    (PT / "status.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    cmd, d = sys.argv[1], date.fromisoformat(sys.argv[2])
    {"plan": plan, "apply": apply}[cmd](d)
