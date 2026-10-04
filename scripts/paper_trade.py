"""Forward ("front") test on live Polymarket data. PAPER ONLY: this never places an order.

    python scripts/paper_trade.py run --minutes 600      # watch live trades, log intended maker orders
    python scripts/paper_trade.py reconcile              # mark which paper orders would have filled / won

How it works
    1. Wallet scores come from data/*.parquet (run scripts/fetch_data.py first), using only
       markets that have already resolved. Refresh the history daily.
    2. Poll the Data API for new taker fills. When a flagged wallet trades, read the CLOB book
       for the OPPOSITE token and log a maker buy at (best bid), sized by fractional Kelly.
    3. `reconcile` checks later prints: a paper order fills only if a later print trades strictly
       through our price within the TTL. Then it marks the position at resolution.

This is the honest forward test: decisions are logged before the outcome exists.
Before any real trading, read "Before going live" in RESEARCH_LOG.md (jurisdiction rules,
fees, risk limits). Real orders would go through the official py-clob-client; that is
deliberately not wired up here.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pmct import fetch  # noqa: E402
from pmct.scoring import scores_at, wallet_history  # noqa: E402
from pmct.strategy import MidBook, _round_down  # noqa: E402
from scripts.final_test import FROZEN  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "results" / "paper_ledger.csv"
CLOB = "https://clob.polymarket.com"
CAPITAL = 10_000.0


def flagged_wallets(cfg=FROZEN) -> tuple[set[str], float]:
    """Wallets that are 'bad' as of now, plus the calibrated edge for sizing."""
    markets = pd.read_parquet(ROOT / "data" / "markets.parquet")
    trades = pd.read_parquet(ROOT / "data" / "trades.parquet")
    wallets = pd.read_parquet(ROOT / "data" / "wallets.parquet")
    book = MidBook(trades, cfg.assumed_half_spread)
    hist = wallet_history(markets, trades, cfg.basis, book)
    now = int(time.time())
    q = pd.DataFrame(dict(wallet=wallets.wallet, ts=now))
    sc = scores_at(hist, q)
    bad = (sc.n_res >= cfg.min_res) & (sc.n_mkts >= cfg.min_mkts) & (sc[cfg.score] <= cfg.threshold)
    addrs = set(wallets.address[bad.to_numpy()])
    # Edge estimate for Kelly: reuse the research pipeline's walk-forward estimate at "now".
    from pmct.strategy import Context
    sig = Context(markets, trades).signals(cfg)
    edge = float(sig["edge_hat"].iloc[-1]) if len(sig) else 0.0
    return addrs, edge


def run(minutes: float, poll_s: float = 15.0):
    bad, edge = flagged_wallets()
    print(f"{len(bad)} flagged wallets; calibrated edge {edge:+.4f}/share; config {FROZEN.key()}")
    seen: set[str] = set()
    t_end = time.time() + minutes * 60
    while time.time() < t_end:
        page = fetch._get(f"{fetch.DATA}/trades", dict(limit=500, takerOnly="true")) or []
        for tr in page:
            key = tr.get("transactionHash", "") + str(tr.get("asset")) + str(tr.get("size"))
            if key in seen or str(tr.get("proxyWallet", "")).lower() not in bad:
                continue
            seen.add(key)
            log_intent(tr, edge)
        time.sleep(poll_s)


def log_intent(tr: dict, edge: float):
    """Counter side = the other outcome token of the same market."""
    mk = fetch._get(f"{fetch.GAMMA}/markets", dict(condition_ids=tr["conditionId"]))
    if not mk:
        return
    import json
    tokens = json.loads(mk[0].get("clobTokenIds") or "[]")
    idx = int(tr["outcomeIndex"])
    buy = str(tr["side"]).upper() == "BUY"
    counter_idx = (1 - idx) if buy else idx       # they bought X -> we buy not-X; they sold X -> we buy X
    if len(tokens) != 2:
        return
    ob = fetch._get(f"{CLOB}/book", dict(token_id=tokens[counter_idx]))
    bids = sorted((float(b["price"]) for b in ob.get("bids", [])), reverse=True)
    if not bids:
        return
    limit = float(_round_down(bids[0], FROZEN.tick))
    if not (FROZEN.px_lo <= limit <= FROZEN.px_hi) or edge <= FROZEN.min_edge:
        return
    p = limit + edge
    f = FROZEN.kelly * (p - limit) / (1 - limit)
    stake = min(f * CAPITAL, FROZEN.max_bet_frac * CAPITAL, FROZEN.max_stake_vs_their * float(tr["size"]) * float(tr["price"]))
    if stake / limit < FROZEN.min_shares:
        return
    row = dict(logged_ts=int(time.time()), their_ts=int(tr["timestamp"]), condition_id=tr["conditionId"],
               token_id=tokens[counter_idx], their_wallet=tr["proxyWallet"], limit=limit, stake=round(stake, 2),
               shares=round(stake / limit, 2), ttl_s=FROZEN.maker_ttl_s, status="posted")
    LEDGER.parent.mkdir(exist_ok=True)
    pd.DataFrame([row]).to_csv(LEDGER, mode="a", header=not LEDGER.exists(), index=False)
    print("PAPER ORDER", row, flush=True)


def reconcile():
    led = pd.read_csv(LEDGER)
    out = []
    for r in led.itertuples():
        trs = fetch.fetch_trades(r.condition_id, max_pages=4)
        fill = None
        if not trs.empty:
            t = trs[(trs.asset == str(r.token_id)) & (trs.timestamp > r.logged_ts)
                    & (trs.timestamp <= r.logged_ts + r.ttl_s)]
            through = t[t.price.astype(float) < r.limit - 1e-9]
            fill = int(through.timestamp.min()) if len(through) else None
        mk = fetch._get(f"{fetch.GAMMA}/markets", dict(condition_ids=r.condition_id)) or [{}]
        out.append(dict(**r._asdict(), filled_ts=fill, closed=bool(mk[0].get("closed")),
                        outcome_prices=mk[0].get("outcomePrices"), clob_token_ids=mk[0].get("clobTokenIds")))
    df = pd.DataFrame(out)
    df.to_csv(ROOT / "results" / "paper_reconciled.csv", index=False)
    print(df[["logged_ts", "limit", "stake", "filled_ts", "closed"]].to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "reconcile"])
    ap.add_argument("--minutes", type=float, default=600)
    a = ap.parse_args()
    run(a.minutes) if a.cmd == "run" else reconcile()
