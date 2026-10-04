"""Real Polymarket data -> canonical tables (see schema.py).

UNTESTED IN THIS SESSION: the cloud environment's network policy blocks Polymarket hosts.
Run locally (or after allowing gamma-api.polymarket.com, data-api.polymarket.com,
clob.polymarket.com) and check the first pages by hand before trusting it.

Sources
    Gamma API   https://gamma-api.polymarket.com/markets   market metadata, outcome prices
    Data API    https://data-api.polymarket.com/trades     fills with the trader's proxyWallet

Known limits / leakage traps handled here
    * Fetch OPEN markets as well as closed ones. Using only markets that later resolved cleanly
      is survivorship bias (you can't know that at decision time). Open markets get outcome NaN.
    * outcome 0.5 = 50/50 (UMA "unknown") resolution, which redeems at $0.50.
    * Two-outcome non-Yes/No markets (Team A vs Team B): outcome index 0 is treated as "YES".
    * The Data API paginates with an offset cap. Very active markets can be truncated.
      For complete history use the Goldsky orderbook subgraph (orderFilledEvents) instead.
    * takerOnly=true: one row per taker fill, so a fill isn't counted twice (maker + taker).
"""
from __future__ import annotations

import json
import time
from datetime import datetime

import numpy as np
import pandas as pd
import requests

GAMMA = "https://gamma-api.polymarket.com"
DATA = "https://data-api.polymarket.com"
S = requests.Session()
S.headers["User-Agent"] = "pmct-research/0.1"


def _get(url, params, tries=5):
    for i in range(tries):
        r = S.get(url, params=params, timeout=30)
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(2 ** i)
            continue
        r.raise_for_status()
        return r.json()
    r.raise_for_status()


def _ts(x):
    if not x:
        return np.nan
    return datetime.fromisoformat(str(x).replace("Z", "+00:00")).timestamp()


def fetch_markets(closed: bool, max_pages: int = 400, min_volume: float = 1000.0) -> pd.DataFrame:
    rows, off = [], 0
    for _ in range(max_pages):
        page = _get(f"{GAMMA}/markets", dict(closed=str(closed).lower(), limit=500, offset=off))
        if not page:
            break
        for m in page:
            try:
                outs = json.loads(m.get("outcomes") or "[]")
                prices = [float(p) for p in json.loads(m.get("outcomePrices") or "[]")]
            except (ValueError, TypeError):
                continue
            if len(outs) != 2 or float(m.get("volumeNum") or 0) < min_volume:
                continue
            outcome = np.nan
            if closed and len(prices) == 2:
                p0 = prices[0]
                outcome = 1.0 if p0 > 0.99 else 0.0 if p0 < 0.01 else 0.5 if abs(p0 - 0.5) < 0.01 else np.nan
            tags = [t.get("label", "") for e in (m.get("events") or []) for t in (e.get("tags") or [])]
            rows.append(dict(condition_id=m["conditionId"], question=m.get("question"),
                             created_ts=_ts(m.get("startDate") or m.get("createdAt")),
                             end_date_ts=_ts(m.get("endDate")), closed_ts=_ts(m.get("closedTime")),
                             outcome=outcome, category=_category(m.get("category"), tags),
                             neg_risk=bool(m.get("negRisk"))))
        off += len(page)
    return pd.DataFrame(rows)


def _category(cat, tags) -> str:
    """Map to the fee-schedule categories in strategy.PM_FEE_RATES (best effort)."""
    text = " ".join([str(cat or "")] + list(tags)).lower()
    for key, words in [("crypto", ["crypto", "bitcoin", "ethereum"]), ("sports", ["sport", "nba", "nfl", "soccer"]),
                       ("geopolitics", ["geopolitic", "world"]), ("politics", ["politic", "election"]),
                       ("finance", ["finance", "stock", "fed"]), ("economics", ["econom"]),
                       ("tech", ["tech", "ai"]), ("culture", ["culture", "pop", "music", "movie"]),
                       ("weather", ["weather"]), ("mentions", ["mention"])]:
        if any(w in text for w in words):
            return key
    return "other"


def fetch_trades(condition_id: str, max_pages: int = 40) -> pd.DataFrame:
    rows, off = [], 0
    for _ in range(max_pages):
        try:
            page = _get(f"{DATA}/trades", dict(market=condition_id, limit=500, offset=off, takerOnly="true"))
        except requests.HTTPError:
            break   # offset cap reached
        if not page:
            break
        rows.extend(page)
        off += len(page)
        if len(page) < 500:
            break
    return pd.DataFrame(rows)


def to_canonical(mk: pd.DataFrame, raw_trades: dict[str, pd.DataFrame]):
    """Build (markets, trades, wallet_map) in the canonical schema."""
    mk = mk.reset_index(drop=True).copy()
    mk["market_id"] = np.arange(len(mk))
    cid2id = dict(zip(mk.condition_id, mk.market_id))
    parts = []
    for cid, t in raw_trades.items():
        if t is None or t.empty:
            continue
        t = t.copy()
        price = t["price"].astype(float)
        size = t["size"].astype(float)
        idx = t["outcomeIndex"].astype(int)
        buy = t["side"].str.upper().eq("BUY")
        d = np.where(idx == 0, np.where(buy, 1, -1), np.where(buy, -1, 1)).astype(np.int8)
        yes_px = np.where(idx == 0, price, 1 - price)
        parts.append(pd.DataFrame(dict(ts=t["timestamp"].astype(np.int64), market_id=cid2id[cid],
                                       wallet_addr=t["proxyWallet"].str.lower(), dir=d, yes_price=yes_px,
                                       usd=size * price, shares=size)))
    tr = pd.concat(parts, ignore_index=True)
    tr = tr[(tr.yes_price > 0) & (tr.yes_price < 1)]
    codes, uniq = pd.factorize(tr["wallet_addr"])
    tr["wallet"] = codes
    wallet_map = pd.DataFrame(dict(wallet=np.arange(len(uniq)), address=uniq))
    last = tr.groupby("market_id").ts.max()
    mk = mk[mk.market_id.isin(last.index)].copy()
    resolved = mk["closed_ts"].where(mk["outcome"].notna(), np.inf)
    mk["end_ts"] = np.minimum(last.reindex(mk.market_id).to_numpy() + 1,
                              np.where(np.isfinite(resolved), resolved, np.inf)).astype("int64")
    mk["resolved_ts"] = np.where(np.isfinite(resolved), np.maximum(resolved, mk["end_ts"]),
                                 np.iinfo(np.int64).max // 4).astype("int64")
    mk["created_ts"] = mk["created_ts"].fillna(tr.groupby("market_id").ts.min().reindex(mk.market_id)).astype("int64")
    tr = tr.merge(mk[["market_id", "end_ts"]], on="market_id")
    tr = tr[tr.ts < tr.end_ts].drop(columns=["end_ts", "wallet_addr"])
    markets = mk[["market_id", "created_ts", "end_ts", "resolved_ts", "outcome", "category", "condition_id", "question"]]
    return markets, tr.sort_values(["ts", "market_id", "wallet"], kind="mergesort").reset_index(drop=True), wallet_map
