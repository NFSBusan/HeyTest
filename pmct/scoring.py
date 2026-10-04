"""Point-in-time wallet scores.

A trade contributes to its wallet's record only from the moment its result is knowable
(`avail_ts`), and only for decisions made strictly after that. This is the main leakage
control: we never use a wallet's *final* PnL (e.g. today's leaderboard) to decide what to do
in the past.

Bases (what "bad" means), per share:
    res      e = dir * (outcome - yes_price)          known at market resolution.
             Includes the spread they paid, so it flags high-volume noise traders too.
    res_mid  e = dir * (outcome - mid_at_trade)        known at resolution. Direction skill only.
    mo<H>    e = dir * (mid(ts + H) - mid_at_trade)    "markout", known at ts + H seconds.
             Much faster feedback than waiting for resolution.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BASES = {"res": None, "res_mid": None, "mo1h": 3600, "mo6h": 6 * 3600, "mo1d": 86400}


def trade_records(markets: pd.DataFrame, trades: pd.DataFrame, basis: str, book=None) -> pd.DataFrame:
    """One scored row per trade whose result is defined, stamped with `avail_ts`."""
    if basis in ("res", "res_mid"):
        res = markets.dropna(subset=["outcome"])[["market_id", "outcome", "resolved_ts"]]
        r = trades.merge(res, on="market_id", how="inner")
        ref = r["yes_price"] if basis == "res" else (r["yes_price"] - r["dir"] * book.assumed_half_spread)
        r["e"] = r["dir"] * (r["outcome"] - ref)
        r["avail_ts"] = r["resolved_ts"]
    else:
        H = BASES[basis]
        r = trades.merge(markets[["market_id", "end_ts"]], on="market_id", how="inner")
        r["avail_ts"] = r["ts"] + H
        r = r[r["avail_ts"] < r["end_ts"]]
        m0 = r["yes_price"] - r["dir"] * book.assumed_half_spread
        m1 = book.at(r["market_id"].to_numpy(), r["avail_ts"].to_numpy())
        r["e"] = r["dir"] * (m1 - m0.to_numpy())
    r["pnl"] = r["e"] * r["shares"]
    return r[["wallet", "avail_ts", "market_id", "e", "pnl", "usd"]].dropna(subset=["e"])


def wallet_history(markets: pd.DataFrame, trades: pd.DataFrame, basis: str = "res", book=None) -> pd.DataFrame:
    """Cumulative per-wallet stats, one row per (wallet, avail_ts) step, sorted by avail_ts."""
    r = trade_records(markets, trades, basis, book)
    r["e2"] = r["e"] ** 2
    g = (r.groupby(["wallet", "avail_ts"], sort=False)
           .agg(n=("e", "size"), se=("e", "sum"), se2=("e2", "sum"), pnl=("pnl", "sum"),
                usd=("usd", "sum"), nm=("market_id", "nunique"))
           .reset_index()
           .sort_values(["wallet", "avail_ts"], kind="mergesort"))
    for c in ["n", "se", "se2", "pnl", "usd", "nm"]:
        g["c_" + c] = g.groupby("wallet", sort=False)[c].cumsum()
    # c_nm is exact for resolution bases (a market resolves once); for markouts it counts
    # (market, time) steps, i.e. an upper bound on distinct markets.
    out = g[["wallet", "avail_ts"] + [f"c_{c}" for c in ["n", "se", "se2", "pnl", "usd", "nm"]]]
    return out.sort_values("avail_ts", kind="mergesort").reset_index(drop=True)


def scores_at(hist: pd.DataFrame, queries: pd.DataFrame, ts_col: str = "ts") -> pd.DataFrame:
    """For each query row (wallet, ts_col), the wallet's stats from records with
    avail_ts < query time (strict). Preserves query order. No history -> zeros."""
    q = queries[["wallet", ts_col]].copy()
    q["_ord"] = np.arange(len(q))
    q = q.sort_values(ts_col, kind="mergesort")
    m = pd.merge_asof(q, hist, left_on=ts_col, right_on="avail_ts", by="wallet",
                      allow_exact_matches=False, direction="backward")
    m = m.sort_values("_ord").reset_index(drop=True)
    cols = ["c_n", "c_se", "c_se2", "c_pnl", "c_usd", "c_nm"]
    m[cols] = m[cols].fillna(0.0)
    n = m["c_n"].to_numpy()
    mean = np.divide(m["c_se"].to_numpy(), n, out=np.zeros(len(m)), where=n > 0)
    var = np.divide(m["c_se2"].to_numpy(), n, out=np.zeros(len(m)), where=n > 0) - mean ** 2
    sd = np.sqrt(np.maximum(var, 1e-6))
    return pd.DataFrame({
        "n_res": n,
        "n_mkts": m["c_nm"].to_numpy(),
        "pnl": m["c_pnl"].to_numpy(),
        "roi": m["c_pnl"].to_numpy() / (m["c_usd"].to_numpy() + 500.0),   # shrunk ROI
        "edge": m["c_se"].to_numpy() / (n + 20.0),                          # shrunk per-share edge
        "tstat": np.where(n > 1, mean / (sd / np.sqrt(np.maximum(n, 1))), 0.0),
    })
