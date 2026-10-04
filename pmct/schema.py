"""Canonical tables shared by the real-data loader and the simulator.

Every backtest component only sees these two tables, so the same code runs on
simulated and real Polymarket data.

markets
    market_id    int     internal id
    created_ts   int64   unix seconds
    end_ts       int64   scheduled/actual trading close (no trades at or after this)
    resolved_ts  int64   time the outcome became public (UMA settlement / closedTime)
    outcome      float   1.0 = YES won, 0.0 = NO won, NaN = unresolved or invalid
    category     str

trades  (one row per fill, from the perspective of `wallet`)
    ts           int64   unix seconds
    market_id    int
    wallet       int     internal id
    dir          int8    +1 = gained YES exposure (buy YES / sell NO)
                         -1 = gained NO exposure  (buy NO  / sell YES)
    yes_price    float   execution price in YES terms (buy NO at q -> 1 - q)
    usd          float   notional paid or received
    shares       float

Edge of a trade versus holding to resolution, per share:
    e = dir * (outcome - yes_price)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DAY = 86_400

MARKET_COLS = ["market_id", "created_ts", "end_ts", "resolved_ts", "outcome", "category"]
TRADE_COLS = ["ts", "market_id", "wallet", "dir", "yes_price", "usd", "shares"]


def validate(markets: pd.DataFrame, trades: pd.DataFrame) -> None:
    missing_m = set(MARKET_COLS) - set(markets.columns)
    missing_t = set(TRADE_COLS) - set(trades.columns)
    if missing_m or missing_t:
        raise ValueError(f"missing columns: markets={missing_m} trades={missing_t}")
    if not markets["market_id"].is_unique:
        raise ValueError("market_id must be unique")
    if not set(np.unique(trades["dir"])) <= {-1, 1}:
        raise ValueError("dir must be +1/-1")
    if ((trades["yes_price"] <= 0) | (trades["yes_price"] >= 1)).any():
        raise ValueError("yes_price must be strictly inside (0, 1)")
    m = trades.merge(markets[["market_id", "end_ts", "resolved_ts"]], on="market_id", how="left")
    if m["end_ts"].isna().any():
        raise ValueError("trades reference unknown markets")
    if (m["ts"] >= m["end_ts"]).any():
        raise ValueError("trades at/after market end_ts")
    res = markets.dropna(subset=["outcome"])
    if (res["resolved_ts"] < res["end_ts"]).any():
        raise ValueError("resolved_ts must be >= end_ts")


def sort_trades(trades: pd.DataFrame) -> pd.DataFrame:
    return trades.sort_values(["ts", "market_id", "wallet"], kind="mergesort").reset_index(drop=True)
