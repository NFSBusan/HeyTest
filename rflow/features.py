"""Point-in-time features, retail scores and forward targets.

Panel schema (one row per ticker-day; real data from fetch.py, simulated from sim.py):
    day (int trading-day index), ticker, close, ret, volume, mentions, offex_share,
    call_vol, put_vol, iv30, call_skew (optional), log_mcap, short_int
Missing columns are allowed (real data rarely has everything); features built from them are NaN.

Rule: a feature on day t uses only rows with day <= t (all rolling windows end at t).
Targets look forward and are named y_*. The model only trains on a target once its
window has fully passed (see model.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

H_ATTN = 5      # horizon for "attention will rise"
H_RET = 5       # horizon for the forward return
H_VOL = 10      # realized-vol window used to judge whether implied vol was rich

FEATURES = ["log_ment", "ment_z20", "ment_d1", "ment_d5", "ment_accel", "ment_xs",
            "offex_share", "offex_z20", "turn_ab", "turn_ab5", "ret1", "ret5", "ret20", "rv20",
            "iv30", "ivrv", "iv_chg5", "cp_ratio", "opt_turn", "call_skew",
            "log_price", "log_mcap", "short_int"]


def _roll(g, col, w, fn):
    return g[col].transform(lambda s: getattr(s.rolling(w, min_periods=max(3, w // 2)), fn)())


def build(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.sort_values(["ticker", "day"], kind="mergesort").reset_index(drop=True).copy()
    for c in ["mentions", "offex_share", "call_vol", "put_vol", "iv30", "call_skew", "log_mcap", "short_int"]:
        if c not in df:
            df[c] = np.nan
    if "ret" not in df:
        df["ret"] = df.groupby("ticker")["close"].transform(lambda s: np.log(s).diff())
    g = df.groupby("ticker", sort=False)

    df["log_ment"] = np.log1p(df["mentions"])
    m20, s20 = _roll(g, "log_ment", 20, "mean"), _roll(g, "log_ment", 20, "std")
    df["ment_z20"] = (df["log_ment"] - m20) / (s20 + 0.25)
    df["ment_d1"] = g["log_ment"].diff(1)
    df["ment_d5"] = g["log_ment"].diff(5)
    df["ment_accel"] = df["ment_d1"] - df.groupby("ticker")["ment_d1"].shift(1)
    df["ment_xs"] = df.groupby("day")["log_ment"].rank(pct=True)

    o20, os20 = _roll(g, "offex_share", 20, "mean"), _roll(g, "offex_share", 20, "std")
    df["offex_z20"] = (df["offex_share"] - o20) / (os20 + 0.01)

    df["log_vol"] = np.log(df["volume"].clip(lower=1))
    g = df.groupby("ticker", sort=False)
    df["turn_ab"] = df["log_vol"] - _roll(g, "log_vol", 20, "mean")
    df["turn_ab5"] = g["turn_ab"].transform(lambda s: s.rolling(5, min_periods=3).mean())

    df["ret1"] = df["ret"]
    df["ret5"] = g["ret"].transform(lambda s: s.rolling(5, min_periods=5).sum())
    df["ret20"] = g["ret"].transform(lambda s: s.rolling(20, min_periods=20).sum())
    df["rv20"] = g["ret"].transform(lambda s: s.rolling(20, min_periods=10).std()) * np.sqrt(252)
    df["ivrv"] = np.log(df["iv30"] / df["rv20"])
    df["iv_chg5"] = np.log(df["iv30"]).groupby(df["ticker"]).diff(5)
    df["cp_ratio"] = np.log((df["call_vol"] + 1) / (df["put_vol"] + 1))
    df["opt_turn"] = np.log((df["call_vol"] + df["put_vol"] + 1) / df["volume"].clip(lower=1))
    df["log_price"] = np.log(df["close"])

    # --- retail scores (cross-sectional ranks per day; heuristics usable without a trained model)
    def xs(c):
        return df.groupby("day")[c].rank(pct=True)
    now_parts = [xs(c) for c in ["log_ment", "offex_share", "turn_ab5", "cp_ratio", "opt_turn"]]
    df["retail_now"] = pd.concat(now_parts, axis=1).mean(axis=1, skipna=True)
    rise_parts = [xs(c) for c in ["ment_d5", "ment_accel", "offex_z20", "turn_ab"]]
    df["retail_rising_heur"] = pd.concat(rise_parts, axis=1).mean(axis=1, skipna=True)
    # structural "meme-prone" score: cheap, small, heavily shorted
    df["meme_prone"] = pd.concat([1 - xs("log_price"), 1 - xs("log_mcap"), xs("short_int")], axis=1).mean(axis=1)
    return add_targets(df)


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("ticker", sort=False)
    ma3 = g["log_ment"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    fut = g["log_ment"].transform(lambda s: s[::-1].rolling(H_ATTN, min_periods=H_ATTN).mean()[::-1].shift(-1))
    df["y_attn"] = fut - ma3                                   # future mean attention vs now
    df["y_ret"] = g["ret"].transform(lambda s: s[::-1].rolling(H_RET, min_periods=H_RET).sum()[::-1].shift(-1))
    fut_rv = g["ret"].transform(lambda s: s[::-1].rolling(H_VOL, min_periods=H_VOL).std()[::-1].shift(-1))
    df["y_vrp"] = np.log(df["iv30"] / (fut_rv * np.sqrt(252)))  # >0: options were rich
    return df
