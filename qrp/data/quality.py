"""Data quality checks. They flag problems; they never silently 'fix' prices."""
from __future__ import annotations

import numpy as np
import pandas as pd


def check_bars(df: pd.DataFrame, max_gap_bdays: int = 5, jump: float = 0.4) -> list[str]:
    issues: list[str] = []
    if df.empty:
        return ["no data"]
    if df["date"].duplicated().any():
        issues.append(f"{int(df['date'].duplicated().sum())} duplicate dates")
    px = df[["open", "high", "low", "close"]]
    if (px <= 0).any().any():
        issues.append("non-positive prices")
    bad_hl = (df["high"] + 1e-9 < px.max(axis=1)) | (df["low"] - 1e-9 > px.min(axis=1))
    if bad_hl.any():
        issues.append(f"{int(bad_hl.sum())} bars where high/low do not bracket open/close")
    if (df["volume"] < 0).any():
        issues.append("negative volume")
    zero_vol = int((df["volume"] == 0).sum())
    if zero_vol > 0.01 * len(df):
        issues.append(f"{zero_vol} zero-volume days")
    d = df["date"].to_numpy().astype("datetime64[D]")
    if len(d) > 1:
        gaps = np.busday_count(d[:-1], d[1:]) - 1
        big = np.flatnonzero(gaps > max_gap_bdays)
        if len(big):
            issues.append(f"{len(big)} gaps longer than {max_gap_bdays} business days "
                          f"(first after {pd.Timestamp(d[big[0]]).date()})")
    r = df["close"].pct_change().abs()
    n_jump = int((r > jump).sum())
    if n_jump:
        first = df.loc[r > jump, "date"].iloc[0].date()
        issues.append(f"{n_jump} one-day moves over {jump:.0%} (first {first}): check split adjustment")
    return issues
