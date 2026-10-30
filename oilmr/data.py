"""Load Brent and WTI daily prices from local CSVs.

Supported formats:
- FRED downloads (DCOILBRENTEU.csv, DCOILWTICO.csv): columns observation_date/DATE and the series id.
- Any CSV with a date column and one price column (pass `col`).

These are *spot assessments* (EIA/Refinitiv), not tradeable contracts. They are fine for
asking "does the relationship mean-revert at all?" and not fine for P&L. P&L needs
same-delivery-month futures settlements (ICE Brent, NYMEX CL) with an explicit roll rule.
"""
from pathlib import Path

import numpy as np
import pandas as pd


def load_series(path, col=None):
    df = pd.read_csv(path)
    date_col = next(c for c in df.columns if c.lower() in ("date", "observation_date", "time", "timestamp"))
    col = col or next(c for c in df.columns if c != date_col)
    s = pd.to_numeric(df[col], errors="coerce")  # FRED writes "." for holidays
    s.index = pd.to_datetime(df[date_col])
    return s.dropna().sort_index().rename(Path(path).stem)


def load_pair(brent_path, wti_path):
    """Inner-join on dates both markets printed. Keeps WTI's negative print of 2020-04-20
    (it really happened, and a model has to survive it) but flags it."""
    b, w = load_series(brent_path), load_series(wti_path)
    df = pd.concat({"brent": b, "wti": w}, axis=1, join="inner").dropna()
    df["wti_nonpositive"] = df["wti"] <= 0
    df["spread"] = df["brent"] - df["wti"]
    return df


def log_ok(df):
    """Rows where log prices are defined."""
    return df[(df["brent"] > 0) & (df["wti"] > 0)].assign(
        lb=lambda d: np.log(d["brent"]), lw=lambda d: np.log(d["wti"]))
