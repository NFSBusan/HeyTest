"""Wide, aligned price panels (dates x symbols) and the dev / holdout split.

The dev panel is physically truncated before the holdout, so no code running on it can read
holdout prices. The holdout is only reachable through `validation.holdout.HoldoutVault`.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

FIELDS = ("open", "high", "low", "close", "volume")


@dataclass
class Panel:
    dates: np.ndarray                 # datetime64[D], shape (T,)
    symbols: list[str]
    open: np.ndarray                  # (T, N) float64, NaN when missing
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    tradable: np.ndarray              # (T, N) bool: has data and is in the universe that day
    meta: dict = field(default_factory=dict)

    @property
    def T(self) -> int:
        return len(self.dates)

    @property
    def N(self) -> int:
        return len(self.symbols)

    def col(self, symbol: str) -> int:
        return self.symbols.index(symbol)

    def head(self, end: int) -> "Panel":
        """Rows [0, end). Used to build the dev view; the rows after `end` are dropped."""
        return Panel(self.dates[:end].copy(), list(self.symbols),
                     *(getattr(self, f)[:end].copy() for f in FIELDS),
                     self.tradable[:end].copy(), dict(self.meta))

    def fingerprint(self, start: int = 0, end: int | None = None) -> str:
        """Hash of the data in rows [start, end): identifies 'the same data' across studies."""
        end = self.T if end is None else end
        h = hashlib.sha1()
        h.update(",".join(self.symbols).encode())
        h.update(self.dates[start:end].astype("int64").tobytes())
        h.update(np.nan_to_num(np.round(self.close[start:end], 6)).tobytes())
        return h.hexdigest()[:16]


def build_panel(long: pd.DataFrame, symbols: list[str], universe: pd.DataFrame | None = None) -> Panel:
    """Long bars (symbol, date, open, ...) -> aligned panel on the union of trading dates."""
    if long.empty:
        raise ValueError("no bars to build a panel from; run `qrp fetch` first")
    long = long.copy()
    long["date"] = pd.to_datetime(long["date"]).dt.normalize()
    long = long.drop_duplicates(["symbol", "date"], keep="last")
    dates = np.sort(long["date"].unique())
    syms = [s for s in symbols if s in set(long["symbol"])]
    missing = sorted(set(symbols) - set(syms))
    arrays = {}
    for f in FIELDS:
        wide = long.pivot(index="date", columns="symbol", values=f)
        arrays[f] = wide.reindex(index=dates, columns=syms).to_numpy(dtype="float64")
    tradable = np.isfinite(arrays["close"]) & np.isfinite(arrays["open"])
    if universe is not None and not universe.empty:
        d = pd.DatetimeIndex(dates)
        member = np.zeros_like(tradable)
        for _, row in universe.iterrows():
            if row["symbol"] in syms:
                j = syms.index(row["symbol"])
                lo = pd.Timestamp(row["start"]) if pd.notna(row.get("start")) else d[0]
                hi = pd.Timestamp(row["end"]) if pd.notna(row.get("end")) and row.get("end") else d[-1]
                member[:, j] |= np.asarray((d >= lo) & (d <= hi))
        tradable &= member
    return Panel(pd.DatetimeIndex(dates).values.astype("datetime64[D]"), syms,
                 arrays["open"], arrays["high"], arrays["low"], arrays["close"], arrays["volume"],
                 tradable, {"missing_symbols": missing})


@dataclass(frozen=True)
class Split:
    warmup_end: int        # first scored row
    dev_end: int           # dev rows are [0, dev_end); scoring in [warmup_end, dev_end)
    holdout_start: int     # holdout rows are [holdout_start, T)
    T: int

    def describe(self, dates: np.ndarray) -> dict:
        def d(i):
            return str(pd.Timestamp(dates[min(i, len(dates) - 1)]).date())
        return {"warmup_start": d(0), "dev_start": d(self.warmup_end), "dev_end": d(self.dev_end - 1),
                "holdout_start": d(self.holdout_start), "holdout_end": d(self.T - 1)}


def make_split(T: int, holdout_fraction: float, embargo_days: int, warmup_days: int) -> Split:
    scored = T - warmup_days
    if scored < 500:
        raise ValueError(f"only {scored} scored days after warm-up; need at least ~2 years")
    holdout_start = T - int(round(scored * holdout_fraction))
    dev_end = holdout_start - embargo_days
    return Split(warmup_days, dev_end, holdout_start, T)


@dataclass(frozen=True)
class Fold:
    train: tuple[int, int]   # [start, end) row indices
    test: tuple[int, int]


def walk_forward_folds(start: int, end: int, n_folds: int, min_train: int, embargo: int) -> list[Fold]:
    """Anchored walk-forward: train on [start, k), skip `embargo` rows, test the next block."""
    if end - start <= min_train + embargo:
        raise ValueError("dev period too short for walk-forward with this min_train")
    test_len = (end - start - min_train) // n_folds
    if test_len < 60:
        raise ValueError("walk-forward test blocks would be shorter than ~3 months; use fewer folds")
    folds = []
    for k in range(n_folds):
        test_start = start + min_train + k * test_len
        test_end = end if k == n_folds - 1 else test_start + test_len
        folds.append(Fold((start, test_start - embargo), (test_start, test_end)))
    return folds
