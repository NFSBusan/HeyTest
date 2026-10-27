"""Technical indicators on (T, N) arrays. Every value at row t uses only rows <= t (causal).

Recursive indicators (EMA, Wilder smoothing) run in Numba; rolling-window ones use pandas.
`IndicatorCache` memoises results by (name, params) so thousands of strategies that share an
indicator compute it once.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Callable

import numpy as np
import pandas as pd
from numba import njit

from .data.panel import Panel


# ---------------------------------------------------------------- primitives

def _roll(x: np.ndarray, n: int, how: str) -> np.ndarray:
    df = pd.DataFrame(x)
    r = df.rolling(n, min_periods=n)
    return getattr(r, how)().to_numpy()


def sma(x, n):
    return _roll(x, n, "mean")


def rolling_std(x, n):
    return _roll(x, n, "std")


def rolling_max(x, n):
    return _roll(x, n, "max")


def rolling_min(x, n):
    return _roll(x, n, "min")


@njit(cache=True)
def _ema(x, alpha):
    T, N = x.shape
    out = np.full((T, N), np.nan)
    for j in range(N):
        prev = np.nan
        for t in range(T):
            v = x[t, j]
            if np.isnan(v):
                out[t, j] = prev
                continue
            prev = v if np.isnan(prev) else alpha * v + (1 - alpha) * prev
            out[t, j] = prev
    return out


@njit(cache=True)
def _wilder(x, n):
    """Wilder's smoothing: seed with the mean of the first n values, then alpha = 1/n."""
    T, N = x.shape
    out = np.full((T, N), np.nan)
    for j in range(N):
        cnt, acc, prev = 0, 0.0, np.nan
        for t in range(T):
            v = x[t, j]
            if np.isnan(v):
                out[t, j] = prev
                continue
            if cnt < n:
                acc += v
                cnt += 1
                if cnt == n:
                    prev = acc / n
                    out[t, j] = prev
            else:
                prev = (prev * (n - 1) + v) / n
                out[t, j] = prev
    return out


def ema(x, n):
    return _ema(np.ascontiguousarray(x, dtype=np.float64), 2.0 / (n + 1))


def lag(x, k=1):
    out = np.full_like(x, np.nan)
    if k < len(x):
        out[k:] = x[:-k] if k else x
    return out


# ---------------------------------------------------------------- indicators

def rsi(close, n):
    d = np.diff(close, axis=0, prepend=np.nan)
    up = _wilder(np.where(np.isnan(d), np.nan, np.maximum(d, 0)), n)
    dn = _wilder(np.where(np.isnan(d), np.nan, np.maximum(-d, 0)), n)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 100 - 100 / (1 + up / dn)
    out = np.where((dn == 0) & (up > 0), 100.0, out)
    return np.where((dn == 0) & (up == 0), 50.0, out)


def true_range(high, low, close):
    pc = lag(close)
    return np.fmax(high - low, np.fmax(np.abs(high - pc), np.abs(low - pc)))


def atr(high, low, close, n):
    return _wilder(true_range(high, low, close), n)


def adx(high, low, close, n):
    up = high - lag(high)
    dn = lag(low) - low
    plus = np.where((up > dn) & (up > 0), up, 0.0)
    minus = np.where((dn > up) & (dn > 0), dn, 0.0)
    nan = np.isnan(up) | np.isnan(dn)
    plus[nan], minus[nan] = np.nan, np.nan
    tr = _wilder(true_range(high, low, close), n)
    with np.errstate(divide="ignore", invalid="ignore"):
        pdi = 100 * _wilder(plus, n) / tr
        mdi = 100 * _wilder(minus, n) / tr
        dx = 100 * np.abs(pdi - mdi) / (pdi + mdi)
    return _wilder(dx, n)


def macd_hist(close, fast, slow, signal):
    line = ema(close, fast) - ema(close, slow)
    return line - ema(line, signal)


def bollinger_z(close, n):
    with np.errstate(divide="ignore", invalid="ignore"):
        return (close - sma(close, n)) / rolling_std(close, n)


def stochastic_k(high, low, close, n):
    hh, ll = rolling_max(high, n), rolling_min(low, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 100 * (close - ll) / (hh - ll)


def donchian(high, low, n):
    """Channel of the previous n bars (excludes today, so a breakout compares to the past)."""
    return lag(rolling_max(high, n)), lag(rolling_min(low, n))


def momentum(close, n, skip=0):
    with np.errstate(divide="ignore", invalid="ignore"):
        return lag(close, skip) / lag(close, n) - 1


def log_returns(close):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(close / lag(close))


def realized_vol(close, n):
    """Annualised close-to-close volatility over n days."""
    return rolling_std(log_returns(close), n) * np.sqrt(252)


def vol_percentile(close, n, lookback):
    """Where today's realised vol sits within its own past `lookback` days (0..1)."""
    v = pd.DataFrame(realized_vol(close, n))
    return v.rolling(lookback, min_periods=lookback // 2).rank(pct=True).to_numpy()


def rolling_vwap(high, low, close, volume, n):
    typ = (high + low + close) / 3
    with np.errstate(divide="ignore", invalid="ignore"):
        return _roll(typ * volume, n, "sum") / _roll(volume, n, "sum")


def volume_z(volume, n):
    lv = np.log(np.where(volume > 0, volume, np.nan))
    with np.errstate(divide="ignore", invalid="ignore"):
        return (lv - sma(lv, n)) / rolling_std(lv, n)


def dollar_volume(close, volume, n):
    return sma(close * volume, n)


# ---------------------------------------------------------------- cache

class IndicatorCache:
    """Per-panel memo of indicator arrays keyed by (name, params). LRU-bounded."""

    def __init__(self, panel: Panel, max_items: int = 512):
        self.p = panel
        self.max_items = max_items
        self._store: OrderedDict = OrderedDict()
        self.hits = self.misses = 0

    def get(self, name: str, *params) -> np.ndarray:
        key = (name, *params)
        if key in self._store:
            self.hits += 1
            self._store.move_to_end(key)
            return self._store[key]
        self.misses += 1
        val = self._compute(name, params)
        val.setflags(write=False)
        self._store[key] = val
        if len(self._store) > self.max_items:
            self._store.popitem(last=False)
        return val

    def _compute(self, name: str, params: tuple) -> np.ndarray:
        p = self.p
        fns: dict[str, Callable[[], np.ndarray]] = {
            "sma": lambda: sma(p.close, *params),
            "ema": lambda: ema(p.close, *params),
            "rsi": lambda: rsi(p.close, *params),
            "atr": lambda: atr(p.high, p.low, p.close, *params),
            "adx": lambda: adx(p.high, p.low, p.close, *params),
            "macd_hist": lambda: macd_hist(p.close, *params),
            "bollinger_z": lambda: bollinger_z(p.close, *params),
            "stoch_k": lambda: stochastic_k(p.high, p.low, p.close, *params),
            "donchian_hi": lambda: donchian(p.high, p.low, *params)[0],
            "donchian_lo": lambda: donchian(p.high, p.low, *params)[1],
            "momentum": lambda: momentum(p.close, *params),
            "realized_vol": lambda: realized_vol(p.close, *params),
            "vol_pct": lambda: vol_percentile(p.close, *params),
            "vwap": lambda: rolling_vwap(p.high, p.low, p.close, p.volume, *params),
            "volume_z": lambda: volume_z(p.volume, *params),
            "dollar_volume": lambda: dollar_volume(p.close, p.volume, *params),
            "log_ret": lambda: log_returns(p.close),
        }
        if name not in fns:
            raise KeyError(f"unknown indicator {name}")
        return np.ascontiguousarray(fns[name](), dtype=np.float64)
