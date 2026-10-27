"""Strategy building blocks. A strategy = signal -> filter -> exit rule -> position sizing.

To add a block: write a function with the signature of its kind and register it in BLOCKS
with a parameter grid. Grids define both the search space and the 'neighbours' used by the
parameter-sensitivity test, so keep them ordered.

Conventions
- Signals take (cache, short, **params) and return D (T, N) in {-1, 0, +1}: the desired
  direction at the close of day t. With short=False they never return -1.
- Everything is causal: row t only uses data up to row t.
- The backtest engine shifts weights so they trade on the NEXT bar.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from numba import njit

from ..indicators import IndicatorCache


@dataclass(frozen=True)
class Block:
    name: str
    kind: str                                   # signal | filter | exit | sizing
    fn: Callable
    grid: dict[str, list] = field(default_factory=dict)
    constraint: Callable[[dict], bool] | None = None
    doc: str = ""
    bounds: dict[str, tuple[float, float]] = field(default_factory=dict)   # allowed range for proposals


# ------------------------------------------------------------------ state machine helpers

@njit(cache=True)
def _latch(enter_long, exit_long, enter_short, exit_short):
    """Enter on an entry condition, hold until the matching exit condition."""
    T, N = enter_long.shape
    out = np.zeros((T, N))
    for j in range(N):
        pos = 0.0
        for t in range(T):
            if pos > 0 and exit_long[t, j]:
                pos = 0.0
            elif pos < 0 and exit_short[t, j]:
                pos = 0.0
            if pos == 0:
                if enter_long[t, j]:
                    pos = 1.0
                elif enter_short[t, j]:
                    pos = -1.0
            out[t, j] = pos
    return out


def _b(x):
    """NaN-safe boolean: comparisons with NaN are False."""
    return np.nan_to_num(x, nan=0.0).astype(bool) if x.dtype != bool else x


def _sign(x):
    return np.nan_to_num(np.sign(x))


def _dir(D, short):
    """Long-only strategies go flat instead of short."""
    return D if short else np.maximum(D, 0.0)


def _no(cond, short):
    """Short-entry condition, disabled for long-only strategies."""
    return _b(cond) if short else np.zeros(cond.shape, dtype=bool)


# ------------------------------------------------------------------ signals

def sig_ma_cross(c: IndicatorCache, short, fast, slow, kind="sma"):
    f, s = c.get(kind, fast), c.get(kind, slow)
    return _dir(_sign(f - s), short)


def sig_tsmom(c: IndicatorCache, short, lookback):
    return _dir(_sign(c.get("momentum", lookback, 0)), short)


def sig_donchian(c: IndicatorCache, short, entry, exit):
    close = c.p.close
    hi_e, lo_e = c.get("donchian_hi", entry), c.get("donchian_lo", entry)
    hi_x, lo_x = c.get("donchian_hi", exit), c.get("donchian_lo", exit)
    with np.errstate(invalid="ignore"):
        return _latch(_b(close > hi_e), _b(close < lo_x), _no(close < lo_e, short), _b(close > hi_x))


def sig_macd(c: IndicatorCache, short, fast, slow, signal):
    return _dir(_sign(c.get("macd_hist", fast, slow, signal)), short)


def sig_rsi_reversion(c: IndicatorCache, short, n, lower, exit_level):
    r = c.get("rsi", n)
    with np.errstate(invalid="ignore"):
        return _latch(_b(r < lower), _b(r > exit_level), _no(r > 100 - lower, short), _b(r < 100 - exit_level))


def sig_bollinger_reversion(c: IndicatorCache, short, n, k, exit_z):
    z = c.get("bollinger_z", n)
    with np.errstate(invalid="ignore"):
        return _latch(_b(z < -k), _b(z > -exit_z), _no(z > k, short), _b(z < exit_z))


def sig_stoch_reversion(c: IndicatorCache, short, n, lower, exit_level):
    k = c.get("stoch_k", n)
    with np.errstate(invalid="ignore"):
        return _latch(_b(k < lower), _b(k > exit_level), _no(k > 100 - lower, short), _b(k < 100 - exit_level))


def _rank_select(score, tradable, top_k, rebal, short, invert=False):
    """+1 for the top_k scores, -1 for the bottom_k, recomputed every `rebal` rows and held."""
    T, N = score.shape
    out = np.zeros((T, N))
    cur = np.zeros(N)
    for t in range(T):
        if t % rebal == 0:
            s = np.where(tradable[t] & np.isfinite(score[t]), score[t], np.nan)
            valid = np.flatnonzero(np.isfinite(s))
            cur = np.zeros(N)
            if len(valid) >= 2 * top_k:
                order = valid[np.argsort(s[valid])]
                lo, hi = order[:top_k], order[-top_k:]
                if invert:
                    lo, hi = hi, lo
                cur[hi] = 1.0
                if short:
                    cur[lo] = -1.0
        out[t] = cur
    return out


def sig_xs_momentum(c: IndicatorCache, short, lookback, skip, top_k, rebal):
    return _rank_select(c.get("momentum", lookback, skip), c.p.tradable, top_k, rebal, short)


def sig_xs_reversal(c: IndicatorCache, short, lookback, top_k, rebal):
    return _rank_select(c.get("momentum", lookback, 0), c.p.tradable, top_k, rebal, short, invert=True)


# ------------------------------------------------------------------ filters (return allowed-mask for longs, shorts)

def flt_none(c: IndicatorCache):
    ones = np.ones((c.p.T, c.p.N), dtype=bool)
    return ones, ones


def flt_trend_regime(c: IndicatorCache, n):
    s = c.get("sma", n)
    with np.errstate(invalid="ignore"):
        return _b(c.p.close > s), _b(c.p.close < s)


def flt_benchmark_regime(c: IndicatorCache, n):
    j = c.p.meta["benchmark_col"]
    s = c.get("sma", n)[:, j]
    with np.errstate(invalid="ignore"):
        up = np.nan_to_num(c.p.close[:, j] > s).astype(bool)
        dn = np.nan_to_num(c.p.close[:, j] < s).astype(bool)
    shape = (c.p.T, c.p.N)
    return np.broadcast_to(up[:, None], shape), np.broadcast_to(dn[:, None], shape)


def flt_adx(c: IndicatorCache, n, threshold, mode):
    a = c.get("adx", n)
    with np.errstate(invalid="ignore"):
        m = _b(a > threshold) if mode == "above" else _b(a < threshold)
    return m, m


def flt_vol_regime(c: IndicatorCache, n, lookback, max_pct):
    v = c.get("vol_pct", n, lookback)
    with np.errstate(invalid="ignore"):
        m = _b(v <= max_pct)
    return m, m


# ------------------------------------------------------------------ exits (stops on top of the signal)

@njit(cache=True)
def _apply_exits(D, close, atr, trail_mult, tp_mult, max_bars):
    """Close-based stops. After a stop the same direction is blocked until the signal resets."""
    T, N = D.shape
    out = np.zeros((T, N))
    for j in range(N):
        pos, entry, peak, a, bars, blocked = 0.0, 0.0, 0.0, np.nan, 0, 0.0
        for t in range(T):
            d = D[t, j]
            px = close[t, j]
            if blocked != 0 and d != blocked:
                blocked = 0.0
            if pos != 0 and d != pos:
                pos = 0.0
            if pos == 0:
                if d != 0 and d != blocked and not np.isnan(px):
                    pos, entry, peak, a, bars = d, px, px, atr[t, j], 0
            else:
                bars += 1
                if not np.isnan(px):
                    peak = max(peak, px) if pos > 0 else min(peak, px)
                    hit = False
                    if not np.isnan(a):
                        if trail_mult > 0:
                            hit = (pos > 0 and px <= peak - trail_mult * a) or \
                                  (pos < 0 and px >= peak + trail_mult * a)
                        if tp_mult > 0 and not hit:
                            hit = (pos > 0 and px >= entry + tp_mult * a) or \
                                  (pos < 0 and px <= entry - tp_mult * a)
                    if max_bars > 0 and bars >= max_bars:
                        hit = True
                    if hit:
                        blocked, pos = pos, 0.0
            out[t, j] = pos
    return out


def ext_none(c: IndicatorCache, D):
    return D


def ext_atr_trail(c: IndicatorCache, D, n, mult):
    return _apply_exits(D, c.p.close, c.get("atr", n), float(mult), 0.0, 0)


def ext_atr_target(c: IndicatorCache, D, n, stop_mult, target_mult):
    return _apply_exits(D, c.p.close, c.get("atr", n), float(stop_mult), float(target_mult), 0)


def ext_time_stop(c: IndicatorCache, D, max_bars):
    return _apply_exits(D, c.p.close, c.get("atr", 14), 0.0, 0.0, int(max_bars))


# ------------------------------------------------------------------ sizing (D -> raw weights)

def siz_slot(c: IndicatorCache, D):
    """Each tradable asset gets a 1/N slot: exposure grows with the number of active signals."""
    n = np.maximum(c.p.tradable.sum(axis=1, keepdims=True), 1)
    return D / n


def siz_equal_active(c: IndicatorCache, D):
    g = np.abs(D).sum(axis=1, keepdims=True)
    return np.divide(D, g, out=np.zeros_like(D), where=g > 0)


def siz_inverse_vol(c: IndicatorCache, D, n):
    v = c.get("realized_vol", n)
    inv = np.where(np.isfinite(v) & (v > 0), 1.0 / np.where(v > 0, v, 1.0), 0.0)
    raw = D * inv
    g = np.abs(raw).sum(axis=1, keepdims=True)
    return np.divide(raw, g, out=np.zeros_like(raw), where=g > 0)


def siz_vol_target(c: IndicatorCache, D, target, n):
    """Inverse-vol weights scaled so trailing realised portfolio vol ~= target (annualised)."""
    w = siz_inverse_vol(c, D, 20)
    r = np.nan_to_num(c.get("log_ret"))
    T = len(w)
    acc = np.zeros(T)
    acc2 = np.zeros(T)
    for k in range(n):                        # portfolio return of today's weights over the last n days
        rk = np.zeros_like(r)
        rk[k:] = r[:T - k] if k else r
        pr = (w * rk).sum(axis=1)
        acc += pr
        acc2 += pr * pr
    var = np.maximum(acc2 / n - (acc / n) ** 2, 0)
    vol = np.sqrt(var * 252)
    scale = np.divide(target, vol, out=np.zeros(T), where=vol > 1e-6)
    scale[: n] = 0.0
    return w * scale[:, None]


# ------------------------------------------------------------------ registry

def _int_range(lo, hi):
    return (float(lo), float(hi))


BLOCKS: dict[str, Block] = {b.name: b for b in [
    # signals
    Block("ma_cross", "signal", sig_ma_cross,
          {"fast": [5, 10, 20, 50], "slow": [50, 100, 150, 200], "kind": ["sma", "ema"]},
          lambda p: p["fast"] * 2 <= p["slow"], "fast MA above slow MA",
          {"fast": _int_range(2, 100), "slow": _int_range(10, 400)}),
    Block("tsmom", "signal", sig_tsmom, {"lookback": [21, 63, 126, 189, 252]}, None,
          "sign of trailing return", {"lookback": _int_range(5, 504)}),
    Block("donchian", "signal", sig_donchian, {"entry": [20, 55, 100, 200], "exit": [10, 20, 50]},
          lambda p: p["exit"] < p["entry"], "channel breakout, exit on opposite shorter channel",
          {"entry": _int_range(5, 400), "exit": _int_range(3, 200)}),
    Block("macd", "signal", sig_macd, {"fast": [8, 12, 19], "slow": [26, 39], "signal": [9]},
          lambda p: p["fast"] < p["slow"], "sign of MACD histogram",
          {"fast": _int_range(2, 60), "slow": _int_range(10, 200), "signal": _int_range(2, 50)}),
    Block("rsi_reversion", "signal", sig_rsi_reversion,
          {"n": [2, 3, 5, 14], "lower": [5, 10, 20, 30], "exit_level": [50, 60, 70]}, None,
          "buy RSI below `lower`, sell above `exit_level` (mirror for shorts)",
          {"n": _int_range(2, 50), "lower": _int_range(1, 45), "exit_level": _int_range(40, 90)}),
    Block("bollinger_reversion", "signal", sig_bollinger_reversion,
          {"n": [10, 20, 50], "k": [1.5, 2.0, 2.5], "exit_z": [0.0, 0.5]}, None,
          "buy below the lower band, exit when z recovers above -exit_z",
          {"n": _int_range(5, 200), "k": (0.5, 4.0), "exit_z": (-1.0, 2.0)}),
    Block("stoch_reversion", "signal", sig_stoch_reversion,
          {"n": [5, 14], "lower": [10, 20], "exit_level": [50, 80]}, None,
          "buy %K below `lower`, sell above `exit_level`",
          {"n": _int_range(3, 50), "lower": _int_range(1, 45), "exit_level": _int_range(40, 95)}),
    Block("xs_momentum", "signal", sig_xs_momentum,
          {"lookback": [63, 126, 189, 252], "skip": [0, 21], "top_k": [3, 5, 8], "rebal": [5, 21]},
          lambda p: p["skip"] < p["lookback"], "long the top_k trailing performers",
          {"lookback": _int_range(10, 504), "skip": _int_range(0, 63), "top_k": _int_range(1, 10),
           "rebal": _int_range(1, 63)}),
    Block("xs_reversal", "signal", sig_xs_reversal,
          {"lookback": [5, 10, 21], "top_k": [3, 5, 8], "rebal": [5, 10]}, None,
          "long the top_k recent losers",
          {"lookback": _int_range(1, 63), "top_k": _int_range(1, 10), "rebal": _int_range(1, 21)}),
    # filters
    Block("none", "filter", flt_none),
    Block("trend_regime", "filter", flt_trend_regime, {"n": [100, 200]}, None,
          "longs only above the asset's SMA(n)", {"n": _int_range(20, 400)}),
    Block("benchmark_regime", "filter", flt_benchmark_regime, {"n": [100, 200]}, None,
          "longs only when the benchmark is above its SMA(n)", {"n": _int_range(20, 400)}),
    Block("adx_above", "filter", lambda c, n, threshold: flt_adx(c, n, threshold, "above"),
          {"n": [14], "threshold": [15, 20, 25]}, None, "trade only in trending conditions",
          {"n": _int_range(5, 50), "threshold": (5.0, 50.0)}),
    Block("adx_below", "filter", lambda c, n, threshold: flt_adx(c, n, threshold, "below"),
          {"n": [14], "threshold": [20, 25]}, None, "trade only in range-bound conditions",
          {"n": _int_range(5, 50), "threshold": (5.0, 50.0)}),
    Block("vol_regime", "filter", flt_vol_regime, {"n": [20], "lookback": [252], "max_pct": [0.8, 0.9]},
          None, "skip the asset when its volatility is in the top of its own history",
          {"n": _int_range(5, 100), "lookback": _int_range(63, 1260), "max_pct": (0.3, 1.0)}),
    # exits
    Block("hold", "exit", ext_none, doc="exit only when the signal says so"),
    Block("atr_trail", "exit", ext_atr_trail, {"n": [14], "mult": [2, 3, 4, 5]}, None,
          "trailing stop at mult x ATR from the best close", {"n": _int_range(5, 50), "mult": (0.5, 10.0)}),
    Block("atr_target", "exit", ext_atr_target, {"n": [14], "stop_mult": [2, 3], "target_mult": [2, 4]},
          None, "stop and take-profit in ATR multiples",
          {"n": _int_range(5, 50), "stop_mult": (0.5, 10.0), "target_mult": (0.5, 20.0)}),
    Block("time_stop", "exit", ext_time_stop, {"max_bars": [5, 10, 20]}, None,
          "exit after max_bars days", {"max_bars": _int_range(1, 252)}),
    # sizing
    Block("slot", "sizing", siz_slot, doc="1/N of capital per tradable asset"),
    Block("equal_active", "sizing", siz_equal_active, doc="split capital equally over active positions"),
    Block("inverse_vol", "sizing", siz_inverse_vol, {"n": [20, 60]}, None,
          "active positions weighted by 1/volatility", {"n": _int_range(10, 252)}),
    Block("vol_target", "sizing", siz_vol_target, {"target": [0.10], "n": [60]}, None,
          "inverse-vol, scaled to a target portfolio volatility",
          {"target": (0.02, 0.4), "n": _int_range(20, 252)}),
]}
