"""Robustness tests for one candidate (dev data only)."""
from __future__ import annotations

import numpy as np

from ..backtest.metrics import sharpe
from ..evaluate import Evaluator
from ..indicators import realized_vol, sma
from ..strategies import neighbours, spec_hash
from .stats import monte_carlo


def regime_breakdown(ev: Evaluator, r: np.ndarray, start: int, end: int) -> dict:
    """Sharpe in bull/bear (benchmark vs its 200-day SMA) and high/low volatility regimes."""
    j = ev.s.bench_col
    c = ev.p.close[:, j:j + 1]
    with np.errstate(invalid="ignore"):
        bull = (c > sma(c, 200))[:, 0]
    vol = realized_vol(c, 63)[:, 0]
    sl = slice(start, end)
    med = np.nanmedian(vol[sl])
    with np.errstate(invalid="ignore"):
        hi_vol = vol > med
    out = {}
    for name, mask in (("bull", bull), ("bear", ~bull), ("high_vol", hi_vol), ("low_vol", ~hi_vol)):
        m = mask[sl] & np.isfinite(vol[sl])
        out[name] = {"sharpe": sharpe(r[sl][m]) if m.sum() > 20 else None, "days": int(m.sum())}
    return out


def yearly_returns(ev: Evaluator, r: np.ndarray, start: int, end: int) -> dict[str, float]:
    yrs = ev.p.dates[start:end].astype("datetime64[Y]")
    x = r[start:end]
    return {str(y): float(np.prod(1 + x[yrs == y]) - 1) for y in np.unique(yrs)}


def robustness_suite(ev: Evaluator, spec: dict, cfg, start: int, end: int,
                     neighbour_results: dict[str, dict] | None = None) -> dict:
    """Parameter sensitivity, cost stress, capacity, Monte Carlo, regimes, yearly consistency.

    neighbour_results: metrics for neighbour hashes that the caller already has (registry
    cache); the rest are computed here and returned in out["new_neighbours"] for storage.
    """
    base, res = ev.run(spec)
    r = res.returns
    out: dict = {"base": base}
    # 1. parameter sensitivity: one grid step in each parameter
    nb_sharpes, new_nb = [], {}
    cache = neighbour_results or {}
    for nb in neighbours(spec):
        h = spec_hash(nb)
        if h in cache:
            m = cache[h]
        else:
            m, nres = ev.run(nb)
            new_nb[h] = (nb, m, nres.returns.astype(np.float32))
        nb_sharpes.append(m["sharpe"])
    out["neighbour_sharpes"] = nb_sharpes
    med = float(np.median(nb_sharpes)) if nb_sharpes else None
    out["neighbour_median_sharpe"] = med
    out["neighbour_ratio"] = (med / base["sharpe"]) if (med is not None and base["sharpe"] > 0) else 0.0
    out["new_neighbours"] = new_nb
    # 2. transaction-cost stress
    out["cost_stress"] = {}
    for mult in cfg["costs"]["stress_multipliers"]:
        m, _ = ev.run(spec, cost_mult=mult)
        out["cost_stress"][str(mult)] = {"sharpe": m["sharpe"], "cagr": m["cagr"], "cost_drag": m["cost_drag"]}
    # 3. capacity: same strategy with 10x and 100x the capital (market impact grows)
    out["capacity"] = {}
    cap = cfg["costs"]["capital"]
    for k in (10, 100):
        m, _ = ev.run(spec, capital=cap * k)
        out["capacity"][str(int(cap * k))] = {"sharpe": m["sharpe"], "cost_drag": m["cost_drag"]}
    # 4. Monte Carlo (block bootstrap of daily returns)
    v = cfg["validation"]
    out["monte_carlo"] = monte_carlo(r[start:end], v["mc_reps"], v["block_len"], cfg["project"]["seed"])
    # 5. regimes and 6. calendar-year consistency
    out["regimes"] = regime_breakdown(ev, r, start, end)
    out["yearly"] = yearly_returns(ev, r, start, end)
    return out
