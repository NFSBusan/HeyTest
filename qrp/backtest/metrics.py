"""Performance metrics from daily returns (and the engine's trade list)."""
from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis, skew

from .engine import BacktestResult

ANN = 252


def sharpe(r: np.ndarray) -> float:
    r = np.asarray(r, dtype=float)
    if len(r) < 2:
        return 0.0
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(ANN)) if sd > 1e-12 else 0.0


def max_drawdown(r: np.ndarray) -> float:
    eq = np.cumprod(1 + np.asarray(r, dtype=float))
    peak = np.maximum.accumulate(np.concatenate([[1.0], eq]))[1:]
    return float(np.max(1 - eq / peak)) if len(eq) else 0.0


def drawdown_series(r: np.ndarray) -> np.ndarray:
    eq = np.cumprod(1 + np.asarray(r, dtype=float))
    peak = np.maximum.accumulate(np.concatenate([[1.0], eq]))[1:]
    return eq / peak - 1


def summarize(r: np.ndarray, res: BacktestResult | None = None, start: int = 0, end: int | None = None,
              dates: np.ndarray | None = None) -> dict:
    """Metrics over rows [start, end). `res` adds cost, turnover, exposure and trade statistics."""
    end = len(r) if end is None else end
    x = np.asarray(r[start:end], dtype=float)
    n = len(x)
    out: dict = {"n_days": n}
    if n < 2:
        return out
    eq = np.prod(1 + x)
    years = n / ANN
    sd = x.std(ddof=1)
    down = x[x < 0]
    dsd = np.sqrt(np.mean(np.minimum(x, 0) ** 2)) if len(down) else 0.0
    mdd = max_drawdown(x)
    cagr = eq ** (1 / years) - 1 if eq > 0 else -1.0
    out.update({
        "total_return": float(eq - 1),
        "cagr": float(cagr),
        "ann_vol": float(sd * np.sqrt(ANN)),
        "sharpe": sharpe(x),
        "sortino": float(x.mean() / dsd * np.sqrt(ANN)) if dsd > 1e-12 else 0.0,
        "max_drawdown": mdd,
        "calmar": float(cagr / mdd) if mdd > 1e-9 else 0.0,
        "skew": float(skew(x)) if sd > 0 else 0.0,
        "kurtosis": float(kurtosis(x, fisher=False)) if sd > 0 else 3.0,
        "best_day": float(x.max()),
        "worst_day": float(x.min()),
    })
    if dates is not None:
        yrs = dates[start:end].astype("datetime64[Y]")
        yearly = [np.prod(1 + x[yrs == y]) - 1 for y in np.unique(yrs)]
        out["positive_year_fraction"] = float(np.mean(np.array(yearly) > 0)) if yearly else 0.0
    if res is not None:
        sl = slice(start, end)
        out["cost_drag"] = float(res.costs[sl].sum() / years)
        out["turnover"] = float(res.turnover[sl].sum() / years)
        out["exposure"] = float(res.exposure[sl].mean())
        m = (res.trade_entry >= start) & (res.trade_entry < end)
        pnl = res.trade_pnl[m]
        out["n_trades"] = int(m.sum())
        out["trades_per_year"] = float(m.sum() / years)
        if len(pnl):
            wins, losses = pnl[pnl > 0], pnl[pnl < 0]
            out["win_rate"] = float(len(wins) / len(pnl))
            out["profit_factor"] = float(wins.sum() / -losses.sum()) if len(losses) and losses.sum() < 0 \
                else (float("inf") if len(wins) else 0.0)
            out["expectancy"] = float(pnl.mean())
            out["avg_hold_days"] = float(res.trade_len[m].mean())
        else:
            out.update({"win_rate": 0.0, "profit_factor": 0.0, "expectancy": 0.0, "avg_hold_days": 0.0})
    return out
