"""Portfolio backtest engine for daily target weights.

Timing (fill = next_open, the default):
  - Weights W[t] are decided from data up to the close of day t.
  - They are executed at the OPEN of day t+1. The overnight move (close t -> open t+1) is
    earned by the old position, the intraday move (open -> close) by the new one.
  - fill = next_close executes at the close of t+1 instead (one extra day of delay).
An asset with no price on a day cannot be traded that day; its position is carried.

Weights are rebalanced to target every day; turnover includes the drift back to target.
Costs per unit of turnover: commission + half-spread + market impact, where
  impact = impact_coef * daily_vol * sqrt(trade_value / avg_daily_dollar_volume),
using volatility and volume known BEFORE the trade (lagged one day). Shorts pay borrow.
Cash earns nothing (Sharpe ratios are therefore versus zero, for strategies and benchmarks alike).
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numba import njit

from ..data.panel import Panel
from ..indicators import lag, log_returns, rolling_std, sma


@dataclass(frozen=True)
class CostModel:
    commission_bps: float = 0.5
    half_spread_bps: float = 1.5
    impact_coef: float = 0.1
    borrow_bps_annual: float = 50.0
    capital: float = 10_000.0

    @classmethod
    def from_config(cls, costs: dict) -> "CostModel":
        return cls(costs["commission_bps"], costs["half_spread_bps"], costs["impact_coef"],
                   costs["borrow_bps_annual"], costs["capital"])

    def scaled(self, mult: float) -> "CostModel":
        return replace(self, commission_bps=self.commission_bps * mult,
                       half_spread_bps=self.half_spread_bps * mult, impact_coef=self.impact_coef * mult,
                       borrow_bps_annual=self.borrow_bps_annual * mult)

    def with_capital(self, capital: float) -> "CostModel":
        return replace(self, capital=capital)


@dataclass
class Market:
    """Per-panel arrays shared by every backtest on that panel."""
    r_on: np.ndarray      # open[t] / close[t-1] - 1
    r_id: np.ndarray      # close[t] / open[t] - 1
    r_cc: np.ndarray      # close[t] / close[t-1] - 1
    vol: np.ndarray       # daily vol known before trading on day t
    adv: np.ndarray       # average daily dollar volume known before day t
    tradable: np.ndarray

    @classmethod
    def from_panel(cls, p: Panel) -> "Market":
        pc = lag(p.close)
        with np.errstate(divide="ignore", invalid="ignore"):
            r_on = p.open / pc - 1
            r_id = p.close / p.open - 1
            r_cc = p.close / pc - 1
        vol = lag(rolling_std(log_returns(p.close), 20))
        adv = lag(sma(p.close * p.volume, 20))
        z = lambda x: np.ascontiguousarray(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0))
        vol = np.where(np.isfinite(vol), vol, 0.02)          # conservative default before 20 days of data
        adv = np.where(np.isfinite(adv) & (adv > 0), adv, np.inf)
        return cls(z(r_on), z(r_id), z(r_cc), np.ascontiguousarray(vol), np.ascontiguousarray(adv),
                   np.ascontiguousarray(p.tradable))


@dataclass
class BacktestResult:
    returns: np.ndarray        # (T,) net daily returns
    gross: np.ndarray          # (T,) before costs
    costs: np.ndarray          # (T,) cost as a fraction of equity
    turnover: np.ndarray       # (T,) sum |change in weight|
    exposure: np.ndarray       # (T,) gross exposure held over the day
    trade_pnl: np.ndarray      # per-trade P&L (fraction of equity, additive)
    trade_len: np.ndarray      # per-trade holding period in days
    trade_entry: np.ndarray    # row index of each trade's entry


@njit(cache=True)
def _run(W, r_on, r_id, r_cc, vol, adv, tradable, fixed_bps, impact_coef, borrow_daily, capital,
         next_open):
    T, N = W.shape
    E = np.zeros((T, N))
    ret = np.zeros(T)
    gross = np.zeros(T)
    cost = np.zeros(T)
    turn = np.zeros(T)
    expo = np.zeros(T)
    max_trades = T * N + N + 1            # at most one trade closes per (day, asset), plus open ones at the end
    tp = np.zeros(max_trades)
    tl = np.zeros(max_trades, dtype=np.int64)
    te = np.zeros(max_trades, dtype=np.int64)
    nt = 0
    cur_pnl = np.zeros(N)
    cur_start = np.zeros(N, dtype=np.int64)
    for t in range(1, T):
        for j in range(N):
            E[t, j] = W[t - 1, j] if tradable[t, j] else E[t - 1, j]
        # Pre-trade weights drift with prices; rebalancing back to target is real turnover.
        r_pre = 0.0
        for j in range(N):
            r_pre += E[t - 1, j] * (r_on[t, j] if next_open else r_cc[t, j])
        r_a = 0.0
        r_b = 0.0
        c_t = 0.0
        b_t = 0.0
        for j in range(N):
            prev, cur = E[t - 1, j], E[t, j]
            drifted = prev * (1 + (r_on[t, j] if next_open else r_cc[t, j])) / (1 + r_pre)
            d = abs(cur - drifted) if tradable[t, j] else 0.0
            cj = 0.0
            if d > 0:
                imp = impact_coef * vol[t, j] * np.sqrt(d * capital / adv[t, j])
                cj = d * (fixed_bps / 1e4 + imp)
            bj = max(-cur, 0.0) * borrow_daily
            if next_open:
                a_j = prev * r_on[t, j]           # overnight, old position
                b_j = cur * r_id[t, j]            # intraday, new position
            else:
                a_j = prev * r_cc[t, j]           # whole day, old position; trade at today's close
                b_j = 0.0
            r_a += a_j
            r_b += b_j
            c_t += cj
            b_t += bj
            turn[t] += d
            # ---- trade bookkeeping (additive per-asset contributions)
            sp = np.sign(prev)
            sc = np.sign(cur)
            if sp != 0:
                cur_pnl[j] += a_j
            if sp != sc:
                if sp != 0:
                    share = abs(prev) / (abs(prev) + abs(cur))
                    cur_pnl[j] -= cj * share
                    tp[nt] = cur_pnl[j]
                    tl[nt] = t - cur_start[j]
                    te[nt] = cur_start[j]
                    nt += 1
                if sc != 0:
                    share = abs(cur) / (abs(prev) + abs(cur))
                    cur_pnl[j] = b_j - cj * share - bj
                    cur_start[j] = t
            elif sc != 0:
                cur_pnl[j] += b_j - cj - bj
            expo[t] += abs(cur)
        g = (1 + r_a) * (1 + r_b) - 1
        gross[t] = g
        cost[t] = c_t + b_t
        ret[t] = (1 + r_a) * (1 - c_t) * (1 + r_b) - 1 - b_t
    for j in range(N):                             # open trades are marked at the last close
        if E[T - 1, j] != 0:
            tp[nt] = cur_pnl[j]
            tl[nt] = T - 1 - cur_start[j]
            te[nt] = cur_start[j]
            nt += 1
    return ret, gross, cost, turn, expo, tp[:nt], tl[:nt], te[:nt]


def run_backtest(W: np.ndarray, market: Market, costs: CostModel, fill: str = "next_open",
                 active_from: int = 0) -> BacktestResult:
    """Simulate target weights W (T, N). Rows before `active_from` are forced flat."""
    W = np.ascontiguousarray(np.nan_to_num(W), dtype=np.float64)
    if active_from:
        W = W.copy()
        W[:active_from] = 0.0
    out = _run(W, market.r_on, market.r_id, market.r_cc, market.vol, market.adv, market.tradable,
               costs.commission_bps + costs.half_spread_bps, costs.impact_coef,
               costs.borrow_bps_annual / 1e4 / 252, costs.capital, fill == "next_open")
    return BacktestResult(*out)


def buy_and_hold(market: Market, col: int, costs: CostModel, fill: str = "next_open",
                 active_from: int = 0) -> BacktestResult:
    T, N = market.r_cc.shape
    W = np.zeros((T, N))
    W[:, col] = 1.0
    return run_backtest(W, market, costs, fill, active_from)
