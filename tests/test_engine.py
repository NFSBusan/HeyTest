"""Backtest engine: timing (no lookahead), cost arithmetic, missing data."""
import numpy as np
import pytest

from qrp.backtest import CostModel, Market, buy_and_hold, run_backtest, sharpe, summarize
from qrp.data.panel import Panel

FREE = CostModel(0, 0, 0, 0, 1e4)


def toy_panel(close, open_=None, vol=1e6):
    close = np.asarray(close, dtype=float)
    if close.ndim == 1:
        close = close[:, None]
    open_ = close.copy() if open_ is None else np.asarray(open_, dtype=float).reshape(close.shape)
    T, N = close.shape
    dates = (np.datetime64("2020-01-01") + np.arange(T)).astype("datetime64[D]")
    return Panel(dates, [f"A{i}" for i in range(N)], open_, np.fmax(open_, close), np.fmin(open_, close), close,
                 np.full((T, N), vol), np.isfinite(close) & np.isfinite(open_))


def test_weights_trade_next_open_not_same_bar():
    # Price jumps +10% from close 1 to close 2 entirely overnight (open 2 = close 2).
    p = toy_panel([100, 100, 110, 110], open_=[100, 100, 110, 110])
    m = Market.from_panel(p)
    W = np.zeros((4, 1))
    W[1] = 1.0                    # decided at close of day 1 ...
    r = run_backtest(W, m, FREE).returns
    assert r[2] == pytest.approx(0.0)   # ... so the overnight jump into day 2 is NOT captured


def test_intraday_move_after_entry_is_captured():
    p = toy_panel([100, 100, 110, 110], open_=[100, 100, 100, 110])
    m = Market.from_panel(p)
    W = np.zeros((4, 1))
    W[1] = 1.0
    r = run_backtest(W, m, FREE).returns
    assert r[2] == pytest.approx(0.10)  # bought at the open of day 2, earned open->close


def test_next_close_adds_a_day_of_delay():
    p = toy_panel([100, 100, 110, 121])
    m = Market.from_panel(p)
    W = np.zeros((4, 1))
    W[1:] = 1.0
    r = run_backtest(W, m, FREE, fill="next_close").returns
    assert r[2] == pytest.approx(0.0) and r[3] == pytest.approx(0.10)


def test_foresight_signal_cannot_profit_on_noise():
    """W[t] = sign of the SAME day's return. If the engine leaked, Sharpe would be enormous."""
    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, (3000, 5)), axis=0))
    p = toy_panel(close)
    m = Market.from_panel(p)
    W = np.sign(m.r_cc) / 5
    r = run_backtest(W, m, FREE).returns
    assert abs(sharpe(r)) < 0.6


def test_costs_charged_per_unit_turnover():
    p = toy_panel([100.0] * 6)
    m = Market.from_panel(p)
    W = np.zeros((6, 1))
    W[1:3] = 1.0                                   # enter (exec day 2), exit (exec day 4)
    c = CostModel(commission_bps=1, half_spread_bps=4, impact_coef=0, borrow_bps_annual=0, capital=1e4)
    res = run_backtest(W, m, c)
    assert res.costs.sum() == pytest.approx(2 * 5e-4)
    assert res.returns.sum() == pytest.approx(-2 * 5e-4, rel=1e-6)
    assert len(res.trade_pnl) == 1 and res.trade_pnl[0] == pytest.approx(-1e-3)


def test_impact_grows_with_capital():
    p = toy_panel(100 + np.cumsum(np.random.default_rng(1).normal(0, 1, 300)), vol=1000)
    m = Market.from_panel(p)
    W = np.zeros((300, 1))
    W[50::2] = 1.0                                  # churn every other day
    small = run_backtest(W, m, CostModel(0, 0, 0.1, 0, 1e3)).costs.sum()
    big = run_backtest(W, m, CostModel(0, 0, 0.1, 0, 1e6)).costs.sum()
    assert big > 10 * small > 0


def test_cannot_trade_an_asset_on_a_day_without_a_price():
    close = np.array([[100, 50], [100, 50], [100, np.nan], [100, 55], [100, 60]], dtype=float)
    p = toy_panel(close)
    m = Market.from_panel(p)
    W = np.zeros((5, 2))
    W[1:, 1] = 1.0                    # wants to buy asset 1 from day 2, but day 2 has no price
    res = run_backtest(W, m, FREE)
    assert res.exposure[2] == 0.0     # order could not execute on the missing day
    assert res.exposure[3] == 1.0     # executed at the next available open
    assert np.isfinite(res.returns).all()
    assert res.returns[4] == pytest.approx(60 / 55 - 1)


def test_buy_and_hold_matches_price_path():
    close = np.array([100, 101, 99, 105, 104.0])
    p = toy_panel(close)
    m = Market.from_panel(p)
    r = buy_and_hold(m, 0, FREE).returns
    # decided at the close of day 0, bought at the open of day 1 (open == close == 101 in this toy panel)
    assert np.prod(1 + r) == pytest.approx(close[-1] / close[1])


def test_summarize_basic():
    r = np.array([0.01, -0.005, 0.002, 0.0, 0.003] * 100)
    m = summarize(r)
    assert m["total_return"] == pytest.approx(np.prod(1 + r) - 1)
    assert 0 <= m["max_drawdown"] < 0.01
    assert m["sharpe"] > 0
