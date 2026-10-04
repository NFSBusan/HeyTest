"""Leakage tests. Run: python -m pytest -q tests"""
import numpy as np
import pandas as pd
import pytest

from pmct.schema import DAY
from pmct.scoring import scores_at, wallet_history
from pmct.strategy import Context, StratConfig, backtest
from pmct.synthetic import SimConfig, simulate


def _tiny():
    markets = pd.DataFrame(dict(market_id=[0, 1], created_ts=[0, 0], end_ts=[100, 300],
                                resolved_ts=[150, 400], outcome=[1.0, 0.0], category=["x", "x"]))
    trades = pd.DataFrame(dict(ts=[10, 20, 200], market_id=[0, 1, 1], wallet=[7, 7, 8], dir=[-1, 1, 1],
                               yes_price=[0.4, 0.6, 0.5], usd=[10.0, 10.0, 10.0], shares=[10 / 0.6, 10 / 0.6, 20.0]))
    return markets, trades


def test_score_only_after_resolution():
    markets, trades = _tiny()
    hist = wallet_history(markets, trades, "res")
    q = pd.DataFrame(dict(wallet=[7, 7, 7, 7], ts=[149, 150, 151, 401]))
    s = scores_at(hist, q)
    assert s["n_res"].tolist() == [0, 0, 1, 2]        # strict: not visible AT resolution time
    assert s["pnl"].iloc[2] == pytest.approx(-(1 - 0.4) * 10 / 0.6)   # bet NO at 0.6, YES won


def _poison(markets, trades, T, seed=1):
    rng = np.random.default_rng(seed)
    m, t = markets.copy(), trades.copy()
    fut = m["resolved_ts"] >= T
    m.loc[fut, "outcome"] = 1 - m.loc[fut, "outcome"]
    ft = t["ts"] >= T
    t.loc[ft, "dir"] = -t.loc[ft, "dir"]
    t.loc[ft, "yes_price"] = rng.uniform(0.02, 0.98, ft.sum())
    t.loc[ft, "wallet"] = rng.permutation(t.loc[ft, "wallet"].to_numpy())
    return m, t


@pytest.fixture(scope="module")
def world():
    return simulate(SimConfig(n_days=200, n_markets=400, n_wallets=800, seed=3))[:2]


@pytest.mark.parametrize("basis,score,thr", [("res", "roi", -0.02), ("mo1d", "tstat", -1.0), ("res_mid", "tstat", -1.0)])
def test_future_poisoning_changes_nothing_before_cutoff(world, basis, score, thr):
    """Rewrite everything after T (outcomes resolving >= T, all trades >= T). Every decision and
    every reported number for a backtest ending at T must be identical."""
    markets, trades = world
    T, start = 140 * DAY, 60 * DAY
    cfg = StratConfig(basis=basis, score=score, threshold=thr, min_res=10, min_mkts=2, calib_min=20, calib_n0=50)
    a = Context(markets, trades)
    b = Context(*_poison(markets, trades, T))
    ra = backtest(cfg, a.signals(cfg), a.book, start, T)
    rb = backtest(cfg, b.signals(cfg), b.book, start, T)
    assert ra.metrics["n_bets"] > 0, "test is vacuous without bets"
    pd.testing.assert_frame_equal(ra.positions, rb.positions)
    pd.testing.assert_series_equal(ra.equity, rb.equity)
    sa, sb = a.signals(cfg), b.signals(cfg)
    sa, sb = sa[sa.exec_ts < T].reset_index(drop=True), sb[sb.exec_ts < T].reset_index(drop=True)
    cols = ["exec_ts", "market_id", "wallet", "our_dir", "q", "edge_hat", "n_calib", "wscore", "settle_ts"]
    pd.testing.assert_frame_equal(sa[cols], sb[cols])


def test_poisoning_does_change_things_after_cutoff(world):
    """Control: the poison must actually matter, otherwise the test above proves nothing."""
    markets, trades = world
    T = 140 * DAY
    cfg = StratConfig(basis="res", score="roi", threshold=-0.02, min_res=10, min_mkts=2, calib_min=20, calib_n0=50)
    a = Context(markets, trades)
    b = Context(*_poison(markets, trades, T))
    ra = backtest(cfg, a.signals(cfg), a.book, T, 200 * DAY)
    rb = backtest(cfg, b.signals(cfg), b.book, T, 200 * DAY)
    assert not ra.positions.equals(rb.positions)


def test_execution_never_at_their_price(world):
    markets, trades = world
    cfg = StratConfig(basis="res", score="roi", threshold=-0.02, min_res=10, min_mkts=2)
    ctx = Context(markets, trades)
    s = ctx.signals(cfg)
    assert (s["exec_ts"] - s["ts"] == cfg.latency_s).all()
    # our cost is mid + exec_cost rounded UP to the tick: never better than that
    assert (s["q"] - s["side_mid"] >= cfg.exec_cost - 1e-9).all()
    assert (s["q"] - s["side_mid"] < cfg.exec_cost + cfg.tick + 1e-9).all()


@pytest.mark.parametrize("extra", [dict(exit_h=6.0), dict(first_only=True, flow_h=24.0, min_flow=50.0), dict(calib_linear=True), dict(maker=True)])
def test_future_poisoning_with_exits_and_flow(world, extra):
    markets, trades = world
    T, start = 140 * DAY, 60 * DAY
    # Unrealistic negative exec_cost makes the estimated edge positive so bets happen and the
    # test can't pass vacuously. It only checks information flow, not profitability.
    cfg = StratConfig(basis="mo1d", score="tstat", threshold=-1.0, min_res=10, min_mkts=2, calib_min=20,
                      calib_n0=5, exec_cost=-0.02, exit_cost=-0.02, **extra)
    a = Context(markets, trades)
    b = Context(*_poison(markets, trades, T))
    ra = backtest(cfg, a.signals(cfg), a.book, start, T)
    rb = backtest(cfg, b.signals(cfg), b.book, start, T)
    assert ra.metrics["n_bets"] > 0
    pd.testing.assert_frame_equal(ra.positions, rb.positions)
    pd.testing.assert_series_equal(ra.equity, rb.equity)
