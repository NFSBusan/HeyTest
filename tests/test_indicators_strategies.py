"""Indicators are causal; specs hash stably; blocks behave as documented."""
import numpy as np
import pytest

from qrp import indicators as ind
from qrp.data.panel import Panel
from qrp.indicators import IndicatorCache
from qrp.strategies import (FAMILIES, SpecError, canonical, enumerate_family, neighbours, sample_family, spec_hash,
                            target_weights, validate_spec)
from qrp.strategies.blocks import _apply_exits


def rand_panel(T=800, N=4, seed=0):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, (T, N)), axis=0))
    o = c * np.exp(rng.normal(0, 0.002, (T, N)))
    h = np.fmax(o, c) * 1.003
    lo = np.fmin(o, c) * 0.997
    v = np.full((T, N), 1e6)
    d = (np.datetime64("2010-01-01") + np.arange(T)).astype("datetime64[D]")
    p = Panel(d, [f"S{i}" for i in range(N)], o, h, lo, c, v, np.ones((T, N), bool))
    p.meta["benchmark_col"] = 0
    return p


@pytest.mark.parametrize("name,params", [
    ("sma", (20,)), ("ema", (20,)), ("rsi", (14,)), ("atr", (14,)), ("adx", (14,)), ("macd_hist", (12, 26, 9)),
    ("bollinger_z", (20,)), ("stoch_k", (14,)), ("donchian_hi", (20,)), ("momentum", (63, 21)),
    ("realized_vol", (20,)), ("vol_pct", (20, 252)), ("vwap", (20,)), ("volume_z", (20,)),
])
def test_indicator_is_causal(name, params):
    """Value at row t must not change when rows after t are altered."""
    p = rand_panel()
    full = IndicatorCache(p).get(name, *params)
    cut = 500
    q = p.head(cut)
    part = IndicatorCache(q).get(name, *params)
    np.testing.assert_allclose(full[:cut], part, equal_nan=True, rtol=1e-9, atol=1e-9)


def test_rsi_extremes():
    up = np.linspace(1, 100, 50)[:, None]
    assert ind.rsi(up, 14)[-1, 0] == pytest.approx(100.0)
    assert ind.rsi(up[::-1].copy(), 14)[-1, 0] == pytest.approx(0.0)


def test_donchian_excludes_today():
    h = np.arange(1, 31, dtype=float)[:, None]
    hi, _ = ind.donchian(h, h, 5)
    assert hi[10, 0] == h[9, 0]          # max of the previous 5 bars, not including bar 10


def test_spec_hash_is_canonical():
    a = {"family": "trend", "signal": {"name": "tsmom", "lookback": 63}, "filter": {"name": "none"},
         "exit": {"name": "hold"}, "sizing": {"name": "slot"}}
    b = {"sizing": {"name": "slot"}, "exit": {"name": "hold"}, "filter": {"name": "none"},
         "signal": {"lookback": 63.0, "name": "tsmom"}, "family": "trend", "direction": "long"}
    assert spec_hash(a) == spec_hash(b)


def test_validate_rejects_bad_specs():
    good = {"family": "trend", "signal": {"name": "ma_cross", "fast": 10, "slow": 50, "kind": "sma"}}
    validate_spec(good)
    with pytest.raises(SpecError):
        validate_spec({**good, "signal": {"name": "ma_cross", "fast": 60, "slow": 50, "kind": "sma"}})
    with pytest.raises(SpecError):
        validate_spec({**good, "signal": {"name": "nope"}})
    with pytest.raises(SpecError):
        validate_spec({**good, "signal": {"name": "tsmom", "lookback": 10_000}}, check_bounds=True)


def test_generator_is_deterministic_unique_and_respects_exclusions():
    a = sample_family("trend", 50, seed=7)
    b = sample_family("trend", 50, seed=7)
    assert [spec_hash(s) for s in a] == [spec_hash(s) for s in b]
    assert len({spec_hash(s) for s in a}) == 50
    c = sample_family("trend", 50, seed=7, exclude={spec_hash(s) for s in a[:10]})
    assert not {spec_hash(s) for s in a[:10]} & {spec_hash(s) for s in c}


def test_every_family_spec_runs_and_respects_limits():
    p = rand_panel(N=6)
    cache = IndicatorCache(p)
    for fam in FAMILIES:
        for s in sample_family(fam, 15, seed=1):
            W = target_weights(s, cache, 0.25, 1.0, False)
            assert W.shape == (p.T, p.N)
            assert np.isfinite(W).all()
            assert (W >= -1e-12).all(), "long-only spec produced a short"
            assert (np.abs(W) <= 0.25 + 1e-9).all()
            assert (np.abs(W).sum(axis=1) <= 1.0 + 1e-9).all()


def test_weights_are_causal():
    p = rand_panel(N=5)
    cut = 600
    for fam in FAMILIES:
        for s in sample_family(fam, 5, seed=3):
            full = target_weights(s, IndicatorCache(p), 0.25, 1.0, False)
            part = target_weights(s, IndicatorCache(p.head(cut)), 0.25, 1.0, False)
            np.testing.assert_allclose(full[:cut], part, atol=1e-12, err_msg=str(s))


def test_trailing_stop_exits_and_blocks_reentry():
    close = np.array([100, 110, 120, 100, 101, 102, 103], dtype=float)[:, None]
    D = np.ones_like(close)
    atr = np.full_like(close, 5.0)
    out = _apply_exits(D, close, atr, 2.0, 0.0, 0)    # stop at peak - 10
    assert list(out[:, 0]) == [1, 1, 1, 0, 0, 0, 0]  # stopped on day 3 and stays out while signal stays on


def test_time_stop():
    close = np.full((8, 1), 100.0)
    D = np.ones_like(close)
    out = _apply_exits(D, close, np.full_like(close, 1.0), 0.0, 0.0, 3)
    assert list(out[:, 0]) == [1, 1, 1, 0, 0, 0, 0, 0]


def test_neighbours_step_one_grid_value():
    s = canonical({"family": "trend", "signal": {"name": "tsmom", "lookback": 126}})
    lbs = sorted(n["signal"]["lookback"] for n in neighbours(s) if n["signal"]["lookback"] != 126)
    assert lbs == [63, 189]


def test_family_sizes_are_finite():
    for fam in FAMILIES:
        assert 0 < len(enumerate_family(fam)) < 50_000
