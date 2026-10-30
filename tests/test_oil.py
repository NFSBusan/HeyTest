"""Tests for the Brent-WTI research code. Run: python -m pytest -q tests/test_oil.py"""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from oilmr import diagnostics, options
from oilmr.data import load_pair


def test_fred_csv_parsing(tmp_path):
    (tmp_path / "b.csv").write_text("observation_date,DCOILBRENTEU\n2020-04-17,28.0\n2020-04-20,25.6\n2020-04-21,.\n")
    (tmp_path / "w.csv").write_text("observation_date,DCOILWTICO\n2020-04-17,18.3\n2020-04-20,-36.98\n2020-04-21,8.9\n")
    df = load_pair(tmp_path / "b.csv", tmp_path / "w.csv")
    assert list(df.index.strftime("%Y-%m-%d")) == ["2020-04-17", "2020-04-20"]
    assert df["wti_nonpositive"].tolist() == [False, True]
    assert df["spread"].iloc[1] == pytest.approx(25.6 + 36.98)


def test_half_life_recovers_ou():
    rng = np.random.default_rng(1)
    hl, n = 20.0, 20_000
    phi = 0.5 ** (1 / hl)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.normal()
    assert diagnostics.half_life(x) == pytest.approx(hl, rel=0.15)
    assert diagnostics.half_life(np.cumsum(rng.normal(size=n))) > 200  # random walk: very slow or inf


@pytest.mark.parametrize("name", list(options.STRUCTURES))
def test_closed_form_matches_monte_carlo(name):
    w = options.World()
    legs = options.STRUCTURES[name]
    e = options.evaluate(w, legs)
    rng = np.random.default_rng(0)
    mean, cov = options.terminal_moments(w)
    x = rng.multivariate_normal(mean, cov, size=400_000)
    pnl = np.zeros(len(x))
    for leg, kind, off, q in legs:
        a, L = options.leg_loadings(w)[leg]
        _, sd = options.leg_change_dist(w, leg)
        k = off * sd if kind == "call" else -off * sd
        dx = a + x @ L
        pnl += q * (np.maximum(dx - k, 0) if kind == "call" else np.maximum(k - dx, 0))
    assert pnl.mean() == pytest.approx(e["exp_payoff"], rel=0.01, abs=0.01)
    # Defined risk: payoff of every long-premium structure is >= 0, so loss <= debit + costs.
    assert pnl.min() >= -1e-12
    assert e["premium"] > 0


def test_no_edge_when_convergence_is_all_brent():
    w = replace(options.World(), phi=0.0)
    e = options.evaluate(w, options.STRUCTURES["wti_call_atm"])
    assert e["exp_pnl"] == pytest.approx(-e["cost"], abs=1e-9)


def test_no_edge_when_fully_priced_in():
    w = replace(options.World(), priced_in=1.0)
    for legs in options.STRUCTURES.values():
        e = options.evaluate(w, legs)
        assert e["exp_pnl"] == pytest.approx(-e["cost"], abs=1e-9)


def test_spread_identity():
    w = options.World(phi=0.3)
    L = options.leg_loadings(w)
    (ab, Lb), (aw, Lw), (as_, Ls) = L["brent"], L["wti"], L["spread"]
    # (B_T - B_0) - (W_T - W_0) == S_T - S_0
    assert ab - aw == pytest.approx(as_)
    assert np.allclose(Lb - Lw, Ls)
