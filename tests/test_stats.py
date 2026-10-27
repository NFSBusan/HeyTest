"""Overfitting statistics behave correctly on noise and on a real effect."""
import numpy as np

from qrp.data.panel import make_split, walk_forward_folds
from qrp.validation.stats import (alpha_returns, deflated_sharpe, effective_trials, expected_max_sharpe, pbo_cscv,
                                  spa_test)


def test_expected_max_sharpe_grows_with_trials():
    assert expected_max_sharpe(1000, 1e-4) > expected_max_sharpe(10, 1e-4) > 0


def test_dsr_penalises_many_trials():
    r = np.random.default_rng(0).normal(0.0006, 0.01, 2500)
    few = deflated_sharpe(r, 2, 1e-4)["dsr"]
    many = deflated_sharpe(r, 5000, 1e-4)["dsr"]
    assert many < few


def test_pbo_high_on_noise_low_on_real_signal():
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 0.01, (2000, 50))
    assert pbo_cscv(noise, 10)["pbo"] > 0.3
    real = noise.copy()
    real[:, :5] += np.linspace(0.0002, 0.001, 5)       # five configs with a genuine, ranked edge
    assert pbo_cscv(real, 10)["pbo"] < 0.1


def test_spa_rejects_noise_detects_effect():
    rng = np.random.default_rng(2)
    noise = rng.normal(0, 0.01, (2000, 30))
    assert spa_test(noise, 200, 10, 0)["spa_p"] > 0.1
    eff = noise.copy()
    eff[:, 3] += 0.0015
    res = spa_test(eff, 200, 10, 0)
    assert res["spa_p"] < 0.05 and res["best"] == 3


def test_alpha_removes_beta():
    rng = np.random.default_rng(3)
    b = rng.normal(0.0005, 0.01, 3000)
    r = 0.5 * b + rng.normal(0, 0.002, 3000)
    a, beta = alpha_returns(r[:, None], b)
    assert abs(beta[0] - 0.5) < 0.03
    assert abs(a[:, 0].mean()) < 1e-4


def test_split_has_embargo_and_holdout_at_end():
    s = make_split(5000, 0.2, 21, 252)
    assert s.warmup_end == 252 and s.holdout_start - s.dev_end == 21 and s.T == 5000
    assert s.holdout_start == 5000 - round((5000 - 252) * 0.2)


def test_walk_forward_folds_never_overlap_and_respect_embargo():
    folds = walk_forward_folds(252, 4000, 5, 756, 21)
    prev_end = None
    for f in folds:
        assert f.train[1] + 21 == f.test[0]
        assert f.train[0] == 252
        if prev_end is not None:
            assert f.test[0] == prev_end
        prev_end = f.test[1]
    assert folds[-1].test[1] == 4000


def test_effective_trials_counts_independent_bets():
    rng = np.random.default_rng(4)
    base = rng.normal(0, 0.01, (2000, 5))
    copies = np.repeat(base, 20, axis=1) + rng.normal(0, 1e-5, (2000, 100))   # 100 configs, 5 ideas
    assert 4 < effective_trials(copies) < 6
    assert effective_trials(rng.normal(0, 0.01, (2000, 50))) > 45


def test_dsr_uses_null_sampling_variance_by_default():
    r = np.random.default_rng(5).normal(0.0008, 0.01, 4000)
    d = deflated_sharpe(r, 20)
    assert abs(d["sr0_daily"] - expected_max_sharpe(20, 1 / 3999)) < 1e-12
