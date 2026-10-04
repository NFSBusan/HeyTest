"""Multiple-testing correction: Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014)."""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

EULER = 0.5772156649


def expected_max_sharpe(n_trials: int, sr_var: float) -> float:
    """Expected max of n_trials Sharpe estimates when the true Sharpe is zero (per-period units)."""
    if n_trials < 2:
        return 0.0
    return float(np.sqrt(sr_var) * ((1 - EULER) * norm.ppf(1 - 1 / n_trials)
                                    + EULER * norm.ppf(1 - 1 / (n_trials * np.e))))


def deflated_sharpe(sr: float, n_obs: int, skew: float, kurt: float, n_trials: int, sr_var: float) -> float:
    """P(true Sharpe > 0) after accounting for n_trials tries. Per-period (daily) Sharpe in."""
    sr0 = expected_max_sharpe(n_trials, sr_var)
    den = np.sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr ** 2, 1e-12))
    return float(norm.cdf((sr - sr0) * np.sqrt(max(n_obs - 1, 1)) / den))
