"""Statistics for telling a real edge from selection luck.

- Probabilistic / Deflated Sharpe ratio (Bailey & Lopez de Prado 2012, 2014)
- Probability of Backtest Overfitting via CSCV (Bailey, Borwein, Lopez de Prado, Zhu 2015)
- White's Reality Check and Hansen's SPA test with the stationary bootstrap (Politis & Romano 1994)
- Alpha: returns minus what the strategy's market exposure (beta) would have earned

Sharpe ratios inside this module are per-period (daily), not annualised, unless noted.
"""
from __future__ import annotations

import itertools

import numpy as np
from numba import njit
from scipy.stats import kurtosis, norm, skew

EULER = 0.5772156649015329


# ------------------------------------------------------------------ alpha

def alpha_returns(R: np.ndarray, bench: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Column-wise OLS on the benchmark: returns (alpha series incl. intercept, betas)."""
    R = np.atleast_2d(R.T).T if R.ndim == 1 else R
    b = bench - bench.mean()
    var = (b * b).sum()
    beta = (b[:, None] * (R - R.mean(axis=0))).sum(axis=0) / var if var > 0 else np.zeros(R.shape[1])
    return R - bench[:, None] * beta, beta


# ------------------------------------------------------------------ Sharpe significance

def sharpe_daily(R: np.ndarray) -> np.ndarray:
    sd = R.std(axis=0, ddof=1)
    return np.divide(R.mean(axis=0), sd, out=np.zeros(R.shape[1] if R.ndim > 1 else 1), where=sd > 1e-12)


def psr(sr: float, n: int, sk: float, ku: float, sr_benchmark: float = 0.0) -> float:
    """P(true Sharpe > sr_benchmark) given an observed daily Sharpe over n days."""
    den = np.sqrt(max(1 - sk * sr + (ku - 1) / 4 * sr ** 2, 1e-12))
    return float(norm.cdf((sr - sr_benchmark) * np.sqrt(max(n - 1, 1)) / den))


def expected_max_sharpe(n_trials: int, sr_var: float) -> float:
    """Expected maximum of n_trials daily Sharpe estimates when every true Sharpe is zero."""
    if n_trials < 2 or sr_var <= 0:
        return 0.0
    return float(np.sqrt(sr_var) * ((1 - EULER) * norm.ppf(1 - 1 / n_trials)
                                    + EULER * norm.ppf(1 - 1 / (n_trials * np.e))))


def effective_trials(R: np.ndarray) -> float:
    """Effective number of independent configs: participation ratio of the correlation eigenvalues.

    K independent series give ~K; K copies of one series give 1. Thousands of grid neighbours
    of the same idea are closer to a handful of independent bets than to thousands.
    """
    R = np.asarray(R, dtype=float)
    keep = R.std(axis=0) > 1e-12
    R = R[:, keep]
    K = R.shape[1]
    if K < 2:
        return float(K)
    lam = np.clip(np.linalg.eigvalsh(np.corrcoef(R, rowvar=False)), 0, None)
    return float(min(K, lam.sum() ** 2 / (lam ** 2).sum()))


def deflated_sharpe(r: np.ndarray, n_trials: float, trial_sr_var: float | None = None) -> dict:
    """DSR of one return series after `n_trials` (effective) independent attempts.

    trial_sr_var=None uses the sampling variance of a daily Sharpe estimate under the null
    (1 / (n - 1)). Passing the empirical cross-trial variance instead is the textbook variant;
    it becomes far too strict when many configs share a real edge, because then the spread of
    Sharpe ratios across trials is real dispersion, not luck.
    """
    r = np.asarray(r, dtype=float)
    n = len(r)
    sd = r.std(ddof=1)
    sr = r.mean() / sd if sd > 1e-12 else 0.0
    sk = float(skew(r)) if sd > 0 else 0.0
    ku = float(kurtosis(r, fisher=False)) if sd > 0 else 3.0
    var = 1.0 / max(n - 1, 1) if trial_sr_var is None else trial_sr_var
    sr0 = expected_max_sharpe(int(round(n_trials)), var)
    return {"dsr": psr(sr, n, sk, ku, sr0), "psr": psr(sr, n, sk, ku, 0.0), "sr_daily": float(sr),
            "sr0_daily": sr0, "sr0_annual": sr0 * np.sqrt(252), "n_trials": float(n_trials)}


# ------------------------------------------------------------------ PBO (CSCV)

def pbo_cscv(R: np.ndarray, n_splits: int = 10) -> dict:
    """Probability that the in-sample best config ranks below the median out of sample.

    R: (T, K) returns of K configurations over the same days. ~0.5 or more means the selection
    procedure is no better than picking at random.
    """
    T, K = R.shape
    if K < 2:
        return {"pbo": float("nan"), "n_combos": 0, "median_logit": float("nan")}
    blocks = np.array_split(np.arange(T), n_splits)
    s1 = np.stack([R[b].sum(axis=0) for b in blocks])
    s2 = np.stack([(R[b] ** 2).sum(axis=0) for b in blocks])
    cnt = np.array([len(b) for b in blocks], dtype=float)
    logits = []
    for is_idx in itertools.combinations(range(n_splits), n_splits // 2):
        is_mask = np.zeros(n_splits, dtype=bool)
        is_mask[list(is_idx)] = True

        def sr(mask):
            n = cnt[mask].sum()
            m = s1[mask].sum(axis=0) / n
            v = s2[mask].sum(axis=0) / n - m ** 2
            return np.divide(m, np.sqrt(np.maximum(v, 0)), out=np.zeros(K), where=v > 1e-18)
        sr_is, sr_oos = sr(is_mask), sr(~is_mask)
        best = int(np.argmax(sr_is))
        rank = (sr_oos < sr_oos[best]).sum() + 0.5 * ((sr_oos == sr_oos[best]).sum() - 1) + 1
        w = rank / (K + 1)
        logits.append(np.log(w / (1 - w)))
    logits = np.array(logits)
    return {"pbo": float(np.mean(logits <= 0)), "n_combos": len(logits),
            "median_logit": float(np.median(logits))}


# ------------------------------------------------------------------ bootstrap tests

@njit(cache=True)
def _chain(jumps, starts, n):
    idx = np.empty(n, dtype=np.int64)
    idx[0] = starts[0]
    for t in range(1, n):
        idx[t] = starts[t] if jumps[t] else (idx[t - 1] + 1) % n
    return idx


def stationary_bootstrap_indices(n: int, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    """Politis-Romano: blocks of geometric length (mean `mean_block`), wrapping around."""
    jumps = rng.random(n) < 1.0 / mean_block
    starts = rng.integers(n, size=n)
    return _chain(jumps, starts, n)


def spa_test(D: np.ndarray, reps: int = 500, mean_block: float = 20, seed: int = 0) -> dict:
    """Hansen's SPA (consistent version) and White's Reality Check.

    D: (T, K) performance differentials vs the benchmark (here: daily alpha, so the null is
    'no configuration has positive expected alpha'). Returns p-values; small = some config
    beats the benchmark even after accounting for having searched all K.
    """
    D = np.asarray(D, dtype=np.float64)
    if D.ndim == 1:
        D = D[:, None]
    n, K = D.shape
    rng = np.random.default_rng(seed)
    dbar = D.mean(axis=0)
    boots = np.empty((reps, K))
    for b in range(reps):
        boots[b] = D[stationary_bootstrap_indices(n, mean_block, rng)].mean(axis=0)
    omega = np.sqrt(np.maximum(n * boots.var(axis=0), 1e-18))
    t_stat = max(float(np.max(np.sqrt(n) * dbar / omega)), 0.0)
    thresh = -np.sqrt(omega ** 2 / n * 2 * np.log(np.log(n)))
    mu_c = np.where(dbar >= thresh, dbar, 0.0)
    z_spa = np.sqrt(n) * (boots - mu_c) / omega
    z_rc = np.sqrt(n) * (boots - dbar)
    t_spa = np.maximum(z_spa.max(axis=1), 0.0)
    rc_stat = float(np.sqrt(n) * dbar.max())
    return {"spa_p": float(np.mean(t_spa >= t_stat)), "rc_p": float(np.mean(z_rc.max(axis=1) >= rc_stat)),
            "t_stat": t_stat, "best": int(np.argmax(np.sqrt(n) * dbar / omega)), "K": K}


def monte_carlo(r: np.ndarray, reps: int = 1000, mean_block: float = 20, seed: int = 0) -> dict:
    """Block-bootstrap the daily returns: how bad could CAGR / drawdown / Sharpe plausibly be?"""
    r = np.asarray(r, dtype=float)
    n = len(r)
    rng = np.random.default_rng(seed)
    sr, cagr, mdd = np.empty(reps), np.empty(reps), np.empty(reps)
    for k in range(reps):
        x = r[stationary_bootstrap_indices(n, mean_block, rng)]
        sd = x.std(ddof=1)
        sr[k] = x.mean() / sd * np.sqrt(252) if sd > 0 else 0.0
        eq = np.cumprod(1 + x)
        cagr[k] = eq[-1] ** (252 / n) - 1 if eq[-1] > 0 else -1.0
        mdd[k] = np.max(1 - eq / np.maximum.accumulate(np.concatenate([[1.0], eq]))[1:])
    q = lambda a, p: float(np.quantile(a, p))
    return {"sharpe_p05": q(sr, 0.05), "sharpe_p50": q(sr, 0.5), "p_sharpe_below_0": float(np.mean(sr < 0)),
            "cagr_p05": q(cagr, 0.05), "cagr_p50": q(cagr, 0.5),
            "mdd_p50": q(mdd, 0.5), "mdd_p95": q(mdd, 0.95)}


def rolling_return_quantile(r: np.ndarray, window: int, q: float, reps: int = 1000, mean_block: float = 20,
                            seed: int = 0) -> float:
    """Quantile of the `window`-day compounded return under block resampling of `r`.

    The paper trader compares its live rolling return with this to detect deterioration.
    """
    r = np.asarray(r, dtype=float)
    rng = np.random.default_rng(seed)
    out = np.empty(reps)
    for k in range(reps):
        idx = stationary_bootstrap_indices(len(r), mean_block, rng)[:window]
        out[k] = np.prod(1 + r[idx]) - 1
    return float(np.quantile(out, q))
