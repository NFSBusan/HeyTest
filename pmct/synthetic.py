"""Simulated Polymarket-like market for testing the pipeline.

Why a simulator: this cloud session can't reach Polymarket's APIs, and a simulator also
lets us test for leakage against a world where the true answer is known.

Model
-----
Each binary market has a latent "fair" probability b(t) that is a martingale:
    S(t) is Brownian motion with total variance 1 over the market's life,
    b(t) = Phi(S(t) / sqrt(remaining variance)), and outcome = 1{S(end) > 0}.
The traded price is b(t) plus mean-reverting noise in logit space. That noise is what
skilled traders exploit and what bad traders lose to.

Each wallet has:
    skill   s  : >0 trades toward the fair value, <0 trades away from it
    longshot c : >=0 tilt toward buying whichever side is cheap (a known retail bias)
    activity    : Pareto-distributed, so a few wallets make most trades
    lifetime    : wallets appear and disappear; bad wallets churn faster (they go broke)
Takers pay a half-spread. Skill can decay over a wallet's life (`skill_half_life_days`).

`null_world=True` sets every skill and bias to zero. Any profit the strategy finds there
is a bug or luck.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import norm

from .schema import DAY, sort_trades, validate


@dataclass
class SimConfig:
    n_days: int = 540
    n_markets: int = 1600
    n_wallets: int = 4000
    trades_per_market_mean: float = 260.0
    half_spread: float = 0.01
    noise_sd: float = 0.35          # logit-space price noise
    noise_phi: float = 0.85         # AR(1) persistence of noise across trades
    frac_sharp: float = 0.10
    frac_bad: float = 0.30
    skill_half_life_days: float = 365.0
    null_world: bool = False
    seed: int = 7


def simulate(cfg: SimConfig = SimConfig()) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (markets, trades, wallets). `wallets` holds the hidden truth, for diagnostics only."""
    rng = np.random.default_rng(cfg.seed)
    T_end = cfg.n_days * DAY

    # --- wallets ---
    W = cfg.n_wallets
    u = rng.random(W)
    kind = np.where(u < cfg.frac_sharp, "sharp", np.where(u < cfg.frac_sharp + cfg.frac_bad, "bad", "noise"))
    skill = np.where(kind == "sharp", rng.uniform(0.5, 1.5, W),
             np.where(kind == "bad", -rng.uniform(0.3, 1.5, W), rng.normal(0, 0.1, W)))
    longshot = np.where(kind == "bad", rng.uniform(0.0, 0.6, W), rng.uniform(0.0, 0.1, W))
    if cfg.null_world:
        skill[:] = 0.0
        longshot[:] = 0.0
    activity = rng.pareto(1.3, W) + 0.05
    size_med = np.exp(rng.normal(np.log(40), 1.0, W))
    birth = rng.uniform(-0.3 * T_end, T_end * 0.95, W)
    life_mean = np.where(kind == "bad", 120, 300) * DAY
    death = birth + rng.exponential(life_mean)
    birth = np.maximum(birth, 0)

    wallets = pd.DataFrame(dict(wallet=np.arange(W), kind=kind, skill=skill, longshot=longshot,
                                activity=activity, birth=birth, death=death))

    # --- markets ---
    M = cfg.n_markets
    created = rng.uniform(0, T_end - 2 * DAY, M)
    dur = np.clip(np.exp(rng.normal(np.log(18), 0.9, M)), 1, 120) * DAY
    end = created + dur
    res_delay = 2 * 3600 + rng.exponential(0.7 * DAY, M)
    resolved = end + res_delay
    cats = rng.choice(["politics", "sports", "crypto", "other"], M, p=[0.3, 0.35, 0.2, 0.15])
    prior_sd = rng.uniform(0.3, 2.0, M)   # how decided the market is at open

    rows = []
    outcomes = np.full(M, np.nan)
    for m in range(M):
        n = max(5, rng.poisson(cfg.trades_per_market_mean * np.sqrt(dur[m] / (18 * DAY))))
        lo, hi = created[m], min(end[m], T_end)
        if hi - lo < 3600:
            continue
        ts = np.sort(rng.uniform(lo, hi - 1, n))
        # latent martingale
        frac_t = (ts - created[m]) / dur[m]
        total_var = 1.0
        rem = total_var * (1 - frac_t) + 1e-6
        dS = rng.normal(0, np.sqrt(np.diff(np.concatenate([[0.0], frac_t * total_var]))))
        S = prior_sd[m] * rng.normal() + np.cumsum(dS)
        b = norm.cdf(S / np.sqrt(rem))
        S_final = S[-1] + rng.normal(0, np.sqrt(rem[-1]))
        out = float(S_final > 0)
        # price noise AR(1)
        eps = rng.normal(0, cfg.noise_sd * np.sqrt(1 - cfg.noise_phi ** 2), n)
        z = np.empty(n)
        z[0] = rng.normal(0, cfg.noise_sd)
        for i in range(1, n):
            z[i] = cfg.noise_phi * z[i - 1] + eps[i]
        lb = logit(np.clip(b, 1e-4, 1 - 1e-4))
        mid = np.clip(expit(lb + z), 0.02, 0.98)
        # who trades: active wallets weighted by activity
        alive = (wallets.birth.values <= hi) & (wallets.death.values >= lo)
        idx = np.flatnonzero(alive)
        if idx.size == 0:
            continue
        p = activity[idx] / activity[idx].sum()
        w = rng.choice(idx, n, p=p)
        alive_now = (birth[w] <= ts) & (death[w] >= ts)
        if alive_now.sum() < 3:
            continue
        ts, mid, lb, w = ts[alive_now], mid[alive_now], lb[alive_now], w[alive_now]
        age = (ts - birth[w]) / DAY
        decay = 0.5 ** (age / cfg.skill_half_life_days)
        lm = logit(mid)
        score = 3.0 * skill[w] * decay * (lb - lm) - longshot[w] * lm * 2.0
        pdir = expit(score)
        d = np.where(rng.random(ts.size) < pdir, 1, -1).astype(np.int8)
        yes_px = np.clip(mid + d * cfg.half_spread, 0.005, 0.995)
        usd = size_med[w] * np.exp(rng.normal(0, 0.8, ts.size))
        side_px = np.where(d == 1, yes_px, 1 - yes_px)
        rows.append(pd.DataFrame(dict(ts=ts.astype(np.int64), market_id=m, wallet=w, dir=d,
                                      yes_price=yes_px, usd=usd, shares=usd / side_px)))
        if resolved[m] <= T_end:
            outcomes[m] = out
        # markets that resolve after the data horizon stay unresolved (NaN)

    trades = sort_trades(pd.concat(rows, ignore_index=True))
    markets = pd.DataFrame(dict(market_id=np.arange(M), created_ts=created.astype(np.int64),
                                end_ts=end.astype(np.int64) + 1, resolved_ts=resolved.astype(np.int64),
                                outcome=outcomes, category=cats))
    trades = trades[trades.market_id.isin(markets.market_id)]
    validate(markets, trades)
    return markets, trades, wallets
