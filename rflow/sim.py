"""Simulated stock + options panel driven by retail attention waves.

Why a simulator: this cloud session can't reach any market-data host, and a world with a known
answer is the only way to test for leakage. Results on it are NOT evidence of real profit.

Model (one row per ticker-day)
------------------------------
Each ticker has a hidden "retail propensity" p in (0, 1). Low price, small cap and high short
interest are observable and correlated with p (that is what makes meme names).

Attention A(t) >= 0 comes in episodes ("waves"). A wave starts with a hazard that rises with p,
with yesterday's absolute return and with short interest, then grows logistically for g days
to a peak and decays with half-life h. So early in a wave, attention keeps rising: that is the
"is going to have a lot of retail" signal a model can learn. Observables:
    mentions      ~ Poisson(base * exp(2.2 * A))         (Reddit / social mentions)
    offex_share   = off-exchange (wholesaler) share of volume, rises with A
    volume        = base * exp(1.1 * A + noise)
    call/put vol  : call share rises with A (retail buys calls)
    iv30          = E[vol over next 30d] * (1 + vrp(A)) + noise, where vrp(A) is the premium
                    retail call demand puts on options; it rises with A
Returns: market factor + idiosyncratic vol (scaled up by attention) + PRICE PRESSURE. Retail
net buying (dA > 0) pushes the price up; the pressure reverses with half-life `rev_hl`.

`null_world=True`: no price pressure and an attention-free, zero vol premium. Attention still
exists and is still predictable, but nothing about it can be traded for profit.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SimConfig:
    n_tickers: int = 300
    n_days: int = 756                 # ~3 years of trading days
    wave_rate: float = 0.012          # base daily wave hazard at p = 1
    grow_days: tuple = (3, 9)         # days from wave start to peak
    decay_hl: tuple = (3.0, 10.0)     # half-life of attention after the peak
    pressure: float = 0.05            # log-price push per unit of attention increase
    rev_hl: float = 8.0               # half-life of price-pressure reversal (days)
    vrp_base: float = 0.04            # baseline implied-over-realized premium
    vrp_attn: float = 0.12            # extra premium per unit of attention
    vol_attn: float = 0.6             # realized vol scales by (1 + vol_attn * A)
    iv_noise: float = 0.03
    null_world: bool = False
    seed: int = 7


def simulate(cfg: SimConfig = SimConfig()) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (panel, tickers). Panel columns ending in `_true` are hidden truth (diagnostics only)."""
    rng = np.random.default_rng(cfg.seed)
    N, T = cfg.n_tickers, cfg.n_days
    p = rng.beta(0.6, 2.5, N)                                   # most names barely retail, a few very
    log_mcap = 23.0 - 3.0 * p + rng.normal(0, 0.8, N)          # ~ $10B median, lower if retail-prone
    short_int = np.clip(0.03 + 0.25 * p * rng.random(N) + rng.normal(0, 0.02, N), 0.005, 0.6)
    sigma = np.exp(rng.normal(np.log(0.35), 0.3, N)) * (1 + 1.2 * p)   # annual vol
    price0 = np.exp(rng.normal(3.3, 0.8, N) - 1.5 * p)
    base_ment = np.exp(rng.normal(0.0, 1.0, N) + 2.5 * p)
    base_vol = np.exp(log_mcap - 5.5 + rng.normal(0, 0.4, N))
    base_offex = np.clip(0.38 + 0.12 * p + rng.normal(0, 0.03, N), 0.2, 0.8)

    A = np.zeros((T, N))
    logp = np.zeros((T, N))
    ret = np.zeros((T, N))
    press = np.zeros(N)
    logp[0] = np.log(price0)
    mkt = rng.normal(0.0003, 0.011, T)
    beta = rng.normal(1.0, 0.3, N)
    # active wave state
    w_start = np.full(N, -1)
    w_peak = np.zeros(N)
    w_g = np.ones(N)
    w_hl = np.ones(N)
    rev = 0.5 ** (1 / cfg.rev_hl)
    for t in range(1, T):
        prev_absret = np.abs(ret[t - 1]) / (sigma / np.sqrt(252))
        haz = cfg.wave_rate * p * (1 + 0.5 * np.clip(prev_absret - 1, 0, 4)) * (1 + 4 * short_int)
        idle = (w_start < 0) | ((t - w_start) > w_g + 6 * w_hl)
        new = idle & (rng.random(N) < haz)
        w_start[new] = t
        w_peak[new] = rng.gamma(2.0, 0.6, new.sum()) * (0.5 + p[new])
        w_g[new] = rng.integers(cfg.grow_days[0], cfg.grow_days[1] + 1, new.sum())
        w_hl[new] = rng.uniform(*cfg.decay_hl, new.sum())
        age = t - w_start
        active = w_start >= 0
        grow = active & (age <= w_g)
        a = np.zeros(N)
        # logistic ramp 0 -> peak over g days, then exponential decay
        x = np.where(grow, age / w_g, 1.0)
        a = np.where(grow, w_peak / (1 + np.exp(-8 * (x - 0.5))), a)
        dec = active & (age > w_g)
        a = np.where(dec, w_peak * 0.5 ** ((age - w_g) / w_hl), a)
        A[t] = a
        dA = A[t] - A[t - 1]
        vol_d = sigma / np.sqrt(252) * (1 + cfg.vol_attn * A[t])
        eps = rng.standard_t(4, N) / np.sqrt(2.0)
        push = 0.0 if cfg.null_world else cfg.pressure * dA
        new_press = press * rev + push
        r = beta * mkt[t] + vol_d * eps + (new_press - press)
        press = new_press
        ret[t] = r
        logp[t] = logp[t - 1] + r

    # observables
    mentions = rng.poisson(base_ment * np.exp(2.2 * A))
    vol_noise = rng.normal(0, 0.25, (T, N))
    volume = base_vol * np.exp(1.1 * A + vol_noise + 2.0 * np.abs(ret) / (sigma / np.sqrt(252)) * 0.1)
    offex = np.clip(base_offex + 0.08 * np.tanh(A) + rng.normal(0, 0.025, (T, N)), 0.05, 0.95)
    call_share = np.clip(0.55 + 0.12 * np.tanh(A) + rng.normal(0, 0.04, (T, N)), 0.05, 0.95)
    opt_vol = volume * 0.05 * np.exp(0.8 * A) * rng.lognormal(0, 0.3, (T, N))
    # forward-looking implied vol: market's expectation of next-30d vol (attention decays ~10d)
    exp_A = A * 0.45
    fair_iv = sigma * (1 + cfg.vol_attn * exp_A)
    vrp = np.zeros_like(A) if cfg.null_world else cfg.vrp_base + cfg.vrp_attn * A
    iv = fair_iv * (1 + vrp) * np.exp(rng.normal(0, cfg.iv_noise, (T, N)))
    # call skew: retail call demand makes upside strikes richer (per unit log-moneyness)
    skew = (0.0 if cfg.null_world else 0.25) * np.tanh(A) + rng.normal(0, 0.03, (T, N))

    day = np.repeat(np.arange(T), N)
    tick = np.tile(np.arange(N), T)
    panel = pd.DataFrame(dict(
        day=day, ticker=[f"T{i:03d}" for i in tick],
        close=np.exp(logp).ravel(), ret=ret.ravel(), volume=volume.ravel(),
        mentions=mentions.ravel(), offex_share=offex.ravel(),
        call_vol=(opt_vol * call_share).ravel(), put_vol=(opt_vol * (1 - call_share)).ravel(),
        iv30=iv.ravel(), call_skew=skew.ravel(),
        log_mcap=log_mcap[tick], short_int=short_int[tick],
        attn_true=A.ravel(),
    ))
    tickers = pd.DataFrame(dict(ticker=[f"T{i:03d}" for i in range(N)], p_true=p, sigma_true=sigma))
    return panel, tickers
