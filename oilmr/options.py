"""Does a correct spread forecast make money through a defined-risk option?

Model ("perfect-signal world"), arithmetic dollars per barrel, horizon T years:

    S_t  = Brent - WTI, an Ornstein-Uhlenbeck process with mean mu, speed kappa, vol sig_s
    C_t  = common oil factor, Brownian with vol sig_c (moves both legs equally)
    W_T  = W_0 + C_T + phi * (S_0 - S_T)
    B_T  = B_0 + C_T - (1 - phi) * (S_0 - S_T)          so B_T - W_T = S_T

phi is the share of spread convergence that happens through WTI moving (phi=1: WTI does all
the moving; phi=0: Brent does all of it). The trader knows mu and kappa exactly, which is far
more than anyone knows in practice. Option sellers know each leg's true terminal variance but
price with zero drift (risk-neutral futures). So the only edge is the drift the trader knows
about. If a structure loses here, it will lose in reality.

`priced_in` relaxes the zero-drift assumption: the option seller's mean for each leg is
priced_in * (trader's mean). Spot Brent-WTI reversion that is forecastable from public data
should already show up in the deferred futures spread, so priced_in near 1 is the efficient-
market case, and then no structure has an edge.

Everything is closed-form Gaussian, so the expected values are exact (no simulation noise in
the means); a Monte Carlo is used only for standard deviation and hit rate.
"""
from dataclasses import dataclass, replace

import numpy as np
from scipy.stats import norm

TRADING_DAYS = 252


@dataclass(frozen=True)
class World:
    w0: float = 70.0            # WTI futures price
    mu: float = 4.0             # true long-run spread
    s0_sigmas: float = 2.0      # entry: spread is this many stationary sds above mu
    half_life_days: float = 30  # spread half-life in trading days
    spread_daily_sd: float = 0.60   # $ sd of daily spread changes
    wti_ann_vol: float = 0.35   # lognormal vol of WTI, converted to $ vol at w0
    phi: float = 0.5            # share of convergence through WTI
    rho: float = 0.0            # corr(common-factor shock, spread shock)
    horizon_days: int = 21
    cost_frac: float = 0.05     # round-trip bid/ask + fees per option leg, fraction of premium
    priced_in: float = 0.0      # share of expected convergence already in futures/option prices

    @property
    def T(self):
        return self.horizon_days / TRADING_DAYS

    @property
    def kappa(self):
        return np.log(2) / (self.half_life_days / TRADING_DAYS)

    @property
    def sig_s(self):
        return self.spread_daily_sd * np.sqrt(TRADING_DAYS)

    @property
    def sig_c(self):
        return self.wti_ann_vol * self.w0

    @property
    def stat_sd(self):
        return self.sig_s / np.sqrt(2 * self.kappa)

    @property
    def s0(self):
        return self.mu + self.s0_sigmas * self.stat_sd

    @property
    def b0(self):
        return self.w0 + self.s0


def terminal_moments(w: World):
    """Mean vector and covariance of (C_T, S_T) under the trader's (physical) measure."""
    k, T = w.kappa, w.T
    m_s = w.mu + (w.s0 - w.mu) * np.exp(-k * T)
    v_s = w.sig_s ** 2 * (1 - np.exp(-2 * k * T)) / (2 * k)
    v_c = w.sig_c ** 2 * T
    cov = w.rho * w.sig_c * w.sig_s * (1 - np.exp(-k * T)) / k
    return np.array([0.0, m_s]), np.array([[v_c, cov], [cov, v_s]])


def leg_loadings(w: World):
    """W_T - W_0 and B_T - B_0 and S_T - S_0 as linear maps a + L @ (C_T, S_T)."""
    return {
        "wti": (w.phi * w.s0, np.array([1.0, -w.phi])),
        "brent": (-(1 - w.phi) * w.s0, np.array([1.0, 1 - w.phi])),
        "spread": (-w.s0, np.array([0.0, 1.0])),
    }


def _gauss_call(m, sd, k):
    """E[(X-k)+] for X ~ N(m, sd^2)."""
    d = (m - k) / sd
    return (m - k) * norm.cdf(d) + sd * norm.pdf(d)


def _gauss_put(m, sd, k):
    d = (k - m) / sd
    return (k - m) * norm.cdf(d) + sd * norm.pdf(d)


def leg_change_dist(w: World, leg):
    mean, cov = terminal_moments(w)
    a, L = leg_loadings(w)[leg]
    return a + L @ mean, float(np.sqrt(L @ cov @ L))


# A structure is a list of (leg, kind, strike_offset_in_sd, quantity). Strike offsets are in
# units of that leg's terminal sd from the current price, so "0" is at-the-money.
STRUCTURES = {
    "wti_call_atm": [("wti", "call", 0.0, 1)],
    "wti_call_spread_0_1": [("wti", "call", 0.0, 1), ("wti", "call", 1.0, -1)],
    "brent_put_atm": [("brent", "put", 0.0, 1)],
    "brent_put_spread_0_1": [("brent", "put", 0.0, 1), ("brent", "put", 1.0, -1)],
    "wti_call_plus_brent_put": [("wti", "call", 0.0, 1), ("brent", "put", 0.0, 1)],
    # A listed Brent-WTI spread option may not exist with usable liquidity; this is the
    # upper bound on what option convexity can do with the signal.
    "spread_put_atm": [("spread", "put", 0.0, 1)],
    "spread_put_spread_0_1": [("spread", "put", 0.0, 1), ("spread", "put", 1.0, -1)],
}


def evaluate(w: World, legs):
    """Exact premium (risk-neutral, zero drift), exact expected payoff (physical), and max
    loss. Costs are charged on every leg's premium, long or short."""
    premium = expected = cost = 0.0
    for leg, kind, off, q in legs:
        m_p, sd_p = leg_change_dist(w, leg)
        # Seller prices with the true variance and priced_in of the drift. Strikes are set
        # relative to the seller's forward, so "atm" means at the forward.
        m_q = w.priced_in * m_p
        k = m_q + (off * sd_p if kind == "call" else -off * sd_p)
        f = _gauss_call if kind == "call" else _gauss_put
        price = f(m_q, sd_p, k)
        premium += q * price
        expected += q * f(m_p, sd_p, k)
        cost += abs(q) * price * w.cost_frac
    pnl = expected - premium - cost
    return {"premium": premium, "exp_payoff": expected, "cost": cost, "exp_pnl": pnl,
            "exp_pnl_pct_of_debit": pnl / premium, "max_loss": premium + cost}


def simulate_pnl(w: World, legs, n=200_000, seed=0):
    """Monte Carlo P&L distribution for sd, hit rate and trades-to-detect."""
    rng = np.random.default_rng(seed)
    mean, cov = terminal_moments(w)
    x = rng.multivariate_normal(mean, cov, size=n)
    pnl = np.zeros(n)
    fixed = 0.0
    for leg, kind, off, q in legs:
        a, L = leg_loadings(w)[leg]
        m_p, sd = leg_change_dist(w, leg)
        m_q = w.priced_in * m_p
        k = m_q + (off * sd if kind == "call" else -off * sd)
        f = _gauss_call if kind == "call" else _gauss_put
        price = f(m_q, sd, k)
        dx = a + x @ L
        pay = np.maximum(dx - k, 0) if kind == "call" else np.maximum(k - dx, 0)
        pnl += q * (pay - price)
        fixed += abs(q) * price * w.cost_frac
    pnl -= fixed
    m, s = pnl.mean(), pnl.std()
    return {"pnl_sd": s, "p_profit": float((pnl > 0).mean()),
            "trades_for_t2": float((2 * s / m) ** 2) if m > 0 else np.inf}


def futures_spread_benchmark(w: World):
    """Short the spread with futures (NOT allowed for trading; reference only)."""
    mean, cov = terminal_moments(w)
    m, sd = (1 - w.priced_in) * (w.s0 - mean[1]), float(np.sqrt(cov[1, 1]))
    return {"exp_pnl": m, "pnl_sd": sd, "trades_for_t2": (2 * sd / m) ** 2}


def table(w: World):
    rows = []
    for name, legs in STRUCTURES.items():
        r = {"structure": name, **evaluate(w, legs), **simulate_pnl(w, legs)}
        rows.append(r)
    return rows


def sweep(base: World, **grids):
    """Cartesian sweep over World fields; returns list of (overrides, rows)."""
    import itertools
    keys = list(grids)
    for combo in itertools.product(*(grids[k] for k in keys)):
        ov = dict(zip(keys, combo))
        yield ov, replace(base, **ov)
