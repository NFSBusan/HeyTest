"""Does the Brent-WTI spread mean-revert, how fast, and is that stable over time?

All functions take a price DataFrame from data.load_pair. Nothing here looks ahead:
rolling statistics use only past windows.
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint


def half_life(x):
    """OU half-life in observations from the AR(1) regression dx_t = a + b x_{t-1} + e.
    Returns inf when b >= 0 (no reversion)."""
    x = pd.Series(x).dropna()
    dx, lag = x.diff().iloc[1:], x.shift(1).iloc[1:]
    b = np.polyfit(lag.values, dx.values, 1)[0]
    return np.inf if b >= 0 else float(np.log(2) / -b)


def full_sample_report(df):
    s = df["spread"]
    adf = adfuller(s, autolag="AIC")
    eg = coint(df["brent"], df["wti"])
    return {
        "n_obs": int(len(s)),
        "start": str(s.index[0].date()),
        "end": str(s.index[-1].date()),
        "spread_mean": float(s.mean()),
        "spread_std": float(s.std()),
        "adf_stat": float(adf[0]),
        "adf_p": float(adf[1]),
        "engle_granger_p": float(eg[1]),
        "half_life_days": half_life(s),
        "corr_daily_price_changes": float(df["brent"].diff().corr(df["wti"].diff())),
        "spread_change_std": float(s.diff().std()),
        "wti_change_std": float(df["wti"].diff().std()),
    }


def rolling_report(df, window=504, step=21):
    """Rolling ADF p-value, half-life and mean of the spread. If these swing wildly the
    'equilibrium' is not a fixed number, and a full-sample Z-score is look-ahead."""
    s = df["spread"]
    rows = []
    for end in range(window, len(s) + 1, step):
        w = s.iloc[end - window:end]
        rows.append({"date": w.index[-1], "mean": w.mean(), "std": w.std(),
                     "adf_p": adfuller(w, autolag="AIC")[1], "half_life": half_life(w)})
    return pd.DataFrame(rows).set_index("date")


def by_regime(df, breaks=("2010-12-31", "2015-12-18", "2020-03-31")):
    """Summary per period between known structural events (Cushing glut, US export-ban
    repeal, COVID). These dates are chosen from history, so treat this as descriptive only."""
    edges = [df.index[0]] + [pd.Timestamp(b) for b in breaks] + [df.index[-1] + pd.Timedelta(days=1)]
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sub = df[(df.index >= lo) & (df.index < hi)]
        if len(sub) < 100:
            continue
        out.append({"from": str(sub.index[0].date()), "to": str(sub.index[-1].date()),
                    "mean": sub["spread"].mean(), "std": sub["spread"].std(),
                    "adf_p": adfuller(sub["spread"], autolag="AIC")[1],
                    "half_life": half_life(sub["spread"])})
    return pd.DataFrame(out)


def spread_share_of_outright(df):
    """How much of a WTI price move is explained by the spread move?
    R^2 of dW on dS. If small, a WTI option is mostly a bet on oil's level, not the spread."""
    d = df[["wti", "spread"]].diff().dropna()
    r = d["wti"].corr(d["spread"])
    return {"corr_dWTI_dSpread": float(r), "r2": float(r * r)}
