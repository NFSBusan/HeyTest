"""Walk-forward prediction models (no lookahead).

At each refit day tau we train only on rows whose target window has fully closed:
    day + horizon < tau        (a 5-day forward return from day d is known on day d+5)
plus an `embargo` gap, then predict days [tau, tau + refit_every). Models:
    attn : P(attention rises over the next 5 days)   -> the "will be retail-heavy" scanner
    ret  : expected 5-day log return                  -> direction (pressure, then reversal)
    vrp  : expected log(IV30 / realized vol next 10d)  -> are options rich or cheap?
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from .features import FEATURES, H_ATTN, H_RET, H_VOL

TARGETS = {"attn": ("y_attn", H_ATTN, "clf"), "ret": ("y_ret", H_RET, "reg"), "vrp": ("y_vrp", H_VOL, "reg")}
ATTN_UP = 0.3     # "rises" = future mean log-mentions at least 0.3 above the current 3-day mean


@dataclass(frozen=True)
class WFConfig:
    start: int = 120          # first prediction day (needs history to train on)
    refit_every: int = 21
    embargo: int = 2
    max_train_days: int = 500
    max_iter: int = 150
    lr: float = 0.06
    min_leaf: int = 200


def _fit(kind, X, y, cfg, seed):
    kw = dict(max_iter=cfg.max_iter, learning_rate=cfg.lr, min_samples_leaf=cfg.min_leaf,
              l2_regularization=1.0, random_state=seed)
    if kind == "clf":
        return HistGradientBoostingClassifier(**kw).fit(X, y)
    return HistGradientBoostingRegressor(loss="absolute_error" if kind == "reg" else "squared_error", **kw).fit(X, y)


def walk_forward(df: pd.DataFrame, cfg: WFConfig = WFConfig(), feats=FEATURES, seed=0) -> pd.DataFrame:
    """Adds p_attn, f_ret, f_vrp columns (NaN before cfg.start)."""
    feats = [f for f in feats if f in df and df[f].notna().any()]
    out = df.copy()
    for name in TARGETS:
        out[{"attn": "p_attn", "ret": "f_ret", "vrp": "f_vrp"}[name]] = np.nan
    last = int(df["day"].max())
    day = df["day"].to_numpy()
    X_all = df[feats].to_numpy(np.float64)
    for tau in range(cfg.start, last + 1, cfg.refit_every):
        pred = (day >= tau) & (day < tau + cfg.refit_every)
        if not pred.any():
            continue
        for name, (col, h, kind) in TARGETS.items():
            y = df[col].to_numpy(np.float64)
            tr = (day + h + cfg.embargo < tau) & (day >= tau - cfg.max_train_days) & np.isfinite(y)
            if tr.sum() < 2000:
                continue
            yt = (y[tr] > ATTN_UP).astype(int) if kind == "clf" else np.clip(y[tr], *np.nanquantile(y[tr], [0.01, 0.99]))
            if kind == "clf" and yt.min() == yt.max():
                continue
            m = _fit(kind, X_all[tr], yt, cfg, seed)
            p = m.predict_proba(X_all[pred])[:, 1] if kind == "clf" else m.predict(X_all[pred])
            out.loc[pred, {"attn": "p_attn", "ret": "f_ret", "vrp": "f_vrp"}[name]] = p
    return out


def skill_report(df: pd.DataFrame, lo: int, hi: int) -> dict:
    """Out-of-sample skill on days [lo, hi): AUC for attention, daily rank-IC (mean, t) for ret and vrp."""
    from sklearn.metrics import roc_auc_score
    d = df[(df.day >= lo) & (df.day < hi)]
    rep = {}
    a = d.dropna(subset=["p_attn", "y_attn"])
    if len(a) and a["y_attn"].gt(ATTN_UP).nunique() == 2:
        rep["attn_auc"] = roc_auc_score(a["y_attn"] > ATTN_UP, a["p_attn"])
        rep["attn_auc_heur"] = roc_auc_score(a["y_attn"] > ATTN_UP, a["retail_rising_heur"].fillna(0.5))
    for f, y in [("f_ret", "y_ret"), ("f_vrp", "y_vrp")]:
        b = d.dropna(subset=[f, y])
        if len(b) == 0:
            continue
        ic = b.groupby("day").apply(lambda x: x[f].corr(x[y], method="spearman"), include_groups=False).dropna()
        # overlapping targets: daily ICs are autocorrelated over the horizon, so t uses non-overlapping days
        step = H_RET if f == "f_ret" else H_VOL
        ic_n = ic.iloc[::step]
        rep[f"{f}_ic"] = float(ic.mean())
        rep[f"{f}_ic_t"] = float(ic_n.mean() / (ic_n.std(ddof=1) / np.sqrt(len(ic_n)))) if len(ic_n) > 2 else np.nan
    return rep
