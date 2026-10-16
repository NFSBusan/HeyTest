"""Do retail-style attention spikes predict what happens next? Real IBKR daily prices.

Pre-registered (written before looking at results; see RETAIL_FLOW_LOG.md):
  Universe  : every data/prices/*.csv except SPY (SPY is the market factor).
  Spike     : volume >= 3x its median over the prior 60 days (strictly before the day).
              At most one spike per ticker per 10 trading days (later ones are skipped).
  Type      : "up" if the spike-day return > +2 sd of the prior 60d returns, "down" if < -2 sd, else "flat".
  Entry     : next day's OPEN (we only know the volume after the close). Exit: close of day t+h, h in 1,5,10,20.
  Return    : abnormal = stock log return - beta * SPY log return over the same window,
              beta from the prior 120 days (point-in-time).
  Vol ratio : realized vol over t+1..t+10 / realized vol over the prior 20 days.
  Inference : events cluster in time (crypto miners spike together), so t-stats come from a
              bootstrap that resamples calendar WEEKS, not single events.
  Split     : dev = 2021-10-18 .. 2024-12-31; holdout = 2025-01-01 .. end. Holdout read once.
  Baseline  : same measures on every non-spike day, so "after a spike" is compared to "any day".

python scripts/retail_spike_study.py [--holdout]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
H = [1, 5, 10, 20]
DEV_END = pd.Timestamp("2024-12-31")


def load():
    px = {}
    for f in sorted((ROOT / "data" / "prices").glob("*.csv")):
        px[f.stem] = pd.read_csv(f, parse_dates=["date"]).set_index("date")
    spy = px.pop("SPY")
    return px, spy


def events_for(t: str, d: pd.DataFrame, spy: pd.DataFrame) -> pd.DataFrame:
    d = d.join(spy[["open", "close"]].add_prefix("spy_"), how="inner")
    r = np.log(d.close).diff()
    rm = np.log(d.spy_close).diff()
    # point-in-time stats: windows end the day BEFORE t
    med_vol = d.volume.shift(1).rolling(60, min_periods=40).median()
    sd = r.shift(1).rolling(60, min_periods=40).std()
    cov = r.shift(1).rolling(120, min_periods=60).cov(rm.shift(1))
    beta = (cov / rm.shift(1).rolling(120, min_periods=60).var()).clip(-1, 4)
    rv_prev = r.shift(1).rolling(20, min_periods=15).std()
    out = pd.DataFrame(index=d.index)
    out["ticker"] = t
    out["vol_x"] = d.volume / med_vol
    out["z"] = r / sd
    out["ret0"] = r
    lo, lm = np.log(d.open), np.log(d.spy_open)
    lc, lmc = np.log(d.close), np.log(d.spy_close)
    for h in H:
        stock = lc.shift(-h) - lo.shift(-1)
        mkt = lmc.shift(-h) - lm.shift(-1)
        out[f"ar{h}"] = stock - beta * mkt
    rv_next = r[::-1].rolling(10, min_periods=10).std()[::-1].shift(-1)
    out["vol_ratio"] = rv_next / rv_prev
    out = out.dropna(subset=["vol_x", "z", "ar20", "vol_ratio"])
    spike = (out.vol_x >= 3).to_numpy()
    keep, last = np.zeros(len(out), bool), -10 ** 9
    for i in np.flatnonzero(spike):
        if i - last >= 10:
            keep[i], last = True, i
    out["spike"] = keep
    out["type"] = np.where(out.z > 2, "up", np.where(out.z < -2, "down", "flat"))
    return out


def week_boot(df: pd.DataFrame, col: str, n=2000, seed=0):
    """Mean and t-stat with weeks as the resampling unit."""
    wk = df.index.to_period("W")
    g = df.groupby(wk)[col].agg(["sum", "count"])
    s, c = g["sum"].to_numpy(), g["count"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(s), (n, len(s)))
    boots = s[idx].sum(1) / c[idx].sum(1)
    m = s.sum() / c.sum()
    return m, m / boots.std(ddof=1), len(df)


def summarize(all_ev: pd.DataFrame, lo, hi) -> pd.DataFrame:
    w = all_ev[(all_ev.index >= lo) & (all_ev.index <= hi)]
    rows = []
    groups = [("all days (baseline)", w[~w.spike])] + [(f"spike: {k}", w[w.spike & (w.type == k)])
                                                     for k in ["up", "flat", "down"]] + [("spike: any", w[w.spike])]
    for name, g in groups:
        row = {"group": name, "n": len(g), "tickers": g.ticker.nunique()}
        for h in H:
            m, t, _ = week_boot(g, f"ar{h}")
            row[f"ar{h}_%"], row[f"t{h}"] = round(100 * m, 2), round(t, 2)
        lv = np.log(g.vol_ratio)
        row["vol_ratio_med"] = round(float(np.exp(lv.median())), 2)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", action="store_true", help="also report the holdout (do this once)")
    a = ap.parse_args()
    px, spy = load()
    ev = pd.concat([events_for(t, d, spy) for t, d in px.items()]).sort_index()
    out = ROOT / "results"
    dev = summarize(ev, ev.index.min(), DEV_END)
    print("DEV 2021-10 .. 2024-12  (abnormal returns in %, week-bootstrap t)\n" + dev.to_string(index=False))
    dev.to_csv(out / "spike_study_dev.csv", index=False)
    by_t = (ev[ev.spike & (ev.index <= DEV_END)].groupby("ticker")
            .agg(n=("ar5", "size"), ar5_pct=("ar5", lambda x: round(100 * x.mean(), 2)),
                 ar20_pct=("ar20", lambda x: round(100 * x.mean(), 2))))
    print("\nDEV spikes by ticker\n" + by_t.to_string())
    if a.holdout:
        ho = summarize(ev, DEV_END + pd.Timedelta(days=1), ev.index.max())
        print("\nHOLDOUT 2025-01 .. 2026-10\n" + ho.to_string(index=False))
        ho.to_csv(out / "spike_study_holdout.csv", index=False)
    spikes = ev[ev.spike].copy()
    spikes.to_csv(out / "spike_events.csv")
    (out / "spike_study_meta.json").write_text(json.dumps(
        {"tickers": sorted(px), "n_spikes": int(ev.spike.sum()), "holdout_reported": a.holdout}, indent=1))


if __name__ == "__main__":
    main()
