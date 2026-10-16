"""Are options cheap or expensive after a retail-style volume spike? Real IBKR prices + DoltHub IV + FINRA.

Pre-registered (written before running; see RETAIL_FLOW_LOG.md "Study 2"):
  IV tickers : those with DoltHub volatility_history covering the price window: GME, TSLA, NVDA, MU, PLTR.
  Spike      : same as Study 1 (volume >= 3x prior-60d median, one per 10 days, up/down/flat by +-2 sd).
               2x threshold reported as a secondary row.
  IV         : iv_current (30-day implied vol) on the spike day, else the next trading day (d0).
  Realized   : annualized std of daily log returns over the 21 trading days after d0.
  VRP        : ln(IV) - ln(realized).  > 0 means options were priced for MORE movement than happened.
  Straddle   : buy a 21-trading-day ATM straddle at d0 close, hold to expiry.
               price ~= 0.7979 * S * IV * sqrt(21/252) (ATM approximation), payoff |S_T - S_0|.
               Net return subtracts 5% of premium (half-spread on entry; no exit cost at expiry).
  Baseline   : the same measures on every non-spike day of the same tickers.
  Retail split (Study 2b, all 15 tickers, price-only measure): spike is "retail-heavy" if its FINRA
               off-exchange share (TotalVolume / IBKR volume) is above that ticker's prior-60d median.
               Measure: Study 1's vol ratio (realized vol next 10d / prior 20d).
  Inference  : week-bootstrap t-stats (events cluster in time).

python scripts/iv_spike_study.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import retail_spike_study as S1  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
IV_TICKERS = ["GME", "TSLA", "NVDA", "MU", "PLTR"]
N = 21
COST = 0.05


def mark_spikes(ev: pd.DataFrame, thr: float) -> pd.Series:
    out = pd.Series(False, index=ev.index)
    for t, g in ev.groupby("ticker"):
        pos = np.flatnonzero(ev.ticker.to_numpy() == t)
        sp = (g.vol_x >= thr).to_numpy()
        keep, last = np.zeros(len(g), bool), -10 ** 9
        for i in np.flatnonzero(sp):
            if i - last >= 10:
                keep[i], last = True, i
        out.iloc[pos] = keep
    return out


def iv_panel(t: str, spy: pd.DataFrame) -> pd.DataFrame:
    px = pd.read_csv(ROOT / "data" / "prices" / f"{t}.csv", parse_dates=["date"]).set_index("date")
    ev = S1.events_for(t, px, spy)
    iv = pd.read_csv(ROOT / "data" / "vol" / f"{t}.csv", parse_dates=["date"]).set_index("date")["iv_current"]
    d = px.join(iv, how="left")
    r = np.log(d.close).diff()
    # IV at d0 = spike day, else next trading day; realized vol over the N days after d0
    iv0 = d.iv_current.fillna(d.iv_current.shift(-1))
    lag = np.where(d.iv_current.notna(), 0, 1)
    c = d.close.to_numpy()
    rv = np.full(len(d), np.nan)
    sT = np.full(len(d), np.nan)
    s0 = np.full(len(d), np.nan)
    rr = r.to_numpy()
    for i in range(len(d)):
        j = i + lag[i]
        if j + N < len(d):
            rv[i] = np.std(rr[j + 1:j + N + 1], ddof=1) * np.sqrt(252)
            s0[i], sT[i] = c[j], c[j + N]
    out = pd.DataFrame({"iv": iv0, "rv": rv, "s0": s0, "sT": sT}, index=d.index)
    out = ev.join(out, how="inner").dropna(subset=["iv", "rv"])
    out["vrp"] = np.log(out.iv / out.rv)
    prem = 0.7979 * out.s0 * out.iv * np.sqrt(N / 252)
    out["straddle_gross"] = np.abs(out.sT - out.s0) / prem - 1
    out["straddle_net"] = np.abs(out.sT - out.s0) / (prem * (1 + COST)) - 1
    out["iv_chg5"] = np.log(out.iv / out.groupby("ticker").iv.shift(5))
    return out


def table(df: pd.DataFrame, spike: pd.Series, cols) -> pd.DataFrame:
    rows = []
    groups = [("all days (baseline)", df[~spike])] + [(f"spike: {k}", df[spike & (df.type == k)])
                                                      for k in ["up", "flat", "down"]] + [("spike: any", df[spike])]
    for name, g in groups:
        row = {"group": name, "n": len(g)}
        for c in cols:
            x = g.dropna(subset=[c])
            if len(x) < 3:
                row[c], row[f"t_{c}"] = np.nan, np.nan
                continue
            m, t, _ = S1.week_boot(x, c)
            row[c], row[f"t_{c}"] = round(m, 3), round(t, 2)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    _, spy = S1.load()
    out = ROOT / "results"
    df = pd.concat([iv_panel(t, spy) for t in IV_TICKERS]).sort_index()
    cols = ["vrp", "straddle_gross", "straddle_net", "iv_chg5"]
    print(f"IV tickers: {IV_TICKERS}; {len(df)} ticker-days with IV, "
          f"{df.index.min().date()} .. {df.index.max().date()}")
    for thr in [3, 2]:
        sp = mark_spikes(df, thr)
        tab = table(df, sp, cols)
        print(f"\n== spike threshold {thr}x  (vrp = ln IV/realized; straddle = return on premium) ==")
        print(tab.to_string(index=False))
        tab.to_csv(out / f"iv_study_thr{thr}.csv", index=False)
        for lo, hi, nm in [(None, S1.DEV_END, "2021-24"), (S1.DEV_END, None, "2025-26")]:
            w = df[(df.index > lo) if lo is not None else (df.index <= hi)]
            t2 = table(w, mark_spikes(w, thr), ["vrp", "straddle_net"])
            print(f"   {nm}: " + "; ".join(f"{r.group} n={r.n} vrp={r.vrp} straddle_net={r.straddle_net}"
                                         for r in t2.itertuples() if r.group in ("all days (baseline)", "spike: up", "spike: down")))
        per = (df[sp].groupby("ticker").agg(n=("vrp", "size"), vrp=("vrp", "mean"), straddle_net=("straddle_net", "mean"))
               .round(3))
        print("   by ticker (spikes):", per.to_dict("index"))

    # --- Study 2b: retail-heavy vs other spikes, all tickers, price-only vol ratio
    px, spy = S1.load()
    ev = pd.concat([S1.events_for(t, d, spy) for t, d in px.items()]).sort_index()
    fin = pd.read_csv(ROOT / "data" / "finra" / "offexchange_volume.csv")
    fin["date"] = pd.to_datetime(fin.Date.astype(str))
    fin = fin.rename(columns={"Symbol": "ticker"})[["date", "ticker", "TotalVolume"]]
    vol = pd.concat([d[["volume"]].assign(ticker=t) for t, d in px.items()]).reset_index()
    m = vol.merge(fin, on=["date", "ticker"], how="left")
    m["offex"] = (m.TotalVolume / m.volume).clip(0, 1)
    m = m.sort_values(["ticker", "date"])
    m["offex_med60"] = m.groupby("ticker").offex.transform(lambda s: s.shift(1).rolling(60, min_periods=40).median())
    ev = ev.reset_index().merge(m[["date", "ticker", "offex", "offex_med60"]], on=["date", "ticker"], how="left").set_index("date")
    ev["lvr"] = np.log(ev.vol_ratio)
    base = ev[~ev.spike].lvr.mean()
    print("\n== Study 2b: spikes split by FINRA off-exchange share (all 15 tickers) ==")
    print(f"   median off-exchange share, all days: {ev.offex.median():.2f}")
    rows = []
    for lab, mask in [("retail-heavy (offex > own 60d median)", ev.offex > ev.offex_med60),
                      ("other", ev.offex <= ev.offex_med60)]:
        for k in ["up", "down", "any"]:
            g = ev[ev.spike & mask & ((ev.type == k) if k != "any" else True)].copy()
            if len(g) < 3:
                continue
            g["d"] = g.lvr - base
            mm, tt, n = S1.week_boot(g, "d")
            a5, t5, _ = S1.week_boot(g, "ar5")
            rows.append(dict(split=lab, type=k, n=n, vol_ratio_vs_base=round(float(np.exp(mm)), 2), t=round(tt, 2),
                             ar5_pct=round(100 * a5, 2), t_ar5=round(t5, 2)))
    t2b = pd.DataFrame(rows)
    print(t2b.to_string(index=False))
    t2b.to_csv(out / "iv_study_retail_split.csv", index=False)


if __name__ == "__main__":
    main()
