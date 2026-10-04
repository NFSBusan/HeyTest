"""Final test of the FROZEN config. Run once, after all iteration.

- Holdout windows of the development worlds (seeds 7, 11): never used for selection.
- Fresh worlds (seeds 101, 102, 103): never seen at all, so dev and holdout windows both count as
  out-of-sample.
- Null world (seed 101, no skill): should not make money.
Every Kelly fraction is reported.     python scripts/final_test.py
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pmct.stats import deflated_sharpe  # noqa: E402
from pmct.strategy import Context, StratConfig, backtest, signal_quality, with_  # noqa: E402
from pmct.synthetic import SimConfig, simulate  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results"
KELLYS = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5]
N_TRIALS = 1300   # every config evaluated during research (quick grid 960 + iteration rounds), for deflation

FROZEN = StratConfig(basis="res_mid", score="roi", threshold=-0.3, min_res=100, min_mkts=5, px_lo=0.3, px_hi=0.95,
                     fee_schedule="pm2026", kelly=0.25, max_bet_frac=0.05, max_mkt_frac=0.10, calib_n0=100.0,
                     calib_min=200, calib_buckets=False, min_edge=0.0, first_only=True,
                     maker=True, maker_ttl_s=3600, maker_offset=0.01)


def windows(trades):
    t0, t1 = int(trades.ts.min()), int(trades.ts.max())
    span = t1 - t0
    return t0 + int(0.22 * span), t0 + int(0.72 * span), t1


def main():
    rows = []
    tests = [(7, False, "hold"), (11, False, "hold"), (101, False, "both"), (102, False, "both"),
             (103, False, "both"), (101, True, "both")]
    for seed, null, which in tests:
        m, t, _ = simulate(SimConfig(seed=seed, null_world=null))
        ctx = Context(m, t)
        dev0, hold0, end = windows(t)
        wins = [("holdout", hold0, end)] + ([("dev-window", dev0, hold0)] if which == "both" else [])
        for wname, a, b in wins:
            sq = signal_quality(ctx.signals(FROZEN), a, b)
            for k in KELLYS:
                cfg = with_(FROZEN, kelly=k)
                r = backtest(cfg, ctx.signals(cfg), ctx.book, a, b)
                mt = r.metrics
                lr = np.diff(np.log(np.concatenate([[10_000.0], r.equity.to_numpy()])))
                rows.append(dict(world=f"seed{seed}" + ("_NULL" if null else ""), window=wname, kelly=k,
                                 growth=mt["log_growth"], cagr=mt["cagr"], sharpe=mt["sharpe"], max_dd=mt["max_dd"],
                                 bets=mt["n_bets"], hit=mt["hit"], final=mt["final"],
                                 sig_edge_c=100 * sq["edge"], sig_t=sq["t"],
                                 daily_sr=float(lr.mean() / lr.std(ddof=1)) if lr.std() > 0 else 0.0,
                                 n_days=mt["n_days"], skew=mt["skew"], kurt=mt["kurt"]))
                print({k2: (round(v, 3) if isinstance(v, float) else v) for k2, v in rows[-1].items()
                       if k2 not in ("daily_sr", "n_days", "skew", "kurt")}, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "final_test.csv", index=False)
    oos = df[~df.world.str.contains("NULL")]
    summ = (oos.groupby("kelly").agg(mean_growth=("growth", "mean"), median_growth=("growth", "median"),
                                     worst_growth=("growth", "min"), pct_positive=("growth", lambda x: (x > 0).mean()),
                                     mean_sharpe=("sharpe", "mean"), worst_dd=("max_dd", "max"))
            .round(4))
    null = df[df.world.str.contains("NULL")].groupby("kelly").growth.mean().round(4).rename("null_growth")
    summ = summ.join(null)
    print(summ.to_string())
    # deflated Sharpe on the pooled out-of-sample daily Sharpe at the frozen Kelly
    f = oos[oos.kelly == FROZEN.kelly]
    sr_var = float(pd.read_csv(OUT / "iterations.csv").filter(like="_sr").stack().div(np.sqrt(365)).var())
    dsr = [deflated_sharpe(r.daily_sr, int(r.n_days), r.skew, r.kurt, N_TRIALS, sr_var) for r in f.itertuples()]
    out = dict(frozen=asdict(FROZEN), by_kelly=summ.reset_index().to_dict("records"),
               dsr_per_test_at_frozen_kelly=dict(zip(f.world + "/" + f.window, np.round(dsr, 3))))
    (OUT / "final_test_summary.json").write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps(out["dsr_per_test_at_frozen_kelly"], indent=1))


if __name__ == "__main__":
    main()
