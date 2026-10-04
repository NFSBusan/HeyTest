"""Staged research run: search on dev only -> freeze -> one holdout (front test) pass.

    python scripts/run_research.py --data sim            # simulated world (default seed 7)
    python scripts/run_research.py --data sim --null     # null world: nobody has skill
    python scripts/run_research.py --data real           # needs data/markets.parquet, data/trades.parquet

Writes results/<tag>_dev_trials.csv, results/<tag>_summary.json.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pmct.schema import DAY  # noqa: E402
from pmct.stats import deflated_sharpe  # noqa: E402
from pmct.strategy import Context, StratConfig, backtest, signal_quality, with_  # noqa: E402

KELLYS = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5]
THRESH = {"roi": [-0.02, -0.05, -0.10], "tstat": [-1.0, -1.5, -2.5], "edge": [-0.005, -0.01, -0.02],
          "pnl": [-50.0, -200.0, -1000.0]}
OUT = Path(__file__).resolve().parents[1] / "results"


def load(args):
    if args.data == "sim":
        from pmct.synthetic import SimConfig, simulate
        m, t, _ = simulate(SimConfig(seed=args.seed, null_world=args.null))
        return m, t
    d = Path(__file__).resolve().parents[1] / "data"
    return pd.read_parquet(d / "markets.parquet"), pd.read_parquet(d / "trades.parquet")


def windows(trades):
    t0, t1 = int(trades.ts.min()), int(trades.ts.max())
    span = t1 - t0
    dev0 = t0 + int(0.22 * span)
    hold0 = t0 + int(0.72 * span)
    return dev0, hold0, t1


def evaluate(ctx, cfg, dev0, hold0):
    sig = ctx.signals(cfg)
    r = backtest(cfg, sig, ctx.book, dev0, hold0)
    mid = (dev0 + hold0) // 2
    h1 = backtest(cfg, sig, ctx.book, dev0, mid).metrics["log_growth"]
    h2 = backtest(cfg, sig, ctx.book, mid, hold0).metrics["log_growth"]
    sq = signal_quality(sig, dev0, hold0)
    lr = np.diff(np.log(np.maximum(np.concatenate([[10_000.0], r.equity.to_numpy()]), 1e-9)))
    d = {**asdict(cfg), **{f"dev_{k}": v for k, v in r.metrics.items()}, "dev_h1": h1, "dev_h2": h2,
         **{f"sig_{k}": v for k, v in sq.items()},
         "daily_sr": float(lr.mean() / lr.std(ddof=1)) if lr.std() > 0 else 0.0}
    return d


def objective(row):
    """Dev selection score: log growth, but only if the config bet enough and made money in
    both halves of dev (stability). Never looks at holdout."""
    if row["dev_n_bets"] < 30 or row["dev_h1"] <= 0 or row["dev_h2"] <= 0:
        return -np.inf
    return row["dev_log_growth"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="sim", choices=["sim", "real"])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--null", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    tag = args.tag or (f"{args.data}{args.seed}" + ("_null" if args.null else ""))
    OUT.mkdir(exist_ok=True)
    t_start = time.time()

    markets, trades = load(args)
    dev0, hold0, end = windows(trades)
    ctx = Context(markets, trades)
    print(f"[{tag}] trades={len(trades):,} dev=[{dev0//DAY},{hold0//DAY}) hold=[{hold0//DAY},{end//DAY}] days")

    rows = []
    # ---- Stage A: what is a "bad" trader? (signal definition), Kelly 0.25, global calibration
    bases = ["res", "res_mid", "mo6h", "mo1d"]
    scores = ["roi", "tstat", "edge", "pnl"]
    grid = [(b, s, th, mr, band) for b in bases for s in scores for th in THRESH[s]
            for mr in ([30, 100] if not args.quick else [30]) for band in [(0.05, 0.95), (0.2, 0.8)]]
    for b, s, th, mr, (lo, hi) in grid:
        cfg = StratConfig(basis=b, score=s, threshold=th, min_res=mr, px_lo=lo, px_hi=hi, calib_buckets=False)
        rows.append({"stage": "A", **evaluate(ctx, cfg, dev0, hold0)})
    df = pd.DataFrame(rows)
    df["obj"] = df.apply(objective, axis=1)
    print(f"stage A: {len(df)} configs, {np.isfinite(df.obj).sum()} pass stability, {time.time()-t_start:.0f}s")

    # ---- Stage B: sizing / calibration for the top signal definitions
    top = df[np.isfinite(df.obj)].sort_values("obj", ascending=False).head(8)
    if top.empty:   # nothing stable: still explore sizing on best raw growth so we can report it
        top = df.sort_values("dev_log_growth", ascending=False).head(4)
    sig_fields = ["basis", "score", "threshold", "min_res", "px_lo", "px_hi"]
    for _, row in top.iterrows():
        base = StratConfig(**{f: row[f] for f in sig_fields})
        for k, n0, me, mb, cb in itertools.product(KELLYS, [100.0, 300.0, 1000.0], [0.005, 0.01, 0.02],
                                                   [0.05, 0.10], [False, True]):
            cfg = with_(base, kelly=k, calib_n0=n0, min_edge=me, max_bet_frac=mb, calib_buckets=cb)
            rows.append({"stage": "B", **evaluate(ctx, cfg, dev0, hold0)})
    df = pd.DataFrame(rows)
    df["obj"] = df.apply(objective, axis=1)
    df.to_csv(OUT / f"{tag}_dev_trials.csv", index=False)
    n_trials = len(df)
    print(f"stage B done: {n_trials} total trials, {time.time()-t_start:.0f}s")

    # ---- Freeze: best by dev objective. Kelly fraction chosen on dev too, but we report all.
    ok = df[np.isfinite(df.obj)]
    best = (ok if len(ok) else df).sort_values("obj" if len(ok) else "dev_log_growth", ascending=False).iloc[0]
    frozen = StratConfig(**{f: (type(getattr(StratConfig(), f))(best[f])) for f in asdict(StratConfig()).keys()})
    sr_var = float(df["daily_sr"].var())
    dsr = deflated_sharpe(best["daily_sr"], int(best["dev_n_days"]), best["dev_skew"], best["dev_kurt"],
                          n_trials, sr_var)
    print("FROZEN:", frozen.key())
    print(f"dev: growth={best.dev_log_growth:.3f} sharpe={best.dev_sharpe:.2f} maxdd={best.dev_max_dd:.2f} "
          f"bets={best.dev_n_bets} DSR={dsr:.3f}")

    # ---- Front test: holdout, touched once, every Kelly fraction of the frozen config
    hold = {}
    for k in KELLYS:
        cfg = with_(frozen, kelly=k)
        r = backtest(cfg, ctx.signals(cfg), ctx.book, hold0, end)
        hold[str(k)] = r.metrics
        print(f"  holdout kelly={k:<5} growth={r.metrics['log_growth']:+.3f} sharpe={r.metrics['sharpe']:+.2f} "
              f"maxdd={r.metrics['max_dd']:.2f} bets={r.metrics['n_bets']} final={r.metrics['final']:,.0f}")
        r.positions.to_csv(OUT / f"{tag}_holdout_positions_k{k}.csv", index=False) if k == frozen.kelly else None
    hq = signal_quality(ctx.signals(frozen), hold0, end)
    summary = dict(tag=tag, n_trials=n_trials, frozen=asdict(frozen),
                   dev={k: best[f"dev_{k}"] for k in ["log_growth", "sharpe", "max_dd", "n_bets", "final"]},
                   dev_halves=[best.dev_h1, best.dev_h2], deflated_sharpe=dsr, dev_signal=
                   {k: best[f"sig_{k}"] for k in ["n_sig", "n_mkt", "edge", "t"]},
                   holdout=hold, holdout_signal=hq,
                   stage_A_pass=int(np.isfinite(df[df.stage == "A"].obj).sum()), runtime_s=time.time() - t_start)
    (OUT / f"{tag}_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"holdout signal edge {hq['edge']:+.4f} t={hq['t']:.2f} n_mkt={hq['n_mkt']}  ({time.time()-t_start:.0f}s)")


if __name__ == "__main__":
    main()
