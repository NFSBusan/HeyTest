"""Structured iterations, judged ONLY on the dev windows of development worlds.

Development worlds: simulator seeds 7 and 11 (dev window only), plus the null world (seed 7) as
a control. The holdout windows and the fresh worlds (seeds 101-103) are not touched here; see
scripts/final_test.py.

    python scripts/iterate.py            -> results/iterations.csv
"""
from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pmct.strategy import Context, StratConfig, backtest, signal_quality, with_  # noqa: E402
from pmct.synthetic import SimConfig, simulate  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results"
DEV_SEEDS = [7, 11]


def windows(trades):
    t0, t1 = int(trades.ts.min()), int(trades.ts.max())
    span = t1 - t0
    return t0 + int(0.22 * span), t0 + int(0.72 * span), t1


def build(seed, null=False):
    m, t, w = simulate(SimConfig(seed=seed, null_world=null))
    ctx = Context(m, t)
    ctx.set_oracle(w)
    return ctx, windows(t)


WORLDS = {f"s{s}": build(s) for s in DEV_SEEDS}
WORLDS["null7"] = build(7, null=True)
ROWS: list[dict] = []


def run(it: str, cfg: StratConfig, note: str = "") -> dict:
    row = {"iter": it, "note": note, **asdict(cfg)}
    for name, (ctx, (dev0, hold0, _)) in WORLDS.items():
        if cfg.basis == "oracle" and name.startswith("null"):
            continue
        sig = ctx.signals(cfg)
        r = backtest(cfg, sig, ctx.book, dev0, hold0)
        sq = signal_quality(sig, dev0, hold0)
        row.update({f"{name}_g": r.metrics["log_growth"], f"{name}_sr": r.metrics["sharpe"],
                    f"{name}_dd": r.metrics["max_dd"], f"{name}_n": r.metrics["n_bets"],
                    f"{name}_edge": sq["edge"], f"{name}_t": sq["t"]})
    gs = [row[f"s{s}_g"] for s in DEV_SEEDS]
    row["obj"] = float(np.mean(gs)) if min(gs) > 0 and min(row[f"s{s}_n"] for s in DEV_SEEDS) >= 30 else -np.inf
    ROWS.append(row)
    print(f"{it:4s} obj={row['obj']:+.3f} " + " ".join(
        f"{n}: g={row.get(n+'_g', np.nan):+.3f} sr={row.get(n+'_sr', np.nan):+.2f} n={row.get(n+'_n', 0):>4} "
        f"t={row.get(n+'_t', np.nan):+.2f}" for n in WORLDS) + f"  | {note}", flush=True)
    return row


def best_of(it_prefix: str) -> StratConfig:
    df = pd.DataFrame([r for r in ROWS if r["iter"].startswith(it_prefix) and r["basis"] != "oracle"])
    # rank by dev objective; break ties (e.g. all -inf) by mean clustered signal t-stat
    df["t_mean"] = df[[f"s{s}_t" for s in DEV_SEEDS]].mean(axis=1).fillna(-99)
    df = df.sort_values(["obj", "t_mean"], ascending=False)
    r = df.iloc[0]
    return StratConfig(**{k: type(v)(r[k]) for k, v in asdict(StratConfig()).items()})


def main():
    K = dict(kelly=0.25)
    # I0: ceiling - counter the truly worst wallets (diagnostic, uses hidden truth)
    for th in [-1.0, -0.6]:
        run("I0", StratConfig(basis="oracle", score="tstat", threshold=th, calib_buckets=False, **K), f"oracle skill<={th}")
    # I1: baseline
    run("I1", StratConfig(basis="res", score="tstat", threshold=-2.0, calib_buckets=True, **K), "baseline raw-PnL t-stat")
    # I2: grid winner from the quick search
    run("I2", StratConfig(basis="mo1d", score="roi", threshold=-0.1, calib_buckets=False, **K), "quick-grid winner")
    # I3: ROI on resolved PnL (best discrimination), threshold x min history
    for b in ["res", "res_mid"]:
        for th in [-0.05, -0.1, -0.2, -0.3]:
            for mr in [30, 100]:
                for me in [0.0, 0.005]:
                    run("I3", StratConfig(basis=b, score="roi", threshold=th, min_res=mr, min_edge=me,
                                          calib_buckets=False, **K), f"{b} roi<={th} n>={mr} min_edge={me}")
    c = best_of("I3")
    # I4: one bet per wallet per market (avoid stacking correlated bets)
    run("I4", with_(c, first_only=True), "first trade per wallet-market only")
    c = best_of("I")
    # I5: bad-money consensus in the market
    for fh in [6.0, 24.0]:
        for mf in [0.0, 100.0, 500.0]:
            run("I5", with_(c, flow_h=fh, min_flow=mf), f"flow {fh}h >= ${mf}")
    # I6: early exit (frees capital, but pays the spread twice)
    for xh in [6.0, 24.0, 72.0]:
        run("I6", with_(c, exit_h=xh), f"exit after {xh}h")
    # I7: price band (longshots are noisy)
    for lo, hi in [(0.2, 0.8), (0.1, 0.9), (0.3, 0.95)]:
        run("I7", with_(c, px_lo=lo, px_hi=hi), f"band {lo}-{hi}")
    c = best_of("I")
    # I8: calibration
    for n0 in [100.0, 300.0, 1000.0]:
        for cb in [False, True]:
            run("I8", with_(c, calib_n0=n0, calib_buckets=cb), f"calib n0={n0} buckets={cb}")
    c = best_of("I8")
    # I8b: size by how bad the wallet is (walk-forward ridge on badness)
    for n0 in [100.0, 300.0, 1000.0]:
        run("I8b", with_(c, calib_n0=n0, calib_linear=True), f"linear calib n0={n0}")
    # I8c: with a looser threshold, the linear model can rank within a wider set of wallets
    for th in [-0.2, -0.1]:
        for n0 in [300.0, 1000.0]:
            run("I8c", with_(c, threshold=th, calib_n0=n0, calib_linear=True), f"linear calib thr={th} n0={n0}")
    c = best_of("I8")
    # I9: Kelly fraction and caps
    for k in [0.1, 0.25, 0.5, 0.75, 1.0, 1.5]:
        for mb in [0.05, 0.10]:
            run("I9", with_(c, kelly=k, max_bet_frac=mb), f"kelly={k} max_bet={mb}")
    # I10: cost sensitivity of the I9 winner (not a selection step)
    c = best_of("I9")
    for ec, lat in [(0.015, 60), (0.02, 60), (0.01, 600), (0.01, 3600)]:
        run("I10", with_(c, exec_cost=ec, latency_s=lat), f"stress exec_cost={ec} latency={lat}s")
    pd.DataFrame(ROWS).to_csv(OUT / "iterations.csv", index=False)
    print("FINAL DEV PICK:", c.key())


if __name__ == "__main__":
    main()
