"""Hold-to-resolution vs early cash-out under Polymarket-style execution rules.
Dev windows of development worlds only.   python scripts/cashout_scenarios.py"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pmct.strategy import StratConfig, backtest, signal_quality, with_  # noqa: E402
from scripts.iterate import DEV_SEEDS, WORLDS  # noqa: E402

BASE = StratConfig(basis="res_mid", score="roi", threshold=-0.3, min_res=100, px_lo=0.3, px_hi=0.95,
                   calib_n0=1000.0, calib_buckets=True, kelly=0.25)


def main():
    rows = []
    for fee in [0.0, 0.01, 0.02]:
        for exit_cost in [0.015, 0.03]:
            for xh in [0.0, 6.0, 24.0, 72.0]:
                if xh == 0 and exit_cost != 0.015:
                    continue   # exit book depth is irrelevant when holding to resolution
                cfg = with_(BASE, fee_rate=fee, exit_cost=exit_cost, exit_h=xh)
                row = dict(fee_rate=fee, exit_cost=exit_cost, exit=("hold" if xh == 0 else f"{xh:.0f}h"))
                for s in DEV_SEEDS:
                    ctx, (dev0, hold0, _) = WORLDS[f"s{s}"]
                    sig = ctx.signals(cfg)
                    sq = signal_quality(sig, dev0, hold0)
                    m = backtest(cfg, sig, ctx.book, dev0, hold0).metrics
                    row.update({f"s{s}_edge_c": round(100 * sq["edge"], 3), f"s{s}_t": round(sq["t"], 2),
                                f"s{s}_growth": round(m["log_growth"], 4), f"s{s}_bets": m["n_bets"]})
                rows.append(row)
                print(row, flush=True)
    df = pd.DataFrame(rows)
    out = Path(__file__).resolve().parents[1] / "results" / "cashout_scenarios.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
