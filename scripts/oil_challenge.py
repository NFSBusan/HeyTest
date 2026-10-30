"""Phase 1 strategy challenge for Brent-WTI mean reversion.

Part A (needs data): spread diagnostics on daily Brent and WTI prices. Put FRED downloads
at data/oil/DCOILBRENTEU.csv and data/oil/DCOILWTICO.csv (or pass --brent/--wti). Skipped
with a message if the files are missing. Never substitutes made-up prices.

Part B (no data): perfect-signal option mapping. Given a trader who knows the spread's true
mean and speed, which defined-risk option structures turn that knowledge into expected
profit, and how many trades would it take to tell that profit from luck?

Run: python scripts/oil_challenge.py   (from the repo root)
"""
import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oilmr import diagnostics, options  # noqa: E402
from oilmr.data import load_pair  # noqa: E402

OUT = Path("results/oil")


def part_a(brent, wti):
    if not (Path(brent).exists() and Path(wti).exists()):
        print(f"[A] skipped: {brent} / {wti} not found. Download FRED series DCOILBRENTEU and "
              "DCOILWTICO as CSV and rerun.")
        return None
    df = load_pair(brent, wti)
    rep = {**diagnostics.full_sample_report(df), **diagnostics.spread_share_of_outright(df)}
    diagnostics.by_regime(df).to_csv(OUT / "spread_by_regime.csv", index=False)
    diagnostics.rolling_report(df).to_csv(OUT / "spread_rolling.csv")
    (OUT / "spread_full_sample.json").write_text(json.dumps(rep, indent=2, default=float))
    print("[A]", json.dumps(rep, indent=2, default=float))
    return rep


def part_b(base):
    rows = [{**r, "phi": base.phi} for r in options.table(base)]
    pd.DataFrame(rows).to_csv(OUT / "option_mapping_base.csv", index=False)
    fut = options.futures_spread_benchmark(base)

    sweep_rows = []
    for ov, w in options.sweep(base, phi=[0.0, 0.25, 0.5, 1.0], half_life_days=[10, 30, 90, 250],
                               s0_sigmas=[0.0, 1.0, 2.0, 3.0], priced_in=[0.0, 0.5, 0.9, 1.0]):
        for name in ["wti_call_atm", "wti_call_plus_brent_put", "spread_put_atm"]:
            e = options.evaluate(w, options.STRUCTURES[name])
            sweep_rows.append({**ov, "structure": name, "exp_pnl_pct_of_debit": e["exp_pnl_pct_of_debit"]})
    sw = pd.DataFrame(sweep_rows)
    sw.to_csv(OUT / "option_mapping_sweep.csv", index=False)

    summary = {
        "world": {**asdict(base), "s0": base.s0, "stationary_sd": base.stat_sd},
        "futures_spread_benchmark_not_tradeable": fut,
        "share_of_sweep_cells_with_positive_ev": sw.groupby("structure")["exp_pnl_pct_of_debit"]
                                                   .apply(lambda x: float((x > 0).mean())).to_dict(),
    }
    (OUT / "option_mapping_summary.json").write_text(json.dumps(summary, indent=2, default=float))

    cols = ["structure", "premium", "exp_pnl", "exp_pnl_pct_of_debit", "max_loss", "pnl_sd",
            "p_profit", "trades_for_t2"]
    print("[B] base world:", {k: round(v, 3) if isinstance(v, float) else v for k, v in asdict(base).items()})
    print(pd.DataFrame(rows)[cols].round(3).to_string(index=False))
    print("[B] futures spread reference:", {k: round(float(v), 3) for k, v in fut.items()})
    piv = sw[(sw.s0_sigmas == 2.0) & (sw.priced_in == 0.0)].pivot_table(index=["structure", "phi"], columns="half_life_days",
                                              values="exp_pnl_pct_of_debit")
    print("[B] EV % of debit at 2-sd entry, by phi (rows) and half-life days (cols):")
    print((100 * piv[piv.index.get_level_values(0) != "spread_put_atm"]).round(1).to_string())
    pin = sw[(sw.s0_sigmas == 2.0) & (sw.phi == 0.5) & (sw.half_life_days == 30)].pivot_table(
        index="structure", columns="priced_in", values="exp_pnl_pct_of_debit")
    print("[B] EV % of debit vs share of convergence already priced in (2-sd entry, phi=0.5, HL=30d):")
    print((100 * pin).round(1).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brent", default="data/oil/DCOILBRENTEU.csv")
    ap.add_argument("--wti", default="data/oil/DCOILWTICO.csv")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    part_a(args.brent, args.wti)
    part_b(options.World())


if __name__ == "__main__":
    main()
