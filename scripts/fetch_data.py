"""Download real Polymarket history into data/*.parquet.

    python scripts/fetch_data.py --max-markets 3000

Needs network access to gamma-api.polymarket.com and data-api.polymarket.com.
Then:  python scripts/run_research.py --data real
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pmct import fetch  # noqa: E402
from pmct.schema import validate  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-markets", type=int, default=3000)
    ap.add_argument("--min-volume", type=float, default=5000.0)
    args = ap.parse_args()
    DATA.mkdir(exist_ok=True)
    closed = fetch.fetch_markets(True, min_volume=args.min_volume)
    open_ = fetch.fetch_markets(False, min_volume=args.min_volume)   # avoid survivorship bias
    mk = pd.concat([closed, open_]).drop_duplicates("condition_id")
    mk = mk.sort_values("created_ts").tail(args.max_markets)
    print(f"{len(mk)} markets ({mk.outcome.notna().sum()} resolved)")
    raw = {}
    for i, cid in enumerate(mk.condition_id):
        raw[cid] = fetch.fetch_trades(cid)
        if i % 100 == 0:
            print(i, cid, len(raw[cid]))
    markets, trades, wallets = fetch.to_canonical(mk, raw)
    validate(markets, trades)
    markets.to_parquet(DATA / "markets.parquet")
    trades.to_parquet(DATA / "trades.parquet")
    wallets.to_parquet(DATA / "wallets.parquet")
    print(f"saved {len(markets)} markets, {len(trades):,} trades, {len(wallets):,} wallets")


if __name__ == "__main__":
    main()
