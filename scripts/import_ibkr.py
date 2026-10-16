"""Convert IBKR get_price_history JSON dumps into data/prices/<TICKER>.csv.

Files carry no symbol, so each is matched by its last close against a {ticker: last_close} map.
    python scripts/import_ibkr.py <dump_dir> TICKER=last_close ...
"""
import json
import sys
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "data" / "prices"


def main():
    src, pairs = Path(sys.argv[1]), sys.argv[2:]
    want = {k: float(v) for k, v in (p.split("=") for p in pairs)}
    OUT.mkdir(parents=True, exist_ok=True)
    for f in sorted(src.glob("mcp-Interactive_Brokers_IBKR-get_price_history-*.txt")):
        d = json.loads(f.read_text())
        if len(d.get("time", [])) < 30:
            continue
        last = d["close"][-1]
        hits = [t for t, c in want.items() if abs(c - last) < 1e-6]
        if len(hits) != 1:
            print(f"skip {f.name}: last close {last} matches {hits}")
            continue
        df = pd.DataFrame({k: d[k] for k in ["time", "open", "high", "low", "close", "volume"]})
        df["date"] = pd.to_datetime(df.pop("time")).dt.date
        df = df[["date", "open", "high", "low", "close", "volume"]]
        df.to_csv(OUT / f"{hits[0]}.csv", index=False)
        print(f"{hits[0]}: {len(df)} days {df.date.iloc[0]} .. {df.date.iloc[-1]}")
        want.pop(hits[0])
    if want:
        print("not found:", list(want))


if __name__ == "__main__":
    main()
