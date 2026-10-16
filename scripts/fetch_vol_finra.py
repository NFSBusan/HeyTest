"""Download implied-vol history (DoltHub) and off-exchange volume (FINRA) for our tickers.

Run this on YOUR computer (the cloud session can't reach these hosts):
    pip install requests pandas
    python scripts/fetch_vol_finra.py            # all tickers in data/prices/
    git add data/vol data/finra && git commit -m "Add IV + FINRA data" && git push

Sources
    DoltHub  post-no-preference/options, table volatility_history (iv_current = 30d implied vol)
             https://www.dolthub.com/repositories/post-no-preference/options
    FINRA    Reg SHO daily short-sale volume, consolidated file CNMSshvol<YYYYMMDD>.txt
             https://www.finra.org/finra-data/browse-catalog/reg-sho-daily
             TotalVolume there = off-exchange (TRF) volume, which is mostly wholesaler/retail flow.
Both are UNVERIFIED from the cloud session: check the first rows by hand.
"""
from __future__ import annotations

import sys
import time
from datetime import date, timedelta
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DOLT = "https://www.dolthub.com/api/v1alpha1/post-no-preference/options/master"
FINRA = "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{:%Y%m%d}.txt"
S = requests.Session()
S.headers["User-Agent"] = "rflow-research/0.1"


def tickers():
    return sorted(f.stem for f in (ROOT / "data" / "prices").glob("*.csv"))


def dolt_query(sql: str) -> list[dict]:
    for i in range(5):
        r = S.get(DOLT, params={"q": sql}, timeout=60)
        if r.status_code == 200:
            j = r.json()
            if j.get("query_execution_status") not in (None, "Success", "RowLimit"):
                raise RuntimeError(j.get("query_execution_message"))
            return j.get("rows", [])
        time.sleep(2 ** i)
    r.raise_for_status()


def fetch_vol(syms):
    out = ROOT / "data" / "vol"
    out.mkdir(parents=True, exist_ok=True)
    cols = dolt_query("SHOW COLUMNS FROM volatility_history")
    print("volatility_history columns:", [c.get("Field") for c in cols])
    for s in syms:
        rows = []
        for y in range(2019, date.today().year + 1):   # one year per query, stays under the API row cap
            rows += dolt_query(f"SELECT * FROM volatility_history WHERE act_symbol = '{s}' "
                               f"AND date BETWEEN '{y}-01-01' AND '{y}-12-31' ORDER BY date")
        df = pd.DataFrame(rows)
        if df.empty:
            print(f"{s}: no rows")
            continue
        df.to_csv(out / f"{s}.csv", index=False)
        print(f"{s}: {len(df)} rows {df['date'].min()} .. {df['date'].max()}")


def fetch_finra(syms, start=date(2021, 10, 1)):
    out = ROOT / "data" / "finra"
    out.mkdir(parents=True, exist_ok=True)
    keep, d, parts = set(syms), start, []
    while d <= date.today():
        if d.weekday() < 5:
            r = S.get(FINRA.format(d), timeout=30)
            if r.status_code == 200 and r.text.strip():
                t = pd.read_csv(StringIO(r.text), sep="|")
                t = t[t["Symbol"].isin(keep)]
                parts.append(t)
            time.sleep(0.2)
        d += timedelta(days=1)
        if d.day == 1:
            print("finra", d)
    df = pd.concat(parts, ignore_index=True)
    df.to_csv(out / "offexchange_volume.csv", index=False)
    print(f"FINRA: {len(df)} rows, {df['Symbol'].nunique()} tickers")


if __name__ == "__main__":
    syms = sys.argv[1:] or tickers()
    fetch_vol(syms)
    fetch_finra(syms)
