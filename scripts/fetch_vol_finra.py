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


def dolt_query(sql: str, tries=4) -> list[dict]:
    """Run one SQL query on DoltHub. Timeouts are retried; the caller splits the window if they persist."""
    for i in range(tries):
        try:
            r = S.get(DOLT, params={"q": sql}, timeout=90)
        except requests.RequestException:
            time.sleep(2 ** i)
            continue
        if r.status_code != 200:
            time.sleep(2 ** i)
            continue
        j = r.json()
        if j.get("query_execution_status") in (None, "Success", "RowLimit"):
            if j.get("query_execution_status") == "RowLimit":
                raise OverflowError("row limit hit")
            return j.get("rows", [])
        msg = j.get("query_execution_message", "")
        if "deadline" not in msg:
            raise RuntimeError(msg)
        time.sleep(2 ** i)
    raise TimeoutError(sql)


def _window(syms, a: date, b: date) -> list[dict]:
    """All symbols for dates [a, b]. Filtering by date first uses the table's index; split on timeout."""
    names = ",".join(f"'{x}'" for x in syms)
    sql = (f"SELECT * FROM volatility_history WHERE date BETWEEN '{a}' AND '{b}' "
           f"AND act_symbol IN ({names}) ORDER BY date")
    try:
        return dolt_query(sql)
    except (TimeoutError, OverflowError):
        if (b - a).days < 2:
            print(f"  giving up on {a}..{b}")
            return []
        mid = a + (b - a) / 2
        return _window(syms, a, mid) + _window(syms, mid + timedelta(days=1), b)


def fetch_vol(syms, start=date(2019, 1, 1)):
    out = ROOT / "data" / "vol"
    out.mkdir(parents=True, exist_ok=True)
    cols = dolt_query("SHOW COLUMNS FROM volatility_history")
    print("volatility_history columns:", [c.get("Field") for c in cols])
    part = out / "_all_partial.csv"
    done = pd.read_csv(part) if part.exists() else pd.DataFrame()
    a = (pd.to_datetime(done["date"]).max().date() + timedelta(days=1)) if len(done) else start
    rows = []
    while a <= date.today():
        b = min((a.replace(day=1) + timedelta(days=32)).replace(day=1) - timedelta(days=1), date.today())
        got = _window(syms, a, b)
        rows += got
        print(f"vol {a:%Y-%m}: {len(got)} rows")
        if got:   # save progress so a rerun resumes here
            pd.concat([done, pd.DataFrame(rows)]).to_csv(part, index=False)
        a = b + timedelta(days=1)
    df = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
    if df.empty:
        print("no vol rows at all")
        return
    for s, g in df.groupby("act_symbol"):
        g.drop_duplicates("date").sort_values("date").to_csv(out / f"{s}.csv", index=False)
        print(f"{s}: {len(g)} rows {g['date'].min()} .. {g['date'].max()}")
    missing = sorted(set(syms) - set(df["act_symbol"]))
    if missing:
        print("not in dataset:", missing)


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
    try:
        fetch_vol(syms)
    except Exception as e:   # keep going: the FINRA part doesn't depend on DoltHub
        print("DoltHub step failed:", repr(e))
    fetch_finra(syms)
