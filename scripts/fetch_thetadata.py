"""Download end-of-day option chains from a running Theta Terminal (ThetaData) on YOUR computer.

1. Start the Theta Terminal and log in (see ThetaData's docs). It serves data on localhost:
     v3 terminal -> http://127.0.0.1:25503     v2 terminal -> http://127.0.0.1:25510
2. python3 scripts/fetch_thetadata.py               (default tickers below, last 365 days)
   python3 scripts/fetch_thetadata.py ASTS RKLB     (just these)
3. git add data/options && git commit -m "Add ThetaData option EOD" && push

What it saves: data/options/<TICKER>.csv.gz, one row per contract per day:
    date, expiration, strike, right (C/P), bid, ask, close, volume
Only REGULAR monthly expirations (3rd Friday) are pulled, each from 50 days before expiry
to expiry. That's what 30-day strategies need, and it keeps the free tier's rate limit manageable.
The script resumes: tickers/expirations already saved are skipped.
If anything fails, it writes the raw reply to data/options/_debug.txt. Paste that (never your key).
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
OUT = ROOT / "data" / "options"
DEFAULT = ["ASTS", "RKLB", "IREN", "ONDS", "APLD", "WULF", "IBRX", "SOFI", "NBIS", "CRWV", "GME", "NVDA"]
DAYS_BACK = 365
WINDOW = 50
PAUSE = 3.2          # seconds between requests (free tier ~20 requests/minute)
S = requests.Session()


def debug(msg: str, text: str = ""):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "_debug.txt", "a") as f:
        f.write(f"--- {msg}\n{text[:3000]}\n")
    print(f"  (wrote details to {OUT / '_debug.txt'})")


def get(url, params):
    for i in range(5):
        try:
            r = S.get(url, params=params, timeout=120)
        except requests.ConnectionError:
            sys.exit(f"Can't reach {url}. Is the Theta Terminal running and logged in?")
        time.sleep(PAUSE)
        if r.status_code == 200:
            return r
        if r.status_code in (429, 570, 571, 474) or r.status_code >= 500:   # rate limit / busy / disconnected
            time.sleep(10 * (i + 1))
            continue
        return r
    return r


def detect():
    for base, ver in [("http://127.0.0.1:25503", 3), ("http://127.0.0.1:25510", 2)]:
        try:
            S.get(base, timeout=3)
            return base, ver
        except requests.ConnectionError:
            continue
    sys.exit("No Theta Terminal found on ports 25503 (v3) or 25510 (v2). Start it and log in first.")


def third_friday(d: date) -> bool:
    return d.weekday() == 4 and 15 <= d.day <= 21


def expirations(base, ver, sym) -> list[date]:
    if ver == 3:
        r = get(f"{base}/v3/option/list/expirations", {"symbol": sym, "format": "csv"})
        if r.status_code != 200:
            debug(f"{sym} expirations HTTP {r.status_code}", r.text)
            return []
        df = pd.read_csv(StringIO(r.text))
        col = [c for c in df.columns if "exp" in c.lower()][0]
        vals = df[col].astype(str).str.replace("-", "")
    else:
        r = get(f"{base}/v2/list/expirations", {"root": sym})
        if r.status_code != 200:
            debug(f"{sym} expirations HTTP {r.status_code}", r.text)
            return []
        vals = [str(x) for x in r.json()["response"]]
    out = []
    for v in vals:
        try:
            out.append(date(int(v[:4]), int(v[4:6]), int(v[6:8])))
        except ValueError:
            pass
    return sorted(out)


def eod(base, ver, sym, exp: date, a: date, b: date) -> pd.DataFrame:
    if ver == 3:
        r = get(f"{base}/v3/option/history/eod", {"symbol": sym, "expiration": f"{exp:%Y%m%d}",
                                                   "start_date": f"{a:%Y%m%d}", "end_date": f"{b:%Y%m%d}",
                                                   "format": "csv"})
        if r.status_code != 200 or not r.text.strip():
            debug(f"{sym} {exp} eod HTTP {r.status_code}", r.text)
            return pd.DataFrame()
        df = pd.read_csv(StringIO(r.text))
        low = {c.lower(): c for c in df.columns}
        pick = lambda *names: next((low[n] for n in names if n in low), None)  # noqa: E731
        cols = {"date": pick("date", "created", "last_trade"), "strike": pick("strike"), "right": pick("right"),
                "bid": pick("bid", "close_bid"), "ask": pick("ask", "close_ask"), "close": pick("close"),
                "volume": pick("volume")}
        if None in (cols["strike"], cols["right"], cols["bid"], cols["ask"], cols["date"]):
            debug(f"{sym} {exp} unexpected v3 columns {list(df.columns)}", r.text)
            return pd.DataFrame()
        out = pd.DataFrame({k: df[v] for k, v in cols.items() if v is not None})
        out["date"] = pd.to_datetime(out["date"].astype(str).str[:10].str.replace("-", ""), format="%Y%m%d")
        out["right"] = out["right"].astype(str).str.upper().str[0]
    else:
        r = get(f"{base}/v2/bulk_hist/option/eod", {"root": sym, "exp": f"{exp:%Y%m%d}",
                                                    "start_date": f"{a:%Y%m%d}", "end_date": f"{b:%Y%m%d}"})
        if r.status_code != 200:
            debug(f"{sym} {exp} eod HTTP {r.status_code}", r.text)
            return pd.DataFrame()
        j = r.json()
        fmt = j["header"]["format"]
        rows = []
        for item in j.get("response", []):
            c = item["contract"]
            for tk in item["ticks"]:
                t = dict(zip(fmt, tk))
                rows.append(dict(date=str(t["date"]), strike=c["strike"] / 1000.0, right=c["right"],
                                 bid=t.get("bid"), ask=t.get("ask"), close=t.get("close"), volume=t.get("volume")))
        out = pd.DataFrame(rows)
        if out.empty:
            return out
        out["date"] = pd.to_datetime(out["date"], format="%Y%m%d")
    out["expiration"] = pd.Timestamp(exp)
    return out[["date", "expiration", "strike", "right", "bid", "ask", "close", "volume"]]


def main():
    syms = sys.argv[1:] or DEFAULT
    base, ver = detect()
    print(f"Theta Terminal v{ver} at {base}")
    OUT.mkdir(parents=True, exist_ok=True)
    today = date.today()
    start = today - timedelta(days=DAYS_BACK)
    for sym in syms:
        path = OUT / f"{sym}.csv.gz"
        have = pd.read_csv(path, parse_dates=["date", "expiration"]) if path.exists() else pd.DataFrame()
        done = set(have["expiration"].dt.date) if len(have) else set()
        exps = [e for e in expirations(base, ver, sym)
                if third_friday(e) and start <= e <= today + timedelta(days=WINDOW)]
        print(f"{sym}: {len(exps)} monthly expirations to check, {len(done)} already saved")
        parts = [have] if len(have) else []
        for e in exps:
            if e in done and e < today:
                continue
            a, b = max(e - timedelta(days=WINDOW), start), min(e, today - timedelta(days=1))
            if a > b:
                continue
            df = eod(base, ver, sym, e, a, b)
            print(f"  {sym} {e}: {len(df)} rows")
            if len(df):
                parts = [p[p["expiration"].dt.date != e] if len(p) else p for p in parts] + [df]
                pd.concat(parts, ignore_index=True).to_csv(path, index=False, compression="gzip")
        if parts:
            tot = pd.concat(parts)
            print(f"{sym}: saved {len(tot)} rows, {tot['date'].min():%Y-%m-%d} .. {tot['date'].max():%Y-%m-%d}")


if __name__ == "__main__":
    main()
