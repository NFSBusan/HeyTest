"""Build a RULE-BASED universe of retail-heavy small/mid caps, then download their data. Run on YOUR Mac.

Rule (fixed in advance, uses only data from BEFORE the test year, so no lookahead or survivorship):
  1. FINRA off-exchange (TRF) volume for every US stock, summed over September 2025 trading days.
     Off-exchange volume is mostly retail orders routed to wholesalers.
  2. Keep stocks with listed options (ThetaData option symbol list).
  3. Drop ETFs/leveraged products (fixed list below + names ThetaData reports as having no stock data)
     and mega caps (fixed S&P 100 list).
  4. Price $2-$150 (average Sept 2025 close, ThetaData stock EOD).
  5. Rank by off-exchange DOLLAR volume (FINRA shares x price). Keep the top N (default 80).
Then: download ~1 year of daily stock prices (ThetaData) and monthly option chains (fetch_thetadata.py)
for the chosen names. Everything resumes if stopped.

  Window 1: java -jar ThetaTerminal*.jar          (leave open)
  Window 2: cd ~/HeyTest && caffeinate -i python3 scripts/build_universe.py
Output: data/universe_2025-09.csv, data/prices_td/<T>.csv, data/options/<T>.csv.gz
"""
from __future__ import annotations

import sys
import time
from datetime import date, timedelta
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fetch_thetadata as F  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TOP_N = int(sys.argv[1]) if len(sys.argv) > 1 else 80
FINRA = "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{:%Y%m%d}.txt"
S = requests.Session()
S.headers["User-Agent"] = "rflow-research/0.1"

# Fixed exclusion lists (written before looking at any results).
SP100 = set("""AAPL ABBV ABT ACN ADBE AIG AMD AMGN AMT AMZN AVGO AXP BA BAC BK BKNG BLK BMY BRK.B C CAT CHTR CL CMCSA COF
COP COST CRM CSCO CVS CVX DE DHR DIS DUK EMR F FDX GD GE GILD GM GOOG GOOGL GS HD HON IBM INTC INTU ISRG JNJ JPM KO LIN
LLY LMT LOW MA MCD MDLZ MDT MET META MMM MO MRK MS MSFT NEE NFLX NKE NOW NVDA ORCL PEP PFE PG PLTR PM PYPL QCOM RTX SBUX
SCHW SO SPG T TGT TMO TMUS TSLA TXN UBER UNH UNP UPS USB V VZ WFC WMT XOM""".split())
ETFS = set("""SPY QQQ IWM DIA TQQQ SQQQ SOXL SOXS SPXL SPXS SPXU UPRO TNA TZA UVXY VXX SVXY UVIX SVIX TSLL TSLQ TSLZ TSLS
NVDL NVDQ NVDX NVDU NVD MSTU MSTX MSTZ CONL AMDL FNGU FNGD LABU LABD YINN YANG FAS FAZ TMF TMV TLT HYG LQD XLF XLE XLK XLV
XLI XLY XLP XLU XLB XLRE XLC GDX GDXJ NUGT DUST JNUG JDST SLV GLD USO UNG UCO SCO BOIL KOLD ARKK ARKG ARKW EEM EFA FXI KWEB
EWZ IBIT BITO BITX ETHA ETHU GBTC FBTC BITU SBIT SMH SOXX XBI XOP KRE KBE IYR VNQ VOO VTI VEA VWO BND AGG SHY IEF SGOV
BIL TBT SH PSQ SDS QID SPDN TSLY NVDY MSTY CONY YMAX ULTY QQQI JEPI JEPQ SCHD DGRO RSP MAGS SSO QLD USD ROM BITI ETHE
IONZ IONX IONL RGTX RGTZ RGTU QBTX QBTZ SOUX OKLL OKLS RKLX RKLZ ASTX ASTN CRWU CWVX CRWG CORD NVDU NVDD NVDS NVDG TSLR TSLG
TSDD CONL CONI MSTZ AMDL AMDD PLTU PLTD PTIR PLTG METU AAPU AAPD GGLL AMZU AMZD BITX ETHT SOLT XRPT HOOX HOOG SMCX SMCL
AVL AVS NUKK IREX IREZ IREG NEBX NBIZ APLX APLZ WULX CIFU CIFG CLSX MRAL RIOX OKLC JOBX ARCX ONDL ONDG ONDU SOFX SOXX BULZ
TECL TECS SPXU SDOW UDOW URTY SRTY DPST DRN DRV ERX ERY GUSH DRIP UTSL MIDU WEBL WEBS HIBL RETL NAIL CURE DFEN""".split())


def finra_sept2025() -> pd.Series:
    out = ROOT / "data" / "finra" / "universe_sep2025_raw.csv"
    if out.exists():
        return pd.read_csv(out, index_col=0).iloc[:, 0]
    parts = []
    d = date(2025, 9, 1)
    while d <= date(2025, 9, 30):
        if d.weekday() < 5:
            r = S.get(FINRA.format(d), timeout=30)
            if r.status_code == 200 and r.text.strip():
                t = pd.read_csv(StringIO(r.text), sep="|")
                parts.append(t[["Symbol", "TotalVolume"]])
                print("finra", d)
            time.sleep(0.3)
        d += timedelta(days=1)
    vol = pd.concat(parts).groupby("Symbol").TotalVolume.sum().sort_values(ascending=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    vol.to_csv(out)
    return vol


def td_csv(base, path, params):
    r = F.get(f"{base}{path}", dict(params, format="csv"))
    if r.status_code != 200 or not r.text.strip():
        return None
    try:
        return pd.read_csv(StringIO(r.text))
    except Exception:  # noqa: BLE001
        F.debug(f"unparseable {path} {params}", r.text)
        return None


def option_symbols(base, ver) -> set:
    tries = [("/v3/option/list/symbols", {}), ("/v3/option/list/roots", {})] if ver == 3 else []
    for path, prm in tries:
        df = td_csv(base, path, prm)
        if df is not None and len(df):
            col = [c for c in df.columns if c.lower() in ("symbol", "root")]
            return set(df[col[0] if col else df.columns[0]].astype(str).str.upper())
    if ver == 2:
        r = F.get(f"{base}/v2/list/roots/option", {})
        if r.status_code == 200:
            return set(map(str.upper, r.json()["response"]))
    F.debug("could not list option symbols", "")
    sys.exit("Could not get the option symbol list from ThetaData. Paste data/options/_debug.txt to Claude.")


def stock_eod(base, ver, sym, a: date, b: date) -> pd.DataFrame | None:
    if ver == 3:
        df = td_csv(base, "/v3/stock/history/eod", {"symbol": sym, "start_date": f"{a:%Y%m%d}", "end_date": f"{b:%Y%m%d}"})
    else:
        r = F.get(f"{base}/v2/hist/stock/eod", {"root": sym, "start_date": f"{a:%Y%m%d}", "end_date": f"{b:%Y%m%d}"})
        df = None
        if r.status_code == 200:
            j = r.json()
            df = pd.DataFrame(j.get("response", []), columns=j["header"]["format"])
    if df is None or df.empty:
        return None
    low = {c.lower(): c for c in df.columns}
    dcol = next((low[k] for k in ("date", "created", "last_trade") if k in low), None)
    if dcol is None or "close" not in low:
        F.debug(f"unexpected stock eod columns for {sym}: {list(df.columns)}", df.head().to_string())
        return None
    out = pd.DataFrame({"date": pd.to_datetime(df[dcol].astype(str).str[:10].str.replace("-", ""), format="%Y%m%d"),
                        "open": df[low.get("open", low["close"])], "high": df[low.get("high", low["close"])],
                        "low": df[low.get("low", low["close"])], "close": df[low["close"]],
                        "volume": df[low["volume"]] if "volume" in low else 0})
    return out[out.close > 0].drop_duplicates("date").sort_values("date")


def main():
    base, ver = F.detect()
    print(f"Theta Terminal v{ver} at {base}")
    uni_path = ROOT / "data" / "universe_2025-09.csv"
    if uni_path.exists():
        uni = pd.read_csv(uni_path)
        print(f"universe already built: {len(uni)} names")
    else:
        vol = finra_sept2025()
        opt = option_symbols(base, ver)
        cand = [s for s in vol.index if s in opt and s not in SP100 and s not in ETFS and s.isalpha()][:400]
        failed = 0
        print(f"{len(cand)} optionable candidates by off-exchange share volume; pricing them (Sept 2025)...")
        rows = []
        for i, s in enumerate(cand):
            px = stock_eod(base, ver, s, date(2025, 9, 1), date(2025, 9, 30))
            if px is None or px.empty:
                failed += 1
                continue
            p = px.close.mean()
            rows.append(dict(symbol=s, finra_shares=int(vol[s]), avg_price=round(p, 2),
                             offex_dollar_vol=round(vol[s] * p), avg_total_volume=px.volume.mean()))
            if i % 25 == 0:
                print(f"  priced {i}/{len(cand)}")
        if failed > 0.5 * len(cand):
            sys.exit(f"Could not price {failed} of {len(cand)} candidates. The Theta Terminal is probably not working "
                     "properly. Restart it and rerun; the universe was NOT saved.")
        df = pd.DataFrame(rows)
        df = df[(df.avg_price >= 2) & (df.avg_price <= 150)].sort_values("offex_dollar_vol", ascending=False)
        uni = df.head(TOP_N).reset_index(drop=True)
        uni_path.parent.mkdir(parents=True, exist_ok=True)
        uni.to_csv(uni_path, index=False)
        print(f"universe: {len(uni)} names -> {uni_path}")
    syms = list(uni.symbol)
    # 1 year of daily prices (ThetaData free tier) for names without an IBKR file
    pdir = ROOT / "data" / "prices_td"
    pdir.mkdir(parents=True, exist_ok=True)
    for s in syms:
        f = pdir / f"{s}.csv"
        if f.exists():
            continue
        px = stock_eod(base, ver, s, date.today() - timedelta(days=364), date.today())
        if px is not None and len(px):
            px.to_csv(f, index=False)
            print(f"prices {s}: {len(px)} days")
    # monthly option chains (skips names already downloaded)
    sys.argv = [sys.argv[0]] + syms
    F.main()


if __name__ == "__main__":
    main()
