"""Historical bar providers.

Every provider returns the same frame: date, open, high, low, close, volume, all adjusted for
splits AND dividends (total-return prices), sorted by date with one row per trading day.
Raw (unadjusted) close is kept in `raw_close` when the source has it.

The real providers (Tiingo, Alpaca) follow their documented REST endpoints. They are tested
offline against recorded response shapes only; this cloud session cannot reach those hosts.
"""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from ..config import secret

log = logging.getLogger(__name__)

BAR_COLUMNS = ["date", "open", "high", "low", "close", "volume"]


class ProviderError(RuntimeError):
    pass


def _empty() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="float64") for c in BAR_COLUMNS}).assign(
        date=pd.Series(dtype="datetime64[ns]"))


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce a provider frame to the canonical schema (naive dates, float columns, sorted, unique)."""
    if df.empty:
        return _empty()
    out = df.copy()
    d = pd.to_datetime(out["date"], utc=True)
    out["date"] = d.dt.tz_localize(None).dt.normalize()
    for c in BAR_COLUMNS[1:]:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")
    keep = BAR_COLUMNS + [c for c in ("raw_close",) if c in out.columns]
    out = out[keep].dropna(subset=["close"])
    out = out.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
    return out


class Provider(ABC):
    name = "base"

    @abstractmethod
    def fetch(self, symbol: str, start: date, end: date, timeframe: str = "1d") -> pd.DataFrame:
        ...


class HttpProvider(Provider):
    """Shared HTTP plumbing: session reuse, retry with exponential backoff, simple rate limit."""

    min_interval_s = 0.0
    max_retries = 4

    def __init__(self, session: requests.Session | None = None):
        self.session = session or requests.Session()
        self._last_call = 0.0

    def _get(self, url: str, params: dict, headers: dict) -> requests.Response:
        for attempt in range(self.max_retries + 1):
            wait = self.min_interval_s - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()
            try:
                r = self.session.get(url, params=params, headers=headers, timeout=30)
            except requests.RequestException as e:
                if attempt == self.max_retries:
                    raise ProviderError(f"{self.name}: network error: {e}") from e
                time.sleep(2 ** attempt)
                continue
            if r.status_code == 200:
                return r
            if r.status_code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                retry_after = r.headers.get("Retry-After")
                time.sleep(float(retry_after) if retry_after and retry_after.isdigit() else 2 ** attempt)
                continue
            raise ProviderError(f"{self.name}: HTTP {r.status_code} for {url}: {r.text[:200]}")
        raise ProviderError(f"{self.name}: retries exhausted for {url}")


class TiingoProvider(HttpProvider):
    """Tiingo end-of-day prices (free tier: ~50 requests/hour, ~1000/day, 500 symbols/month).

    Endpoint: GET https://api.tiingo.com/tiingo/daily/{ticker}/prices
    Adjusted fields (adjOpen/adjHigh/adjLow/adjClose/adjVolume) include splits and dividends.
    """

    name = "tiingo"
    base_url = "https://api.tiingo.com/tiingo/daily/{symbol}/prices"
    min_interval_s = 1.0

    def __init__(self, api_key: str | None = None, session: requests.Session | None = None):
        super().__init__(session)
        self.api_key = api_key or secret("TIINGO_API_KEY")

    def fetch(self, symbol: str, start: date, end: date, timeframe: str = "1d") -> pd.DataFrame:
        if timeframe != "1d":
            raise ProviderError("tiingo provider here supports daily bars only (use alpaca for intraday)")
        r = self._get(self.base_url.format(symbol=symbol.lower()),
                      params={"startDate": str(start), "endDate": str(end), "format": "json",
                              "resampleFreq": "daily"},
                      headers={"Authorization": f"Token {self.api_key}",
                               "Content-Type": "application/json"})
        rows = r.json()
        if not rows:
            return _empty()
        raw = pd.DataFrame(rows)
        df = pd.DataFrame({"date": raw["date"], "open": raw["adjOpen"], "high": raw["adjHigh"],
                           "low": raw["adjLow"], "close": raw["adjClose"], "volume": raw["adjVolume"],
                           "raw_close": raw["close"]})
        return normalize(df)


class AlpacaProvider(HttpProvider):
    """Alpaca market data v2 (free Basic plan: IEX feed, ~200 requests/minute).

    Endpoint: GET https://data.alpaca.markets/v2/stocks/bars with adjustment=all (splits+dividends).
    Note: the free IEX feed only covers IEX volume (a few % of the market); prices are fine for
    daily research, volume-based features are not. History starts around 2016.
    """

    name = "alpaca"
    base_url = "https://data.alpaca.markets/v2/stocks/bars"
    min_interval_s = 0.35
    timeframes = {"1d": "1Day", "1h": "1Hour", "1m": "1Min"}

    def __init__(self, key_id: str | None = None, secret_key: str | None = None, feed: str = "iex",
                 session: requests.Session | None = None):
        super().__init__(session)
        self.key_id = key_id or secret("ALPACA_API_KEY_ID")
        self.secret_key = secret_key or secret("ALPACA_API_SECRET_KEY")
        self.feed = feed

    def fetch(self, symbol: str, start: date, end: date, timeframe: str = "1d") -> pd.DataFrame:
        if timeframe != "1d":
            # The API serves 1Hour/1Min too; the research panel is daily-only for now (phase 5).
            raise ProviderError("alpaca: only daily bars are wired into the research panel so far")
        params = {"symbols": symbol, "timeframe": self.timeframes[timeframe], "start": str(start),
                  "end": str(end), "adjustment": "all", "feed": self.feed, "limit": 10000}
        headers = {"APCA-API-KEY-ID": self.key_id, "APCA-API-SECRET-KEY": self.secret_key}
        bars: list[dict] = []
        while True:
            data = self._get(self.base_url, params, headers).json()
            bars.extend((data.get("bars") or {}).get(symbol, []))
            token = data.get("next_page_token")
            if not token:
                break
            params = {**params, "page_token": token}
        if not bars:
            return _empty()
        df = pd.DataFrame(bars).rename(columns={"t": "date", "o": "open", "h": "high", "l": "low",
                                                "c": "close", "v": "volume"})
        return normalize(df)


class CsvProvider(Provider):
    """Import files you already have (e.g. exports from Norgate, Sharadar or a broker).

    Expected columns: date, open, high, low, close, volume (adjusted). Extra columns are ignored.
    """

    name = "csv"

    def __init__(self, pattern: str):
        self.pattern = pattern

    def fetch(self, symbol: str, start: date, end: date, timeframe: str = "1d") -> pd.DataFrame:
        p = Path(self.pattern.format(symbol=symbol))
        if not p.exists():
            raise ProviderError(f"csv: file not found {p}")
        df = normalize(pd.read_csv(p))
        m = (df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))
        return df[m].reset_index(drop=True)


class SyntheticProvider(Provider):
    """Deterministic simulated market for offline development and for testing the pipeline.

    edge = "none": no exploitable structure. Anything that "passes" here is a false positive.
    edge = "trend": slowly drifting expected returns, so trend-following has a real but weak edge.
    edge = "reversal": short-term negative autocorrelation in idiosyncratic returns.
    Results on synthetic data say nothing about real markets.
    """

    name = "synthetic"

    def __init__(self, symbols: list[str], start: str, end: str, seed: int = 42, edge: str = "none",
                 edge_strength: float = 1.0, late_starters: int = 2):
        if edge not in ("none", "trend", "reversal"):
            raise ProviderError(f"synthetic: unknown edge {edge}")
        self.symbols = list(symbols)
        self._frames = _simulate(self.symbols, start, end, seed, edge, edge_strength, late_starters)

    def fetch(self, symbol: str, start: date, end: date, timeframe: str = "1d") -> pd.DataFrame:
        if symbol not in self._frames:
            raise ProviderError(f"synthetic: unknown symbol {symbol}")
        df = self._frames[symbol]
        m = (df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))
        return df[m].reset_index(drop=True)


def _simulate(symbols, start, end, seed, edge, strength, late_starters) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    T, N = len(dates), len(symbols)
    # Market factor with two volatility regimes (calm / stressed) and fat tails.
    regime = np.zeros(T, dtype=int)
    for t in range(1, T):
        stay = 0.995 if regime[t - 1] == 0 else 0.98
        regime[t] = regime[t - 1] if rng.random() < stay else 1 - regime[t - 1]
    mkt_vol = np.where(regime == 0, 0.007, 0.018)
    shocks = rng.standard_t(5, size=T) / np.sqrt(5 / 3)
    mkt = 0.0003 + mkt_vol * shocks
    beta = rng.uniform(-0.2, 1.3, N)
    idio_vol = rng.uniform(0.004, 0.012, N)
    beta[0], idio_vol[0] = 1.0, 0.0005      # the first symbol (SPY in the default config) is the market
    # With no planted edge every asset has the same expected return given its market exposure,
    # so the only real premium is market beta; any "alpha" found there is a false positive.
    mu = np.full(N, 0.00005) if edge == "none" else rng.uniform(-0.0001, 0.0002, N)
    idio = rng.standard_t(5, size=(T, N)) / np.sqrt(5 / 3) * idio_vol
    if edge == "reversal":
        phi = 0.12 * strength
        for t in range(1, T):
            idio[t] -= phi * idio[t - 1]
    drift = np.zeros((T, N))
    if edge == "trend":
        rho, sd = 0.995, 0.0007 * strength
        innov = rng.normal(0, sd * np.sqrt(1 - rho ** 2), size=(T, N))
        for t in range(1, T):
            drift[t] = rho * drift[t - 1] + innov[t]
    ret = mu + beta * mkt[:, None] + idio + drift
    logp = np.log(100.0) + np.cumsum(np.log1p(ret), axis=0)
    close = np.exp(logp)
    prev = np.vstack([close[:1], close[:-1]])
    split = rng.uniform(0.2, 0.5, size=(T, N))      # share of the day's move that happens overnight
    open_ = prev * np.exp(split * np.log(close / prev) + rng.normal(0, 0.001, (T, N)))
    tot_vol = np.sqrt((beta * mkt_vol[:, None]) ** 2 + idio_vol ** 2)
    hi = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, 0.4, (T, N))) * tot_vol)
    lo = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, 0.4, (T, N))) * tot_vol)
    dollar_vol = np.exp(rng.uniform(np.log(2e8), np.log(3e10), N))
    volume = dollar_vol / close * np.exp(rng.normal(0, 0.3, (T, N))) * (1 + regime[:, None])
    frames = {}
    for i, s in enumerate(symbols):
        first = 0
        if late_starters and i >= N - late_starters:
            first = min(T - 1, 756)     # a few funds launch three years later
        frames[s] = pd.DataFrame({"date": dates[first:], "open": open_[first:, i], "high": hi[first:, i],
                                  "low": lo[first:, i], "close": close[first:, i],
                                  "volume": volume[first:, i]})
    return frames


def make_provider(cfg) -> Provider:
    d = cfg["data"]
    p = d["provider"]
    if p == "synthetic":
        syn = d.get("synthetic", {})
        return SyntheticProvider(d["symbols"], d["start"], d["end"], seed=cfg["project"]["seed"],
                                 edge=syn.get("edge", "none"), edge_strength=syn.get("edge_strength", 1.0))
    if p == "tiingo":
        return TiingoProvider()
    if p == "alpaca":
        return AlpacaProvider()
    if p == "csv":
        return CsvProvider(d["csv_pattern"])
    raise ProviderError(f"unknown provider {p}")
