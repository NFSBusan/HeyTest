"""Bar storage: one compressed Parquet file per symbol, queried with DuckDB.

Layout: {root}/bars/{provider}/{timeframe}/{SYMBOL}.parquet

`root` can be a local folder or s3://bucket/prefix (Cloudflare R2, Backblaze B2, AWS S3).
Only missing date ranges are downloaded, so re-running a fetch costs no API calls.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import date, timedelta

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq

from .providers import BAR_COLUMNS, Provider, normalize
from .quality import check_bars

log = logging.getLogger(__name__)


def _s3_filesystem() -> pafs.S3FileSystem:
    return pafs.S3FileSystem(
        access_key=os.environ.get("QRP_S3_ACCESS_KEY_ID"),
        secret_key=os.environ.get("QRP_S3_SECRET_ACCESS_KEY"),
        endpoint_override=os.environ.get("QRP_S3_ENDPOINT") or None,
        region=os.environ.get("QRP_S3_REGION", "auto"),
    )


@dataclass
class FetchReport:
    symbol: str
    fetched_rows: int
    stored_rows: int
    issues: list[str]


class BarStore:
    def __init__(self, root: str, provider_name: str, timeframe: str = "1d"):
        self.remote = root.startswith("s3://")
        if self.remote:
            self.fs = _s3_filesystem()
            self.base = root[len("s3://"):].rstrip("/")
        else:
            self.fs = pafs.LocalFileSystem()
            self.base = os.path.abspath(root)
        self.root = root.rstrip("/")
        self.dir = f"{self.base}/bars/{provider_name}/{timeframe}"
        self.fs.create_dir(self.dir, recursive=True)

    def _path(self, symbol: str) -> str:
        return f"{self.dir}/{symbol.upper()}.parquet"

    def _uri(self, symbol: str) -> str:
        return ("s3://" if self.remote else "") + self._path(symbol)

    def exists(self, symbol: str) -> bool:
        return self.fs.get_file_info(self._path(symbol)).type == pafs.FileType.File

    def read(self, symbol: str) -> pd.DataFrame:
        if not self.exists(symbol):
            return normalize(pd.DataFrame(columns=BAR_COLUMNS))
        return pq.read_table(self._path(symbol), filesystem=self.fs).to_pandas()

    def write(self, symbol: str, df: pd.DataFrame) -> None:
        table = pa.Table.from_pandas(normalize(df), preserve_index=False)
        tmp = self._path(symbol) + ".tmp"
        pq.write_table(table, tmp, filesystem=self.fs, compression="zstd")
        self.fs.move(tmp, self._path(symbol))

    def coverage(self, symbol: str) -> tuple[pd.Timestamp, pd.Timestamp] | None:
        df = self.read(symbol)
        if df.empty:
            return None
        return df["date"].min(), df["date"].max()

    def update(self, provider: Provider, symbol: str, start: date, end: date,
               timeframe: str = "1d") -> FetchReport:
        """Download only the ranges not already stored, merge, validate and save."""
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        have = self.coverage(symbol)
        old = self.read(symbol)
        ranges: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        if have is None:
            ranges.append((start, end))
        else:
            lo, hi = have
            # A gap of a few days at either end is weekends/holidays, not missing data.
            if start < lo - timedelta(days=5):
                ranges.append((start, lo - timedelta(days=1)))
            if end > hi + timedelta(days=4):
                # Re-fetch a short overlap: adjusted prices are rewritten back through history after
                # every dividend or split. If the overlap moved, the stored history is stale.
                overlap_start = hi - timedelta(days=14)
                fresh = provider.fetch(symbol, overlap_start.date(), end.date(), timeframe)
                if self._adjustment_changed(old, fresh):
                    log.info("%s: adjustment factors changed; re-downloading full history", symbol)
                    full = provider.fetch(symbol, min(start, lo).date(), end.date(), timeframe)
                    self.write(symbol, full)
                    return FetchReport(symbol, len(full), len(full), check_bars(full) + ["re-adjusted"])
                ranges_parts = [fresh]
                for a, b in ranges:
                    ranges_parts.append(provider.fetch(symbol, a.date(), b.date(), timeframe))
                return self._merge(symbol, old, ranges_parts)
        new_parts = [provider.fetch(symbol, a.date(), b.date(), timeframe) for a, b in ranges]
        return self._merge(symbol, old, new_parts)

    @staticmethod
    def _adjustment_changed(old: pd.DataFrame, fresh: pd.DataFrame, tol: float = 1e-6) -> bool:
        if old.empty or fresh.empty:
            return False
        m = old[["date", "close"]].merge(fresh[["date", "close"]], on="date", suffixes=("_old", "_new"))
        if m.empty:
            return False
        return bool((abs(m["close_new"] / m["close_old"] - 1) > tol).any())

    def _merge(self, symbol: str, old: pd.DataFrame, new_parts: list[pd.DataFrame]) -> FetchReport:
        fetched = sum(len(p) for p in new_parts)
        if fetched:
            merged = normalize(pd.concat([old, *new_parts], ignore_index=True))
            self.write(symbol, merged)
        else:
            merged = old
        issues = check_bars(merged)
        for msg in issues:
            log.warning("%s: %s", symbol, msg)
        return FetchReport(symbol, fetched, len(merged), issues)

    def query(self, symbols: list[str], start: str | None = None, end: str | None = None) -> pd.DataFrame:
        """Long-format bars for several symbols, filtered inside DuckDB (no full download needed)."""
        files = [self._uri(s) for s in symbols if self.exists(s)]
        if not files:
            return pd.DataFrame(columns=["symbol", *BAR_COLUMNS])
        con = duckdb.connect()
        if self.remote:
            _configure_duckdb_s3(con)
        where, params = [], []
        if start:
            where.append("date >= ?")
            params.append(pd.Timestamp(start))
        if end:
            where.append("date <= ?")
            params.append(pd.Timestamp(end))
        clause = ("WHERE " + " AND ".join(where)) if where else ""
        sql = (f"SELECT regexp_extract(filename, '([^/]+)\\.parquet$', 1) AS symbol, "
               f"{', '.join(BAR_COLUMNS)} FROM read_parquet(?, filename=true, union_by_name=true) "
               f"{clause} ORDER BY symbol, date")
        df = con.execute(sql, [files, *params]).df()
        con.close()
        return df


def _configure_duckdb_s3(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("INSTALL httpfs; LOAD httpfs;")
    endpoint = os.environ.get("QRP_S3_ENDPOINT", "").replace("https://", "")
    con.execute(
        "CREATE OR REPLACE SECRET qrp_s3 (TYPE s3, KEY_ID ?, SECRET ?, REGION ?, ENDPOINT ?, URL_STYLE 'path')",
        [os.environ.get("QRP_S3_ACCESS_KEY_ID", ""), os.environ.get("QRP_S3_SECRET_ACCESS_KEY", ""),
         os.environ.get("QRP_S3_REGION", "auto"), endpoint])
