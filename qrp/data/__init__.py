"""Data engine: providers -> Parquet store -> aligned panels."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .panel import Panel, build_panel
from .providers import make_provider
from .store import BarStore, FetchReport

log = logging.getLogger(__name__)


def store_for(cfg) -> BarStore:
    return BarStore(cfg["storage"]["root"], cfg["data"]["provider"], cfg["data"]["timeframe"])


def fetch_all(cfg) -> list[FetchReport]:
    provider = make_provider(cfg)
    store = store_for(cfg)
    d = cfg["data"]
    reports = []
    for s in d["symbols"]:
        rep = store.update(provider, s, pd.Timestamp(d["start"]).date(), pd.Timestamp(d["end"]).date(),
                           d["timeframe"])
        log.info("%s: +%d rows (%d stored)", s, rep.fetched_rows, rep.stored_rows)
        reports.append(rep)
    return reports


def load_panel(cfg, fetch_missing: bool = True) -> Panel:
    store = store_for(cfg)
    d = cfg["data"]
    if fetch_missing and any(not store.exists(s) for s in d["symbols"]):
        fetch_all(cfg)
    long = store.query(d["symbols"], d["start"], d["end"])
    universe = None
    if d.get("universe_file"):
        universe = pd.read_csv(Path(d["universe_file"]))
    panel = build_panel(long, d["symbols"], universe)
    if d["benchmark"] not in panel.symbols:
        raise ValueError(f"benchmark {d['benchmark']} has no data")
    return panel
