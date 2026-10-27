"""Experiment registry (DuckDB): every study, every configuration tried, every holdout look.

Two rules are enforced here:
1. A configuration tested on the same dev data and costs is never re-run; its stored
   result is reused. It still counts once toward the multiple-testing correction.
2. Trial counts are per dev-data fingerprint, across ALL studies, so starting a new study
   (or letting Claude propose more experiments) cannot reset the count.
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA = """
CREATE TABLE IF NOT EXISTS studies (
    study_id VARCHAR PRIMARY KEY, created_at TIMESTAMP, name VARCHAR, config_json VARCHAR,
    dev_fp VARCHAR, holdout_fp VARCHAR, cost_fp VARCHAR, split_json VARCHAR, status VARCHAR);
CREATE TABLE IF NOT EXISTS trials (
    dev_fp VARCHAR, cost_fp VARCHAR, spec_hash VARCHAR, family VARCHAR, source VARCHAR,
    counted BOOLEAN, study_id VARCHAR, spec_json VARCHAR, created_at TIMESTAMP, returns_file VARCHAR,
    sharpe DOUBLE, alpha_sharpe DOUBLE, cagr DOUBLE, max_drawdown DOUBLE, n_trades INTEGER,
    cost_drag DOUBLE, metrics_json VARCHAR,
    PRIMARY KEY (dev_fp, cost_fp, spec_hash));
CREATE TABLE IF NOT EXISTS family_results (
    study_id VARCHAR, family VARCHAR, metrics_json VARCHAR, PRIMARY KEY (study_id, family));
CREATE TABLE IF NOT EXISTS candidates (
    study_id VARCHAR, spec_hash VARCHAR, family VARCHAR, status VARCHAR, failed_json VARCHAR,
    gates_json VARCHAR, robustness_json VARCHAR, PRIMARY KEY (study_id, spec_hash));
CREATE TABLE IF NOT EXISTS holdout_access (
    access_id VARCHAR PRIMARY KEY, holdout_fp VARCHAR, study_id VARCHAR, accessed_at TIMESTAMP,
    spec_hashes_json VARCHAR, forced BOOLEAN, reason VARCHAR, results_json VARCHAR);
CREATE TABLE IF NOT EXISTS proposals (
    proposal_id VARCHAR PRIMARY KEY, created_at TIMESTAMP, status VARCHAR, dev_fp VARCHAR,
    payload_json VARCHAR, n_configs INTEGER, decided_at TIMESTAMP, executed_at TIMESTAMP, notes VARCHAR);
CREATE TABLE IF NOT EXISTS llm_usage (
    ts TIMESTAMP, purpose VARCHAR, model VARCHAR, input_tokens BIGINT, output_tokens BIGINT,
    cost_usd DOUBLE, request_id VARCHAR);
"""


def now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Registry:
    def __init__(self, db_path: str, results_root: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(db_path)
        self.con.execute(SCHEMA)
        self.results_root = Path(results_root)

    def close(self):
        self.con.close()

    # ---------------------------------------------------------------- studies
    def create_study(self, name: str, config: dict, dev_fp: str, holdout_fp: str, cost_fp: str,
                     split: dict) -> str:
        sid = now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        self.con.execute("INSERT INTO studies VALUES (?,?,?,?,?,?,?,?,?)",
                         [sid, now(), name, json.dumps(config, default=str), dev_fp, holdout_fp, cost_fp,
                          json.dumps(split), "running"])
        return sid

    def set_study_status(self, sid: str, status: str):
        self.con.execute("UPDATE studies SET status=? WHERE study_id=?", [status, sid])

    def study(self, sid: str | None = None) -> dict:
        q = "SELECT * FROM studies " + ("WHERE study_id=?" if sid else "ORDER BY created_at DESC LIMIT 1")
        df = self.con.execute(q, [sid] if sid else []).df()
        if df.empty:
            raise KeyError("no such study" if sid else "no studies yet; run `qrp research` first")
        return df.iloc[0].to_dict()

    # ---------------------------------------------------------------- trials
    def tested_hashes(self, dev_fp: str, cost_fp: str, family: str | None = None) -> set[str]:
        q = "SELECT spec_hash FROM trials WHERE dev_fp=? AND cost_fp=?"
        args = [dev_fp, cost_fp]
        if family:
            q += " AND family=?"
            args.append(family)
        return set(self.con.execute(q, args).df()["spec_hash"])

    def n_trials(self, dev_fp: str, family: str | None = None) -> int:
        """Distinct counted configurations ever evaluated on this dev data (any costs, any study)."""
        q = "SELECT COUNT(DISTINCT spec_hash) FROM trials WHERE dev_fp=? AND counted"
        args = [dev_fp]
        if family:
            q += " AND family=?"
            args.append(family)
        return int(self.con.execute(q, args).fetchone()[0])

    def save_trials(self, dev_fp: str, cost_fp: str, study_id: str, family: str, source: str,
                    counted: bool, rows: list[dict], returns: dict[str, np.ndarray]) -> None:
        """rows: dicts with spec_hash, spec, metrics. returns: spec_hash -> float32 daily returns."""
        if not rows:
            return
        d = self.results_root / f"{dev_fp}_{cost_fp}" / family
        d.mkdir(parents=True, exist_ok=True)
        fname = d / f"{source}_{now().strftime('%Y%m%d%H%M%S%f')}_{uuid.uuid4().hex[:6]}.parquet"
        table = pa.table({h: pa.array(np.asarray(v, dtype=np.float32)) for h, v in returns.items()})
        pq.write_table(table, fname, compression="zstd")
        recs = []
        for r in rows:
            m = r["metrics"]
            recs.append([dev_fp, cost_fp, r["spec_hash"], family, source, counted, study_id,
                         json.dumps(r["spec"]), now(), str(fname), m.get("sharpe"), m.get("alpha_sharpe"),
                         m.get("cagr"), m.get("max_drawdown"), m.get("n_trades"), m.get("cost_drag"),
                         json.dumps(m)])
        self.con.executemany("INSERT OR IGNORE INTO trials VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", recs)

    def trials(self, dev_fp: str, cost_fp: str, family: str | None = None,
               counted_only: bool = True) -> pd.DataFrame:
        q = "SELECT * FROM trials WHERE dev_fp=? AND cost_fp=?"
        args = [dev_fp, cost_fp]
        if family:
            q += " AND family=?"
            args.append(family)
        if counted_only:
            q += " AND counted"
        return self.con.execute(q + " ORDER BY spec_hash", args).df()

    def trial_returns(self, trials: pd.DataFrame) -> tuple[list[str], np.ndarray]:
        """(T, K) matrix of stored daily returns for the given trial rows (column order = rows)."""
        cols: dict[str, np.ndarray] = {}
        for fname, grp in trials.groupby("returns_file"):
            t = pq.read_table(fname, columns=list(grp["spec_hash"]))
            for h in grp["spec_hash"]:
                cols[h] = t.column(h).to_numpy()
        hashes = list(trials["spec_hash"])
        if not hashes:
            return [], np.zeros((0, 0))
        return hashes, np.column_stack([cols[h] for h in hashes]).astype(np.float64)

    # ---------------------------------------------------------------- results
    def save_family(self, sid: str, family: str, metrics: dict):
        self.con.execute("INSERT OR REPLACE INTO family_results VALUES (?,?,?)",
                         [sid, family, json.dumps(metrics, default=float)])

    def family_results(self, sid: str) -> dict[str, dict]:
        df = self.con.execute("SELECT family, metrics_json FROM family_results WHERE study_id=?", [sid]).df()
        return {r.family: json.loads(r.metrics_json) for r in df.itertuples()}

    def save_candidate(self, sid: str, spec_hash: str, family: str, status: str, failed: list[str],
                       gates: dict, robustness: dict):
        self.con.execute("INSERT OR REPLACE INTO candidates VALUES (?,?,?,?,?,?,?)",
                         [sid, spec_hash, family, status, json.dumps(failed), json.dumps(gates, default=float),
                          json.dumps(robustness, default=float)])

    def candidates(self, sid: str) -> pd.DataFrame:
        return self.con.execute("SELECT * FROM candidates WHERE study_id=? ORDER BY family, spec_hash",
                                [sid]).df()

    def set_candidate_status(self, sid: str, spec_hash: str, status: str):
        self.con.execute("UPDATE candidates SET status=? WHERE study_id=? AND spec_hash=?",
                         [status, sid, spec_hash])

    # ---------------------------------------------------------------- holdout log
    def holdout_accesses(self, holdout_fp: str) -> pd.DataFrame:
        return self.con.execute("SELECT * FROM holdout_access WHERE holdout_fp=? ORDER BY accessed_at",
                                [holdout_fp]).df()

    def log_holdout_access(self, holdout_fp: str, sid: str, hashes: list[str], forced: bool,
                           reason: str) -> str:
        aid = uuid.uuid4().hex[:12]
        self.con.execute("INSERT INTO holdout_access VALUES (?,?,?,?,?,?,?,?)",
                         [aid, holdout_fp, sid, now(), json.dumps(hashes), forced, reason, None])
        return aid

    def save_holdout_results(self, access_id: str, results: dict):
        self.con.execute("UPDATE holdout_access SET results_json=? WHERE access_id=?",
                         [json.dumps(results, default=float), access_id])

    # ---------------------------------------------------------------- llm usage
    def log_llm(self, purpose: str, model: str, tin: int, tout: int, cost: float, request_id: str = ""):
        self.con.execute("INSERT INTO llm_usage VALUES (?,?,?,?,?,?,?)",
                         [now(), purpose, model, tin, tout, cost, request_id])

    def llm_spend_this_month(self) -> float:
        start = now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        v = self.con.execute("SELECT COALESCE(SUM(cost_usd),0) FROM llm_usage WHERE ts>=?", [start]).fetchone()
        return float(v[0])


def open_registry(cfg) -> Registry:
    root = cfg["storage"]["root"]
    results_root = os.path.join(root, "results") if not root.startswith("s3://") else "data/qrp/results"
    return Registry(cfg["storage"]["db_path"], results_root)
