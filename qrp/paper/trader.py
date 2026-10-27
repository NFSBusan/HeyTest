"""Paper trading: run approved strategies forward on new data with simulated fills.

Run once per trading day after the close (cron / GitHub Actions). Each run:
  1. fills the orders created on the previous run at today's open (or close, per config),
  2. marks positions to today's close and records equity,
  3. checks risk limits (daily loss, drawdown kill, stale data, kill-switch file, deterioration),
  4. computes tomorrow's target weights and stores them as pending orders.

There is no broker connection in this module and no code path that sends a real order.
`paper.live_trading_enabled: true` makes the config loader refuse to start.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import requests

from ..backtest import Market
from ..data.panel import Panel
from ..evaluate import EvalSettings, Evaluator
from ..registry import now

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS strategies (
    strategy_id VARCHAR PRIMARY KEY, spec_json VARCHAR, study_id VARCHAR, description VARCHAR,
    started_at TIMESTAMP, status VARCHAR, cash DOUBLE, peak_equity DOUBLE, last_date DATE,
    deterioration_threshold DOUBLE, notes VARCHAR);
CREATE TABLE IF NOT EXISTS orders (
    strategy_id VARCHAR, signal_date DATE, symbol VARCHAR, target_weight DOUBLE, status VARCHAR,
    fill_date DATE, fill_price DOUBLE, qty DOUBLE, cost DOUBLE);
CREATE TABLE IF NOT EXISTS positions (strategy_id VARCHAR, symbol VARCHAR, qty DOUBLE,
    PRIMARY KEY (strategy_id, symbol));
CREATE TABLE IF NOT EXISTS equity (strategy_id VARCHAR, date DATE, equity DOUBLE, cash DOUBLE,
    gross_exposure DOUBLE, daily_return DOUBLE, PRIMARY KEY (strategy_id, date));
CREATE TABLE IF NOT EXISTS events (ts TIMESTAMP, strategy_id VARCHAR, level VARCHAR, kind VARCHAR,
    message VARCHAR);
"""


class LiveTradingRefused(RuntimeError):
    pass


@dataclass
class PaperConfig:
    starting_cash: float
    max_daily_loss: float
    kill_drawdown: float
    max_stale_days: int
    deterioration_window: int
    kill_switch_file: str

    @classmethod
    def from_config(cls, cfg) -> "PaperConfig":
        p = cfg["paper"]
        if p.get("live_trading_enabled"):
            raise LiveTradingRefused("live trading is not implemented and stays off without explicit approval")
        return cls(p["starting_cash"], p["max_daily_loss"], p["kill_drawdown"], p["max_stale_days"],
                   p["deterioration_window"], p["kill_switch_file"])


class PaperTrader:
    def __init__(self, cfg, db_path: str | None = None):
        self.cfg = cfg
        self.pc = PaperConfig.from_config(cfg)
        path = db_path or cfg["storage"]["paper_db_path"]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(path)
        self.con.execute(SCHEMA)

    def close(self):
        self.con.close()

    # ---------------------------------------------------------------- setup
    def add_strategy(self, spec_hash: str, spec: dict, study_id: str, description: str,
                     deterioration_threshold: float | None, notes: str = ""):
        self.con.execute("INSERT OR REPLACE INTO strategies VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                         [spec_hash, json.dumps(spec), study_id, description, now(), "active",
                          self.pc.starting_cash, self.pc.starting_cash, None, deterioration_threshold, notes])
        self.event(spec_hash, "info", "start", f"paper trading started with ${self.pc.starting_cash:,.0f}")

    def strategies(self, active_only: bool = True) -> pd.DataFrame:
        q = "SELECT * FROM strategies" + (" WHERE status='active'" if active_only else "")
        return self.con.execute(q).df()

    def event(self, sid: str, level: str, kind: str, msg: str):
        self.con.execute("INSERT INTO events VALUES (?,?,?,?,?)", [now(), sid, level, kind, msg])
        (log.warning if level != "info" else log.info)("[%s] %s: %s", sid, kind, msg)
        if level in ("warn", "critical"):
            _alert(f"[qrp paper] {sid} {kind}: {msg}")

    # ---------------------------------------------------------------- daily step
    def run_day(self, panel: Panel, today: pd.Timestamp | None = None) -> dict:
        """Process the latest bar in `panel` for every active strategy. Idempotent per date."""
        if not len(panel.dates):
            return {}
        t = panel.T - 1
        date = pd.Timestamp(panel.dates[t])
        today = today or pd.Timestamp.today().normalize()
        stale = int(np.busday_count(panel.dates[t], np.datetime64(today.date(), "D")))
        kill_switch = os.path.exists(self.pc.kill_switch_file)
        settings = EvalSettings.from_config(self.cfg, panel, 0)
        market = Market.from_panel(panel)
        ev = None
        summary = {}
        for s in self.strategies().itertuples():
            if s.last_date is not None and pd.Timestamp(s.last_date) >= date:
                summary[s.strategy_id] = "already processed"
                continue
            if stale > self.pc.max_stale_days:
                self.event(s.strategy_id, "warn", "stale_data",
                           f"latest bar {date.date()} is {stale} business days old; skipped")
                summary[s.strategy_id] = "stale"
                continue
            ev = ev or Evaluator(panel, settings)
            summary[s.strategy_id] = self._step(s, panel, market, ev, t, date, kill_switch)
        return summary

    def _positions(self, sid: str) -> dict[str, float]:
        df = self.con.execute("SELECT symbol, qty FROM positions WHERE strategy_id=?", [sid]).df()
        return dict(zip(df["symbol"], df["qty"]))

    def _step(self, s, panel: Panel, market: Market, ev: Evaluator, t: int, date, kill_switch: bool) -> str:
        sid = s.strategy_id
        cash = float(s.cash)
        pos = self._positions(sid)
        col = {sym: j for j, sym in enumerate(panel.symbols)}
        opn, cls = panel.open[t], panel.close[t]
        fill_px = opn if self.cfg["execution"]["fill"] == "next_open" else cls
        costs = ev.s.costs
        # 1. fill yesterday's orders
        pend = self.con.execute("SELECT * FROM orders WHERE strategy_id=? AND status='pending'", [sid]).df()
        if not pend.empty:
            eq_at_fill = cash + sum(q * _px(fill_px, cls, col, sym, panel, t) for sym, q in pos.items())
            for o in pend.itertuples():
                j = col.get(o.symbol)
                if j is None or not panel.tradable[t, j]:
                    continue                       # no price today: try again tomorrow
                px = fill_px[j]
                target = o.target_weight * eq_at_fill / px
                qty = target - pos.get(o.symbol, 0.0)
                value = abs(qty * px)
                imp = costs.impact_coef * market.vol[t, j] * np.sqrt(value / market.adv[t, j]) if value else 0.0
                fee = value * ((costs.commission_bps + costs.half_spread_bps) / 1e4 + imp)
                cash -= qty * px + fee
                pos[o.symbol] = target
                self.con.execute("UPDATE orders SET status='filled', fill_date=?, fill_price=?, qty=?, cost=? "
                                 "WHERE strategy_id=? AND signal_date=? AND symbol=? AND status='pending'",
                                 [date.date(), float(px), float(qty), float(fee), sid, o.signal_date, o.symbol])
        # 2. mark to market
        mv = sum(q * _px(cls, cls, col, sym, panel, t) for sym, q in pos.items())
        equity = cash + mv
        prev = self.con.execute("SELECT equity FROM equity WHERE strategy_id=? ORDER BY date DESC LIMIT 1",
                                [sid]).fetchone()
        prev_eq = prev[0] if prev else self.pc.starting_cash
        r = equity / prev_eq - 1
        gross = sum(abs(q * _px(cls, cls, col, sym, panel, t)) for sym, q in pos.items()) / equity if equity > 0 else 0
        self.con.execute("INSERT OR REPLACE INTO equity VALUES (?,?,?,?,?,?)",
                         [sid, date.date(), equity, cash, gross, r])
        peak = max(float(s.peak_equity), equity)
        status, block_orders = "active", kill_switch
        # 3. risk checks
        if kill_switch:
            self.event(sid, "warn", "kill_switch", f"{self.pc.kill_switch_file} exists: no new orders")
        if r < -self.pc.max_daily_loss:
            self.event(sid, "warn", "daily_loss", f"lost {r:.2%} today (limit {self.pc.max_daily_loss:.0%}); "
                       "no new orders today")
            block_orders = True
        dd = 1 - equity / peak
        if dd >= self.pc.kill_drawdown:
            status = "killed"
            # Flatten now at today's close (worse than waiting for a better fill, on purpose).
            for sym, q in list(pos.items()):
                j = col[sym]
                if not panel.tradable[t, j]:
                    continue
                value = abs(q * cls[j])
                imp = costs.impact_coef * market.vol[t, j] * np.sqrt(value / market.adv[t, j]) if value else 0.0
                cash += q * cls[j] - value * ((costs.commission_bps + costs.half_spread_bps) / 1e4 + imp)
                del pos[sym]
            equity = cash + sum(q * _px(cls, cls, col, sym, panel, t) for sym, q in pos.items())
            self.con.execute("UPDATE equity SET equity=?, cash=? WHERE strategy_id=? AND date=?",
                             [equity, cash, sid, date.date()])
            self.event(sid, "critical", "kill_drawdown",
                       f"drawdown {dd:.1%} from paper peak: strategy stopped and positions closed")
        if s.deterioration_threshold is not None and not pd.isna(s.deterioration_threshold):
            w = self.pc.deterioration_window
            hist = self.con.execute("SELECT daily_return FROM equity WHERE strategy_id=? ORDER BY date DESC LIMIT ?",
                                    [sid, w]).df()["daily_return"].to_numpy()
            if len(hist) >= w:
                roll = float(np.prod(1 + hist) - 1)
                if roll < s.deterioration_threshold:
                    self.event(sid, "warn", "deterioration",
                               f"{w}-day paper return {roll:.2%} is below the backtest's 5th percentile "
                               f"({s.deterioration_threshold:.2%})")
        # 4. tomorrow's targets
        self.con.execute("UPDATE orders SET status='cancelled' WHERE strategy_id=? AND status='pending'", [sid])
        if status == "killed" or block_orders:
            targets = {}
        else:
            W = ev.weights(json.loads(s.spec_json))[t]
            targets = {sym: float(W[j]) for sym, j in col.items() if abs(W[j]) > 1e-9 or pos.get(sym)}
        rows = [[sid, date.date(), sym, w, "pending", None, None, None, None] for sym, w in targets.items()]
        if rows:
            self.con.executemany("INSERT INTO orders VALUES (?,?,?,?,?,?,?,?,?)", rows)
        self.con.execute("DELETE FROM positions WHERE strategy_id=?", [sid])
        live = [[sid, sym, q] for sym, q in pos.items() if abs(q) > 1e-12]
        if live:
            self.con.executemany("INSERT INTO positions VALUES (?,?,?)", live)
        self.con.execute("UPDATE strategies SET cash=?, peak_equity=?, last_date=?, status=? WHERE strategy_id=?",
                         [cash, peak, date.date(), "killed" if status == "killed" else "active", sid])
        return f"equity {equity:,.2f} ({r:+.2%})"

    # ---------------------------------------------------------------- reporting
    def status(self) -> pd.DataFrame:
        return self.con.execute("""
            SELECT s.strategy_id, s.description, s.status, s.started_at, s.last_date,
                   e.equity, e.gross_exposure, (SELECT COUNT(*) FROM equity x WHERE x.strategy_id=s.strategy_id) AS days
            FROM strategies s LEFT JOIN equity e ON e.strategy_id=s.strategy_id AND e.date=s.last_date""").df()

    def equity_curve(self, sid: str) -> pd.DataFrame:
        return self.con.execute("SELECT * FROM equity WHERE strategy_id=? ORDER BY date", [sid]).df()

    def events(self, limit: int = 50) -> pd.DataFrame:
        return self.con.execute("SELECT * FROM events ORDER BY ts DESC LIMIT ?", [limit]).df()


def _px(px_row, close_row, col, sym, panel, t) -> float:
    """Price for valuation; falls back to the last known close when today's bar is missing."""
    j = col[sym]
    v = px_row[j]
    if np.isfinite(v):
        return float(v)
    past = panel.close[: t + 1, j]
    past = past[np.isfinite(past)]
    return float(past[-1]) if len(past) else 0.0


def _alert(text: str) -> None:
    url = os.environ.get("QRP_ALERT_WEBHOOK")
    if not url:
        return
    try:   # Slack uses "text", Discord uses "content"
        requests.post(url, json={"text": text, "content": text}, timeout=10)
    except requests.RequestException as e:
        log.warning("alert webhook failed: %s", e)


def replay(cfg, panel: Panel, spec: dict, start: int, end: int, db_path: str) -> pd.DataFrame:
    """Run the paper trader day by day over panel rows [start, end) as if live (for verification)."""
    pt = PaperTrader(cfg, db_path)
    pt.add_strategy("replay", spec, "replay", "replay", None)
    for t in range(start, end):
        view = panel.head(t + 1)
        pt.run_day(view, today=pd.Timestamp(view.dates[-1]))
    curve = pt.equity_curve("replay")
    pt.close()
    return curve
