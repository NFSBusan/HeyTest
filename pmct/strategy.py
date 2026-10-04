"""Counter-trade the worst wallets, sized with fractional Kelly.

Pipeline (all point-in-time):
1. Score every trade's wallet using only markets resolved strictly before that trade.
2. Flag trades from wallets whose score is below a threshold (with minimum history).
3. We act `latency_s` after their fill, on the opposite side, at the mid at that time
   (estimated from prints) plus our half-spread, plus size-dependent impact, plus fees.
4. Our win probability for Kelly is  p = q + edge_hat, where edge_hat is the shrunk
   average realized edge of *earlier* signals whose markets already resolved.
5. Kelly stake = kelly * (p - q) / (1 - q) of capital, subject to caps.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, asdict, replace

import numpy as np
import pandas as pd

from .schema import DAY
from .scoring import scores_at

EDGES = np.array([0.0, 0.15, 0.35, 0.65, 0.85, 1.0])   # price buckets for calibration


@dataclass(frozen=True)
class StratConfig:
    basis: str = "res"            # what "bad" means: res | res_mid | mo1h | mo6h | mo1d (see scoring.py)
    score: str = "tstat"          # pnl | roi | edge | tstat
    threshold: float = -2.0       # flag wallet if score <= threshold
    min_res: int = 30             # min resolved trades in wallet history
    min_mkts: int = 5             # min distinct resolved markets
    min_usd: float = 0.0          # ignore their trades smaller than this
    px_lo: float = 0.05           # our entry price band (side terms)
    px_hi: float = 0.95
    latency_s: int = 60
    assumed_half_spread: float = 0.01   # to back out mid from a print
    exec_cost: float = 0.01       # what we pay over mid
    impact: float = 0.02          # extra price per unit of stake/(stake+their_usd)
    fee_rate: float = 0.0         # Polymarket-style taker fee: rate * min(q,1-q) per share
    kelly: float = 0.25
    max_bet_frac: float = 0.05
    max_mkt_frac: float = 0.10
    max_stake_vs_their: float = 1.0
    calib_n0: float = 300.0       # shrinkage pseudo-count toward zero edge
    calib_min: int = 200          # min resolved prior signals before betting
    calib_buckets: bool = True
    min_edge: float = 0.005       # need p - q_fill above this

    def key(self) -> str:
        return ",".join(f"{k}={v}" for k, v in asdict(self).items())


# ---------------------------------------------------------------- price lookup
class MidBook:
    """Mid-price estimate per market from prints: mid = yes_price - dir * assumed_half_spread."""

    def __init__(self, trades: pd.DataFrame, assumed_half_spread: float):
        self.assumed_half_spread = assumed_half_spread
        t = trades.sort_values(["market_id", "ts"], kind="mergesort")
        self.key = t["market_id"].to_numpy(np.int64) * (1 << 34) + t["ts"].to_numpy(np.int64)
        self.mid = np.clip(t["yes_price"].to_numpy() - t["dir"].to_numpy() * assumed_half_spread, 0.001, 0.999)
        self.mkt = t["market_id"].to_numpy(np.int64)

    def at(self, market_id, ts, inclusive=True) -> np.ndarray:
        """Last mid at or before ts (or strictly before if not inclusive). NaN if none."""
        market_id = np.asarray(market_id, np.int64)
        k = market_id * (1 << 34) + np.asarray(ts, np.int64)
        i = np.searchsorted(self.key, k, side="right" if inclusive else "left") - 1
        ok = (i >= 0) & (self.mkt[np.clip(i, 0, None)] == market_id)
        out = np.full(len(k), np.nan)
        out[ok] = self.mid[i[ok]]
        return out


# ---------------------------------------------------------------- signals
def make_signals(cfg: StratConfig, markets: pd.DataFrame, trades: pd.DataFrame,
                 hist: pd.DataFrame, book: MidBook, scores: pd.DataFrame | None = None) -> pd.DataFrame:
    """All counter-trade signals for a config, with walk-forward edge estimates."""
    sc = scores if scores is not None else scores_at(hist, trades)
    flag = ((sc["n_res"].to_numpy() >= cfg.min_res) & (sc["n_mkts"].to_numpy() >= cfg.min_mkts)
            & (sc[cfg.score].to_numpy() <= cfg.threshold) & (trades["usd"].to_numpy() >= cfg.min_usd))
    s = trades.loc[flag, ["ts", "market_id", "wallet", "dir", "usd"]].copy()
    s["wscore"] = sc.loc[flag, cfg.score].to_numpy()
    s["exec_ts"] = s["ts"] + cfg.latency_s
    mk = markets.set_index("market_id")
    s["end_ts"] = mk.loc[s["market_id"], "end_ts"].to_numpy()
    s["resolved_ts"] = mk.loc[s["market_id"], "resolved_ts"].to_numpy()
    s["outcome"] = mk.loc[s["market_id"], "outcome"].to_numpy()
    s = s[s["exec_ts"] < s["end_ts"]]
    s["our_dir"] = -s["dir"]
    mid = book.at(s["market_id"].to_numpy(), s["exec_ts"].to_numpy())
    s["side_mid"] = np.where(s["our_dir"] == 1, mid, 1 - mid)
    s["q"] = s["side_mid"] + cfg.exec_cost
    s = s[(s["q"] >= cfg.px_lo) & (s["q"] <= cfg.px_hi)].copy()
    s["win"] = np.where(s["our_dir"] == 1, s["outcome"], 1 - s["outcome"])   # NaN if unresolved
    s["bucket"] = np.clip(np.searchsorted(EDGES, s["q"].to_numpy(), side="right") - 1, 0, len(EDGES) - 2)
    s = s.sort_values(["exec_ts", "market_id"], kind="mergesort").reset_index(drop=True)
    return add_edge_estimates(s, cfg)


def _asof_cum(s: pd.DataFrame, done: pd.DataFrame, by: str | None) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative (n, sum r) over `done` rows with avail_ts < s.exec_ts (strict), optionally by group."""
    cols = ["avail_ts", "r"] + ([by] if by else [])
    d = done[cols].sort_values("avail_ts", kind="mergesort").copy()
    if by:
        d["cn"] = d.groupby(by).cumcount() + 1
        d["cr"] = d.groupby(by)["r"].cumsum()
    else:
        d["cn"] = np.arange(1, len(d) + 1)
        d["cr"] = d["r"].cumsum()
    q = s[["exec_ts"] + ([by] if by else [])].copy()
    q["_o"] = np.arange(len(q))
    m = pd.merge_asof(q.sort_values("exec_ts", kind="mergesort"), d[["avail_ts", "cn", "cr"] + ([by] if by else [])],
                      left_on="exec_ts", right_on="avail_ts", by=by, allow_exact_matches=False)
    m = m.sort_values("_o")
    return m["cn"].fillna(0).to_numpy(), m["cr"].fillna(0).to_numpy()


def add_edge_estimates(s: pd.DataFrame, cfg: StratConfig) -> pd.DataFrame:
    """edge_hat for each signal from earlier signals of the same config that resolved before it.
    Hierarchical shrinkage: bucket -> global -> 0."""
    done = s.dropna(subset=["win"]).copy()
    done["r"] = done["win"] - done["q"]
    done["avail_ts"] = done["resolved_ts"]
    n_g, r_g = _asof_cum(s, done, None) if len(done) else (np.zeros(len(s)), np.zeros(len(s)))
    g_edge = r_g / (n_g + cfg.calib_n0)
    if cfg.calib_buckets and len(done):
        n_b, r_b = _asof_cum(s, done, "bucket")
        k = cfg.calib_n0 / 3
        edge = (r_b + k * g_edge) / (n_b + k)
    else:
        edge = g_edge
    s["n_calib"] = n_g
    s["edge_hat"] = edge
    return s


# ---------------------------------------------------------------- backtest
@dataclass
class BTResult:
    cfg: StratConfig
    positions: pd.DataFrame
    equity: pd.Series         # daily, mark-to-market
    metrics: dict


def backtest(cfg: StratConfig, sig: pd.DataFrame, book: MidBook, start_ts: int, end_ts: int,
             capital: float = 10_000.0) -> BTResult:
    """Bets on signals with start_ts <= exec_ts < end_ts. Positions still open at end_ts are
    marked to the last mid before end_ts minus exec_cost (so no outcome after end_ts is used)."""
    s = sig[(sig["exec_ts"] >= start_ts) & (sig["exec_ts"] < end_ts)]
    cash = capital
    open_cost = 0.0
    heap: list = []
    mkt_exp: dict[int, float] = {}
    rec = []
    cols = [s[c].to_numpy() for c in ["exec_ts", "market_id", "our_dir", "q", "edge_hat", "n_calib", "usd",
                                       "resolved_ts", "win"]]
    for ets, mid_, od, q, eh, nc, their, rts, win in zip(*cols):
        while heap and heap[0][0] < ets:
            _, i = heapq.heappop(heap)
            r = rec[i]
            if r["settled"]:
                continue
            payoff = r["shares"] * r["win"]
            cash += payoff
            open_cost -= r["stake"]
            mkt_exp[r["market_id"]] -= r["stake"]
            r["settled"] = True
            r["payoff"] = payoff
        if nc < cfg.calib_min:
            continue
        p = min(max(q + eh, 0.001), 0.999)
        if p - q <= cfg.min_edge:
            continue
        cap = cash + open_cost
        f = cfg.kelly * (p - q) / (1 - q)
        stake = min(f * cap, cfg.max_bet_frac * cap, cfg.max_stake_vs_their * their,
                    cfg.max_mkt_frac * cap - mkt_exp.get(mid_, 0.0), cash * 0.999)
        if stake < 1.0:
            continue
        qf = q + cfg.impact * stake / (stake + their)
        if qf >= 0.999 or p - qf <= cfg.min_edge:
            continue
        shares = stake / qf
        fee = cfg.fee_rate * min(qf, 1 - qf) * shares
        if stake + fee > cash:
            continue
        cash -= stake + fee
        open_cost += stake
        mkt_exp[mid_] = mkt_exp.get(mid_, 0.0) + stake
        i = len(rec)
        rec.append(dict(exec_ts=ets, market_id=mid_, our_dir=od, q=qf, p=p, stake=stake, fee=fee,
                        shares=shares, resolved_ts=rts, win=win, settled=False, payoff=np.nan))
        if rts < end_ts and not np.isnan(win):
            heapq.heappush(heap, (rts, i))
    # settle what resolves before end
    while heap:
        _, i = heapq.heappop(heap)
        r = rec[i]
        r["settled"] = True
        r["payoff"] = r["shares"] * r["win"]
    pos = pd.DataFrame(rec, columns=["exec_ts", "market_id", "our_dir", "q", "p", "stake", "fee", "shares",
                                     "resolved_ts", "win", "settled", "payoff"])
    if len(pos):
        op = ~pos["settled"]
        pos.loc[op, "win"] = np.nan   # outcome not knowable at end_ts; don't carry it in the output
        if op.any():
            mid = book.at(pos.loc[op, "market_id"].to_numpy(), np.full(op.sum(), end_ts), inclusive=False)
            side = np.where(pos.loc[op, "our_dir"] == 1, mid, 1 - mid)
            pos.loc[op, "payoff"] = pos.loc[op, "shares"] * np.clip(side - cfg.exec_cost, 0, 1)
        pos["pnl"] = pos["payoff"] - pos["stake"] - pos["fee"]
    else:
        pos["pnl"] = []
    eq = equity_curve(pos, book, start_ts, end_ts, capital, cfg.exec_cost)
    return BTResult(cfg, pos, eq, metrics(pos, eq, capital))


def equity_curve(pos: pd.DataFrame, book: MidBook, start_ts: int, end_ts: int, capital: float,
                 exec_cost: float) -> pd.Series:
    days = np.arange(start_ts + DAY, end_ts + 1, DAY)
    if len(days) == 0 or days[-1] != end_ts:
        days = np.append(days, end_ts)
    if len(pos) == 0:
        return pd.Series(capital, index=days)
    e = pos["exec_ts"].to_numpy()
    r = np.where(pos["settled"].to_numpy(), pos["resolved_ts"].to_numpy(), np.iinfo(np.int64).max)
    pnl_final = pos["pnl"].to_numpy()
    cost = (pos["stake"] + pos["fee"]).to_numpy()
    out = np.empty(len(days))
    mk, od, sh = pos["market_id"].to_numpy(), pos["our_dir"].to_numpy(), pos["shares"].to_numpy()
    for j, d in enumerate(days):
        done = (e < d) & (r < d)
        opn = (e < d) & ~(r < d)
        v = pnl_final[done].sum()
        if opn.any():
            if d == end_ts:
                v += pnl_final[opn].sum()
            else:
                m = book.at(mk[opn], np.full(opn.sum(), d), inclusive=False)
                side = np.where(od[opn] == 1, m, 1 - m)
                v += (sh[opn] * np.nan_to_num(side, nan=0.5) - cost[opn]).sum()
        out[j] = capital + v
    return pd.Series(out, index=days)


def metrics(pos: pd.DataFrame, eq: pd.Series, capital: float) -> dict:
    v = np.maximum(eq.to_numpy(), 1e-9)
    v = np.concatenate([[capital], v])
    lr = np.diff(np.log(v))
    years = (len(v) - 1) / 365.0
    sd = lr.std(ddof=1) if len(lr) > 2 else 0.0
    peak = np.maximum.accumulate(v)
    out = dict(
        final=float(v[-1]),
        log_growth=float(np.log(v[-1] / capital)),
        cagr=float((v[-1] / capital) ** (1 / max(years, 1e-9)) - 1) if v[-1] > 0 else -1.0,
        sharpe=float(lr.mean() / sd * np.sqrt(365)) if sd > 0 else 0.0,
        max_dd=float((1 - v / peak).max()),
        n_bets=int(len(pos)),
        hit=float((pos["pnl"] > 0).mean()) if len(pos) else np.nan,
        avg_edge=float((pos["payoff"] / pos["shares"] - pos["q"]).mean()) if len(pos) else np.nan,
        turnover=float(pos["stake"].sum() / capital) if len(pos) else 0.0,
        n_days=int(len(lr)),
        skew=float(pd.Series(lr).skew()) if len(lr) > 3 else 0.0,
        kurt=float(pd.Series(lr).kurt() + 3) if len(lr) > 3 else 3.0,
    )
    return out


def signal_quality(sig: pd.DataFrame, start_ts: int, end_ts: int) -> dict:
    """Sizing-free check: realized edge per signal (net of exec_cost, before impact),
    with a t-stat clustered by market. Only signals resolved before end_ts count."""
    s = sig[(sig["exec_ts"] >= start_ts) & (sig["exec_ts"] < end_ts) & (sig["resolved_ts"] < end_ts)].dropna(subset=["win"])
    if len(s) < 10:
        return dict(n_sig=len(s), n_mkt=0, edge=np.nan, t=np.nan)
    r = s["win"] - s["q"]
    by_m = r.groupby(s["market_id"]).mean()
    t = by_m.mean() / (by_m.std(ddof=1) / np.sqrt(len(by_m))) if len(by_m) > 2 else np.nan
    return dict(n_sig=int(len(s)), n_mkt=int(len(by_m)), edge=float(r.mean()), t=float(t))


def with_(cfg: StratConfig, **kw) -> StratConfig:
    return replace(cfg, **kw)


SIZING_FIELDS = {"kelly", "max_bet_frac", "max_mkt_frac", "max_stake_vs_their", "min_edge", "impact", "fee_rate"}


class Context:
    """Caches the expensive point-in-time pieces (wallet scores per basis, signals per config)."""

    def __init__(self, markets: pd.DataFrame, trades: pd.DataFrame, assumed_half_spread: float = 0.01):
        from .scoring import wallet_history
        self._wh = wallet_history
        self.markets, self.trades = markets, trades
        self.book = MidBook(trades, assumed_half_spread)
        self._scores: dict = {}
        self._sigs: dict = {}

    def scores(self, basis: str) -> pd.DataFrame:
        if basis not in self._scores:
            hist = self._wh(self.markets, self.trades, basis, self.book)
            self._scores[basis] = scores_at(hist, self.trades)
        return self._scores[basis]

    def signals(self, cfg: StratConfig) -> pd.DataFrame:
        k = tuple((f, v) for f, v in asdict(cfg).items() if f not in SIZING_FIELDS)
        if k not in self._sigs:
            self._sigs[k] = make_signals(cfg, self.markets, self.trades, None, self.book, self.scores(cfg.basis))
        return self._sigs[k]
