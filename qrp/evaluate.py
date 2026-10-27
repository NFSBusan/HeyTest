"""Evaluate strategy specs on a panel, serially or across CPU cores."""
from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np

from .backtest import CostModel, Market, buy_and_hold, run_backtest, summarize
from .data.panel import Panel
from .indicators import IndicatorCache
from .strategies import spec_hash, target_weights, validate_spec
from .validation.stats import alpha_returns

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvalSettings:
    fill: str
    allow_short: bool
    max_weight: float
    max_gross: float
    costs: CostModel
    score_start: int          # first scored row (end of warm-up)
    bench_col: int

    @classmethod
    def from_config(cls, cfg, panel: Panel, score_start: int) -> "EvalSettings":
        e = cfg["execution"]
        return cls(e["fill"], e["allow_short"], e["max_weight"], e["max_gross_leverage"],
                   CostModel.from_config(cfg["costs"]), score_start, panel.col(cfg["data"]["benchmark"]))


class Evaluator:
    """Holds the indicator cache and market arrays for one panel; reuse it for many specs."""

    def __init__(self, panel: Panel, s: EvalSettings):
        panel.meta["benchmark_col"] = s.bench_col
        self.p, self.s = panel, s
        self.cache = IndicatorCache(panel)
        self.market = Market.from_panel(panel)
        self.bench = buy_and_hold(self.market, s.bench_col, s.costs, s.fill, s.score_start).returns

    def weights(self, spec: dict) -> np.ndarray:
        return target_weights(spec, self.cache, self.s.max_weight, self.s.max_gross, self.s.allow_short)

    def run(self, spec: dict, cost_mult: float = 1.0, capital: float | None = None,
            active_from: int | None = None, score_start: int | None = None):
        costs = self.s.costs.scaled(cost_mult) if cost_mult != 1.0 else self.s.costs
        if capital:
            costs = costs.with_capital(capital)
        a = self.s.score_start if active_from is None else active_from
        res = run_backtest(self.weights(spec), self.market, costs, self.s.fill, a)
        start = a if score_start is None else score_start
        m = summarize(res.returns, res, start, dates=self.p.dates)
        alpha, beta = alpha_returns(res.returns[start:, None], self.bench[start:])
        m["beta"] = float(beta[0])
        sd = alpha[:, 0].std(ddof=1)
        m["alpha_sharpe"] = float(alpha[:, 0].mean() / sd * np.sqrt(252)) if sd > 1e-12 else 0.0
        m["alpha_ann"] = float(alpha[:, 0].mean() * 252)
        return m, res

    def evaluate(self, spec: dict) -> tuple[str, dict, dict, np.ndarray]:
        spec = validate_spec(spec)
        m, res = self.run(spec)
        return spec_hash(spec), spec, m, res.returns.astype(np.float32)


_WORKER: Evaluator | None = None


def _init(panel: Panel, s: EvalSettings):
    global _WORKER
    os.environ.setdefault("NUMBA_NUM_THREADS", "1")
    _WORKER = Evaluator(panel, s)


def _work(specs: list[dict]):
    out = []
    for spec in specs:
        try:
            out.append(_WORKER.evaluate(spec))
        except Exception as e:  # one bad spec must not kill a batch of thousands
            out.append(("error", spec, {"error": f"{type(e).__name__}: {e}"}, None))
    return out


def evaluate_many(panel: Panel, s: EvalSettings, specs: list[dict], n_jobs: int = 0,
                  batch_size: int = 100, evaluator: Evaluator | None = None):
    """Yield (hash, spec, metrics, returns) per spec; errors come back with hash == 'error'."""
    if not specs:
        return
    n_jobs = n_jobs or os.cpu_count() or 1
    batches = [specs[i:i + batch_size] for i in range(0, len(specs), batch_size)]
    if n_jobs == 1 or len(specs) < 2 * batch_size:
        ev = evaluator or Evaluator(panel, s)
        global _WORKER
        prev, _WORKER = _WORKER, ev
        try:
            for b in batches:
                yield from _work(b)
        finally:
            _WORKER = prev
        return
    with ProcessPoolExecutor(max_workers=n_jobs, initializer=_init, initargs=(panel, s)) as ex:
        for res in ex.map(_work, batches):
            yield from res
