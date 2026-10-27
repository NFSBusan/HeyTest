"""Search strategies for spending a family's trial budget: grid, random, or Bayesian (Optuna TPE).

Whatever the sampler, every evaluated configuration is a counted trial in the registry, so a
smarter search finds good-looking configs faster but does not dodge the multiple-testing
penalty. The objective rewards consistency across time, not the single best backtest:

    objective = median(Sharpe of 5 equal time blocks) - 0.5 * std(those Sharpes)

computed on the dev period only.
"""
from __future__ import annotations

import logging
import tempfile

import numpy as np
import optuna

from .backtest.metrics import sharpe
from .strategies import BLOCKS, FAMILIES, canonical, enumerate_family, sample_family, spec_hash

log = logging.getLogger(__name__)


def robust_objective(r: np.ndarray, n_blocks: int = 5) -> float:
    blocks = np.array_split(np.asarray(r, dtype=float), n_blocks)
    s = np.array([sharpe(b) for b in blocks])
    return float(np.median(s) - 0.5 * s.std())


def _suggest(trial: optuna.Trial, family: str, direction: str) -> dict | None:
    f = FAMILIES[family]
    spec = {"family": family, "direction": direction}
    for slot, names in (("signal", f.signals), ("filter", f.filters), ("exit", f.exits), ("sizing", f.sizings)):
        name = trial.suggest_categorical(slot, list(names))
        b = BLOCKS[name]
        params = {k: trial.suggest_categorical(f"{name}.{k}", list(v)) for k, v in b.grid.items()}
        if b.constraint and not b.constraint(params):
            return None
        spec[slot] = {"name": name, **params}
    return canonical(spec)


def optimize_family(ctx, family: str, n_trials: int, sampler: str = "tpe") -> dict:
    """Spend up to n_trials of the family's remaining budget. Returns a summary."""
    from .pipeline import search          # local import avoids a circular import
    budget_left = ctx.cfg["search"]["max_trials_per_family"] - ctx.reg.n_trials(ctx.dev_fp, family)
    n = min(n_trials, max(budget_left, 0))
    if n <= 0:
        raise RuntimeError(f"{family}: trial budget exhausted ({ctx.cfg['search']['max_trials_per_family']}). "
                           "Raising the budget is allowed, but the multiple-testing penalty grows with it.")
    direction = "long_short" if ctx.cfg["execution"]["allow_short"] else "long"
    sc = ctx.score
    seed = ctx.cfg["project"]["seed"]
    if sampler in ("random", "grid"):
        if sampler == "random":
            specs = sample_family(family, n, seed, ctx.reg.tested_hashes(ctx.dev_fp, ctx.cost_fp, family), direction)
        else:
            tested = ctx.reg.tested_hashes(ctx.dev_fp, ctx.cost_fp, family)
            specs = [s for s in enumerate_family(family, direction) if spec_hash(s) not in tested][:n]
        search(ctx, family, specs, sampler)
        return _summary(ctx, family, [spec_hash(s) for s in specs], sampler)

    # Configs already in the registry are scored from stored returns, never re-run.
    prior = ctx.reg.trials(ctx.dev_fp, ctx.cost_fp, family)
    known: dict[str, float] = {}
    if not prior.empty:
        hs, M = ctx.reg.trial_returns(prior)
        known = {h: robust_objective(M[sc, i]) for i, h in enumerate(hs)}
    evaluated: list[str] = []
    buf_rows, buf_rets = [], {}

    def flush():
        ctx.reg.save_trials(ctx.dev_fp, ctx.cost_fp, ctx.study_id, family, "optuna", True, buf_rows, buf_rets)
        buf_rows.clear()
        buf_rets.clear()

    def objective(trial: optuna.Trial) -> float:
        spec = _suggest(trial, family, direction)
        if spec is None:
            raise optuna.TrialPruned("constraint")
        h = spec_hash(spec)
        if h in known:
            return known[h]
        if len(evaluated) >= n:
            raise optuna.TrialPruned("budget")
        h, spec, m, r = ctx.evaluator.evaluate(spec)
        buf_rows.append({"spec_hash": h, "spec": spec, "metrics": m})
        buf_rets[h] = r
        evaluated.append(h)
        if len(buf_rows) >= 50:
            flush()
        known[h] = robust_objective(r[sc].astype(np.float64))
        return known[h]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    # Optuna proposes duplicates and constraint violations; cap total proposals so the loop ends.
    try:
        study.optimize(objective, n_trials=n * 5,
                       callbacks=[lambda st, t: st.stop() if len(evaluated) >= n else None])
    finally:
        flush()
    return _summary(ctx, family, evaluated, "tpe")


def _summary(ctx, family: str, hashes: list[str], sampler: str) -> dict:
    t = ctx.reg.trials(ctx.dev_fp, ctx.cost_fp, family)
    t = t[t["spec_hash"].isin(hashes)]
    if t.empty:
        return {"family": family, "sampler": sampler, "evaluated": 0}
    hs, M = ctx.reg.trial_returns(t)
    objs = [robust_objective(M[ctx.score, i]) for i in range(len(hs))]
    best = int(np.argmax(objs))
    return {"family": family, "sampler": sampler, "evaluated": len(hs), "best_hash": hs[best],
            "best_objective": objs[best], "objective_trace": np.maximum.accumulate(objs).tolist(),
            "n_trials_family": ctx.reg.n_trials(ctx.dev_fp, family)}


def compare_samplers(cfg, family: str, budget: int, samplers=("grid", "random", "tpe")) -> dict:
    """Same budget per sampler, each in its own throwaway registry. Synthetic data only.

    On real data this comparison would itself be untracked data snooping, so it is refused.
    With a planted edge, the better sampler finds a high objective with fewer trials.
    """
    from .pipeline import prepare
    from .registry import Registry
    if cfg["data"]["provider"] != "synthetic":
        raise RuntimeError("compare-samplers runs on synthetic data only (it would snoop real data untracked)")
    out = {}
    for s in samplers:
        with tempfile.TemporaryDirectory() as d:
            c = cfg.with_overrides({"search.max_trials_per_family": budget})
            reg = Registry(f"{d}/r.duckdb", f"{d}/results")
            ctx = prepare(c, reg)
            ctx.study_id = "compare"
            res = optimize_family(ctx, family, budget, s)
            reg.close()
        out[s] = {k: res[k] for k in ("evaluated", "best_objective", "objective_trace") if k in res}
    return out
