"""The research study: search -> family statistics -> candidates -> robustness -> gates.

Only the dev panel exists inside this module. The holdout is evaluated in `run_holdout`,
through the vault, after a study has produced a frozen list of candidates.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass

import numpy as np

from .backtest import buy_and_hold
from .backtest.metrics import sharpe, summarize
from .config import Config
from .data import load_panel
from .data.panel import Panel, Split, make_split, walk_forward_folds
from .evaluate import EvalSettings, Evaluator, evaluate_many
from .registry import Registry, open_registry
from .strategies import FAMILIES, describe, sample_family, spec_hash
from .validation.holdout import HoldoutVault
from .validation.robustness import robustness_suite
from .validation.stats import alpha_returns, deflated_sharpe, effective_trials, pbo_cscv, sharpe_daily, spa_test

log = logging.getLogger(__name__)


@dataclass
class StudyContext:
    cfg: Config
    reg: Registry
    dev: Panel
    split: Split
    dev_fp: str
    holdout_fp: str
    cost_fp: str
    settings: EvalSettings
    evaluator: Evaluator
    split_info: dict
    study_id: str = ""

    @property
    def score(self) -> slice:
        return slice(self.split.warmup_end, self.split.dev_end)


def cost_fingerprint(cfg) -> str:
    key = {"costs": cfg["costs"], "execution": cfg["execution"], "benchmark": cfg["data"]["benchmark"],
           "warmup": cfg["splits"]["warmup_days"], "engine": 2}
    return hashlib.sha1(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:12]


def prepare(cfg: Config, reg: Registry | None = None) -> StudyContext:
    """Load data, split it, and keep only the dev rows in memory."""
    full = load_panel(cfg)
    s = cfg["splits"]
    split = make_split(full.T, s["holdout_fraction"], s["embargo_days"], s["warmup_days"])
    holdout_fp = full.fingerprint(split.holdout_start, split.T)
    split_info = split.describe(full.dates)
    dev = full.head(split.dev_end)
    del full
    reg = reg or open_registry(cfg)
    settings = EvalSettings.from_config(cfg, dev, split.warmup_end)
    return StudyContext(cfg, reg, dev, split, dev.fingerprint(), holdout_fp, cost_fingerprint(cfg), settings,
                        Evaluator(dev, settings), split_info)


# ------------------------------------------------------------------ search

def search(ctx: StudyContext, family: str, specs: list[dict], source: str, counted: bool = True) -> int:
    """Evaluate specs not yet in the registry and store them. Returns how many were new."""
    tested = ctx.reg.tested_hashes(ctx.dev_fp, ctx.cost_fp)
    todo, seen = [], set()
    for s in specs:
        h = spec_hash(s)
        if h not in tested and h not in seen:
            todo.append(s)
            seen.add(h)
    if not todo:
        return 0
    sc = ctx.cfg["search"]
    rows, rets, n_new, n_err = [], {}, 0, 0
    for h, spec, m, r in evaluate_many(ctx.dev, ctx.settings, todo, sc.get("n_jobs", 0),
                                       sc.get("batch_size", 100), evaluator=ctx.evaluator):
        if h == "error":
            n_err += 1
            log.warning("spec failed: %s (%s)", describe(spec), m["error"])
            continue
        rows.append({"spec_hash": h, "spec": spec, "metrics": m})
        rets[h] = r
        if len(rows) >= 500:
            ctx.reg.save_trials(ctx.dev_fp, ctx.cost_fp, ctx.study_id, family, source, counted, rows, rets)
            n_new += len(rows)
            rows, rets = [], {}
    ctx.reg.save_trials(ctx.dev_fp, ctx.cost_fp, ctx.study_id, family, source, counted, rows, rets)
    n_new += len(rows)
    if n_err:
        log.warning("%d specs failed in family %s", n_err, family)
    return n_new


def run_generator(ctx: StudyContext, family: str) -> int:
    """Fill the family up to its trial budget with sampled, untested configs."""
    budget = ctx.cfg["search"]["max_trials_per_family"]
    done = ctx.reg.n_trials(ctx.dev_fp, family)
    if done >= budget:
        log.info("%s: trial budget used (%d/%d); nothing new to generate", family, done, budget)
        return 0
    direction = "long_short" if ctx.cfg["execution"]["allow_short"] else "long"
    specs = sample_family(family, budget - done, ctx.cfg["project"]["seed"],
                          ctx.reg.tested_hashes(ctx.dev_fp, ctx.cost_fp, family), direction)
    return search(ctx, family, specs, "generator")


# ------------------------------------------------------------------ family statistics

def walk_forward(ctx: StudyContext, M: np.ndarray, hashes: list[str]) -> dict:
    """Re-run the selection procedure 'pick the best in-sample Sharpe' on anchored folds."""
    sp = ctx.cfg["splits"]
    folds = walk_forward_folds(ctx.split.warmup_end, ctx.split.dev_end, sp["wf_folds"],
                               int(sp["wf_min_train_years"] * 252), sp["embargo_days"])
    oos, chosen, bench_oos = [], [], []
    for f in folds:
        tr = M[f.train[0]:f.train[1]]
        best = int(np.argmax(sharpe_daily(tr)))
        oos.append(M[f.test[0]:f.test[1], best])
        bench_oos.append(ctx.evaluator.bench[f.test[0]:f.test[1]])
        d = ctx.dev.dates
        chosen.append({"train": [int(f.train[0]), int(f.train[1])], "test": [int(f.test[0]), int(f.test[1])],
                       "test_dates": [str(d[f.test[0]]), str(d[f.test[1] - 1])],
                       "spec_hash": hashes[best], "is_sharpe": sharpe(tr[:, best]),
                       "oos_sharpe": sharpe(M[f.test[0]:f.test[1], best])})
    r = np.concatenate(oos)
    b = np.concatenate(bench_oos)
    a, beta = alpha_returns(r[:, None], b)
    m = summarize(r)
    return {"oos": m, "oos_alpha_sharpe": sharpe(a[:, 0]), "oos_beta": float(beta[0]),
            "benchmark_oos_sharpe": sharpe(b), "folds": chosen,
            "distinct_choices": len({c["spec_hash"] for c in chosen}), "_returns": r}


def dedup(M: np.ndarray, order: np.ndarray, max_corr: float, k: int) -> list[int]:
    """Greedy: walk configs best-first, keep one unless it correlates > max_corr with a kept one."""
    keep: list[int] = []
    sd = M.std(axis=0)
    Z = np.divide(M - M.mean(axis=0), sd, out=np.zeros_like(M), where=sd > 1e-12)
    n = len(M)
    for i in order:
        if sd[i] <= 1e-12:
            continue
        if all(abs(float(Z[:, i] @ Z[:, j]) / n) <= max_corr for j in keep):
            keep.append(int(i))
        if len(keep) >= k:
            break
    return keep


def analyze_family(ctx: StudyContext, family: str) -> dict:
    trials = ctx.reg.trials(ctx.dev_fp, ctx.cost_fp, family)
    if trials.empty:
        return {"family": family, "n_trials": 0}
    hashes, M_full = ctx.reg.trial_returns(trials)
    M = M_full[ctx.score]
    bench = ctx.evaluator.bench[ctx.score]
    v = ctx.cfg["validation"]
    A, _ = alpha_returns(M, bench)
    alpha_sr = sharpe_daily(A)
    out = {"family": family, "n_trials": len(hashes), "hypothesis": FAMILIES[family].hypothesis,
           "trial_alpha_sr_var_daily": float(np.var(alpha_sr, ddof=1)) if len(hashes) > 1 else 0.0,
           "trial_sharpe_median": float(np.median(trials["sharpe"])),
           "trial_sharpe_max": float(trials["sharpe"].max()),
           "trial_alpha_sharpe_median": float(np.median(trials["alpha_sharpe"]))}
    # Counted on alpha: shared market exposure makes long-only configs look like one bet.
    out["n_effective"] = effective_trials(A)
    out["pbo"] = pbo_cscv(M, v["cscv_splits"])
    out["spa"] = spa_test(A, v["bootstrap_reps"], v["block_len"], ctx.cfg["project"]["seed"])
    out["spa"]["best_hash"] = hashes[out["spa"]["best"]]
    wf = walk_forward(ctx, M_full, hashes)
    wf["returns"] = np.round(wf.pop("_returns"), 7).tolist()   # for the dashboard's OOS curve
    wf["start_row"] = wf["folds"][0]["test"][0]
    out["walk_forward"] = wf
    g = ctx.cfg["gates"]
    eligible = np.flatnonzero(trials["n_trades"].to_numpy() >= g["min_trades"])
    order = eligible[np.argsort(-trials["sharpe"].to_numpy()[eligible])]
    keep = dedup(M, order, ctx.cfg["search"]["dedup_corr"], ctx.cfg["search"]["candidates_per_family"])
    out["candidates"] = [hashes[i] for i in keep]
    out["n_clusters_note"] = "candidates are the best config of distinct return clusters (corr <= dedup_corr)"
    return out


# ------------------------------------------------------------------ gates

def apply_gates(cfg, fam: dict, cand: dict, rob: dict, bench_sharpe: float) -> tuple[list[str], dict]:
    g = cfg["gates"]
    b = rob["base"]
    stressed = rob["cost_stress"].get(str(cfg["costs"]["stress_multipliers"][0]), {}).get("sharpe", 0.0)
    checks = {
        "min_trades": (b.get("n_trades", 0), b.get("n_trades", 0) >= g["min_trades"]),
        "min_dev_sharpe": (b["sharpe"], b["sharpe"] >= g["min_dev_sharpe"]),
        "min_dsr": (cand["dsr"]["dsr"], cand["dsr"]["dsr"] >= g["min_dsr"]),
        "max_pbo": (fam["pbo"]["pbo"], fam["pbo"]["pbo"] <= g["max_pbo"]),
        "max_spa_pvalue": (fam["spa"]["spa_p"], fam["spa"]["spa_p"] <= g["max_spa_pvalue"]),
        "min_wf_oos_sharpe": (fam["walk_forward"]["oos"]["sharpe"],
                              fam["walk_forward"]["oos"]["sharpe"] >= g["min_wf_oos_sharpe"]),
        "max_drawdown": (b["max_drawdown"], b["max_drawdown"] <= g["max_drawdown"]),
        "min_neighbor_ratio": (rob["neighbour_ratio"], rob["neighbour_ratio"] >= g["min_neighbor_ratio"]),
        "min_stressed_sharpe": (stressed, stressed >= g["min_stressed_sharpe"]),
        "min_positive_year_fraction": (b.get("positive_year_fraction", 0),
                                       b.get("positive_year_fraction", 0) >= g["min_positive_year_fraction"]),
        "min_excess_sharpe_vs_benchmark": (b["sharpe"] - bench_sharpe,
                                           b["sharpe"] - bench_sharpe >= g["min_excess_sharpe_vs_benchmark"]),
        "max_capacity_cost_drag": (b.get("cost_drag", 0), b.get("cost_drag", 0) <= g["max_capacity_cost_drag"]),
    }
    failed = [k for k, (_, ok) in checks.items() if not ok]
    return failed, {k: {"value": float(v), "pass": bool(ok), "threshold": g[k]} for k, (v, ok) in checks.items()}


def evaluate_candidates(ctx: StudyContext, family: str, fam: dict, n_eff_total: float) -> list[dict]:
    trials = ctx.reg.trials(ctx.dev_fp, ctx.cost_fp, family).set_index("spec_hash")
    n_total = ctx.reg.n_trials(ctx.dev_fp)
    bench_sharpe = sharpe(ctx.evaluator.bench[ctx.score])
    nb_trials = ctx.reg.trials(ctx.dev_fp, ctx.cost_fp, family, counted_only=False).set_index("spec_hash")
    known = {h: json.loads(r.metrics_json) for h, r in nb_trials.iterrows()}
    out = []
    for h in fam.get("candidates", []):
        spec = json.loads(trials.loc[h, "spec_json"])
        rob = robustness_suite(ctx.evaluator, spec, ctx.cfg, ctx.split.warmup_end, ctx.split.dev_end, known)
        new = rob.pop("new_neighbours")
        if new:
            rows = [{"spec_hash": k, "spec": s, "metrics": m} for k, (s, m, _) in new.items()]
            ctx.reg.save_trials(ctx.dev_fp, ctx.cost_fp, ctx.study_id, family, "neighbour", False, rows,
                                {k: r for k, (_, _, r) in new.items()})
            known.update({k: m for k, (_, m, _) in new.items()})
        _, res = ctx.evaluator.run(spec)
        a, _ = alpha_returns(res.returns[ctx.score, None], ctx.evaluator.bench[ctx.score])
        cand = {"spec_hash": h, "spec": spec, "description": describe(spec),
                # Gate: effective independent trials across every family tested on this data.
                "dsr": deflated_sharpe(a[:, 0], n_eff_total),
                # Reference only: raw trial count with the empirical cross-trial variance (much stricter).
                "dsr_strict": deflated_sharpe(a[:, 0], n_total, fam["trial_alpha_sr_var_daily"])}
        failed, gates = apply_gates(ctx.cfg, fam, cand, rob, bench_sharpe)
        status = "passed" if not failed else "rejected"
        ctx.reg.save_candidate(ctx.study_id, h, family, status, failed, gates, {**rob, "dsr": cand["dsr"],
                               "dsr_strict": cand["dsr_strict"], "description": cand["description"],
                               "spec": spec})
        out.append({**cand, "status": status, "failed": failed})
        log.info("%s %s: %s %s", family, cand["description"], status, failed or "")
    return out


# ------------------------------------------------------------------ study

def run_study(cfg: Config, name: str | None = None, families: list[str] | None = None,
              generate: bool = True, sampler: str = "random") -> str:
    """Spend each family's remaining budget (random sample, grid order, or Optuna TPE), then analyse.

    With generate=False nothing new is tested; the study re-analyses everything already in the
    registry for this data (e.g. after running approved Claude proposals)."""
    ctx = prepare(cfg)
    ctx.study_id = ctx.reg.create_study(name or cfg["project"]["name"], cfg, ctx.dev_fp, ctx.holdout_fp,
                                        ctx.cost_fp, ctx.split_info)
    fams = families or cfg["search"]["families"]
    for f in fams:
        if generate:
            if sampler == "random":
                n = run_generator(ctx, f)
            else:
                from .optimize import optimize_family
                left = cfg["search"]["max_trials_per_family"] - ctx.reg.n_trials(ctx.dev_fp, f)
                n = optimize_family(ctx, f, left, sampler)["evaluated"] if left > 0 else 0
            log.info("%s: %d new configs evaluated (%d total on this data)", f, n, ctx.reg.n_trials(ctx.dev_fp, f))
    analyzed = {f: analyze_family(ctx, f) for f in fams}
    # Families tested on this data but not analysed in this run count with their raw trial numbers.
    others = ctx.reg.n_trials(ctx.dev_fp) - sum(m["n_trials"] for m in analyzed.values())
    n_eff_total = sum(m.get("n_effective", 0.0) for m in analyzed.values()) + max(others, 0)
    for f, fam in analyzed.items():
        if fam["n_trials"]:
            fam["candidate_results"] = evaluate_candidates(ctx, f, fam, n_eff_total)
        ctx.reg.save_family(ctx.study_id, f, fam)
    bench = summarize(ctx.evaluator.bench, None, ctx.split.warmup_end)
    ctx.reg.save_family(ctx.study_id, "_benchmark", {"symbol": cfg["data"]["benchmark"], "metrics": bench,
                                                     "n_trials_total": ctx.reg.n_trials(ctx.dev_fp),
                                                     "n_effective_total": n_eff_total,
                                                     "dev_fp": ctx.dev_fp})
    ctx.reg.set_study_status(ctx.study_id, "done")
    return ctx.study_id


# ------------------------------------------------------------------ holdout

def run_holdout(cfg: Config, study_id: str | None = None, force: bool = False, reason: str = "") -> dict:
    reg = open_registry(cfg)
    st = reg.study(study_id)
    sid = st["study_id"]
    cands = reg.candidates(sid)
    # A forced re-look re-tests the same frozen list, whatever the first look decided.
    passed = cands[cands["status"].isin(["passed", "holdout_passed", "holdout_failed", "paper"])]
    full = load_panel(cfg, fetch_missing=False)
    s = cfg["splits"]
    split = make_split(full.T, s["holdout_fraction"], s["embargo_days"], s["warmup_days"])
    if full.fingerprint(0, split.dev_end) != st["dev_fp"]:
        raise RuntimeError("data changed since the study ran (dev fingerprint differs); re-run the study")
    vault = HoldoutVault(reg, full, split)
    view = vault.open(sid, list(passed["spec_hash"]), force=force, reason=reason)
    settings = EvalSettings.from_config(cfg, view.panel, split.warmup_end)
    ev = Evaluator(view.panel, settings)
    ev.bench = buy_and_hold(ev.market, settings.bench_col, settings.costs, settings.fill, view.start).returns
    g = cfg["gates"]
    results = {"times_opened": view.times_opened, "contaminated": view.times_opened > 1 or force,
               "window": split.describe(full.dates), "benchmark": summarize(ev.bench, None, view.start),
               "benchmark_returns": np.round(ev.bench[view.start:], 7).tolist(),
               "dates": [str(d) for d in full.dates[view.start:]], "strategies": {}}
    for row in passed.itertuples():
        rob = json.loads(row.robustness_json)
        spec = rob["spec"]
        m, res = ev.run(spec, active_from=view.start, score_start=view.start)
        dev_sr = rob["base"]["sharpe"]
        ok = m["sharpe"] >= g["holdout_min_sharpe"] and m["sharpe"] >= g["holdout_min_sharpe_retention"] * dev_sr
        results["strategies"][row.spec_hash] = {"description": rob["description"], "family": row.family,
                                                "holdout": m, "dev_sharpe": dev_sr, "pass": bool(ok),
                                                "returns": np.round(res.returns[view.start:], 7).tolist()}
        reg.set_candidate_status(sid, row.spec_hash, "holdout_passed" if ok else "holdout_failed")
    reg.save_holdout_results(view.access_id, results)
    reg.close()
    return results


# ------------------------------------------------------------------ promotion to paper trading

def promote(cfg: Config, spec_hash_: str, study_id: str | None = None, force: bool = False,
            reason: str = "") -> dict:
    """Move a strategy to paper trading. Requires a passed holdout unless forced with a reason."""
    from .paper import PaperTrader
    from .validation.stats import rolling_return_quantile
    reg = open_registry(cfg)
    st = reg.study(study_id)
    c = reg.candidates(st["study_id"])
    row = c[c["spec_hash"] == spec_hash_]
    if row.empty:
        raise KeyError(f"{spec_hash_} is not a candidate of study {st['study_id']}")
    status = row.iloc[0]["status"]
    if status != "holdout_passed" and not (force and reason):
        raise RuntimeError(f"candidate status is {status!r}; only holdout_passed strategies go to paper "
                           "trading (use --force with --reason to override, which is logged)")
    rob = json.loads(row.iloc[0]["robustness_json"])
    t = reg.trials(st["dev_fp"], st["cost_fp"])
    _, M = reg.trial_returns(t[t["spec_hash"] == spec_hash_])
    split = json.loads(st["split_json"])
    warm = cfg["splits"]["warmup_days"]
    pcfg = cfg["paper"]
    thr = rolling_return_quantile(M[warm:, 0], pcfg["deterioration_window"], pcfg["deterioration_pctile"],
                                  seed=cfg["project"]["seed"]) if len(M) else None
    pt = PaperTrader(cfg)
    note = f"forced: {reason}" if status != "holdout_passed" else ""
    pt.add_strategy(spec_hash_, rob["spec"], st["study_id"], rob["description"], thr, note)
    pt.close()
    reg.set_candidate_status(st["study_id"], spec_hash_, "paper")
    reg.close()
    return {"strategy": spec_hash_, "deterioration_threshold": thr, "dev_period": split, "forced": bool(note)}
