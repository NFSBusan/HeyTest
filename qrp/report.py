"""Research report (Markdown) built only from numbers stored in the registry.

Nothing here computes or invents results; it formats what the pipeline saved. Optional
Claude commentary (qrp.research.agent) is appended in its own clearly labelled section.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .registry import Registry


def _f(x, pct=False, nd=2):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "n/a"
    if isinstance(x, str):
        return x
    return f"{x * 100:.1f}%" if pct else f"{x:.{nd}f}"


def build_report(reg: Registry, study_id: str | None = None, commentary: str | None = None) -> str:
    st = reg.study(study_id)
    sid = st["study_id"]
    cfg = json.loads(st["config_json"])
    split = json.loads(st["split_json"])
    fams = reg.family_results(sid)
    bench = fams.pop("_benchmark", {})
    cands = reg.candidates(sid)
    holdouts = reg.holdout_accesses(st["holdout_fp"])
    L: list[str] = []
    L.append(f"# Research report: {st['name']}")
    L.append(f"Study `{sid}`, created {st['created_at']}. Status: {st['status']}.\n")
    provider = cfg["data"]["provider"]
    if provider == "synthetic":
        edge = cfg["data"].get("synthetic", {}).get("edge", "none")
        L.append(f"> **Simulated data** (synthetic provider, planted edge: `{edge}`). These results test the "
                 "software; they say nothing about real markets.\n")
    L.append("## Setup")
    L.append(f"- Data: {provider}, {cfg['data']['timeframe']} bars, {len(cfg['data']['symbols'])} symbols, "
             f"benchmark {cfg['data']['benchmark']}")
    L.append(f"- Dev period (all search and tuning): {split['dev_start']} to {split['dev_end']} "
             f"(warm-up from {split['warmup_start']})")
    L.append(f"- Final holdout (locked): {split['holdout_start']} to {split['holdout_end']}")
    c = cfg["costs"]
    L.append(f"- Execution: {cfg['execution']['fill']}; costs {c['commission_bps']} bps commission + "
             f"{c['half_spread_bps']} bps half-spread + impact (coef {c['impact_coef']}) at ${c['capital']:,} capital")
    L.append(f"- Configurations tested on this dev data, all studies: **{bench.get('n_trials_total', 'n/a')}**, "
             f"equivalent to about **{_f(bench.get('n_effective_total'), nd=0)}** independent bets given how "
             "correlated they are (the N used to deflate Sharpe ratios)\n")
    if bench:
        m = bench["metrics"]
        L.append(f"Benchmark buy-and-hold {bench['symbol']} over the dev period: Sharpe {_f(m.get('sharpe'))}, "
                 f"CAGR {_f(m.get('cagr'), True)}, max drawdown {_f(m.get('max_drawdown'), True)}.\n")
    L.append("## Strategy families (dev period)")
    L.append("| Family | Configs | Effective N | Median Sharpe | Best Sharpe | Walk-forward OOS Sharpe | "
             "WF OOS alpha Sharpe | PBO | SPA p | Reality Check p |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for f, m in fams.items():
        if not m.get("n_trials"):
            continue
        wf = m["walk_forward"]
        L.append(f"| {f} | {m['n_trials']} | {_f(m.get('n_effective'), nd=0)} | {_f(m['trial_sharpe_median'])} | {_f(m['trial_sharpe_max'])} | "
                 f"{_f(wf['oos']['sharpe'])} | {_f(wf['oos_alpha_sharpe'])} | {_f(m['pbo']['pbo'])} | "
                 f"{_f(m['spa']['spa_p'], nd=3)} | {_f(m['spa']['rc_p'], nd=3)} |")
    L.append("\nDSR uses the effective number of independent configs; *DSR (strict)* in the candidate table "
             "uses the raw count and the cross-trial Sharpe spread, which over-penalises families where many "
             "configs share a real edge.")
    L.append("\nHow to read this: *walk-forward OOS* re-runs 'pick the best config so far' on expanding "
             "windows and scores it only on the following block, so it estimates what the selection "
             "procedure earns out of sample. *PBO* near or above 0.5 means the in-sample winner is no better "
             "than a random pick out of sample. *SPA / Reality Check p* test whether any config in the "
             "family has positive alpha after accounting for the whole search (small is good).\n")
    for f, m in fams.items():
        if not m.get("n_trials"):
            continue
        L.append(f"### {f}")
        L.append(f"Hypothesis: {m['hypothesis']}\n")
        wf = m["walk_forward"]
        L.append(f"Walk-forward picked {wf['distinct_choices']} different configs across {len(wf['folds'])} folds "
                 f"(benchmark Sharpe over the same OOS days: {_f(wf['benchmark_oos_sharpe'])}).\n")
        L.append("| Test period | Chosen config | In-sample Sharpe | Out-of-sample Sharpe |")
        L.append("|---|---|---|---|")
        for fo in wf["folds"]:
            period = " to ".join(fo["test_dates"]) if "test_dates" in fo else f"rows {fo['test'][0]}-{fo['test'][1]}"
            L.append(f"| {period} | `{fo['spec_hash']}` | {_f(fo['is_sharpe'])} | {_f(fo['oos_sharpe'])} |")
        L.append("")
    L.append("## Candidates and gates")
    if cands.empty:
        L.append("No candidates (no configuration met the minimum trade count).\n")
    else:
        n_pass = int(cands["status"].isin(["passed", "holdout_passed", "holdout_failed", "paper"]).sum())
        L.append(f"{len(cands)} candidates (best of distinct return clusters per family); "
                 f"**{n_pass} passed every dev gate**.\n")
        L.append("| Family | Strategy | Dev Sharpe | Alpha Sharpe | CAGR | Max DD | Trades | DSR | DSR (strict) | "
                 "Neighbour ratio | Sharpe @2x costs | Status | Failed gates |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for row in cands.itertuples():
            rob = json.loads(row.robustness_json)
            b = rob["base"]
            stress = next(iter(rob["cost_stress"].values()), {})
            failed = json.loads(row.failed_json)
            L.append(f"| {row.family} | `{row.spec_hash}` {rob['description']} | {_f(b['sharpe'])} | "
                     f"{_f(b['alpha_sharpe'])} | {_f(b['cagr'], True)} | {_f(b['max_drawdown'], True)} | "
                     f"{b.get('n_trades')} | {_f(rob['dsr']['dsr'])} | {_f(rob.get('dsr_strict', {}).get('dsr'))} | "
                     f"{_f(rob['neighbour_ratio'])} | "
                     f"{_f(stress.get('sharpe'))} | {row.status} | {', '.join(failed) or '-'} |")
        L.append("")
        L.append("### Robustness detail (candidates)")
        for row in cands.itertuples():
            rob = json.loads(row.robustness_json)
            mc, rg = rob["monte_carlo"], rob["regimes"]
            cap = ", ".join(f"${int(float(k)):,}: Sharpe {_f(v['sharpe'])}" for k, v in rob["capacity"].items())
            L.append(f"- `{row.spec_hash}`: bootstrap Sharpe 5th pct {_f(mc['sharpe_p05'])}, "
                     f"P(Sharpe<0) {_f(mc['p_sharpe_below_0'])}, 95th pct drawdown {_f(mc['mdd_p95'], True)}; "
                     f"regime Sharpe bull {_f(rg['bull']['sharpe'])} / bear {_f(rg['bear']['sharpe'])} / "
                     f"high-vol {_f(rg['high_vol']['sharpe'])} / low-vol {_f(rg['low_vol']['sharpe'])}; "
                     f"capacity {cap}")
        L.append("")
    L.append("## Final holdout")
    if holdouts.empty:
        L.append("Not opened. Run `qrp holdout` once the candidate list is final; it can be opened once.\n")
    else:
        L.append(f"Opened {len(holdouts)} time(s)." + (" **Contaminated: opened more than once, so these "
                 "numbers are no longer a clean out-of-sample test.**" if len(holdouts) > 1 else ""))
        last = holdouts.iloc[-1]
        if last["results_json"]:
            res = json.loads(last["results_json"])
            bm = res["benchmark"]
            L.append(f"\nBenchmark over the holdout: Sharpe {_f(bm.get('sharpe'))}, CAGR {_f(bm.get('cagr'), True)}.\n")
            L.append("| Strategy | Dev Sharpe | Holdout Sharpe | Holdout CAGR | Holdout max DD | Pass |")
            L.append("|---|---|---|---|---|---|")
            for h, s in res["strategies"].items():
                ho = s["holdout"]
                L.append(f"| `{h}` {s['description']} | {_f(s['dev_sharpe'])} | {_f(ho.get('sharpe'))} | "
                         f"{_f(ho.get('cagr'), True)} | {_f(ho.get('max_drawdown'), True)} | {s['pass']} |")
            if not res["strategies"]:
                L.append("| (no candidates had passed the dev gates) | | | | | |")
        L.append("")
    L.append("## Labels")
    L.append("- **In-sample (dev)**: everything in the candidate table. Selected on this data, so optimistic.")
    L.append("- **Walk-forward OOS**: honest estimate of the selection procedure, still inside the dev period.")
    L.append("- **Holdout**: one look at untouched data. **Paper/live**: see `qrp paper status`.\n")
    if commentary:
        L.append("## Commentary (written by Claude from the tables above; numbers checked against them)")
        L.append(commentary)
    return "\n".join(L)


def write_report(reg: Registry, out_dir: str, study_id: str | None = None, commentary: str | None = None) -> Path:
    text = build_report(reg, study_id, commentary)
    sid = reg.study(study_id)["study_id"]
    p = Path(out_dir) / f"report_{sid}.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p
