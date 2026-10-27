"""Command line: python -m qrp <command> [options]   (or `qrp <command>` after `pip install -e .`)

Typical order:
  fetch -> research -> report/dashboard -> (propose -> approve -> run-proposal -> research --no-generate)
  -> holdout (once) -> promote -> paper run (daily)
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
from pathlib import Path

from .config import load_config


def _cfg(a):
    overrides = {}
    for kv in a.set or []:
        k, v = kv.split("=", 1)
        try:
            overrides[k] = json.loads(v)
        except json.JSONDecodeError:
            overrides[k] = v
    return load_config(a.config, overrides)


def cmd_fetch(a):
    from .data import fetch_all
    for r in fetch_all(_cfg(a)):
        print(f"{r.symbol:6s} +{r.fetched_rows:5d} rows, {r.stored_rows:5d} stored"
              + (f"  issues: {'; '.join(r.issues)}" if r.issues else ""))


def _outputs(cfg, sid):
    from .dashboard import write_dashboard
    from .pipeline import prepare
    from .registry import open_registry
    from .report import write_report
    reg = open_registry(cfg)
    ctx = prepare(cfg, reg)
    out = Path(cfg["storage"].get("reports_dir", "reports"))
    rp = write_report(reg, str(out), sid)
    dp = write_dashboard(reg, str(out), ctx.evaluator.bench, ctx.split.warmup_end, sid)
    reg.close()
    return rp, dp


def cmd_research(a):
    from .pipeline import run_study
    cfg = _cfg(a)
    if a.budget:
        cfg = cfg.with_overrides({"search.max_trials_per_family": a.budget})
    sid = run_study(cfg, a.name, a.families.split(",") if a.families else None, not a.no_generate, a.sampler)
    rp, dp = _outputs(cfg, sid)
    print(f"study {sid}\nreport    {rp}\ndashboard {dp}")


def cmd_report(a):
    rp, dp = _outputs(_cfg(a), a.study)
    print(f"report    {rp}\ndashboard {dp}")


def cmd_holdout(a):
    from .pipeline import run_holdout
    cfg = _cfg(a)
    res = run_holdout(cfg, a.study, a.force, a.reason or "")
    for h, s in res["strategies"].items():
        print(f"{h} {s['description']}\n   dev Sharpe {s['dev_sharpe']:.2f} -> holdout Sharpe "
              f"{s['holdout'].get('sharpe', float('nan')):.2f}  {'PASS' if s['pass'] else 'FAIL'}")
    _outputs(cfg, a.study)


def cmd_optimize(a):
    from .optimize import optimize_family
    from .pipeline import prepare
    cfg = _cfg(a)
    ctx = prepare(cfg)
    ctx.study_id = "optimize"
    res = optimize_family(ctx, a.family, a.trials, a.sampler)
    res.pop("objective_trace", None)
    print(json.dumps(res, indent=2))
    print("run `qrp research --no-generate` to analyse these trials with the full validation suite")


def cmd_compare(a):
    from .optimize import compare_samplers
    res = compare_samplers(_cfg(a), a.family, a.trials)
    for s, r in res.items():
        tr = r.get("objective_trace", [])
        marks = {k: round(tr[k - 1], 3) for k in (10, 25, 50, 100, 200) if len(tr) >= k}
        print(f"{s:6s} evaluated {r.get('evaluated', 0):4d}  best objective {r.get('best_objective', float('nan')):.3f}"
              f"  best-so-far at n={marks}")


def cmd_propose(a):
    from .registry import open_registry
    from .research.agent import list_proposals, propose
    cfg = _cfg(a)
    reg = open_registry(cfg)
    ids = propose(cfg, reg, a.study, a.guidance or "")
    df = list_proposals(reg)
    for r in df[df["proposal_id"].isin(ids)].itertuples():
        p = json.loads(r.payload_json)
        print(f"{r.proposal_id} [{r.status}] {p['proposal']['title']} ({r.n_configs} configs)\n"
              f"   {p['proposal']['hypothesis']}\n   falsified if: {p['proposal']['falsified_if']}"
              + (f"\n   problems: {r.notes}" if r.notes else ""))
    reg.close()


def cmd_proposals(a):
    from .registry import open_registry
    from .research.agent import list_proposals
    reg = open_registry(_cfg(a))
    for r in list_proposals(reg).itertuples():
        print(f"{r.proposal_id} {r.created_at:%Y-%m-%d} [{r.status}] {json.loads(r.payload_json)['proposal']['title']}"
              f" ({r.n_configs} configs)")
    reg.close()


def cmd_decide(a, approve: bool):
    from .registry import open_registry
    from .research.agent import decide
    reg = open_registry(_cfg(a))
    decide(reg, a.proposal_id, approve, a.note or "")
    reg.close()
    print("approved" if approve else "rejected")


def cmd_run_proposal(a):
    from .pipeline import prepare
    from .research.agent import run_proposal
    cfg = _cfg(a)
    ctx = prepare(cfg)
    ctx.study_id = f"proposal-{a.proposal_id}"
    print(f"{run_proposal(ctx, a.proposal_id)} new trials; now run `qrp research --no-generate`")


def cmd_commentary(a):
    from .registry import open_registry
    from .report import build_report, write_report
    from .research.agent import commentary
    cfg = _cfg(a)
    reg = open_registry(cfg)
    text = commentary(cfg, reg, build_report(reg, a.study))
    print(write_report(reg, cfg["storage"].get("reports_dir", "reports"), a.study, text))
    reg.close()


def cmd_promote(a):
    from .pipeline import promote
    print(json.dumps(promote(_cfg(a), a.spec_hash, a.study, a.force, a.reason or ""), indent=2, default=str))


def cmd_paper(a):
    from .data import fetch_all, load_panel
    from .paper import PaperTrader
    cfg = _cfg(a)
    pt = PaperTrader(cfg)
    if a.action == "run":
        live = cfg.with_overrides({"data.end": datetime.date.today().isoformat()})
        fetch_all(live)
        panel = load_panel(live, fetch_missing=False)
        for sid, msg in pt.run_day(panel).items():
            print(sid, msg)
    elif a.action == "status":
        print(pt.status().to_string(index=False))
        print("\nrecent events:\n" + pt.events(20).to_string(index=False))
    pt.close()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="qrp", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="path to YAML config (default config/qrp.yaml)")
    ap.add_argument("--set", action="append", metavar="KEY=VALUE", help="override a config value, e.g. data.provider=tiingo")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch", help="download missing bars into the Parquet store").set_defaults(fn=cmd_fetch)
    p = sub.add_parser("research", help="run a study: search, validate, report")
    p.add_argument("--name")
    p.add_argument("--families", help="comma-separated subset")
    p.add_argument("--budget", type=int, help="override search.max_trials_per_family")
    p.add_argument("--sampler", choices=["random", "grid", "tpe"], default="random")
    p.add_argument("--no-generate", action="store_true", help="only analyse trials already in the registry")
    p.set_defaults(fn=cmd_research)
    p = sub.add_parser("report", help="rebuild report + dashboard")
    p.add_argument("--study")
    p.set_defaults(fn=cmd_report)
    p = sub.add_parser("holdout", help="ONE look at the final holdout for the passed candidates")
    p.add_argument("--study")
    p.add_argument("--force", action="store_true")
    p.add_argument("--reason")
    p.set_defaults(fn=cmd_holdout)
    p = sub.add_parser("optimize", help="spend part of a family's budget with a chosen sampler")
    p.add_argument("--family", required=True)
    p.add_argument("--trials", type=int, default=200)
    p.add_argument("--sampler", choices=["random", "grid", "tpe"], default="tpe")
    p.set_defaults(fn=cmd_optimize)
    p = sub.add_parser("compare-samplers", help="grid vs random vs TPE at equal budget (synthetic data only)")
    p.add_argument("--family", required=True)
    p.add_argument("--trials", type=int, default=200)
    p.set_defaults(fn=cmd_compare)
    p = sub.add_parser("propose", help="ask Claude for experiments (needs ANTHROPIC_API_KEY)")
    p.add_argument("--study")
    p.add_argument("--guidance")
    p.set_defaults(fn=cmd_propose)
    sub.add_parser("proposals", help="list proposals").set_defaults(fn=cmd_proposals)
    for name, ok in (("approve", True), ("reject", False)):
        p = sub.add_parser(name, help=f"{name} a pending proposal")
        p.add_argument("proposal_id")
        p.add_argument("--note")
        p.set_defaults(fn=lambda a, ok=ok: cmd_decide(a, ok))
    p = sub.add_parser("run-proposal", help="run an approved proposal on dev data")
    p.add_argument("proposal_id")
    p.set_defaults(fn=cmd_run_proposal)
    p = sub.add_parser("commentary", help="Claude writes an analysis of the latest report (numbers checked)")
    p.add_argument("--study")
    p.set_defaults(fn=cmd_commentary)
    p = sub.add_parser("promote", help="move a holdout-passed strategy to paper trading")
    p.add_argument("spec_hash")
    p.add_argument("--study")
    p.add_argument("--force", action="store_true")
    p.add_argument("--reason")
    p.set_defaults(fn=cmd_promote)
    p = sub.add_parser("paper", help="paper trading: run (daily) or status")
    p.add_argument("action", choices=["run", "status"])
    p.set_defaults(fn=cmd_paper)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    try:
        a.fn(a)
    except (RuntimeError, KeyError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
