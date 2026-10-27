"""Claude in the research loop: propose experiments, then comment on finished results.

Safeguards
- Claude sees dev-period summaries only. Holdout results are never put in a prompt.
- Proposals are structured JSON over the registered building blocks (no free-form code is
  executed). Each proposal is validated, sized against the remaining trial budget, and saved
  as `pending` until you approve it (unless research.require_approval is false).
- Every configuration a proposal runs is a counted trial, so the deflated Sharpe ratio pays
  for Claude's exploration exactly as it pays for the random generator's.
- Commentary is checked: any number in Claude's text that does not appear in the report
  tables is flagged in the output.
- A monthly spending cap is checked before every API call.
"""
from __future__ import annotations

import itertools
import json
import logging
import re
import uuid

from ..registry import Registry, now
from ..strategies import BLOCKS, FAMILIES, SpecError, canonical, spec_hash, validate_spec

log = logging.getLogger(__name__)

SYSTEM = """You are a careful quantitative researcher working inside an automated research platform.
You propose experiments; the platform runs them and reports results. Rules:
- Never state or guess performance numbers. Only the platform's backtests produce numbers.
- Every configuration you propose is counted as a trial and makes the multiple-testing penalty
  (deflated Sharpe ratio) stricter for everything found on this data. Propose few, focused
  experiments with a clear economic rationale, not broad parameter sweeps.
- Prefer hypotheses that can fail clearly, and say what result would falsify each one.
- You only see the development period. The final holdout is locked and is not shown to you.
- Use only the building blocks and parameter ranges in the catalog."""

_BLOCK_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["name", "params"],
    "properties": {
        "name": {"type": "string", "enum": list(BLOCKS)},
        "params": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["param", "values"],
                "properties": {
                    "param": {"type": "string"},
                    "values": {"type": "array", "items": {"anyOf": [{"type": "number"}, {"type": "string"}]}},
                },
            },
        },
    },
}

PROPOSAL_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["proposals", "notes"],
    "properties": {
        "notes": {"type": "string"},
        "proposals": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["title", "hypothesis", "economic_rationale", "family", "signal", "filters",
                             "exits", "sizings", "falsified_if", "expected_failure_modes"],
                "properties": {
                    "title": {"type": "string"},
                    "hypothesis": {"type": "string"},
                    "economic_rationale": {"type": "string"},
                    "family": {"type": "string", "enum": list(FAMILIES)},
                    "signal": _BLOCK_SCHEMA,
                    "filters": {"type": "array", "items": _BLOCK_SCHEMA},
                    "exits": {"type": "array", "items": _BLOCK_SCHEMA},
                    "sizings": {"type": "array", "items": _BLOCK_SCHEMA},
                    "falsified_if": {"type": "string"},
                    "expected_failure_modes": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
    },
}


class BudgetExceeded(RuntimeError):
    pass


# ------------------------------------------------------------------ context for the model

def catalog() -> dict:
    out = {"families": {}, "blocks": {}}
    for f in FAMILIES.values():
        out["families"][f.name] = {"hypothesis": f.hypothesis, "signals": f.signals, "filters": f.filters,
                                   "exits": f.exits, "sizings": f.sizings}
    for b in BLOCKS.values():
        out["blocks"][b.name] = {"kind": b.kind, "doc": b.doc, "default_grid": b.grid,
                                 "allowed_ranges": {k: list(v) for k, v in b.bounds.items()}}
    return out


def study_context(reg: Registry, study_id: str | None, cfg) -> dict:
    """Dev-only summary of a finished study. Deliberately excludes anything from the holdout."""
    st = reg.study(study_id)
    fams = reg.family_results(st["study_id"])
    bench = fams.pop("_benchmark", {})
    trials = reg.trials(st["dev_fp"], st["cost_fp"])
    budget = cfg["search"]["max_trials_per_family"]
    out = {"benchmark_dev": bench.get("metrics", {}), "n_trials_total": bench.get("n_trials_total"),
           "families": {}, "candidates": []}
    for f, m in fams.items():
        if not m.get("n_trials"):
            continue
        t = trials[trials["family"] == f].sort_values("sharpe", ascending=False)
        top = [{"description": _desc(json.loads(r.spec_json)), "sharpe": round(r.sharpe, 3),
                "alpha_sharpe": round(r.alpha_sharpe, 3), "max_drawdown": round(r.max_drawdown, 3),
                "n_trades": int(r.n_trades)} for r in t.head(8).itertuples()]
        out["families"][f] = {
            "n_trials": m["n_trials"], "remaining_budget": max(budget - reg.n_trials(st["dev_fp"], f), 0),
            "median_sharpe": round(m["trial_sharpe_median"], 3), "best_sharpe": round(m["trial_sharpe_max"], 3),
            "walk_forward_oos_sharpe": round(m["walk_forward"]["oos"]["sharpe"], 3),
            "walk_forward_oos_alpha_sharpe": round(m["walk_forward"]["oos_alpha_sharpe"], 3),
            "pbo": round(m["pbo"]["pbo"], 3), "spa_p": round(m["spa"]["spa_p"], 3), "top_configs_dev": top}
    for c in reg.candidates(st["study_id"]).itertuples():
        rob = json.loads(c.robustness_json)
        out["candidates"].append({"family": c.family, "description": rob["description"],
                                  "dev_gate_status": "passed" if c.status != "rejected" else "rejected",
                                  "failed_gates": json.loads(c.failed_json)})
    return out


def _desc(spec: dict) -> str:
    def part(d):
        a = ",".join(f"{k}={v}" for k, v in d.items() if k != "name")
        return f"{d['name']}({a})" if a else d["name"]
    return " | ".join(part(spec[k]) for k in ("signal", "filter", "exit", "sizing"))


# ------------------------------------------------------------------ proposal validation

def expand_proposal(p: dict, direction: str = "long") -> tuple[list[dict], list[str]]:
    """Turn one proposal into concrete specs. Returns (specs, errors)."""
    errors: list[str] = []
    fam = FAMILIES.get(p.get("family"))
    if fam is None:
        return [], [f"unknown family {p.get('family')!r}"]

    def variants(block: dict, slot: str, allowed: tuple[str, ...]) -> list[dict]:
        name = block["name"]
        if name not in allowed:
            errors.append(f"{slot} block {name!r} is not part of family {fam.name} (allowed: {list(allowed)})")
            return []
        grid = {d["param"]: d["values"] for d in block.get("params", [])}
        expected = set(BLOCKS[name].grid)
        if set(grid) != expected:
            errors.append(f"{name}: params must be exactly {sorted(expected)}, got {sorted(grid)}")
            return []
        keys = sorted(grid)
        return [{"name": name, **dict(zip(keys, combo))} for combo in itertools.product(*(grid[k] for k in keys))]

    sig = variants(p["signal"], "signal", fam.signals)
    flt = [v for b in (p.get("filters") or [{"name": "none", "params": []}]) for v in variants(b, "filter", fam.filters)]
    ext = [v for b in (p.get("exits") or [{"name": "hold", "params": []}]) for v in variants(b, "exit", fam.exits)]
    siz = [v for b in (p.get("sizings") or [{"name": "slot", "params": []}]) for v in variants(b, "sizing", fam.sizings)]
    specs, seen = [], set()
    for s, f, e, z in itertools.product(sig, flt, ext, siz):
        spec = canonical({"family": fam.name, "direction": direction, "signal": s, "filter": f, "exit": e, "sizing": z})
        try:
            validate_spec(spec, check_bounds=True)
        except SpecError as ex:
            if len(errors) < 20:
                errors.append(str(ex))
            continue
        h = spec_hash(spec)
        if h not in seen:
            seen.add(h)
            specs.append(spec)
    return specs, errors


# ------------------------------------------------------------------ Claude calls

def _client():
    try:
        import anthropic
    except ImportError as e:
        raise RuntimeError("pip install anthropic (or `pip install -e .[claude]`)") from e
    return anthropic.Anthropic()


def _call(client, cfg, reg: Registry, purpose: str, messages: list[dict], schema: dict | None = None) -> str:
    import anthropic
    r = cfg["research"]
    spent = reg.llm_spend_this_month()
    if spent >= r["monthly_budget_usd"]:
        raise BudgetExceeded(f"Claude spend this month ${spent:.2f} >= cap ${r['monthly_budget_usd']:.2f}")
    output_config: dict = {"effort": r.get("effort", "high")}
    if schema:
        output_config["format"] = {"type": "json_schema", "schema": schema}
    try:
        resp = client.beta.messages.create(
            model=r["model"], max_tokens=16000, system=SYSTEM, messages=messages, output_config=output_config,
            betas=["server-side-fallback-2026-07-01"], fallbacks="default")
    except anthropic.RateLimitError as e:
        raise RuntimeError(f"Claude rate limit; retry later ({e.message})") from e
    except anthropic.APIStatusError as e:
        raise RuntimeError(f"Claude API error {e.status_code}: {e.message}") from e
    except anthropic.APIConnectionError as e:
        raise RuntimeError("could not reach the Claude API (network)") from e
    u = resp.usage
    cost = (u.input_tokens * r["price_in_per_mtok"] + u.output_tokens * r["price_out_per_mtok"]) / 1e6
    reg.log_llm(purpose, getattr(resp, "model", r["model"]), u.input_tokens, u.output_tokens, cost,
                getattr(resp, "_request_id", "") or "")
    if resp.stop_reason == "refusal":
        raise RuntimeError("Claude declined this request (stop_reason=refusal)")
    if resp.stop_reason == "max_tokens":
        raise RuntimeError("Claude's answer was cut off at max_tokens; nothing was saved")
    return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")


def propose(cfg, reg: Registry, study_id: str | None = None, guidance: str = "", client=None) -> list[str]:
    """Ask Claude for experiments, validate them, store them as proposals. Returns proposal ids."""
    st = reg.study(study_id)
    ctx = study_context(reg, st["study_id"], cfg)
    msg = ("Here is the building-block catalog and a summary of the finished study (development period "
           "only).\n\nCATALOG:\n" + json.dumps(catalog()) + "\n\nSTUDY:\n" + json.dumps(ctx, default=float) +
           f"\n\nPropose at most 3 experiments. Each must expand to at most "
           f"{cfg['research']['max_configs_per_proposal']} configurations and fit the family's remaining budget. "
           "Explain the weaknesses you see in the results so far in `notes`." +
           (f"\n\nGuidance from the researcher: {guidance}" if guidance else ""))
    text = _call(client or _client(), cfg, reg, "propose", [{"role": "user", "content": msg}], PROPOSAL_SCHEMA)
    data = json.loads(text)
    ids = []
    direction = "long_short" if cfg["execution"]["allow_short"] else "long"
    cap = cfg["research"]["max_configs_per_proposal"]
    for p in data.get("proposals", []):
        specs, errors = expand_proposal(p, direction)
        fam = p.get("family")
        remaining = ctx["families"].get(fam, {}).get("remaining_budget", cfg["search"]["max_trials_per_family"])
        status = "pending" if cfg["research"]["require_approval"] else "approved"
        if not specs:
            status, errors = "invalid", errors or ["expands to zero valid configurations"]
        elif len(specs) > cap:
            status = "invalid"
            errors.append(f"expands to {len(specs)} configs, more than the cap of {cap}")
        elif len(specs) > remaining:
            status = "invalid"
            errors.append(f"expands to {len(specs)} configs; family {fam} has {remaining} trials left")
        pid = uuid.uuid4().hex[:10]
        payload = {"proposal": p, "errors": errors, "specs": specs, "notes": data.get("notes", "")}
        reg.con.execute("INSERT INTO proposals VALUES (?,?,?,?,?,?,?,?,?)",
                        [pid, now(), status, st["dev_fp"], json.dumps(payload), len(specs), None, None,
                         "; ".join(errors)])
        ids.append(pid)
    return ids


def list_proposals(reg: Registry):
    return reg.con.execute("SELECT proposal_id, created_at, status, n_configs, notes, payload_json FROM proposals "
                           "ORDER BY created_at DESC").df()


def decide(reg: Registry, proposal_id: str, approve: bool, note: str = ""):
    row = reg.con.execute("SELECT status FROM proposals WHERE proposal_id=?", [proposal_id]).fetchone()
    if not row:
        raise KeyError(proposal_id)
    if row[0] != "pending":
        raise RuntimeError(f"proposal is {row[0]}, not pending")
    reg.con.execute("UPDATE proposals SET status=?, decided_at=?, notes=? WHERE proposal_id=?",
                    ["approved" if approve else "rejected", now(), note, proposal_id])


def run_proposal(ctx, proposal_id: str) -> int:
    """Execute an approved proposal on the dev data. Returns the number of new trials."""
    from ..pipeline import search
    row = ctx.reg.con.execute("SELECT status, payload_json, dev_fp FROM proposals WHERE proposal_id=?",
                              [proposal_id]).fetchone()
    if not row:
        raise KeyError(proposal_id)
    status, payload, dev_fp = row
    if status != "approved":
        raise RuntimeError(f"proposal is {status}; only approved proposals run")
    if dev_fp != ctx.dev_fp:
        raise RuntimeError("proposal was made for different data; propose again")
    p = json.loads(payload)
    fam = p["proposal"]["family"]
    n = search(ctx, fam, p["specs"], "claude")
    ctx.reg.con.execute("UPDATE proposals SET status='executed', executed_at=? WHERE proposal_id=?", [now(), proposal_id])
    return n


# ------------------------------------------------------------------ commentary with a number check

NUM = re.compile(r"(?<![\w.])[-+]?\d+(?:\.\d+)?%?")


def unverified_numbers(text: str, source: str) -> list[str]:
    """Numbers in `text` (2+ significant chars) that do not appear in `source`."""
    have = set(NUM.findall(source))
    have |= {h.rstrip("%") for h in have}
    out = []
    for n in NUM.findall(text):
        bare = n.rstrip("%").lstrip("+")
        if len(bare.replace("-", "").replace(".", "")) < 2 or bare.isdigit() and len(bare) == 4 and bare[:2] in ("19", "20"):
            continue                      # tiny integers and years are not results
        if n not in have and bare not in have:
            out.append(n)
    return sorted(set(out))


def commentary(cfg, reg: Registry, report_md: str, client=None) -> str:
    if "## Final holdout" in report_md:
        report_md = report_md.split("## Final holdout")[0]       # Claude never sees holdout results
    msg = ("Below is a research report produced by the platform (development period only). Write a short "
           "analysis: (1) what the evidence does and does not support, (2) weaknesses and failure conditions, "
           "(3) which strategy families look worth more research and which should be dropped, (4) concrete next "
           "experiments. Quote numbers only exactly as they appear in the report.\n\n" + report_md)
    text = _call(client or _client(), cfg, reg, "commentary", [{"role": "user", "content": msg}])
    bad = unverified_numbers(text, report_md)
    if bad:
        text += ("\n\n> Number check: these figures in the commentary do not appear in the report tables and "
                 f"should not be trusted: {', '.join(bad)}")
    return text
