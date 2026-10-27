"""End-to-end: study, registry dedup and trial counting, holdout lock, report, paper replay."""
import json

import numpy as np
import pandas as pd
import pytest

from qrp.backtest import run_backtest
from qrp.dashboard import build_dashboard
from qrp.paper import LiveTradingRefused, PaperTrader, replay
from qrp.pipeline import prepare, promote, run_holdout, run_study
from qrp.registry import open_registry
from qrp.report import build_report
from qrp.strategies import sample_family
from qrp.validation.holdout import HoldoutLocked, HoldoutVault

from .conftest import make_cfg


def test_dev_panel_never_contains_holdout_rows(cfg):
    ctx = prepare(cfg)
    assert ctx.dev.T == ctx.split.dev_end
    assert ctx.split.holdout_start - ctx.split.dev_end == cfg["splits"]["embargo_days"]
    assert pd.Timestamp(ctx.dev.dates[-1]) < pd.Timestamp(ctx.split_info["holdout_start"])
    ctx.reg.close()


def test_study_counts_trials_once_across_studies(cfg):
    sid1 = run_study(cfg, "a")
    reg = open_registry(cfg)
    st = reg.study(sid1)
    n1 = reg.n_trials(st["dev_fp"])
    assert n1 == 3 * cfg["search"]["max_trials_per_family"]
    reg.close()
    sid2 = run_study(cfg, "b")                           # budget already spent: nothing new is tested
    reg = open_registry(cfg)
    assert reg.n_trials(st["dev_fp"]) == n1
    assert reg.study(sid2)["dev_fp"] == st["dev_fp"]
    fams = reg.family_results(sid2)
    assert fams["_benchmark"]["n_trials_total"] == n1
    report = build_report(reg, sid2)
    assert "Final holdout" in report and "Not opened" in report
    reg.close()


def test_holdout_opens_once(cfg):
    ctx = prepare(cfg)
    from qrp.data import load_panel
    full = load_panel(cfg)
    vault = HoldoutVault(ctx.reg, full, ctx.split)
    with pytest.raises(HoldoutLocked):
        vault.open("s", [])                              # needs frozen candidates
    vault.open("s", ["abc"])
    with pytest.raises(HoldoutLocked):
        vault.open("s", ["abc"])
    with pytest.raises(HoldoutLocked):
        vault.open("s", ["abc"], force=True)             # forced look needs a reason
    view = vault.open("s", ["abc"], force=True, reason="testing")
    assert view.times_opened == 2                        # refused attempts are not looks
    ctx.reg.close()


def test_run_holdout_refuses_without_candidates_then_marks_contamination(cfg):
    sid = run_study(cfg)
    with pytest.raises(HoldoutLocked):
        run_holdout(cfg, sid)                            # nothing passed: the holdout stays unused
    reg = open_registry(cfg)
    assert reg.holdout_accesses(reg.study(sid)["holdout_fp"]).empty
    h = reg.candidates(sid)["spec_hash"].iloc[0]
    reg.set_candidate_status(sid, h, "passed")           # pretend it passed every dev gate
    reg.close()
    res = run_holdout(cfg, sid)
    assert set(res["strategies"]) == {h}
    assert res["times_opened"] == 1 and not res["contaminated"]
    with pytest.raises(HoldoutLocked):
        run_holdout(cfg, sid)
    reg = open_registry(cfg)
    reg.set_candidate_status(sid, h, "passed")
    reg.close()
    res2 = run_holdout(cfg, sid, force=True, reason="test")
    assert res2["contaminated"]
    reg = open_registry(cfg)
    assert "Contaminated" in build_report(reg, sid)
    reg.close()


def test_null_world_passes_nothing(tmp_path):
    """With no planted edge, a full search must not produce a strategy that passes every gate."""
    cfg = make_cfg(tmp_path, **{"search.max_trials_per_family": 120})
    sid = run_study(cfg)
    reg = open_registry(cfg)
    c = reg.candidates(sid)
    assert len(c) > 0
    assert (c["status"] == "rejected").all(), c[["family", "failed_json"]]
    # defence in depth: noise should fail several independent gates, not squeak past one
    assert all(len(json.loads(f)) >= 2 for f in c["failed_json"])
    reg.close()


def test_planted_trend_edge_is_detected(tmp_path):
    """With a real (simulated) trend effect, the trend family's statistics should say so."""
    cfg = make_cfg(tmp_path, **{"data.synthetic.edge": "trend", "data.synthetic.edge_strength": 1.5,
                                "search.max_trials_per_family": 120, "search.families": ["trend"]})
    sid = run_study(cfg)
    reg = open_registry(cfg)
    fam = reg.family_results(sid)["trend"]
    assert fam["walk_forward"]["oos_alpha_sharpe"] > 0.3
    assert fam["spa"]["spa_p"] < 0.05
    c = reg.candidates(sid)
    best = max(json.loads(r)["dsr"]["dsr"] for r in c["robustness_json"])
    assert best > 0.95
    assert (c["status"] == "passed").any(), c[["failed_json"]]      # the platform can accept a real edge
    reg.close()


def test_dashboard_builds_and_is_valid_json(cfg):
    sid = run_study(cfg)
    ctx = prepare(cfg)
    html = build_dashboard(ctx.reg, ctx.evaluator.bench, ctx.split.warmup_end, sid)
    payload = html.split("const D = ", 1)[1].split(";\nconst $", 1)[0]
    data = json.loads(payload)
    assert data["trials"] and data["wf"]
    assert "NaN" not in payload
    ctx.reg.close()


def test_paper_replay_matches_backtest(cfg, tmp_path):
    """The live paper trader, stepped day by day, must reproduce the backtest engine."""
    ctx = prepare(cfg)
    spec = sample_family("trend", 1, seed=11)[0]
    start, end = 1500, 1700
    curve = replay(cfg, ctx.dev, spec, start, end, str(tmp_path / "replay.duckdb"))
    W = ctx.evaluator.weights(spec)
    bt = run_backtest(W, ctx.evaluator.market, ctx.settings.costs, ctx.settings.fill, active_from=start)
    paper_r = curve["daily_return"].to_numpy()
    bt_r = bt.returns[start:end]
    assert len(paper_r) == len(bt_r)
    assert np.corrcoef(paper_r[2:], bt_r[2:])[0, 1] > 0.999
    assert abs(np.prod(1 + paper_r) - np.prod(1 + bt_r)) < 0.003
    ctx.reg.close()


def test_paper_refuses_live_trading(cfg):
    bad = cfg.with_overrides({"paper.live_trading_enabled": True})
    with pytest.raises(LiveTradingRefused):
        PaperTrader(bad)


def test_promote_requires_holdout_pass(cfg):
    sid = run_study(cfg)
    reg = open_registry(cfg)
    h = reg.candidates(sid)["spec_hash"].iloc[0]
    reg.close()
    with pytest.raises(RuntimeError):
        promote(cfg, h, sid)
    out = promote(cfg, h, sid, force=True, reason="test only")
    assert out["forced"]
    pt = PaperTrader(cfg)
    assert h in set(pt.strategies()["strategy_id"])
    pt.close()


def test_paper_kill_switch_blocks_orders(cfg, tmp_path, monkeypatch):
    ctx = prepare(cfg)
    ks = tmp_path / "KILL"
    c2 = cfg.with_overrides({"paper.kill_switch_file": str(ks)})
    pt = PaperTrader(c2, str(tmp_path / "ks.duckdb"))
    pt.add_strategy("x", sample_family("trend", 1, seed=2)[0], "s", "x", None)
    ks.write_text("stop")
    view = ctx.dev.head(1600)
    pt.run_day(view, today=pd.Timestamp(view.dates[-1]))
    assert pt.con.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 0
    ks.unlink()
    view = ctx.dev.head(1601)
    pt.run_day(view, today=pd.Timestamp(view.dates[-1]))
    ev = pt.events()
    assert (ev["kind"] == "kill_switch").any()
    pt.close()
    ctx.reg.close()


def test_paper_skips_stale_data(cfg, tmp_path):
    ctx = prepare(cfg)
    pt = PaperTrader(cfg, str(tmp_path / "st.duckdb"))
    pt.add_strategy("x", sample_family("trend", 1, seed=2)[0], "s", "x", None)
    view = ctx.dev.head(1600)
    out = pt.run_day(view, today=pd.Timestamp(view.dates[-1]) + pd.Timedelta(days=30))
    assert out["x"] == "stale"
    pt.close()
    ctx.reg.close()
