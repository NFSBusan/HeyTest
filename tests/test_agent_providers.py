"""Claude research loop (mocked client) and HTTP providers (mocked responses). No network."""
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from qrp.data.providers import AlpacaProvider, ProviderError, TiingoProvider, normalize
from qrp.data.quality import check_bars
from qrp.data.store import BarStore
from qrp.pipeline import prepare, run_study
from qrp.registry import open_registry
from qrp.research import agent


class FakeResp:
    def __init__(self, status, payload, headers=None):
        self.status_code, self._p, self.headers, self.text = status, payload, headers or {}, json.dumps(payload)

    def json(self):
        return self._p


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append((url, dict(params or {}), dict(headers or {})))
        return self.responses.pop(0)


def test_tiingo_parses_adjusted_fields_and_sends_token():
    rows = [{"date": "2024-01-02T00:00:00.000Z", "open": 10, "high": 11, "low": 9, "close": 10.5, "volume": 100,
             "adjOpen": 5, "adjHigh": 5.5, "adjLow": 4.5, "adjClose": 5.25, "adjVolume": 200, "divCash": 0,
             "splitFactor": 1}]
    s = FakeSession([FakeResp(200, rows)])
    df = TiingoProvider("KEY", session=s).fetch("SPY", pd.Timestamp("2024-01-01").date(), pd.Timestamp("2024-01-05").date())
    assert df["close"].iloc[0] == 5.25 and df["raw_close"].iloc[0] == 10.5 and df["volume"].iloc[0] == 200
    assert s.calls[0][2]["Authorization"] == "Token KEY"


def test_alpaca_follows_pagination_and_retries_429(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    p1 = {"bars": {"SPY": [{"t": "2024-01-02T05:00:00Z", "o": 1, "h": 2, "l": 0.5, "c": 1.5, "v": 10}]},
          "next_page_token": "abc"}
    p2 = {"bars": {"SPY": [{"t": "2024-01-03T05:00:00Z", "o": 1.5, "h": 2, "l": 1, "c": 1.8, "v": 12}]},
          "next_page_token": None}
    s = FakeSession([FakeResp(429, {}, {"Retry-After": "0"}), FakeResp(200, p1), FakeResp(200, p2)])
    df = AlpacaProvider("id", "secret", session=s).fetch("SPY", pd.Timestamp("2024-01-01").date(),
                                                         pd.Timestamp("2024-01-05").date())
    assert list(df["close"]) == [1.5, 1.8]
    assert s.calls[-1][1]["page_token"] == "abc" and s.calls[0][1]["adjustment"] == "all"


def test_provider_raises_on_auth_error():
    s = FakeSession([FakeResp(401, {"detail": "bad key"})])
    with pytest.raises(ProviderError):
        TiingoProvider("bad", session=s).fetch("SPY", pd.Timestamp("2024-01-01").date(), pd.Timestamp("2024-01-05").date())


def test_store_only_fetches_missing_ranges(tmp_path):
    calls = []

    class P:
        name = "fake"

        def fetch(self, symbol, start, end, timeframe="1d"):
            calls.append((start, end))
            d = pd.bdate_range(start, end)
            return normalize(pd.DataFrame({"date": d, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0}))

    st = BarStore(str(tmp_path), "fake")
    st.update(P(), "AAA", pd.Timestamp("2020-01-01").date(), pd.Timestamp("2020-06-30").date())
    st.update(P(), "AAA", pd.Timestamp("2020-01-01").date(), pd.Timestamp("2020-06-30").date())
    assert len(calls) == 1                                   # second run: nothing missing, no API call
    st.update(P(), "AAA", pd.Timestamp("2020-01-01").date(), pd.Timestamp("2020-12-31").date())
    assert len(calls) == 2 and pd.Timestamp(calls[1][0]) > pd.Timestamp("2020-06-10")   # short overlap only
    assert st.coverage("AAA")[1] >= pd.Timestamp("2020-12-30")
    q = st.query(["AAA"], "2020-03-01", "2020-03-31")
    assert q["date"].min() >= pd.Timestamp("2020-03-01") and set(q["symbol"]) == {"AAA"}


def test_store_refetches_history_when_adjustment_changes(tmp_path):
    """After a dividend, the provider's adjusted history shifts; the stored copy must be replaced."""
    state = {"factor": 1.0}
    calls = []

    class P:
        name = "fake"

        def fetch(self, symbol, start, end, timeframe="1d"):
            calls.append((start, end))
            d = pd.bdate_range(start, end)
            c = 100.0 * state["factor"]
            return normalize(pd.DataFrame({"date": d, "open": c, "high": c, "low": c, "close": c, "volume": 1.0}))

    st = BarStore(str(tmp_path), "fake")
    st.update(P(), "AAA", pd.Timestamp("2020-01-01").date(), pd.Timestamp("2020-06-30").date())
    state["factor"] = 0.98                                  # a dividend re-scales all past adjusted prices
    rep = st.update(P(), "AAA", pd.Timestamp("2020-01-01").date(), pd.Timestamp("2020-09-30").date())
    assert "re-adjusted" in rep.issues
    df = st.read("AAA")
    assert df["close"].nunique() == 1 and df["close"].iloc[0] == pytest.approx(98.0)
    assert pd.Timestamp(calls[-1][0]) <= pd.Timestamp("2020-01-01")


def test_quality_flags_unadjusted_split():
    d = pd.bdate_range("2020-01-01", periods=10)
    c = [100.0] * 5 + [50.0] * 5
    df = pd.DataFrame({"date": d, "open": c, "high": c, "low": c, "close": c, "volume": 1.0})
    assert any("split" in m for m in check_bars(df))


# ------------------------------------------------------------------ Claude loop

def fake_client(text):
    usage = SimpleNamespace(input_tokens=1000, output_tokens=500)
    resp = SimpleNamespace(content=[SimpleNamespace(type="text", text=text)], usage=usage, stop_reason="end_turn",
                           model="claude-opus-5-5", _request_id="req_1")
    calls = []

    def create(**kw):
        calls.append(kw)
        return resp
    return SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=create))), calls


def block(name, **grid):
    return {"name": name, "params": [{"param": k, "values": v} for k, v in grid.items()]}


def good_proposal():
    return {"title": "Slow trend with vol filter", "hypothesis": "h", "economic_rationale": "r", "family": "trend",
            "signal": block("tsmom", lookback=[100, 150]), "filters": [block("none")], "exits": [block("hold")],
            "sizings": [block("inverse_vol", n=[30])], "falsified_if": "f", "expected_failure_modes": ["x"]}


def test_proposal_expansion_and_validation():
    specs, errors = agent.expand_proposal(good_proposal())
    assert len(specs) == 2 and not errors
    bad = {**good_proposal(), "signal": block("tsmom", lookback=[100, 99999])}
    specs, errors = agent.expand_proposal(bad)
    assert len(specs) == 1 and errors                         # out-of-range value dropped with a reason
    wrong_family = {**good_proposal(), "signal": block("rsi_reversion", n=[2], lower=[10], exit_level=[50])}
    assert agent.expand_proposal(wrong_family)[0] == []


def test_propose_approve_run_counts_trials(cfg):
    sid = run_study(cfg)
    reg = open_registry(cfg)
    payload = {"notes": "n", "proposals": [good_proposal(), {**good_proposal(), "family": "nope"}]}
    client, calls = fake_client(json.dumps(payload))
    ids = agent.propose(cfg, reg, sid, client=client)
    sent = json.dumps(calls[0]["messages"])
    assert "holdout_passed" not in sent and "Final holdout" not in sent
    assert calls[0]["output_config"]["format"]["type"] == "json_schema"
    df = agent.list_proposals(reg).set_index("proposal_id")
    statuses = sorted(df.loc[ids, "status"])
    # the first proposal fits, but the family budget (40) is already spent by the generator
    assert statuses == ["invalid", "invalid"]
    reg.close()
    big = cfg.with_overrides({"search.max_trials_per_family": 100})
    reg = open_registry(big)
    ids = agent.propose(big, reg, sid, client=client)
    ok = [i for i in ids if agent.list_proposals(reg).set_index("proposal_id").loc[i, "status"] == "pending"]
    assert len(ok) == 1
    with pytest.raises(RuntimeError):
        ctx = prepare(big, reg)
        agent.run_proposal(ctx, ok[0])                        # not approved yet
    agent.decide(reg, ok[0], True)
    before = reg.n_trials(ctx.dev_fp)
    n = agent.run_proposal(ctx, ok[0])
    assert n == 2 and reg.n_trials(ctx.dev_fp) == before + 2
    assert reg.llm_spend_this_month() > 0
    reg.close()


def test_budget_cap_blocks_calls(cfg):
    run_study(cfg)
    reg = open_registry(cfg)
    reg.log_llm("x", "m", 0, 0, 999.0)
    client, _ = fake_client("{}")
    with pytest.raises(agent.BudgetExceeded):
        agent.propose(cfg, reg, client=client)
    reg.close()


def test_commentary_flags_invented_numbers(cfg):
    sid = run_study(cfg)
    reg = open_registry(cfg)
    from qrp.report import build_report
    rep = build_report(reg, sid)
    client, calls = fake_client("The trend family reached a Sharpe of 7.77 which is great.")
    out = agent.commentary(cfg, reg, rep, client=client)
    assert "7.77" in out.split("Number check")[1]
    assert "## Final holdout" not in calls[0]["messages"][0]["content"]
    reg.close()
