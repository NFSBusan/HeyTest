# Quant research platform (`qrp`): plan, setup and operation

This is the improved version of "Strategy 2". It is built and tested on simulated data. Nothing
in this repository is evidence that any strategy makes money in real markets.

## 1. What changed from the original plan, and why

| Original plan | Problem | What `qrp` does instead |
|---|---|---|
| Generate 10,000-100,000+ indicator combinations | Searching more configs mostly finds more luck; simple technical rules on liquid markets have a low prior of working | Strategy **families** built around a documented economic idea (trend, short-term mean reversion, cross-sectional momentum/reversal), each with a hard trial budget (default 1,500) |
| Overfitting checks after the search | Checks applied afterwards can't undo an unrecorded search | Every configuration ever tested on the same data is counted (across studies, Optuna runs and Claude proposals). The deflated Sharpe uses the *effective* number of independent bets (from the correlation of the trials' alpha returns), and the raw count is reported alongside |
| No pass bar written down | Gates chosen after seeing results are just more tuning | Gates are in `config/qrp.yaml` and stored with each study. "Nothing passed" is a valid, expected result |
| "Prevent repeated holdout use" | No mechanism was specified | `HoldoutVault`: the dev panel is physically truncated; the holdout opens once per window, every look is logged before any number is computed, and a second look is labelled *contaminated* in every report |
| Claude writes and runs strategy code | Hidden trials, prompt-injected code, invented numbers | Claude proposes **structured JSON** over registered blocks; proposals are validated, sized against the budget and wait for your approval; report numbers come from the database; Claude's commentary is number-checked; Claude never sees the holdout |
| Sharpe vs zero | A long-only fund strategy "wins" just by holding the market | Edge tests (deflated Sharpe, SPA/Reality Check) use **alpha**: returns minus what the strategy's market beta would have earned. Raw Sharpe and excess-over-benchmark are gated separately |
| Fixed slippage | Underestimates costs, ignores capacity | Commission + half-spread + square-root market impact using lagged volatility and dollar volume, drift-aware turnover, borrow for shorts, capacity checks at 10x/100x capital, cost stress at 2x/3x |
| Cloud first | Daily data for a few dozen ETFs is a few MB | Runs on a laptop. Cloud storage and compute are optional switches (section 4) |
| Six asset classes incl. options from day one | Each is its own data and modelling project | US ETFs, daily bars first. Expand only after something survives the holdout and paper trading |
| Strategies judged one by one | Thousands of near-duplicates | Candidates are the best of distinct return clusters (correlation ≤ 0.9) |
| Paper trading as the final check | Paper fills are optimistic | The paper trader is verified against the backtest engine (replay test: daily return correlation > 0.999) and tracks a deterioration threshold from the backtest's bootstrap distribution |

Assumptions I made because the setup questions weren't answered yet: US ETFs, daily bars,
long-only, about $10k of capital, $0/month to start, a laptop or small VM. All of these are config values.

## 2. Architecture

```
                 config/qrp.yaml (every number that affects a result; stored with each study)
                          |
  providers (Tiingo / Alpaca / CSV / synthetic)        <- API keys from env vars only
        |   only missing date ranges are fetched
  BarStore: Parquet (zstd) per symbol, local or s3:// (R2/B2)  -- DuckDB queries it in place
        |
  Panel (dates x symbols, point-in-time tradable mask) --split--> dev | embargo | HOLDOUT (vault)
        |
  IndicatorCache (Numba/pandas, causal, memoised)
        |
  Strategy spec = signal -> filter -> exit -> sizing   (JSON, stable hash)
        |   generator (random/grid) | Optuna TPE | Claude proposals (approved)
  Backtest engine (Numba; next-open fills; costs; trades)   x N cores
        |
  Registry (DuckDB): studies, trials (dedup + counting), returns (Parquet), candidates,
                     holdout log, proposals, LLM spend
        |
  Validation: walk-forward of the selection procedure, PBO (CSCV), Hansen SPA + White RC,
              deflated Sharpe on alpha, neighbours, cost stress, capacity, Monte Carlo, regimes
        |
  Gates -> report.md + dashboard.html  -> holdout (once) -> promote -> paper trader (daily)
```

Code map: `qrp/data` (providers, store, quality, panel), `qrp/indicators.py`, `qrp/strategies`
(blocks, families, specs), `qrp/backtest` (engine, metrics), `qrp/evaluate.py` (parallel),
`qrp/registry.py`, `qrp/validation` (stats, robustness, holdout), `qrp/pipeline.py`,
`qrp/optimize.py`, `qrp/research/agent.py` (Claude), `qrp/paper/trader.py`, `qrp/report.py`,
`qrp/dashboard.py`, `qrp/cli.py`.

## 3. Data providers (prices checked October 2026; confirm before paying)

| Provider | Good for | History | Free tier | Paid | Notes |
|---|---|---|---|---|---|
| **Tiingo** | Daily US stocks/ETFs, adjusted (splits + dividends) | Decades for most ETFs | Yes: ~50 req/hour, ~1,000/day, 500 symbols/month | "Power" about $30/month per recent third-party listings (Tiingo's own page shows no public price) | **Default real provider for the MVP.** Wired in (`provider: tiingo`) |
| **Alpaca** | Free data plus a broker for later; intraday | Bars from ~2016 | Basic: IEX real-time, 15-min delayed SIP via REST, 200 req/min | Algo Trader Plus $99/month (full SIP, OPRA options) | Wired in (`provider: alpaca`). The free IEX feed has only partial volume |
| **Massive (formerly Polygon.io)** | Stocks, options, indices, FX, crypto; intraday | Plan-dependent | Basic: 5 calls/min, end-of-day, ~2 years | Individual plans listed at roughly $29-$399/month by third parties; one source says custom pricing | Renamed Oct 30 2025. Not wired in yet. Add it as a `Provider` subclass |
| **Databento** | Futures, options, equities tick/minute (institutional-grade) | Deep | $125 credit for new accounts | Pay-as-you-go $/GB for history, or flat plans | Best choice when you move to futures or intraday |
| **QuantConnect** | Hosted research + bundled data | Deep | Free tier, cloud backtests only | Researcher plan $60-$84/month (sources differ) | Data generally stays on their platform; it doesn't feed this pipeline |
| **Norgate** | Survivorship-free US stocks incl. delisted, historical index membership | To 1990s / 1950 | 21-day trial | Roughly $270-$790/year depending on tier | Needed before any single-stock strategy. Windows app; export to CSV, then `provider: csv` |

Bias handling built in:
- **Splits/dividends**: providers return total-return-adjusted prices; the quality check flags
  one-day moves over 40% (a typical sign of a missed split).
- **Survivorship**: optional `universe_file` (symbol, start, end) restricts each symbol to the days
  it was really in the universe. Broad ETFs are low-risk; single stocks need Norgate-style data.
- **Lookahead**: indicators are causal (tested by truncating the future), signals at the close of day t
  trade at the open of day t+1, and Donchian channels exclude the current bar.
- **Missing data**: never forward-filled into returns; an asset with no price that day can't be traded.

## 4. Infrastructure and monthly cost

| Setup | What runs where | Monthly cost |
|---|---|---|
| **A. Laptop (recommended start)** | Everything local; daily data for 25 ETFs is under 10 MB | **$0** (Tiingo/Alpaca free) |
| B. Laptop + cloud storage | Parquet store on Cloudflare R2 (10 GB free, then $0.015/GB, free egress) or Backblaze B2 (~$6.95/TB, 10 GB free) | **$0** at this data size |
| C. Daily paper trading in the cloud | GitHub Actions cron (`docs/examples/paper-trading.yml`): free for public repos; private repos on the Free plan get ~2,000 minutes/month (one run ≈ 1-2 min) | **$0** |
| D. Bigger searches | On-demand VM only while a study runs (e.g. Hetzner shared vCPU, raised twice in 2026 and roughly €5-€15/month if left on), or a spot VM on AWS/GCP | $0-15 |
| E. Claude research loop | ~20k input + 4k output tokens per proposal round on Claude Opus 5.5 ($4 / $20 per million tokens) ≈ $0.16 per round; monthly cap in config (`research.monthly_budget_usd: 10`) | $1-10 |
| F. Paid data later | Tiingo Power (~$30) or Alpaca Algo Trader Plus ($99); Databento pay-per-GB for futures | $30-100+ |

Realistic total for the MVP: **$0-10/month**. Spending more becomes worth it only once a strategy
survives the holdout and several months of paper trading.

## 5. Libraries

numpy, pandas, numba (backtest engine and indicators), pyarrow + DuckDB (storage/queries),
scipy (statistics), optuna (TPE search), pyyaml, requests, anthropic (optional), pytest.
I left out VectorBT: the engine here is ~200 lines of Numba with exact next-open timing, drift-aware
costs and per-trade bookkeeping, all of which are tested. That's easier to audit than a
large dependency, and it ran about 2,200 backtests per minute on 4 cores (23 ETFs × 20 years, full validation included).

## 6. Roadmap with exit criteria

| Phase | Content | Done when |
|---|---|---|
| 0 Rules | Config with gates, registry, trial counting, holdout vault, cost model | ✅ built and tested |
| 1 MVP | Data engine, indicators, 3 families, engine, metrics, walk-forward, report | ✅ built; run on **real** data next |
| 2 Robustness | DSR on alpha, PBO, SPA/RC, neighbours, cost stress, capacity, MC, regimes, dedup, Optuna, dashboard | ✅ built |
| 3 Claude loop | Proposals → approval → run → re-analyse; commentary with number check; monthly cap | ✅ built (tested with a mocked client) |
| 4 Paper trading | Daily runner, fills, ledger, kill switch, drawdown kill, stale-data guard, deterioration alerts, webhook | ✅ built; needs live data |
| 5 Scale-out | Intraday bars, more providers, futures (Databento), single stocks (Norgate), cloud workers | Only after a strategy has passed the holdout **and** ≥ 3 months of paper trading without breaching its deterioration threshold |
| Live money | Not implemented. Needs your explicit decision, a broker adapter, and its own review | — |

## 7. Exact first steps

```bash
git clone <this repo> && cd HeyTest
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev,claude]"
pytest -q                                              # ~2 minutes, all offline
```

1. **Try it on simulated data** (no keys needed):
   ```bash
   python -m qrp research --budget 300
   ```
   Open `reports/dashboard_<study>.html` in a browser. Try `--set data.synthetic.edge=trend` to see what a
   detected edge looks like and `edge=none` to see everything rejected.
2. **Get a free Tiingo key** (tiingo.com → account → API token). Then `cp .env.example .env` and put the key
   in `.env`. `.env` is git-ignored; keys are only ever read from the environment.
3. **Fetch real data and run the first real study**:
   ```bash
   python -m qrp --set data.provider=tiingo fetch
   python -m qrp --set data.provider=tiingo research --name first-real
   ```
   The free tier allows ~50 requests/hour; 23 symbols take one request each, and re-runs fetch nothing.
4. **Read the report before touching anything.** Note how many configs were tried and which gates failed.
   If nothing passed, that is information: don't loosen gates to make something pass.
5. **Optional: Claude proposals**: put `ANTHROPIC_API_KEY` in `.env`, then
   ```bash
   python -m qrp --set data.provider=tiingo propose --guidance "focus on why trend fails in 2010-2019"
   python -m qrp proposals
   python -m qrp approve <id>
   python -m qrp --set data.provider=tiingo run-proposal <id>
   python -m qrp --set data.provider=tiingo research --no-generate
   ```
6. **Holdout, once**, after the candidate list is final: `python -m qrp --set data.provider=tiingo holdout`.
7. **Paper trading** for holdout passers: `python -m qrp promote <hash>`, then run
   `python -m qrp --set data.provider=tiingo paper run` after each US close (or use the GitHub Actions example)
   and check `python -m qrp paper status`.

Instead of repeating `--set data.provider=tiingo`, edit `data.provider` in `config/qrp.yaml`.

## 8. API keys and security

- Keys live in environment variables (`.env` locally, repository secrets in GitHub Actions). They are never
  written to config, logs or the database. `.env.example` lists every variable.
- Data keys only need read access. The Claude key should be in its own workspace with a spend limit set
  in the Anthropic console, in addition to `research.monthly_budget_usd`.
- No broker credentials exist anywhere in this system. Paper trading simulates fills.

## 9. Running in this cloud session

This Claude Code cloud environment blocks the data hosts (Tiingo, Alpaca, Polygon/Massive, Databento,
Yahoo, Binance were all unreachable when tested). Either run `fetch` on your own machine and commit or
upload the Parquet files, or add those hosts to the environment's network allowlist in the
environment settings. Everything else (tests, synthetic studies) works here.

## 10. How the platform was checked

Three simulated worlds, 3,990 configurations each, full default config (`docs/QRP_LOG.md` has details):

| World | Truth | Candidates passing every dev gate | Holdout |
|---|---|---|---|
| No edge | Nothing to find | **0 of 15**: each fails ≥ 3 independent gates (best DSR 0.87) | not opened |
| Planted trend | Strong trend + cross-sectional effects | **10 of 15** (DSR 1.00, walk-forward OOS Sharpe 1.3-1.5) | 10 of 10 held up (Sharpe 1.3-1.7) |
| Planted short-term reversal | Weak, high-turnover effect | 0 of 15: walk-forward and SPA see it, but DSR 0.84-0.945 and cost drag reject it | not opened |

Sampler comparison at 150 trials (planted trend): grid 1.21, random 1.28, Optuna TPE 1.56 best objective.

## 11. Known limitations (honest list)

- Results so far are on simulated data only. The Tiingo/Alpaca connectors and the Claude calls are tested
  against mocked responses; neither has been run live from this environment (network blocked, no key).
- Cash earns 0% in the engine, so strategies that sit in cash are understated vs a T-bill-aware
  benchmark; both sides use the same convention.
- Delisted assets: a position in an asset whose data stops is carried with zero return. Survivorship-free
  single-stock data needs explicit delisting returns.
- Close-based stops (evaluated at the close, executed next open). No intrabar stop fills.
- Market impact uses a fixed starting capital, not current equity, in the backtest.
- Intraday bars are fetched by the Alpaca provider code path but not yet supported by the panel.
- The holdout is a single window. Once used it's spent: wait for new data to get a fresh one.
