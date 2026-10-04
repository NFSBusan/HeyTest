# Research Log: Counter-Trading the Worst Polymarket Traders

This file is the record of the work. It is updated after every step so nothing is lost
if the chat gets compressed. Newest entries are at the bottom of each section.

## Goal (from user)

1. Find the worst-performing Polymarket traders and take the opposite side of their trades.
2. Backtest and forward ("front") test, sizing with Kelly at several fractions.
3. No data leakage.
4. Keep iterating until the best variant is found.
5. List anything that's missing, with the end goal of running it live on Polymarket.

## Environment constraint (2026-10-04)

The cloud session's network policy blocks every Polymarket host (`gamma-api.polymarket.com`,
`data-api.polymarket.com`, `clob.polymarket.com`, the Goldsky subgraph). The proxy rejects
the connection, so this session can't download real trade history.

What that means:
- The pipeline is built against Polymarket's real data schema (`pmct/fetch.py`), but the
  backtests here run on a **simulated** market (`pmct/synthetic.py`).
- Simulated results check that the code works and has no leakage. They are **not** evidence
  that the strategy makes money on Polymarket. Don't trade real money on them.
- To get real results: allow those hosts in the environment's network settings (or run
  locally), then `python scripts/fetch_data.py` and `python scripts/run_research.py --data real`.

## Plan

| Step | What | Status |
|---|---|---|
| 1 | Data layer: simulated market done (`pmct/synthetic.py`); real fetcher | sim done, fetcher todo |
| 2 | Point-in-time trader scoring (`pmct/scoring.py`), 5 bases × 4 scores | done |
| 3 | Counter-signal + walk-forward probability calibration + Kelly sizing (`pmct/strategy.py`) | done |
| 4 | Event-driven backtester (latency, spread, impact, fees, caps, cash limits) | done |
| 5 | Leakage tests (`tests/test_leakage.py`, 6 pass); null world runs in research script | done |
| 6 | Iterations: parameter search on dev period only, deflated Sharpe for multiple testing | todo |
| 7 | Front test: one run on the untouched holdout period, all Kelly fractions | todo |
| 8 | Live paper-trading script for Polymarket (dry run, no real orders) | todo |
| 9 | List of what's missing before going live | todo |

## Design decisions (leakage controls)

- **Time split**: warm-up (history only, no trading) → dev/backtest (all tuning happens here)
  → front test/holdout (touched once, after the config is frozen).
- **Point-in-time scoring**: a trade only counts toward a wallet's score once its market has
  *resolved*, strictly before the decision timestamp (`merge_asof(..., allow_exact_matches=False)`).
- **Execution**: we react to a bad trader's fill after a latency delay and pay the price at
  that later time, plus half-spread, slippage and fees. We never fill at their price.
- **Dev evaluation boundary**: positions still open at the end of the dev period are marked to
  the last price before the boundary, so outcomes that resolve inside the holdout never reach
  parameter selection.
- **Probability model for Kelly**: calibrated only on earlier signals whose markets already
  resolved.
- **Multiple testing**: every config tried is counted, and the winner's Sharpe is deflated
  for the number of trials.

## Iteration log

(filled in as runs complete)

Simulated world: 540 days, 1,600 markets, 4,000 wallets, ~405k trades. Warm-up 0–120d,
dev 120–390d, holdout 390–540d. Costs: pay mid + 1¢, plus impact 2¢ × stake/(stake+their size).

**It 1. Baseline: score = t-stat of raw resolved PnL per share, threshold −2.**
Dev signal edge −0.9¢/share (t = −3.4 clustered by market). Kelly 0.1–1.0 lose 59–74%.
Why: raw PnL includes the spread the wallet paid, so the "worst" list fills up with
high-volume *noise* traders who lose only the spread. Counter-trading them pays the spread
a second time. **Lesson for real Polymarket: "lowest PnL on the leaderboard" ≠ "anti-predictive".**
Price-bucket calibration also chased noise in the cheap buckets (hit rate 19%).

**It 2. Other ways to define "bad"** (resolution vs mid, markouts at 1h/6h/1d), global calibration.
Signal edges +0.3¢ to +1.6¢ but t ≤ 1.75. Not significant. Few or no Kelly bets.

**Oracle check (uses hidden truth, diagnostic only):** even counter-trading wallets we *know*
are worst (skill < −0.6) earns only +0.8–1.0¢/share after the spread. Latency up to 1h barely
matters here because the simulated price doesn't react to their trades (real markets may differ).
So the ceiling is thin, and the job is to identify those wallets well.

**Discrimination check** (Spearman of point-in-time score vs true skill, wallets with ≥30
resolved trades): ROI on resolved PnL is best (0.47). t-stat scores are 0.15–0.32. ROI is
dollar-weighted, so it reflects where the wallet actually puts its money.

**Leakage fix found by the future-poisoning test:** positions still open at the end of a
window carried their future outcome in the output table. It wasn't used in P&L, but it is now
blanked. All 6 tests pass.

**It 3. Quick full grid (960 configs) → overfit.** Dev winner (`mo1d`/`roi` ≤ −0.10, Kelly 1.0) made
+28% on dev, Sharpe 1.6, but deflated Sharpe = 0.25 (not significant after 960 tries), and it
**lost at every Kelly fraction on the holdout** (−2% at 0.1 to −44% at 1.5). The protocol caught it.
(This peek at seed-7's holdout is disclosed. The final test uses fresh worlds, seeds 101–103.)

**It 4–10. Structured iterations** (`scripts/iterate.py`, `results/iterations.csv`), judged on
dev windows of two worlds (seeds 7, 11) plus a null-world control:
- Oracle ceiling (counter the truly worst wallets): +15–20% log growth over 270 dev days at
  Kelly 0.25, Sharpe 0.8–1.2. **Even with perfect knowledge, Kelly above 0.5 lowered growth**
  because edge estimates are noisy.
- Best real signal: `res_mid` / `roi` ≤ −0.30, ≥100 resolved trades, our price 0.30–0.95.
  Clustered t = +2.8 / +3.7 on the two dev worlds and +0.9 on the null world. Real but small edge.
- Kelly sizing turns that into only +1.5–2% growth (Kelly 0.25). Kelly ≥ 0.5 is flat or negative.
- One bet per wallet-market, bad-money consensus, and linear "badness" calibration: no robust gain.
- Early exit after 6/24/72h: strongly negative (round trip pays the spread twice).
- Stress: exec cost 1.5¢ → negative; 2¢ → negative; latency 10 min ≈ fine; 1h → mixed.

## Realistic trading rules (added at user request)

Engine now models: 1¢ tick (buys round up, sells round down), cash-out by selling into the bid
(`exit_cost` below mid, thinner books), taker fee on **both** legs of an early cash-out (redemption
at resolution is free), 5-share minimum order, and a 2h UMA liveness delay before cash is usable.
Bug found and fixed: `fee_rate` was cached as a sizing-only field, so fee scenarios reused stale signals.

**Cash-out scenarios** (`scripts/cashout_scenarios.py`, `results/cashout_scenarios.csv`; dev, seeds 7/11,
edge in ¢/share, clustered t):

| fee | exit | s7 edge | s7 t | s11 edge | s11 t |
|---|---|---|---|---|---|
| 0% | hold | see csv | ≈+2.9 | | ≈+2.9 |
| 1% | hold | +1.24 | +2.64 | −0.10 | +2.63 |
| 1% | 6h cash-out | −3.76 | −13.9 | −3.92 | −17.6 |
| 1% | 24h cash-out | −3.21 | −6.8 | −3.48 | −6.8 |
| 1% | 72h cash-out | −2.38 | −1.9 | −2.82 | −2.8 |
| 2% | hold | +0.96 | +2.43 | −0.38 | +2.37 |
| 2% | 6h, thin book (3¢) | −5.84 | −21.8 | −6.01 | −27.2 |

**Conclusion: never cash out early with this strategy. Hold to resolution.** Each early exit costs
roughly 2.5–6¢ per share (spread + tick + fee twice), while the edge is about 1¢. Fees alone push
the trade-weighted edge to ~0. With the 1¢ tick rounding, the Kelly calibrator often finds no
bet worth making. That's the honest outcome for a thin edge.
