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
| 1 | Data layer: simulator (`pmct/synthetic.py`); real fetcher (`pmct/fetch.py`, offline-tested only) | done |
| 2 | Point-in-time trader scoring (`pmct/scoring.py`), 5 bases × 4 scores | done |
| 3 | Counter-signal + walk-forward probability calibration + Kelly sizing (`pmct/strategy.py`) | done |
| 4 | Event-driven backtester (latency, spread, impact, fees, caps, cash limits) | done |
| 5 | Leakage tests (`tests/test_leakage.py`, 6 pass); null world runs in research script | done |
| 6 | Iterations on dev only (`scripts/iterate.py`, `scripts/run_research.py`) + realistic Polymarket rules | done |
| 7 | Final test on untouched data, all Kelly fractions (`scripts/final_test.py`) | done: **failed** |
| 8 | Live paper-trading script (`scripts/paper_trade.py`, never places orders) | done, untested (network) |
| 9 | What's missing before going live (bottom of this file) | done |

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

**It 11–12 (realistic rules, rerun of all iterations).** With Polymarket's 2026 fee formula and the 1¢ tick,
**taker** versions place zero bets, even the oracle. Fees plus spread exceed the edge. A **maker** version
(post a limit 1¢ below mid on the counter side, cancel after 1h, fill only if a later print trades *through*
our price, 0 fee, rebates ignored) was the first config positive on both dev worlds and flat on the null world:
dev growth +6.6% / +13.4% at Kelly 0.25. The automatic pick was Kelly 1.0 (best *average*). I froze **Kelly 0.25**
instead (best *worst case* across worlds), and wrote that rule down before running the final test.

## Final test (frozen config, run once): `results/final_test.csv`

Frozen: `res_mid` ROI ≤ −30% over ≥100 resolved trades (≥5 markets); first flagged trade per wallet-market;
our price 0.30–0.95; maker limit = mid − 1¢, TTL 1h; global walk-forward calibration (n0=100);
Kelly 0.25, ≤5% per bet, ≤10% per market; Polymarket 2026 fees; hold to resolution.

Kelly 0.25, untouched data:

| world / window | growth | Sharpe | max DD | bets | signal t |
|---|---|---|---|---|---|
| seed 7 holdout | −6.5% | −1.87 | 10.6% | 208 | −0.15 |
| seed 11 holdout | +9.6% | +1.45 | 6.4% | 183 | +0.57 |
| seed 101 holdout | +3.8% | +0.92 | 4.5% | 119 | +1.10 |
| seed 101 dev-window | −1.2% | −0.64 | 1.9% | 49 | +0.96 |
| seed 102 holdout | −1.5% | −0.38 | 5.6% | 231 | −0.34 |
| seed 102 dev-window | −0.8% | −0.26 | 2.7% | 183 | −0.25 |
| seed 103 both windows | 0 | – | – | 0 | −0.4 / −0.7 |
| null world (both windows) | 0 | – | – | 0 | +0.26 / −0.95 |

All Kelly fractions, 8 out-of-sample windows:

| Kelly | mean growth | median | worst | % windows positive | worst max DD |
|---|---|---|---|---|---|
| 0.10 | −0.0% | −0.2% | −5.1% | 25% | 6.3% |
| 0.25 | +0.4% | −0.4% | −6.5% | 25% | 10.6% |
| 0.50 | +0.6% | 0.0% | −6.4% | 38% | 14.2% |
| 0.75 | +1.2% | 0.0% | −7.4% | 38% | 16.6% |
| 1.00 | +1.6% | 0.0% | −8.2% | 38% | 18.3% |
| 1.50 | +2.3% | 0.0% | −7.4% | 38% | 19.5% |

**Verdict: the strategy does not survive out of sample in the simulated world.** No window has a
significant signal (|t| ≤ 1.1). Deflated Sharpe (1,300 trials) is 0.01–0.44, nowhere near the 0.95 bar.
Bigger Kelly fractions raise the mean only through a few lucky windows (the median stays at 0) while
drawdowns roughly double. The dev results were selection luck. No leakage showed up: the null world never
traded, and all 11 leakage/conversion tests pass.

I stopped iterating here on purpose. More tuning against the same simulated worlds would just fit noise
again; their holdouts are now spent. **The next iterations must run on real Polymarket data:**
`python scripts/fetch_data.py` → `python scripts/run_research.py --data real` → `python scripts/final_test.py`
(edit it to load real data with a fresh, never-touched final period) → weeks of
`python scripts/paper_trade.py run` + `reconcile`.

## Rules and regulation that apply (researched 2026-10-04; verify yourself, this is not legal advice)

1. **Where you can trade.** The international site (polymarket.com) blocks OFAC-sanctioned places (Iran, Syria,
   Cuba, North Korea, occupied Ukrainian regions) and puts 30+ countries (e.g. France, Germany, Australia,
   Brazil, Singapore) in **close-only** mode. Some (Ireland, Japan, Netherlands; Malta for sports) are reported
   as frontend-blocked with API access open. **Using a VPN to get around a geoblock breaks the Terms of Use**
   and can get the account put into close-only mode or closed. Check your own country's status at
   help.polymarket.com before building anything.
2. **US persons** must use **Polymarket US** (CFTC-regulated exchange via the QCEX acquisition, with KYC). It has a
   different fee schedule (taker 0.05, maker rebate). **Accounts there are not public wallets, so you cannot see
   other traders' fills. The core input of this strategy (who the bad traders are) does not exist on the US
   venue.** The strategy only works where trades are on-chain with public proxy wallets (the international CLOB).
   CFTC proposed a formal event-contract rule in June 2026, so expect more change.
3. **Fees (international, Fee Structure V2, 2026).** Takers pay `shares × feeRate × p × (1−p)`. feeRate by
   category: crypto 0.07; sports/economics/culture/weather/other 0.05; politics/finance/tech/mentions 0.04;
   geopolitics 0. Makers pay nothing and get a share of taker fees as rebates. Redemption at resolution is free.
   Now modelled in `pmct/strategy.py` (`PM_FEE_RATES`).
4. **Resolution (UMA).** A proposal plus a 2-hour challenge window (a $500 bond to dispute); a dispute adds a
   second round, and a second dispute goes to a UMA vote (~48h plus a debate period). Ambiguous events can
   resolve 50/50 ($0.50 per share). Capital stays locked meanwhile. Modelled as a 2h delay; a 3-day stress
   barely changed results.
5. **Market integrity rules (updated March 2026).** Prohibited: trading on stolen/confidential information,
   trading on illegal tips, and trading by people who can influence the outcome; also wash trading and
   spoofing. Polymarket says it has referred 90+ accounts to law enforcement. **Counter-trading wallets based on
   public on-chain data is not insider trading.** But don't post-and-cancel orders to move prices, and don't
   trade against yourself across wallets.
6. **Bots/API.** Automated trading through the CLOB API is allowed. Reported limits are about 100 req/min on
   public endpoints and 60 orders/min. Order minimum is about 5 shares; the tick is 1¢ (finer near 0/1).
7. **Tax (US example).** No 1099 is issued. Treatment is unsettled (other income, capital gains, or gambling).
   If it is gambling, 2026 rules (OBBBA) cap loss deductions at 90% of losses, and only against winnings.
   High-turnover strategies need per-trade records; the ledger in `paper_trade.py` is a start.
   Talk to a tax professional where you live.

Sources: datawallet.com/crypto/polymarket-restricted-countries, help.polymarket.com/en/articles/13364163,
docs.polymarket.com/trading/fees, docs.polymarket.us/fees, congress.gov/crs-product/LSB11441,
integrity.polymarket.com, venable.com (2026/04 insider trading), docs.polymarket.com/concepts/resolution,
startpolymarket.com/learn/how-markets-resolve, keepertax.com and marketmath.io (prediction-market taxes),
quantvps.com (automated trading on Polymarket).

## What's missing / before going live

1. **Real data.** Everything above is simulated. Allow the Polymarket hosts (environment network settings) or run
   locally. Prefer the Goldsky orderbook subgraph for complete fills; the Data API has offset caps.
2. **Order-book data.** Maker fills are inferred from prints, with no queue position or depth. Record CLOB book
   snapshots (websocket) during paper trading to measure real fill rates and adverse selection.
3. **The leaderboard trap.** Never pick "worst traders" from today's leaderboard and backtest them in the past.
   That is lookahead plus survivorship bias. Scores here are strictly point-in-time.
4. **Wallet identity.** One person can run many proxy wallets, and bad wallets go broke and disappear. Consider
   clustering wallets by funding source.
5. **Correlated markets.** Neg-risk groups (multi-outcome events) share outcomes. Cap exposure per *event*, not
   just per market.
6. **Kelly inputs.** Edge estimates are noisy. Keep Kelly ≤ 0.25 until real out-of-sample data shows a stable
   edge. Even the oracle lost growth above Kelly 0.5.
7. **Operational risk.** Polygon/USDC custody, key security (use a dedicated wallet), API outages, and Polymarket
   rule changes (fees changed three times in 2026).
8. **Kill switch.** Before any real money: at least 4–8 weeks of paper trading, a hard drawdown stop (e.g. −10%),
   per-day loss limits, and a pre-registered rule for what counts as success.
