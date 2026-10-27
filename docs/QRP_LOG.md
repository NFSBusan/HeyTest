# qrp build and review log

Newest entries at the bottom. All runs use simulated data (`provider: synthetic`).

## Build (phases 0-5)
Built in order: rules layer (config gates, DuckDB registry with trial counting and dedup, holdout vault,
cost model) → data engine (Tiingo/Alpaca/CSV/synthetic, Parquet store, quality checks, panel) → indicators
and strategy blocks → Numba engine and metrics → walk-forward, PBO, SPA/RC, DSR, robustness suite, gates →
report and dashboard → Optuna → Claude proposal loop → paper trader → CLI and docs.

## Review cycle 1: tests (76 tests, all offline)
Bugs the tests caught and fixed:
1. **Engine crash**: the per-trade buffer was sized for T·N/2 trades; positions that flip daily overflowed it
   and Numba aborted the process. Now T·N + N.
2. **Tiingo parser** produced duplicate `open`/`close` columns (raw and adjusted). Now built explicitly.
3. Two of my own tests had wrong expectations (buy-and-hold entry day; refused holdout attempts aren't looks).

## Review cycle 2: code reading
4. **Cost model ignored weight drift**: rebalancing back to target weights was free. Turnover now includes drift.
5. **Stale adjusted prices**: after each dividend, providers rewrite adjusted history, so appending only
   new days leaves a jump at the seam. The store re-fetches a 14-day overlap and reloads full history if it moved.
6. A forced second holdout look found no candidates (statuses had changed after the first look). Fixed.
7. Null-world false "alpha": the synthetic benchmark wasn't the market factor and assets had different drifts,
   so "alpha" existed in the no-edge world. The benchmark is now the market and drifts are equal when edge=none.

## Review cycle 3: full-size runs (3 worlds × 3,990 configs, ~1m50s each on 4 cores)
8. **Deflated Sharpe could never accept a real edge.** It used the cross-trial spread of Sharpe ratios as
   "noise"; in the planted-trend world most of that spread is real, so the bar was a 1.9-3.4 annual Sharpe
   and 0 of 15 strong candidates passed. Now: null sampling variance and an effective number of independent
   trials. Result: planted trend 10/15 pass, no-edge world still 0/15.
9. The first effective-N version (on raw returns) said 3,990 configs ≈ 11 bets, because shared market
   exposure dominates the correlations. The best no-edge candidate then reached DSR 0.93, too close to 0.95.
   Effective N is now computed on alpha returns (≈ 22 bets); the no-edge maximum is 0.87, and every no-edge
   candidate fails at least 3 independent gates (asserted in tests).

## Review cycle 4: end-to-end CLI and UI
- Holdout → second look refused → promote → paper status: works. Planted-edge strategies held up on the
  holdout (dev Sharpe 1.4-1.7 → holdout 1.3-1.7).
- compare-samplers (150 trials, planted trend): grid 1.21, random 1.28, TPE 1.56. It refuses to run on real data.
- Dashboard in Chromium (light/dark/phone): no console errors. Fixed an 8 px horizontal overflow on phones
  and a 17,000 px page (the table now scrolls on its own).

## What's left that can't be checked here
- Live data: Tiingo/Alpaca hosts are blocked from this cloud session. Tested only against mocked responses.
- Live Claude calls: no API key here. The request shape follows the current SDK docs; tested with a mocked client.
- Real-market results: none yet. The next step is `qrp fetch` + `qrp research` with `provider: tiingo`.
