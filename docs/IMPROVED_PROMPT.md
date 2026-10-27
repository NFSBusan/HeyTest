# Strategy 2, revised prompt

A tightened version of the original request. The main changes: a hypothesis-first search with a fixed
trial budget, pass/fail rules fixed before testing, a locked holdout, alpha-based edge tests, and Claude
limited to proposing structured experiments. The platform in `qrp/` implements this prompt.

---

Act as a quantitative researcher and engineer. Help me build a research platform whose job is to
**find out honestly whether any trading strategy has an edge after costs**, not to produce impressive
backtests. "No strategy passed" is an acceptable and likely result.

**Scope for version 1.** One market (US ETFs), daily bars, long-only, about $10k capital, free data,
runs on a laptop. Expand markets, frequency or infrastructure only after a strategy has passed the final
holdout and at least 3 months of paper trading.

**Rules fixed before any test.**
1. Write the pass criteria (gates) into a config file before running anything: minimum trades, deflated
   Sharpe ≥ 0.95 on alpha, PBO ≤ 0.5, SPA p ≤ 0.10, walk-forward OOS Sharpe, max drawdown, Sharpe at 2x costs,
   parameter-neighbour stability, yearly consistency, cost drag. Changing a gate after seeing results starts
   a new, recorded study.
2. Count every configuration ever tested on the same data, across studies, optimisers and AI proposals,
   and use that count in the multiple-testing correction.
3. Split the data into warm-up, development and a final holdout with an embargo gap. Code that searches or
   tunes must not be able to read holdout prices. The holdout is opened once, with a logged list of frozen
   candidates; a second look marks it as contaminated in every report.
4. Measure edge as alpha (returns minus what the strategy's market exposure would have earned), not as
   Sharpe versus zero.

**Search.** Group strategies into families with a stated economic rationale (trend, short-term mean
reversion, cross-sectional momentum/reversal). Give each family a fixed trial budget. Spend it by random
sampling, grid order or Bayesian optimisation (Optuna) that maximises a consistency score across time
blocks, not peak Sharpe. Never re-test an identical configuration. Pick candidates as the best of distinct
return clusters, not the top N near-duplicates.

**Realism.** Signals at the close trade at the next open. Costs = commission + half-spread + square-root
market impact from lagged volatility and volume, including the turnover from rebalancing drifted weights.
Borrow costs for shorts. Stress costs at 2x and 3x; test capacity at 10x and 100x capital. Handle splits,
dividends (including re-downloading history when adjustments change), missing days, and point-in-time
universe membership.

**Validation per candidate.** Walk-forward of the whole selection procedure, PBO (CSCV), Hansen SPA and
White's Reality Check, deflated Sharpe, parameter neighbours, cost stress, capacity, block-bootstrap Monte
Carlo, bull/bear and high/low-vol regimes, calendar-year consistency, and comparison with buy-and-hold.

**Claude's role.** Propose at most a few experiments at a time as structured JSON over the registered
building blocks, with a hypothesis, a falsification condition and expected failure modes. The platform
validates and sizes each proposal against the remaining budget, and I approve it before it runs. Claude
never sees holdout results, never executes free-form code, and only interprets numbers that the platform
produced; any number in its commentary that isn't in the report tables gets flagged. There is a monthly
spending cap.

**Outputs.** A Markdown report and a static HTML dashboard generated from the results database, with
in-sample, walk-forward OOS, holdout and paper results labelled separately. Filters by family, block type,
drawdown and trade count.

**Paper trading.** Only for holdout passers. A daily job fills yesterday's orders at today's open, marks
to market, enforces daily-loss and drawdown kill limits, refuses stale data, honours a kill-switch file, alerts
when the rolling return falls below the backtest's 5th percentile, and is verified to reproduce the backtest
on replayed data. No real-money trading without my explicit approval.

**Engineering.** Modular Python with tests for lookahead (indicators and weights unchanged when the future
is altered), cost arithmetic, holdout locking, trial counting, and an end-to-end check on simulated data: a
world with no edge must produce zero passing strategies, and a world with a planted edge must be detected.
Keys only from environment variables. Verify provider and cloud prices before recommending paid services.
