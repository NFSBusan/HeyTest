# Phase 1: Challenging the Brent–WTI Mean-Reversion Idea

Status: **written before any model is fitted.** Part B numbers come from a stated model with
assumed parameters, not from market data. No real prices were available in this session
(the network policy blocks FRED, EIA, Yahoo and Stooq), so Part A of
`scripts/oil_challenge.py` has not run yet.

Run everything from the repo root:

```bash
python scripts/oil_challenge.py
python -m pytest -q tests/test_oil.py
```

Outputs go to `results/oil/`.

---

## 1. Why should a Brent–WTI mean-reversion strategy generate returns?

A trading return needs someone on the other side who is willing to lose to you, on average.
Here are the candidates, from most to least plausible:

| Source of return | Who pays | Plausible for us? |
|---|---|---|
| **Risk-transfer premium.** Hedgers (US producers selling WTI, refiners buying Brent-linked crude, index funds rolling) push the spread around and pay to unload risk. | Hedgers and index rolls | Possible but small. The premium goes to whoever holds the risk when nobody else wants it, often during stress, and that is when spreads can widen a lot further. |
| **Physical arbitrage.** If the spread goes above the cost of moving crude (pipeline from Cushing/Permian to the Gulf, plus a tanker to Europe), cargoes flow and the spread closes. | Nobody. This is a no-arbitrage bound. | **No.** The traders who capture it own pipeline capacity, storage and ships. A paper trader without those can't take the arbitrage. They can only bet that someone else will. |
| **Slow reaction to public information** (EIA inventories, Cushing stocks, export data, freight rates) | Slower traders | Unlikely. These are among the most-watched numbers in commodities. |
| **Behavioral overreaction** to headlines | Noise traders | Unproven, and it would be competed away quickly. |

**What the physics implies.** Since the US export ban ended (18 Dec 2015), and since WTI
Midland joined the Dated Brent basket (deliveries from June 2023), the spread is held inside
a **band** set by transport costs. Inside the band, moves are mostly noise with no strong
force pulling them back. Outside the band, physical arbitrage closes the gap, but that is not
our edge. A Z-score treats the spread as if it had one fixed equilibrium. In reality the
equilibrium itself shifts whenever infrastructure, freight or regulation changes.

**The 2011–2013 example.** A glut at Cushing pushed the spread above $25 (its long-run level
had been near zero). A Z-score model fitted on 2000–2010 would have shorted the spread at +3σ
and been underwater for years. Reversion only came once pipelines were reversed or built. The
spread did revert in the end, but the timing depended on construction schedules, and those
aren't in the price data. **The largest deviations are usually structural breaks, not
temporary noise.** The Z-score bets biggest exactly when it is most likely to be wrong.

## 2. The key objection: reversion is probably already priced in

Most backtests skip this point.

- Spot Brent–WTI series (FRED `DCOILBRENTEU` / `DCOILWTICO`) are **not tradeable**. Futures
  are.
- If the front spread is unusually wide and everyone expects it to narrow, the **deferred**
  futures spread will already be narrower. A futures position earns only the gap between
  **actual** convergence and the convergence **the curve already implies**.
- So the right null hypothesis is not "the spread is a random walk". It is "**the futures
  spread curve is an unbiased forecast of where the spread goes**". A spread that is strongly
  mean-reverting can still offer zero edge.

Part B puts a number on this. Let `priced_in` be the share of the expected convergence that
is already in the prices. Take a trader who knows the true mean and speed of the spread
(perfect knowledge), entering at 2σ wide with a 30-day half-life:

| Structure (EV as % of debit, after 5% round-trip cost) | priced_in = 0 | 0.5 | 0.9 | 1.0 |
|---|---|---|---|---|
| Spread put, ATM (if a listed spread option existed) | +161% | +68% | +7.7% | −5% |
| WTI call, ATM | +14.9% | +4.7% | −3.1% | −5% |
| WTI call + Brent put | +14.9% | +4.7% | −3.1% | −5% |

**Test 1 on real data is therefore a Fama-style unbiasedness regression on the Brent–WTI
futures curve**, not a stationarity test on spot prices. Does the realized change in the
front spread beat what the curve forecast, out of sample, with HAC/overlap-corrected errors?
If it doesn't, stop here.

## 3. Even with a perfect signal, outright options are weak at capturing it

This is the payoff-mapping problem from the brief. The perfect-signal model (`oilmr/options.py`):

- Spread `S` is an OU process. A common oil factor `C` moves both legs equally.
  WTI = W₀ + C + φ·(S₀ − S), Brent = W₀ + S₀ + C − (1−φ)·(S₀ − S), so Brent − WTI = S exactly.
- φ is the share of convergence that comes from WTI moving. Nobody knows φ ahead of time.
- Option sellers know each leg's true variance. Strikes and prices use their forward.
  Every leg pays a 5% round-trip cost.
- **Assumed** parameters (to be replaced by real estimates): WTI $70 and 35% vol, daily spread
  sd $0.60, half-life 30 trading days, mean $4, entry at +2σ stationary (S₀ ≈ $9.58),
  21-day horizon, priced_in = 0.

Results (`results/oil/option_mapping_base.csv`), per barrel:

| Structure | Debit | E[P&L] | E[P&L] / debit | Max loss | P(profit) | Independent trades for t = 2 |
|---|---|---|---|---|---|---|
| WTI call ATM | 2.86 | +0.43 | +14.9% | 3.00 | 39.5% | **~430** |
| WTI call spread (ATM / +1σ) | 2.26 | +0.21 | +9.1% | 2.43 | 42.5% | ~800 |
| Brent put ATM | 2.86 | +0.43 | +14.9% | 3.00 | 39.5% | ~450 |
| WTI call + Brent put | 5.71 | +0.85 | +14.9% | 6.00 | 50.5% | ~100 |
| Spread put ATM (hypothetical) | 0.88 | +1.42 | +161% | 0.92 | 71.0% | ~7 |
| Futures spread (not allowed, reference) | — | +2.15 | — | unlimited | — | ~4 |

What this shows:

1. **Signal dilution.** Over a month, a WTI option's value is driven almost entirely by the
   flat oil price (≈$7 sd) and very little by the spread (≈$2 sd). The perfect signal turns
   into a **~15% edge buried in a P&L whose sd is about 160% of the debit**.
2. **It can't be validated.** With roughly 430 independent trades needed to separate a
   *perfect* signal from luck, and 2σ events with a 30-day half-life happening only a few
   times a year, it would take more than a century of data. A real, imperfect signal would
   need even more.
3. **φ matters and is unknown.** If convergence comes entirely from Brent falling (φ = 0),
   the WTI call has **zero edge, minus costs, in every cell of the sweep**. Pairing the WTI
   call with a Brent put removes the dependence on φ, but it pays two premiums.
4. **Debit spreads make the signal-to-noise ratio worse here.** They cap the upside, and with
   a small drift the upside is where the edge sits.
5. **Only an option written on the spread itself** keeps the signal intact. Whether such an
   option is listed, liquid, and settles in a way our risk rules allow has to be checked
   (§5). If none exists, the options version of this strategy probably isn't viable.

Sweep (`results/oil/option_mapping_sweep.csv`): EV turns negative for the WTI call once the
half-life reaches ~250 days at φ = 0.25, and at φ = 0 everywhere. With priced_in ≥ 0.9, 254 of
256 outright-option cells are negative. The two exceptions (WTI call, φ = 1, entry at 3σ,
half-life 10–30 days) make +0.5% to +1.5%, which is noise.

## 4. Other ways a naive backtest will mislead

- **Contract-month mismatch.** Brent and WTI front months roll on different dates. A naive
  "front vs front" series mixes delivery months for part of every month and creates fake
  spread jumps that "revert". Use same-delivery-month pairs with an explicit roll rule.
- **Settlement timing.** ICE Brent settles at 19:30 London and NYMEX WTI at 14:30 New York.
  These are the same instant except during the weeks when US and UK daylight-saving dates
  differ. Check this; don't assume it.
- **The 20 Apr 2020 negative WTI print** (spot −$36.98 on FRED). It was a delivery and storage
  squeeze specific to physically delivered CL. Log-price models break on it, and it is a live
  reminder of what physical-delivery exposure means.
- **Full-sample Z-score and hedge ratio are look-ahead.** Every parameter has to be estimated
  on a rolling or expanding window.
- **Structural breaks** (2010–11 Cushing, Dec 2015 export repeal, 2020 COVID, 2023 Midland in
  Dated Brent). Test for stability. Don't pick break dates after looking at the chart and then
  report in-sample results.
- **Multiple testing.** Every threshold, window or lookback tried counts as a trial and has to
  go into a deflated Sharpe or White's Reality Check.

## 5. Instrument and risk-rule screen (to verify against current rulebooks)

These need to be **checked against current CME/ICE rulebooks and the broker's rules before any
structure is accepted.** Nothing below has been verified in this session.

| Instrument | Concern under your rules |
|---|---|
| NYMEX WTI options (LO) | American style, exercise into **physically delivered** CL futures. An in-the-money long option auto-exercises at expiry into a futures position that is close to delivery. A short leg in a debit spread can be **assigned early**, which leaves an uncovered futures leg with margin calls. Reject unless every position is closed before expiry under a hard, tested operational rule, and even then early assignment on short legs remains. |
| CME European-style or cash-settled WTI options (e.g. European and average-price contracts) | Better fit: no early assignment, and cash settlement removes delivery risk. Check the settlement mechanics, liquidity and strike coverage. |
| ICE Brent options | Exercise into ICE Brent futures, which are **cash-settled** against the ICE Brent Index, so there is no barrel delivery. Exercise still creates a futures position. Check the style and the automatic-exercise rules. |
| Brent–WTI spread options (any exchange) | The only instrument that maps directly to the signal. Check whether any are listed now, how much open interest they have, and the bid/ask. |
| USO / BNO ETF options | Equity options, American style, physically settled in shares. Exercise requires the cash to buy 100 shares per contract, and short legs can be assigned early. The ETFs have roll costs and have changed structure in the past (USO in 2020). BNO options are thin. |

Max-loss check for any accepted structure: max loss = Σ(long premiums) − Σ(short premiums)
+ all fees, × multiplier (1,000 bbl for CL, Brent and LO). This holds only if **no**
exercise, assignment or settlement path can create a position outside the structure. Tests in
`tests/test_oil.py` confirm that every modeled structure has a non-negative payoff. That
covers the payoff maths only, not the exchange mechanics above.

## 6. Pre-registered go / no-go gates

| Gate | Test | Kill if |
|---|---|---|
| 0 Data | Daily settlements for the same delivery month: ICE Brent and NYMEX CL, front plus at least 3 deferred months, 2008 onward. Spot series are allowed only for descriptive work. | Can't obtain them |
| 1 Stability | Rolling ADF and half-life, break tests, regime table (`oilmr/diagnostics.py`) | Half-life unstable by more than ×3 across windows, or no reversion after 2015 |
| 2 Beyond the curve | Fama regression: does the realized change in the front spread beat the curve-implied change, walk-forward, HAC errors | Slope not significantly beyond the curve out of sample |
| 3 Costs | Futures-spread backtest as a reference (not traded), with realistic bid/ask, fees and rolls | Net Sharpe < 0.5 out of sample, or deflated Sharpe not significant |
| 4 Mapping | Re-run Part B with the estimated half-life, φ distribution, priced_in and real option quotes | No defined-risk, rule-compliant instrument with positive EV **and** a trade count for t = 2 that fits inside the available history |
| 5 Paper | Forward paper trading, size fixed in advance | Results outside the backtest's 90% band |

## 7. Current verdict

- **Prior: low.** For a participant with no physical assets, a simple Z-score on Brent–WTI
  most likely has no edge left after costs. The physical arbitrage belongs to firms with
  infrastructure. The biggest deviations are structural. Forecastable reversion is probably
  already in the futures curve.
- **Even if Gates 1–3 pass**, outright WTI or Brent options lose most of the signal in
  flat-price noise (Part B). The plan is viable only if a liquid, cash-settled,
  European-style spread option or an equivalent exists. That needs checking.
- **Recommendation:** don't build the full modeling stack yet. Get futures-curve data and run
  Gate 2. It is cheap and decisive. If Gate 2 fails, report that the strategy isn't supported,
  rather than tuning thresholds until a backtest looks good.

## Data access needed

To run Part A and Gates 0–2, one of:

- Add `fred.stlouisfed.org` (spot, descriptive only) to the environment's allowed domains.
- Better: a futures-settlement source covering ICE Brent and NYMEX CL deferred months
  (e.g. a Nasdaq Data Link / CHRIS-style continuous series, or exchange data).
- The IBKR connector in this session can return price history for listed contracts. Its
  coverage of expired contracts is limited, so it can't replace a long history.
