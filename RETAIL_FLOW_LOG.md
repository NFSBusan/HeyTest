# Research Log: Retail-Attention Options Strategy

## Goal (from user)

Find stocks that retail investors are piling into (or soon will be), predict short-term
moves, and trade options around those retail waves.

## Data (2026-10-16)

- `data/prices/`: 5 years of daily IBKR bars for 15 stocks plus SPY. See `data/prices/README.md`.
- `results/retail_scan_2026-10-16.md`: a manual social-media scan (web search only; counts aren't available).
- No retail-flow history yet: no Reddit mention history, no FINRA off-exchange volume, no options history.
  This study uses **volume spikes** as the stand-in for retail attention.

## Study 1: volume spikes on real prices (`scripts/retail_spike_study.py`)

Rules were fixed before running (see the script docstring):
- **Spike:** volume ≥ 3× the prior 60-day median, at most one spike per ticker per 10 days.
- **Up/down spike:** the spike-day move is beyond ±2 sd.
- **Trade:** enter at the next day's open; returns are SPY-beta-adjusted.
- **t-stats:** bootstrapped by calendar week, because spikes cluster in time.
- **Split:** dev 2021-10..2024-12; holdout 2025-01..2026-10, read once.

Extra trials: thresholds 2× and 5× (dev only), plus a volatility-ratio test added before the holdout was read.

### Dev (223 spikes)

| | n | next day | 5d | 20d | vol next 10d vs normal |
|---|---|---|---|---|---|
| up spike | 65 | +1.74% (t 1.84) | −1.46% (t −0.6) | −0.97% | **1.30× (t 3.8)** |
| down spike | 45 | −0.54% | −1.18% | −2.39% | **1.18× (t 2.4)** |
| flat spike | 113 | 0.0% | +0.98% | −2.82% | 0.97× |

### Holdout (79 spikes, read once)

| | n | next day | 5d | 10d | 20d | vol next 10d vs normal |
|---|---|---|---|---|---|---|
| up spike | 29 | −0.18% | +0.48% | +4.11% (t 1.55) | +3.67% | **1.40× (t 4.5)** |
| down spike | 15 | +1.75% | +0.89% | +6.05% (t 1.36) | 0.0% | 1.01× |
| flat spike | 35 | −0.40% | +1.21% | −1.72% | +0.37% | 1.02× |

### Conclusions

1. **Direction: no edge.** No return is significant in either period. The dev "next-day pop after an up
   spike" (+1.7%) did not repeat in the holdout (−0.2%), and the 2×/5× thresholds don't agree with each other.
   Price and volume alone don't predict where these stocks go next. That matches the earlier advice not
   to predict price direction.
2. **Volatility: a real, repeatable effect.** After an *up* spike (heavy volume plus a big up move,
   the typical retail pile-in), the stock moves about 30–40% more than usual over the next 10 days.
   This held in both dev and holdout (t 3.8 and 4.5). After down spikes it showed in dev only (n = 15 in the holdout).
3. **What this means for options:** don't sell premium straight after an up spike unless implied vol already
   prices in more than about 1.4× the normal movement. You'd be short exactly when the stock keeps swinging.
   Buying options (straddles, or calls for direction) after a spike pays only if implied vol hasn't already
   jumped by that much. **Which side wins depends on implied vol, which we don't have historically.**

### Caveats

- **Survivorship bias:** the tickers were picked because they're popular *today*, which tilts toward stocks
  that went up. That's why the 20-day "baseline" drifts. Abnormal returns reduce this but don't remove it.
- **Small sample:** 15 stocks; the crypto-miner/AI-cloud group (IREN, WULF, APLD, MSTR, NBIS) moves together.
- **Volume isn't retail:** it's only a stand-in. Institutions also cause volume spikes (earnings, index changes).

### Next steps

- **Record implied vol for these tickers every trading day from IBKR, starting now.** After a few weeks,
  test whether IV after an up spike is above or below the 1.4× realized volatility. That decides buy vs sell.
- Add a real retail measure: FINRA off-exchange volume (free; needs the network allowed) or Reddit mention history.
- Widen the universe to 100+ names, chosen by a rule (e.g. top options volume each month, point-in-time),
  to remove the survivorship bias.

## Study 2: implied vol after spikes (`scripts/iv_spike_study.py`), 2026-10-17

Data: DoltHub `volatility_history` (30-day IV) for GME, TSLA, NVDA, MU, PLTR (the mid-tier names
aren't in the dataset; MSTR stops in 2021). FINRA Reg SHO daily files for off-exchange volume, all 16 tickers.
IV readings are about 3 per week until 2024, then daily. Rules fixed before running (see the docstring).
No parameters were tuned.

**VRP** = ln(IV at the spike) − ln(realized vol over the next 21 trading days). Above 0 means options were
priced for more movement than happened. **Straddle** = buy a 21-day ATM straddle, hold to expiry, return on premium.

| 3× threshold | n | VRP | t | straddle net | t |
|---|---|---|---|---|---|
| all days (baseline) | 5014 | +0.067 | 5.8 | −0.7% | −0.3 |
| up spike | 19 | +0.112 | 1.1 | −31.6% | −1.7 |
| any spike | 47 | **+0.175** | **2.8** | **−22.1%** | −1.9 |
| 2× threshold, any spike | 110 | **+0.132** | **3.5** | −3.8% | −0.4 |

By ticker (2× threshold): GME VRP +0.32 (straddle −22%), NVDA +0.15, TSLA −0.02, MU −0.02 (straddle +21%), PLTR −0.06.
By period, up spikes: 2021–24 VRP +0.22 / +0.34 (3× / 2×), but 2025–26 VRP −0.06 / −0.09.

**Study 2b**: is a spike "retail-heavy" (FINRA off-exchange share above the stock's own 60-day median)?
After up spikes, realized vol is 1.33× normal for retail-heavy spikes (t 4.2, n 53) and 1.31× for the
others (t 3.7, n 40). Off-exchange share **doesn't separate** the two, and neither group predicts 5-day returns.

### Conclusions

1. **Options normally cost a bit more than the movement that follows** (VRP +0.07, t 5.8). That's the usual
   volatility risk premium: on an ordinary day a straddle buyer about breaks even before costs and loses
   slightly after them.
2. **After spikes, options get even richer.** Realized vol does jump (Study 1), but implied vol jumps more,
   so buying straddles after a spike lost money on average. Selling premium after a spike is the side the
   data favors.
3. **But it's mostly GME.** It holds for GME (and weakly NVDA). It's flat to reversed for MU, PLTR and TSLA,
   and up spikes in 2025–26 went the other way. With 47–110 spikes, five tickers and several cuts, this is
   a **lead, not a proven edge**.
4. **FINRA off-exchange share didn't add information** in this form.

### What to do next

- **Test "sell premium after a spike" properly** on meme-type names, with defined risk (iron condors or call
  spreads, not naked) and real option prices (ThetaData) including the skew. The straddle approximation
  ignores the skew and the bid/ask on each strike.
- **Get IV for the mid-tier names** (ASTS, RKLB, IREN, ONDS…): ThetaData's free tier, or daily IBKR snapshots
  from now on.
- **Don't trade on this yet.** Sample: 27 GME spikes drive most of the effect.

## Study 3: option strategy scan (`scripts/option_strategy_scan.py`), 2026-10-17

Ten standard ~30-day strategies on GME, TSLA, NVDA, MU, PLTR (SPY as benchmark), entered on every day with
an IV reading and held to expiry. Black-Scholes at 30-day IV with **no skew**, entry costs (4% of each leg's
premium, min $0.02, plus commission). 4,126 entries; 100 strategy/filter/ticker rows = 100 trials.
Results: `results/option_scan_pooled.csv`, `results/option_scan_by_ticker.csv`.

Pooled, all entries (return per trade on the capital each strategy ties up):

| strategy | mean/trade | t | win | worst | Sharpe |
|---|---|---|---|---|---|
| cash-secured short put (1 sd OTM) | +0.63% | **4.1** | 91% | −30% | 1.32 |
| covered call (1 sd) | +3.07% | 3.3 | 59% | −42% | 0.88 |
| stock (reference) | +4.30% | 3.4 | 55% | −45% | 0.86 |
| long call spread | +12.4% | 1.5 | 41% | −100% | 0.28 |
| long straddle | −0.1% | 0.0 | 38% | −100% | −0.01 |
| short straddle (naked, 20% margin) | −3.8% | −1.1 | 58% | −1837% | −0.12 |
| short strangle (naked) | −3.7% | −1.6 | 77% | −1779% | −0.26 |
| iron condor | −3.4% | −2.3 | 76% | −100% | −0.38 |
| short call spread | −4.8% | −3.2 | 82% | −100% | −0.69 |
| long put spread | −29.1% | −4.8 | 30% | −100% | −1.06 |

### Conclusions

1. **Bullish strategies "won" because the sample is a bull market in hand-picked winners** (MU up about 13×,
   NVDA 10×, PLTR 8×). Stock, long call spreads, covered calls, short puts winning and put spreads losing is
   mostly that, not an option edge.
2. **The one candidate that beats simply owning the stock on a risk-adjusted basis: cash-secured short puts**
   (Sharpe 1.32 vs 0.86; t 4.1, which survives 100 trials; positive in both 2021–24, which includes the 2022
   crash, and 2025–26). It beat the stock on GME, PLTR, MU and SPY, not on NVDA, and tied on TSLA. Selling
   puts when IV rank ≥ 0.7 raised the return per trade from 0.6% to 1.6% (n = 334, clustered). This matches
   the long-documented put-write premium (CBOE PUT index). Small and steady: about 0.6–1.6% a month on cash,
   with a worst month of −30%.
3. **Selling volatility without a view loses on these single stocks after costs** (short straddles,
   strangles and iron condors are all negative). The exceptions are SPY and GME. Big trending moves blow
   through the strikes.
4. **After spikes**, short straddles made +14.9% on margin (t 1.9), consistent with Study 2, but iron
   condors didn't (−1.4%). That's a lead, not significant.
5. **Covered calls ≈ owning the stock** (Sharpe 0.88 vs 0.86). They just trade upside for income; no edge.
6. Sharpe ratios for the filtered rows come from very few non-overlapping months and are unreliable (e.g. 10.7).

### Biggest caveats

- **No skew.** NVDA, MU and TSLA usually have normal put skew (OTM puts cost more), which understates the
  short-put result. Meme names like GME often have call skew (OTM puts cost less), so the GME short-put
  result is probably overstated. Needs real chain prices (ThetaData).
- **Survivorship.** The tickers were picked because they are popular now, i.e. after they went up.
- Physical settlement and assignment, margin calls and gap risk aren't modeled.

## Study 4: real option quotes, small/mid caps (`scripts/real_options_backtest.py`), 2026-10-18

Data: ThetaData end-of-day bid/ask for every strike, monthly expirations, Oct 2025 – Oct 2026, for 26
small/mid caps plus GME and NVDA. CRWV was skipped (no IBKR price file); BMEA, INDI and NAK have chains too
thin for most structures. Entry every day at the ~30-day monthly expiry, hold to expiry. **Sell at the bid,
buy at the ask** (a mid-price version is the optimistic bound). Legs with spreads over 50% of mid are skipped.
106 rows = 106 trials. Rules fixed before running. Trades: `results/real_options_trades.csv.gz`.

**Regime warning:** this was a violent, net-down year for these names. Their average 21-day forward return
swung between −33% and +26% depending on the entry month, and averaged −1.8%. One year is one regime.

### What held up

1. **Options on retail small/mid caps are priced well above the movement that follows.** ATM IV exceeded
   the next 21 days' realized vol by **+16% (log), t = 10**, across 5,123 ticker-days. After volume spikes
   it was +14% (t 2.8): no extra richness after spikes in this sample, unlike GME in Study 2.
2. **Buying options lost.** Long straddles −8% per trade, long call spreads −23% (bid/ask fills).
3. **Retail call skew is real in meme names.** 1 sd OTM calls were priced at a higher IV than ATM: AMC +21%,
   BMEA +24%, GME +19%, BBAI +9%, IBRX +10%. NVDA had normal (negative) call skew. Put skew was mild
   (0–13%) on most names.

### Strategy results (small/mid caps, bid/ask fills, mean return per trade on capital at risk)

| strategy | all days | IV rich (IV/RV20 ≥ 1.3) | after spike |
|---|---|---|---|
| cash-secured short put | −0.3% (t −0.4) | +0.3% | +1.8% (t 3.0, n 66) |
| short call spread 1/2 sd | +0.4% | +1.9% (t 2.4) | +3.2% (t 7.8, n 39) |
| short strangle (naked, 20% margin) | +4.5% (t 1.6), worst −747% | **+11.8% (t 4.0)**, worst −284% | +0.4% |
| iron condor | −3.8% (t −1.9) | +1.9% | too few |
| long straddle | −8.1% | −13.1% | −11.1% |
| stock (reference) | −1.8% | −0.7% | −0.9% |

GME + NVDA, for comparison: short puts +0.95% (t 6.4), put credit spreads +8.2% (t 17), consistent with Study 3.

### Reading it honestly

- **The edge is the volatility premium (point 1), not direction.** Selling options is the right side, but:
  - **naked** short strangles/straddles had −747% / −838% worst trades. One gap can wipe out a year;
  - **defined-risk** versions (iron condors, put credit spreads) often can't be filled: the far wings are
    illiquid or have >50% spreads (only 487 condors and 780 put spreads out of ~5,400 entries), and the
    bid/ask on the rest eats 1–3% per trade.
- **Short call spreads look best**, especially after spikes (+3.2%, worst −6%) and when IV is rich. But
  (a) only 39 after-spike trades over 20 weeks and 13 tickers, (b) the stocks *fell* 3.2% on average in those
  trades and the sample year was net down, so some of it is direction, not volatility premium, and (c) it
  lost badly on the names that rallied (RKLB −18%, NBIS −6%, RIOT −6%).
- **Cash-secured puts did NOT work on small/mid caps this year** (−0.3%), unlike the large-cap 5-year
  result (Study 3). In a falling year, short puts get assigned.

### Next

- **Delta-hedged** short-vol test (hedge daily with stock) to separate the volatility premium from direction.
  If short calls/strangles still pay after hedging, the edge is real; if not, it was the down year.
- More history: ThetaData free tier is 1 year only. Either a paid tier for 2020–2025, or keep recording.
- Get CRWV's IBKR price file and re-run.

## Study 5: delta-hedged (`scripts/hedged_backtest.py`), 2026-10-18

Same entries and bid/ask fills as Study 4, plus a daily close-to-close stock hedge (Black-Scholes delta at
that day's ATM IV) with 5 bps of cost on every share traded. This separates volatility premium from direction.
29 rows = 29 trials.

| small/mid caps | unhedged | **hedged** | hedged worst |
|---|---|---|---|
| short call spread, all | +0.4% | **−1.2% (t −4.2)** | −48% |
| short call spread, after spike | +3.2% | **−2.1% (t −2.4)** | −12% |
| short put | −0.3% | +0.2% (t 1.4) | −10% |
| short strangle, all | +4.5% | **+2.8% (t 3.2)** | −134% |
| short strangle, IV rich | +11.8% | **+5.8% (t 6.2)** | −46% |
| short straddle, IV rich | +2.4% | +5.1% (t 2.4) | −140% |
| long straddle | −8.1% | −14.9% (t −10.9) | −76% |

(returns per trade: strangle/straddle on 20% of spot as margin; spreads on max loss; puts on strike)

### Conclusions

1. **The short call spread fails the test.** Its profit was the stocks falling, not overpriced calls. Hedged,
   it loses. Study 4's "best candidate" was a disguised bearish bet. Drop it as a retail-premium strategy.
2. **The volatility premium is real and survives hedging:** delta-hedged short strangles made +2.8% per
   trade (t 3.2), and **+5.8% (t 6.2) when IV was ≥ 1.3× the last 20 days' realized vol.** The mirror
   image (hedged long straddles) lost −15% (t −11).
3. It worked on most of the retail meme names (SOUN, PLUG, BBAI, AMC, RGTI, JOBY, NAK, ACHR, SOFI, IREN,
   MARA) and **lost on the big-jump names** (ASTS, LUNR, NBIS, RKLB, QBTS, RIOT).
4. **Spike timing didn't help** once hedged (−3% to −4.5%). The useful timing signal is "IV rich vs recent
   movement", not "volume spike".

### Caveats

- **Margin:** brokers require far more than 20% for naked options on very volatile small caps. Real
  returns on capital will be lower, perhaps by half or more.
- **Hedging:** close-to-close hedging in a model; real hedging needs daily stock trades, and overnight gaps
  still hurt (hedged worst trade −134% of the assumed margin).
- One year, 26 hand-picked names, 135 trials across Studies 4–5.

## Study 6: realistic margin, practical hedging, capped-risk version (`scripts/practical_backtest.py`), 2026-10-18

Small/mid caps only, bid/ask fills, 1 sd short strangles at ~30 days. Returns are on the real capital tied up:
the Reg-T naked-option margin (20%/10% rule), or a "high-vol" margin at double that, plus 50% Reg-T margin on
the largest stock hedge. 21 rows + 4 cost-stress rows.

Per trade (IV rich = ATM IV / RV20 ≥ 1.3):

| | all days | IV rich | worst trade |
|---|---|---|---|
| strangle, hedged daily, Reg-T | +2.5% (t 4.6) | +3.4% (t 6.4) | −44% / −21% |
| strangle, hedged daily, high-vol margin | +1.8% | +2.6% | |
| strangle, hedged weekly, Reg-T | +5.5% (t 5.8) | +6.2% (t 5.4) | −79% / −60% |
| strangle, unhedged, Reg-T | +5.4% (t 1.4) | +15.4% (t 3.8) | −923% / −407% |
| iron condor, liquid wings (capped risk) | −4.9% (t −2.3) | +0.9% (t 0.3) | −100% |

Account simulation: IV rich only, one trade per stock per monthly cycle, equal capital, 103 trades over 11 months:

| | total | avg month | worst month | max drawdown | losing months |
|---|---|---|---|---|---|
| hedged daily, Reg-T | +51% | +3.9% | −4.4% | 4.4% | 2 of 11 |
| hedged daily, high-vol margin | +37% | +3.0% | −3.5% | 3.5% | 2 |
| hedged weekly, Reg-T | +105% | +6.9% | −4.6% | 4.6% | 1 |
| unhedged, Reg-T | +34% | +10.0% | **−71.6%** | **73%** | 3 |
| iron condor, liquid wings | +24% | +2.7% | −24.6% | 29% | 3 |

Hedge-cost stress (per trade, all days / IV rich; account total): 20 bps: daily +1.7% / +2.7%, +39%;
weekly +5.1% / +5.8%, +97%. 50 bps: daily +0.3% / +1.2%, +18%; weekly +4.3% / +5.0%, +80%.

### Conclusions

1. **The volatility premium survives realistic margin and costs** when the strangle is delta-hedged and
   only opened when IV is rich. Weekly hedging did better than daily and is far less sensitive to costs.
   One likely reason: these names often snap back within days, and daily hedging locks in losses on each
   swing. That's a hypothesis, not a proven fact.
2. **Unhedged is a disaster waiting to happen:** −72% in a single month (Jan 2026).
3. **The capped-risk version (iron condor) doesn't capture the premium.** The protective wings cost about
   as much as the premium collected, because of the skew.
4. **These results look too good to trust yet:** 11 months, one regime, hand-picked stocks, ~150 trials
   across Studies 4–6, end-of-day quotes (real fills at the bid on size may be worse), and no margin calls
   between entry and expiry are modeled. Treat this as the candidate to paper-trade, not as an expected return.

## Study 7: robustness / bias checks (`scripts/robustness.py`), 2026-10-18

Base = paper-trade rule (1 sd strangle, ~30 DTE, IV/RV20 ≥ 1.3, hedge every 5 days, sell at bid, Reg-T margin).
Results: `results/robustness_summary.csv`. 22 more trials (≈180 in total).

| check | per trade | t | account avg month | losing months |
|---|---|---|---|---|
| base | +6.2% | 5.5 | +6.7% | 1/11 |
| threshold 0 / 1.0 / 1.2 / 1.4 / 1.5 | +5.5 / 5.4 / 5.7 / 6.2 / 6.7% | 3.6–6.4 | 5.1–6.6% | 1–3/11 |
| strikes 0.75 / 1.25 sd | +4.0 / 6.0% | 2.6 / 4.0 | 5.1 / 7.9% | 3 / 1 |
| DTE 45 (DTE 20 = same expiries as base) | +6.5% | 6.1 | 6.2% | 1/11 |
| hedge every 1 / 3 / 10 days | +3.4 / 5.8 / 6.5% | 6.3 / 7.0 / 3.6 | 4.0 / 6.4 / 5.9% | 1 / 1 / 3 |
| **enter next day** (no same-close lookahead) | +6.0% | 5.1 | 8.0% | 1/11 |
| sell at bid − 25% of spread | +5.3% | 4.8 | 5.8% | 1/11 |
| only options that traded that day | +6.2% | 5.4 | 6.7% | 1/11 |
| **next day + worse fills + traded only (most realistic)** | **+5.2%** | **4.6** | 7.1% | 3/11 |
| first half (Oct–Mar) / second half (Apr–Oct) | +5.2% / +7.2% | 2.7 / 6.9 | 6.7 / 6.8% | 1/7, 0/5 |
| drop the 3 best tickers (AMC, ONDS, ACHR) | +5.2% | 4.2 | 6.5% | 1/11 |

- Placebo: random picks of the same number of trades averaged +5.5% (95% range 4.5–6.4%) vs +6.2% for
  IV-rich. Only 6% of random draws matched it, so **the IV/RV filter adds little and isn't statistically proven.**
- Path risk: 1.5% of trades (9/587) had an interim loss larger than the initial margin (a margin call /
  forced close). The 5th-percentile worst interim P&L was −25% of capital.
- Deflated Sharpe (160 trials, 112 non-overlapping trades): the probability that the true Sharpe is above 0
  is high, but with one year of data this test has little power against regime risk.
- 20 of 24 tickers were positive.

### Conclusions

1. **No sign of overfitting to a parameter.** Results form a broad plateau across thresholds, strikes,
   expiries and hedge frequencies. The edge doesn't hinge on the exact settings chosen.
2. **No lookahead or fill artifact.** Next-day entry and worse fills cut it from +6.2% to +5.2% per trade.
3. **The edge is the general volatility premium on these names, not the timing filter.** IV/RV ≥ 1.3
   barely beats random selection. Keep it for the paper trade (it was fixed in advance), but don't count on it.
4. **Weekly hedging was a post-hoc choice.** Hedging every 3, 5 or 10 days all beat daily, and 3 days was
   about as good with smaller tails. The gain from hedging less often is real in this sample but unexplained.
5. **What these checks can't fix:** one year, one regime (no 2008/2020-style market crash), a hand-picked
   universe, end-of-day quotes, and IBKR margin above Reg-T.

### Most realistic figure

Most realistic per-trade return ≈ +5% on margin. With the paper-trade sizing (≈13% of the account in margin,
capped so a 50% gap in every name costs ≤ 25%), that's about **+0.9% a month on the whole account (≈ 11%/yr)
before haircuts**. Halving it for regime risk, selection bias and real-world frictions gives a planning figure of
**≈ 5–6% a year**, with a real chance of a losing year if a market-wide crash hits while positions are open.

## Study 8: searching for improvements (`scripts/improvements.py`), 2026-10-18

Literature checked: Goyal & Saretto (2009) cross-sectional IV−RV sorts; the short-dated/retail VRP evidence;
earnings IV-crush studies (gap risk usually offsets the crush, and we have no earnings dates). IBKR pays about
3.1% on USD cash above $10k (BM − 0.5%). Realistic execution throughout (next-day entry, bid − 25% of spread,
traded legs only, weekly hedge, Reg-T margin). A variant "counts" only if it beats the base in BOTH halves.

| variant | 1st half | 2nd half | full | verdict |
|---|---|---|---|---|
| V0 base (1 sd strangle, IV/RV ≥ 1.3) | +4.0% | +6.7% | +5.3% (t 4.6) | — |
| no filter, all days | +5.6% | +3.7% | +4.7% | mixed |
| V1 top third by IV/RV each day (Goyal–Saretto style) | +5.2% | +5.6% | +5.4% (t 5.6), 2.2× the trades | similar per trade, more positions |
| V2 ATM straddle | −2.9% | +2.4% | −0.3% | worse |
| V3 sell only the richer side | +3.0% | +4.0% | +3.5% | worse |
| V4 short-dated (~14 DTE) | +9.6% | +2.2% | +5.3% | unstable, fails |
| V5 Goyal–Saretto long-short (long leg: hedged long straddles) | −1.8% | −0.5% | −1.2% | fails (buying vol loses after costs) |
| V6 strangle + 2.5 sd call as squeeze insurance | +3.1% | +0.9% | +2.1% (vs +5.1% unprotected on the same trades) | insurance costs more than it saves |

**Tail risk, measured properly.** Historical replay of 5-day joint moves (2021–2026, IV +50% shock, each stock's
first 65 trading days skipped): meme squeezes dominate (BBAI +301% in 5 days in Jan 2023, QBTS +245%, GME and APLD
about +200%). The first version of this stress was wrong twice: it included LUNR's pre-options SPAC week (+717%),
and it averaged over all 24 names, which understates a single-name squeeze in an 8-position book. Fixed version:
1,000 random books, sized so the 90th-percentile book's worst week costs ≤ 20% of the account.

| positions in the book | unlucky worst week (per $ margin) | margin use | est. annual return incl. 3% interest (in-sample) |
|---|---|---|---|
| 4 | −225% | 9% | 11.0% |
| 8 | −145% | 14% | 15.6% |
| 12 | −113% | 18% | 19.4% |
| 16 | −96% | 21% | 22.5% |
| 24 | −76% | 26% | 28.3% |

### Conclusions

1. **No trade-structure tweak reliably improves the return.** The famous academic long-short version fails
   here after costs, which matches published doubts about transaction costs (Do, Foster & Gray 2016).
2. **Two real improvements, neither a "pattern":**
   - **interest on idle cash** (~85% of the account): +2.6%/yr that was missing from earlier figures;
   - **more simultaneous positions.** The binding constraint is single-name squeeze risk, so diversification is
     the main lever: 8 → 16 positions raises safe margin use from 14% to 21% (est. 15.6% → 22.5%/yr before haircuts).
     The ranking rule (V1) gives about twice as many positions at the same per-trade return.
3. Getting 16 positions needs a bigger universe of retail-heavy names with liquid options, chosen by a rule.
   That also removes the hand-picking bias.
4. **Updated realistic figures (after the same ~50% haircut as Study 7):** 8 positions ≈ 8%/yr; 16 positions ≈ 11%/yr.

## Study 9: rule-based universe (`scripts/universe_study.py`), 2026-10-25

Universe: top 80 by FINRA off-exchange dollar volume in Sept 2025, optionable, $2–$150, no S&P 100
(`scripts/build_universe.py`), minus 9 ETFs that slipped through the exclusion list = 71 stocks. Chosen before
the test year, so there's no hindsight in the stock selection. 54 of the 71 were never part of the hand-picked list:
a true out-of-sample test. Realistic execution as in Studies 7–8.

| per trade | 1st half | 2nd half | full |
|---|---|---|---|
| all 71, V0 (IV/RV ≥ 1.3) | +1.6% | +4.9% | +3.2% (t 4.3) |
| all 71, no filter | +2.2% | +2.2% | +2.2% (t 5.2) |
| all 71, top third (V1) | +2.6% | +2.7% | +2.6% (t 4.9) |
| retail-heavy half / less-retail half (V0) | −0.3% / +3.3% | +6.4% / +3.6% | +2.9% / +3.4% |
| **NEW names only (out-of-sample)** | +1.5% | +4.7% | **+3.0% (t 4.2)** |
| original names | +1.9% | +5.7% | +3.6% (t 2.7) |

52 of 71 stocks were positive. Worst: QBTS −17%, LRCX −14%, WDC −12%, HIMS −12%, CRCL −11% per trade.

Account level (cluster cap of 3 per sector per cycle; stress = an unlucky book's worst common week over the last year
+ one position taking the worst single-name squeeze ever seen, −722% of margin; IV +50%; ≤ 20% loss allowed;
3% on idle cash). The first stress version was distorted by uneven history lengths and was replaced; both runs are disclosed.

| | positions/month | return on margin/month | margin use | backtest annual | planning (÷2) |
|---|---|---|---|---|---|
| all 71, V0 | 22 | 4.0% | 30% | 17.7% | **8.8%** |
| all 71, top third | 45 | 3.3% | 43% | 20.7% | **10.3%** |
| retail-heavy half, V0 | 12 | 4.7% | 20% | 14.8% | 7.4% |
| new names only, V0 | 19 | 3.2% | 30% | 14.2% | 7.1% |
| (before: 26 hand-picked names, 8 positions) | 8 | | 14% | 15.6% | 7.8% |

### Conclusions

1. **The edge holds out of sample:** +3.0% per trade on the 54 stocks that played no part in designing it, and
   positive in both halves. It's smaller than on the hand-picked names (+5.2%), as you'd expect once hindsight is removed.
2. **More stocks → much bigger safe share:** margin use rises from 14% to 30–43%, because a single squeeze matters
   less in a 22–45 position book. That's what lifts the account figure, not a better per-trade edge.
3. **"Retail-heavy" doesn't add extra premium** within this universe (the more retail half did no better). The edge is
   the general overpricing of options on high-volatility, heavily traded stocks.
4. **The top-third ranking (V1)** was the most stable across halves (+2.6% / +2.7%) and gives the most positions, so it
   has the best planning figure (~10%/yr). Choosing it now is a choice among a few disclosed variants.
5. Still one year of data; the stress test is an estimate; IBKR margin on volatile names may exceed Reg-T.

## Study 10: exit rules and the risk budget (`scripts/exit_rules.py`), 2026-10-25

Hedged 1 sd strangles, top third by IV/RV each day, rule-based 71-stock universe, realistic execution; buy-backs
at ask + 25% of spread. 3,327 trades. "Per 30 days" assumes freed capital is re-deployed at the same rate.

| rule | days held | per trade (1st / 2nd half) | per 30 days (1st / 2nd half) | worst |
|---|---|---|---|---|
| E0 hold to expiry | 32.7 | +2.5% / +2.7% | +2.4% / +2.7% | −110% |
| E1 take profit at 50% | 21.2 | +0.3% / +0.3% | +2.6% / +2.3% | −105% |
| E2 stop at 2× credit | 31.1 | +1.8% / +2.3% | −3.0% / −2.0% | −110% |
| E3 both | 19.9 | −0.4% / 0.0% | −2.4% / −2.1% | −105% |
| sensitivity: take profit at 25% | 15.3 | −1.2% / −1.1% | +0.5% / −0.4% | −95% |
| sensitivity: take profit at 75% | 26.6 | +1.6% / +1.4% | +2.9% / +3.0% | −110% |
| sensitivity: stop at 3× | 32.1 | +2.2% / +2.6% | −0.1% / +1.1% | −110% |

### Conclusions

1. **Stop-losses hurt.** They buy back right after spikes that would mostly have reverted, which matches the
   weekly-beats-daily hedging result. They also don't reduce the worst trade: gaps jump past the stop.
2. **Taking profit at 50% doesn't beat holding** once buy-back costs are paid (it isn't better in both halves).
3. Taking profit at 75% was better per 30 days in both halves (+2.9% vs +2.5%), but it was a sensitivity row, not a
   pre-selected rule, and it relies on the re-deployment assumption. It's a hint to paper-trade, not an adopted rule.
4. **The remaining lever is the risk budget, which is a choice, not an edge.** Top-third version, return on margin
   3.3%/month, 3% on idle cash:

| worst stress week allowed | margin use | backtest annual | planning (÷2) |
|---|---|---|---|
| 15% of account | 32% | ~16% | ~8% |
| 20% (current) | 43% | ~20.5% | ~10% |
| 25% | 54% | ~25% | ~12.5% |
| 30% | 65% | ~30% | ~15% |
