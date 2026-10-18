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
