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
