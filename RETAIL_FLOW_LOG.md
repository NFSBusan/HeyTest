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
