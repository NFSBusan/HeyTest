# Retail attention scan, 2026-10-16 (manual, from web search)

Built from news and tracker pages found by web search. The cloud session can't reach Reddit,
Stocktwits, X or the mention trackers directly, so these are **not counted mentions**. They are
names that several recent sources independently flag. Re-run with live data (rflow/fetch.py, once
the network allows it) before using any of this.

| Tier | Ticker | Where it's showing up | Note |
|---|---|---|---|
| Heavy, broad | TSLA | WSB, X, Stocktwits (msg volume +75% in 24h), Robinhood most-held | Roadster event; Q3 earnings Oct 21; sentiment turned bearish |
| Heavy, broad | NVDA | WSB, Robinhood most-held, Stocktwits top dozen | Slipping, "momentum traders cautious" |
| Heavy, broad | MU | WSB + X weekly top 5, Stocktwits #3 | AI memory / earnings chatter |
| Heavy, broad | GME | Stocktwits top 5, social trending lists | Perennial meme name |
| Heavy, broad | SPCX (SpaceX) | Stocktwits, Robinhood lists, Tesla merger talk | New listing, very retail-heavy |
| Heavy, broad | ASTS | Stocktwits #4 | Space theme |
| Rising | MSTR | WSB momentum heating up; +16.4% on Friday | Bitcoin proxy |
| Rising | CRWV, NBIS, IREN, APLD | WSB/X weekly top 5 (CRWV); Stocktwits poll: 57% of ~4,000 voters plan to buy the dip | Neocloud selloff + dip buying |
| Rising | RKLB | WSB + X weekly top 5 | Space theme |
| Rising | WULF | Stocktwits sentiment turned bullish | Biggest weekly drop since Nov |
| Small-cap, speculative | ONDS | Stocktwits "extremely bullish", high short interest, "epic like GameStop" | Down ~35% YTD |
| Small-cap, speculative | IBRX | Stocktwits message spike, sentiment bearish | FDA win, founder trial talk |
| Small-cap, speculative | VEEA, BEAT, ALEC, MI, BLGO | Small-cap Reddit subs (pennystocks, Shortsqueeze, smallstreetbets): top movers by mention change, Oct 6 | Tiny counts (≤20 mentions); a few posts move the ranking |
| Squeeze setups (short interest, not mentions) | BDRX, BMEA, NAK, INDI | Short-squeeze screens, short ratio ~45–75% | Candidates *before* retail arrives |
| ETFs | SPY, QQQ, SOXL | Stocktwits #1/#2, WSB | Not single-stock retail signals |

Backdrop: single-stock options volume is back near 2021 meme levels (~$700B daily notional, per a WSJ
report on Oct 12), and call prices are above that era's levels. The AAII bull share is 40.3%, the highest
since mid-July.

## Platform gaps

- **Reddit:** WSB coverage is decent. For r/stocks, r/investing and r/Daytrading I found no ticker rankings.
- **X/Twitter:** Covered only through news roundups. Real cashtag counts need the X API (paid).
- **Facebook / Instagram / TikTok:** No public ticker-level data. Meta's research API is restricted,
  and TikTok's research API is limited to academics. Treat these as not measurable for now.

## Sources

- https://www.tradingview.com/news/benzinga:a0e90f9d8094b:0-tsla-mu-crwv-and-more-5-stocks-investors-couldn-t-stop-buzzing-about-this-week-corrected/
- https://www.marketscreener.com/news/social-buzz-wallstreetbets-stocks-mixed-premarket-friday-tesla-to-advance-coreweave-to-decline-ce785cdadb8ffe26
- https://timothysykes.com/news/strategy-inc-mstr-news-2026_10_16
- https://www.timothysykes.com/news/nvidia-corporation-nvda-news-2026_10_15/
- https://altindex.com/stocktwits-stocks
- https://stocktwits.com/news-articles/markets/equity/tsla-stock-roadster-event-elon-musk-greatest-product-demo-of-all-time/cZxWmPlRBA3
- https://stocktwits.com/news-articles/markets/equity/onds-stock-heads-for-third-red-week-but-retail-stands-firm-on-drone-defense-opportunity-will-be-epic-like-game-stop/cZx0uiJRBsb
- https://stocktwits.com/news-articles/markets/equity/crwv-nbis-iren-stocks-extend-slide-overnight-retail-eyes-the-dip-as-core-weave-ceo-says-market-acting-irrationally/cZxWLMRRBqw
- https://stocktwits.com/news-articles/markets/equity/wulf-stock-set-for-biggest-weekly-loss-of-2026-morgan-stanley-says-shares-are-mispriced/cZxhbweRBr7
- https://stocktwits.com/news-articles/markets/equity/ibrx-stock-anktiva-fda-win-retail-cautious-ahead-of-founder-trial-talk/cZxWWioRBqM
- https://chartexchange.com/trends/reddit/mentions/pennystocks+PennyStocksDD+RobinHoodPennyStocks+Shortsqueeze+smallstreetbets+SPACs+Spacstocks+SqueezePlays+WebullPennyStocks/
- https://altindex.com/short-squeeze-stocks
- https://www.fool.com/investing/2026/02/02/10-most-owned-stocks-on-robinhood-the-best/
- https://note.com/hirokimiyano/n/n49bda9aad424?hl=en
