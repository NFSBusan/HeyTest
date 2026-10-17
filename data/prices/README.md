# Daily prices (IBKR)

Source: Interactive Brokers `get_price_history`, daily bars, regular trading hours only,
split-adjusted, pulled 2026-10-16 (15-minute delayed feed). Converted with `scripts/import_ibkr.py`.

- Window: 2021-10-18 to 2026-10-16 (5 years) unless the stock listed later. IREN starts 2021-11-17.
- NBIS: trimmed to start after the 2022–2024 Yandex trading halt (IBKR shows flat prices during the halt).
- CRWV is missing (listed 2025-03-28; the IBKR result was not saved to a file). Re-pull it.
- Prices only. No mentions, off-exchange volume or options data yet.
- IBKR market data is for your own use; check their terms before sharing it publicly.

## Added 2026-10-17 (small/mid-cap expansion)

SOUN, BBAI, QBTS, RGTI, LUNR, OKLO, ACHR, JOBY, CIFR, CLSK, MARA, RIOT, AMC, PLUG, BMEA, NAK, INDI.
Same source and window. SPAC shell periods removed (price pinned near the trust value before the merger):
LUNR before 2023-02-14, OKLO before 2024-05-10, QBTS before 2022-08-08.
BDRX was skipped: it has no listed options.
