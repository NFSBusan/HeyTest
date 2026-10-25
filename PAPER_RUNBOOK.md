# Daily paper-trading run (automatic, weekdays ~3:40pm ET)

Read-only IBKR market data only. NEVER create orders, order instructions, alerts or watchlists.
Branch: `claude/admiring-brown-tol4xd`. `D` = today's date (YYYY-MM-DD, New York time).

0. `git pull origin claude/admiring-brown-tol4xd`. If `data/live/D/` already has `legs.json` and `status.md`
   shows today's date, stop (already done).
1. **Market open check.** Snapshot one stock (e.g. HOOD) with `top_status`. If the market is closed today
   (weekend/holiday: the last trade timestamp is not today), stop and do nothing.
2. **Underlyings.** For each of the 71 stocks (`data/universe_2025-09.csv` minus the ETFs listed in
   `scripts/universe_study.py` ETF_LEAKS), call `get_price_snapshot` with
   `["last", "prior_close", "implied_vol_underlying"]`. Contract ids are cached in `data/live/ibkr_ids.json`
   (`{SYM: {"conid": n, "exp": {"YYYY-MM-DD": expiration_id}, "opt": {"YYYY-MM-DD|C|strike": conid}}}`);
   look up missing ones with `search_contracts` (exact symbol, US listing with OPT) and add them.
   Write `data/live/D/underlyings.json` = `{SYM: {"last": price, "prior_close": close or null, "iv": annual_iv or null}}`.
3. `python3 scripts/paper_tracker.py plan D` → `data/live/D/plan.json` lists the legs to quote.
4. **Option quotes.** For each leg in plan.json:
   - `fill` / `mark`: strikes are given.
   - `new`: get the expiry's chain (`get_option_parameters` → regular expiration with that date, no odd
     trading class → `get_option_data` bounded around the targets) and take the listed strike NEAREST to
     `call_target` for the call and to `put_target` for the put.
   Snapshot each option with `["bid_ask", "option_volume"]`. Volume = the call/put volume number in the
   response for that contract (null if absent). Write `data/live/D/legs.json` =
   `{key: {"call": {"strike", "bid", "ask", "volume"}, "put": {...}}}` (key = `SYM|expiry` from plan.json).
   Skip a leg whose quote is empty (the script then skips it).
5. `python3 scripts/paper_tracker.py apply D` → updates `results/paper_trades/book.csv`, `account.csv`, `status.md`.
6. Append one line to section 8 of `PROJECT_NOTES.txt` only if something happened (entries, settlements, or a
   Friday mark): date, equity, open positions, stress %. Then commit
   (`Paper run D`, with the session trailers) and `git push -u origin claude/admiring-brown-tol4xd`.
7. Tell the user in 2-4 lines: what was entered / settled, equity, stress %. Flag anything odd (stress > 20%,
   a position losing more than 3x its credit, missing data).
