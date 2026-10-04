"""Offline test of the API -> canonical conversion, on hand-made API-shaped rows."""
import numpy as np
import pandas as pd

from pmct.fetch import to_canonical
from pmct.schema import validate


def test_to_canonical_directions_and_prices():
    mk = pd.DataFrame(dict(condition_id=["0xa", "0xb"], question=["A?", "B?"], created_ts=[1000.0, 1000.0],
                           end_date_ts=[5000.0, 9000.0], closed_ts=[6000.0, np.nan], outcome=[1.0, np.nan],
                           category=["politics", "sports"], neg_risk=[False, False]))
    raw = {
        "0xa": pd.DataFrame(dict(proxyWallet=["0xW1", "0xw2", "0xW1", "0xw2"], side=["BUY", "BUY", "SELL", "SELL"],
                                 outcomeIndex=[0, 1, 0, 1], price=[0.6, 0.3, 0.65, 0.2], size=[10, 10, 5, 5],
                                 timestamp=[2000, 2100, 2200, 2300])),
        "0xb": pd.DataFrame(dict(proxyWallet=["0xw3"], side=["BUY"], outcomeIndex=[0], price=[0.5], size=[2],
                                 timestamp=[3000])),
    }
    markets, trades, wallets = to_canonical(mk, raw)
    validate(markets, trades)
    a = trades[trades.market_id == 0].sort_values("ts")
    assert a["dir"].tolist() == [1, -1, -1, 1]          # buy YES, buy NO, sell YES, sell NO
    assert np.allclose(a["yes_price"], [0.6, 0.7, 0.65, 0.8])
    assert len(wallets) == 3                            # addresses lower-cased
    assert markets.loc[markets.market_id == 1, "outcome"].isna().all()   # open market kept, unresolved
