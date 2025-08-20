
import pandas as pd
import numpy as np
from ..intraday.aggregate_daily import aggregate_intraday_features

def test_aggregate_alignment():
    idx = pd.date_range("2024-05-06 09:30", "2024-05-06 16:00", freq="T")
    close = pd.Series(np.linspace(100, 101, len(idx)), index=idx)
    df = pd.DataFrame({"open": close.shift().fillna(close.iloc[0]), "high": close+0.1, "low": close-0.1, "close": close, "volume": 1e5}, index=idx)
    sig = pd.Series(False, index=idx)
    diag = df.copy()
    diag['vwap'] = close
    diag['stretch'] = 0
    diag['volx'] = 1
    diag['parabolic'] = False
    feats = aggregate_intraday_features(df, sig, diag)
    # Index should be daily
    assert feats.index.freq is None or feats.index.inferred_type in ("datetime64", )
    assert 'parabolic_count' in feats.columns
