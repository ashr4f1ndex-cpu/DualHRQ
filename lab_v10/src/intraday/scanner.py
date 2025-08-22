
import numpy as np
import pandas as pd


def vwap(df: pd.DataFrame) -> pd.Series:
    pv = (df['close'] * df['volume']).cumsum()
    vv = df['volume'].cumsum().replace(0, np.nan)
    return pv / vv

def atr(high, low, close, n=14):
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def detect_parabolic_backside(df: pd.DataFrame, atr_n: int=14, stretch=2.0, vol_mult=2.5, lookback=30):
    """Return a boolean Series: potential backside short trigger (lower-high break) after parabolic extension."""
    d = df.copy()
    d['vwap'] = vwap(d)
    d['atr'] = atr(d['high'], d['low'], d['close'], n=atr_n)
    d['stretch'] = (d['close'] - d['vwap']) / (d['atr'] + 1e-9)
    d['volx'] = d['volume'] / (d['volume'].rolling(20).mean() + 1e-9)

    # Identify blow-off region
    d['parabolic'] = (d['stretch'] >= stretch) & (d['volx'] >= vol_mult)
    # Lower-high: peak then a bounce that fails under the peak
    d['peak'] = (d['high'] == d['high'].rolling(lookback, min_periods=3).max())
    # Create a simple LH condition: last peak price, then price makes a lower high and breaks that pivot low
    lh = []
    last_peak = None
    for _i, row in d.iterrows():
        if bool(row['peak']) and bool(row['parabolic']):
            last_peak = row['high']
            lh.append(False)
        elif last_peak is not None and row['high'] < last_peak and row['close'] < row['open']:
            lh.append(True)
            last_peak = None
        else:
            lh.append(False)
    d['lower_high_break'] = pd.Series(lh, index=d.index)

    return d['lower_high_break']
