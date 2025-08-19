
import pandas as pd
import numpy as np

def vwap(df: pd.DataFrame) -> pd.Series:
    pv = (df['close'] * df['volume']).cumsum()
    vv = df['volume'].cumsum().replace(0, np.nan)
    return pv / vv

def atr(df: pd.DataFrame, n:int=14) -> pd.Series:
    high, low, close = df['high'], df['low'], df['close']
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=1).mean()

def _find_lower_high_triggers(d: pd.DataFrame, lookback:int=30) -> pd.DataFrame:
    # Identify peaks after parabolic on each day session
    d = d.copy()
    d['peak'] = (d['high'] == d['high'].rolling(lookback, min_periods=3).max())
    # Build state machine: after a parabolic=True bar triggers, watch for first peak, then a bounce failing under that peak; 
    # define LH pivot as that bounce's swing low; trigger when price breaks pivot low.
    state = "idle"
    peak_price = None
    pivot_low = None
    lh_level = []
    signal = []
    for i, row in d.iterrows():
        if state == "idle":
            lh_level.append(np.nan); signal.append(False)
            if bool(row['parabolic']):
                state = "parabolic"
        elif state == "parabolic":
            lh_level.append(np.nan); signal.append(False)
            if bool(row['peak']):
                peak_price = float(row['high'])
                state = "after_peak"
        elif state == "after_peak":
            # Waiting for a failed bounce after the parabolic peak.  A valid
            # lower‑high is signalled by a bar whose high is below the peak
            # and whose close is below its open (a red candle).  Without the
            # red close requirement the algorithm would often mis‑identify
            # noisy price action as a reversal.  See review for details.
            if peak_price is not None and row['high'] < peak_price and row['close'] < row['open']:
                pivot_low = float(row['low'])
                state = "pivot_set"
            lh_level.append(np.nan)
            signal.append(False)
        elif state == "pivot_set":
            # Trigger when price breaks below pivot low
            trig = row['close'] < (pivot_low if pivot_low is not None else row['close'] - 1e9)
            signal.append(bool(trig))
            lh_level.append(pivot_low if pivot_low is not None else np.nan)
            if trig:
                state = "idle"; peak_price=None; pivot_low=None
            # Reset if new parabolic detected (rare but possible)
            if bool(row['parabolic']):
                state = "parabolic"; peak_price=None; pivot_low=None
    d['lh_level'] = pd.Series(lh_level, index=d.index)
    d['enter_signal'] = pd.Series(signal, index=d.index).astype(bool)
    return d

def detect_parabolic_reversal(
    df: pd.DataFrame,
    atr_n:int=14, stretch_thr:float=2.0, vol_mult:float=2.5, lookback:int=30
):
    """Video-accurate detector: parabolic stretch + lower-high breakdown trigger.
    Returns (signal Series, lh_level Series, enriched DataFrame with diagnostics)."""
    d = df.copy()
    d['vwap'] = vwap(d)
    d['atr'] = atr(d, n=atr_n).replace(0, np.nan)
    d['stretch'] = (d['close'] - d['vwap']) / (d['atr'] + 1e-9)
    d['volx'] = d['volume'] / (d['volume'].rolling(20, min_periods=1).mean() + 1e-9)
    d['parabolic'] = (d['stretch'] >= stretch_thr) & (d['volx'] >= vol_mult)
    d = _find_lower_high_triggers(d, lookback=lookback)
    return d['enter_signal'], d['lh_level'], d
