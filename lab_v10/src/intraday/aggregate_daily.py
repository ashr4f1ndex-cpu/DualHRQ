
import numpy as np
import pandas as pd


def aggregate_intraday_features(df: pd.DataFrame, signal: pd.Series, diag: pd.DataFrame) -> pd.DataFrame:
    """Aggregate intraday indicators to daily features for the options HRM.
    Assumes df index is intraday timestamps; includes columns: open, high, low, close, volume, and diag has vwap, stretch, volx, parabolic.
    Returns daily DataFrame indexed by date with engineered features.
    """
    d = df.copy()
    d['date'] = d.index.date
    diag2 = diag.copy()
    diag2['date'] = diag2.index.date
    sig = signal.copy()
    sig.index = d.index
    sig = sig.astype(int)
    # Features per day
    grp = d.groupby('date')
    grpd = diag2.groupby('date')
    feats = pd.DataFrame(index=pd.to_datetime(sorted(set(d['date']))))
    feats['parabolic_count'] = grpd['parabolic'].sum()
    feats['max_stretch'] = grpd['stretch'].max()
    feats['max_volx'] = grpd['volx'].max()
    feats['lh_confirmed'] = grp.apply(lambda g: int(sig.reindex(g.index).any()))
    feats['range_atr_ratio'] = (grp['high'].max() - grp['low'].min()) / (grp['close'].apply(lambda s: s.diff().abs()).rolling(14).mean().iloc[-1] + 1e-9)
    # Time to VWAP reversion (bars from first signal to first time close <= vwap)
    def _tt_vwap(gdf, gdiag, gsig):
        if not gsig.any():
            return np.nan
        t0 = gsig[gsig].index[0]
        # find first bar after t0 where close <= vwap
        post = gdiag[gdiag.index >= t0]
        returned = post[post['close'] <= post['vwap']]
        if len(returned) == 0:
            return np.nan
        return (returned.index[0] - t0).seconds/60.0
    feats['time_to_vwap_min'] = [
        _tt_vwap(df[df.index.date==d], diag[diag.index.date==d], sig[sig.index.date==d])
        for d in feats.index.date
    ]
    feats.index.name = 'date'
    # Shift features forward by one day to avoid lookahead when using for next-day trades.
    return feats.shift(1)
