
import numpy as np
import pandas as pd

from ..intraday.scanner_video import detect_parabolic_reversal


def test_scanner_basic_structure():
    # Build a synthetic parabolic then fade
    idx = pd.date_range("2024-05-06 09:30", "2024-05-06 16:00", freq="T")
    base = np.linspace(100, 108, len(idx)) + np.sin(np.linspace(0, 10, len(idx)))
    # inject a blow-off near mid-session
    blow = int(len(idx)*0.4)
    base[blow:blow+5] += np.linspace(0, 3.5, 5)  # vertical push
    vol = np.ones(len(idx))*1e5
    vol[blow:blow+5] *= 5
    df = pd.DataFrame({"open": base, "high": base+0.1, "low": base-0.1, "close": base, "volume": vol}, index=idx)
    signal, lh_level, diag = detect_parabolic_reversal(df, stretch_thr=1.5, vol_mult=2.0)
    # Expect at least one signal after the blow-off
    assert signal.any()
