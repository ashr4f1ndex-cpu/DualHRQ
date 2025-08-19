
import numpy as np
import pandas as pd

def vol_target_sizer(signal: pd.Series, target_daily_vol: float = 0.01, est_vol: float = 0.02, max_contracts:int=5):
    """Position size proportional to conviction, scaled to target volatility.
    Very simplified: contracts = clip( signal_z * target/est )
    """
    sig = (signal - signal.mean()) / (signal.std() + 1e-9)
    raw = sig * (target_daily_vol / max(est_vol, 1e-6))
    size = raw.round().clip(lower=-max_contracts, upper=max_contracts)
    return size

def apply_drawdown_guard(equity: pd.Series, max_dd: float = -0.1) -> pd.Series:
    """Return a trade-allowed boolean series; pauses trading after breach until recovery."""
    roll_max = equity.cummax()
    dd = (equity - roll_max) / (roll_max.replace(0, 1))
    allowed = (dd >= max_dd).astype(int)
    return allowed
