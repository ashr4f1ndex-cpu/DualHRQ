
from typing import Optional
import pandas as pd
from .pricing import bsm_straddle, crr_straddle, year_fraction_calendar_days

def price_straddle(
    S: float, T: float, r: float, q: float, sigma: float,
    style: str = "european", dividends: Optional[List[Tuple[float,float]]] = None
) -> float:
    if style.lower() == "american":
        return crr_straddle(S, T, r, sigma, steps=300, dividends=dividends)
    return bsm_straddle(S, T, r, q, sigma)

def simulate_atm_straddle_roundtrip(
    S_entry: float,
    iv_entry: float,
    S_exit: float,
    iv_exit: float,
    *,
    entry_ts: Optional[pd.Timestamp] = None,
    expiry_ts: Optional[pd.Timestamp] = None,
    exit_ts: Optional[pd.Timestamp] = None,
    dte_days: Optional[int] = None,
    hold_days: Optional[int] = None,
    r_annual: float = 0.01,
    q_annual: float = 0.0,
    bid_ask_bps: float = 12.5,
    commission_per_contract: float = 0.65,
    contracts: int = 1,
    multiplier: int = 100,
    style: str = "european",
    dividends: Optional[list[tuple[float,float]]] = None,
    session: str = "open",  # widen spreads at open
    open_widen_bps: float = 5.0
) -> tuple[float, float] | float:
    """
    Price an at‑the‑money straddle at entry and exit.

    There are two modes of operation:

    1. **Calendar timestamp mode** (default): provide ``entry_ts``, ``expiry_ts`` and ``exit_ts``
       as pandas Timestamps.  The time to expiry at entry is computed as the calendar
       day count between ``entry_ts`` and ``expiry_ts``.  The exit time to expiry is
       similarly computed from ``exit_ts``.

    2. **Day‑count mode**: provide ``dte_days`` and ``hold_days``.  In this mode
       the option is assumed to have ``dte_days`` calendar days to expiry at entry and
       the trade is held for ``hold_days`` days.  The year fraction is computed as
       ``dte_days/365.0`` and ``max(dte_days - hold_days, 0)/365.0`` for the exit.

    In both modes the function returns either a single P&L figure (day‑count mode)
    or a tuple ``(pnl, costs)`` (calendar timestamp mode).  The P&L is in currency
    units for the specified number of ``contracts`` and ``multiplier``.  Costs
    include bid/ask spread and commissions per contract.
    """
    # Determine the year fractions to expiry for entry and exit
    if dte_days is not None and hold_days is not None:
        # Day‑count mode: compute times in years
        T_entry = max(float(dte_days), 0.0) / 365.0
        T_exit = max(float(dte_days) - float(hold_days), 0.0) / 365.0
        # Spread fraction: no open widening in day‑count mode unless specified
        spread_frac = (
            (bid_ask_bps + open_widen_bps) / 1e4 if session == "open" 
            else (bid_ask_bps / 1e4)
        )
    elif entry_ts is not None and expiry_ts is not None and exit_ts is not None:
        # Calendar timestamp mode
        T_entry = year_fraction_calendar_days(entry_ts, expiry_ts, basis="ACT/365")
        T_exit = max(
            0.0, year_fraction_calendar_days(exit_ts, expiry_ts, basis="ACT/365")
        )
        spread_frac = (
            bid_ask_bps + (open_widen_bps if session == "open" else 0.0)
        ) / 1e4
    else:
        raise ValueError(
            "Either (entry_ts, expiry_ts, exit_ts) or (dte_days, hold_days) must be provided"
        )

    # Price at entry
    mid_entry = price_straddle(S_entry, T_entry, r_annual, q_annual, max(iv_entry, 1e-6), style=style, dividends=dividends)
    entry_price = mid_entry * (1.0 + spread_frac) + 2 * commission_per_contract / multiplier
    # Price at exit
    mid_exit = price_straddle(S_exit, T_exit, r_annual, q_annual, max(iv_exit, 1e-6), style=style, dividends=dividends)
    exit_price = mid_exit * (1.0 - spread_frac) - 2 * commission_per_contract / multiplier
    # Compute P&L per contract and total costs
    pnl_per_contract = (exit_price - entry_price) * multiplier
    costs = (entry_price - mid_entry) + (mid_exit - exit_price)
    # If day‑count mode, return only P&L for clarity
    if dte_days is not None and hold_days is not None:
        return pnl_per_contract * contracts
    # Otherwise return both P&L and total friction costs
    return pnl_per_contract * contracts, costs * contracts * multiplier
