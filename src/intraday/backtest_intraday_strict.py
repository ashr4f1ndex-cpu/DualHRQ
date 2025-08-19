"""
backtest_intraday_strict.py
==========================

This module contains a more realistic intraday short sale simulator
building on the simplified version in ``backtest_intraday.py``.  The
enhanced implementation enforces several regulatory and microstructure
constraints:

* **Short Sale Restriction persistence**: Once the SSR trigger fires
  (a 10 % drop from the previous day's close) it remains active
  through the end of that trading day *and* for the entire
  following day【515680207816213†L213-L218】.
* **Uptick price test**: Short sales cannot execute at or below the
  current national best bid; in this simulation we approximate this by
  requiring the entry price to be strictly above the prior bar's
  closing price【515680207816213†L213-L218】.
* **Limit Up–Limit Down (LULD) bands**: Price bands are doubled
  during the last 25 minutes of the trading day for Tier 1
  securities and Tier 2 securities below $3【457522738594840†L70-L71】.

All other parameters from the base simulator are retained.  Traders
can specify slippage, stop distance, optional VWAP targets, random
shock probability, borrow availability and rates.  The function
returns per‑bar P&L, an equity curve and the number of executed trades.
"""

from __future__ import annotations

import pandas as pd
import numpy as np


def simulate_short_with_constraints(
    df: pd.DataFrame,
    signal: pd.Series,
    lh_level: pd.Series,
    stop_bps: float = 35,
    target_to_vwap: bool = True,
    slippage_bps: int = 5,
    apply_ssr: bool = True,
    apply_luld: bool = True,
    luld_pct: float = 0.10,
    luld_state_bars: int = 3,
    luld_pause_bars: int = 5,
    shock_prob: float = 0.0,
    shock_bps: int = 30,
    seed: int | None = None,
    borrow_availability: float = 1.0,
    borrow_rate_bps: float = 0.0,
) -> tuple[pd.Series, pd.Series, int]:
    """Simulate intraday short trades with regulatory constraints.

    Parameters
    ----------
    df : pd.DataFrame
        Intraday OHLCV data indexed by timestamp.  Must contain
        columns ``open``, ``high``, ``low``, ``close`` and ``volume``.
    signal : pd.Series
        Boolean series aligned to ``df`` indicating trade entry points
        (fires on the bar immediately preceding entry).
    lh_level : pd.Series
        Series of pivot (lower–high) levels; the stop is placed above
        this level.
    stop_bps : float
        Stop distance in basis points above the pivot.
    target_to_vwap : bool
        If True, exit trades when the running VWAP is hit; otherwise
        exit at the session close.
    slippage_bps : int
        Base slippage applied to entry and exit prices (in basis points).
    apply_ssr : bool
        If True, apply the short sale restriction persistence logic.
    apply_luld : bool
        If True, enforce limit up–limit down price bands and doubling in
        the last 25 minutes of trading.【457522738594840†L70-L71】
    luld_pct : float
        Percentage width of the LULD price band (e.g. 0.10 for ±10 %).
    luld_state_bars : int
        Number of consecutive bars at a band edge required to trigger a
        limit state (approx. 15 s if using one‑minute bars).
    luld_pause_bars : int
        Number of bars to pause trading after a limit state is triggered
        (approx. 5 minutes if using one‑minute bars).
    shock_prob : float
        Probability of encountering an additional slippage shock on entry
        or exit.
    shock_bps : int
        Extra slippage (in basis points) applied when a shock occurs.
    seed : int, optional
        Random seed for reproducibility of slippage shocks.
    borrow_availability : float
        Probability that shares are available to borrow.  Values in (0, 1].
    borrow_rate_bps : float
        Borrow fee in basis points deducted from P&L per trade.

    Returns
    -------
    pnl : pd.Series
        Per‑bar P&L (per share).  Length ``len(df)`` minus one.
    equity : pd.Series
        Cumulative equity curve.
    trades : int
        Total number of executed trades.
    """
    d = df.copy()
    # align signals and pivot levels
    d["signal"] = signal.reindex(d.index).fillna(False).astype(bool)
    d["lh_level"] = lh_level.reindex(d.index)
    # running VWAP for each bar
    d["vwap"] = (d["close"] * d["volume"]).cumsum() / (d["volume"].cumsum().replace(0, np.nan))
    # rolling five‑minute reference price for LULD (min_periods=1 to seed early bars)
    if apply_luld:
        d["ref_price"] = d["close"].rolling(window=5, min_periods=1).mean()
    # compute day index
    day_index = pd.to_datetime(d.index.date)
    # compute SSR active days
    ssr_active_by_day: dict[pd.Timestamp, bool] = {}
    if apply_ssr:
        # previous day's close per day
        last_close_by_day = d["close"].groupby(day_index).last()
        prev_close_by_day = last_close_by_day.shift(1)
        for day, prev_close in prev_close_by_day.items():
            # if no previous close, SSR cannot trigger
            prev_c = prev_close
            if pd.isna(prev_c):
                ssr_active_by_day[day] = False
                continue
            # compute intraday low for that day
            day_mask = (day_index == day)
            low_today = d.loc[day_mask, "low"].cummin().iloc[-1]
            triggered = low_today <= prev_c * 0.9
            # active on this day if previous day triggered
            prev_day = (last_close_by_day.index.get_loc(day) - 1)
            prev_triggered = False
            if prev_day >= 0:
                prev_day_key = last_close_by_day.index[prev_day]
                # was SSR triggered on the previous day?
                prev_low = d.loc[day_index == prev_day_key, "low"].cummin().iloc[-1]
                prev_prev_close = prev_close_by_day.get(prev_day_key)
                if pd.notna(prev_prev_close):
                    prev_triggered = prev_low <= prev_prev_close * 0.9
            ssr_active_by_day[day] = bool(triggered or prev_triggered)
    # Precompute LULD doubling flags per bar
    double_band = np.zeros(len(d), dtype=bool)
    if apply_luld:
        # group bars by day and flag last 25 bars
        for day in np.unique(day_index):
            indices = np.where(day_index == day)[0]
            # mark last 25 bars (or all if fewer)
            if len(indices) > 0:
                last_n = indices[-25:]
                double_band[last_n] = True
    # state trackers
    limit_counter = 0
    pause_counter = 0
    pnl: list[float] = []
    eq_curve: list[float] = []
    trades = 0
    eq = 0.0
    rng = np.random.default_rng(seed)
    rng_borrow = np.random.default_rng(seed + 1234 if seed is not None else None)
    for i in range(1, len(d)):
        # update LULD limit state if enabled
        if apply_luld:
            if pause_counter > 0:
                pause_counter -= 1
            else:
                # compute band width, doubling in final 25 minutes
                ref = d["ref_price"].iloc[i]
                pct = luld_pct * (2.0 if double_band[i] else 1.0)
                band_hi = ref * (1 + pct)
                band_lo = ref * (1 - pct)
                price = d["close"].iloc[i]
                if price >= band_hi or price <= band_lo:
                    limit_counter += 1
                    if limit_counter >= luld_state_bars:
                        pause_counter = luld_pause_bars
                        limit_counter = 0
                else:
                    limit_counter = 0
        # determine if SSR restricts this bar
        ssr_restrict = False
        if apply_ssr:
            current_day = day_index[i]
            # SSR active on this day?  Map missing keys to False
            ssr_restrict = bool(ssr_active_by_day.get(current_day, False))
        # decide whether we can place a trade on this bar
        can_trade = True
        if apply_luld and pause_counter > 0:
            can_trade = False
        if ssr_restrict:
            can_trade = False
        # apply uptick test: entry must be above prior close
        prior_close = d["close"].iloc[i - 1]
        # If prior close is NaN (unlikely), allow trade
        # When the prior bar signalled, attempt a trade on this bar
        if d["signal"].iloc[i - 1] and can_trade and d["open"].iloc[i] > prior_close:
            # check borrow availability
            trades += 1
            if rng_borrow.random() > borrow_availability:
                pnl.append(0.0)
                eq_curve.append(eq)
                continue
            # compute entry slippage
            slip_in = slippage_bps + (shock_bps if (rng.random() < shock_prob) else 0)
            entry_price = d["open"].iloc[i] * (1 - slip_in / 1e4)
            pivot = d["lh_level"].iloc[i - 1]
            raw_stop = pivot if pd.notna(pivot) else d["high"].iloc[i - 1]
            stop_price = raw_stop * (1 + stop_bps / 1e4)
            # choose target price
            target_price = d["vwap"].iloc[i] if target_to_vwap else d["close"].iloc[-1]
            # exit slippage
            slip_out = slippage_bps + (shock_bps if (rng.random() < shock_prob) else 0)
            exit_price: float
            if d["high"].iloc[i] >= stop_price:
                exit_price = stop_price * (1 + slip_out / 1e4)
            elif d["low"].iloc[i] <= target_price:
                exit_price = target_price * (1 + slip_out / 1e4)
            else:
                exit_price = d["close"].iloc[i] * (1 + slip_out / 1e4)
            trade_pnl = entry_price - exit_price
            if borrow_rate_bps > 0.0:
                borrow_cost = borrow_rate_bps / 1e4 * entry_price
                trade_pnl -= borrow_cost
            pnl.append(trade_pnl)
            eq += trade_pnl
        else:
            # no signal or trade blocked
            pnl.append(0.0)
            eq += 0.0
        eq_curve.append(eq)
    # convert lists to series
    pnl_series = pd.Series(pnl, index=d.index[: len(pnl)])
    equity_series = pd.Series(eq_curve, index=d.index[: len(eq_curve)])
    return pnl_series, equity_series, trades