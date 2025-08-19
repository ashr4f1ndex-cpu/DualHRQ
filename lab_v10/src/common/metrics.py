
import numpy as np
import pandas as pd
from math import sqrt
from scipy.stats import skew, kurtosis, norm

def sharpe(pnl_series: pd.Series, periods_per_year: int = 252) -> float:
    """Compute the annualised Sharpe ratio of a P&L series.

    The Sharpe ratio is the mean return divided by the standard
    deviation of returns, multiplied by the square root of the
    annualisation factor.  A small constant is added to the
    denominator to avoid division by zero.  If the input series
    contains no variation (zero standard deviation) the function
    returns 0.0.

    Parameters
    ----------
    pnl_series : pd.Series
        A time series of per-period P&L or returns.
    periods_per_year : int, optional
        The number of periods in a year (252 for daily data).

    Returns
    -------
    float
        The annualised Sharpe ratio.
    """
    s = pnl_series.dropna()
    if s.std(ddof=0) == 0:
        return 0.0
    return (s.mean() / (s.std(ddof=0) + 1e-12)) * np.sqrt(periods_per_year)

def max_drawdown(equity: pd.Series) -> float:
    """
    Compute the maximum drawdown of a cumulative equity curve.

    Returns the worst (most negative) fractional drawdown, defined as
    ``(equity - rolling_max) / rolling_max``.  If the input series is empty
    or contains no variation, the function returns 0.0.  A negative return
    indicates a drawdown; 0.0 means no drawdown.
    """
    s = equity.dropna()
    if s.empty:
        return 0.0
    roll_max = s.cummax()
    # Avoid division by zero; if roll_max is zero replace with 1.0
    denom = roll_max.replace(0.0, np.nan)
    dd_ratio = (s - roll_max) / denom
    # If denom was zero, dd_ratio may be NaN; drop NaNs
    if dd_ratio.dropna().empty:
        return 0.0
    return float(dd_ratio.min())

def sortino(pnl_series: pd.Series, periods_per_year: int = 252) -> float:
    s = pnl_series.dropna()
    downside = s[s < 0]
    if downside.std(ddof=0) == 0:
        return 0.0
    return (s.mean() / (downside.std(ddof=0) + 1e-12)) * np.sqrt(periods_per_year)

def deflated_sharpe(pnl_series: pd.Series, trials: int = 1, periods_per_year: int = 252) -> float:
    """Compute the deflated Sharpe ratio of a P&L series.

    The deflated Sharpe ratio (DSR) adjusts an observed Sharpe ratio
    for estimation error, non‑normal returns and multiple testing.  It
    follows the framework proposed by Bailey and López de Prado:

    1. Compute the sample Sharpe ratio ``S``.
    2. Estimate the standard error of ``S`` accounting for skewness and
       kurtosis of the returns.
    3. Compute a z‑score threshold ``z*`` based on the effective number
       of trials (strategies) tested.
    4. Deflate ``S`` by subtracting ``z* × SE`` and apply a floor at 0.

    This implementation returns the deflated Sharpe in the same units
    as the Sharpe ratio and ensures the result does not exceed the
    sample Sharpe.  If fewer than 10 observations are supplied, the
    function returns 0.0.

    Parameters
    ----------
    pnl_series : pd.Series
        Per-period P&L or returns.
    trials : int, optional
        Effective number of strategies tested.  Must be >= 1.
    periods_per_year : int, optional
        The number of periods per year used for annualisation.

    Returns
    -------
    float
        The deflated Sharpe ratio (Sharpe units).
    """
    s = pnl_series.dropna()
    if len(s) < 10:
        return 0.0
    # sample Sharpe
    S = sharpe(s, periods_per_year)
    n = len(s)
    # skewness and excess kurtosis
    g1 = float(skew(s, bias=False))
    g2 = float(kurtosis(s, fisher=True, bias=False))
    # ensure at least one trial
    trials = max(int(trials), 1)
    # z* threshold for the maximum Sharpe expected across trials
    z_star = norm.ppf(1.0 - 1.0 / trials) if trials > 1 else 0.0
    # standard error of the Sharpe ratio (Bailey & López de Prado, 2014)
    # se^2 = (1 - g1*S + 0.25*(g2 - 1)*S^2) / (n - 1)
    se_num = max(1e-12, (1 - g1 * S + 0.25 * (g2 - 1.0) * S * S))
    se = np.sqrt(se_num / max(n - 1, 1))
    # deflate the Sharpe by z*·se and floor at zero
    return float(max(0.0, S - z_star * se))

def summarize(pnl: pd.Series, equity: pd.Series, label: str = "", trials: int = 1) -> dict:
    """Aggregate key performance metrics for a P&L series.

    This helper computes a dictionary of common metrics including the
    total P&L, Sharpe ratio, Sortino ratio, deflated Sharpe ratio,
    maximum drawdown and the number of non‑zero trade outcomes.  It
    uses the new `deflated_sharpe` function to report a statistically
    conservative risk‑adjusted measure.

    Parameters
    ----------
    pnl : pd.Series
        Per-period P&L or returns.
    equity : pd.Series
        Cumulative equity curve (same length as ``pnl``).
    label : str, optional
        Optional label identifying the series.
    trials : int, optional
        Effective number of trials for the deflated Sharpe ratio.

    Returns
    -------
    dict
        A dictionary containing the computed statistics.
    """
    return {
        "label": label,
        "samples": int(len(pnl.dropna())),
        "pnl_sum": float(pnl.sum()),
        "sharpe": float(sharpe(pnl)),
        "sortino": float(sortino(pnl)),
        "deflated_sharpe": float(deflated_sharpe(pnl, trials=trials)),
        "mdd": float(max_drawdown(equity)),
        "trades": int((pnl != 0).sum()),
    }
