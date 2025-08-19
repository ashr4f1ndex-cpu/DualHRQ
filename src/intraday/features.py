import numpy as np

def vwap(price, volume):
    """Compute the volume‑weighted average price (VWAP) of a series.

    Parameters
    ----------
    price : sequence of float
        Sequence of trade or quote prices.
    volume : sequence of float
        Corresponding sequence of traded volumes.

    Returns
    -------
    float
        The volume‑weighted average price.  If the total volume is zero the
        last price is returned.
    """
    p = np.asarray(price, dtype=float)
    v = np.asarray(volume, dtype=float)
    total_vol = v.sum()
    if total_vol > 0.0:
        return float((p * v).sum() / total_vol)
    return float(p[-1])

def lower_high_pivot(high, lookback: int = 10) -> bool:
    """Detect a lower‑high pivot.

    A lower‑high pivot occurs when the most recent high is strictly below
    the maximum of the prior ``lookback`` highs.  This function avoids
    look‑ahead bias by only using past data.

    Parameters
    ----------
    high : sequence of float
        Sequence of intraday high prices.
    lookback : int, optional
        Number of prior highs to consider when defining the pivot.  The
        default is 10 bars.

    Returns
    -------
    bool
        True if the most recent high is lower than all highs in the
        preceding window, False otherwise.
    """
    h = np.asarray(high, dtype=float)
    if len(h) <= lookback:
        return False
    window_max = np.max(h[-(lookback + 1):-1])
    return bool(h[-1] < window_max)

def time_to_vwap(price, volume, max_lookback: int = 10) -> int:
    """Compute the number of bars since the price last crossed the VWAP.

    The function looks back up to ``max_lookback`` bars and finds the
    most recent index where the sign of (price - VWAP) differs from the
    current sign.  A value of zero indicates no cross in the lookback
    window.

    Parameters
    ----------
    price : sequence of float
        Sequence of intraday prices.
    volume : sequence of float
        Corresponding sequence of traded volumes.
    max_lookback : int, optional
        Maximum number of bars to search backward for a VWAP cross.

    Returns
    -------
    int
        Number of bars since the last VWAP cross.  If no cross is found
        in the lookback window, returns zero.
    """
    p = np.asarray(price, dtype=float)
    v = np.asarray(volume, dtype=float)
    vw = vwap(p, v)
    # sign of distance to vwap
    signs = np.sign(p - vw)
    current_sign = signs[-1]
    # examine the last max_lookback bars excluding the current bar
    lookback_slice = signs[-(max_lookback + 1):-1] if max_lookback > 0 else []
    if len(lookback_slice) == 0:
        return 0
    diffs = np.where(lookback_slice != current_sign)[0]
    if len(diffs) == 0:
        return 0
    # distance from current bar (bars since cross)
    return len(lookback_slice) - diffs[-1]


def volx(price, window: int = 20) -> float:
    """Compute a realised volatility estimate over a trailing window.

    This helper computes the standard deviation of log returns over the
    past ``window`` bars (excluding the current return) and scales it
    by the square root of the window.  The input series should be
    ordered chronologically.  If there are insufficient observations
    the function returns zero.

    Parameters
    ----------
    price : sequence of float
        Sequence of intraday prices.
    window : int, optional
        Length of the trailing window in bars.  Default is 20.

    Returns
    -------
    float
        Realised volatility estimate for the most recent bar.
    """
    p = np.asarray(price, dtype=float)
    if len(p) <= window:
        return 0.0
    log_p = np.log(p)
    returns = np.diff(log_p)
    recent = returns[-window:]
    # protect against near‑zero variance
    if np.allclose(recent, 0.0):
        return 0.0
    return float(np.sqrt(window) * np.std(recent, ddof=1))


def parabolic_spike(price, window: int = 30) -> float:
    """Detect and quantify parabolic price spikes.

    This indicator compares the most recent log return to the mean and
    standard deviation of returns over a trailing window.  It returns
    the z‑score of the current return relative to the past window.
    Larger positive values indicate an upwards price spike, whereas
    negative values indicate a sharp drop.  The function does not
    include the current return in the window statistics to avoid
    look‑ahead bias.

    Parameters
    ----------
    price : sequence of float
        Sequence of intraday prices.
    window : int, optional
        Trailing window length over which to compute the mean and standard
        deviation of log returns.  Default is 30 bars.

    Returns
    -------
    float
        Z‑score of the most recent log return relative to the trailing
        distribution.  Values above approximately 3 may be treated
        as significant spikes by higher‑level scanners.
    """
    p = np.asarray(price, dtype=float)
    if len(p) <= window + 1:
        return 0.0
    log_p = np.log(p)
    returns = np.diff(log_p)
    current_return = returns[-1]
    window_returns = returns[-(window + 1):-1]
    mu = np.mean(window_returns)
    sigma = np.std(window_returns, ddof=1) + 1e-8
    z = (current_return - mu) / sigma
    return float(z)
