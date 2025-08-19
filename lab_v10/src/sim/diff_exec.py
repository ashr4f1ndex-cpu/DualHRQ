"""
diff_exec.py
=============

Differentiable execution surrogate for trading strategies.  This module
provides a smooth mapping from raw model signals to portfolio PnL
series, accounting for position sizing, turnover and transaction
costs.  The functions defined here are fully differentiable and
therefore suitable for gradient‐based optimisation.  They do not
attempt to model all aspects of realistic execution, but supply a
simple yet flexible framework for connecting predictions to returns
within the loss function.

Example
-------

>>> import torch
>>> from src.sim.diff_exec import positions_to_returns_from_signal
>>> signals = torch.randn(2, 5)
>>> rets = torch.randn(2, 5) * 0.01
>>> pnl = positions_to_returns_from_signal(signals, rets, w_cap=1.0, beta=2.0)
>>> pnl.shape
torch.Size([2, 5])

Note
----
The functions here operate in units of simple returns and basis
points.  Costs are linear in turnover by default.  More elaborate
execution logic such as soft stop‐loss or profit‐target exits can be
implemented via additional helper functions (see ``soft_barrier_mix``).
"""

from __future__ import annotations

import torch
from ..common.portfolio_loss import EPS

def soft_abs(x: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """Smooth approximation of absolute value.

    ``soft_abs(x) = sqrt(x^2 + eps)``.  A small epsilon prevents
    gradients from blowing up when ``x`` approaches zero.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    eps : float
        Small constant for numerical stability.

    Returns
    -------
    torch.Tensor
        The smoothed absolute value.
    """
    return torch.sqrt(x * x + eps)

def smooth_position_from_signal(signal: torch.Tensor,
                                w_cap: float = 1.0,
                                beta: float = 2.0) -> torch.Tensor:
    """Map raw signals to continuous positions.

    Signals are transformed via a hyperbolic tangent into the range
    ``[-w_cap, w_cap]``.  The ``beta`` parameter controls the slope of
    the tanh; larger values make the mapping more binary.

    Parameters
    ----------
    signal : torch.Tensor
        Raw model outputs.  Shape ``[B, T]`` or ``[T]``.
    w_cap : float
        Maximum absolute position size.
    beta : float
        Temperature parameter for the tanh.

    Returns
    -------
    torch.Tensor
        Positions of the same shape as ``signal``.
    """
    return w_cap * torch.tanh(beta * signal)

def turnover(w: torch.Tensor, detach_prev: bool = True) -> torch.Tensor:
    """Compute per‐step turnover of positions.

    Turnover at time ``t`` is defined as ``|w_t - w_{t-1}|``.  The
    first timestep assumes a previous position of zero.  A smooth
    approximation of the absolute value is used to preserve
    differentiability.  To prevent gradients from propagating
    undesirably through previous positions when optimising, the
    ``detach_prev`` flag may be set to ``True`` (default) which
    detaches ``w_{t-1}`` from the computation graph.

    Parameters
    ----------
    w : torch.Tensor
        Position tensor of shape ``[B, T]`` or ``[T]``.
    detach_prev : bool
        If True, previous positions are detached from the gradient
        graph when computing turnover.

    Returns
    -------
    torch.Tensor
        Turnover tensor of the same shape as ``w``.
    """
    if w.dim() == 1:
        prev = torch.zeros_like(w)
        prev[1:] = w[:-1]
        if detach_prev:
            prev = prev.detach()
        return soft_abs(w - prev)
    # B x T
    prev = torch.zeros_like(w)
    prev[:, 1:] = w[:, :-1]
    if detach_prev:
        prev = prev.detach()
    return soft_abs(w - prev)

def trading_cost(turnover_t: torch.Tensor,
                 spread_bps: float = 1.0,
                 fee_bps: float = 0.1) -> torch.Tensor:
    """Compute linear transaction costs per timestep.

    Costs are modelled as proportional to turnover measured in basis
    points (one‐hundredth of a percent).  The effective cost per unit
    turnover is given by ``(spread_bps + fee_bps) / 1e4``.  Costs are
    subtracted from gross PnL when converting positions to returns.

    Parameters
    ----------
    turnover_t : torch.Tensor
        Per‐step turnover.  Shape matches that of positions.
    spread_bps : float
        Effective bid–ask spread in basis points.
    fee_bps : float
        Additional fees and slippage in basis points.

    Returns
    -------
    torch.Tensor
        Costs of the same shape as ``turnover_t``.
    """
    bps = (float(spread_bps) + float(fee_bps)) / 10000.0
    return bps * turnover_t

def positions_to_returns(w: torch.Tensor,
                         asset_returns: torch.Tensor,
                         spread_bps: float = 1.0,
                         fee_bps: float = 0.1,
                         detach_prev: bool = True) -> torch.Tensor:
    """Convert positions and asset returns into net PnL.

    The gross PnL at time ``t`` is ``w_{t-1} * r_t`` – that is,
    today's return multiplied by yesterday's position.  Costs are
    deducted based on the change in position (turnover) and the
    specified basis point costs.

    Parameters
    ----------
    w : torch.Tensor
        Position tensor.  Shape ``[B, T]`` or ``[T]``.
    asset_returns : torch.Tensor
        Realised simple returns for the traded asset.  Shape must be
        broadcastable with ``w``.
    spread_bps : float
        Effective bid–ask spread (one‐way) in basis points.
    fee_bps : float
        Miscellaneous fee component in basis points.
    detach_prev : bool
        Whether to detach ``w_{t-1}`` when computing turnover costs.

    Returns
    -------
    torch.Tensor
        Net PnL series of the same shape as ``w``.
    """
    # broadcast asset_returns to match shape of positions
    if asset_returns.dim() == 1:
        asset_returns = asset_returns.unsqueeze(0).expand_as(w)
    # gross PnL uses previous positions
    prev_w = w.roll(shifts=1, dims=-1)
    # at t=0 there is no previous position
    if w.dim() == 1:
        prev_w[0] = 0.0
    else:
        prev_w[:, 0] = 0.0
    gross = prev_w * asset_returns
    costs = trading_cost(turnover(w, detach_prev=detach_prev),
                         spread_bps=spread_bps,
                         fee_bps=fee_bps)
    return gross - costs

def positions_to_returns_from_signal(signal: torch.Tensor,
                                     asset_returns: torch.Tensor,
                                     w_cap: float = 1.0,
                                     beta: float = 2.0,
                                     spread_bps: float = 1.0,
                                     fee_bps: float = 0.1,
                                     detach_prev: bool = True) -> torch.Tensor:
    """Convert raw model signals to net PnL via smooth positions.

    This convenience function chains the mapping ``signal -> position``
    using ``smooth_position_from_signal`` and then passes the result
    through ``positions_to_returns``.  It is the preferred entry point
    when computing portfolio losses from raw network outputs.

    Parameters
    ----------
    signal : torch.Tensor
        Raw model outputs.  Shape ``[B, T]`` or ``[T]``.
    asset_returns : torch.Tensor
        Simple returns for the traded asset.  Shape must be
        broadcastable with ``signal``.
    w_cap : float
        Maximum absolute position.
    beta : float
        Temperature controlling the steepness of the tanh mapping.
    spread_bps : float
        Spread component of transaction costs.
    fee_bps : float
        Fee component of transaction costs.
    detach_prev : bool
        Whether to detach previous positions when computing turnover.

    Returns
    -------
    torch.Tensor
        Net PnL series of the same shape as ``signal``.
    """
    w = smooth_position_from_signal(signal, w_cap=w_cap, beta=beta)
    return positions_to_returns(w, asset_returns,
                               spread_bps=spread_bps,
                               fee_bps=fee_bps,
                               detach_prev=detach_prev)

def soft_barrier_mix(r_base: torch.Tensor,
                     r_stop: torch.Tensor,
                     r_target: torch.Tensor,
                     high: torch.Tensor,
                     low: torch.Tensor,
                     stop_level: torch.Tensor,
                     target_level: torch.Tensor,
                     k: float = 25.0) -> torch.Tensor:
    """Smooth mixture of base, stop and target returns.

    Given a base return series ``r_base`` and alternative return
    series for stop‐loss and profit‐target behaviours (``r_stop`` and
    ``r_target``), this function constructs a differentiable blend
    using soft indicator functions based on the high and low price
    series.  The resulting return is a weighted sum where weights
    transition smoothly from 0 to 1 as the price approaches stop
    and target levels.  This can be used to model stop‐loss and
    take‐profit exits in a differentiable manner.

    Parameters
    ----------
    r_base : torch.Tensor
        Base return series.
    r_stop : torch.Tensor
        Return series if stop loss is triggered.
    r_target : torch.Tensor
        Return series if profit target is triggered.
    high : torch.Tensor
        Series of high prices.
    low : torch.Tensor
        Series of low prices.
    stop_level : torch.Tensor
        Stop loss level.
    target_level : torch.Tensor
        Profit target level.
    k : float
        Steepness of the sigmoid used for the soft indicator.

    Returns
    -------
    torch.Tensor
        Blended return series of the same shape as inputs.
    """
    # soft indicators for stop and target conditions
    p_stop = torch.sigmoid(k * (stop_level - low))
    p_target = torch.sigmoid(k * (high - target_level))
    p_none = 1.0 - torch.clamp(p_stop + p_target, max=1.0)
    return p_none * r_base + p_stop * r_stop + p_target * r_target