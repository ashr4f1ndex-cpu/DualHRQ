"""
portfolio_loss.py
===================

Differentiable surrogate loss functions for portfolio optimisation.  This
module provides numerically stable, PyTorch‐based implementations of
common risk‐adjusted return metrics (Sharpe, Sortino and CVaR) which may
be used as auxiliary objectives during model training.  It also
includes helper routines to enforce consistent tensor shapes and to
smooth out otherwise non‐differentiable operations such as the
value–at–risk hinge.

The functions defined here operate on sequences of returns.  Returns
may be passed either as a one‐dimensional tensor (interpreted as a
single batch example) or a two‐dimensional tensor of shape
``[B, T]`` where ``B`` is the batch size and ``T`` the number of
timesteps.  Internally, one‐dimensional inputs are reshaped to
``[1, T]`` to unify broadcasting.  All objectives return a scalar
``torch.Tensor`` when ``reduce=True``.  When ``reduce=False`` they
return a one‐dimensional tensor of length ``B`` containing per
example losses.

Example
-------

>>> import torch
>>> from src.common.portfolio_loss import sharpe_loss
>>> r = torch.tensor([[0.01, -0.02, 0.03],[0.02, -0.01, 0.01]])
>>> sharpe_loss(r)
tensor(???)

Note
----
These objectives should be used with care.  When combined with
standard supervised losses they can help nudge a model toward
favourable risk–adjusted behaviour, but they are not substitutes
for evaluating a trading strategy with a realistic simulator.
"""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F

# small value to prevent division by zero
EPS: float = 1e-8

def _ensure_2d(x: torch.Tensor) -> torch.Tensor:
    """Ensure that ``x`` has shape [B, T].

    If ``x`` is one dimensional, it is interpreted as a single batch
    example and reshaped to ``[1, T]``.  If ``x`` has more than two
    dimensions, the trailing dimensions are flattened into the time
    dimension via a mean.  This helper guarantees that the first
    dimension always indexes batch examples and the second dimension
    indexes timesteps.
    """
    if x.dim() == 1:
        return x.unsqueeze(0)
    if x.dim() > 2:
        b, t = x.shape[0], x.shape[1]
        return x.reshape(b, t, -1).mean(-1)
    return x

def stable_mean_std(x: torch.Tensor, eps: float = EPS) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute mean and standard deviation along the time axis.

    Parameters
    ----------
    x : torch.Tensor
        A tensor of shape ``[B, T]`` or ``[T]`` representing a batch
        of return sequences.
    eps : float
        Small constant added to the standard deviation to avoid
        division by zero.

    Returns
    -------
    mu : torch.Tensor
        The mean returns, shape ``[B]``.
    sigma : torch.Tensor
        The standard deviation of returns, shape ``[B]``.
    """
    x = _ensure_2d(x)
    mu = x.mean(dim=1)
    # population standard deviation (unbiased=False) for Sharpe
    sigma = x.std(dim=1, unbiased=False).clamp_min(eps)
    return mu, sigma

def sharpe_loss(returns: torch.Tensor, reduce: bool = True) -> torch.Tensor:
    """Compute the negative Sharpe ratio as a loss.

    The Sharpe ratio is the mean return divided by the standard
    deviation.  Since we want to maximise Sharpe, the loss is defined
    as the negative ratio.

    Parameters
    ----------
    returns : torch.Tensor
        A tensor of daily returns.  Shape ``[B, T]`` or ``[T]``.
    reduce : bool
        If True, average across the batch and return a scalar.  If
        False, return one value per batch example.

    Returns
    -------
    torch.Tensor
        The loss.  If ``reduce=True`` this is a scalar tensor;
        otherwise it is a one‐dimensional tensor of shape ``[B]``.
    """
    r = _ensure_2d(returns)
    mu, sigma = stable_mean_std(r)
    loss = -(mu / sigma)
    return loss.mean() if reduce else loss

def sortino_loss(returns: torch.Tensor, reduce: bool = True) -> torch.Tensor:
    """Compute the negative Sortino ratio as a loss.

    The Sortino ratio replaces the standard deviation in the Sharpe
    ratio with the downside deviation (i.e. the standard deviation of
    negative returns only).  A higher Sortino is better; therefore the
    loss is the negative ratio.  If all returns are positive the
    downside deviation becomes extremely small; a small epsilon is
    added for numerical stability.

    Parameters
    ----------
    returns : torch.Tensor
        A tensor of returns.  Shape ``[B, T]`` or ``[T]``.
    reduce : bool
        If True, average across the batch and return a scalar.  If
        False, return one value per batch example.

    Returns
    -------
    torch.Tensor
        The loss.  If ``reduce=True`` this is a scalar tensor;
        otherwise it is a one‐dimensional tensor of shape ``[B]``.
    """
    r = _ensure_2d(returns)
    mu = r.mean(dim=1)
    downside = torch.minimum(r, torch.zeros_like(r))
    dd = torch.sqrt((downside.pow(2)).mean(dim=1).clamp_min(EPS))
    loss = -(mu / dd)
    return loss.mean() if reduce else loss

def cvar_loss(returns: torch.Tensor,
              alpha: float = 0.95,
              k: float = 25.0,
              reduce: bool = True) -> torch.Tensor:
    """Compute a smooth Conditional Value‐at‐Risk (CVaR) loss.

    This function implements the Rockafellar–Uryasev formulation of
    CVaR using a soft hinge (``softplus``) to make the objective
    differentiable.  We treat ``returns`` as the portfolio returns,
    meaning that ``losses = -returns``.  CVaR is then the expected
    tail loss beyond the VaR at the specified confidence level.

    Parameters
    ----------
    returns : torch.Tensor
        A tensor of returns.  Shape ``[B, T]`` or ``[T]``.
    alpha : float
        Confidence level for CVaR (e.g. 0.95).  Higher values
        penalise deeper tail risk.
    k : float
        Steepness parameter for the softplus hinge.  Higher values
        make the approximation closer to the true hinge.
    reduce : bool
        If True, return the mean CVaR across the batch.  If False,
        return one value per batch example.

    Returns
    -------
    torch.Tensor
        The CVaR loss.  If ``reduce=True`` this is a scalar tensor;
        otherwise it is a one‐dimensional tensor of shape ``[B]``.
    """
    r = _ensure_2d(returns)
    losses = -r  # convert returns to losses
    # stop‐gradient quantile anchor; we detach to prevent backprop through the quantile
    q = torch.quantile(losses.detach(), alpha, dim=1, keepdim=True)
    hinge = F.softplus(k * (losses - q)) / k
    cvar = q + hinge.mean(dim=1, keepdim=True) / (1.0 - alpha)
    loss = cvar.squeeze(1)
    return loss.mean() if reduce else loss

def evar_loss(returns: torch.Tensor,
              tau: float = 10.0,
              reduce: bool = True) -> torch.Tensor:
    """Compute the entropic Value‐at‐Risk (EVaR) loss.

    EVaR is an alternative tail risk measure that upper bounds
    CVaR and is sometimes more stable in optimisation.  It is
    defined as ``EVaR_tau(L) = (1/tau) * log E[exp(tau * L)]``
    where ``L`` are portfolio losses.  We again convert returns
    to losses via ``losses = -returns``.

    Parameters
    ----------
    returns : torch.Tensor
        A tensor of returns.  Shape ``[B, T]`` or ``[T]``.
    tau : float
        Temperature parameter controlling the curvature.  Larger
        values place more weight on extreme losses.
    reduce : bool
        If True, return the mean EVaR across the batch.  If False,
        return one value per batch example.

    Returns
    -------
    torch.Tensor
        The EVaR loss.  If ``reduce=True`` this is a scalar tensor;
        otherwise it is a one‐dimensional tensor of shape ``[B]``.
    """
    r = _ensure_2d(returns)
    losses = -r
    # logsumexp along the time axis to compute E[exp(tau * L)]
    lse = torch.logsumexp(losses * tau, dim=1) - torch.log(torch.tensor(r.shape[1], dtype=r.dtype, device=r.device))
    evar = lse / tau
    return evar.mean() if reduce else evar

def make_portfolio_loss(kind: str,
                        **kwargs) -> callable:
    """Factory to create a portfolio loss function.

    Parameters
    ----------
    kind : str
        One of ``"sharpe"``, ``"sortino"``, ``"cvar"`` or
        ``"evar"``.
    **kwargs
        Additional keyword arguments forwarded to the underlying
        objective (e.g. ``alpha``, ``k_barrier`` for CVaR, ``tau`` for EVaR).

    Returns
    -------
    callable
        A function ``loss_fn(returns: torch.Tensor) -> torch.Tensor``
        computing the corresponding portfolio loss.
    """
    kind = (kind or "sharpe").lower()
    if kind == "sharpe":
        return lambda r: sharpe_loss(r, reduce=True)
    if kind == "sortino":
        return lambda r: sortino_loss(r, reduce=True)
    if kind == "cvar":
        alpha = float(kwargs.get("alpha", 0.95))
        k = float(kwargs.get("k_barrier", 25.0))
        return lambda r: cvar_loss(r, alpha=alpha, k=k, reduce=True)
    if kind == "evar":
        tau = float(kwargs.get("tau", 10.0))
        return lambda r: evar_loss(r, tau=tau, reduce=True)
    raise ValueError(f"Unknown portfolio loss kind: {kind}")