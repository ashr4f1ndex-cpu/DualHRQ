"""
sim
===

Simulation utilities for differentiable portfolio optimisation.

This package exposes differentiable surrogate functions to map
predicted signals to portfolio returns.  See :mod:`.diff_exec` for
details.
"""

from .diff_exec import (
    soft_abs,
    smooth_position_from_signal,
    turnover,
    trading_cost,
    positions_to_returns,
    positions_to_returns_from_signal,
    soft_barrier_mix,
)

__all__ = [
    "soft_abs",
    "smooth_position_from_signal",
    "turnover",
    "trading_cost",
    "positions_to_returns",
    "positions_to_returns_from_signal",
    "soft_barrier_mix",
]