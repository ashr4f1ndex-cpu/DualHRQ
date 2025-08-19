"""
act.py
======

Utilities to compute auxiliary quantities for the Adaptive Computation
Time (ACT) mechanism used in the Hierarchical Reasoning Model (HRM).
In practice, ACT introduces a ponder cost that penalizes models for
taking too many computational segments, and uses a Q-head to decide
whether to halt or continue at each reasoning segment.

Functions
---------

act_ponder_penalty
    Compute the mean ponder cost for a batch of examples given their
    number of segments used and a hyperparameter cost per segment.

q_head_loss
    Compute the loss for the Q-head of ACT using a cross entropy between
    logits and integer targets.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def act_ponder_penalty(steps_used: torch.Tensor, ponder_cost: float) -> torch.Tensor:
    """Compute the ACT ponder penalty.

    Parameters
    ----------
    steps_used : torch.Tensor
        A one-dimensional tensor of shape (B,) where B is the batch size
        containing the number of reasoning segments used by the model for
        each example.
    ponder_cost : float
        The cost per reasoning segment.  A higher value penalizes
        models for using more segments.

    Returns
    -------
    torch.Tensor
        The mean ponder cost for the batch.  This should be added to
        the total loss function during training when ACT is enabled.
    """
    return ponder_cost * steps_used.float().mean()


def q_head_loss(logits: torch.Tensor, q_targets: torch.Tensor) -> torch.Tensor:
    """Compute the Q-head loss for ACT.

    The Q-head is a small classifier that predicts whether the model
    should continue or halt at each segment.  This function uses the
    cross entropy loss on the logits against integer targets.

    Parameters
    ----------
    logits : torch.Tensor
        A tensor of shape (B, 2) containing the raw Q-head logits for
        the two classes: [continue, halt].
    q_targets : torch.Tensor
        A tensor of shape (B,) containing the ground truth labels
        indicating whether the model should continue (0) or halt (1).

    Returns
    -------
    torch.Tensor
        The mean cross entropy loss for the Q-head.
    """
    return F.cross_entropy(logits, q_targets.long())