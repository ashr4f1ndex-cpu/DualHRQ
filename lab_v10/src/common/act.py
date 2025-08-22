"""
act.py
======

Adaptive Computation Time (ACT) mechanism for the Hierarchical Reasoning Model (HRM).
Provides both utility functions and a structured ACT controller class for managing
adaptive computation with continuous learning and optimization.

Classes
-------
AdaptiveComputationTime
    Controller class for managing ACT behavior with learning capabilities.

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
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


def act_ponder_penalty(steps_used: torch.Tensor, ponder_cost: float) -> torch.Tensor:
    """Compute the ACT ponder penalty.

    Parameters
    ----------
    steps_used : torch.Tensor
        A one-dimensional tensor of shape (B, ) where B is the batch size
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
        A tensor of shape (B, ) containing the ground truth labels
        indicating whether the model should continue (0) or halt (1).

    Returns
    -------
    torch.Tensor
        The mean cross entropy loss for the Q-head.
    """
    return F.cross_entropy(logits, q_targets.long())


class AdaptiveComputationTime(nn.Module):
    """
    Advanced ACT controller with continuous learning capabilities.
    
    This controller manages adaptive computation decisions, learns from
    computation patterns, and optimizes resource usage over time.
    """
    
    def __init__(
        self, 
        threshold: float = 0.01,
        max_steps: int = 8,
        ponder_cost: float = 0.01,
        learning_rate: float = 0.1,
        adaptation_window: int = 100
    ):
        super().__init__()
        self.threshold = threshold
        self.max_steps = max_steps
        self.ponder_cost = ponder_cost
        self.learning_rate = learning_rate
        self.adaptation_window = adaptation_window
        
        # Learning state
        self.computation_history: List[Dict] = []
        self.efficiency_metrics: List[float] = []
        self.adaptation_count = 0
        
        # Adaptive thresholds (learnable parameters)
        self.adaptive_threshold = nn.Parameter(torch.tensor(threshold))
        self.confidence_bias = nn.Parameter(torch.zeros(1))
        
        logger.info(f"ACT Controller initialized with threshold={threshold}, max_steps={max_steps}")
    
    def should_halt(self, halt_logits: torch.Tensor, step: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Determine halting decisions with adaptive thresholds.
        
        Parameters
        ----------
        halt_logits : torch.Tensor
            Raw logits from Q-head [continue, halt] of shape (B, 2)
        step : int
            Current computation step
            
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            (halt_decisions, halt_probabilities)
        """
        # Apply learned confidence bias
        adjusted_logits = halt_logits + self.confidence_bias
        halt_probs = torch.sigmoid(adjusted_logits[:, 1])  # Probability of halting
        
        # Adaptive threshold based on step and learned parameters
        current_threshold = self.adaptive_threshold * (1.0 + 0.1 * step)
        
        # Make halting decisions
        halt_decisions = halt_probs > current_threshold
        
        # Force halt at max steps
        if step >= self.max_steps - 1:
            halt_decisions = torch.ones_like(halt_decisions, dtype=torch.bool)
        
        return halt_decisions, halt_probs
    
    def compute_ponder_loss(self, steps_used: torch.Tensor) -> torch.Tensor:
        """Compute adaptive ponder penalty with learning."""
        base_penalty = act_ponder_penalty(steps_used, self.ponder_cost)
        
        # Add adaptive component based on efficiency history
        if len(self.efficiency_metrics) > 10:
            avg_efficiency = sum(self.efficiency_metrics[-10:]) / 10
            efficiency_factor = 1.0 / (avg_efficiency + 1e-6)
            adaptive_penalty = base_penalty * efficiency_factor
        else:
            adaptive_penalty = base_penalty
            
        return adaptive_penalty
    
    def update_efficiency_metrics(self, actual_steps: torch.Tensor, target_accuracy: torch.Tensor):
        """Update efficiency metrics based on performance."""
        avg_steps = actual_steps.float().mean().item()
        avg_accuracy = target_accuracy.float().mean().item()
        
        # Compute efficiency as accuracy per computational step
        efficiency = avg_accuracy / (avg_steps + 1e-6)
        self.efficiency_metrics.append(efficiency)
        
        # Keep only recent metrics
        if len(self.efficiency_metrics) > self.adaptation_window:
            self.efficiency_metrics = self.efficiency_metrics[-self.adaptation_window:]
        
        # Record computation pattern
        self.computation_history.append({
            'avg_steps': avg_steps,
            'avg_accuracy': avg_accuracy, 
            'efficiency': efficiency,
            'threshold': self.adaptive_threshold.item(),
            'step_count': len(self.efficiency_metrics)
        })
    
    def adapt_parameters(self):
        """Adapt ACT parameters based on learning history."""
        if len(self.efficiency_metrics) < 20:  # Need sufficient history
            return
            
        recent_efficiency = self.efficiency_metrics[-10:]
        historical_efficiency = self.efficiency_metrics[-20:-10]
        
        recent_avg = sum(recent_efficiency) / len(recent_efficiency)
        historical_avg = sum(historical_efficiency) / len(historical_efficiency)
        
        # If efficiency is declining, adapt
        if recent_avg < historical_avg * 0.95:  # 5% degradation threshold
            # Increase threshold to reduce computation
            with torch.no_grad():
                self.adaptive_threshold *= (1.0 + self.learning_rate)
                self.adaptive_threshold.clamp_(0.001, 0.1)  # Reasonable bounds
                
            self.adaptation_count += 1
            logger.info(f"ACT adapted threshold to {self.adaptive_threshold.item():.4f} "
                       f"(adaptation #{self.adaptation_count})")
        
        # If efficiency is very high, we can be more aggressive
        elif recent_avg > historical_avg * 1.1:  # 10% improvement
            with torch.no_grad():
                self.adaptive_threshold *= (1.0 - self.learning_rate * 0.5)
                self.adaptive_threshold.clamp_(0.001, 0.1)
                
            self.adaptation_count += 1
            logger.info(f"ACT reduced threshold to {self.adaptive_threshold.item():.4f} "
                       f"for more computation (adaptation #{self.adaptation_count})")
    
    def get_learning_metrics(self) -> Dict[str, float]:
        """Get current learning and adaptation metrics."""
        if not self.efficiency_metrics:
            return {'efficiency': 0.0, 'adaptations': 0, 'threshold': self.threshold}
        
        recent_efficiency = self.efficiency_metrics[-5:] if len(self.efficiency_metrics) >= 5 else self.efficiency_metrics
        
        return {
            'current_efficiency': sum(recent_efficiency) / len(recent_efficiency),
            'total_adaptations': self.adaptation_count,
            'current_threshold': self.adaptive_threshold.item(),
            'confidence_bias': self.confidence_bias.item(),
            'computation_samples': len(self.computation_history),
            'max_efficiency': max(self.efficiency_metrics) if self.efficiency_metrics else 0.0,
            'min_efficiency': min(self.efficiency_metrics) if self.efficiency_metrics else 0.0
        }
    
    def reset_learning_state(self):
        """Reset learning state for new training phase."""
        self.computation_history.clear()
        self.efficiency_metrics.clear()
        self.adaptation_count = 0
        
        with torch.no_grad():
            self.adaptive_threshold.fill_(self.threshold)
            self.confidence_bias.fill_(0.0)
            
        logger.info("ACT learning state reset")
