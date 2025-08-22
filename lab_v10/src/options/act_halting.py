"""
act_halting.py
==============

Advanced Adaptive Computation Time (ACT) halting mechanism for the HRM.
This module implements sophisticated halting logic with Q-learning style
targets and uncertainty-based stopping criteria.

Key Features:
- Learned halting probabilities with temperature scaling
- Dynamic ponder cost adjustment based on complexity
- Q-learning style targets for halting decisions
- Uncertainty-based early stopping
- Memory-efficient computation tracking
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ACTConfig:
    """Configuration for ACT halting mechanism."""
    max_steps: int = 8
    threshold: float = 0.99  # Cumulative probability threshold for halting
    ponder_cost: float = 0.01  # Cost per computation step
    temperature: float = 1.0  # Temperature for halting probability
    min_steps: int = 1  # Minimum computation steps
    use_uncertainty: bool = True  # Use uncertainty for halting decisions
    q_learning_alpha: float = 0.1  # Q-learning update rate


class ACTHaltingHead(nn.Module):
    """Learned halting head with Q-learning style targets."""
    
    def __init__(self, hidden_dim: int, config: ACTConfig):
        super().__init__()
        self.config = config
        
        # Main halting network
        self.halt_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2, bias=False)  # [continue, halt] logits
        )
        
        # Uncertainty estimation head (optional)
        if config.use_uncertainty:
            self.uncertainty_head = nn.Linear(hidden_dim, 1, bias=False)
        else:
            self.uncertainty_head = None
            
        # Learnable temperature parameter
        self.temperature = nn.Parameter(torch.tensor(config.temperature))
        
    def forward(self, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute halting probabilities and optional uncertainty.
        
        Args:
            hidden_state: [B, D] hidden state tensor
            
        Returns:
            halt_logits: [B, 2] logits for [continue, halt]
            uncertainty: [B, 1] uncertainty estimate (optional)
        """
        # Compute halt logits with temperature scaling
        halt_logits = self.halt_net(hidden_state) / self.temperature
        
        # Compute uncertainty if enabled
        uncertainty = None
        if self.uncertainty_head is not None:
            uncertainty = torch.sigmoid(self.uncertainty_head(hidden_state))
            
        return halt_logits, uncertainty


class ACTController:
    """Controller for managing ACT halting during forward pass."""
    
    def __init__(self, config: ACTConfig, device: str = 'cpu'):
        self.config = config
        self.device = device
        
        # Q-learning state for target generation
        self.q_values = {}  # Store Q-values for different states
        self.step_rewards = []  # Track rewards per step
        
    def should_halt(self, 
                   halt_logits: torch.Tensor, 
                   step: int, 
                   cumulative_probs: torch.Tensor,
                   uncertainty: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Determine which examples should halt at the current step.
        
        Args:
            halt_logits: [B, 2] logits for [continue, halt]
            step: Current computation step
            cumulative_probs: [B] cumulative halting probabilities so far
            uncertainty: [B, 1] uncertainty estimates (optional)
            
        Returns:
            should_halt: [B] boolean mask indicating which examples should halt
            halt_probs: [B] halting probabilities for this step
        """
        B = halt_logits.shape[0]
        
        # Compute halting probabilities
        halt_probs = F.softmax(halt_logits, dim=-1)[:, 1]  # Probability of halting
        
        # Apply minimum steps constraint
        if step < self.config.min_steps:
            should_halt = torch.zeros(B, dtype=torch.bool, device=self.device)
            return should_halt, halt_probs
            
        # Compute updated cumulative probabilities
        new_cumulative = cumulative_probs + halt_probs
        
        # Base halting decision on threshold
        threshold_halt = new_cumulative >= self.config.threshold
        
        # Apply uncertainty-based early stopping if enabled
        if uncertainty is not None and self.config.use_uncertainty:
            # Halt if uncertainty is low (confident)
            confident_halt = uncertainty.squeeze(-1) < 0.3
            should_halt = threshold_halt | confident_halt
        else:
            should_halt = threshold_halt
            
        # Force halt at maximum steps
        if step >= self.config.max_steps - 1:
            should_halt = torch.ones(B, dtype=torch.bool, device=self.device)
            
        return should_halt, halt_probs
    
    def compute_q_targets(self, 
                         outputs_sequence: list,
                         final_loss: torch.Tensor,
                         steps_used: torch.Tensor) -> list:
        """
        Compute Q-learning style targets for halting decisions.
        
        Args:
            outputs_sequence: List of (output, halt_logits, uncertainty) tuples
            final_loss: [B] final task loss for each example
            steps_used: [B] number of steps used by each example
            
        Returns:
            q_targets: List of target labels for each step
        """
        num_steps = len(outputs_sequence)
        B = final_loss.shape[0]
        
        q_targets = []
        
        for step in range(num_steps):
            # Compute reward for halting at this step vs continuing
            step_cost = self.config.ponder_cost * (step + 1)
            
            # Create targets based on optimal stopping theory
            # Earlier stopping is rewarded if loss doesn't improve much
            if step == num_steps - 1:
                # Always halt at last step
                targets = torch.ones(B, dtype=torch.long, device=self.device)
            else:
                # Determine if halting at this step would be optimal
                # This is a simplified heuristic - in practice you might use
                # more sophisticated optimal stopping criteria
                future_improvement = 0.1  # Expected improvement from more computation
                should_halt_target = (step_cost + final_loss) < (final_loss - future_improvement)
                targets = should_halt_target.long()
                
            q_targets.append(targets)
            
        return q_targets
    
    def compute_ponder_cost(self, steps_used: torch.Tensor) -> torch.Tensor:
        """
        Compute adaptive ponder cost based on steps used.
        
        Args:
            steps_used: [B] number of steps used per example
            
        Returns:
            ponder_cost: [B] adaptive ponder cost per example
        """
        # Progressive ponder cost - more expensive for longer sequences
        base_cost = self.config.ponder_cost
        progressive_multiplier = torch.pow(1.2, steps_used - 1)  # 1.2^(steps-1)
        
        return base_cost * progressive_multiplier
    
    def reset_episode(self):
        """Reset controller state for new episode."""
        self.step_rewards = []
        
        
def act_loss(halt_logits_sequence: list, 
            q_targets: list, 
            steps_used: torch.Tensor,
            config: ACTConfig) -> torch.Tensor:
    """
    Compute comprehensive ACT loss including Q-head loss and ponder cost.
    
    Args:
        halt_logits_sequence: List of [B, 2] halt logits for each step
        q_targets: List of [B] target labels for each step
        steps_used: [B] number of steps used per example
        config: ACT configuration
        
    Returns:
        total_act_loss: Scalar ACT loss
    """
    if not halt_logits_sequence:
        return torch.tensor(0.0, device=steps_used.device)
    
    # Q-head classification loss
    q_loss = 0.0
    for halt_logits, targets in zip(halt_logits_sequence, q_targets):
        q_loss += F.cross_entropy(halt_logits, targets)
    q_loss = q_loss / len(halt_logits_sequence)
    
    # Adaptive ponder cost
    controller = ACTController(config, device=steps_used.device)
    ponder_costs = controller.compute_ponder_cost(steps_used)
    ponder_loss = ponder_costs.mean()
    
    return q_loss + ponder_loss


def compute_act_weights(halt_probs_sequence: list, 
                       should_halt_sequence: list) -> torch.Tensor:
    """
    Compute ACT weights for weighted output combination.
    
    Args:
        halt_probs_sequence: List of [B] halt probabilities for each step
        should_halt_sequence: List of [B] boolean halt decisions for each step
        
    Returns:
        weights: [num_steps, B] normalized weights for each step and example
    """
    if not halt_probs_sequence:
        return torch.ones(1, halt_probs_sequence[0].shape[0])
    
    num_steps = len(halt_probs_sequence)
    B = halt_probs_sequence[0].shape[0]
    device = halt_probs_sequence[0].device
    
    weights = torch.zeros(num_steps, B, device=device)
    remainders = torch.zeros(B, device=device)
    
    for step, (halt_probs, should_halt) in enumerate(zip(halt_probs_sequence, should_halt_sequence)):
        # Weight is halt_prob for examples that should halt, 
        # remainder for examples continuing to next step
        step_weights = torch.where(should_halt, 1.0 - remainders, halt_probs)
        weights[step] = step_weights
        
        # Update remainders for continuing examples
        remainders += halt_probs
        remainders = torch.clamp(remainders, max=1.0)
    
    # Normalize weights to sum to 1 for each example
    weight_sums = weights.sum(dim=0, keepdim=True)
    weights = weights / (weight_sums + 1e-8)
    
    return weights