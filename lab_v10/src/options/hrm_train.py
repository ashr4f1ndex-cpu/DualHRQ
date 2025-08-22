"""
hrm_train.py
============

Basic training utilities for the HRM model.  This module defines a
simple trainer class that takes care of batching data, running a
forward/backward pass on the network, and applying gradient updates.
It also implements early stopping and optional gradient clipping.

Note: This implementation is deliberately minimal; it supports two
tasks (Head-A regression and Head-B binary classification) and
uncertainty‐weighted loss aggregation.  It can be extended to
support more sophisticated objectives or output heads as needed.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ..common.act import act_ponder_penalty, q_head_loss


def heteroscedastic_loss(y_pred_tuple, y_true):
    """Compute heteroscedastic regression loss.
    
    Args:
        y_pred_tuple: (mu, log_var) predictions
        y_true: true targets
    
    Returns:
        negative log-likelihood loss
    """
    mu, log_var = y_pred_tuple
    var = torch.exp(log_var)
    loss = 0.5 * (torch.log(var) + (y_true - mu)**2 / var)
    return loss.mean()


def compute_task_losses(outA, outB, yA, yB, use_heteroscedastic=True):
    """Compute individual task losses.
    
    Returns:
        lossA, lossB (both scalar tensors)
    """
    # Head-A loss (vol-gap regression)
    if use_heteroscedastic and isinstance(outA, tuple):
        lossA = heteroscedastic_loss(outA, yA)
    else:
        if isinstance(outA, tuple):
            outA = outA[0]  # Use only mu if tuple but heteroscedastic disabled
        lossA = torch.mean((outA - yA) ** 2)
    
    # Head-B loss (intraday trigger classification)
    lossB = torch.mean(torch.nn.functional.binary_cross_entropy_with_logits(outB, yB))
    
    return lossA, lossB
from .hrm_net import HRMNet

# Import GradNorm from the common module
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from src.common.gradnorm import GradNorm


@dataclass
class TrainConfig:
    lr: float = 1e-4
    weight_decay: float = 0.01
    batch_size: int = 64
    max_epochs: int = 30
    precision: str = "bf16"
    grad_clip: float = 1.0
    early_stop_patience: int = 5
    schedule: str = "cosine"
    use_gradnorm: bool = False
    gradnorm_alpha: float = 0.5
    uncertainty_weighting: bool = True


class MultiTaskDataset(Dataset):
    """A simple container to hold H/L tokens and labels for two tasks."""

    def __init__(self, H: torch.Tensor, L: torch.Tensor, yA: torch.Tensor, yB: torch.Tensor):
        assert len(H) == len(L) == len(yA) == len(yB)
        self.H = H
        self.L = L
        self.yA = yA
        self.yB = yB

    def __len__(self) -> int:
        return self.H.shape[0]

    def __getitem__(self, idx: int):
        return self.H[idx], self.L[idx], self.yA[idx], self.yB[idx]


class HRMTrainer:
    """Trainer class for HRM models."""

    def __init__(self, net: HRMNet, cfg: TrainConfig):
        self.net = net
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.net.to(self.device)
        
        # Multi-task learning parameters
        if cfg.uncertainty_weighting:
            # parameters for uncertainty weighting: learn log variances
            self.log_sA = nn.Parameter(torch.zeros(1, device=self.device))
            self.log_sB = nn.Parameter(torch.zeros(1, device=self.device))
        else:
            self.log_sA = None
            self.log_sB = None
            
        # GradNorm setup
        if cfg.use_gradnorm:
            initial_task_weights = torch.ones(2, device=self.device)
            self.gradnorm = GradNorm(initial_task_weights, alpha=cfg.gradnorm_alpha)
            self.task_weights = nn.Parameter(initial_task_weights.clone())
        else:
            self.gradnorm = None
            self.task_weights = None

    def _make_optimizer(self, steps: int):
        params = list(self.net.parameters())
        
        # Add uncertainty weighting parameters if enabled
        if self.cfg.uncertainty_weighting:
            params.extend([self.log_sA, self.log_sB])
            
        # Add GradNorm task weights if enabled
        if self.cfg.use_gradnorm:
            params.append(self.task_weights)
            
        opt = optim.AdamW(params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        if self.cfg.schedule == "cosine":
            sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps * self.cfg.max_epochs)
        else:
            sched = optim.lr_scheduler.LambdaLR(opt, lambda step: 1.0)
        return opt, sched

    def _precision_context(self):
        # use torch.autocast if bfloat16/fp16 is requested
        if self.cfg.precision.lower() == "bf16":
            return torch.autocast(self.device, dtype=torch.bfloat16)
        if self.cfg.precision.lower() == "fp16":
            return torch.autocast(self.device, dtype=torch.float16)
        class Dummy:
            def __enter__(self):
                pass
            def __exit__(self, exc_type, exc_val, exc_tb):
                pass
        return Dummy()

    def fit(self, train_dl: DataLoader, val_dl: DataLoader) -> dict[str, float]:
        # set up optimiser and scheduler
        opt, sched = self._make_optimizer(len(train_dl))
        best = float("inf")
        bad = 0
        for epoch in range(self.cfg.max_epochs):
            self.net.train()
            for H, L, yA, yB in tqdm(train_dl, desc=f"train epoch {epoch}"):
                # move data to device
                H, L = H.to(self.device), L.to(self.device)
                yA, yB = yA.to(self.device), yB.to(self.device)
                opt.zero_grad(set_to_none=True)
                with self._precision_context():
                    # forward pass; returns different structures depending on DEQ mode
                    out = self.net(H, L)
                    
                    if self.net.cfg.deq_style and self.net.cfg.act_enable:
                        # DEQ + ACT mode: (h_tokens, l_tokens), (outA_final, outB_final, all_outputs, n_updates)
                        (_, _), (outA_final, outB_final, all_outputs, n_updates) = out
                        
                        # Compute task losses
                        lossA, lossB = compute_task_losses(
                            outA_final, outB_final, yA, yB, 
                            use_heteroscedastic=self.net.cfg.use_heteroscedastic
                        )
                        
                        # Multi-task loss aggregation
                        if self.cfg.use_gradnorm and self.gradnorm is not None:
                            # GradNorm weighting
                            shared_params = [p for n, p in self.net.named_parameters() 
                                           if 'headA' not in n and 'headB' not in n]
                            gradnorm_loss = self.gradnorm.step([lossA, lossB], shared_params)
                            total = self.task_weights[0] * lossA + self.task_weights[1] * lossB + gradnorm_loss
                        elif self.cfg.uncertainty_weighting and self.log_sA is not None:
                            # Uncertainty weighting
                            total = (
                                torch.exp(-self.log_sA) * lossA + self.log_sA +
                                torch.exp(-self.log_sB) * lossB + self.log_sB
                            )
                        else:
                            # Simple equal weighting
                            total = lossA + lossB
                        
                        # ACT auxiliary losses
                        if all_outputs:
                            # Q-head loss across all steps
                            q_loss_sum = 0.0
                            for step_outputs in all_outputs:
                                _, _, q_logits = step_outputs
                                # Target: halt with probability inversely related to step number
                                step_idx = all_outputs.index(step_outputs)
                                is_last_step = (step_idx == len(all_outputs) - 1)
                                Bsz = q_logits.shape[0]
                                q_targets = torch.ones(Bsz, dtype=torch.long, device=self.device) if is_last_step else torch.zeros(Bsz, dtype=torch.long, device=self.device)
                                q_loss_sum += q_head_loss(q_logits, q_targets)
                            q_loss = q_loss_sum / len(all_outputs)
                            
                            # Ponder penalty
                            ponder = act_ponder_penalty(n_updates, self.net.cfg.ponder_cost)
                            total = total + q_loss + ponder
                            
                    elif not self.net.cfg.deq_style:
                        # Original deep supervision mode
                        (_, _), segments = out
                        num_segments = len(segments)
                        lossA_sum = 0.0
                        lossB_sum = 0.0
                        
                        for outA_s, outB_s, _ in segments:
                            lossA_s, lossB_s = compute_task_losses(
                                outA_s, outB_s, yA, yB,
                                use_heteroscedastic=self.net.cfg.use_heteroscedastic
                            )
                            lossA_sum += lossA_s
                            lossB_sum += lossB_s
                            
                        lossA_avg = lossA_sum / num_segments
                        lossB_avg = lossB_sum / num_segments
                        
                        # Multi-task loss aggregation
                        if self.cfg.uncertainty_weighting and self.log_sA is not None:
                            total = (
                                torch.exp(-self.log_sA) * lossA_avg + self.log_sA +
                                torch.exp(-self.log_sB) * lossB_avg + self.log_sB
                            )
                        else:
                            total = lossA_avg + lossB_avg
                            
                        # Optional ACT auxiliary loss
                        if getattr(self.net.cfg, "act_enable", False):
                            q_logits_last = segments[-1][2]
                            Bsz = q_logits_last.shape[0]
                            q_targets = torch.ones(Bsz, dtype=torch.long, device=self.device)
                            q_loss = q_head_loss(q_logits_last, q_targets)
                            steps_used = torch.full((Bsz,), fill_value=num_segments, device=self.device)
                            ponder = act_ponder_penalty(steps_used, self.net.cfg.ponder_cost)
                            total = total + q_loss + ponder
                    else:
                        # Fallback: simple forward
                        (_, _), (outA_final, outB_final, _) = out
                        lossA, lossB = compute_task_losses(
                            outA_final, outB_final, yA, yB,
                            use_heteroscedastic=self.net.cfg.use_heteroscedastic
                        )
                        total = lossA + lossB
                # backward pass
                total.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.grad_clip)
                opt.step()
                sched.step()
            # validation loop
            self.net.eval()
            val_loss = 0.0
            count = 0
            with torch.no_grad():
                for H, L, yA, yB in val_dl:
                    H, L = H.to(self.device), L.to(self.device)
                    yA, yB = yA.to(self.device), yB.to(self.device)
                    out = self.net(H, L)
                    
                    if self.net.cfg.deq_style and self.net.cfg.act_enable:
                        # DEQ + ACT mode
                        (_, _), (outA_final, outB_final, all_outputs, n_updates) = out
                        lossA, lossB = compute_task_losses(
                            outA_final, outB_final, yA, yB,
                            use_heteroscedastic=self.net.cfg.use_heteroscedastic
                        )
                        total_val = lossA + lossB
                    elif not self.net.cfg.deq_style:
                        # Deep supervision mode
                        (_, _), segments = out
                        num_segments = len(segments)
                        lossA_sum = 0.0
                        lossB_sum = 0.0
                        
                        for outA_s, outB_s, _ in segments:
                            lossA_s, lossB_s = compute_task_losses(
                                outA_s, outB_s, yA, yB,
                                use_heteroscedastic=self.net.cfg.use_heteroscedastic
                            )
                            lossA_sum += lossA_s
                            lossB_sum += lossB_s
                            
                        lossA_avg = lossA_sum / num_segments
                        lossB_avg = lossB_sum / num_segments
                        total_val = lossA_avg + lossB_avg
                    else:
                        # Fallback mode
                        (_, _), (outA_final, outB_final, _) = out
                        lossA, lossB = compute_task_losses(
                            outA_final, outB_final, yA, yB,
                            use_heteroscedastic=self.net.cfg.use_heteroscedastic
                        )
                        total_val = lossA + lossB
                    val_loss += total_val.item()
                    count += 1
            val_loss /= max(1, count)
            if val_loss < best:
                best, bad = val_loss, 0
                torch.save(self.net.state_dict(), ".hrm_best.pt")
            else:
                bad += 1
                if bad >= self.cfg.early_stop_patience:
                    break
        return {"val_loss": best}

    def load_best(self):
        state = torch.load(".hrm_best.pt", map_location=self.device)
        self.net.load_state_dict(state)
