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
from typing import TYPE_CHECKING
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .hrm_net import HRMNet, HRMConfig
from ..common.act import q_head_loss, act_ponder_penalty


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
        # parameters for uncertainty weighting: learn log variances
        self.log_sA = nn.Parameter(torch.zeros(1, device=self.device))
        self.log_sB = nn.Parameter(torch.zeros(1, device=self.device))

    def _make_optimizer(self, steps: int):
        params = list(self.net.parameters()) + [self.log_sA, self.log_sB]
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
                    # forward pass; returns different structures depending on ACT
                    out = self.net(H, L)
                    if self.net.cfg.act_enable:
                        # output: (h_tokens, l_tokens), (outA_final, outB_final, q_list, weights)
                        (_, _), (outA_final, outB_final, q_list, weights) = out
                        # supervised losses on final weighted outputs
                        lossA_avg = torch.mean((outA_final - yA) ** 2)
                        lossB_avg = torch.mean(
                            torch.nn.functional.binary_cross_entropy_with_logits(outB_final, yB)
                        )
                        # uncertainty weighting
                        total = (
                            torch.exp(-self.log_sA) * lossA_avg
                            + torch.exp(-self.log_sB) * lossB_avg
                            + self.log_sA + self.log_sB
                        )
                        # Q-head loss: cross entropy across all segments
                        if q_list:
                            num_steps = len(q_list)
                            # build target: 0 for all but last, 1 for last
                            Bsz = q_list[0].shape[0]
                            q_targets = torch.zeros((num_steps, Bsz), dtype=torch.long, device=self.device)
                            q_targets[-1] = 1  # final segment halts
                            # compute cross-entropy per step and average
                            q_loss_sum = 0.0
                            for s in range(num_steps):
                                q_loss_sum += q_head_loss(q_list[s], q_targets[s])
                            q_loss = q_loss_sum / num_steps
                        else:
                            q_loss = torch.tensor(0.0, device=self.device)
                        # Ponder penalty: penalise number of segments used
                        steps_used = torch.full((H.shape[0],), fill_value=len(q_list), device=self.device)
                        ponder = act_ponder_penalty(steps_used, self.net.cfg.ponder_cost)
                        total = total + q_loss + ponder
                    else:
                        # deep supervision branch: we receive segments list
                        (_, _), segments = out
                        num_segments = len(segments)
                        lossA_sum = 0.0
                        lossB_sum = 0.0
                        for outA_s, outB_s, _ in segments:
                            lossA_sum += torch.mean((outA_s - yA) ** 2)
                            lossB_sum += torch.mean(
                                torch.nn.functional.binary_cross_entropy_with_logits(outB_s, yB)
                            )
                        lossA_avg = lossA_sum / num_segments
                        lossB_avg = lossB_sum / num_segments
                        # uncertainty weighting of supervised losses
                        total = (
                            torch.exp(-self.log_sA) * lossA_avg
                            + torch.exp(-self.log_sB) * lossB_avg
                            + self.log_sA + self.log_sB
                        )
                        # optional ACT auxiliary loss (if act_enable is False this block not executed)
                        if getattr(self.net.cfg, "act_enable", False):
                            q_logits_last = segments[-1][2]
                            Bsz = q_logits_last.shape[0]
                            q_targets = torch.ones(Bsz, dtype=torch.long, device=self.device)
                            q_loss = q_head_loss(q_logits_last, q_targets)
                            steps_used = torch.full((Bsz,), fill_value=num_segments, device=self.device)
                            ponder = act_ponder_penalty(steps_used, self.net.cfg.ponder_cost)
                            total = total + q_loss + ponder
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
                    if self.net.cfg.act_enable:
                        (_, _), (outA_final, outB_final, q_list, weights) = out
                        # compute supervised validation loss
                        lossA_avg = torch.mean((outA_final - yA) ** 2)
                        lossB_avg = torch.mean(
                            torch.nn.functional.binary_cross_entropy_with_logits(outB_final, yB)
                        )
                        total_val = lossA_avg + lossB_avg
                    else:
                        (_, _), segments = out
                        num_segments = len(segments)
                        lossA_sum = 0.0
                        lossB_sum = 0.0
                        for outA_s, outB_s, _ in segments:
                            lossA_sum += torch.mean((outA_s - yA) ** 2)
                            lossB_sum += torch.mean(
                                torch.nn.functional.binary_cross_entropy_with_logits(outB_s, yB)
                            )
                        lossA_avg = lossA_sum / num_segments
                        lossB_avg = lossB_sum / num_segments
                        total_val = lossA_avg + lossB_avg
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