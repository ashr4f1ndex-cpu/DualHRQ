"""
hrm_adapter.py
==============

Adapter module that wraps the neural Hierarchical Reasoning Model (HRM)
and exposes a simple fit/predict interface akin to the ridge‐based
model used in earlier lab versions.  This adapter is responsible for
building token sequences, fitting scalers, instantiating the HRM
network and trainer, and producing predictions for both daily and
intraday tasks.

Usage
-----

```
adapter = HRMAdapter(config)
adapter.fit(X_daily, X_intraday, yA, yB_df, train_idx, val_idx)
mu_series = adapter.predict_daily_mu(X_daily)
proba_day = adapter.predict_intraday_proba_for_day(X_intraday, specific_day)
```

"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .hrm_input import FittedScalers, TokenConfig, fit_scalers, make_h_tokens, make_l_tokens_for_day
from .hrm_net import HRMConfig, HRMNet
from .hrm_train import HRMTrainer, MultiTaskDataset, TrainConfig


class HRMAdapter:
    """Wrap a neural HRM network for use in the trading lab.

    The adapter hides the details of tokenization, scaling, training
    and inference.  After calling :meth:`fit`, the attributes
    ``artifacts`` will hold the trained model and scalers.
    """

    def __init__(self, config: dict):
        self.config = config
        hrm_cfg = config.get("hrm", {})
        # tokenization settings
        self.token_cfg = TokenConfig(
            daily_window=hrm_cfg.get("tokens", {}).get("daily_window", 192),
            minutes_per_day=hrm_cfg.get("tokens", {}).get("minutes_per_day", 390),
        )
        self.artifacts: dict[str, any] | None = None

    def _build_model(self) -> HRMNet:
        hrm_cfg = self.config.get("hrm", {})
        h_cfg = hrm_cfg.get("h", {})
        l_cfg = hrm_cfg.get("l", {})
        # Use exact 26.82M parameter configuration (within 27M budget)
        cfg = HRMConfig(
            # H-module: 4 layers, 512 d_model, 8 heads, 384 FFN
            h_layers=h_cfg.get("layers", 4), 
            h_dim=h_cfg.get("d_model", 512), 
            h_heads=h_cfg.get("heads", 8),
            h_ffn_mult=h_cfg.get("ffn_mult", 0.75),  # 384 / 512 = 0.75
            h_dropout=h_cfg.get("dropout", 0.1),
            # L-module: 6 layers, 768 d_model, 12 heads, 384 FFN
            l_layers=l_cfg.get("layers", 6), 
            l_dim=l_cfg.get("d_model", 768), 
            l_heads=l_cfg.get("heads", 12),
            l_ffn_mult=l_cfg.get("ffn_mult", 0.5),  # 384 / 768 = 0.5
            l_dropout=l_cfg.get("dropout", 0.1),
            # Training configuration
            segments_N=hrm_cfg.get("segments_N", 4), 
            l_inner_T=hrm_cfg.get("l_inner_T", 4),
            act_enable=hrm_cfg.get("act", {}).get("enable", True),
            act_max_segments=hrm_cfg.get("act", {}).get("max_segments", 8),
            ponder_cost=hrm_cfg.get("act", {}).get("ponder_cost", 0.01),
            act_threshold=hrm_cfg.get("act", {}).get("threshold", 0.01),
            # Architecture features
            use_cross_attn=hrm_cfg.get("use_cross_attn", False),
            use_heteroscedastic=hrm_cfg.get("use_heteroscedastic", True),
            deq_style=hrm_cfg.get("deq_style", True),
            uncertainty_weighting=hrm_cfg.get("uncertainty_weighting", True),
        )
        return HRMNet(cfg)

    def fit(
        self,
        X_daily: pd.DataFrame,
        X_intraday: pd.DataFrame,
        yA: pd.Series,
        yB_df: pd.DataFrame,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
    ) -> None:
        """Fit the HRM model.

        Parameters
        ----------
        X_daily : pd.DataFrame
            Daily features indexed by date.
        X_intraday : pd.DataFrame
            Intraday features indexed by timestamp.
        yA : pd.Series
            Daily target for Head-A (volatility gap).  Indexed by date.
        yB_df : pd.DataFrame
            Minute-level target for Head-B.  Indexed by timestamp; the
            intraday dimension will be reduced to one value per day by
            taking the mean across each day's rows.
        train_idx : np.ndarray
            Integer indices for the training dates.
        val_idx : np.ndarray
            Integer indices for the validation dates.
        """
        # fit scalers on training data to avoid leakage
        scalers = fit_scalers(X_daily, X_intraday, train_idx)
        # build tokens for H-module
        train_days = X_daily.index[train_idx]
        val_days = X_daily.index[val_idx]
        H_train = make_h_tokens(X_daily, train_days, scalers, self.token_cfg)
        H_val = make_h_tokens(X_daily, val_days, scalers, self.token_cfg)
        # collapse minute-level target into one value per day by taking the mean
        def collapse_day(day):
            block = yB_df.loc[yB_df.index.normalize() == day.normalize()]
            return float(block.mean().values.squeeze()) if len(block) else 0.0
        yB_train = torch.tensor([collapse_day(d) for d in train_days], dtype=torch.float32)
        yB_val = torch.tensor([collapse_day(d) for d in val_days], dtype=torch.float32)
        # build L tokens
        L_train = torch.cat(
            [make_l_tokens_for_day(X_intraday, d, scalers, self.token_cfg) for d in train_days], dim=0
        )
        L_val = torch.cat(
            [make_l_tokens_for_day(X_intraday, d, scalers, self.token_cfg) for d in val_days], dim=0
        )
        # labels for Head-A
        yA_train = torch.from_numpy(yA.loc[train_days].values).float()
        yA_val = torch.from_numpy(yA.loc[val_days].values).float()
        # instantiate model and trainer
        model = self._build_model()
        # Training configuration with new multi-task learning options
        tcfg = TrainConfig(
            lr=self.config.get("hrm", {}).get("optimizer", {}).get("lr", 1e-4),
            weight_decay=self.config.get("hrm", {}).get("regularization", {}).get("weight_decay", 0.01),
            batch_size=self.config.get("hrm", {}).get("batch_size", 32),  # Reduced for 27M model
            max_epochs=self.config.get("hrm", {}).get("max_epochs", 50),
            precision=self.config.get("hrm", {}).get("precision", "bf16"),
            grad_clip=self.config.get("hrm", {}).get("regularization", {}).get("grad_clip_norm", 1.0),
            early_stop_patience=self.config.get("hrm", {}).get("early_stop_patience", 8),
            schedule=self.config.get("hrm", {}).get("optimizer", {}).get("schedule", "cosine"),
            # Multi-task learning configuration
            use_gradnorm=self.config.get("hrm", {}).get("losses", {}).get("gradnorm", {}).get("enabled", False),
            gradnorm_alpha=self.config.get("hrm", {}).get("losses", {}).get("gradnorm", {}).get("alpha", 0.5),
            uncertainty_weighting=self.config.get("hrm", {}).get("losses", {}).get("uncertainty_weighting", True),
        )
        trainer = HRMTrainer(model, tcfg)
        train_ds = MultiTaskDataset(H_train, L_train, yA_train, yB_train)
        val_ds = MultiTaskDataset(H_val, L_val, yA_val, yB_val)
        train_dl = DataLoader(train_ds, batch_size=tcfg.batch_size, shuffle=True, drop_last=False)
        val_dl = DataLoader(val_ds, batch_size=tcfg.batch_size, shuffle=False, drop_last=False)
        trainer.fit(train_dl, val_dl)
        trainer.load_best()
        # store artifacts
        self.artifacts = {
            "model": model,
            "scalers": scalers,
            "token_cfg": self.token_cfg,
            "device": trainer.device,
        }

    def predict_daily_mu(self, X_daily: pd.DataFrame) -> pd.Series:
        """Return daily predictions for Head-A (mu)."""
        assert self.artifacts is not None, "model not fitted"
        model: HRMNet = self.artifacts["model"]
        scalers: FittedScalers = self.artifacts["scalers"]
        token_cfg: TokenConfig = self.artifacts["token_cfg"]
        device: str = self.artifacts["device"]
        dates = X_daily.index
        H = make_h_tokens(X_daily, dates, scalers, token_cfg).to(device)
        model.eval()
        with torch.no_grad():
            # dummy L tokens because we only want H head
            L_dummy = torch.zeros(
                (H.shape[0], token_cfg.minutes_per_day, scalers.intraday.n_features_in_), device=device
            )
            out = model(H, L_dummy)
            
            # Handle different output formats
            if model.cfg.deq_style and model.cfg.act_enable:
                # DEQ + ACT mode: (h_tokens, l_tokens), (outA_final, outB_final, all_outputs, n_updates)
                (_, _), (outA, _, _, _) = out
            else:
                # Deep supervision mode: (h_tokens, l_tokens), segments
                (_, _), segments = out
                outA, _, _ = segments[-1]
            
            # Extract mean from heteroscedastic output if needed
            if isinstance(outA, tuple):
                mu = outA[0].cpu().numpy()  # Use mu, ignore log_var
            else:
                mu = outA.cpu().numpy()
                
        return pd.Series(mu, index=dates, name="hrm_mu")

    def predict_intraday_proba_for_day(
        self, X_intraday: pd.DataFrame, day: pd.Timestamp, X_daily: pd.DataFrame
    ) -> pd.Series:
        """Return a single Head-B prediction for a given day.

        Because the simplified HRM emits one output per day for Head-B,
        this method computes the intraday token sequence for ``day`` and
        returns the model's probability of the downside trigger.
        """
        assert self.artifacts is not None, "model not fitted"
        model: HRMNet = self.artifacts["model"]
        scalers: FittedScalers = self.artifacts["scalers"]
        token_cfg: TokenConfig = self.artifacts["token_cfg"]
        device: str = self.artifacts["device"]
        model.eval()
        with torch.no_grad():
            H = make_h_tokens(X_daily, pd.DatetimeIndex([day]), scalers, token_cfg).to(device)
            L = make_l_tokens_for_day(X_intraday, day, scalers, token_cfg).to(device)
            out = model(H, L)
            
            # Handle different output formats
            if model.cfg.deq_style and model.cfg.act_enable:
                # DEQ + ACT mode
                (_, _), (_, outB, _, _) = out
            else:
                # Deep supervision mode
                (_, _), segments = out
                _, outB, _ = segments[-1]
                
            proba = torch.sigmoid(outB.squeeze(0)).cpu().numpy()
        return pd.Series([proba], index=[day], name="hrm_intraday_proba")

# Backwards compatibility: alias HRMModel to HRMAdapter so existing tests
# referencing HRMModel continue to work.
HRMModel = HRMAdapter
