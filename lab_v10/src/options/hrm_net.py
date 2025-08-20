"""
hrm_net.py
==========

This module implements a simplified version of the Hierarchical
Reasoning Model (HRM) for the trading lab.  The HRM uses two
transformer encoders: a slow module (H-module) that operates on
daily‐frequency tokens and a fast module (L-module) that operates on
intraday tokens.  The H-module produces a context that conditions
the L-module via FiLM (feature-wise linear modulation).  The model
supports deep supervision at multiple reasoning segments and an
optional Adaptive Computation Time (ACT) mechanism.

Note: This is a simplified implementation designed to integrate into
the existing lab.  It omits some advanced features such as
heteroscedastic regression and per-minute outputs for Head-B, but it
provides a working neural network replacement for the ridge‐based
HRMAdapter used in v7.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common.act import act_ponder_penalty, q_head_loss


def rope_freqs(T: int, D: int, base: float = 10000.0, device=None):
    """Compute rotary positional embedding frequencies."""
    half = D // 2
    theta = torch.arange(half, device=device).float()
    inv_freq = 1.0 / (base ** (theta / half))
    t = torch.arange(T, device=device).float()
    freqs = torch.einsum("t,f->tf", t, inv_freq)
    return torch.stack([freqs.cos(), freqs.sin()], dim=-1)


def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply rotary positional embeddings to a tensor x."""
    B, T, D = x.shape
    x_ = x.view(B, T, D // 2, 2)
    cos, sin = freqs[..., 0], freqs[..., 1]
    xr, xi = x_[..., 0], x_[..., 1]
    rotr = xr * cos - xi * sin
    roti = xr * sin + xi * cos
    return torch.stack([rotr, roti], dim=-1).flatten(2)


class RMSNorm(nn.Module):
    """Root mean square normalization."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.w = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return self.w * x


class GLU(nn.Module):
    """Gated Linear Unit feed forward block."""

    def __init__(self, dim: int, mult: float = 3.0, dropout: float = 0.0):
        super().__init__()
        inner = int(dim * mult)
        self.fc = nn.Linear(dim, inner * 2, bias=False)
        self.proj = nn.Linear(inner, dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, g = self.fc(x).chunk(2, dim=-1)
        a = a * torch.sigmoid(g)
        return self.drop(self.proj(a))


class MHSA(nn.Module):
    """Multi-head self-attention without biases and with RoPE."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        assert dim % heads == 0
        self.dim = dim
        self.heads = heads
        self.dk = dim // heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        # apply RoPE
        q = apply_rope(q, rope)
        k = apply_rope(k, rope)
        # reshape into heads
        q = q.view(B, T, self.heads, self.dk).transpose(1, 2)
        k = k.view(B, T, self.heads, self.dk).transpose(1, 2)
        v = v.view(B, T, self.heads, self.dk).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)
        att = att.softmax(dim=-1)
        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.proj(out)


class EncoderBlock(nn.Module):
    """Single transformer block with RMSNorm, attention and GLU FFN."""

    def __init__(self, dim: int, heads: int, ffn_mult: float, dropout: float):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = MHSA(dim, heads)
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = RMSNorm(dim)
        self.ffn = GLU(dim, mult=ffn_mult, dropout=dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
        h = self.attn(self.norm1(x), rope)
        x = x + self.drop1(h)
        h = self.ffn(self.norm2(x))
        x = x + self.drop2(h)
        return x


class Encoder(nn.Module):
    """Stack of encoder blocks."""

    def __init__(self, layers: int, dim: int, heads: int, ffn_mult: float, dropout: float):
        super().__init__()
        self.blocks = nn.ModuleList([
            EncoderBlock(dim, heads, ffn_mult, dropout) for _ in range(layers)
        ])

    def forward(self, x: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x, rope)
        return x


class FiLM(nn.Module):
    """Feature-wise linear modulation for conditioning L on H context."""

    def __init__(self, h_dim: int, l_dim: int):
        super().__init__()
        self.g = nn.Linear(h_dim, l_dim, bias=False)
        self.b = nn.Linear(h_dim, l_dim, bias=False)

    def forward(self, l_tokens: torch.Tensor, h_ctx: torch.Tensor) -> torch.Tensor:
        gamma = self.g(h_ctx).unsqueeze(1)
        beta = self.b(h_ctx).unsqueeze(1)
        return l_tokens * (1.0 + gamma) + beta


class CrossAttention(nn.Module):
    """Multi-head cross attention allowing L-tokens to attend over H-tokens.

    This module projects query vectors from the L-module and key/value vectors
    from the H-module into a shared latent space and performs scaled dot
    product attention.  The output is projected back to the L-module
    dimensionality.  It is bias-free and does not apply positional
    embeddings by default; RoPE may be applied to inputs prior to calling.

    Parameters
    ----------
    q_dim : int
        Dimensionality of the query (L) tokens.
    kv_dim : int
        Dimensionality of the key and value (H) tokens.
    heads : int
        Number of attention heads.
    """
    def __init__(self, q_dim: int, kv_dim: int, heads: int):
        super().__init__()
        assert q_dim % heads == 0, "q_dim must be divisible by number of heads"
        # We project queries to q_dim, keys and values to q_dim internally
        self.q_proj = nn.Linear(q_dim, q_dim, bias=False)
        self.k_proj = nn.Linear(kv_dim, q_dim, bias=False)
        self.v_proj = nn.Linear(kv_dim, q_dim, bias=False)
        self.out_proj = nn.Linear(q_dim, q_dim, bias=False)
        self.heads = heads
        self.dk = q_dim // heads

    def forward(self, q_tokens: torch.Tensor, kv_tokens: torch.Tensor) -> torch.Tensor:
        """Perform cross-attention from ``q_tokens`` onto ``kv_tokens``.

        Parameters
        ----------
        q_tokens : torch.Tensor
            Query tokens of shape [B, T_q, q_dim].
        kv_tokens : torch.Tensor
            Key/value tokens of shape [B, T_kv, kv_dim].

        Returns
        -------
        torch.Tensor
            Attended representation of shape [B, T_q, q_dim].
        """
        B, T_q, q_dim = q_tokens.shape
        T_kv = kv_tokens.shape[1]
        # project
        q = self.q_proj(q_tokens)  # [B, T_q, q_dim]
        k = self.k_proj(kv_tokens)  # [B, T_kv, q_dim]
        v = self.v_proj(kv_tokens)  # [B, T_kv, q_dim]
        # reshape to heads
        q = q.view(B, T_q, self.heads, self.dk).transpose(1, 2)  # [B, H, T_q, dk]
        k = k.view(B, T_kv, self.heads, self.dk).transpose(1, 2)  # [B, H, T_kv, dk]
        v = v.view(B, T_kv, self.heads, self.dk).transpose(1, 2)  # [B, H, T_kv, dk]
        # scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)  # [B, H, T_q, T_kv]
        attn = attn.softmax(dim=-1)
        out = attn @ v  # [B, H, T_q, dk]
        out = out.transpose(1, 2).contiguous().view(B, T_q, q_dim)
        return self.out_proj(out)


@dataclass
class HRMConfig:
    h_layers: int
    h_dim: int
    h_heads: int
    h_ffn_mult: float
    h_dropout: float
    l_layers: int
    l_dim: int
    l_heads: int
    l_ffn_mult: float
    l_dropout: float
    segments_N: int
    l_inner_T: int
    act_enable: bool
    act_max_segments: int
    ponder_cost: float
    use_cross_attn: bool = False

    # by default, cross attention is off; when enabled, the L-module
    # will attend to the H-module state via a CrossAttention layer.


class HRMNet(nn.Module):
    """Simplified HRM with FiLM conditioning and deep supervision."""

    def __init__(self, cfg: HRMConfig = None):
        super().__init__()
        if cfg is None:
            # Default config targeting ~27M params
            cfg = HRMConfig(
                h_layers=4, h_dim=512, h_heads=8, h_ffn_mult=3.0, h_dropout=0.1,
                l_layers=6, l_dim=768, l_heads=12, l_ffn_mult=3.0, l_dropout=0.1,
                segments_N=4, l_inner_T=16, act_enable=True, act_max_segments=8,
                ponder_cost=0.01, use_cross_attn=False
            )
        self.cfg = cfg
        # H- and L-encoders
        self.h_enc = Encoder(cfg.h_layers, cfg.h_dim, cfg.h_heads, cfg.h_ffn_mult, cfg.h_dropout)
        self.h_norm = RMSNorm(cfg.h_dim)
        self.l_enc = Encoder(cfg.l_layers, cfg.l_dim, cfg.l_heads, cfg.l_ffn_mult, cfg.l_dropout)
        self.l_norm = RMSNorm(cfg.l_dim)
        # FiLM conditioning from H context to L tokens
        self.film = FiLM(cfg.h_dim, cfg.l_dim)
        # Optional cross attention from L tokens to H tokens
        self.cross_attn = CrossAttention(cfg.l_dim, cfg.h_dim, cfg.l_heads) if cfg.use_cross_attn else None
        # Heads for the two tasks
        self.headA = nn.Linear(cfg.h_dim, 1, bias=False)
        self.headB = nn.Linear(cfg.l_dim, 1, bias=False)
        # ACT Q-head: two logits [continue, halt]
        self.q_head = nn.Linear(cfg.h_dim, 2, bias=False)

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=1)

    def _forward_once(self, h_tokens: torch.Tensor, l_tokens: torch.Tensor) -> tuple:
        """Perform a single reasoning step of the HRM.

        This method runs one slow update on the H-module, conditions the
        L-module on the updated H context via FiLM and optional cross
        attention, and then runs one fast update on the L-module.  It
        returns the updated token sequences along with the per-step
        predictions from both heads and the Q-head logits.

        Parameters
        ----------
        h_tokens : torch.Tensor
            H-module tokens of shape [B, T_h, d_h].
        l_tokens : torch.Tensor
            L-module tokens of shape [B, T_l, d_l].

        Returns
        -------
        tuple
            ((h_tokens_new, l_tokens_new), (outA, outB, q_logits))
        """
        B, Th, Dh = h_tokens.shape
        Tl = l_tokens.shape[1]
        # compute rotary frequencies on the fly for efficiency
        rope_h = rope_freqs(Th, Dh, device=h_tokens.device)
        rope_l = rope_freqs(Tl, self.cfg.l_dim, device=l_tokens.device)
        # Update H tokens first (slow step)
        h_tokens = self.h_enc(h_tokens, rope_h)
        # pool to context and normalise
        h_pool = self._pool(self.h_norm(h_tokens))  # [B, d_h]
        # Condition L tokens via FiLM using H context
        cond_l = self.film(l_tokens, h_pool)
        # Optional cross attention: allow L tokens to attend over H tokens
        if self.cross_attn is not None:
            # compute cross-attention between cond_l and h_tokens (normalised)
            cond_l = cond_l + self.cross_attn(cond_l, h_tokens)
        # Update L tokens (fast step)
        l_tokens = self.l_enc(cond_l, rope_l)
        # heads: compute pooled outputs
        l_pool = self._pool(self.l_norm(l_tokens))
        outA = self.headA(h_pool).squeeze(-1)  # shape [B]
        outB = self.headB(l_pool).squeeze(-1)  # shape [B]
        # Q-head logits for ACT: predicts [continue, halt]
        q_logits = self.q_head(h_pool)
        return (h_tokens, l_tokens), (outA, outB, q_logits)

    def forward(self, h_tokens: torch.Tensor, l_tokens: torch.Tensor) -> tuple:
        """Run the HRM for N*T-1 steps without gradient and one step with gradient."""
        N = self.cfg.segments_N
        T = self.cfg.l_inner_T
        segments = []
        with torch.no_grad():
            for i in range(N * T - 1):
                (h_tokens, l_tokens), heads = self._forward_once(h_tokens, l_tokens)
                # deep supervision every T steps
                if (i + 1) % T == 0:
                    segments.append([h.detach() if isinstance(h, torch.Tensor) else h for h in heads])
        # final step with gradient
        (h_tokens, l_tokens), heads = self._forward_once(h_tokens, l_tokens)
        segments.append(heads)
        return (h_tokens, l_tokens), segments