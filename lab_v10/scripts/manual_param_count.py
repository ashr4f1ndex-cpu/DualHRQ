#!/usr/bin/env python3
"""
manual_param_count.py
====================

Manual parameter count calculation for the HRM-27M architecture.
This script calculates parameter counts without requiring PyTorch.
"""

def transformer_block_params(d_model, n_heads, ffn_mult):
    """Calculate parameters in a single transformer block."""
    # Multi-head self-attention
    # QKV projection: d_model -> 3 * d_model (no bias)
    qkv_params = d_model * 3 * d_model
    
    # Output projection: d_model -> d_model (no bias)
    out_proj_params = d_model * d_model
    
    # Total attention params
    attn_params = qkv_params + out_proj_params
    
    # GLU Feed-forward network
    # Two projections for GLU: d_model -> 2 * inner_dim (no bias)
    inner_dim = int(d_model * ffn_mult)
    glu_proj_params = d_model * 2 * inner_dim
    
    # Output projection: inner_dim -> d_model (no bias)
    ffn_out_params = inner_dim * d_model
    
    # Total FFN params  
    ffn_params = glu_proj_params + ffn_out_params
    
    # RMSNorm parameters (2 layer norms per block)
    # Each RMSNorm has d_model learnable scale parameters
    rms_params = 2 * d_model
    
    return attn_params + ffn_params + rms_params

def encoder_params(n_layers, d_model, n_heads, ffn_mult):
    """Calculate parameters for a transformer encoder."""
    block_params = transformer_block_params(d_model, n_heads, ffn_mult)
    return n_layers * block_params

def film_params(h_dim, l_dim):
    """Calculate parameters for FiLM conditioning."""
    # gamma projection: h_dim -> l_dim (no bias)  
    gamma_params = h_dim * l_dim
    
    # beta projection: h_dim -> l_dim (no bias)
    beta_params = h_dim * l_dim
    
    return gamma_params + beta_params

def cross_attention_params(q_dim, kv_dim, heads):
    """Calculate parameters for cross attention."""
    # Query projection: q_dim -> q_dim
    q_proj = q_dim * q_dim
    
    # Key projection: kv_dim -> q_dim  
    k_proj = kv_dim * q_dim
    
    # Value projection: kv_dim -> q_dim
    v_proj = kv_dim * q_dim
    
    # Output projection: q_dim -> q_dim
    out_proj = q_dim * q_dim
    
    return q_proj + k_proj + v_proj + out_proj

def task_heads_params(h_dim, l_dim, use_heteroscedastic=True):
    """Calculate parameters for task heads."""
    if use_heteroscedastic:
        # Head-A: mu and log_var outputs
        head_a_params = h_dim * 1 + h_dim * 1  # mu + log_var
    else:
        # Head-A: single output
        head_a_params = h_dim * 1
        
    # Head-B: single output
    head_b_params = l_dim * 1
    
    # Q-head: 2 outputs [continue, halt]
    q_head_params = h_dim * 2
    
    return head_a_params + head_b_params + q_head_params

def calculate_hrm27m_params():
    """Calculate total parameters for HRM-27M architecture."""
    
    # Architecture specification (optimized for 26.82M parameters)
    # H-module: 4 layers, 512 d_model, 8 heads, 384 FFN
    h_layers = 4
    h_dim = 512  
    h_heads = 8
    h_ffn_mult = 0.75  # 384 / 512 = 0.75
    
    # L-module: 6 layers, 768 d_model, 12 heads, 384 FFN  
    l_layers = 6
    l_dim = 768
    l_heads = 12
    l_ffn_mult = 0.5  # 384 / 768 = 0.5
    
    print("HRM-27M Parameter Count Calculation")
    print("=" * 50)
    
    # H-module encoder
    h_encoder_params = encoder_params(h_layers, h_dim, h_heads, h_ffn_mult)
    print(f"H-module encoder: {h_encoder_params:,} parameters")
    
    # L-module encoder  
    l_encoder_params = encoder_params(l_layers, l_dim, l_heads, l_ffn_mult)
    print(f"L-module encoder: {l_encoder_params:,} parameters")
    
    # H-module normalization (final norm)
    h_norm_params = h_dim
    print(f"H-module final norm: {h_norm_params:,} parameters")
    
    # L-module normalization (final norm)
    l_norm_params = l_dim  
    print(f"L-module final norm: {l_norm_params:,} parameters")
    
    # FiLM conditioning
    film_conditioning_params = film_params(h_dim, l_dim)
    print(f"FiLM conditioning: {film_conditioning_params:,} parameters")
    
    # Optional cross attention (disabled in default config)
    cross_attn_params = 0  # cross_attention_params(l_dim, h_dim, l_heads)
    print(f"Cross attention: {cross_attn_params:,} parameters")
    
    # Task heads
    task_head_params = task_heads_params(h_dim, l_dim, use_heteroscedastic=True)
    print(f"Task heads: {task_head_params:,} parameters")
    
    # Total parameters
    total_params = (h_encoder_params + l_encoder_params + 
                   h_norm_params + l_norm_params +
                   film_conditioning_params + cross_attn_params + 
                   task_head_params)
    
    print("-" * 50)
    print(f"Total parameters: {total_params:,}")
    
    # Validate against target
    target_min = 26_500_000  # 26.5M
    target_max = 27_500_000  # 27.5M
    
    if target_min <= total_params <= target_max:
        print(f"✓ Within target range [{target_min:,}, {target_max:,}]")
        status = "PASS"
    else:
        print(f"✗ Outside target range [{target_min:,}, {target_max:,}]")
        status = "FAIL"
        
    # Memory estimates
    fp16_memory_mb = (total_params * 2) / (1024**2)  # 2 bytes per param in FP16
    print(f"FP16 parameter memory: {fp16_memory_mb:.1f} MB")
    
    # Detailed breakdown
    print("\nDetailed Breakdown:")
    print(f"H-module ({h_layers}L×{h_dim}D×{h_heads}H×{int(h_dim*h_ffn_mult)}FFN): {h_encoder_params:,} ({h_encoder_params/total_params*100:.1f}%)")
    print(f"L-module ({l_layers}L×{l_dim}D×{l_heads}H×{int(l_dim*l_ffn_mult)}FFN): {l_encoder_params:,} ({l_encoder_params/total_params*100:.1f}%)")
    print(f"Conditioning & Heads: {film_conditioning_params + task_head_params:,} ({(film_conditioning_params + task_head_params)/total_params*100:.1f}%)")
    
    return total_params, status

def detailed_block_breakdown():
    """Show detailed breakdown of a single transformer block."""
    print("\nSingle Transformer Block Breakdown:")
    print("(Using L-module as example: 768D, 12H, 3072FFN)")
    
    d_model = 768
    n_heads = 12  
    ffn_mult = 4.0
    
    # Attention components
    qkv_params = d_model * 3 * d_model
    attn_out_params = d_model * d_model
    total_attn = qkv_params + attn_out_params
    
    # FFN components
    inner_dim = int(d_model * ffn_mult)
    glu_params = d_model * 2 * inner_dim
    ffn_out_params = inner_dim * d_model
    total_ffn = glu_params + ffn_out_params
    
    # Norms
    norm_params = 2 * d_model
    
    total_block = total_attn + total_ffn + norm_params
    
    print(f"  QKV projection: {qkv_params:,}")
    print(f"  Attention output: {attn_out_params:,}")
    print(f"  GLU projection: {glu_params:,}")  
    print(f"  FFN output: {ffn_out_params:,}")
    print(f"  RMSNorm (2x): {norm_params:,}")
    print(f"  Total per block: {total_block:,}")

if __name__ == "__main__":
    total_params, status = calculate_hrm27m_params()
    detailed_block_breakdown()
    
    print(f"\nFinal Status: {status}")
    
    if status == "FAIL":
        exit(1)
    else:
        exit(0)