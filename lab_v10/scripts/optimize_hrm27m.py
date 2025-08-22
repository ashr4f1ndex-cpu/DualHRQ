#!/usr/bin/env python3
"""
optimize_hrm27m.py
=================

Optimize HRM architecture to hit exactly 27M parameters.
This script finds the correct FFN multipliers and layer configurations.
"""

def transformer_block_params(d_model, n_heads, ffn_mult):
    """Calculate parameters in a single transformer block."""
    # Multi-head self-attention
    qkv_params = d_model * 3 * d_model  # QKV projection
    out_proj_params = d_model * d_model  # Output projection
    attn_params = qkv_params + out_proj_params
    
    # GLU Feed-forward network
    inner_dim = int(d_model * ffn_mult)
    glu_proj_params = d_model * 2 * inner_dim  # Two projections for GLU
    ffn_out_params = inner_dim * d_model  # Output projection
    ffn_params = glu_proj_params + ffn_out_params
    
    # RMSNorm parameters (2 per block)
    rms_params = 2 * d_model
    
    return attn_params + ffn_params + rms_params

def calculate_total_params(h_layers, h_dim, h_ffn_mult, l_layers, l_dim, l_ffn_mult):
    """Calculate total parameters for given architecture."""
    # H-module
    h_block_params = transformer_block_params(h_dim, 8, h_ffn_mult)  # 8 heads fixed
    h_encoder_params = h_layers * h_block_params
    h_norm_params = h_dim
    
    # L-module  
    l_block_params = transformer_block_params(l_dim, 12, l_ffn_mult)  # 12 heads fixed
    l_encoder_params = l_layers * l_block_params
    l_norm_params = l_dim
    
    # FiLM conditioning: h_dim -> l_dim (gamma + beta)
    film_params = 2 * h_dim * l_dim
    
    # Task heads
    head_a_params = 2 * h_dim  # mu + log_var
    head_b_params = l_dim  # single output
    q_head_params = 2 * h_dim  # [continue, halt]
    task_params = head_a_params + head_b_params + q_head_params
    
    total = (h_encoder_params + h_norm_params + 
             l_encoder_params + l_norm_params + 
             film_params + task_params)
    
    return total, {
        'h_encoder': h_encoder_params,
        'l_encoder': l_encoder_params, 
        'film': film_params,
        'heads': task_params,
        'norms': h_norm_params + l_norm_params
    }

def find_optimal_config():
    """Find configuration that hits ~27M parameters."""
    target = 27_000_000  # 27M target
    tolerance = 500_000   # ±500K tolerance
    
    # Fixed architecture constraints from specification
    h_layers = 4
    h_dim = 512
    l_layers = 6  
    l_dim = 768
    
    best_config = None
    best_diff = float('inf')
    
    print("Searching for optimal FFN multipliers...")
    print("Target: 27,000,000 parameters")
    print()
    
    # Search through FFN multipliers (broader range)
    ffn_range = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    for h_ffn_mult in ffn_range:
        for l_ffn_mult in ffn_range:
            total_params, breakdown = calculate_total_params(
                h_layers, h_dim, h_ffn_mult, 
                l_layers, l_dim, l_ffn_mult
            )
            
            diff = abs(total_params - target)
            
            if diff < best_diff:
                best_diff = diff
                best_config = {
                    'h_layers': h_layers,
                    'h_dim': h_dim, 
                    'h_ffn_mult': h_ffn_mult,
                    'l_layers': l_layers,
                    'l_dim': l_dim,
                    'l_ffn_mult': l_ffn_mult,
                    'total_params': total_params,
                    'breakdown': breakdown
                }
            
            # Print promising candidates
            if diff < tolerance:
                h_ffn_dim = int(h_dim * h_ffn_mult)
                l_ffn_dim = int(l_dim * l_ffn_mult)
                print(f"H_FFN={h_ffn_mult} ({h_ffn_dim}), L_FFN={l_ffn_mult} ({l_ffn_dim}): {total_params:,} (±{diff:,})")
    
    return best_config

def main():
    print("HRM-27M Architecture Optimization")
    print("=" * 50)
    
    optimal = find_optimal_config()
    
    if optimal:
        print(f"\nOptimal Configuration:")
        print(f"H-module: {optimal['h_layers']} layers, {optimal['h_dim']} d_model, FFN_mult={optimal['h_ffn_mult']}")
        print(f"L-module: {optimal['l_layers']} layers, {optimal['l_dim']} d_model, FFN_mult={optimal['l_ffn_mult']}")
        print(f"Total parameters: {optimal['total_params']:,}")
        
        print(f"\nParameter Breakdown:")
        for component, params in optimal['breakdown'].items():
            pct = params / optimal['total_params'] * 100
            print(f"  {component}: {params:,} ({pct:.1f}%)")
            
        # Generate exact FFN dimensions for specification
        h_ffn_dim = int(optimal['h_dim'] * optimal['h_ffn_mult'])
        l_ffn_dim = int(optimal['l_dim'] * optimal['l_ffn_mult'])
        
        print(f"\nExact Architecture Specification:")
        print(f"H_CONFIG = {{")
        print(f"    'n_layers': {optimal['h_layers']}, 'd_model': {optimal['h_dim']}, 'n_heads': 8, 'd_ff': {h_ffn_dim},")
        print(f"    'dropout': 0.1, 'layer_norm_eps': 1e-5")
        print(f"}}")
        print(f"L_CONFIG = {{")
        print(f"    'n_layers': {optimal['l_layers']}, 'd_model': {optimal['l_dim']}, 'n_heads': 12, 'd_ff': {l_ffn_dim},")
        print(f"    'dropout': 0.1, 'layer_norm_eps': 1e-5")
        print(f"}}")
        
        # Memory estimate
        fp16_mb = (optimal['total_params'] * 2) / (1024**2)
        print(f"\nFP16 Memory: {fp16_mb:.1f} MB")
        
        target_min = 26_500_000
        target_max = 27_500_000
        if target_min <= optimal['total_params'] <= target_max:
            print("✓ Within 27M parameter budget")
            return 0
        else:
            print("✗ Outside 27M parameter budget")
            return 1
    else:
        print("No suitable configuration found")
        return 1

if __name__ == "__main__":
    exit(main())