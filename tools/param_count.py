#!/usr/bin/env python3
"""
param_count.py - Parameter Counter + CI Gate
============================================

CRITICAL WEEK 1 BLOCKING TOOL - Nothing else proceeds until this is GREEN.

This tool:
1. Counts HRM parameters accurately 
2. Enforces 26.5M-27.5M parameter budget
3. Provides CI gate with proper exit codes
4. Iterates configs until budget compliance

Current Issue: 46.68M parameters (74% over budget)
Target: 26.5M-27.5M parameters
"""

import sys
import argparse
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    import torch
    import torch.nn as nn
    from lab_v10.src.options.hrm_net import HRMNet, HRMConfig
except ImportError as e:
    print(f"ERROR: Cannot import required modules: {e}")
    print("Make sure you're running from project root and dependencies are installed")
    sys.exit(1)


class ParameterCounter:
    """Parameter counting and budget enforcement."""
    
    def __init__(self, min_params: int = 26_500_000, max_params: int = 27_500_000):
        self.min_params = min_params
        self.max_params = max_params
    
    def count_hrm_parameters(self, config_path: Optional[str] = None, 
                           config_obj: Optional[HRMConfig] = None,
                           return_breakdown: bool = False) -> Dict[str, Any]:
        """Count HRM parameters with optional breakdown."""
        
        if config_path and config_obj:
            raise ValueError("Provide either config_path or config_obj, not both")
        
        if config_path:
            config_obj = self._load_config(config_path)
        elif config_obj is None:
            raise ValueError("Must provide either config_path or config_obj")
        
        # Create model to count parameters
        model = HRMNet(config_obj)
        
        # Count parameters by component
        h_module_params = 0
        l_module_params = 0
        head_params = 0
        other_params = 0
        
        for name, param in model.named_parameters():
            param_count = param.numel()
            
            if 'h_module' in name or 'H_' in name:
                h_module_params += param_count
            elif 'l_module' in name or 'L_' in name:
                l_module_params += param_count
            elif 'head' in name.lower() or 'attention' in name.lower():
                head_params += param_count
            else:
                other_params += param_count
        
        total_params = h_module_params + l_module_params + head_params + other_params
        
        result = {
            'total': total_params,
            'is_compliant': self.min_params <= total_params <= self.max_params,
            'budget_utilization': total_params / self.max_params,
            'over_budget_by': max(0, total_params - self.max_params),
            'under_budget_by': max(0, self.min_params - total_params)
        }
        
        if return_breakdown:
            result.update({
                'h_module': h_module_params,
                'l_module': l_module_params,
                'heads': head_params,
                'other': other_params
            })
        
        return result
    
    def _load_config(self, config_path: str) -> HRMConfig:
        """Load HRM config from YAML file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Extract HRM-specific config
        if 'hrm' in config_dict:
            hrm_config = config_dict['hrm']
        elif 'model' in config_dict:
            hrm_config = config_dict['model']
        else:
            hrm_config = config_dict
        
        # Create HRMConfig object
        return HRMConfig(**hrm_config)
    
    def suggest_config_adjustments(self, current_params: int) -> Dict[str, Any]:
        """Suggest configuration adjustments to meet budget."""
        
        if self.min_params <= current_params <= self.max_params:
            return {'status': 'compliant', 'suggestions': []}
        
        suggestions = []
        target_reduction = current_params - self.max_params
        
        if current_params > self.max_params:
            # Over budget - suggest reductions
            reduction_pct = target_reduction / current_params
            
            suggestions.append({
                'type': 'dimension_reduction',
                'description': f'Reduce dimensions by ~{reduction_pct:.1%}',
                'examples': [
                    'H: 4×384 → 4×320, L: 4×512 → 4×480',
                    'H: 4×384 → 3×400, L: 4×512 → 5×460',
                    'Reduce FFN multiplier: 3.0 → 2.0'
                ]
            })
            
            suggestions.append({
                'type': 'layer_reduction', 
                'description': 'Reduce layer count',
                'examples': [
                    'H: 4 layers → 3 layers',
                    'L: 4 layers → 3 layers'
                ]
            })
            
            suggestions.append({
                'type': 'head_reduction',
                'description': 'Reduce attention heads',
                'examples': [
                    'H heads: 8 → 6, L heads: 8 → 6',
                    'Keep dimensions, reduce heads for efficiency'
                ]
            })
        
        elif current_params < self.min_params:
            # Under budget - suggest increases
            increase_needed = self.min_params - current_params
            increase_pct = increase_needed / current_params
            
            suggestions.append({
                'type': 'dimension_increase',
                'description': f'Increase dimensions by ~{increase_pct:.1%}',
                'examples': [
                    'H: 4×320 → 4×384, L: 4×480 → 4×512',
                    'Add more capacity to utilize budget'
                ]
            })
        
        return {
            'status': 'non_compliant',
            'current_params': current_params,
            'target_range': f'{self.min_params:,} - {self.max_params:,}',
            'adjustment_needed': target_reduction if current_params > self.max_params else increase_needed,
            'suggestions': suggestions
        }
    
    def verify_budget_compliance(self, param_count: int, 
                               max_params: Optional[int] = None, 
                               min_params: Optional[int] = None) -> bool:
        """Verify parameter budget compliance."""
        max_p = max_params or self.max_params
        min_p = min_params or self.min_params
        return min_p <= param_count <= max_p


def generate_config_variants() -> list:
    """Generate config variants to try for budget compliance."""
    
    variants = [
        # Variant 1: Target ~27M parameters with balanced architecture
        {
            'name': 'optimal_27m_v1',
            'h_layers': 4, 'h_dim': 512, 'h_heads': 8, 'h_ffn_mult': 3.0, 'h_dropout': 0.1,
            'l_layers': 6, 'l_dim': 768, 'l_heads': 12, 'l_ffn_mult': 3.0, 'l_dropout': 0.1,
            'segments_N': 3, 'l_inner_T': 8, 'act_enable': True, 'act_max_segments': 5, 'ponder_cost': 0.01,
            'use_cross_attn': True
        },
        
        # Variant 2: Slightly smaller H, larger L  
        {
            'name': 'optimal_27m_v2',
            'h_layers': 3, 'h_dim': 480, 'h_heads': 8, 'h_ffn_mult': 3.0, 'h_dropout': 0.1,
            'l_layers': 6, 'l_dim': 896, 'l_heads': 14, 'l_ffn_mult': 2.5, 'l_dropout': 0.1,
            'segments_N': 3, 'l_inner_T': 10, 'act_enable': True, 'act_max_segments': 5, 'ponder_cost': 0.01,
            'use_cross_attn': True
        },
        
        # Variant 3: Deeper but narrower
        {
            'name': 'optimal_27m_v3',
            'h_layers': 5, 'h_dim': 448, 'h_heads': 7, 'h_ffn_mult': 2.5, 'h_dropout': 0.1,
            'l_layers': 7, 'l_dim': 672, 'l_heads': 12, 'l_ffn_mult': 2.5, 'l_dropout': 0.1,
            'segments_N': 3, 'l_inner_T': 8, 'act_enable': True, 'act_max_segments': 5, 'ponder_cost': 0.01,
            'use_cross_attn': True
        },
        
        # Variant 4: Wider but shallower
        {
            'name': 'optimal_27m_v4',
            'h_layers': 3, 'h_dim': 640, 'h_heads': 10, 'h_ffn_mult': 3.0, 'h_dropout': 0.1,
            'l_layers': 4, 'l_dim': 896, 'l_heads': 16, 'l_ffn_mult': 3.0, 'l_dropout': 0.1,
            'segments_N': 4, 'l_inner_T': 12, 'act_enable': True, 'act_max_segments': 6, 'ponder_cost': 0.01,
            'use_cross_attn': True
        },

        # Variant 5: Medium size configurations to hit the target
        {
            'name': 'optimal_27m_med_v1',
            'h_layers': 4, 'h_dim': 384, 'h_heads': 8, 'h_ffn_mult': 2.2, 'h_dropout': 0.1,
            'l_layers': 4, 'l_dim': 512, 'l_heads': 8, 'l_ffn_mult': 2.5, 'l_dropout': 0.1,
            'segments_N': 3, 'l_inner_T': 8, 'act_enable': True, 'act_max_segments': 5, 'ponder_cost': 0.01,
            'use_cross_attn': True
        },

        {
            'name': 'optimal_27m_med_v2',
            'h_layers': 4, 'h_dim': 416, 'h_heads': 8, 'h_ffn_mult': 2.3, 'h_dropout': 0.1,
            'l_layers': 4, 'l_dim': 544, 'l_heads': 8, 'l_ffn_mult': 2.4, 'l_dropout': 0.1,
            'segments_N': 3, 'l_inner_T': 8, 'act_enable': True, 'act_max_segments': 5, 'ponder_cost': 0.01,
            'use_cross_attn': True
        },

        {
            'name': 'optimal_27m_med_v3',
            'h_layers': 3, 'h_dim': 448, 'h_heads': 8, 'h_ffn_mult': 2.4, 'h_dropout': 0.1,
            'l_layers': 5, 'l_dim': 544, 'l_heads': 8, 'l_ffn_mult': 2.3, 'l_dropout': 0.1,
            'segments_N': 3, 'l_inner_T': 8, 'act_enable': True, 'act_max_segments': 5, 'ponder_cost': 0.01,
            'use_cross_attn': True
        },

        {
            'name': 'optimal_27m_med_v4',
            'h_layers': 4, 'h_dim': 384, 'h_heads': 6, 'h_ffn_mult': 2.5, 'h_dropout': 0.1,
            'l_layers': 4, 'l_dim': 576, 'l_heads': 9, 'l_ffn_mult': 2.3, 'l_dropout': 0.1,
            'segments_N': 3, 'l_inner_T': 8, 'act_enable': True, 'act_max_segments': 5, 'ponder_cost': 0.01,
            'use_cross_attn': True
        },

        {
            'name': 'optimal_27m_med_v5',
            'h_layers': 4, 'h_dim': 400, 'h_heads': 8, 'h_ffn_mult': 2.2, 'h_dropout': 0.1,
            'l_layers': 4, 'l_dim': 560, 'l_heads': 8, 'l_ffn_mult': 2.2, 'l_dropout': 0.1,
            'segments_N': 3, 'l_inner_T': 8, 'act_enable': True, 'act_max_segments': 5, 'ponder_cost': 0.01,
            'use_cross_attn': True
        },

        # Fine-tuned to hit exactly 26.5M-27.5M range
        {
            'name': 'optimal_27m_target',
            'h_layers': 4, 'h_dim': 448, 'h_heads': 8, 'h_ffn_mult': 2.4, 'h_dropout': 0.1,
            'l_layers': 5, 'l_dim': 560, 'l_heads': 8, 'l_ffn_mult': 2.3, 'l_dropout': 0.1,
            'segments_N': 3, 'l_inner_T': 8, 'act_enable': True, 'act_max_segments': 5, 'ponder_cost': 0.01,
            'use_cross_attn': True
        },

        {
            'name': 'optimal_27m_target_v2',
            'h_layers': 3, 'h_dim': 480, 'h_heads': 8, 'h_ffn_mult': 2.5, 'h_dropout': 0.1,
            'l_layers': 5, 'l_dim': 576, 'l_heads': 9, 'l_ffn_mult': 2.2, 'l_dropout': 0.1,
            'segments_N': 3, 'l_inner_T': 8, 'act_enable': True, 'act_max_segments': 5, 'ponder_cost': 0.01,
            'use_cross_attn': True
        },

        # Fallback variants if above don't work
        # Variant: Reduced dimensions
        {
            'name': 'reduced_dims_v1',
            'h_layers': 4, 'h_dim': 320, 'h_heads': 8, 'h_ffn_mult': 2.0, 'h_dropout': 0.1,
            'l_layers': 4, 'l_dim': 480, 'l_heads': 8, 'l_ffn_mult': 2.0, 'l_dropout': 0.1,
            'segments_N': 3, 'l_inner_T': 8, 'act_enable': True, 'act_max_segments': 5, 'ponder_cost': 0.01,
            'use_cross_attn': False
        }
    ]
    
    return variants


# Functions expected by tests
def count_hrm_parameters(config_path_or_obj, return_breakdown=False):
    """Count HRM parameters - wrapper for test compatibility."""
    counter = ParameterCounter()
    if isinstance(config_path_or_obj, str):
        return counter.count_hrm_parameters(config_path=config_path_or_obj, return_breakdown=return_breakdown)
    else:
        return counter.count_hrm_parameters(config_obj=config_path_or_obj, return_breakdown=return_breakdown)

def verify_budget_compliance(param_count, max_params=None, min_params=None):
    """Verify budget compliance - wrapper for test compatibility."""
    counter = ParameterCounter()
    return counter.verify_budget_compliance(param_count, max_params, min_params)


def main():
    parser = argparse.ArgumentParser(description='HRM Parameter Counter + CI Gate')
    parser.add_argument('--config', type=str, help='Path to HRM config YAML')
    parser.add_argument('--format', choices=['json', 'text'], default='text',
                       help='Output format')
    parser.add_argument('--strict', action='store_true',
                       help='Strict mode - exit 1 if not compliant')
    parser.add_argument('--suggest', action='store_true', 
                       help='Suggest config adjustments')
    parser.add_argument('--iterate', action='store_true',
                       help='Try config variants until compliant')
    parser.add_argument('--breakdown', action='store_true',
                       help='Show parameter breakdown by component')
    
    args = parser.parse_args()
    
    counter = ParameterCounter()
    
    try:
        if args.iterate:
            # Try variants until we find compliant one
            print("ITERATING CONFIG VARIANTS TO FIND BUDGET COMPLIANCE...")
            
            variants = generate_config_variants()
            compliant_config = None
            
            for variant in variants:
                print(f"\nTrying variant: {variant['name']}")
                
                # Remove name from config for HRMConfig
                config_dict = {k: v for k, v in variant.items() if k != 'name'}
                config = HRMConfig(**config_dict)
                
                result = counter.count_hrm_parameters(config_obj=config, return_breakdown=args.breakdown)
                
                print(f"  Total params: {result['total']:,}")
                print(f"  Budget utilization: {result['budget_utilization']:.1%}")
                print(f"  Compliant: {result['is_compliant']}")
                
                if result['is_compliant']:
                    compliant_config = (variant['name'], config, result)
                    print(f"  ✅ FOUND COMPLIANT CONFIG: {variant['name']}")
                    break
                else:
                    over_by = result.get('over_budget_by', 0)
                    under_by = result.get('under_budget_by', 0)
                    if over_by > 0:
                        print(f"  ❌ Over budget by: {over_by:,} parameters")
                    elif under_by > 0:
                        print(f"  ❌ Under budget by: {under_by:,} parameters")
            
            if compliant_config:
                name, config, result = compliant_config
                
                if args.format == 'json':
                    output = {
                        'compliant_config_found': True,
                        'config_name': name,
                        'config_params': config.__dict__,
                        'parameter_count': result
                    }
                    print(json.dumps(output, indent=2))
                else:
                    print(f"\n🎯 SUCCESS: Found compliant configuration '{name}'")
                    print(f"Parameters: {result['total']:,} (within {counter.min_params:,} - {counter.max_params:,})")
                    
                    if args.breakdown and 'h_module' in result:
                        print("\nParameter Breakdown:")
                        print(f"  H-module: {result['h_module']:,}")
                        print(f"  L-module: {result['l_module']:,}")
                        print(f"  Heads: {result['heads']:,}")
                        print(f"  Other: {result['other']:,}")
                
                # Save compliant config
                output_path = Path('config/compliant_hrm27m.yaml')
                output_path.parent.mkdir(exist_ok=True)
                
                with open(output_path, 'w') as f:
                    yaml.dump({'hrm': config.__dict__}, f, default_flow_style=False)
                
                print(f"\n💾 Saved compliant config to: {output_path}")
                
                sys.exit(0)  # Success
            else:
                print("\n❌ NO COMPLIANT CONFIG FOUND")
                print("Manual adjustment needed - see suggestions with --suggest")
                sys.exit(1)  # Failure
        
        else:
            # Single config check
            if args.config:
                result = counter.count_hrm_parameters(args.config, return_breakdown=args.breakdown)
            else:
                # Use default config
                default_config = Path('config/default_hrm27m.yaml')
                if default_config.exists():
                    result = counter.count_hrm_parameters(str(default_config), return_breakdown=args.breakdown)
                else:
                    print("ERROR: No config specified and no default config found")
                    sys.exit(1)
            
            if args.format == 'json':
                print(json.dumps(result, indent=2))
            else:
                print(f"Total Parameters: {result['total']:,}")
                print(f"Budget Range: {counter.min_params:,} - {counter.max_params:,}")
                print(f"Budget Utilization: {result['budget_utilization']:.1%}")
                print(f"Compliant: {'✅ YES' if result['is_compliant'] else '❌ NO'}")
                
                if not result['is_compliant']:
                    over_by = result.get('over_budget_by', 0)
                    under_by = result.get('under_budget_by', 0)
                    if over_by > 0:
                        print(f"Over budget by: {over_by:,} parameters ({over_by/result['total']:.1%})")
                    elif under_by > 0:
                        print(f"Under budget by: {under_by:,} parameters")
                
                if args.breakdown and 'h_module' in result:
                    print("\nParameter Breakdown:")
                    print(f"  H-module: {result['h_module']:,}")
                    print(f"  L-module: {result['l_module']:,}")
                    print(f"  Heads: {result['heads']:,}")
                    print(f"  Other: {result['other']:,}")
            
            if args.suggest:
                suggestions = counter.suggest_config_adjustments(result['total'])
                
                if args.format == 'json':
                    print(json.dumps(suggestions, indent=2))
                else:
                    print(f"\nSuggestions ({suggestions['status']}):")
                    for suggestion in suggestions.get('suggestions', []):
                        print(f"\n{suggestion['type'].upper()}:")
                        print(f"  {suggestion['description']}")
                        for example in suggestion.get('examples', []):
                            print(f"  - {example}")
            
            # Exit code for CI
            if args.strict and not result['is_compliant']:
                sys.exit(1)  # Failure
            else:
                sys.exit(0)  # Success
    
    except Exception as e:
        if args.format == 'json':
            error_result = {'error': str(e), 'success': False}
            print(json.dumps(error_result, indent=2))
        else:
            print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()