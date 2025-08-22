"""
HRM Model Compatibility Layer

This module provides a compatibility layer that aliases the HRM implementation
from options.hrm_net to maintain backward compatibility with the main orchestrator
while providing enhanced adaptive computation capabilities.
"""

from ..options.hrm_net import HRMNet, HRMConfig as BaseHRMConfig
from ..common.act import AdaptiveComputationTime
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

def HRMConfig(**kwargs):
    """Create HRM configuration with backward compatibility."""
    # Map old parameter names to new ones for compatibility
    param_mapping = {
        'h_dim': 'h_dim',
        'l_dim': 'l_dim', 
        'num_h_layers': 'h_layers',
        'num_l_layers': 'l_layers',
        'num_heads': 'h_heads',
        'dropout': 'h_dropout',
        'max_sequence_length': 'l_inner_T',
        'deq_threshold': 'act_threshold',
        'max_deq_iterations': 'act_max_segments'
    }
    
    # Convert old-style config to new-style config
    converted_kwargs = {}
    for old_key, new_key in param_mapping.items():
        if old_key in kwargs:
            converted_kwargs[new_key] = kwargs[old_key]
    
    # Set defaults for any missing parameters
    defaults = {
        'h_layers': converted_kwargs.get('h_layers', 4),
        'h_dim': converted_kwargs.get('h_dim', 512),
        'h_heads': converted_kwargs.get('h_heads', 8),
        'h_ffn_mult': 0.75,
        'h_dropout': converted_kwargs.get('h_dropout', 0.1),
        'l_layers': converted_kwargs.get('l_layers', 6),
        'l_dim': converted_kwargs.get('l_dim', 768),
        'l_heads': converted_kwargs.get('l_heads', 12),
        'l_ffn_mult': 0.5,
        'l_dropout': 0.1,
        'segments_N': 4,
        'l_inner_T': converted_kwargs.get('l_inner_T', 16),
        'act_enable': True,
        'act_max_segments': converted_kwargs.get('act_max_segments', 8),
        'ponder_cost': 0.01,
        'use_cross_attn': False,
        'use_heteroscedastic': True,
        'act_threshold': converted_kwargs.get('act_threshold', 0.01),
        'deq_style': True,
        'uncertainty_weighting': True
    }
    
    # Create config using the base class
    config = BaseHRMConfig(**defaults)
    logger.info(f"HRM Config created with converted parameters")
    return config

class HierarchicalReasoningModel(nn.Module):
    """
    Enhanced HRM with adaptive computation time and continuous learning.
    
    This model wraps the core HRMNet implementation with additional
    capabilities for adaptive resource usage and learning loop integration.
    """
    
    def __init__(self, config: HRMConfig):
        super().__init__()
        self.config = config
        self.hrm_core = HRMNet(config)
        
        # Add adaptive computation time controller
        self.act_controller = AdaptiveComputationTime(
            threshold=config.act_threshold,
            max_steps=config.act_max_segments,
            ponder_cost=config.ponder_cost
        )
        
        # Initialize learning metrics
        self.computation_history = []
        self.adaptation_count = 0
        
        logger.info("Enhanced HRM initialized with adaptive computation")
    
    @property 
    def h_module(self):
        """Provide compatibility access to H-module."""
        return self.hrm_core.h_enc
    
    @property
    def l_module(self):
        """Provide compatibility access to L-module.""" 
        return self.hrm_core.l_enc
    
    @property
    def act_module(self):
        """Provide access to ACT module."""
        return self.act_controller
    
    def forward(self, h_tokens, l_tokens):
        """Forward pass with adaptive computation tracking."""
        # Track computation before execution
        initial_computation = self._estimate_computation_cost()
        
        # Run HRM forward pass
        outputs = self.hrm_core(h_tokens, l_tokens)
        
        # Track adaptation metrics
        final_computation = self._estimate_computation_cost()
        self.computation_history.append({
            'initial_cost': initial_computation,
            'final_cost': final_computation,
            'adaptation_ratio': final_computation / (initial_computation + 1e-8)
        })
        
        # Adapt computation strategy based on history
        if len(self.computation_history) > 10:
            self._adapt_computation_strategy()
        
        return outputs
    
    def _estimate_computation_cost(self):
        """Estimate current computational cost."""
        # Simple heuristic based on model parameters and config
        base_cost = sum(p.numel() for p in self.parameters()) / 1e6  # Millions of params
        dynamic_cost = self.config.act_max_segments * self.config.l_inner_T
        return base_cost * dynamic_cost
    
    def _adapt_computation_strategy(self):
        """Adapt computation strategy based on historical performance."""
        recent_history = self.computation_history[-10:]
        avg_adaptation = sum(h['adaptation_ratio'] for h in recent_history) / len(recent_history)
        
        # If we're consistently over-computing, reduce max segments
        if avg_adaptation > 1.2:
            self.config.act_max_segments = max(4, self.config.act_max_segments - 1)
            self.adaptation_count += 1
            logger.info(f"Adapted max segments to {self.config.act_max_segments} (adaptation #{self.adaptation_count})")
        
        # If we're consistently under-computing, increase threshold
        elif avg_adaptation < 0.8:
            self.config.act_threshold = min(0.1, self.config.act_threshold * 1.1)
            self.adaptation_count += 1
            logger.info(f"Adapted threshold to {self.config.act_threshold:.4f} (adaptation #{self.adaptation_count})")
    
    def get_adaptation_metrics(self):
        """Get current adaptation metrics for monitoring."""
        if not self.computation_history:
            return {'adaptations': 0, 'efficiency': 1.0}
        
        recent = self.computation_history[-5:]
        avg_efficiency = sum(h['adaptation_ratio'] for h in recent) / len(recent)
        
        return {
            'adaptations': self.adaptation_count,
            'efficiency': avg_efficiency,
            'total_computations': len(self.computation_history),
            'current_threshold': self.config.act_threshold,
            'current_max_segments': self.config.act_max_segments
        }

# Alias for backward compatibility
HRMNet = HierarchicalReasoningModel

__all__ = ['HierarchicalReasoningModel', 'HRMConfig', 'HRMNet']