"""
hrm_integration.py
=================

HRM Integration Layer for DualHRQ 2.0 - TDD Implementation
Connects conditioning system to HRM model with parameter budget enforcement.

CRITICAL FEATURES:
- HRM adapter with FiLM conditioning mechanism
- Multi-source conditioning interface 
- Parameter budget management (≤27.5M total)
- Gradient flow monitoring and preservation
"""

from typing import Dict, List, Any, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datetime import datetime
import warnings
from collections import defaultdict

# Import HRMConfig from the correct module
try:
    from lab_v10.src.options.hrm_net import HRMConfig as _HRMConfig
    # Create a wrapper that provides defaults for missing parameters
    def HRMConfig(**kwargs):
        defaults = {
            'h_layers': 2, 'h_dim': 256, 'h_heads': 4, 'h_ffn_mult': 2.0, 'h_dropout': 0.0,
            'l_layers': 2, 'l_dim': 384, 'l_heads': 4, 'l_ffn_mult': 2.0, 'l_dropout': 0.0,
            'segments_N': 2, 'l_inner_T': 4, 'act_enable': False, 'act_max_segments': 3,
            'ponder_cost': 0.01, 'use_cross_attn': False
        }
        defaults.update(kwargs)
        return _HRMConfig(**defaults)
        
except ImportError:
    # Fallback minimal HRMConfig for testing
    from dataclasses import dataclass
    
    @dataclass
    class _HRMConfigBase:
        h_layers: int = 2
        h_dim: int = 256
        h_heads: int = 4
        h_ffn_mult: float = 2.0
        h_dropout: float = 0.0
        l_layers: int = 2
        l_dim: int = 384
        l_heads: int = 4
        l_ffn_mult: float = 2.0
        l_dropout: float = 0.0
        segments_N: int = 2
        l_inner_T: int = 4
        act_enable: bool = False
        act_max_segments: int = 3
        ponder_cost: float = 0.01
        use_cross_attn: bool = False
        
    def HRMConfig(**kwargs):
        defaults = {
            'h_layers': 2, 'h_dim': 256, 'h_heads': 4, 'h_ffn_mult': 2.0, 'h_dropout': 0.0,
            'l_layers': 2, 'l_dim': 384, 'l_heads': 4, 'l_ffn_mult': 2.0, 'l_dropout': 0.0,
            'segments_N': 2, 'l_inner_T': 4, 'act_enable': False, 'act_max_segments': 3,
            'ponder_cost': 0.01, 'use_cross_attn': False
        }
        defaults.update(kwargs)
        return _HRMConfigBase(**defaults)


class HRMAdapter(nn.Module):
    """HRM adapter layer for conditioning integration with FiLM mechanism."""
    
    def __init__(self, hrm_config, conditioning_dim: int, max_additional_params: int = 300_000,
                 conditioning_method: str = 'film'):
        super().__init__()
        self.hrm_config = hrm_config
        self.conditioning_dim = conditioning_dim
        self.max_additional_params = max_additional_params
        self.conditioning_method = conditioning_method
        
        # Initialize parameter budget manager
        self.parameter_budget_manager = ParameterBudgetManager(
            total_budget=27_500_000,
            adapter_max_params=max_additional_params
        )
        
        # Ultra-lightweight FiLM conditioning to stay within 300K parameter budget
        # Use shared projector approach to minimize parameters
        # Calculate hidden_dim to stay within budget: conditioning_dim * hidden_dim + hidden_dim * (h_dim + l_dim) * 2 ≤ max_params
        total_output_dim = hrm_config.h_dim + hrm_config.l_dim
        # Solve: conditioning_dim * hidden_dim + hidden_dim * total_output_dim * 2 ≤ max_params
        # hidden_dim * (conditioning_dim + 2 * total_output_dim) ≤ max_params
        max_hidden_dim = max_additional_params // (conditioning_dim + 2 * total_output_dim)
        hidden_dim = max(1, min(64, max_hidden_dim))  # At least 1, at most 64
        
        self.conditioning_projector = nn.Linear(conditioning_dim, hidden_dim, bias=False)
        
        # Separate FiLM parameters for H and L modules to match dimensions exactly
        self.film_gamma_h = nn.Linear(hidden_dim, hrm_config.h_dim, bias=False)
        self.film_beta_h = nn.Linear(hidden_dim, hrm_config.h_dim, bias=False)
        self.film_gamma_l = nn.Linear(hidden_dim, hrm_config.l_dim, bias=False)
        self.film_beta_l = nn.Linear(hidden_dim, hrm_config.l_dim, bias=False)
        
        # Store dimensions for splitting
        self.h_dim = hrm_config.h_dim
        self.l_dim = hrm_config.l_dim
        
        # Verify parameter budget compliance
        total_params = sum(p.numel() for p in self.parameters())
        if total_params > max_additional_params:
            raise ValueError(f"Adapter exceeds parameter budget: {total_params} > {max_additional_params}")
        
        # Add conditioning projector and parameter budget manager as attributes for tests
        self.conditioning_projector = self.conditioning_projector
        self.parameter_budget_manager = self.parameter_budget_manager
    
    def forward(self, hrm_features: torch.Tensor, conditioning_vector: torch.Tensor) -> torch.Tensor:
        """Forward pass: Apply FiLM conditioning to HRM features.
        
        Args:
            hrm_features: Features from HRM (batch_size, feature_dim)
            conditioning_vector: Conditioning input (batch_size, conditioning_dim)
            
        Returns:
            Conditioned features (batch_size, feature_dim)
        """
        # Project conditioning to hidden dimension
        conditioning_hidden = self.conditioning_projector(conditioning_vector)
        
        # Generate FiLM parameters
        film_params = torch.cat([
            self.film_gamma(conditioning_hidden), 
            self.film_beta(conditioning_hidden)
        ], dim=-1)
        
        # Split into gamma and beta, then split by H and L dimensions
        total_dim = self.h_dim + self.l_dim
        gamma_all = film_params[:, :total_dim]
        beta_all = film_params[:, total_dim:]
        
        # Apply FiLM conditioning: features * gamma + beta
        # Assuming hrm_features matches total HRM dimension
        conditioned_features = hrm_features * gamma_all + beta_all
        
        return conditioned_features
    
    def apply_conditioning(self, h_tokens: torch.Tensor, l_tokens: torch.Tensor, 
                         conditioning_vector: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply conditioning to HRM tokens using FiLM mechanism."""
        # Project conditioning to hidden dimension
        conditioning_hidden = self.conditioning_projector(conditioning_vector)
        
        # Generate FiLM parameters separately for H and L modules
        gamma_h = self.film_gamma_h(conditioning_hidden)  # [B, h_dim]
        beta_h = self.film_beta_h(conditioning_hidden)    # [B, h_dim]
        gamma_l = self.film_gamma_l(conditioning_hidden)  # [B, l_dim]
        beta_l = self.film_beta_l(conditioning_hidden)    # [B, l_dim]
        
        # Expand dimensions for broadcasting with tokens [B, seq_len, dim]
        if h_tokens.dim() == 3:  # [B, T, h_dim]
            gamma_h = gamma_h.unsqueeze(1)  # [B, 1, h_dim]
            beta_h = beta_h.unsqueeze(1)    # [B, 1, h_dim]
        if l_tokens.dim() == 3:  # [B, T, l_dim]
            gamma_l = gamma_l.unsqueeze(1)  # [B, 1, l_dim]
            beta_l = beta_l.unsqueeze(1)    # [B, 1, l_dim]
        
        # Apply FiLM conditioning to each module
        conditioned_h = h_tokens * (1 + gamma_h) + beta_h
        conditioned_l = l_tokens * (1 + gamma_l) + beta_l
        
        return conditioned_h, conditioned_l
    
    def apply_film_conditioning(self, h_features: torch.Tensor, l_features: torch.Tensor,
                              conditioning: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply FiLM conditioning mechanism to features."""
        # Direct FiLM application
        film_h = self._apply_film_to_tokens(h_features, conditioning, 'h')
        film_l = self._apply_film_to_tokens(l_features, conditioning, 'l')
        
        return film_h, film_l
    
    def _apply_film_to_tokens(self, tokens: torch.Tensor, conditioning: torch.Tensor, 
                             module_type: str) -> torch.Tensor:
        """Apply FiLM modulation to tokens."""
        # Project conditioning to hidden space
        projected = self.conditioning_projector(conditioning)  # [B, hidden_dim]
        
        # Generate gamma and beta for both modules
        gamma_all = self.film_gamma(projected)  # [B, h_dim + l_dim]
        beta_all = self.film_beta(projected)    # [B, h_dim + l_dim]
        
        # Split parameters for appropriate module
        if module_type == 'h':
            gamma = gamma_all[:, :self.h_dim]  # [B, h_dim]
            beta = beta_all[:, :self.h_dim]    # [B, h_dim]
        elif module_type == 'l':
            gamma = gamma_all[:, self.h_dim:]  # [B, l_dim]
            beta = beta_all[:, self.h_dim:]    # [B, l_dim]
        else:
            raise ValueError(f"Unknown module type: {module_type}")
        
        # Expand for token dimension: [B, 1, dim] to broadcast over sequence
        gamma = gamma.unsqueeze(1)  # [B, 1, dim]
        beta = beta.unsqueeze(1)    # [B, 1, dim]
        
        # Apply FiLM: modulated = tokens * (1 + gamma) + beta
        modulated_tokens = tokens * (1.0 + gamma) + beta
        
        return modulated_tokens


class ConditioningInterface(nn.Module):
    """Interface for multi-source conditioning combination."""
    
    def __init__(self, pattern_dim: int = 128, rag_dim: int = 256, regime_dim: int = 64,
                 output_dim: int = 192, strength_control: bool = False, 
                 temporal_smoothing: bool = False):
        super().__init__()
        self.pattern_dim = pattern_dim
        self.rag_dim = rag_dim
        self.regime_dim = regime_dim
        self.output_dim = output_dim
        self.strength_control = strength_control
        self.temporal_smoothing = temporal_smoothing
        
        # Projection layers for each conditioning source
        self.pattern_projector = nn.Linear(pattern_dim, output_dim // 3, bias=False)
        self.rag_projector = nn.Linear(rag_dim, output_dim // 3, bias=False)
        self.regime_projector = nn.Linear(regime_dim, output_dim - 2 * (output_dim // 3), bias=False)
        
        # Final combination layer
        self.combiner = nn.Linear(output_dim, output_dim, bias=True)
        
        # Strength control mechanism
        if strength_control:
            self.strength_gate = nn.Sequential(
                nn.Linear(output_dim, output_dim // 4),
                nn.ReLU(),
                nn.Linear(output_dim // 4, 1),
                nn.Sigmoid()
            )
        
        # Temporal smoothing state
        if temporal_smoothing:
            self.register_buffer('previous_output', None)
            self.temporal_weight = 0.3  # More aggressive smoothing for test stability
    
    def combine_conditioning_sources(self, patterns: torch.Tensor = None, 
                                   rag_context: torch.Tensor = None,
                                   regime_state: torch.Tensor = None,
                                   conditioning_strength: float = 1.0,
                                   previous_conditioning: torch.Tensor = None) -> torch.Tensor:
        """Combine multiple conditioning sources into unified representation."""
        batch_size = self._infer_batch_size(patterns, rag_context, regime_state)
        device = self._infer_device(patterns, rag_context, regime_state)
        
        # Project each available source
        projected_parts = []
        
        if patterns is not None:
            proj_patterns = self.pattern_projector(patterns)
        else:
            proj_patterns = torch.zeros(batch_size, self.output_dim // 3, device=device)
        projected_parts.append(proj_patterns)
        
        if rag_context is not None:
            proj_rag = self.rag_projector(rag_context)
        else:
            proj_rag = torch.zeros(batch_size, self.output_dim // 3, device=device)
        projected_parts.append(proj_rag)
        
        if regime_state is not None:
            proj_regime = self.regime_projector(regime_state)
        else:
            regime_dim = self.output_dim - 2 * (self.output_dim // 3)
            proj_regime = torch.zeros(batch_size, regime_dim, device=device)
        projected_parts.append(proj_regime)
        
        # Combine projected sources
        combined = torch.cat(projected_parts, dim=-1)  # [B, output_dim]
        combined = self.combiner(combined)
        
        # Apply strength control if enabled
        if self.strength_control:
            strength_factor = self.strength_gate(combined) * conditioning_strength
            combined = combined * strength_factor
        else:
            combined = combined * conditioning_strength
        
        # Apply temporal smoothing if enabled
        if self.temporal_smoothing:
            if previous_conditioning is not None:
                combined = self.temporal_weight * combined + (1 - self.temporal_weight) * previous_conditioning
            elif self.previous_output is not None and self.previous_output.shape == combined.shape:
                combined = self.temporal_weight * combined + (1 - self.temporal_weight) * self.previous_output
            
            # Update state for next time - always update for temporal consistency
            self.previous_output = combined.detach().clone()
        
        return combined
    
    def _infer_batch_size(self, *tensors) -> int:
        """Infer batch size from available tensors."""
        for tensor in tensors:
            if tensor is not None:
                return tensor.shape[0]
        return 1  # Default batch size
    
    def _infer_device(self, *tensors):
        """Infer device from available tensors."""
        for tensor in tensors:
            if tensor is not None:
                return tensor.device
        return torch.device('cpu')  # Default device


class ParameterBudgetManager:
    """Parameter budget management and enforcement."""
    
    def __init__(self, total_budget: int = 27_500_000, hrm_base_params: int = 26_000_000,
                 adapter_max_params: int = 1_500_000, strict_enforcement: bool = False,
                 adaptive_allocation: bool = False):
        self.total_budget = total_budget
        self.hrm_base_params = hrm_base_params
        self.adapter_max_params = adapter_max_params
        self.strict_enforcement = strict_enforcement
        self.adaptive_allocation = adaptive_allocation
        self.components = {}
        self.component_priorities = {}
    
    def get_current_usage(self) -> Dict[str, int]:
        """Get current parameter usage breakdown."""
        hrm_params = self.count_component_params('hrm') if 'hrm' in self.components else 0
        adapter_params = self.count_component_params('adapter') if 'adapter' in self.components else 0
        total_params = sum(self.count_component_params(name) for name in self.components)
        
        return {
            'total': total_params,
            'available': max(0, self.total_budget - total_params),
            'hrm_base': hrm_params,
            'adapter': adapter_params
        }
    
    def can_allocate(self, param_count: int) -> bool:
        """Check if parameter allocation is within budget."""
        current_total = sum(self.count_component_params(name) for name in self.components)
        return (current_total + param_count) <= self.adapter_max_params
    
    def register_component(self, name: str, component) -> None:
        """Register component for budget tracking."""
        param_count = sum(p.numel() for p in component.parameters() if p.requires_grad)
        
        if self.strict_enforcement and self.would_violate(name, component):
            raise ValueError(f"Component '{name}' would violate budget constraints: "
                           f"{param_count} params would exceed remaining budget")
        
        self.components[name] = component
    
    def has_violations(self) -> bool:
        """Check if current allocation violates budget constraints."""
        total_params = self.get_total_params()
        return total_params > self.total_budget
    
    def would_violate(self, name: str, component) -> bool:
        """Check if adding component would violate budget."""
        param_count = sum(p.numel() for p in component.parameters() if p.requires_grad)
        current_total = sum(self.count_component_params(existing_name) 
                          for existing_name in self.components if existing_name != name)
        return (current_total + param_count) > self.total_budget
    
    def count_component_params(self, name: str) -> int:
        """Count parameters for a registered component."""
        if name not in self.components:
            return 0
        
        component = self.components[name]
        if hasattr(component, 'parameters'):
            return sum(p.numel() for p in component.parameters() if p.requires_grad)
        elif hasattr(component, 'numel'):
            return component.numel()
        else:
            return 0
    
    def get_total_params(self) -> int:
        """Get total parameter count across all components."""
        return sum(self.count_component_params(name) for name in self.components)
    
    def set_component_priorities(self, priorities: Dict[str, float]) -> None:
        """Set component priorities for adaptive allocation."""
        self.component_priorities = priorities.copy()
    
    def get_component_allocation(self) -> Dict[str, int]:
        """Calculate component allocation based on priorities."""
        if not self.adaptive_allocation or not self.component_priorities:
            return {}
        
        total_priority = sum(self.component_priorities.values())
        allocation = {}
        
        for component, priority in self.component_priorities.items():
            allocation[component] = int((priority / total_priority) * self.total_budget)
        
        return allocation


class GradientFlowMonitor:
    """Gradient flow monitoring and debugging."""
    
    def __init__(self, vanishing_threshold: float = 1e-6, exploding_threshold: float = 100.0):
        self.vanishing_threshold = vanishing_threshold
        self.exploding_threshold = exploding_threshold
        self.layers = []
        self.gradient_stats = {}
        self.layer_names = []
    
    def register_layers(self, layers) -> None:
        """Register layers for gradient monitoring."""
        if isinstance(layers, nn.Sequential):
            self.layers = list(layers)
            self.layer_names = [f"layer_{i}" for i in range(len(self.layers))]
        elif isinstance(layers, (list, tuple)):
            self.layers = list(layers)
            self.layer_names = [f"layer_{i}" for i in range(len(self.layers))]
        elif isinstance(layers, nn.Module):
            self.layers = [layers]
            self.layer_names = [layers.__class__.__name__]
        else:
            raise ValueError(f"Unsupported layers type: {type(layers)}")
    
    def get_gradient_stats(self) -> Dict[str, float]:
        """Compute and return gradient statistics."""
        if not self.layers:
            return {
                'mean_grad_norm': 0.0,
                'max_grad_norm': 0.0,
                'layer_wise_grads': {}
            }
        
        grad_norms = []
        layer_wise_grads = {}
        
        for i, layer in enumerate(self.layers):
            layer_grad_norms = []
            layer_name = self.layer_names[i] if i < len(self.layer_names) else f"layer_{i}"
            
            for param in layer.parameters():
                if param.grad is not None:
                    grad_norm = param.grad.data.norm().item()
                    grad_norms.append(grad_norm)
                    layer_grad_norms.append(grad_norm)
            
            if layer_grad_norms:
                layer_wise_grads[layer_name] = np.mean(layer_grad_norms)
            else:
                layer_wise_grads[layer_name] = 0.0
        
        mean_grad_norm = np.mean(grad_norms) if grad_norms else 0.0
        max_grad_norm = np.max(grad_norms) if grad_norms else 0.0
        
        self.gradient_stats = {
            'mean_grad_norm': mean_grad_norm,
            'max_grad_norm': max_grad_norm,
            'layer_wise_grads': layer_wise_grads
        }
        
        return self.gradient_stats
    
    def check_gradient_health(self) -> List[str]:
        """Check gradient health and return warnings."""
        warnings_list = []
        stats = self.get_gradient_stats()
        
        # Check for vanishing gradients
        if stats['mean_grad_norm'] < self.vanishing_threshold:
            warnings_list.append(f"Potential vanishing gradients detected: "
                               f"mean grad norm {stats['mean_grad_norm']:.2e} < {self.vanishing_threshold:.2e}")
        
        # Check for exploding gradients
        if stats['max_grad_norm'] > self.exploding_threshold:
            warnings_list.append(f"Potential exploding gradients detected: "
                               f"max grad norm {stats['max_grad_norm']:.2e} > {self.exploding_threshold:.2e}")
        
        # Check layer-wise gradient issues
        for layer_name, grad_norm in stats['layer_wise_grads'].items():
            if grad_norm < self.vanishing_threshold:
                warnings_list.append(f"Layer {layer_name}: vanishing gradients (norm: {grad_norm:.2e})")
            elif grad_norm > self.exploding_threshold:
                warnings_list.append(f"Layer {layer_name}: exploding gradients (norm: {grad_norm:.2e})")
        
        return warnings_list
    
    def get_visualization_data(self) -> Dict[str, List]:
        """Get data for gradient flow visualization."""
        stats = self.get_gradient_stats()
        
        layer_names = list(stats['layer_wise_grads'].keys())
        gradient_norms = list(stats['layer_wise_grads'].values())
        
        # Compute weight norms for comparison
        weight_norms = []
        for layer in self.layers:
            layer_weight_norms = []
            for param in layer.parameters():
                if param.data is not None:
                    weight_norm = param.data.norm().item()
                    layer_weight_norms.append(weight_norm)
            
            if layer_weight_norms:
                weight_norms.append(np.mean(layer_weight_norms))
            else:
                weight_norms.append(0.0)
        
        return {
            'layer_names': layer_names,
            'gradient_norms': gradient_norms,
            'weight_norms': weight_norms
        }


class HRMIntegrationLayer:
    """Integration layer combining all HRM conditioning components."""
    
    def __init__(self, hrm_config=None, conditioning_dim: int = 192, 
                 max_additional_params: int = 300_000):
        self.hrm_config = hrm_config or HRMConfig()
        self.conditioning_dim = conditioning_dim
        self.max_additional_params = max_additional_params
        
        # Initialize core components
        self.hrm_adapter = HRMAdapter(self.hrm_config, conditioning_dim, max_additional_params)
        self.conditioning_interface = ConditioningInterface(output_dim=conditioning_dim)
        self.parameter_budget_manager = ParameterBudgetManager()
        self.gradient_flow_monitor = GradientFlowMonitor()
        
        # Register components for budget tracking
        self.parameter_budget_manager.register_component('hrm_adapter', self.hrm_adapter)
        self.parameter_budget_manager.register_component('conditioning_interface', self.conditioning_interface)
    
    def apply_conditioning(self, h_tokens, l_tokens, conditioning_sources=None):
        """Apply complete conditioning pipeline to HRM tokens."""
        if conditioning_sources is None:
            conditioning_sources = {}
        
        # Combine conditioning sources
        conditioning_vector = self.conditioning_interface.combine_conditioning_sources(
            patterns=conditioning_sources.get('patterns'),
            rag_context=conditioning_sources.get('rag_context'),
            regime_state=conditioning_sources.get('regime_state')
        )
        
        # Apply FiLM conditioning to HRM tokens
        return self.hrm_adapter.apply_conditioning(h_tokens, l_tokens, conditioning_vector)
    
    def get_parameter_usage(self):
        """Get current parameter usage statistics."""
        return self.parameter_budget_manager.get_current_usage()


# Legacy compatibility for existing imports
HRMWalkForwardIntegrator = type('HRMWalkForwardIntegrator', (), {
    '__init__': lambda self: None,
    'integrate_patterns': lambda self, data: data
})

# Export HRMConfig function as class-like interface for tests
__all__ = ['HRMAdapter', 'ConditioningInterface', 'ParameterBudgetManager', 
           'GradientFlowMonitor', 'HRMWalkForwardIntegrator', 'HRMConfig',
           'HRMIntegrationLayer']