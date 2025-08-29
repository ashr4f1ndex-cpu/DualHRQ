"""
HRM Integration Tests - Week 5 DRQ-104 Implementation
====================================================

Comprehensive tests for HRM adapter layer with conditioning integration.
These tests enforce the acceptance criteria from the 26-week plan.
"""

import torch
import pytest
import sys
import os
sys.path.insert(0, os.path.abspath('src'))

from src.conditioning.hrm_integration import HRMIntegrationLayer
from src.models.conditioning.film_conditioning import FiLMConditioning
from tools.param_count import count_hrm_parameters


class TestHRMConditioningIntegration:
    """Test HRM adapter integration with conditioning system."""
    
    def test_hrm_conditioning_integration_functional(self):
        """Test that conditioning actually affects HRM outputs."""
        # Initialize integration layer
        integration_layer = HRMIntegrationLayer()
        
        # Create mock HRM tokens with correct dimensions from config
        batch_size, seq_len = 2, 10
        h_dim = integration_layer.hrm_adapter.h_dim  # 256
        l_dim = integration_layer.hrm_adapter.l_dim  # 384
        h_tokens = torch.randn(batch_size, seq_len, h_dim)
        l_tokens = torch.randn(batch_size, seq_len, l_dim)
        
        # Test with different conditioning sources
        conditioning_sources_1 = {
            'patterns': torch.randn(batch_size, 128),
            'rag_context': torch.randn(batch_size, 256),
            'regime_state': torch.randn(batch_size, 64)
        }
        
        conditioning_sources_2 = {
            'patterns': torch.randn(batch_size, 128),
            'rag_context': torch.randn(batch_size, 256), 
            'regime_state': torch.randn(batch_size, 64)
        }
        
        # Apply conditioning with different sources
        h_conditioned_1, l_conditioned_1 = integration_layer.apply_conditioning(
            h_tokens, l_tokens, conditioning_sources_1
        )
        
        h_conditioned_2, l_conditioned_2 = integration_layer.apply_conditioning(
            h_tokens, l_tokens, conditioning_sources_2
        )
        
        # Verify conditioning actually changes outputs
        h_diff = torch.norm(h_conditioned_1 - h_conditioned_2)
        l_diff = torch.norm(l_conditioned_1 - l_conditioned_2)
        
        assert h_diff > 0.01, "H-module conditioning should affect outputs"
        assert l_diff > 0.01, "L-module conditioning should affect outputs"
        
        # Verify shapes are preserved
        assert h_conditioned_1.shape == h_tokens.shape
        assert l_conditioned_1.shape == l_tokens.shape
    
    def test_total_parameter_budget_compliance(self):
        """Test that total parameters are within 26.5M-27.5M range."""
        integration_layer = HRMIntegrationLayer()
        usage_stats = integration_layer.get_parameter_usage()
        
        total_params = usage_stats['total']
        
        # Verify within budget range
        assert 26_500_000 <= total_params <= 27_500_000, \
            f"Total parameters {total_params:,} outside budget range 26.5M-27.5M"
        
        # Log parameter breakdown for analysis
        print(f"Parameter usage: {usage_stats}")
        
    def test_conditioning_effect_strength(self):
        """Test that conditioning effects scale appropriately."""
        integration_layer = HRMIntegrationLayer()
        
        batch_size, seq_len = 1, 5
        h_dim = integration_layer.hrm_adapter.h_dim  # 256
        l_dim = integration_layer.hrm_adapter.l_dim  # 384
        h_tokens = torch.randn(batch_size, seq_len, h_dim)
        l_tokens = torch.randn(batch_size, seq_len, l_dim)
        
        # Weak conditioning
        weak_conditioning = {
            'patterns': torch.randn(batch_size, 128) * 0.1,
            'regime_state': torch.randn(batch_size, 64) * 0.1
        }
        
        # Strong conditioning  
        strong_conditioning = {
            'patterns': torch.randn(batch_size, 128) * 2.0,
            'regime_state': torch.randn(batch_size, 64) * 2.0
        }
        
        h_weak, l_weak = integration_layer.apply_conditioning(
            h_tokens, l_tokens, weak_conditioning
        )
        
        h_strong, l_strong = integration_layer.apply_conditioning(
            h_tokens, l_tokens, strong_conditioning
        )
        
        # Strong conditioning should have larger effect
        weak_h_change = torch.norm(h_weak - h_tokens)
        strong_h_change = torch.norm(h_strong - h_tokens)
        
        assert strong_h_change > weak_h_change, \
            "Strong conditioning should have larger effect than weak conditioning"


class TestMemoryOptimization:
    """Test memory usage optimization."""
    
    def test_memory_usage_within_bounds(self):
        """Test that memory usage is optimized."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Initialize integration layer
        integration_layer = HRMIntegrationLayer()
        
        # Simulate realistic batch processing with correct dimensions
        batch_size, seq_len = 8, 50
        h_dim = 256  # From HRM config
        l_dim = 384  # From HRM config
        
        for _ in range(10):  # Multiple batches
            h_tokens = torch.randn(batch_size, seq_len, h_dim)
            l_tokens = torch.randn(batch_size, seq_len, l_dim)
            
            conditioning_sources = {
                'patterns': torch.randn(batch_size, 128),
                'rag_context': torch.randn(batch_size, 256),
                'regime_state': torch.randn(batch_size, 64)
            }
            
            h_conditioned, l_conditioned = integration_layer.apply_conditioning(
                h_tokens, l_tokens, conditioning_sources
            )
            
            # Force cleanup
            del h_tokens, l_tokens, h_conditioned, l_conditioned
            
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - baseline_memory
        
        # Memory increase should be reasonable (less than 500MB for this test)
        assert memory_increase < 500, \
            f"Memory usage increased by {memory_increase:.1f}MB, exceeds 500MB limit"


class TestTrainingDynamicsPreservation:
    """Test that training dynamics are preserved."""
    
    def test_gradient_flow_preservation(self):
        """Test that gradients flow properly through conditioning."""
        integration_layer = HRMIntegrationLayer()
        
        batch_size, seq_len = 2, 10
        h_dim = integration_layer.hrm_adapter.h_dim  # 256
        l_dim = integration_layer.hrm_adapter.l_dim  # 384
        h_tokens = torch.randn(batch_size, seq_len, h_dim, requires_grad=True)
        l_tokens = torch.randn(batch_size, seq_len, l_dim, requires_grad=True)
        
        conditioning_sources = {
            'patterns': torch.randn(batch_size, 128, requires_grad=True),
            'regime_state': torch.randn(batch_size, 64, requires_grad=True)
        }
        
        h_conditioned, l_conditioned = integration_layer.apply_conditioning(
            h_tokens, l_tokens, conditioning_sources
        )
        
        # Create a simple loss
        loss = torch.mean(h_conditioned) + torch.mean(l_conditioned)
        loss.backward()
        
        # Verify gradients exist
        assert h_tokens.grad is not None, "H-tokens should have gradients"
        assert l_tokens.grad is not None, "L-tokens should have gradients" 
        assert conditioning_sources['patterns'].grad is not None, \
            "Conditioning patterns should have gradients"
        
        # Verify gradient magnitudes are reasonable
        h_grad_norm = torch.norm(h_tokens.grad)
        l_grad_norm = torch.norm(l_tokens.grad)
        
        assert h_grad_norm > 1e-6, f"H-tokens gradient too small: {h_grad_norm}"
        assert l_grad_norm > 1e-6, f"L-tokens gradient too small: {l_grad_norm}"
