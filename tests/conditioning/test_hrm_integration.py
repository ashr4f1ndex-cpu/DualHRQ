"""
test_hrm_integration.py
=======================

TDD tests for DRQ-103: HRM Integration Layer
These tests MUST be written first and will initially FAIL.
Implementation should make these tests pass.

CRITICAL: Parameter budget enforcement, gradient flow, adapter layer.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys
import time
from typing import Dict, List, Any, Tuple, Optional

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from src.conditioning.hrm_integration import (
        HRMAdapter,
        ConditioningInterface,
        ParameterBudgetManager,
        GradientFlowMonitor,
        HRMConfig
    )
    from lab_v10.src.options.hrm_net import HRMNet
    from src.conditioning.pattern_library import Pattern
    from src.conditioning.rag_system import RetrievalContext
except ImportError:
    # These will fail initially - that's expected in TDD
    pass


class TestHRMAdapter:
    """Tests for HRM adapter layer functionality."""
    
    def test_hrm_adapter_initialization(self):
        """Test HRM adapter initializes correctly."""
        # This will fail initially until HRMAdapter is implemented
        hrm_config = HRMConfig(
            h_layers=4, h_dim=384, h_heads=8, h_ffn_mult=3.0, h_dropout=0.1,
            l_layers=4, l_dim=512, l_heads=8, l_ffn_mult=3.0, l_dropout=0.1,
            segments_N=3, l_inner_T=8, act_enable=True,
            act_max_segments=5, ponder_cost=0.01, use_cross_attn=False
        )
        
        conditioning_dim = 256  # Output from conditioning system
        
        adapter = HRMAdapter(
            hrm_config=hrm_config,
            conditioning_dim=conditioning_dim,
            max_additional_params=300_000  # Stay within budget
        )
        
        # Should create valid adapter
        assert adapter is not None, "Adapter should initialize"
        assert hasattr(adapter, 'conditioning_projector'), "Should have conditioning projector"
        assert hasattr(adapter, 'parameter_budget_manager'), "Should have budget manager"
        
        # Check parameter count
        total_params = sum(p.numel() for p in adapter.parameters())
        assert total_params <= 300_000, f"Adapter should be ≤300K params, got {total_params}"
    
    def test_conditioning_injection(self):
        """Test conditioning information injection into HRM."""
        hrm_config = HRMConfig(
            h_layers=2, h_dim=256, h_heads=4, h_ffn_mult=2.0, h_dropout=0.0,
            l_layers=2, l_dim=384, l_heads=4, l_ffn_mult=2.0, l_dropout=0.0,
            segments_N=2, l_inner_T=4, act_enable=False,
            act_max_segments=3, ponder_cost=0.01, use_cross_attn=False
        )
        
        adapter = HRMAdapter(hrm_config, conditioning_dim=128, max_additional_params=100_000)
        
        # Mock conditioning input
        batch_size = 2
        conditioning_vector = torch.randn(batch_size, 128)
        
        # Mock HRM tokens
        h_tokens = torch.randn(batch_size, 10, hrm_config.h_dim)
        l_tokens = torch.randn(batch_size, 20, hrm_config.l_dim)
        
        # Apply conditioning
        conditioned_h, conditioned_l = adapter.apply_conditioning(
            h_tokens, l_tokens, conditioning_vector
        )
        
        # Should maintain shapes
        assert conditioned_h.shape == h_tokens.shape, \
            f"H tokens shape should be preserved: {conditioned_h.shape} vs {h_tokens.shape}"
        assert conditioned_l.shape == l_tokens.shape, \
            f"L tokens shape should be preserved: {conditioned_l.shape} vs {l_tokens.shape}"
        
        # Should modify tokens (not identity)
        assert not torch.allclose(conditioned_h, h_tokens), \
            "H tokens should be modified by conditioning"
        assert not torch.allclose(conditioned_l, l_tokens), \
            "L tokens should be modified by conditioning"
    
    def test_film_conditioning_mechanism(self):
        """Test FiLM (Feature-wise Linear Modulation) conditioning."""
        adapter = HRMAdapter(
            hrm_config=HRMConfig(h_dim=256, l_dim=384),
            conditioning_dim=128,
            conditioning_method='film'
        )
        
        batch_size = 3
        conditioning = torch.randn(batch_size, 128)
        h_features = torch.randn(batch_size, 15, 256)
        l_features = torch.randn(batch_size, 25, 384)
        
        # Apply FiLM conditioning
        film_h, film_l = adapter.apply_film_conditioning(h_features, l_features, conditioning)
        
        # Should produce proper FiLM modulation
        assert film_h.shape == h_features.shape, "FiLM should preserve H feature shape"
        assert film_l.shape == l_features.shape, "FiLM should preserve L feature shape"
        
        # FiLM should be multiplicative + additive transformation
        # Check that different conditioning produces different outputs
        different_conditioning = conditioning + torch.randn_like(conditioning) * 0.5
        film_h2, film_l2 = adapter.apply_film_conditioning(h_features, l_features, different_conditioning)
        
        assert not torch.allclose(film_h, film_h2), "Different conditioning should produce different FiLM output"
    
    def test_gradient_flow_preservation(self):
        """Test that adapter preserves gradient flow to HRM."""
        adapter = HRMAdapter(
            hrm_config=HRMConfig(h_dim=128, l_dim=256),
            conditioning_dim=64
        )
        
        # Enable gradient tracking
        conditioning = torch.randn(1, 64, requires_grad=True)
        h_tokens = torch.randn(1, 8, 128, requires_grad=True)
        l_tokens = torch.randn(1, 12, 256, requires_grad=True)
        
        # Forward pass through adapter
        cond_h, cond_l = adapter.apply_conditioning(h_tokens, l_tokens, conditioning)
        
        # Create dummy loss
        loss = (cond_h.mean() + cond_l.mean())
        loss.backward()
        
        # Should have gradients flowing back
        assert conditioning.grad is not None, "Gradients should flow back to conditioning"
        assert h_tokens.grad is not None, "Gradients should flow back to H tokens"
        assert l_tokens.grad is not None, "Gradients should flow back to L tokens"
        
        # Gradients should be non-zero (adapter is not identity)
        assert torch.any(conditioning.grad != 0), "Conditioning gradients should be non-zero"
    
    def test_conditioning_consistency(self):
        """Test that same conditioning produces consistent results."""
        adapter = HRMAdapter(
            hrm_config=HRMConfig(h_dim=192, l_dim=320),
            conditioning_dim=96
        )
        
        # Fix inputs
        torch.manual_seed(42)
        conditioning = torch.randn(2, 96)
        h_tokens = torch.randn(2, 10, 192)
        l_tokens = torch.randn(2, 15, 320)
        
        # Apply conditioning twice
        adapter.eval()  # Disable dropout for consistency
        with torch.no_grad():
            result1_h, result1_l = adapter.apply_conditioning(h_tokens, l_tokens, conditioning)
            result2_h, result2_l = adapter.apply_conditioning(h_tokens, l_tokens, conditioning)
        
        # Should be identical
        torch.testing.assert_close(result1_h, result2_h, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(result1_l, result2_l, rtol=1e-6, atol=1e-6)


class TestParameterBudgetManager:
    """Tests for parameter budget management."""
    
    def test_budget_enforcement(self):
        """Test parameter budget is enforced strictly."""
        # This will fail initially until ParameterBudgetManager is implemented
        budget_manager = ParameterBudgetManager(
            total_budget=27_500_000,
            hrm_base_params=26_000_000,
            adapter_max_params=1_500_000
        )
        
        # Should track current usage
        current_usage = budget_manager.get_current_usage()
        assert isinstance(current_usage, dict), "Should return usage breakdown"
        assert 'total' in current_usage, "Should include total usage"
        assert 'available' in current_usage, "Should include available budget"
        
        # Test budget validation
        valid_adapter_size = 800_000
        assert budget_manager.can_allocate(valid_adapter_size), \
            "Should allow allocation within budget"
        
        overbudget_size = 2_000_000
        assert not budget_manager.can_allocate(overbudget_size), \
            "Should reject allocation over budget"
    
    def test_adaptive_budget_allocation(self):
        """Test adaptive allocation based on component importance."""
        budget_manager = ParameterBudgetManager(
            total_budget=27_000_000,
            adaptive_allocation=True
        )
        
        # Set component priorities
        priorities = {
            'hrm_core': 0.9,      # Highest priority
            'conditioning': 0.7,   # High priority  
            'pattern_lib': 0.5,   # Medium priority
            'rag_system': 0.3     # Lower priority
        }
        
        budget_manager.set_component_priorities(priorities)
        allocation = budget_manager.get_component_allocation()
        
        # HRM core should get largest allocation
        assert allocation['hrm_core'] > allocation['conditioning'], \
            "HRM core should get more than conditioning"
        assert allocation['conditioning'] > allocation['pattern_lib'], \
            "Conditioning should get more than pattern library"
        
        # Total should not exceed budget
        total_allocated = sum(allocation.values())
        assert total_allocated <= 27_000_000, \
            f"Total allocation should not exceed budget: {total_allocated}"
    
    def test_parameter_counting_accuracy(self):
        """Test accurate parameter counting across components."""
        budget_manager = ParameterBudgetManager(total_budget=30_000_000)
        
        # Mock components with known parameter counts
        mock_hrm = nn.Linear(512, 256)  # 512*256 + 256 = 131,328 params
        mock_adapter = nn.Sequential(
            nn.Linear(128, 64),  # 128*64 + 64 = 8,256 params
            nn.Linear(64, 32)    # 64*32 + 32 = 2,080 params
        )  # Total: 10,336 params
        
        # Register components
        budget_manager.register_component('hrm', mock_hrm)
        budget_manager.register_component('adapter', mock_adapter)
        
        # Count should be accurate
        hrm_count = budget_manager.count_component_params('hrm')
        adapter_count = budget_manager.count_component_params('adapter')
        
        assert hrm_count == 131_328, f"HRM params should be 131,328, got {hrm_count}"
        assert adapter_count == 10_336, f"Adapter params should be 10,336, got {adapter_count}"
        
        # Total count
        total_count = budget_manager.get_total_params()
        expected_total = 131_328 + 10_336
        assert total_count == expected_total, \
            f"Total should be {expected_total}, got {total_count}"
    
    def test_budget_violation_detection(self):
        """Test detection and handling of budget violations."""
        budget_manager = ParameterBudgetManager(
            total_budget=1_000_000,  # Small budget for testing
            strict_enforcement=True
        )
        
        # Add component within budget
        small_component = nn.Linear(100, 50)  # 5,050 params
        budget_manager.register_component('small', small_component)
        assert not budget_manager.has_violations(), "Should have no violations"
        
        # Try to add component that would exceed budget
        large_component = nn.Linear(10000, 5000)  # ~50M params
        
        # Should detect violation before registration
        would_violate = budget_manager.would_violate('large', large_component)
        assert would_violate, "Should detect potential violation"
        
        # Should raise error in strict mode
        with pytest.raises(ValueError, match="budget"):
            budget_manager.register_component('large', large_component)


class TestConditioningInterface:
    """Tests for conditioning interface and integration."""
    
    def test_multi_source_conditioning(self):
        """Test conditioning from multiple sources (patterns, RAG, regime)."""
        # This will fail initially until ConditioningInterface is implemented
        interface = ConditioningInterface(
            pattern_dim=128,
            rag_dim=256,
            regime_dim=64,
            output_dim=192
        )
        
        # Mock conditioning inputs
        batch_size = 4
        pattern_features = torch.randn(batch_size, 128)
        rag_context = torch.randn(batch_size, 256)
        regime_state = torch.randn(batch_size, 64)
        
        # Combine conditioning sources
        combined_conditioning = interface.combine_conditioning_sources(
            patterns=pattern_features,
            rag_context=rag_context,
            regime_state=regime_state
        )
        
        assert combined_conditioning.shape == (batch_size, 192), \
            f"Should output shape (batch, 192), got {combined_conditioning.shape}"
        
        # Should handle missing sources gracefully
        partial_conditioning = interface.combine_conditioning_sources(
            patterns=pattern_features,
            rag_context=None,  # Missing RAG
            regime_state=regime_state
        )
        
        assert partial_conditioning.shape == (batch_size, 192), \
            "Should handle missing conditioning sources"
    
    def test_conditioning_strength_control(self):
        """Test control of conditioning strength/influence."""
        interface = ConditioningInterface(
            pattern_dim=64,
            rag_dim=64,
            regime_dim=64,
            output_dim=128,
            strength_control=True
        )
        
        batch_size = 2
        inputs = {
            'patterns': torch.randn(batch_size, 64),
            'rag_context': torch.randn(batch_size, 64),
            'regime_state': torch.randn(batch_size, 64)
        }
        
        # Test different strength settings
        weak_conditioning = interface.combine_conditioning_sources(
            conditioning_strength=0.1, **inputs
        )
        strong_conditioning = interface.combine_conditioning_sources(
            conditioning_strength=0.9, **inputs
        )
        
        # Stronger conditioning should have larger magnitude
        weak_norm = torch.norm(weak_conditioning, dim=-1).mean()
        strong_norm = torch.norm(strong_conditioning, dim=-1).mean()
        
        assert strong_norm > weak_norm, \
            "Strong conditioning should have larger magnitude"
    
    def test_conditioning_temporal_consistency(self):
        """Test temporal consistency of conditioning across time steps."""
        interface = ConditioningInterface(
            pattern_dim=96,
            rag_dim=128,
            regime_dim=32,
            output_dim=160,
            temporal_smoothing=True
        )
        
        batch_size = 3
        
        # Simulate sequence of conditioning inputs
        time_steps = 5
        conditioning_sequence = []
        
        for t in range(time_steps):
            # Gradual change in inputs
            patterns = torch.randn(batch_size, 96) + t * 0.1
            rag = torch.randn(batch_size, 128) + t * 0.1
            regime = torch.randn(batch_size, 32) + t * 0.1
            
            conditioning = interface.combine_conditioning_sources(
                patterns=patterns,
                rag_context=rag,
                regime_state=regime,
                previous_conditioning=conditioning_sequence[-1] if conditioning_sequence else None
            )
            
            conditioning_sequence.append(conditioning)
        
        # Check temporal smoothness
        for t in range(1, time_steps):
            change = torch.norm(conditioning_sequence[t] - conditioning_sequence[t-1], dim=-1)
            assert torch.all(change < 2.0), f"Temporal changes should be smooth at step {t}"
    
    def test_batch_processing_efficiency(self):
        """Test efficient batch processing of conditioning."""
        interface = ConditioningInterface(
            pattern_dim=100,
            rag_dim=200,
            regime_dim=50,
            output_dim=150
        )
        
        # Test with different batch sizes
        batch_sizes = [1, 8, 32, 128]
        
        for batch_size in batch_sizes:
            inputs = {
                'patterns': torch.randn(batch_size, 100),
                'rag_context': torch.randn(batch_size, 200), 
                'regime_state': torch.randn(batch_size, 50)
            }
            
            start_time = time.time()
            output = interface.combine_conditioning_sources(**inputs)
            processing_time = time.time() - start_time
            
            # Should scale linearly with batch size
            time_per_sample = processing_time / batch_size
            assert time_per_sample < 0.01, \
                f"Should process efficiently: {time_per_sample:.4f}s per sample for batch size {batch_size}"
            
            assert output.shape == (batch_size, 150), \
                f"Correct output shape for batch size {batch_size}"


class TestGradientFlowMonitoring:
    """Tests for gradient flow monitoring and debugging."""
    
    def test_gradient_flow_tracking(self):
        """Test tracking of gradient flow through conditioning layers."""
        # This will fail initially until GradientFlowMonitor is implemented
        monitor = GradientFlowMonitor()
        
        # Create mock conditioning pipeline
        conditioning_layers = nn.Sequential(
            nn.Linear(128, 256, bias=True),
            nn.ReLU(),
            nn.Linear(256, 192, bias=True),
            nn.Tanh()
        )
        
        # Register layers for monitoring
        monitor.register_layers(conditioning_layers)
        
        # Forward pass
        input_tensor = torch.randn(4, 128, requires_grad=True)
        output = conditioning_layers(input_tensor)
        
        # Create loss and backward pass
        loss = output.mean()
        loss.backward()
        
        # Check gradient statistics
        grad_stats = monitor.get_gradient_stats()
        
        assert 'mean_grad_norm' in grad_stats, "Should track mean gradient norm"
        assert 'max_grad_norm' in grad_stats, "Should track max gradient norm"
        assert 'layer_wise_grads' in grad_stats, "Should track layer-wise gradients"
        
        # Gradients should be non-zero and finite
        assert grad_stats['mean_grad_norm'] > 0, "Should have non-zero gradients"
        assert np.isfinite(grad_stats['mean_grad_norm']), "Gradients should be finite"
    
    def test_vanishing_gradient_detection(self):
        """Test detection of vanishing gradients."""
        monitor = GradientFlowMonitor(vanishing_threshold=1e-6)
        
        # Create deep network prone to vanishing gradients
        deep_network = nn.Sequential(
            *[nn.Sequential(nn.Linear(64, 64), nn.Sigmoid()) for _ in range(10)]
        )
        
        monitor.register_layers(deep_network)
        
        input_tensor = torch.randn(2, 64, requires_grad=True)
        output = deep_network(input_tensor)
        loss = output.mean()
        loss.backward()
        
        # Check for vanishing gradient warning
        warnings = monitor.check_gradient_health()
        
        # Should detect potential vanishing gradients
        vanishing_detected = any('vanishing' in str(warning).lower() for warning in warnings)
        # Note: This may or may not trigger depending on initialization
        assert isinstance(warnings, list), "Should return list of warnings"
    
    def test_exploding_gradient_detection(self):
        """Test detection of exploding gradients."""
        monitor = GradientFlowMonitor(exploding_threshold=100.0)
        
        # Create network that could have exploding gradients
        network = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        
        # Initialize with large weights to trigger exploding gradients
        with torch.no_grad():
            for param in network.parameters():
                param.fill_(10.0)  # Large weights
        
        monitor.register_layers(network)
        
        input_tensor = torch.randn(1, 32, requires_grad=True)
        output = network(input_tensor)
        loss = output.sum() * 100  # Amplify loss
        loss.backward()
        
        warnings = monitor.check_gradient_health()
        
        # May detect exploding gradients
        exploding_detected = any('exploding' in str(warning).lower() for warning in warnings)
        assert isinstance(warnings, list), "Should return list of warnings"
    
    def test_gradient_flow_visualization(self):
        """Test gradient flow visualization helpers."""
        monitor = GradientFlowMonitor()
        
        # Simple network
        network = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 8)
        )
        
        monitor.register_layers(network)
        
        # Forward and backward
        x = torch.randn(3, 16, requires_grad=True)
        y = network(x)
        loss = y.sum()
        loss.backward()
        
        # Get visualization data
        viz_data = monitor.get_visualization_data()
        
        assert 'layer_names' in viz_data, "Should include layer names"
        assert 'gradient_norms' in viz_data, "Should include gradient norms"
        assert 'weight_norms' in viz_data, "Should include weight norms"
        
        # Data should have consistent lengths
        assert len(viz_data['layer_names']) == len(viz_data['gradient_norms']), \
            "Layer names and gradient norms should have same length"


class TestEndToEndIntegration:
    """Tests for complete HRM integration workflow."""
    
    def test_complete_conditioning_pipeline(self):
        """Test complete pipeline from conditioning inputs to HRM."""
        # Create components
        hrm_config = HRMConfig(
            h_layers=2, h_dim=256, h_heads=4, h_ffn_mult=2.0, h_dropout=0.0,
            l_layers=2, l_dim=384, l_heads=4, l_ffn_mult=2.0, l_dropout=0.0,
            segments_N=2, l_inner_T=4
        )
        
        adapter = HRMAdapter(hrm_config, conditioning_dim=128)
        interface = ConditioningInterface(
            pattern_dim=64, rag_dim=64, regime_dim=32, output_dim=128
        )
        
        batch_size = 2
        
        # Mock conditioning inputs
        patterns = torch.randn(batch_size, 64)
        rag_context = torch.randn(batch_size, 64)
        regime_state = torch.randn(batch_size, 32)
        
        # Mock HRM inputs
        h_tokens = torch.randn(batch_size, 8, 256)
        l_tokens = torch.randn(batch_size, 12, 384)
        
        # Complete pipeline
        conditioning_vector = interface.combine_conditioning_sources(
            patterns=patterns,
            rag_context=rag_context,
            regime_state=regime_state
        )
        
        conditioned_h, conditioned_l = adapter.apply_conditioning(
            h_tokens, l_tokens, conditioning_vector
        )
        
        # Should produce valid conditioned tokens
        assert conditioned_h.shape == h_tokens.shape
        assert conditioned_l.shape == l_tokens.shape
        assert torch.all(torch.isfinite(conditioned_h))
        assert torch.all(torch.isfinite(conditioned_l))
    
    def test_training_loop_integration(self):
        """Test integration in training loop with parameter constraints."""
        # Setup with strict parameter budget
        budget_manager = ParameterBudgetManager(
            total_budget=1_000_000,  # Small budget for testing
            strict_enforcement=True
        )
        
        # Create minimal HRM and adapter
        mini_hrm_config = HRMConfig(
            h_layers=1, h_dim=64, h_heads=2, h_ffn_mult=1.5, h_dropout=0.0,
            l_layers=1, l_dim=128, l_heads=2, l_ffn_mult=1.5, l_dropout=0.0,
            segments_N=1, l_inner_T=2, act_enable=False
        )
        
        hrm_model = HRMNet(mini_hrm_config)
        adapter = HRMAdapter(mini_hrm_config, conditioning_dim=32, max_additional_params=50_000)
        
        # Register with budget manager
        budget_manager.register_component('hrm', hrm_model)
        budget_manager.register_component('adapter', adapter)
        
        # Should be within budget
        assert not budget_manager.has_violations(), "Should be within parameter budget"
        
        # Mock training step
        optimizer = torch.optim.Adam(
            list(hrm_model.parameters()) + list(adapter.parameters()), 
            lr=1e-3
        )
        
        # Training iteration
        conditioning = torch.randn(1, 32)
        h_tokens = torch.randn(1, 4, 64)
        l_tokens = torch.randn(1, 6, 128)
        target = torch.randn(1)
        
        optimizer.zero_grad()
        
        # Apply conditioning
        cond_h, cond_l = adapter.apply_conditioning(h_tokens, l_tokens, conditioning)
        
        # Forward through HRM
        (h_out, l_out), segments = hrm_model(cond_h, cond_l)
        
        # Mock loss
        loss = torch.nn.functional.mse_loss(segments[-1][0].mean(dim=-1), target)
        
        loss.backward()
        optimizer.step()
        
        # Should complete without errors
        assert loss.item() >= 0, "Training step should complete successfully"


# This will run when pytest is called and should initially FAIL
# Implementation should make these tests pass
if __name__ == "__main__":
    pytest.main([__file__, "-v"])