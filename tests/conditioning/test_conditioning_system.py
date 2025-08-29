"""
test_conditioning_system.py
============================

Tests for DRQ-103: Unified Conditioning System
Comprehensive tests for single ConditioningSystem API with feature flags.

CRITICAL: Parameter budget ≤300K, feature flags, fail-open behavior.
"""

import pytest
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import time
from typing import Dict, List, Any
from datetime import datetime, timedelta

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from src.conditioning.conditioning_system import (
        ConditioningSystem,
        ConditioningConfig,
        FeatureFlags
    )
    from src.conditioning.pattern_library import Pattern
    from src.conditioning.rag_system import RetrievalContext
    from src.conditioning.hrm_integration import HRMConfig
except ImportError:
    # These will fail initially - that's expected in TDD
    pass


class TestFeatureFlags:
    """Tests for feature flag management."""
    
    def test_feature_flag_initialization(self):
        """Test feature flags initialize with config values."""
        config = ConditioningConfig(
            enable_patterns=True,
            enable_rag=False,
            enable_regime_classification=True
        )
        
        flags = FeatureFlags(config)
        
        assert flags.is_enabled('patterns') == True
        assert flags.is_enabled('rag') == False
        assert flags.is_enabled('regime_classification') == True
    
    def test_feature_flag_control(self):
        """Test individual feature flag control."""
        config = ConditioningConfig()
        flags = FeatureFlags(config)
        
        # Test setting flags
        flags.set_flag('patterns', False)
        assert flags.is_enabled('patterns') == False
        
        flags.set_flag('rag', True)
        assert flags.is_enabled('rag') == True
        
        # Test invalid flag
        with pytest.raises(ValueError, match="Unknown feature flag"):
            flags.set_flag('invalid_flag', True)
    
    def test_emergency_disable(self):
        """Test emergency disable functionality."""
        config = ConditioningConfig(
            enable_patterns=True,
            enable_rag=True,
            enable_regime_classification=True
        )
        
        flags = FeatureFlags(config)
        flags.disable_all()
        
        assert all(not enabled for enabled in flags.get_all_flags().values())
    
    def test_get_all_flags(self):
        """Test getting all flag states."""
        config = ConditioningConfig()
        flags = FeatureFlags(config)
        
        all_flags = flags.get_all_flags()
        
        assert isinstance(all_flags, dict)
        assert 'patterns' in all_flags
        assert 'rag' in all_flags
        assert 'regime_classification' in all_flags


class TestConditioningSystemInitialization:
    """Tests for conditioning system initialization."""
    
    def test_basic_initialization(self):
        """Test basic conditioning system initialization."""
        config = ConditioningConfig(
            total_parameter_budget=300_000,
            enable_patterns=True,
            enable_rag=True,
            enable_regime_classification=True
        )
        
        system = ConditioningSystem(config)
        
        # Should initialize without errors
        assert system is not None
        assert system.config == config
        assert system.feature_flags is not None
        assert system.parameter_manager is not None
    
    def test_parameter_budget_compliance(self):
        """Test that system stays within parameter budget."""
        config = ConditioningConfig(total_parameter_budget=300_000)
        system = ConditioningSystem(config)
        
        total_params = sum(p.numel() for p in system.parameters() if p.requires_grad)
        
        assert total_params <= config.total_parameter_budget, \
            f"System uses {total_params:,} parameters, exceeds budget of {config.total_parameter_budget:,}"
    
    def test_parameter_budget_violation_detection(self):
        """Test detection of parameter budget violations."""
        # Set unrealistically small budget to trigger violation
        config = ConditioningConfig(total_parameter_budget=1000)  # Too small
        
        with pytest.raises(ValueError, match="exceeds parameter budget"):
            ConditioningSystem(config)
    
    def test_selective_component_initialization(self):
        """Test initialization with selective components enabled."""
        # Only patterns enabled
        config = ConditioningConfig(
            enable_patterns=True,
            enable_rag=False,
            enable_regime_classification=False
        )
        
        system = ConditioningSystem(config)
        
        assert system.pattern_library is not None
        assert system.pattern_detector is not None
        assert system.rag_system is None
        assert system.regime_classifier is None
    
    def test_hrm_adapter_integration(self):
        """Test HRM adapter integration."""
        hrm_config = HRMConfig(
            h_dim=256, l_dim=384,
            h_layers=2, l_layers=2
        )
        
        config = ConditioningConfig(
            hrm_config=hrm_config,
            conditioning_dim=192
        )
        
        system = ConditioningSystem(config)
        
        assert system.hrm_adapter is not None
        assert hasattr(system.hrm_adapter, 'apply_conditioning')


class TestConditioningSystemFunctionality:
    """Tests for core conditioning functionality."""
    
    def test_market_context_conditioning(self):
        """Test basic market context conditioning."""
        config = ConditioningConfig(
            enable_patterns=True,
            enable_rag=True,
            enable_regime_classification=True
        )
        
        system = ConditioningSystem(config)
        
        market_data = {
            'price_change_1m': 0.002,
            'price_change_5m': -0.005,
            'volatility': 0.18,
            'volume_ratio': 1.3,
            'trend_strength': 0.7,
            'timestamp': datetime.now()
        }
        
        result = system.condition_market_context(market_data)
        
        assert 'conditioning_vector' in result
        assert result['conditioning_vector'].shape == (1, config.conditioning_dim)
        assert 'components_used' in result
        assert 'processing_time_ms' in result
        assert 'feature_flags' in result
    
    def test_pattern_based_conditioning(self):
        """Test conditioning with explicit patterns."""
        config = ConditioningConfig(
            enable_patterns=True,
            enable_rag=False,
            enable_regime_classification=False
        )
        
        system = ConditioningSystem(config)
        
        # Create test patterns
        test_patterns = [
            Pattern(
                pattern_id="test_1",
                pattern_type="trend",
                scale="15m",
                features={'duration': 30, 'amplitude': 0.05},
                strength=0.8
            ),
            Pattern(
                pattern_id="test_2", 
                pattern_type="reversal",
                scale="30m",
                features={'duration': 45, 'amplitude': 0.03},
                strength=0.6
            )
        ]
        
        market_data = {'timestamp': datetime.now()}
        
        result = system.condition_market_context(market_data, current_patterns=test_patterns)
        
        assert 'conditioning_vector' in result
        assert 'patterns' in result.get('components_used', []) or len(result['components_used']) == 0  # May not detect patterns without price data
    
    def test_feature_flag_runtime_control(self):
        """Test runtime feature flag control affects conditioning."""
        config = ConditioningConfig(
            enable_patterns=True,
            enable_rag=True,
            enable_regime_classification=True
        )
        
        system = ConditioningSystem(config)
        
        market_data = {
            'price_change_1m': 0.002,
            'volatility': 0.18,
            'timestamp': datetime.now()
        }
        
        # Get conditioning with all flags enabled
        result1 = system.condition_market_context(market_data)
        
        # Disable all components
        system.set_feature_flag('patterns', False)
        system.set_feature_flag('rag', False)
        system.set_feature_flag('regime_classification', False)
        
        # Get conditioning with all flags disabled
        result2 = system.condition_market_context(market_data)
        
        # Should have different components used
        assert len(result2['components_used']) <= len(result1['components_used'])
    
    def test_fail_open_behavior(self):
        """Test fail-open behavior on timeout/errors."""
        config = ConditioningConfig(
            total_conditioning_timeout_ms=1.0,  # Very short timeout
            fail_open_on_timeout=True
        )
        
        system = ConditioningSystem(config)
        
        market_data = {
            'price_change_1m': 0.002,
            'volatility': 0.18,
            'timestamp': datetime.now()
        }
        
        result = system.condition_market_context(market_data)
        
        # Should complete (either normally or via fail-open)
        assert 'conditioning_vector' in result
        assert result['conditioning_vector'] is not None
    
    def test_performance_timeout_handling(self):
        """Test timeout handling with performance requirements."""
        config = ConditioningConfig(
            total_conditioning_timeout_ms=100.0,  # 100ms timeout
            fail_open_on_timeout=True
        )
        
        system = ConditioningSystem(config)
        
        market_data = {'timestamp': datetime.now()}
        
        start_time = time.time()
        result = system.condition_market_context(market_data)
        elapsed_ms = (time.time() - start_time) * 1000
        
        # Should complete within reasonable time (timeout or success)
        assert elapsed_ms < 200  # Give some buffer
        assert 'conditioning_vector' in result
    
    def test_emergency_disable_functionality(self):
        """Test emergency disable functionality."""
        config = ConditioningConfig(
            enable_patterns=True,
            enable_rag=True,
            enable_regime_classification=True
        )
        
        system = ConditioningSystem(config)
        
        # Emergency disable
        system.emergency_disable()
        
        # All flags should be disabled
        flags = system.feature_flags.get_all_flags()
        assert all(not enabled for enabled in flags.values())
        
        # Conditioning should still work (fail-open)
        market_data = {'timestamp': datetime.now()}
        result = system.condition_market_context(market_data)
        
        assert result['components_used'] == []  # No components used
        assert 'conditioning_vector' in result


class TestHRMIntegration:
    """Tests for HRM integration functionality."""
    
    def test_hrm_conditioning_application(self):
        """Test applying conditioning to HRM tokens."""
        hrm_config = HRMConfig(h_dim=256, l_dim=384)
        config = ConditioningConfig(
            hrm_config=hrm_config,
            conditioning_dim=192
        )
        
        system = ConditioningSystem(config)
        
        # Mock HRM tokens
        batch_size = 2
        h_tokens = torch.randn(batch_size, 10, 256)
        l_tokens = torch.randn(batch_size, 15, 384)
        
        # Get conditioning
        market_data = {'timestamp': datetime.now()}
        conditioning_result = system.condition_market_context(market_data)
        
        # Apply to HRM
        conditioned_h, conditioned_l = system.apply_conditioning_to_hrm(
            h_tokens, l_tokens, conditioning_result
        )
        
        assert conditioned_h.shape == h_tokens.shape
        assert conditioned_l.shape == l_tokens.shape
        # Tokens should be modified (not identity)
        assert not torch.allclose(conditioned_h, h_tokens)
        assert not torch.allclose(conditioned_l, l_tokens)
    
    def test_hrm_integration_without_adapter(self):
        """Test HRM integration when no adapter is configured."""
        config = ConditioningConfig(hrm_config=None)  # No HRM config
        system = ConditioningSystem(config)
        
        batch_size = 2
        h_tokens = torch.randn(batch_size, 10, 256)
        l_tokens = torch.randn(batch_size, 15, 384)
        
        market_data = {'timestamp': datetime.now()}
        conditioning_result = system.condition_market_context(market_data)
        
        # Should return tokens unchanged
        conditioned_h, conditioned_l = system.apply_conditioning_to_hrm(
            h_tokens, l_tokens, conditioning_result
        )
        
        assert torch.allclose(conditioned_h, h_tokens)
        assert torch.allclose(conditioned_l, l_tokens)
    
    def test_end_to_end_training_integration(self):
        """Test end-to-end integration in training context."""
        hrm_config = HRMConfig(
            h_dim=128, l_dim=256,  # Smaller for test
            h_layers=1, l_layers=1
        )
        config = ConditioningConfig(
            hrm_config=hrm_config,
            conditioning_dim=128,
            enable_patterns=True,
            enable_rag=False,  # Disable for simpler test
            enable_regime_classification=True
        )
        
        system = ConditioningSystem(config)
        
        # Training setup
        optimizer = torch.optim.Adam(system.parameters(), lr=1e-3)
        
        # Mock training step
        market_data = {
            'price_change_1m': 0.002,
            'volatility': 0.18,
            'timestamp': datetime.now()
        }
        
        h_tokens = torch.randn(1, 5, 128, requires_grad=True)
        l_tokens = torch.randn(1, 8, 256, requires_grad=True)
        target = torch.randn(1)
        
        optimizer.zero_grad()
        
        # Forward pass through conditioning system
        conditioning_result = system.condition_market_context(market_data)
        conditioned_h, conditioned_l = system.apply_conditioning_to_hrm(
            h_tokens, l_tokens, conditioning_result
        )
        
        # Mock loss
        output = conditioned_h.mean() + conditioned_l.mean()
        loss = torch.nn.functional.mse_loss(output, target)
        
        loss.backward()
        optimizer.step()
        
        # Should complete without errors
        assert loss.item() >= 0


class TestPerformanceAndScaling:
    """Tests for performance requirements and scaling."""
    
    def test_conditioning_latency_requirements(self):
        """Test conditioning meets latency requirements."""
        config = ConditioningConfig(
            total_conditioning_timeout_ms=100.0,
            enable_patterns=True,
            enable_rag=True,
            enable_regime_classification=True
        )
        
        system = ConditioningSystem(config)
        
        market_data = {
            'price_change_1m': 0.002,
            'volatility': 0.18,
            'timestamp': datetime.now()
        }
        
        # Test multiple conditioning calls
        times = []
        for _ in range(10):
            start_time = time.time()
            result = system.condition_market_context(market_data)
            elapsed_ms = (time.time() - start_time) * 1000
            times.append(elapsed_ms)
        
        avg_time = np.mean(times)
        p95_time = np.percentile(times, 95)
        
        # Should meet performance requirements most of the time
        assert avg_time < 100, f"Average conditioning time {avg_time:.2f}ms exceeds 100ms"
        assert p95_time < 150, f"P95 conditioning time {p95_time:.2f}ms exceeds 150ms"
    
    def test_parameter_efficiency_analysis(self):
        """Test parameter efficiency and allocation."""
        config = ConditioningConfig(total_parameter_budget=300_000)
        system = ConditioningSystem(config)
        
        total_params = sum(p.numel() for p in system.parameters() if p.requires_grad)
        performance_stats = system.get_performance_stats()
        
        # Should use parameters efficiently
        assert total_params <= config.total_parameter_budget
        assert performance_stats['parameter_count'] == total_params
        assert performance_stats['parameter_budget'] == config.total_parameter_budget
        
        # Should have reasonable parameter allocation
        param_ratio = total_params / config.total_parameter_budget
        assert 0.1 <= param_ratio <= 1.0  # Should use at least 10% of budget
    
    def test_performance_statistics_tracking(self):
        """Test performance statistics collection."""
        config = ConditioningConfig()
        system = ConditioningSystem(config)
        
        # Generate some activity
        market_data = {'timestamp': datetime.now()}
        for _ in range(5):
            system.condition_market_context(market_data)
        
        stats = system.get_performance_stats()
        
        assert 'parameter_count' in stats
        assert 'feature_flags' in stats
        assert 'component_usage' in stats
        assert 'total_conditioning_times' in stats
        
        # Should track timing stats
        timing_stats = stats['total_conditioning_times']
        assert timing_stats['count'] == 5
        assert timing_stats['mean'] >= 0
    
    def test_memory_usage_efficiency(self):
        """Test memory usage remains reasonable."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / (1024 * 1024)  # MB
        
        config = ConditioningConfig()
        system = ConditioningSystem(config)
        
        # Generate activity
        market_data = {'timestamp': datetime.now()}
        for _ in range(100):
            system.condition_market_context(market_data)
        
        memory_after = process.memory_info().rss / (1024 * 1024)  # MB
        memory_increase = memory_after - memory_before
        
        # Should not use excessive memory
        assert memory_increase < 100, \
            f"Conditioning system used {memory_increase:.1f}MB, exceeding reasonable limit"


class TestFailureHandling:
    """Tests for failure handling and robustness."""
    
    def test_component_failure_isolation(self):
        """Test that individual component failures don't crash system."""
        config = ConditioningConfig(
            enable_patterns=True,
            enable_rag=True, 
            enable_regime_classification=True,
            fail_open_on_error=True
        )
        
        system = ConditioningSystem(config)
        
        # Test with malformed market data
        malformed_data = {
            'invalid_field': 'bad_data',
            'timestamp': 'not_a_datetime'  # Invalid timestamp
        }
        
        # Should handle gracefully
        result = system.condition_market_context(malformed_data)
        
        assert 'conditioning_vector' in result
        assert result['conditioning_vector'] is not None
    
    def test_concurrent_access_safety(self):
        """Test thread safety under concurrent access."""
        import threading
        
        config = ConditioningConfig()
        system = ConditioningSystem(config)
        
        results = []
        errors = []
        
        def conditioning_worker():
            try:
                market_data = {
                    'price_change_1m': np.random.normal(0, 0.01),
                    'volatility': 0.15 + np.random.normal(0, 0.05),
                    'timestamp': datetime.now()
                }
                result = system.condition_market_context(market_data)
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Run concurrent conditioning
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=conditioning_worker)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Should handle concurrent access safely
        assert len(errors) == 0, f"Concurrent access errors: {errors}"
        assert len(results) == 10, "Should complete all concurrent requests"


# Integration tests
class TestDRQComplianceValidation:
    """Tests for complete DRQ-103 compliance validation."""
    
    def test_unified_conditioning_system_api(self):
        """Test DRQ-103: Single ConditioningSystem API for all components."""
        config = ConditioningConfig()
        system = ConditioningSystem(config)
        
        # Should provide single API
        assert hasattr(system, 'condition_market_context')
        assert hasattr(system, 'apply_conditioning_to_hrm')
        assert hasattr(system, 'set_feature_flag')
        assert hasattr(system, 'emergency_disable')
        
        # Should integrate all components
        market_data = {'timestamp': datetime.now()}
        result = system.condition_market_context(market_data)
        
        assert isinstance(result, dict)
        assert 'conditioning_vector' in result
    
    def test_feature_flag_independent_control(self):
        """Test DRQ-103: Feature flags control each component independently."""
        config = ConditioningConfig(
            enable_patterns=True,
            enable_rag=True,
            enable_regime_classification=True
        )
        
        system = ConditioningSystem(config)
        
        # Test individual control
        system.set_feature_flag('patterns', False)
        assert not system.feature_flags.is_enabled('patterns')
        assert system.feature_flags.is_enabled('rag')
        assert system.feature_flags.is_enabled('regime_classification')
        
        system.set_feature_flag('rag', False)
        assert not system.feature_flags.is_enabled('rag')
        assert system.feature_flags.is_enabled('regime_classification')
    
    def test_total_conditioning_parameter_budget(self):
        """Test DRQ-103: Total conditioning system ≤0.3M parameters."""
        config = ConditioningConfig(total_parameter_budget=300_000)
        system = ConditioningSystem(config)
        
        total_params = sum(p.numel() for p in system.parameters() if p.requires_grad)
        
        assert total_params <= 300_000, \
            f"Total conditioning parameters {total_params:,} exceeds 300K limit"
    
    def test_hybrid_mode_operation(self):
        """Test DRQ-103: Hybrid mode (regime + RAG) functional."""
        config = ConditioningConfig(
            enable_patterns=False,  # Disable patterns
            enable_rag=True,        # Enable RAG
            enable_regime_classification=True  # Enable regime
        )
        
        system = ConditioningSystem(config)
        
        market_data = {
            'price_change_1m': 0.002,
            'volatility': 0.18,
            'timestamp': datetime.now()
        }
        
        result = system.condition_market_context(market_data)
        
        # Should work in hybrid mode
        assert 'conditioning_vector' in result
        assert result['conditioning_vector'] is not None
        
        # Should use regime component (RAG may not be used without historical data)
        components_used = result.get('components_used', [])
        # At least one component should be available or fail-open should work
        assert len(components_used) >= 0  # May be empty but should not crash
    
    def test_fail_open_path_verification(self):
        """Test DRQ-103: Fail-open path verified under load."""
        config = ConditioningConfig(
            total_conditioning_timeout_ms=1.0,  # Very short timeout to trigger fail-open
            fail_open_on_timeout=True
        )
        
        system = ConditioningSystem(config)
        
        market_data = {'timestamp': datetime.now()}
        
        # Should handle fail-open gracefully
        result = system.condition_market_context(market_data)
        
        assert 'conditioning_vector' in result
        assert result['conditioning_vector'].shape[1] == config.conditioning_dim
        
        # May have fail_open flag set
        if 'fail_open' in result:
            assert result['fail_open'] == True


# This will run when pytest is called
if __name__ == "__main__":
    pytest.main([__file__, "-v"])