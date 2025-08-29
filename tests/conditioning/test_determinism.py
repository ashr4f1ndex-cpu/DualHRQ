"""
test_determinism.py
===================

TDD tests for DRQ-003: Deterministic Reproducibility Validation
These tests MUST be written first and will initially FAIL.
Implementation should make these tests pass.

CRITICAL: Same seeds must produce identical outputs.
"""

import pytest
import numpy as np
import torch
import random
import time
from pathlib import Path
import sys
from typing import Dict, Any, Callable

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from tools.determinism import (
        set_all_seeds,
        validate_determinism,
        ReproducibilityValidator
    )
    from lab_v10.src.options.hrm_net import HRMNet, HRMConfig
except ImportError:
    # These will fail initially - that's expected in TDD
    pass


class TestDeterministicReproduction:
    """Tests for deterministic reproduction across runs."""
    
    def test_exact_reproduction_same_seed(self):
        """Test that same seeds produce identical outputs."""
        def test_function(data):
            """Simple test function that should be deterministic."""
            # Use multiple sources of randomness
            torch_result = torch.randn(10)
            numpy_result = np.random.randn(5)
            python_result = [random.random() for _ in range(3)]
            
            return {
                'torch': torch_result,
                'numpy': numpy_result,
                'python': python_result
            }
        
        test_data = {'input': np.random.randn(100, 10)}
        seed = 42
        
        # This will fail initially until set_all_seeds is implemented
        set_all_seeds(seed)
        result1 = test_function(test_data)
        
        set_all_seeds(seed)  # Reset to same seed
        result2 = test_function(test_data)
        
        # Results should be identical
        torch.testing.assert_close(result1['torch'], result2['torch'], rtol=0, atol=0)
        np.testing.assert_array_equal(result1['numpy'], result2['numpy'])
        assert result1['python'] == result2['python'], \
            f"Python random should be identical: {result1['python']} vs {result2['python']}"
    
    def test_different_seeds_different_outputs(self):
        """Test that different seeds produce different outputs."""
        def test_function(data):
            return torch.randn(20).sum().item()
        
        test_data = {'input': 'dummy'}
        
        set_all_seeds(42)
        result1 = test_function(test_data)
        
        set_all_seeds(123)  # Different seed
        result2 = test_function(test_data)
        
        # Results should be different (with very high probability)
        assert result1 != result2, \
            f"Different seeds should produce different results: {result1} == {result2}"
    
    def test_hrm_model_determinism(self):
        """Test that HRM model produces identical results with same seed."""
        config = HRMConfig(
            h_layers=2, h_dim=128, h_heads=4, h_ffn_mult=2.0, h_dropout=0.0,  # No dropout for determinism
            l_layers=2, l_dim=256, l_heads=4, l_ffn_mult=2.0, l_dropout=0.0,
            segments_N=2, l_inner_T=4, act_enable=False,
            act_max_segments=3, ponder_cost=0.01, use_cross_attn=False
        )
        
        # Create test inputs
        batch_size = 2
        h_tokens = torch.randn(batch_size, 10, config.h_dim)
        l_tokens = torch.randn(batch_size, 20, config.l_dim)
        
        # First run
        set_all_seeds(789)
        model1 = HRMNet(config)
        model1.eval()  # Disable dropout
        
        with torch.no_grad():
            (h_out1, l_out1), segments1 = model1(h_tokens, l_tokens)
        
        # Second run with same seed
        set_all_seeds(789)
        model2 = HRMNet(config)
        model2.eval()
        
        with torch.no_grad():
            (h_out2, l_out2), segments2 = model2(h_tokens, l_tokens)
        
        # Outputs should be identical
        torch.testing.assert_close(h_out1, h_out2, rtol=1e-7, atol=1e-7)
        torch.testing.assert_close(l_out1, l_out2, rtol=1e-7, atol=1e-7)
        
        # Segments should be identical
        assert len(segments1) == len(segments2), \
            "Number of segments should be identical"
        
        for i, (seg1, seg2) in enumerate(zip(segments1, segments2)):
            torch.testing.assert_close(seg1[0], seg2[0], rtol=1e-7, atol=1e-7)  # outA
            torch.testing.assert_close(seg1[1], seg2[1], rtol=1e-7, atol=1e-7)  # outB
            torch.testing.assert_close(seg1[2], seg2[2], rtol=1e-7, atol=1e-7)  # q_logits
    
    def test_cuda_determinism(self):
        """Test deterministic behavior on CUDA (if available)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        def cuda_test_function(data):
            x = torch.randn(100, 50).cuda()
            y = torch.mm(x, x.t())
            return y.cpu()
        
        test_data = {'dummy': 'data'}
        
        set_all_seeds(999)
        result1 = cuda_test_function(test_data)
        
        set_all_seeds(999)
        result2 = cuda_test_function(test_data)
        
        # CUDA results should also be deterministic
        torch.testing.assert_close(result1, result2, rtol=1e-6, atol=1e-6)
    
    def test_determinism_with_training(self):
        """Test determinism during model training."""
        config = HRMConfig(
            h_layers=1, h_dim=64, h_heads=2, h_ffn_mult=2.0, h_dropout=0.0,
            l_layers=1, l_dim=128, l_heads=2, l_ffn_mult=2.0, l_dropout=0.0,
            segments_N=1, l_inner_T=2, act_enable=False,
            act_max_segments=2, ponder_cost=0.01, use_cross_attn=False
        )
        
        def train_one_step(seed):
            set_all_seeds(seed)
            model = HRMNet(config)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            # Create dummy data
            h_tokens = torch.randn(1, 5, config.h_dim)
            l_tokens = torch.randn(1, 10, config.l_dim)
            target = torch.randn(1)
            
            # Forward pass
            (h_out, l_out), segments = model(h_tokens, l_tokens)
            loss = torch.nn.functional.mse_loss(segments[-1][0], target)  # Use outA
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            return loss.item(), list(model.parameters())[0].clone().detach()
        
        # Train with same seed twice
        loss1, param1 = train_one_step(555)
        loss2, param2 = train_one_step(555)
        
        # Loss and parameters should be identical after one training step
        assert abs(loss1 - loss2) < 1e-8, \
            f"Training loss should be identical: {loss1} vs {loss2}"
        torch.testing.assert_close(param1, param2, rtol=1e-7, atol=1e-7)
    
    def test_reproducibility_validator(self):
        """Test the ReproducibilityValidator class."""
        def simple_model_function(data):
            return torch.randn(5).sum().item()
        
        test_data = {'test': 'data'}
        
        # This will fail initially until ReproducibilityValidator is implemented
        validator = ReproducibilityValidator(tolerance=1e-8)
        
        validation_results = validator.validate_reproducibility(
            simple_model_function, test_data, n_runs=3
        )
        
        assert validation_results['is_reproducible'], \
            f"Simple function should be reproducible: {validation_results}"
        assert validation_results['n_runs'] == 3, \
            "Should report correct number of runs"
        assert 'consistency_check' in validation_results, \
            "Should include consistency check results"
    
    def test_seed_isolation(self):
        """Test that seed setting doesn't interfere across functions."""
        def function_a(seed):
            set_all_seeds(seed)
            return torch.randn(3).tolist()
        
        def function_b(seed):
            set_all_seeds(seed)
            return np.random.randn(3).tolist()
        
        # Call functions in different order
        result_a1 = function_a(111)
        result_b1 = function_b(222)
        
        result_b2 = function_b(222)  # Same seed as b1
        result_a2 = function_a(111)  # Same seed as a1
        
        # Results should be identical regardless of call order
        assert result_a1 == result_a2, \
            "Function A should be reproducible regardless of other calls"
        assert result_b1 == result_b2, \
            "Function B should be reproducible regardless of other calls"
    
    def test_comprehensive_seed_coverage(self):
        """Test that all randomness sources are controlled."""
        def comprehensive_random_function(data):
            results = {}
            
            # PyTorch randomness
            results['torch_randn'] = torch.randn(2)
            results['torch_randint'] = torch.randint(0, 100, (2,))
            results['torch_rand'] = torch.rand(2)
            
            # NumPy randomness  
            results['numpy_randn'] = np.random.randn(2)
            results['numpy_randint'] = np.random.randint(0, 100, 2)
            results['numpy_choice'] = np.random.choice([1, 2, 3, 4, 5], 2)
            
            # Python built-in randomness
            results['python_random'] = random.random()
            results['python_randint'] = random.randint(1, 100)
            results['python_choice'] = random.choice(['a', 'b', 'c', 'd'])
            
            return results
        
        test_data = {'comprehensive': 'test'}
        
        # Run twice with same seed
        set_all_seeds(777)
        results1 = comprehensive_random_function(test_data)
        
        set_all_seeds(777)
        results2 = comprehensive_random_function(test_data)
        
        # All random sources should be identical
        for key in results1.keys():
            if isinstance(results1[key], torch.Tensor):
                torch.testing.assert_close(results1[key], results2[key], rtol=0, atol=0)
            elif isinstance(results1[key], np.ndarray):
                np.testing.assert_array_equal(results1[key], results2[key])
            else:
                assert results1[key] == results2[key], \
                    f"Mismatch in {key}: {results1[key]} vs {results2[key]}"
    
    def test_determinism_across_python_versions(self):
        """Test determinism considerations across Python versions."""
        # Note: This test documents expected behavior but may vary across Python versions
        set_all_seeds(1234)
        
        # Use operations that should be stable across Python versions
        torch_result = torch.randn(5, generator=torch.Generator().manual_seed(1234))
        numpy_result = np.random.RandomState(1234).randn(5)
        
        # These should be reproducible within the same Python version
        set_all_seeds(1234)
        torch_result2 = torch.randn(5, generator=torch.Generator().manual_seed(1234))
        numpy_result2 = np.random.RandomState(1234).randn(5)
        
        torch.testing.assert_close(torch_result, torch_result2, rtol=0, atol=0)
        np.testing.assert_array_equal(numpy_result, numpy_result2)
    
    def test_performance_impact_of_determinism(self):
        """Test that determinism settings don't severely impact performance."""
        def performance_test_function():
            # CPU operations
            x = torch.randn(1000, 1000)
            y = torch.mm(x, x.t())
            
            # Some randomness
            z = torch.randn(1000)
            
            return y.sum() + z.sum()
        
        # Measure time with determinism
        set_all_seeds(2468)
        start_time = time.time()
        for _ in range(5):
            result_det = performance_test_function()
        det_time = time.time() - start_time
        
        # Time without explicit seed setting (but still deterministic within run)
        start_time = time.time()
        for _ in range(5):
            result_normal = performance_test_function()
        normal_time = time.time() - start_time
        
        # Determinism shouldn't cause major performance regression (>50% slowdown)
        slowdown_factor = det_time / normal_time if normal_time > 0 else 1
        assert slowdown_factor < 2.0, \
            f"Determinism causing excessive slowdown: {slowdown_factor:.2f}x"


class TestSeedManagement:
    """Tests for seed management functionality."""
    
    def test_seed_state_preservation(self):
        """Test that we can save and restore random state."""
        # Set initial state
        set_all_seeds(999)
        
        # Generate some random numbers to change state
        torch.randn(10)
        np.random.randn(10)
        random.random()
        
        # Save state (this functionality may need to be implemented)
        # For now, just test that we can reset reliably
        checkpoint_value_torch = torch.randn(1).item()
        checkpoint_value_numpy = np.random.randn()
        checkpoint_value_python = random.random()
        
        # Reset to same seed
        set_all_seeds(999)
        
        # Skip the same number of random generations
        torch.randn(10)
        np.random.randn(10)
        random.random()
        
        # Should get same values as checkpoint
        new_value_torch = torch.randn(1).item()
        new_value_numpy = np.random.randn()
        new_value_python = random.random()
        
        assert abs(checkpoint_value_torch - new_value_torch) < 1e-8, \
            "Torch state should be resettable"
        assert abs(checkpoint_value_numpy - new_value_numpy) < 1e-8, \
            "NumPy state should be resettable"
        assert abs(checkpoint_value_python - new_value_python) < 1e-8, \
            "Python state should be resettable"


# This will run when pytest is called and should initially FAIL
# Implementation should make these tests pass
if __name__ == "__main__":
    pytest.main([__file__, "-v"])