"""
test_param_gate.py
==================

TDD tests for DRQ-001: Parameter Counter + CI Gate
These tests MUST be written first and will initially FAIL.
Implementation should make these tests pass.

CRITICAL: This is a P0 blocker for all other tickets.
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import sys
import os

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from tools.param_count import count_hrm_parameters, verify_budget_compliance
    from lab_v10.src.options.hrm_net import HRMConfig, HRMNet
except ImportError:
    # These will fail initially - that's expected in TDD
    pass


class TestParameterGate:
    """Tests for parameter counting and budget enforcement."""
    
    def test_param_count_current_config(self):
        """Test that current config shows 46.68M parameters (broken state)."""
        # This test documents the current broken state
        config_path = "config/default_hrm27m.yaml"
        
        # This will fail initially until param_count.py is implemented
        param_count = count_hrm_parameters(config_path)
        
        # Current broken state - 46.68M parameters (74% over budget)
        expected_range = (46_000_000, 47_000_000)
        assert expected_range[0] <= param_count <= expected_range[1], \
            f"Current config should show ~46.68M params, got {param_count}"
        
        # Verify it's over budget (should fail compliance)
        is_compliant = verify_budget_compliance(param_count)
        assert not is_compliant, "Current config should fail budget compliance"
    
    def test_param_count_target_config(self):
        """Test that target config shows ~26.7M parameters (target state)."""
        # This tests the target configuration we need to achieve
        target_config = HRMConfig(
            h_layers=4, h_dim=384, h_heads=8, h_ffn_mult=3.0, h_dropout=0.1,
            l_layers=4, l_dim=512, l_heads=8, l_ffn_mult=3.0, l_dropout=0.1,
            segments_N=3, l_inner_T=8, act_enable=True, 
            act_max_segments=5, ponder_cost=0.01, use_cross_attn=False
        )
        
        # Create model to count parameters
        model = HRMNet(target_config)
        param_count = sum(p.numel() for p in model.parameters())
        
        # Should be in target range: 26.5M ≤ params ≤ 27.5M
        assert 26_500_000 <= param_count <= 27_500_000, \
            f"Target config should be 26.5M-27.5M params, got {param_count}"
        
        # Should pass compliance
        is_compliant = verify_budget_compliance(param_count)
        assert is_compliant, f"Target config should pass compliance, got {param_count}"
    
    def test_ci_gate_blocks_over_budget(self):
        """Test CI gate blocks models over 27.5M parameter budget."""
        # Test that CI gate properly blocks over-budget models
        over_budget_count = 28_000_000  # Over 27.5M limit
        
        # Should fail compliance check
        is_compliant = verify_budget_compliance(over_budget_count, max_params=27_500_000)
        assert not is_compliant, "Over-budget model should fail CI gate"
        
        # Under budget should pass
        under_budget_count = 26_800_000  # Within 27.5M limit
        is_compliant = verify_budget_compliance(under_budget_count, max_params=27_500_000)
        assert is_compliant, "Under-budget model should pass CI gate"
    
    def test_parameter_breakdown_by_component(self):
        """Test parameter counting breaks down by component (H, L, heads, etc)."""
        config = HRMConfig(
            h_layers=4, h_dim=384, h_heads=8, h_ffn_mult=3.0, h_dropout=0.1,
            l_layers=4, l_dim=512, l_heads=8, l_ffn_mult=3.0, l_dropout=0.1,
            segments_N=3, l_inner_T=8, act_enable=True,
            act_max_segments=5, ponder_cost=0.01, use_cross_attn=False
        )
        
        # This will fail initially - param_count.py needs to provide breakdown
        breakdown = count_hrm_parameters(config, return_breakdown=True)
        
        # Should return breakdown by component
        required_components = ['h_module', 'l_module', 'heads', 'total']
        assert all(comp in breakdown for comp in required_components), \
            f"Breakdown should contain {required_components}, got {breakdown.keys()}"
        
        # Components should sum to total
        component_sum = breakdown['h_module'] + breakdown['l_module'] + breakdown['heads']
        assert abs(component_sum - breakdown['total']) < 10, \
            f"Components should sum to total: {component_sum} vs {breakdown['total']}"
        
        # H-module should be smaller than L-module (384 vs 512 dims)
        assert breakdown['h_module'] < breakdown['l_module'], \
            "H-module should have fewer parameters than L-module"
    
    def test_param_count_accuracy(self):
        """Test parameter counting matches PyTorch's built-in counting."""
        config = HRMConfig(
            h_layers=2, h_dim=128, h_heads=4, h_ffn_mult=2.0, h_dropout=0.1,
            l_layers=2, l_dim=256, l_heads=4, l_ffn_mult=2.0, l_dropout=0.1,
            segments_N=2, l_inner_T=4, act_enable=False,
            act_max_segments=3, ponder_cost=0.01, use_cross_attn=False
        )
        
        # Count using PyTorch built-in
        model = HRMNet(config)
        pytorch_count = sum(p.numel() for p in model.parameters())
        
        # Count using our tool
        our_count = count_hrm_parameters(config)
        
        # Should match exactly
        assert pytorch_count == our_count, \
            f"Parameter counts should match: PyTorch={pytorch_count}, Ours={our_count}"
    
    def test_budget_enforcement_edge_cases(self):
        """Test budget enforcement handles edge cases correctly."""
        max_params = 27_500_000
        
        # Test exact boundary cases
        assert verify_budget_compliance(27_500_000, max_params), \
            "Exactly at limit should pass"
        assert not verify_budget_compliance(27_500_001, max_params), \
            "One over limit should fail"
        
        # Test minimum boundary
        min_params = 26_500_000
        assert verify_budget_compliance(26_500_000, max_params, min_params), \
            "At minimum should pass"
        assert not verify_budget_compliance(26_499_999, max_params, min_params), \
            "Below minimum should fail"
    
    @pytest.mark.performance
    def test_param_counting_performance(self):
        """Test parameter counting completes within reasonable time."""
        import time
        
        config = HRMConfig(
            h_layers=4, h_dim=384, h_heads=8, h_ffn_mult=3.0, h_dropout=0.1,
            l_layers=4, l_dim=512, l_heads=8, l_ffn_mult=3.0, l_dropout=0.1,
            segments_N=3, l_inner_T=8, act_enable=True,
            act_max_segments=5, ponder_cost=0.01, use_cross_attn=False
        )
        
        start_time = time.time()
        param_count = count_hrm_parameters(config)
        elapsed_time = time.time() - start_time
        
        # Should complete within 30 seconds (CI requirement)
        assert elapsed_time < 30, \
            f"Parameter counting should complete in <30s, took {elapsed_time:.2f}s"
        
        # Should return valid count
        assert param_count > 0, "Should return positive parameter count"


class TestCIIntegration:
    """Tests for CI integration and automation."""
    
    def test_ci_script_exists(self):
        """Test that CI script exists and is executable."""
        # This will fail initially until we create the CI integration
        ci_script_path = Path("tools/param_count.py")
        assert ci_script_path.exists(), "Parameter counting script should exist"
        
        # Should be executable
        assert os.access(ci_script_path, os.X_OK), "Script should be executable"
    
    def test_ci_exit_codes(self):
        """Test CI script returns correct exit codes."""
        # Test that the script returns proper exit codes for CI
        
        # This will fail initially - need to implement exit codes
        import subprocess
        
        # Should return 0 for compliant model
        result = subprocess.run([
            "python", "tools/param_count.py", 
            "--config", "config/target_hrm27m.yaml"
        ], capture_output=True)
        
        assert result.returncode == 0, \
            "Compliant model should return exit code 0"
        
        # Should return 1 for non-compliant model
        result = subprocess.run([
            "python", "tools/param_count.py",
            "--config", "config/default_hrm27m.yaml",
            "--strict"
        ], capture_output=True)
        
        assert result.returncode == 1, \
            "Over-budget model should return exit code 1"
    
    def test_ci_output_format(self):
        """Test CI script output format for parsing."""
        # CI needs parseable output format
        import subprocess
        
        result = subprocess.run([
            "python", "tools/param_count.py",
            "--config", "config/target_hrm27m.yaml", 
            "--format", "json"
        ], capture_output=True, text=True)
        
        # Should return valid JSON
        import json
        try:
            output = json.loads(result.stdout)
        except json.JSONDecodeError:
            pytest.fail("CI script should return valid JSON")
        
        # Should contain required fields
        required_fields = ['total_params', 'budget_compliant', 'breakdown']
        assert all(field in output for field in required_fields), \
            f"JSON output should contain {required_fields}"


# This will run when pytest is called and should initially FAIL
# Implementation of tools/param_count.py should make these tests pass
if __name__ == "__main__":
    pytest.main([__file__, "-v"])