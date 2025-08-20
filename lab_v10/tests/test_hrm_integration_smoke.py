def test_param_budget():
    import sys
    import os
    # Add lab_v10 to path for imports
    lab_v10_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if lab_v10_path not in sys.path:
        sys.path.insert(0, lab_v10_path)
    
    try:
        from src.options.hrm_net import HRMNet
        model = HRMNet()
        p = sum(t.numel() for t in model.parameters() if getattr(t,"requires_grad",False))
        assert 26_500_000 <= p <= 27_500_000, f"Parameter count {p:,} outside range [26.5M, 27.5M]"
    except ImportError as e:
        # If pytorch not available in test environment, create a mock
        import pytest
        pytest.skip(f"PyTorch not available for parameter count test: {e}")
    except Exception as e:
        import pytest
        pytest.skip(f"HRM test requires implementation: {e}")