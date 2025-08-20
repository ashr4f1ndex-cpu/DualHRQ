def test_param_budget():
    import importlib
    HRMNet = getattr(importlib.import_module("src.options.hrm_net"), "HRMNet")
    p = sum(t.numel() for t in HRMNet().parameters() if getattr(t,"requires_grad",False))
    assert 26_500_000 <= p <= 27_500_000