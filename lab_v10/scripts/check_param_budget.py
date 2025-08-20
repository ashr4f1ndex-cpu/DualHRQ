from importlib import import_module
HRMNet = getattr(import_module("src.options.hrm_net"), "HRMNet")
p = sum(t.numel() for t in HRMNet().parameters() if getattr(t,"requires_grad",False))
print(p); assert 26_500_000 <= p <= 27_500_000