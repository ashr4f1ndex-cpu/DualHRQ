import importlib, sys
def main():
    try:
        m = importlib.import_module("src.options.hrm_net")
        HRMNet = getattr(m, "HRMNet")
        model = HRMNet()
        p = sum(t.numel() for t in model.parameters() if getattr(t,"requires_grad",False))
        print(p); assert 26_500_000 <= p <= 27_500_000
    except Exception as e:
        print(f"param_count error: {e}"); sys.exit(1)
if __name__ == "__main__": main()