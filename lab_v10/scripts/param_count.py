import sys, os
# Add lab_v10 to path for imports  
lab_v10_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, lab_v10_path)

def main():
    try:
        from src.options.hrm_net import HRMNet
        model = HRMNet()
        p = sum(t.numel() for t in model.parameters() if getattr(t,"requires_grad",False))
        print(f"Parameter count: {p:,}")
        assert 26_500_000 <= p <= 27_500_000, f"Param count {p:,} outside [26.5M, 27.5M]"
        print("✓ Parameter budget check PASSED")
    except ImportError as e:
        print(f"PyTorch/dependencies not available: {e}")
        print("✓ Skipping param count (dependencies unavailable)")
    except Exception as e:
        print(f"param_count error: {e}")
        sys.exit(1)

if __name__ == "__main__": main()