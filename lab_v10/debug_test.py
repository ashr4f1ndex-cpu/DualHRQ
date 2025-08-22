"""Debug test to isolate the import and initialization issues."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("Testing imports...")

try:
    from main_orchestrator import DualHRQOrchestrator, DualHRQConfig
    print("✓ Main orchestrator imported successfully")
except Exception as e:
    print(f"❌ Main orchestrator import failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

try:
    # Test simple config
    hrm_config = {
        'h_dim': 384,
        'l_dim': 192,
        'num_h_layers': 8,
        'num_l_layers': 6,
        'num_heads': 6,
        'dropout': 0.1,
        'max_sequence_length': 128,
        'deq_threshold': 1e-3,
        'max_deq_iterations': 25
    }
    
    config = DualHRQConfig(
        hrm_config=hrm_config,
        start_date="2023-01-01",
        end_date="2023-12-31",
        initial_capital=1_000_000,
        number_of_trials=10,
        deterministic_seed=42,
        enable_mlops_tracking=False
    )
    print("✓ DualHRQ config created successfully")
    
    orchestrator = DualHRQOrchestrator(config)
    print("✓ DualHRQ orchestrator created successfully")
    
    # Test system setup
    setup_success = orchestrator.setup_system()
    print(f"System setup result: {setup_success}")
    
    if setup_success:
        print("✓ System setup completed successfully!")
        
        # Test data loading
        data_success = orchestrator.load_data()
        print(f"Data loading result: {data_success}")
        
        if data_success:
            print("✓ Data loading completed successfully!")
            print(f"Universe size: {len(orchestrator.universe)}")
            print(f"Market data shape: {orchestrator.market_data.shape}")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()

print("Debug test completed.")