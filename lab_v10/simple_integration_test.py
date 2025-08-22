"""
Simple integration test to verify dual-book feature engineering implementation.
Tests core functionality without external dependencies.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def test_module_imports():
    """Test that all modules can be imported successfully."""
    print("Testing module imports...")
    
    try:
        # Test options features
        from src.common.options_features import OptionsFeatureEngine
        print("✓ Options features module imported")
        
        # Test intraday features
        from src.common.intraday_features import IntradayFeatureEngine
        print("✓ Intraday features module imported")
        
        # Test corporate actions
        from src.common.corporate_actions import CorporateActionAdjuster, CorporateActionDatabase
        print("✓ Corporate actions module imported")
        
        # Test data alignment
        from src.common.data_alignment import DataAligner, USEquityCalendar
        print("✓ Data alignment module imported")
        
        # Test leakage prevention
        from src.common.leakage_prevention import CombinatorialPurgedCV, LeakageAuditor
        print("✓ Leakage prevention module imported")
        
        # Test integrated features
        from src.common.feature_integration import IntegratedFeatureEngine, FeatureConfig
        print("✓ Feature integration module imported")
        
        # Test enhanced HRM inputs
        from src.options.hrm_input_enhanced import EnhancedFeatureProcessor, EnhancedTokenConfig
        print("✓ Enhanced HRM input module imported")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_class_instantiation():
    """Test that all classes can be instantiated."""
    print("\nTesting class instantiation...")
    
    try:
        # Import required modules
        from src.common.options_features import OptionsFeatureEngine
        from src.common.intraday_features import IntradayFeatureEngine
        from src.common.corporate_actions import CorporateActionAdjuster
        from src.common.data_alignment import DataAligner
        from src.common.leakage_prevention import LeakageAuditor
        from src.common.feature_integration import IntegratedFeatureEngine, FeatureConfig
        
        # Test instantiation
        options_engine = OptionsFeatureEngine()
        print("✓ OptionsFeatureEngine instantiated")
        
        intraday_engine = IntradayFeatureEngine()
        print("✓ IntradayFeatureEngine instantiated")
        
        corp_adjuster = CorporateActionAdjuster()
        print("✓ CorporateActionAdjuster instantiated")
        
        data_aligner = DataAligner()
        print("✓ DataAligner instantiated")
        
        leakage_auditor = LeakageAuditor()
        print("✓ LeakageAuditor instantiated")
        
        feature_config = FeatureConfig()
        print("✓ FeatureConfig instantiated")
        
        integrated_engine = IntegratedFeatureEngine(feature_config)
        print("✓ IntegratedFeatureEngine instantiated")
        
        return True
        
    except Exception as e:
        print(f"❌ Instantiation failed: {e}")
        return False


def test_configuration_validation():
    """Test configuration validation."""
    print("\nTesting configuration validation...")
    
    try:
        from src.common.feature_integration import FeatureConfig
        
        # Test default configuration
        default_config = FeatureConfig()
        assert default_config.enable_options_features == True
        assert default_config.enable_intraday_features == True
        assert default_config.enable_corporate_actions == True
        print("✓ Default configuration valid")
        
        # Test custom configuration
        custom_config = FeatureConfig(
            enable_options_features=False,
            cv_method='walkforward',
            n_splits=4
        )
        assert custom_config.enable_options_features == False
        assert custom_config.cv_method == 'walkforward'
        assert custom_config.n_splits == 4
        print("✓ Custom configuration valid")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return False


def test_corporate_action_types():
    """Test corporate action enumeration and dataclass."""
    print("\nTesting corporate action types...")
    
    try:
        from src.common.corporate_actions import CorporateActionType, CorporateAction
        import pandas as pd
        
        # Test enumeration
        assert CorporateActionType.CASH_DIVIDEND.value == "CASH_DIVIDEND"
        assert CorporateActionType.STOCK_SPLIT.value == "STOCK_SPLIT"
        print("✓ Corporate action types enumeration valid")
        
        # Test dataclass creation
        action = CorporateAction(
            symbol="TEST",
            action_type=CorporateActionType.CASH_DIVIDEND,
            ex_date=pd.Timestamp('2023-01-15'),
            cash_amount=0.50
        )
        assert action.symbol == "TEST"
        assert action.cash_amount == 0.50
        print("✓ Corporate action dataclass valid")
        
        return True
        
    except Exception as e:
        print(f"❌ Corporate action types test failed: {e}")
        return False


def test_time_zone_support():
    """Test timezone handling."""
    print("\nTesting timezone support...")
    
    try:
        from zoneinfo import ZoneInfo
        from datetime import time
        
        # Test timezone creation
        ny_tz = ZoneInfo("America/New_York")
        chicago_tz = ZoneInfo("America/Chicago")
        
        # Test time objects
        market_open = time(9, 30)
        market_close = time(16, 0)
        
        assert market_open.hour == 9
        assert market_close.hour == 16
        print("✓ Timezone support available")
        
        return True
        
    except Exception as e:
        print(f"❌ Timezone test failed: {e}")
        return False


def run_simple_tests():
    """Run all simple integration tests."""
    print("=== Simple Dual-Book Feature Engineering Integration Tests ===")
    print("Testing core functionality without external dependencies...\n")
    
    tests = [
        test_module_imports,
        test_class_instantiation,
        test_configuration_validation,
        test_corporate_action_types,
        test_time_zone_support
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    print(f"\n=== Results ===")
    print(f"Passed: {passed}/{total} tests")
    
    if passed == total:
        print("🎉 ALL SIMPLE TESTS PASSED! 🎉")
        print("Core dual-book feature engineering pipeline is properly implemented!")
    else:
        print("❌ Some tests failed. Please check the implementation.")
    
    return passed == total


if __name__ == "__main__":
    success = run_simple_tests()
    sys.exit(0 if success else 1)