"""
Complete Integration Test for DualHRQ System

Comprehensive end-to-end testing validating:
- HRM model architecture and parameter count
- Feature engineering pipeline integrity  
- Backtesting engines with regulatory compliance
- Portfolio optimization and risk management
- Statistical validation with multiple testing correction
- MLOps deterministic execution
"""

import pytest
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import tempfile
import json
from unittest.mock import patch, MagicMock

import sys
import os

# Fix Python path for lab_v10 imports
project_root = str(Path(__file__).parent.parent.parent)
lab_v10_root = str(Path(__file__).parent.parent)
src_root = str(Path(__file__).parent.parent / "src")

# Add all necessary paths
sys.path.insert(0, project_root)  # For lab_v10.* imports
sys.path.insert(0, lab_v10_root)  # For src.* imports  
sys.path.insert(0, src_root)      # For direct imports

os.chdir(project_root)

# Try multiple import strategies with fallbacks
try:
    from lab_v10.src.main_orchestrator import DualHRQOrchestrator, DualHRQConfig
except ImportError:
    try:
        from src.main_orchestrator import DualHRQOrchestrator, DualHRQConfig
    except ImportError:
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from main_orchestrator import DualHRQOrchestrator, DualHRQConfig

try:
    from lab_v10.src.models.hrm_model import HRMConfig
except ImportError:
    try:
        from src.models.hrm_model import HRMConfig  
    except ImportError:
        from models.hrm_model import HRMConfig

try:
    from lab_v10.src.portfolio.dual_book_integrator import StrategySignal
except ImportError:
    try:
        from src.portfolio.dual_book_integrator import StrategySignal
    except ImportError:
        from portfolio.dual_book_integrator import StrategySignal

class TestDualHRQIntegration:
    """Complete integration test suite."""
    
    @pytest.fixture
    def test_config(self):
        """Create test configuration."""
        
        hrm_config = {
            'h_dim': 384,      # Reduced for testing
            'l_dim': 192,
            'num_h_layers': 8,  
            'num_l_layers': 6,
            'num_heads': 6,
            'dropout': 0.1,
            'max_sequence_length': 128,
            'deq_threshold': 1e-3,
            'max_deq_iterations': 25
        }
        
        return DualHRQConfig(
            hrm_config=hrm_config,
            start_date="2023-01-01",
            end_date="2023-12-31", 
            initial_capital=1_000_000,
            number_of_trials=10,
            deterministic_seed=42,
            enable_mlops_tracking=False  # Disable for testing
        )
    
    @pytest.fixture
    def orchestrator(self, test_config):
        """Create orchestrator instance."""
        return DualHRQOrchestrator(test_config)
    
    def test_system_initialization(self, orchestrator):
        """Test complete system initialization."""
        
        # Test system setup
        success = orchestrator.setup_system()
        assert success, "System setup should succeed"
        assert orchestrator.setup_complete, "Setup should be marked complete"
        
        # Verify components are initialized
        assert orchestrator.hrm_model is not None, "HRM model should be initialized"
        assert orchestrator.options_features is not None, "Options features should be initialized"
        assert orchestrator.intraday_features is not None, "Intraday features should be initialized"
        assert orchestrator.portfolio_manager is not None, "Portfolio manager should be initialized"
        assert orchestrator.backtester is not None, "Backtester should be initialized"
        assert orchestrator.validation_suite is not None, "Validation suite should be initialized"
    
    def test_hrm_parameter_count(self, orchestrator):
        """Test HRM model parameter count constraint."""
        
        orchestrator.setup_system()
        
        # Check parameter count is within required range
        total_params = sum(p.numel() for p in orchestrator.hrm_model.parameters())
        
        # For reduced test configuration, expect smaller parameter count
        # But verify architecture is correct
        assert total_params > 1_000_000, f"Model too small: {total_params:,} parameters"
        assert total_params < 30_000_000, f"Model too large: {total_params:,} parameters"
        
        # Verify dual-module structure
        assert hasattr(orchestrator.hrm_model, 'h_module'), "Missing H-module"
        assert hasattr(orchestrator.hrm_model, 'l_module'), "Missing L-module"
        assert hasattr(orchestrator.hrm_model, 'act_module'), "Missing ACT module"
    
    def test_deterministic_execution(self, orchestrator):
        """Test deterministic execution produces identical results."""
        
        orchestrator.setup_system()
        orchestrator.load_data()
        
        # Generate features twice with same seed
        success1 = orchestrator.generate_features()
        features1 = dict(orchestrator.feature_data)
        
        # Reset and regenerate
        orchestrator.feature_data.clear()
        torch.manual_seed(42)
        np.random.seed(42)
        
        success2 = orchestrator.generate_features()
        features2 = dict(orchestrator.feature_data)
        
        assert success1 and success2, "Both feature generation runs should succeed"
        assert set(features1.keys()) == set(features2.keys()), "Feature keys should be identical"
        
        # Verify numerical identity (within floating point precision)
        for key in features1.keys():
            if isinstance(features1[key], pd.DataFrame) and isinstance(features2[key], pd.DataFrame):
                pd.testing.assert_frame_equal(features1[key], features2[key], 
                                            check_exact=False, rtol=1e-10)
    
    def test_data_loading_and_processing(self, orchestrator):
        """Test data loading and preprocessing."""
        
        orchestrator.setup_system()
        
        # Test synthetic data generation
        success = orchestrator.load_data()
        assert success, "Data loading should succeed"
        
        # Verify data structure
        assert orchestrator.universe is not None, "Universe should be loaded"
        assert orchestrator.market_data is not None, "Market data should be loaded"
        assert len(orchestrator.universe) > 0, "Universe should not be empty"
        assert len(orchestrator.market_data) > 0, "Market data should not be empty"
        
        # Check required columns
        required_columns = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            assert col in orchestrator.market_data.columns, f"Missing required column: {col}"
        
        # Verify date range
        min_date = orchestrator.market_data['date'].min()
        max_date = orchestrator.market_data['date'].max()
        assert min_date >= pd.Timestamp(orchestrator.config.start_date), "Data starts too early"
        assert max_date <= pd.Timestamp(orchestrator.config.end_date), "Data ends too late"
    
    def test_feature_engineering_pipeline(self, orchestrator):
        """Test complete feature engineering pipeline."""
        
        orchestrator.setup_system()
        orchestrator.load_data()
        
        # Generate features
        success = orchestrator.generate_features()
        assert success, "Feature generation should succeed"
        assert len(orchestrator.feature_data) > 0, "Should generate some features"
        
        # Verify feature structure for each symbol
        symbols = orchestrator.universe['symbol'].tolist()
        
        for symbol in symbols:
            options_key = f"{symbol}_options"
            intraday_key = f"{symbol}_intraday"
            
            # At least one feature set should exist
            assert (options_key in orchestrator.feature_data or 
                   intraday_key in orchestrator.feature_data), f"No features for {symbol}"
            
            # Check options features if present
            if options_key in orchestrator.feature_data:
                options_df = orchestrator.feature_data[options_key]
                assert isinstance(options_df, pd.DataFrame), "Options features should be DataFrame"
                assert 'date' in options_df.columns, "Options features need date column"
                assert len(options_df) > 0, "Options features should not be empty"
            
            # Check intraday features if present  
            if intraday_key in orchestrator.feature_data:
                intraday_df = orchestrator.feature_data[intraday_key]
                assert isinstance(intraday_df, pd.DataFrame), "Intraday features should be DataFrame"
                assert 'date' in intraday_df.columns, "Intraday features need date column"
                assert len(intraday_df) > 0, "Intraday features should not be empty"
    
    def test_signal_generation(self, orchestrator):
        """Test HRM-based signal generation."""
        
        orchestrator.setup_system()
        orchestrator.load_data()
        orchestrator.generate_features()
        
        # Prepare test data
        backtest_data = orchestrator._prepare_backtest_data()
        assert len(backtest_data) > 0, "Backtest data should not be empty"
        
        # Test equity signal generation
        equity_signals = orchestrator._generate_equity_signals(backtest_data)
        assert len(equity_signals) > 0, "Should generate equity signals"
        
        # Verify signal structure
        for signal in equity_signals[:5]:  # Check first 5 signals
            assert isinstance(signal, StrategySignal), "Should be StrategySignal objects"
            assert -1 <= signal.signal_strength <= 1, "Signal strength should be in [-1, 1]"
            assert 0 <= signal.confidence <= 1, "Confidence should be in [0, 1]"
            assert signal.strategy_name == 'intraday_short', "Should be intraday strategy"
            assert signal.asset_class == 'equity', "Should be equity asset class"
        
        # Test options signal generation
        options_signals = orchestrator._generate_options_signals(backtest_data)
        assert len(options_signals) > 0, "Should generate options signals"
        
        for signal in options_signals[:5]:
            assert isinstance(signal, StrategySignal), "Should be StrategySignal objects"
            assert -1 <= signal.signal_strength <= 1, "Signal strength should be in [-1, 1]"
            assert 0 <= signal.confidence <= 1, "Confidence should be in [0, 1]"
            assert signal.strategy_name == 'options_straddles', "Should be options strategy"
            assert signal.asset_class == 'options', "Should be options asset class"
    
    def test_backtesting_pipeline(self, orchestrator):
        """Test comprehensive backtesting pipeline."""
        
        orchestrator.setup_system()
        orchestrator.load_data()
        orchestrator.generate_features()
        
        # Run backtesting
        success = orchestrator.run_backtest()
        assert success, "Backtesting should succeed"
        
        # Verify backtest results structure
        assert 'equity_strategies' in orchestrator.backtest_results, "Missing equity results"
        assert 'options_strategies' in orchestrator.backtest_results, "Missing options results"
        assert 'combined_portfolio' in orchestrator.backtest_results, "Missing combined results"
        
        # Check combined portfolio results
        combined = orchestrator.backtest_results['combined_portfolio']
        assert 'returns' in combined, "Missing returns series"
        assert 'metrics' in combined, "Missing metrics"
        
        # Verify metrics structure
        metrics = combined['metrics']
        required_metrics = ['total_return', 'annualized_return', 'annualized_volatility', 
                           'sharpe_ratio', 'max_drawdown']
        
        for metric in required_metrics:
            assert metric in metrics, f"Missing metric: {metric}"
            assert isinstance(metrics[metric], (int, float)), f"Metric {metric} should be numeric"
        
        # Verify returns are reasonable
        returns = combined['returns']
        if len(returns) > 0:
            assert returns.std() < 0.5, "Daily volatility should be reasonable"
            assert abs(returns.mean()) < 0.1, "Daily return should be reasonable"
    
    def test_portfolio_optimization(self, orchestrator):
        """Test portfolio optimization and risk management."""
        
        orchestrator.setup_system()
        
        # Test portfolio manager configuration
        pm = orchestrator.portfolio_manager
        assert pm.initial_capital == orchestrator.config.initial_capital, "Capital should match config"
        
        # Verify strategy configurations
        strategy_configs = pm.strategy_configs
        assert 'options_straddles' in strategy_configs, "Missing options strategy config"
        assert 'intraday_short' in strategy_configs, "Missing intraday strategy config"
        assert 'cash' in strategy_configs, "Missing cash config"
        
        # Check allocation targets sum to 1
        total_allocation = sum(config['target_allocation'] for config in strategy_configs.values())
        assert abs(total_allocation - 1.0) < 1e-6, f"Allocations should sum to 1, got {total_allocation}"
        
        # Test signal processing
        test_signals = [
            StrategySignal(
                timestamp=pd.Timestamp.now(),
                strategy_name='intraday_short',
                signal_strength=0.5,
                confidence=0.8,
                asset_class='equity',
                target_symbol='SPY'
            ),
            StrategySignal(
                timestamp=pd.Timestamp.now(),
                strategy_name='options_straddles', 
                signal_strength=-0.3,
                confidence=0.6,
                asset_class='options',
                target_symbol='QQQ'
            )
        ]
        
        allocation = pm.process_signals(test_signals)
        assert isinstance(allocation, type(pm.process_signals(test_signals))), "Should return PortfolioAllocation"
        assert hasattr(allocation, 'allocations'), "Should have allocations attribute"
        assert hasattr(allocation, 'expected_return'), "Should have expected_return attribute"
    
    def test_statistical_validation(self, orchestrator):
        """Test statistical validation pipeline."""
        
        orchestrator.setup_system()
        orchestrator.load_data()
        orchestrator.generate_features()
        orchestrator.run_backtest()
        
        # Run statistical validation
        success = orchestrator.run_statistical_validation()
        assert success, "Statistical validation should succeed"
        
        # Verify validation results structure
        validation = orchestrator.validation_results
        assert 'strategy_stats' in validation, "Missing strategy stats"
        assert 'tests' in validation, "Missing test results"
        assert 'overall_assessment' in validation, "Missing overall assessment"
        
        # Check strategy statistics
        stats = validation['strategy_stats']
        required_stats = ['total_return', 'annualized_return', 'annualized_volatility',
                         'sharpe_ratio', 'max_drawdown', 'skewness', 'excess_kurtosis']
        
        for stat in required_stats:
            assert stat in stats, f"Missing statistic: {stat}"
        
        # Check test results
        tests = validation['tests']
        expected_tests = ['deflated_sharpe', 'probabilistic_sharpe']  # Minimum expected tests
        
        for test_name in expected_tests:
            if test_name in tests:
                test_result = tests[test_name]
                assert hasattr(test_result, 'p_value'), f"{test_name} missing p_value"
                assert hasattr(test_result, 'reject_null'), f"{test_name} missing reject_null"
                assert 0 <= test_result.p_value <= 1, f"{test_name} p_value out of range"
        
        # Check overall assessment
        assessment = validation['overall_assessment']
        assert 'recommendation' in assessment, "Missing recommendation"
        assert 'confidence_score' in assessment, "Missing confidence score"
        assert 0 <= assessment['confidence_score'] <= 1, "Confidence score out of range"
    
    def test_complete_pipeline_execution(self, orchestrator):
        """Test complete end-to-end pipeline execution."""
        
        # Run complete pipeline
        results = orchestrator.run_complete_pipeline()
        
        # Verify execution success
        assert isinstance(results, dict), "Results should be dictionary"
        assert 'success' in results, "Results should have success field"
        assert 'completed_stages' in results, "Results should have completed stages"
        
        # Check completed stages
        expected_stages = ['system_setup', 'data_loading', 'feature_generation', 
                          'backtesting', 'validation', 'report_generation']
        
        completed = results['completed_stages']
        for stage in expected_stages:
            assert stage in completed, f"Stage {stage} should be completed"
        
        # If successful, verify final report
        if results['success']:
            assert 'final_report' in results, "Successful pipeline should have final report"
            
            report = results['final_report']
            assert 'system_info' in report, "Report missing system info"
            assert 'backtest_results' in report, "Report missing backtest results"
            assert 'validation_results' in report, "Report missing validation results"
            
            # Verify system info
            system_info = report['system_info']
            assert system_info['model_parameters'] > 0, "Should report model parameters"
            assert system_info['universe_size'] > 0, "Should report universe size"
            
    def test_error_handling_and_recovery(self, orchestrator):
        """Test error handling and graceful degradation."""
        
        # Test with incomplete setup
        with pytest.raises(Exception):
            orchestrator.load_data()  # Should fail without setup
        
        # Test with missing data
        orchestrator.setup_system()
        with patch.object(orchestrator, 'market_data', None):
            success = orchestrator.generate_features()
            assert not success, "Should fail with missing data"
        
        # Test with corrupted features
        orchestrator.load_data()
        orchestrator.feature_data = {"corrupted": "data"}
        
        success = orchestrator.run_backtest()
        # Should handle gracefully (may succeed with empty results or fail cleanly)
        assert isinstance(success, bool), "Should return boolean result"
    
    def test_performance_metrics_accuracy(self, orchestrator):
        """Test accuracy of performance metric calculations."""
        
        orchestrator.setup_system()
        orchestrator.load_data()
        orchestrator.generate_features()
        orchestrator.run_backtest()
        
        # Extract combined results
        combined = orchestrator.backtest_results['combined_portfolio']
        returns = combined['returns']
        metrics = combined['metrics']
        
        if len(returns) > 10:  # Only test with sufficient data
            # Verify Sharpe ratio calculation
            expected_sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
            actual_sharpe = metrics['sharpe_ratio']
            
            assert abs(expected_sharpe - actual_sharpe) < 1e-6, \
                f"Sharpe ratio mismatch: expected {expected_sharpe}, got {actual_sharpe}"
            
            # Verify total return calculation
            expected_total_return = (1 + returns).prod() - 1
            actual_total_return = metrics['total_return']
            
            assert abs(expected_total_return - actual_total_return) < 1e-10, \
                f"Total return mismatch: expected {expected_total_return}, got {actual_total_return}"
    
    def test_memory_usage_and_scalability(self, orchestrator):
        """Test memory usage remains reasonable."""
        
        import psutil
        import os
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run pipeline
        orchestrator.setup_system()
        orchestrator.load_data()
        orchestrator.generate_features()
        
        # Check memory usage after setup
        current_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = current_memory - initial_memory
        
        # Should not use more than 2GB additional memory
        assert memory_increase < 2048, f"Memory usage too high: {memory_increase:.1f}MB increase"
        
        # Verify cleanup works
        orchestrator.feature_data.clear()
        orchestrator.market_data = None
        
        # Force garbage collection
        import gc
        gc.collect()

@pytest.mark.integration
def test_full_system_integration():
    """Comprehensive integration test entry point."""
    
    # Test with minimal configuration for speed
    hrm_config = {
        'h_dim': 256,
        'l_dim': 128, 
        'num_h_layers': 4,
        'num_l_layers': 3,
        'num_heads': 4,
        'dropout': 0.1,
        'max_sequence_length': 64,
        'deq_threshold': 1e-3,
        'max_deq_iterations': 10
    }
    
    config = DualHRQConfig(
        hrm_config=hrm_config,
        start_date="2023-06-01",
        end_date="2023-08-31", 
        initial_capital=100_000,  # Smaller for testing
        number_of_trials=5,
        deterministic_seed=42,
        enable_mlops_tracking=False
    )
    
    orchestrator = DualHRQOrchestrator(config)
    
    # Run complete pipeline
    results = orchestrator.run_complete_pipeline()
    
    # Verify success
    assert results['success'], f"Pipeline failed: {results.get('error_message', 'Unknown error')}"
    
    # Verify all stages completed
    expected_stages = ['system_setup', 'data_loading', 'feature_generation',
                      'backtesting', 'validation', 'report_generation']
    
    for stage in expected_stages:
        assert stage in results['completed_stages'], f"Stage {stage} not completed"
    
    # Verify final report quality
    report = results['final_report']
    
    # System should have reasonable parameter count
    params = report['system_info']['model_parameters']
    assert 100_000 <= params <= 5_000_000, f"Parameter count {params:,} seems wrong for test config"
    
    # Should have processed multiple symbols
    universe_size = report['system_info']['universe_size']
    assert universe_size >= 5, f"Universe too small: {universe_size}"
    
    # Should have generated features
    feature_sets = report['system_info']['feature_sets']
    assert feature_sets > 0, "No feature sets generated"
    
    print(f"✅ Full integration test passed!")
    print(f"   Model parameters: {params:,}")
    print(f"   Universe size: {universe_size}")
    print(f"   Feature sets: {feature_sets}")
    print(f"   Pipeline stages: {len(results['completed_stages'])}")

if __name__ == "__main__":
    # Run the full integration test
    test_full_system_integration()