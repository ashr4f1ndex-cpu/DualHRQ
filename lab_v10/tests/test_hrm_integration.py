"""
Tests for HRM Integration with Walk-Forward Testing.

Tests the integration between pattern library, adaptive budget system,
and walk-forward testing to ensure proper HRM-inspired learning.
"""

import pytest
import tempfile
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import sys
sys.path.append('/Users/aziymandias/Desktop/dual_book_trading_lab_v10_complete/lab_v10')

from src.models.hrm_integration import HRMWalkForwardIntegrator
from src.models.adaptive_budget import ComputeBudget, UncertaintyMethod


def create_test_market_data(n_days: int = 100, regime: str = 'normal') -> pd.DataFrame:
    """Create test market data for HRM integration testing."""
    np.random.seed(42)
    
    dates = pd.date_range('2023-01-01', periods=n_days, freq='D')
    
    # Base volatility based on regime
    vol_multipliers = {'low': 0.5, 'normal': 1.0, 'high': 2.0}
    base_vol = 0.02 * vol_multipliers.get(regime, 1.0)
    
    # Generate price series
    prices = [100.0]
    for i in range(1, n_days):
        return_ = np.random.normal(0, base_vol)
        prices.append(prices[-1] * (1 + return_))
    
    # Create OHLCV data
    data = []
    for i, date in enumerate(dates):
        close = prices[i]
        high = close * (1 + abs(np.random.normal(0, base_vol * 0.5)))
        low = close * (1 - abs(np.random.normal(0, base_vol * 0.5)))
        open_price = low + (high - low) * np.random.uniform(0.3, 0.7)
        
        data.append({
            'timestamp': date,
            'open': open_price,
            'high': max(high, open_price, close),
            'low': min(low, open_price, close),
            'close': close,
            'volume': int(np.random.uniform(100000, 1000000)),
            'symbol': 'TEST'
        })
    
    df = pd.DataFrame(data)
    df['returns'] = df['close'].pct_change()
    
    return df


def create_test_training_data(n_samples: int = 100, n_features: int = 5,
                             task_type: str = 'classification') -> tuple:
    """Create test training data."""
    np.random.seed(42)
    
    X = torch.randn(n_samples, n_features)
    
    if task_type == 'classification':
        # Binary classification based on feature sum
        y = (torch.sum(X, dim=1) > 0).long()
    else:
        # Regression task
        y = torch.sum(X * torch.randn(n_features), dim=1, keepdim=True) + torch.randn(n_samples, 1) * 0.1
    
    return X, y


class SimpleTestModel(nn.Module):
    """Simple model for testing HRM integration."""
    
    def __init__(self, input_dim: int = 5, output_dim: int = 2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim)
        )
        self.out_features = output_dim
    
    def forward(self, x):
        return self.layers(x)


class TestHRMWalkForwardIntegrator:
    """Test HRM walk-forward integration functionality."""
    
    def test_integrator_initialization(self):
        """Test integrator initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            integrator = HRMWalkForwardIntegrator(
                pattern_library_dir=temp_dir,
                uncertainty_method=UncertaintyMethod.ENTROPY
            )
            
            assert integrator.pattern_library is not None
            assert integrator.uncertainty_method == UncertaintyMethod.ENTROPY
            assert len(integrator.adaptation_history) == 0
            assert isinstance(integrator.default_budget, ComputeBudget)
    
    def test_adaptive_model_creation(self):
        """Test creating adaptive model from base model."""
        with tempfile.TemporaryDirectory() as temp_dir:
            integrator = HRMWalkForwardIntegrator(pattern_library_dir=temp_dir)
            base_model = SimpleTestModel()
            
            adaptive_model = integrator.create_adaptive_model(base_model)
            
            assert adaptive_model is not None
            assert adaptive_model.base_model is base_model
            assert adaptive_model.scheduler is not None
            assert hasattr(adaptive_model, 'refinement_module')
    
    def test_model_adaptation_for_period(self):
        """Test adapting model for walk-forward period."""
        with tempfile.TemporaryDirectory() as temp_dir:
            integrator = HRMWalkForwardIntegrator(pattern_library_dir=temp_dir)
            
            # Create test data
            market_data = create_test_market_data(50, 'high')
            training_data = create_test_training_data(100, 5, 'classification')
            base_model = SimpleTestModel()
            reference_date = datetime(2023, 1, 15)
            
            # Adapt model
            adapted_model, adaptation_info = integrator.adapt_model_for_period(
                base_model, market_data, training_data, reference_date, 'period_1'
            )
            
            # Verify adaptation
            assert adapted_model is not None
            assert adaptation_info['period_id'] == 'period_1'
            assert 'pattern_id' in adaptation_info
            assert 'regime_signature' in adaptation_info
            assert isinstance(adaptation_info['training_samples'], int)
            
            # Check adaptation history
            assert len(integrator.adaptation_history) == 1
            assert integrator.adaptation_history[0]['period_id'] == 'period_1'
    
    def test_pattern_aware_prediction(self):
        """Test making predictions with pattern awareness."""
        with tempfile.TemporaryDirectory() as temp_dir:
            integrator = HRMWalkForwardIntegrator(pattern_library_dir=temp_dir)
            
            # Set up model and data
            base_model = SimpleTestModel()
            adaptive_model = integrator.create_adaptive_model(base_model)
            market_data = create_test_market_data(40, 'normal')
            reference_date = datetime(2023, 1, 20)
            
            # Create test input
            X = torch.randn(10, 5)
            
            # Make prediction
            predictions, prediction_info = integrator.predict_with_pattern_awareness(
                adaptive_model, X, market_data, reference_date, use_adaptive_compute=True
            )
            
            # Verify predictions
            assert predictions.shape == (10, 2)
            assert 'pattern_id' in prediction_info
            assert 'regime_signature' in prediction_info
            assert 'adaptive_compute_used' in prediction_info
            assert prediction_info['adaptive_compute_used'] == True
    
    def test_period_performance_evaluation(self):
        """Test evaluating performance for a period."""
        with tempfile.TemporaryDirectory() as temp_dir:
            integrator = HRMWalkForwardIntegrator(pattern_library_dir=temp_dir)
            
            # Create test predictions and targets
            predictions = torch.tensor([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1]])
            targets = torch.tensor([0, 1, 0])
            
            # Evaluate performance
            performance = integrator.evaluate_period_performance(
                'test_period', predictions, targets, 'test_pattern'
            )
            
            # Verify performance metrics
            assert 'accuracy' in performance
            assert 'task_type' in performance
            assert performance['task_type'] == 'classification'
            assert performance['period_id'] == 'test_period'
            assert performance['pattern_id'] == 'test_pattern'
            assert 0.0 <= performance['accuracy'] <= 1.0
            
            # Check performance tracking
            assert 'test_pattern' in integrator.performance_tracking
            assert len(integrator.performance_tracking['test_pattern']) == 1
    
    def test_multiple_period_adaptation(self):
        """Test adaptation across multiple walk-forward periods."""
        with tempfile.TemporaryDirectory() as temp_dir:
            integrator = HRMWalkForwardIntegrator(pattern_library_dir=temp_dir)
            base_model = SimpleTestModel()
            
            # Simulate multiple periods
            periods = [
                ('2023-01-15', 'high'),
                ('2023-02-15', 'normal'), 
                ('2023-03-15', 'low'),
                ('2023-04-15', 'high')
            ]
            
            adapted_models = []
            
            for i, (date_str, regime) in enumerate(periods):
                reference_date = datetime.strptime(date_str, '%Y-%m-%d')
                market_data = create_test_market_data(30, regime)
                training_data = create_test_training_data(80, 5)
                
                adapted_model, adaptation_info = integrator.adapt_model_for_period(
                    base_model, market_data, training_data, reference_date, f'period_{i+1}'
                )
                
                adapted_models.append((adapted_model, adaptation_info))
            
            # Verify all adaptations
            assert len(integrator.adaptation_history) == 4
            assert len(set(a['pattern_id'] for a in integrator.adaptation_history)) >= 2  # At least 2 different patterns
            
            # Check pattern library has patterns
            assert len(integrator.pattern_library.patterns) > 0
    
    def test_adaptation_summary(self):
        """Test getting adaptation summary."""
        with tempfile.TemporaryDirectory() as temp_dir:
            integrator = HRMWalkForwardIntegrator(pattern_library_dir=temp_dir)
            
            # Initially empty
            summary = integrator.get_adaptation_summary()
            assert summary['total_adaptations'] == 0
            
            # Add some adaptations
            base_model = SimpleTestModel()
            for i in range(3):
                market_data = create_test_market_data(25, 'normal')
                training_data = create_test_training_data(50, 5)
                reference_date = datetime(2023, 1, 1) + timedelta(days=i*30)
                
                integrator.adapt_model_for_period(
                    base_model, market_data, training_data, reference_date, f'period_{i}'
                )
            
            # Get summary
            summary = integrator.get_adaptation_summary()
            
            assert summary['total_adaptations'] == 3
            assert 'unique_patterns' in summary
            assert 'adaptation_success_rate' in summary
            assert 'library_stats' in summary
            assert 0.0 <= summary['adaptation_success_rate'] <= 1.0
    
    def test_budget_optimization(self):
        """Test budget optimization for patterns."""
        with tempfile.TemporaryDirectory() as temp_dir:
            integrator = HRMWalkForwardIntegrator(pattern_library_dir=temp_dir)
            
            # Create model and validation data
            base_model = SimpleTestModel()
            adaptive_model = integrator.create_adaptive_model(base_model)
            
            validation_data = [create_test_training_data(20, 5) for _ in range(5)]
            
            # Optimize budget
            optimized_budget = integrator.optimize_budget_for_patterns(validation_data, adaptive_model)
            
            assert isinstance(optimized_budget, ComputeBudget)
            assert optimized_budget.max_refinement_steps >= optimized_budget.min_refinement_steps
            assert 0.0 <= optimized_budget.uncertainty_threshold <= 1.0
            assert 0.0 <= optimized_budget.confidence_threshold <= 1.0
    
    def test_state_persistence(self):
        """Test saving and loading integrator state."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create integrator and add some patterns
            integrator1 = HRMWalkForwardIntegrator(pattern_library_dir=temp_dir)
            
            market_data = create_test_market_data(30)
            training_data = create_test_training_data(40, 5)
            base_model = SimpleTestModel()
            reference_date = datetime(2023, 1, 10)
            
            integrator1.adapt_model_for_period(
                base_model, market_data, training_data, reference_date, 'test_period'
            )
            
            # Save state
            integrator1.save_state()
            
            # Create new integrator (should load existing patterns)
            integrator2 = HRMWalkForwardIntegrator(pattern_library_dir=temp_dir)
            
            # Should have loaded the pattern
            assert len(integrator2.pattern_library.patterns) == 1


class TestHRMIntegrationEdgeCases:
    """Test edge cases for HRM integration."""
    
    def test_insufficient_training_data(self):
        """Test behavior with insufficient training data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            integrator = HRMWalkForwardIntegrator(pattern_library_dir=temp_dir)
            
            # Very small training dataset
            market_data = create_test_market_data(20)
            training_data = create_test_training_data(5, 5)  # Only 5 samples
            base_model = SimpleTestModel()
            reference_date = datetime(2023, 1, 15)
            
            # Should handle gracefully
            adapted_model, adaptation_info = integrator.adapt_model_for_period(
                base_model, market_data, training_data, reference_date
            )
            
            assert adapted_model is not None
            assert adaptation_info['specialist_adapted'] == False  # Should not adapt with insufficient data
    
    def test_empty_market_data(self):
        """Test behavior with empty market data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            integrator = HRMWalkForwardIntegrator(pattern_library_dir=temp_dir)
            
            # Empty market data
            empty_data = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            training_data = create_test_training_data(50, 5)
            base_model = SimpleTestModel()
            reference_date = datetime(2023, 1, 15)
            
            # Should handle gracefully
            adapted_model, adaptation_info = integrator.adapt_model_for_period(
                base_model, empty_data, training_data, reference_date
            )
            
            assert adapted_model is not None
            assert 'pattern_id' in adaptation_info
    
    def test_regression_task_evaluation(self):
        """Test performance evaluation for regression tasks."""
        with tempfile.TemporaryDirectory() as temp_dir:
            integrator = HRMWalkForwardIntegrator(pattern_library_dir=temp_dir)
            
            # Create regression predictions and targets
            predictions = torch.tensor([[1.5], [2.3], [0.8]])
            targets = torch.tensor([[1.2], [2.1], [1.0]])
            
            # Evaluate performance
            performance = integrator.evaluate_period_performance(
                'regression_period', predictions, targets, 'regression_pattern'
            )
            
            # Verify regression metrics
            assert 'mse' in performance
            assert 'rmse' in performance
            assert performance['task_type'] == 'regression'
            assert performance['mse'] >= 0
            assert performance['rmse'] >= 0


def test_hrm_integration_end_to_end():
    """End-to-end test of HRM integration in walk-forward context."""
    with tempfile.TemporaryDirectory() as temp_dir:
        integrator = HRMWalkForwardIntegrator(
            pattern_library_dir=temp_dir,
            uncertainty_method=UncertaintyMethod.ENTROPY
        )
        
        base_model = SimpleTestModel(input_dim=4, output_dim=2)
        
        # Simulate walk-forward testing with multiple periods
        periods = [
            ('2023-01-15', 'high', 100),
            ('2023-02-15', 'normal', 120), 
            ('2023-03-15', 'low', 80),
            ('2023-04-15', 'crisis', 90),
            ('2023-05-15', 'normal', 110)
        ]
        
        all_performances = []
        
        for period_idx, (date_str, regime, n_samples) in enumerate(periods):
            reference_date = datetime.strptime(date_str, '%Y-%m-%d')
            period_id = f'period_{period_idx+1}'
            
            # Create period-specific data
            market_data = create_test_market_data(40, regime)
            training_data = create_test_training_data(n_samples, 4, 'classification')
            test_data = create_test_training_data(30, 4, 'classification')
            
            # Adapt model for this period
            adapted_model, adaptation_info = integrator.adapt_model_for_period(
                base_model, market_data, training_data, reference_date, period_id
            )
            
            # Make predictions on test data
            X_test, y_test = test_data
            predictions, prediction_info = integrator.predict_with_pattern_awareness(
                adapted_model, X_test, market_data, reference_date
            )
            
            # Evaluate performance
            performance = integrator.evaluate_period_performance(
                period_id, predictions, y_test, prediction_info['pattern_id']
            )
            
            all_performances.append(performance)
        
        # Verify end-to-end results
        assert len(all_performances) == 5
        assert len(integrator.adaptation_history) == 5
        assert len(integrator.pattern_library.patterns) > 0
        
        # Get final summary
        summary = integrator.get_adaptation_summary()
        
        assert summary['total_adaptations'] == 5
        assert summary['unique_patterns'] >= 2  # Should have found different patterns
        assert len(summary['pattern_performance']) > 0
        
        # Save state
        integrator.save_state()
        
        print(f"End-to-end test completed:")
        print(f"- {summary['total_adaptations']} adaptations")
        print(f"- {summary['unique_patterns']} unique patterns")
        print(f"- {summary['adaptation_success_rate']:.2%} adaptation success rate")
        
        # Verify all accuracies are reasonable
        accuracies = [p['accuracy'] for p in all_performances]
        assert all(0.2 <= acc <= 1.0 for acc in accuracies), f"Unrealistic accuracies: {accuracies}"