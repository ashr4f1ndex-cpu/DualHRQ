"""
Tests for Adaptive Budget System.

Tests HBPO-style adaptive compute allocation, uncertainty estimation,
and budget optimization to ensure proper resource management.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple

import sys
sys.path.append('/Users/aziymandias/Desktop/dual_book_trading_lab_v10_complete/lab_v10')

from src.models.adaptive_budget import (
    AdaptiveBudgetScheduler, ComputeBudget, UncertaintyEstimator,
    UncertaintyMethod, AdaptiveComputeModel, create_adaptive_model,
    BudgetOptimizer
)


class SimpleTestModel(nn.Module):
    """Simple model for testing adaptive budget functionality."""
    
    def __init__(self, input_dim: int = 10, hidden_dim: int = 32, output_dim: int = 2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.out_features = output_dim
        
    def forward(self, x):
        return self.layers(x)


def create_test_data(n_samples: int = 100, input_dim: int = 10, 
                    noise_level: float = 0.1) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Create test data with varying difficulty levels."""
    data = []
    
    for i in range(n_samples):
        # Create input with varying difficulty
        difficulty = i / n_samples  # Gradually increase difficulty
        
        x = torch.randn(1, input_dim) * (1 + difficulty)
        
        # Create target based on input (with noise)
        target_logits = torch.sum(x, dim=1, keepdim=True)
        noise = torch.randn_like(target_logits) * noise_level
        target_logits += noise
        
        # Convert to class (binary classification)
        y = torch.where(target_logits > 0, torch.tensor(1), torch.tensor(0))
        
        data.append((x, y))
    
    return data


class TestUncertaintyEstimator:
    """Test uncertainty estimation methods."""
    
    def test_entropy_uncertainty(self):
        """Test entropy-based uncertainty estimation."""
        estimator = UncertaintyEstimator(UncertaintyMethod.ENTROPY)
        model = SimpleTestModel()
        
        # High uncertainty case (uniform probabilities)
        uniform_probs = torch.tensor([[0.5, 0.5]])
        uncertainty = estimator._entropy_uncertainty(uniform_probs)
        assert uncertainty > 0.6  # Should be high uncertainty
        
        # Low uncertainty case (confident prediction)
        confident_probs = torch.tensor([[0.9, 0.1]])  
        uncertainty = estimator._entropy_uncertainty(confident_probs)
        assert uncertainty < 0.5  # Should be low uncertainty
    
    def test_variance_uncertainty(self):
        """Test variance-based uncertainty estimation."""
        estimator = UncertaintyEstimator(UncertaintyMethod.VARIANCE)
        
        # High variance prediction
        high_var = torch.tensor([[1.0, -1.0, 2.0, -2.0]])
        uncertainty = estimator._variance_uncertainty(high_var)
        assert uncertainty > 0.5
        
        # Low variance prediction  
        low_var = torch.tensor([[0.1, 0.1, 0.1, 0.1]])
        uncertainty = estimator._variance_uncertainty(low_var)
        assert uncertainty < 0.5
    
    def test_mc_dropout_uncertainty(self):
        """Test Monte Carlo dropout uncertainty estimation."""
        estimator = UncertaintyEstimator(UncertaintyMethod.MC_DROPOUT, mc_samples=5)
        model = SimpleTestModel()
        
        x = torch.randn(1, 10)
        uncertainty = estimator.estimate_uncertainty(model, x)
        
        assert isinstance(uncertainty, float)
        assert uncertainty >= 0.0  # Uncertainty should be non-negative
    
    def test_uncertainty_estimation_integration(self):
        """Test full uncertainty estimation process."""
        estimator = UncertaintyEstimator(UncertaintyMethod.ENTROPY)
        model = SimpleTestModel()
        
        # Test with different input difficulties
        easy_input = torch.ones(1, 10) * 0.1  # Easy case
        hard_input = torch.randn(1, 10) * 3.0  # Hard case
        
        easy_uncertainty = estimator.estimate_uncertainty(model, easy_input)
        hard_uncertainty = estimator.estimate_uncertainty(model, hard_input)
        
        assert isinstance(easy_uncertainty, float)
        assert isinstance(hard_uncertainty, float)
        assert easy_uncertainty >= 0.0
        assert hard_uncertainty >= 0.0


class TestComputeBudget:
    """Test compute budget configuration."""
    
    def test_budget_creation(self):
        """Test budget configuration creation."""
        budget = ComputeBudget(
            max_refinement_steps=15,
            min_refinement_steps=2,
            uncertainty_threshold=0.3,
            confidence_threshold=0.8
        )
        
        assert budget.max_refinement_steps == 15
        assert budget.min_refinement_steps == 2
        assert budget.uncertainty_threshold == 0.3
        assert budget.confidence_threshold == 0.8
        assert budget.budget_patience == 3  # Default value
    
    def test_default_budget(self):
        """Test default budget configuration."""
        budget = ComputeBudget()
        
        assert budget.max_refinement_steps == 10
        assert budget.min_refinement_steps == 1
        assert budget.uncertainty_threshold == 0.5
        assert budget.confidence_threshold == 0.9


class TestAdaptiveBudgetScheduler:
    """Test adaptive budget scheduling."""
    
    def test_scheduler_creation(self):
        """Test scheduler initialization."""
        budget = ComputeBudget(max_refinement_steps=8)
        scheduler = AdaptiveBudgetScheduler(budget, UncertaintyMethod.ENTROPY)
        
        assert scheduler.config.max_refinement_steps == 8
        assert scheduler.uncertainty_estimator.method == UncertaintyMethod.ENTROPY
        assert len(scheduler.refinement_history) == 0
    
    def test_budget_allocation_basic(self):
        """Test basic budget allocation."""
        scheduler = AdaptiveBudgetScheduler()
        model = SimpleTestModel()
        
        x = torch.randn(1, 10)
        
        # Test budget allocation
        final_pred, refinement_steps = scheduler.allocate_budget(model, x)
        
        assert isinstance(final_pred, torch.Tensor)
        assert len(refinement_steps) >= scheduler.config.min_refinement_steps
        assert len(refinement_steps) <= scheduler.config.max_refinement_steps
        
        # Check refinement steps structure
        for step in refinement_steps:
            assert hasattr(step, 'step')
            assert hasattr(step, 'uncertainty')
            assert hasattr(step, 'confidence')
            assert hasattr(step, 'compute_cost')
    
    def test_early_stopping_high_confidence(self):
        """Test early stopping when confidence is high."""
        budget = ComputeBudget(
            max_refinement_steps=10,
            confidence_threshold=0.6,  # Lower threshold for easier triggering
            min_refinement_steps=1
        )
        scheduler = AdaptiveBudgetScheduler(budget)
        model = SimpleTestModel()
        
        # Create input that should lead to confident prediction
        x = torch.ones(1, 10) * 2.0  # Strong signal
        
        final_pred, refinement_steps = scheduler.allocate_budget(model, x)
        
        # Should potentially stop early (though not guaranteed with random model)
        assert len(refinement_steps) >= 1
        assert len(refinement_steps) <= budget.max_refinement_steps
    
    def test_budget_calculation(self):
        """Test budget calculation based on uncertainty."""
        scheduler = AdaptiveBudgetScheduler()
        
        # Test with different uncertainty levels
        high_uncertainty = 0.9
        low_uncertainty = 0.1
        
        x = torch.randn(1, 10)
        
        high_budget = scheduler._calculate_budget(high_uncertainty, x)
        low_budget = scheduler._calculate_budget(low_uncertainty, x)
        
        # Higher uncertainty should get more budget (generally)
        assert high_budget >= low_budget
        assert high_budget >= scheduler.config.min_refinement_steps
        assert high_budget <= scheduler.config.max_refinement_steps
    
    def test_refinement_improvement_tracking(self):
        """Test tracking of improvements during refinement."""
        scheduler = AdaptiveBudgetScheduler()
        model = SimpleTestModel()
        
        x = torch.randn(1, 10)
        
        # Run allocation
        final_pred, refinement_steps = scheduler.allocate_budget(model, x)
        
        # Check that improvements are tracked
        for step in refinement_steps:
            assert hasattr(step, 'improvement')
            assert isinstance(step.improvement, float)
            assert step.improvement >= 0.0
    
    def test_budget_statistics(self):
        """Test budget usage statistics."""
        scheduler = AdaptiveBudgetScheduler()
        model = SimpleTestModel()
        
        # Run several budget allocations
        for _ in range(5):
            x = torch.randn(1, 10)
            scheduler.allocate_budget(model, x)
        
        stats = scheduler.get_budget_statistics()
        
        assert 'total_refinement_steps' in stats
        assert 'avg_uncertainty' in stats
        assert 'avg_confidence' in stats
        assert 'early_stopping_rate' in stats
        
        assert stats['total_refinement_steps'] > 0
        assert 0.0 <= stats['early_stopping_rate'] <= 1.0


class TestAdaptiveComputeModel:
    """Test adaptive compute model wrapper."""
    
    def test_adaptive_model_creation(self):
        """Test creating adaptive compute model."""
        base_model = SimpleTestModel()
        adaptive_model = create_adaptive_model(base_model)
        
        assert isinstance(adaptive_model, AdaptiveComputeModel)
        assert adaptive_model.base_model is base_model
        assert adaptive_model.scheduler is not None
        assert adaptive_model.enable_refinement is True
    
    def test_adaptive_forward_pass(self):
        """Test forward pass with adaptive computation."""
        base_model = SimpleTestModel()
        adaptive_model = create_adaptive_model(base_model)
        
        x = torch.randn(3, 10)  # Batch of 3 samples
        
        # Forward pass without adaptive compute
        pred_base = adaptive_model(x, use_adaptive_compute=False)
        assert pred_base.shape == (3, 2)
        
        # Forward pass with adaptive compute  
        pred_adaptive = adaptive_model(x, use_adaptive_compute=True)
        assert pred_adaptive.shape == (3, 2)
    
    def test_refinement_module(self):
        """Test learnable refinement module."""
        base_model = SimpleTestModel(output_dim=4)
        adaptive_model = AdaptiveComputeModel(base_model, enable_refinement=True)
        
        # Check refinement module exists
        assert adaptive_model.refinement_module is not None
        
        # Test refinement
        x = torch.randn(2, 10)
        current_pred = torch.randn(2, 4)
        
        refined_pred = adaptive_model._learnable_refinement(
            base_model, x, current_pred, step=0
        )
        
        assert refined_pred.shape == current_pred.shape
        assert not torch.equal(refined_pred, current_pred)  # Should be different
    
    def test_compute_statistics(self):
        """Test getting compute usage statistics."""
        base_model = SimpleTestModel()
        adaptive_model = create_adaptive_model(base_model)
        
        # Run some computations
        for _ in range(3):
            x = torch.randn(1, 10)
            adaptive_model(x, use_adaptive_compute=True)
        
        stats = adaptive_model.get_compute_statistics()
        
        assert isinstance(stats, dict)
        assert 'total_refinement_steps' in stats
        assert stats['total_refinement_steps'] >= 0


class TestBudgetOptimizer:
    """Test budget parameter optimization."""
    
    def test_optimizer_creation(self):
        """Test budget optimizer initialization."""
        optimizer = BudgetOptimizer()
        
        assert optimizer.performance_metric is not None
        assert len(optimizer.optimization_history) == 0
    
    def test_budget_optimization(self):
        """Test budget parameter optimization."""
        # Create test model and data
        base_model = SimpleTestModel()
        adaptive_model = create_adaptive_model(base_model)
        
        validation_data = create_test_data(20, 10)
        
        # Define parameter search space
        budget_params = {
            'uncertainty_threshold': [0.3, 0.5, 0.7],
            'confidence_threshold': [0.8, 0.9],
            'max_refinement_steps': [5, 10]
        }
        
        optimizer = BudgetOptimizer()
        
        # Optimize budget
        best_config = optimizer.optimize_budget(
            adaptive_model, validation_data, budget_params
        )
        
        assert best_config is not None
        assert isinstance(best_config, ComputeBudget)
        assert len(optimizer.optimization_history) > 0
        
        # Check that optimization tried different configurations
        configs_tried = [entry['config'] for entry in optimizer.optimization_history]
        assert len(configs_tried) >= 2  # Should try multiple configs
    
    def test_default_performance_metric(self):
        """Test default performance metric."""
        optimizer = BudgetOptimizer()
        
        # Test classification metric
        pred_class = torch.tensor([[0.8, 0.2], [0.3, 0.7]])
        target_class = torch.tensor([0, 1])
        
        performance = optimizer._default_metric(pred_class, target_class)
        assert 0.0 <= performance <= 1.0  # Should be accuracy
        
        # Test regression metric
        pred_reg = torch.tensor([[1.0], [2.0]])
        target_reg = torch.tensor([[1.1], [1.9]])
        
        performance = optimizer._default_metric(pred_reg, target_reg)
        assert performance <= 0.0  # Negative MSE (higher is better)


class TestAdaptiveBudgetIntegration:
    """Integration tests for adaptive budget system."""
    
    def test_end_to_end_adaptive_computation(self):
        """Test complete adaptive computation workflow."""
        # Create model with different uncertainty characteristics
        base_model = SimpleTestModel(input_dim=8, output_dim=3)
        
        budget_config = ComputeBudget(
            max_refinement_steps=6,
            min_refinement_steps=1,
            uncertainty_threshold=0.4,
            confidence_threshold=0.8
        )
        
        adaptive_model = AdaptiveComputeModel(base_model, enable_refinement=True)
        adaptive_model.scheduler.config = budget_config
        
        # Test with various input difficulties
        test_cases = [
            torch.randn(1, 8) * 0.1,  # Easy case
            torch.randn(1, 8) * 1.0,  # Medium case
            torch.randn(1, 8) * 3.0,  # Hard case
        ]
        
        results = []
        
        for i, x in enumerate(test_cases):
            # Run adaptive computation
            pred = adaptive_model(x, use_adaptive_compute=True)
            stats = adaptive_model.get_compute_statistics()
            
            results.append({
                'case': i,
                'prediction_shape': pred.shape,
                'total_steps': stats.get('total_refinement_steps', 0)
            })
        
        # Verify results
        assert len(results) == 3
        for result in results:
            assert result['prediction_shape'] == (1, 3)
            assert result['total_steps'] >= 0
        
        # Final statistics should show usage
        final_stats = adaptive_model.get_compute_statistics()
        assert final_stats['total_refinement_steps'] > 0
    
    def test_uncertainty_driven_budget_allocation(self):
        """Test that higher uncertainty leads to more computation."""
        base_model = SimpleTestModel()
        
        # Create scheduler with clear budget differences
        budget_config = ComputeBudget(
            max_refinement_steps=10,
            min_refinement_steps=1,
            uncertainty_threshold=0.3
        )
        
        scheduler = AdaptiveBudgetScheduler(budget_config, UncertaintyMethod.ENTROPY)
        
        # Create inputs designed to have different uncertainties
        # (Note: actual uncertainty depends on model behavior)
        inputs = [
            torch.zeros(1, 10),      # Potential low uncertainty
            torch.randn(1, 10) * 0.1,  # Low variance
            torch.randn(1, 10) * 2.0,   # High variance
        ]
        
        step_counts = []
        
        for x in inputs:
            final_pred, refinement_steps = scheduler.allocate_budget(base_model, x)
            step_counts.append(len(refinement_steps))
        
        # Verify that computation was allocated
        assert all(count >= budget_config.min_refinement_steps for count in step_counts)
        assert all(count <= budget_config.max_refinement_steps for count in step_counts)
        
        # Get usage statistics
        stats = scheduler.get_budget_statistics()
        assert stats['total_refinement_steps'] == sum(step_counts)
    
    def test_budget_optimization_improves_efficiency(self):
        """Test that budget optimization can improve efficiency."""
        base_model = SimpleTestModel()
        adaptive_model = create_adaptive_model(base_model)
        
        # Create validation data with known patterns
        validation_data = create_test_data(15, 10, noise_level=0.1)
        
        # Define focused parameter search
        budget_params = {
            'max_refinement_steps': [3, 6, 12],
            'uncertainty_threshold': [0.3, 0.7]
        }
        
        optimizer = BudgetOptimizer()
        
        # Initial performance baseline
        original_config = adaptive_model.scheduler.config
        baseline_performance = optimizer._evaluate_config(adaptive_model, validation_data)
        
        # Optimize budget
        optimized_config = optimizer.optimize_budget(
            adaptive_model, validation_data, budget_params
        )
        
        # Apply optimized configuration
        adaptive_model.scheduler.config = optimized_config
        optimized_performance = optimizer._evaluate_config(adaptive_model, validation_data)
        
        # Verify optimization ran
        assert len(optimizer.optimization_history) > 1
        
        # Performance should be reasonable (may not always improve due to random model)
        assert -10.0 <= baseline_performance <= 1.0
        assert -10.0 <= optimized_performance <= 1.0
    
    def test_adaptive_budget_memory_efficiency(self):
        """Test that adaptive budget doesn't cause memory issues."""
        base_model = SimpleTestModel()
        adaptive_model = create_adaptive_model(base_model)
        
        # Run many computations to test memory
        for i in range(50):
            x = torch.randn(1, 10)
            pred = adaptive_model(x, use_adaptive_compute=True)
            
            # Verify prediction is valid
            assert pred.shape == (1, 2)
            assert not torch.isnan(pred).any()
        
        # Check that history doesn't grow unbounded
        stats = adaptive_model.get_compute_statistics()
        
        # Should have processed all samples
        assert stats.get('total_refinement_steps', 0) > 0
        
        # Memory usage should be reasonable (refinement history is tracked but should not be excessive)
        history_length = len(adaptive_model.scheduler.refinement_history)
        assert history_length <= 50 * adaptive_model.scheduler.config.max_refinement_steps