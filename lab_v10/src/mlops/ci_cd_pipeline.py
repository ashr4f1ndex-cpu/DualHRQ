"""
CI/CD Pipeline for Production ML Systems

Advanced continuous integration and deployment:
- Automated model validation and testing
- Performance regression detection
- A/B testing framework
- Canary deployments with rollback
- Integration with existing GitHub Actions
"""

import os
import subprocess
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
import shutil
import tempfile

import torch
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class ModelValidationResult:
    """Result of model validation tests."""
    model_name: str
    version: str
    passed: bool
    test_results: Dict[str, Any]
    performance_metrics: Dict[str, float]
    regression_analysis: Dict[str, Any]
    timestamp: str
    
    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class DeploymentConfig:
    """Deployment configuration."""
    model_name: str
    version: str
    environment: str  # 'staging', 'production'
    deployment_type: str  # 'blue_green', 'canary', 'rolling'
    rollback_threshold: float
    monitoring_duration_hours: int
    traffic_percentage: float = 100.0

class ModelValidator:
    """Comprehensive model validation for production deployment."""
    
    def __init__(self, validation_config: Dict[str, Any] = None):
        self.validation_config = validation_config or {}
        self.baseline_models = {}
        
    def validate_model_architecture(self, model: torch.nn.Module, 
                                  expected_config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate model architecture matches specifications."""
        
        validation_results = {
            'architecture_valid': True,
            'parameter_count_valid': True,
            'layer_structure_valid': True,
            'issues': []
        }
        
        # Check parameter count
        actual_params = sum(p.numel() for p in model.parameters())
        expected_params = expected_config.get('expected_parameters', 0)
        
        if expected_params > 0:
            param_tolerance = expected_config.get('parameter_tolerance', 0.05)
            param_diff_pct = abs(actual_params - expected_params) / expected_params
            
            if param_diff_pct > param_tolerance:
                validation_results['parameter_count_valid'] = False
                validation_results['issues'].append(
                    f"Parameter count mismatch: expected {expected_params:,}, got {actual_params:,}"
                )
        
        # Check layer structure
        expected_layers = expected_config.get('expected_layers', [])
        if expected_layers:
            actual_layers = [name for name, _ in model.named_modules()]
            
            for expected_layer in expected_layers:
                if expected_layer not in actual_layers:
                    validation_results['layer_structure_valid'] = False
                    validation_results['issues'].append(f"Missing expected layer: {expected_layer}")
        
        # Overall validation status
        validation_results['architecture_valid'] = (
            validation_results['parameter_count_valid'] and 
            validation_results['layer_structure_valid']
        )
        
        return validation_results
    
    def validate_model_performance(self, model: torch.nn.Module, 
                                 validation_data: torch.utils.data.DataLoader,
                                 baseline_metrics: Dict[str, float] = None) -> Dict[str, Any]:
        """Validate model performance against baseline."""
        
        model.eval()
        total_loss = 0
        num_batches = 0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch in validation_data:
                inputs, batch_targets = batch
                outputs = model(inputs)
                
                # Assuming MSE loss for simplicity
                loss = torch.nn.functional.mse_loss(outputs, batch_targets)
                total_loss += loss.item()
                num_batches += 1
                
                predictions.extend(outputs.cpu().numpy().flatten())
                targets.extend(batch_targets.cpu().numpy().flatten())
        
        # Calculate metrics
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        mse = np.mean((predictions - targets) ** 2)
        mae = np.mean(np.abs(predictions - targets))
        correlation = np.corrcoef(predictions, targets)[0, 1] if len(predictions) > 1 else 0
        
        current_metrics = {
            'mse': mse,
            'mae': mae,
            'correlation': correlation,
            'avg_loss': avg_loss
        }
        
        # Compare against baseline
        performance_regression = False
        regression_details = {}
        
        if baseline_metrics:
            for metric_name, baseline_value in baseline_metrics.items():
                if metric_name in current_metrics:
                    current_value = current_metrics[metric_name]
                    
                    # For loss metrics, higher is worse
                    if metric_name in ['mse', 'mae', 'avg_loss']:
                        if current_value > baseline_value * 1.05:  # 5% tolerance
                            performance_regression = True
                            regression_details[metric_name] = {
                                'baseline': baseline_value,
                                'current': current_value,
                                'regression_pct': ((current_value - baseline_value) / baseline_value) * 100
                            }
                    
                    # For correlation, higher is better
                    elif metric_name == 'correlation':
                        if current_value < baseline_value * 0.95:  # 5% tolerance
                            performance_regression = True
                            regression_details[metric_name] = {
                                'baseline': baseline_value,
                                'current': current_value,
                                'regression_pct': ((baseline_value - current_value) / baseline_value) * 100
                            }
        
        return {
            'performance_valid': not performance_regression,
            'metrics': current_metrics,
            'baseline_comparison': regression_details,
            'regression_detected': performance_regression
        }
    
    def validate_model_stability(self, model: torch.nn.Module,
                               test_inputs: torch.Tensor, num_runs: int = 5) -> Dict[str, Any]:
        """Validate model output stability across multiple runs."""
        
        model.eval()
        outputs = []
        
        with torch.no_grad():
            for _ in range(num_runs):
                output = model(test_inputs)
                outputs.append(output.cpu().numpy())
        
        # Calculate stability metrics
        outputs = np.array(outputs)  # Shape: (num_runs, batch_size, output_dim)
        
        # Standard deviation across runs
        output_std = np.std(outputs, axis=0)
        max_std = np.max(output_std)
        mean_std = np.mean(output_std)
        
        # Coefficient of variation
        output_mean = np.mean(outputs, axis=0)
        cv = output_std / (np.abs(output_mean) + 1e-8)
        max_cv = np.max(cv)
        
        stability_threshold = 1e-6  # Very small threshold for deterministic models
        
        return {
            'stability_valid': max_std < stability_threshold,
            'max_std_deviation': float(max_std),
            'mean_std_deviation': float(mean_std),
            'max_coefficient_variation': float(max_cv),
            'stability_threshold': stability_threshold
        }
    
    def run_comprehensive_validation(self, model: torch.nn.Module,
                                   validation_data: torch.utils.data.DataLoader,
                                   model_config: Dict[str, Any],
                                   baseline_metrics: Dict[str, float] = None) -> ModelValidationResult:
        """Run complete model validation suite."""
        
        model_name = model_config.get('model_name', 'unknown')
        version = model_config.get('version', 'unknown')
        
        # Architecture validation
        arch_results = self.validate_model_architecture(model, model_config)
        
        # Performance validation
        perf_results = self.validate_model_performance(model, validation_data, baseline_metrics)
        
        # Stability validation
        test_batch = next(iter(validation_data))
        test_inputs = test_batch[0][:1]  # Single sample for stability test
        stability_results = self.validate_model_stability(model, test_inputs)
        
        # Overall validation result
        all_valid = (
            arch_results['architecture_valid'] and
            perf_results['performance_valid'] and
            stability_results['stability_valid']
        )
        
        validation_result = ModelValidationResult(
            model_name=model_name,
            version=version,
            passed=all_valid,
            test_results={
                'architecture': arch_results,
                'performance': perf_results,
                'stability': stability_results
            },
            performance_metrics=perf_results['metrics'],
            regression_analysis=perf_results['baseline_comparison'],
            timestamp=pd.Timestamp.now().isoformat()
        )
        
        return validation_result

class ABTestingFramework:
    """A/B testing framework for model deployment."""
    
    def __init__(self):
        self.active_tests = {}
        self.test_results = {}
        
    def create_ab_test(self, test_id: str, model_a: str, model_b: str,
                      traffic_split: float = 0.5, duration_days: int = 7) -> Dict[str, Any]:
        """Create new A/B test configuration."""
        
        test_config = {
            'test_id': test_id,
            'model_a': model_a,  # Control
            'model_b': model_b,  # Treatment
            'traffic_split': traffic_split,  # Percentage going to model_b
            'duration_days': duration_days,
            'start_time': pd.Timestamp.now(),
            'end_time': pd.Timestamp.now() + pd.Timedelta(days=duration_days),
            'metrics': {
                'model_a': {'requests': 0, 'errors': 0, 'latency_sum': 0, 'performance_sum': 0},
                'model_b': {'requests': 0, 'errors': 0, 'latency_sum': 0, 'performance_sum': 0}
            }
        }
        
        self.active_tests[test_id] = test_config
        logger.info(f"Created A/B test: {test_id} ({model_a} vs {model_b})")
        
        return test_config
    
    def route_request(self, test_id: str, user_id: str) -> str:
        """Route request to appropriate model based on A/B test configuration."""
        
        if test_id not in self.active_tests:
            return "model_a"  # Default to control
        
        test_config = self.active_tests[test_id]
        
        # Check if test is still active
        if pd.Timestamp.now() > test_config['end_time']:
            return "model_a"  # Test expired, use control
        
        # Deterministic routing based on user_id hash
        import hashlib
        user_hash = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        if (user_hash % 100) < (test_config['traffic_split'] * 100):
            return "model_b"
        else:
            return "model_a"
    
    def record_request_result(self, test_id: str, model_used: str,
                            latency: float, performance_score: float = None,
                            error_occurred: bool = False) -> None:
        """Record result of A/B test request."""
        
        if test_id not in self.active_tests:
            return
        
        metrics = self.active_tests[test_id]['metrics'][model_used]
        metrics['requests'] += 1
        metrics['latency_sum'] += latency
        
        if error_occurred:
            metrics['errors'] += 1
        
        if performance_score is not None:
            metrics['performance_sum'] += performance_score
    
    def analyze_ab_test(self, test_id: str) -> Dict[str, Any]:
        """Analyze A/B test results and determine winner."""
        
        if test_id not in self.active_tests:
            raise ValueError(f"Test {test_id} not found")
        
        test_config = self.active_tests[test_id]
        metrics_a = test_config['metrics']['model_a']
        metrics_b = test_config['metrics']['model_b']
        
        # Calculate summary statistics
        def calculate_stats(metrics):
            requests = metrics['requests']
            if requests == 0:
                return {'error_rate': 0, 'avg_latency': 0, 'avg_performance': 0}
            
            return {
                'error_rate': metrics['errors'] / requests,
                'avg_latency': metrics['latency_sum'] / requests,
                'avg_performance': metrics['performance_sum'] / requests if metrics['performance_sum'] > 0 else 0
            }
        
        stats_a = calculate_stats(metrics_a)
        stats_b = calculate_stats(metrics_b)
        
        # Simple winner determination (can be enhanced with statistical tests)
        winner = "model_a"  # Default to control
        
        # Model B wins if it has lower error rate AND lower latency AND higher performance
        if (stats_b['error_rate'] <= stats_a['error_rate'] and
            stats_b['avg_latency'] < stats_a['avg_latency'] * 1.1 and  # 10% latency tolerance
            stats_b['avg_performance'] >= stats_a['avg_performance']):
            winner = "model_b"
        
        analysis_result = {
            'test_id': test_id,
            'winner': winner,
            'model_a_stats': stats_a,
            'model_b_stats': stats_b,
            'total_requests': metrics_a['requests'] + metrics_b['requests'],
            'statistical_significance': 'not_calculated',  # Would implement proper statistical tests
            'recommendation': f"Deploy {winner} to production" if winner == "model_b" else "Keep current model"
        }
        
        self.test_results[test_id] = analysis_result
        return analysis_result

class CanaryDeploymentManager:
    """Canary deployment with automated rollback."""
    
    def __init__(self):
        self.active_deployments = {}
        self.rollback_triggers = {
            'error_rate_threshold': 0.05,  # 5%
            'latency_p95_threshold_ms': 1000,
            'performance_degradation_threshold': 0.1  # 10%
        }
    
    def start_canary_deployment(self, deployment_config: DeploymentConfig) -> str:
        """Start canary deployment with monitoring."""
        
        deployment_id = f"{deployment_config.model_name}_{deployment_config.version}"
        
        canary_state = {
            'config': deployment_config,
            'start_time': pd.Timestamp.now(),
            'current_traffic_pct': min(deployment_config.traffic_percentage, 5.0),  # Start with 5%
            'metrics': {
                'requests': 0,
                'errors': 0,
                'latencies': [],
                'performance_scores': []
            },
            'status': 'active',
            'rollback_triggered': False
        }
        
        self.active_deployments[deployment_id] = canary_state
        
        logger.info(f"Started canary deployment: {deployment_id} with {canary_state['current_traffic_pct']:.1f}% traffic")
        
        return deployment_id
    
    def record_canary_request(self, deployment_id: str, latency_ms: float,
                            performance_score: float = None, error_occurred: bool = False) -> None:
        """Record canary deployment request result."""
        
        if deployment_id not in self.active_deployments:
            return
        
        canary = self.active_deployments[deployment_id]
        metrics = canary['metrics']
        
        metrics['requests'] += 1
        metrics['latencies'].append(latency_ms)
        
        if error_occurred:
            metrics['errors'] += 1
        
        if performance_score is not None:
            metrics['performance_scores'].append(performance_score)
        
        # Check rollback conditions
        self._check_rollback_conditions(deployment_id)
    
    def _check_rollback_conditions(self, deployment_id: str) -> bool:
        """Check if canary deployment should be rolled back."""
        
        canary = self.active_deployments[deployment_id]
        metrics = canary['metrics']
        
        if metrics['requests'] < 100:  # Need minimum requests for reliable statistics
            return False
        
        # Error rate check
        error_rate = metrics['errors'] / metrics['requests']
        if error_rate > self.rollback_triggers['error_rate_threshold']:
            self._trigger_rollback(deployment_id, f"High error rate: {error_rate:.3f}")
            return True
        
        # Latency check
        if metrics['latencies']:
            p95_latency = np.percentile(metrics['latencies'], 95)
            if p95_latency > self.rollback_triggers['latency_p95_threshold_ms']:
                self._trigger_rollback(deployment_id, f"High P95 latency: {p95_latency:.1f}ms")
                return True
        
        # Performance degradation check
        if metrics['performance_scores'] and len(metrics['performance_scores']) > 50:
            recent_performance = np.mean(metrics['performance_scores'][-50:])
            early_performance = np.mean(metrics['performance_scores'][:50])
            
            if early_performance > 0:
                performance_degradation = (early_performance - recent_performance) / early_performance
                if performance_degradation > self.rollback_triggers['performance_degradation_threshold']:
                    self._trigger_rollback(deployment_id, f"Performance degradation: {performance_degradation:.3f}")
                    return True
        
        return False
    
    def _trigger_rollback(self, deployment_id: str, reason: str) -> None:
        """Trigger automatic rollback of canary deployment."""
        
        canary = self.active_deployments[deployment_id]
        canary['status'] = 'rolled_back'
        canary['rollback_triggered'] = True
        canary['rollback_reason'] = reason
        canary['rollback_time'] = pd.Timestamp.now()
        
        logger.error(f"ROLLBACK TRIGGERED for {deployment_id}: {reason}")
        
        # In production, this would trigger actual infrastructure changes
        # For now, just log and mark as rolled back
    
    def promote_canary(self, deployment_id: str) -> bool:
        """Promote successful canary to full production."""
        
        if deployment_id not in self.active_deployments:
            return False
        
        canary = self.active_deployments[deployment_id]
        
        if canary['rollback_triggered']:
            logger.error(f"Cannot promote rolled back deployment: {deployment_id}")
            return False
        
        # Check if deployment has been running long enough
        runtime = pd.Timestamp.now() - canary['start_time']
        min_runtime = pd.Timedelta(hours=canary['config'].monitoring_duration_hours)
        
        if runtime < min_runtime:
            logger.warning(f"Deployment {deployment_id} needs more monitoring time")
            return False
        
        # Mark as promoted
        canary['status'] = 'promoted'
        canary['promotion_time'] = pd.Timestamp.now()
        
        logger.info(f"Successfully promoted canary deployment: {deployment_id}")
        return True

class CICDPipeline:
    """Complete CI/CD pipeline orchestration."""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config = self._load_config(config_path)
        self.validator = ModelValidator()
        self.ab_framework = ABTestingFramework()
        self.canary_manager = CanaryDeploymentManager()
        
    def _load_config(self, config_path: Optional[Path]) -> Dict[str, Any]:
        """Load CI/CD configuration."""
        
        default_config = {
            'validation': {
                'architecture_validation': True,
                'performance_validation': True,
                'stability_validation': True
            },
            'deployment': {
                'staging_required': True,
                'canary_traffic_start_pct': 5.0,
                'canary_ramp_up_hours': 24,
                'monitoring_duration_hours': 48
            },
            'rollback': {
                'auto_rollback_enabled': True,
                'error_rate_threshold': 0.05,
                'latency_threshold_ms': 1000
            }
        }
        
        if config_path and config_path.exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
            # Merge with defaults
            default_config.update(user_config)
        
        return default_config
    
    def run_ci_pipeline(self, model_path: Path, model_config: Dict[str, Any],
                       validation_data_path: Path) -> ModelValidationResult:
        """Run continuous integration pipeline."""
        
        logger.info("Starting CI pipeline...")
        
        # Load model
        model_class = self._get_model_class(model_config)
        model = model_class()
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        
        # Load validation data
        validation_data = self._load_validation_data(validation_data_path)
        
        # Run validation
        validation_result = self.validator.run_comprehensive_validation(
            model, validation_data, model_config
        )
        
        # Save validation report
        report_path = model_path.parent / "validation_report.json"
        with open(report_path, 'w') as f:
            json.dump(validation_result.to_dict(), f, indent=2, default=str)
        
        logger.info(f"CI pipeline completed. Validation {'PASSED' if validation_result.passed else 'FAILED'}")
        
        return validation_result
    
    def run_cd_pipeline(self, validation_result: ModelValidationResult,
                       deployment_config: DeploymentConfig) -> str:
        """Run continuous deployment pipeline."""
        
        if not validation_result.passed:
            raise ValueError("Cannot deploy model that failed validation")
        
        logger.info(f"Starting CD pipeline for {deployment_config.model_name} v{deployment_config.version}")
        
        if deployment_config.deployment_type == 'canary':
            deployment_id = self.canary_manager.start_canary_deployment(deployment_config)
            return deployment_id
        else:
            # Other deployment types would be implemented here
            logger.info("Non-canary deployment not implemented yet")
            return "deployment_completed"
    
    def _get_model_class(self, model_config: Dict[str, Any]):
        """Get model class from configuration."""
        # This would be implemented to dynamically load model classes
        # For now, return a dummy class
        return torch.nn.Linear(10, 1)  # Placeholder
    
    def _load_validation_data(self, data_path: Path) -> torch.utils.data.DataLoader:
        """Load validation data for testing."""
        # This would load actual validation data
        # For now, return dummy data
        dummy_data = torch.utils.data.TensorDataset(
            torch.randn(100, 10),
            torch.randn(100, 1)
        )
        return torch.utils.data.DataLoader(dummy_data, batch_size=32)
    
    def run_full_pipeline(self, model_path: Path, model_config: Dict[str, Any],
                         validation_data_path: Path, deployment_config: DeploymentConfig) -> Dict[str, Any]:
        """Run complete CI/CD pipeline."""
        
        try:
            # CI Pipeline
            validation_result = self.run_ci_pipeline(model_path, model_config, validation_data_path)
            
            # CD Pipeline (only if CI passes)
            if validation_result.passed:
                deployment_id = self.run_cd_pipeline(validation_result, deployment_config)
                
                return {
                    'success': True,
                    'validation_result': validation_result.to_dict(),
                    'deployment_id': deployment_id,
                    'message': 'Pipeline completed successfully'
                }
            else:
                return {
                    'success': False,
                    'validation_result': validation_result.to_dict(),
                    'deployment_id': None,
                    'message': 'Deployment blocked due to validation failures'
                }
                
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'message': 'Pipeline failed with error'
            }