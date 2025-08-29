"""
feature_flags.py - Feature Flags System
========================================

Comprehensive feature flags system for DualHRQ 2.0 with A/B testing framework,
rollout controller, performance isolation, and feature gating.

CRITICAL: Safe rollout, performance isolation, A/B testing framework.
"""

import hashlib
import json
import time
from collections import defaultdict
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Set, Tuple
import numpy as np
from scipy import stats
import threading
import uuid


class FeatureFlagManager:
    """Feature flag management system with comprehensive targeting and rollout capabilities."""
    
    def __init__(self):
        self._lock = threading.Lock()
        self.flags: Dict[str, bool] = {}
        self.rollout_percentages: Dict[str, int] = {}
        self.user_targets: Dict[str, Set[str]] = defaultdict(set)
        self.group_targets: Dict[str, Set[str]] = defaultdict(set)
        self.scheduled_features: Dict[str, datetime] = {}
        self.time_windows: Dict[str, Tuple[datetime, datetime]] = {}
        self.conditions: Dict[str, Callable] = {}
        self.dependencies: Dict[str, str] = {}
        
    def _hash_user(self, feature_name: str, user_id: str) -> float:
        """Create deterministic hash for user assignment."""
        combined = f"{feature_name}:{user_id}"
        hash_obj = hashlib.md5(combined.encode('utf-8'))
        hash_int = int(hash_obj.hexdigest(), 16)
        return (hash_int % 10000) / 100.0  # Return percentage 0-99.99
    
    def is_enabled(self, feature_name: str) -> bool:
        """Check if feature is enabled globally."""
        with self._lock:
            # Check basic flag
            if feature_name in self.flags:
                enabled = self.flags[feature_name]
            else:
                enabled = False
            
            # Check dependencies
            if enabled and feature_name in self.dependencies:
                prerequisite = self.dependencies[feature_name]
                if not self.is_enabled(prerequisite):
                    return False
            
            # Check time-based scheduling
            if feature_name in self.scheduled_features:
                now = datetime.now()
                if now < self.scheduled_features[feature_name]:
                    return False
                    
            # Check time windows
            if feature_name in self.time_windows:
                now = datetime.now()
                start_time, end_time = self.time_windows[feature_name]
                if not (start_time <= now <= end_time):
                    return False
                    
            return enabled
    
    def enable_feature(self, feature_name: str) -> None:
        """Enable a feature."""
        with self._lock:
            self.flags[feature_name] = True
    
    def disable_feature(self, feature_name: str) -> None:
        """Disable a feature."""
        with self._lock:
            self.flags[feature_name] = False
    
    def enable_features(self, feature_names: List[str]) -> None:
        """Enable multiple features."""
        with self._lock:
            for name in feature_names:
                self.flags[name] = True
    
    def set_rollout_percentage(self, feature_name: str, percentage: int) -> None:
        """Set percentage-based rollout."""
        with self._lock:
            self.rollout_percentages[feature_name] = max(0, min(100, percentage))
    
    def is_enabled_for_user(self, feature_name: str, user_id: str) -> bool:
        """Check if feature is enabled for specific user."""
        with self._lock:
            # Check user targeting first - this can override global settings
            if feature_name in self.user_targets and user_id in self.user_targets[feature_name]:
                # Still need to check time-based constraints and dependencies for targeted users
                if feature_name in self.scheduled_features:
                    now = datetime.now()
                    if now < self.scheduled_features[feature_name]:
                        return False
                        
                if feature_name in self.time_windows:
                    now = datetime.now()
                    start_time, end_time = self.time_windows[feature_name]
                    if not (start_time <= now <= end_time):
                        return False
                
                # Check dependencies
                if feature_name in self.dependencies:
                    prerequisite = self.dependencies[feature_name]
                    if prerequisite not in self.flags or not self.flags[prerequisite]:
                        return False
                
                return True
            
            # If user targeting is set but user not in list, return False (unless percentage rollout)
            if feature_name in self.user_targets and feature_name not in self.rollout_percentages:
                return False
            
            # If feature has percentage rollout, use that logic
            if feature_name in self.rollout_percentages:
                # Still check time-based constraints
                if feature_name in self.scheduled_features:
                    now = datetime.now()
                    if now < self.scheduled_features[feature_name]:
                        return False
                        
                if feature_name in self.time_windows:
                    now = datetime.now()
                    start_time, end_time = self.time_windows[feature_name]
                    if not (start_time <= now <= end_time):
                        return False
                
                # Check dependencies
                if feature_name in self.dependencies:
                    prerequisite = self.dependencies[feature_name]
                    if prerequisite not in self.flags or not self.flags[prerequisite]:
                        return False
                
                # Apply percentage rollout
                user_hash = self._hash_user(feature_name, user_id)
                return user_hash < self.rollout_percentages[feature_name]
            
            # Check if feature is explicitly enabled/disabled
            if feature_name not in self.flags:
                return False  # Not enabled by default
                
            if not self.flags[feature_name]:
                return False
            
            # Check time-based scheduling
            if feature_name in self.scheduled_features:
                now = datetime.now()
                if now < self.scheduled_features[feature_name]:
                    return False
                    
            # Check time windows
            if feature_name in self.time_windows:
                now = datetime.now()
                start_time, end_time = self.time_windows[feature_name]
                if not (start_time <= now <= end_time):
                    return False
            
            # Check dependencies (avoid infinite recursion by checking basic flags only)
            if feature_name in self.dependencies:
                prerequisite = self.dependencies[feature_name]
                if prerequisite not in self.flags or not self.flags[prerequisite]:
                    return False
                
            return True  # Feature is enabled globally
    
    def set_user_targets(self, feature_name: str, target_users: List[str]) -> None:
        """Set user targeting."""
        with self._lock:
            self.user_targets[feature_name] = set(target_users)
    
    def set_group_targets(self, feature_name: str, target_groups: List[str]) -> None:
        """Set group targeting."""
        with self._lock:
            self.group_targets[feature_name] = set(target_groups)
    
    def is_enabled_for_group(self, feature_name: str, group: str) -> bool:
        """Check if feature is enabled for group."""
        with self._lock:
            if not self.is_enabled(feature_name):
                return False
            return group in self.group_targets.get(feature_name, set())
    
    def schedule_enable(self, feature_name: str, enable_time: datetime) -> None:
        """Schedule feature enabling."""
        with self._lock:
            self.scheduled_features[feature_name] = enable_time
            self.flags[feature_name] = True  # Enable but time-gated
    
    def set_time_window(self, feature_name: str, start_time: datetime, end_time: datetime) -> None:
        """Set time-limited feature window."""
        with self._lock:
            self.time_windows[feature_name] = (start_time, end_time)
            self.flags[feature_name] = True  # Enable but time-gated
    
    def set_condition(self, feature_name: str, condition_func: Callable) -> None:
        """Set conditional feature enabling."""
        with self._lock:
            self.conditions[feature_name] = condition_func
    
    def is_enabled_with_context(self, feature_name: str, context: Dict[str, Any]) -> bool:
        """Check if feature is enabled with context."""
        with self._lock:
            if not self.is_enabled(feature_name):
                return False
                
            if feature_name in self.conditions:
                try:
                    return self.conditions[feature_name](context)
                except Exception:
                    return False
                    
            return True
    
    def set_dependency(self, feature_name: str, prerequisite: str) -> None:
        """Set feature dependency."""
        with self._lock:
            self.dependencies[feature_name] = prerequisite


@dataclass
class ExperimentConfig:
    """A/B test experiment configuration."""
    name: str
    variants: Dict[str, float]
    start_date: datetime
    end_date: datetime
    metrics: Optional[List[str]] = None
    sample_size: Optional[int] = None
    min_effect_size: Optional[float] = None
    early_stop_confidence: Optional[float] = None
    significance_level: float = 0.05
    
    def __post_init__(self):
        """Validate configuration."""
        # Ensure variant probabilities sum to ~1.0
        total = sum(self.variants.values())
        if abs(total - 1.0) > 0.001:
            # Normalize probabilities
            self.variants = {k: v/total for k, v in self.variants.items()}


class ABTestManager:
    """A/B testing framework with statistical significance testing and early stopping."""
    
    def __init__(self, min_sample_size: int = 100, early_stopping: bool = False):
        self.min_sample_size = min_sample_size
        self.early_stopping = early_stopping
        self.experiments: Dict[str, ExperimentConfig] = {}
        self.metrics_data: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
        self.user_assignments: Dict[str, Dict[str, str]] = defaultdict(dict)
        self.experiment_status: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'stopped_early': False, 'is_significant': False, 'stop_reason': None
        })
        self._lock = threading.Lock()
    
    def _hash_user_variant(self, experiment_name: str, user_id: str) -> str:
        """Deterministically assign user to variant based on hash."""
        if experiment_name not in self.experiments:
            return 'control'
            
        # Check if user already assigned
        if user_id in self.user_assignments[experiment_name]:
            return self.user_assignments[experiment_name][user_id]
            
        config = self.experiments[experiment_name]
        variants = list(config.variants.keys())
        probabilities = list(config.variants.values())
        
        # Create deterministic hash
        combined = f"{experiment_name}:{user_id}"
        hash_obj = hashlib.md5(combined.encode('utf-8'))
        hash_value = int(hash_obj.hexdigest(), 16) / (16**32)  # Normalize to 0-1
        
        # Assign based on cumulative probabilities
        cumulative = 0
        for variant, prob in zip(variants, probabilities):
            cumulative += prob
            if hash_value <= cumulative:
                self.user_assignments[experiment_name][user_id] = variant
                return variant
                
        # Fallback to last variant
        variant = variants[-1]
        self.user_assignments[experiment_name][user_id] = variant
        return variant
    
    def create_experiment(self, config: ExperimentConfig) -> None:
        """Create A/B test experiment."""
        with self._lock:
            self.experiments[config.name] = config
            self.metrics_data[config.name] = defaultdict(list)
            self.user_assignments[config.name] = {}
    
    def get_variant(self, experiment_name: str, user_id: str) -> str:
        """Get user's variant assignment."""
        with self._lock:
            return self._hash_user_variant(experiment_name, user_id)
    
    def record_metrics(self, experiment_name: str, user_id: str, metrics: Dict[str, float]) -> None:
        """Record experiment metrics."""
        if experiment_name not in self.experiments:
            return
            
        with self._lock:
            variant = self.get_variant(experiment_name, user_id)
            
            metric_record = {
                'user_id': user_id,
                'timestamp': datetime.now(),
                'metrics': metrics.copy()
            }
            
            self.metrics_data[experiment_name][variant].append(metric_record)
    
    def analyze_experiment(self, experiment_name: str) -> Dict[str, Any]:
        """Analyze experiment results."""
        if experiment_name not in self.experiments:
            return {}
            
        with self._lock:
            config = self.experiments[experiment_name]
            results = {}
            
            for variant in config.variants.keys():
                variant_data = self.metrics_data[experiment_name][variant]
                if not variant_data:
                    results[variant] = {'sample_size': 0}
                    continue
                
                # Aggregate metrics
                variant_metrics = {}
                sample_size = len(variant_data)
                
                # Calculate means for each metric
                if config.metrics:
                    for metric in config.metrics:
                        values = []
                        for record in variant_data:
                            if metric in record['metrics']:
                                values.append(record['metrics'][metric])
                        
                        if values:
                            variant_metrics[metric] = {
                                'mean': np.mean(values),
                                'std': np.std(values),
                                'count': len(values)
                            }
                else:
                    # Analyze all available metrics
                    all_metrics = set()
                    for record in variant_data:
                        all_metrics.update(record['metrics'].keys())
                    
                    for metric in all_metrics:
                        values = []
                        for record in variant_data:
                            if metric in record['metrics']:
                                values.append(record['metrics'][metric])
                        
                        if values:
                            variant_metrics[metric] = {
                                'mean': np.mean(values),
                                'std': np.std(values),
                                'count': len(values)
                            }
                
                results[variant] = {
                    'sample_size': sample_size,
                    **variant_metrics
                }
            
            return results
    
    def test_significance(self, experiment_name: str, metric_name: str) -> Dict[str, Any]:
        """Test statistical significance between variants."""
        if experiment_name not in self.experiments:
            return {'p_value': 1.0, 'confidence_interval': (0, 0), 'is_significant': False}
        
        with self._lock:
            config = self.experiments[experiment_name]
            variants = list(config.variants.keys())
            
            if len(variants) < 2:
                return {'p_value': 1.0, 'confidence_interval': (0, 0), 'is_significant': False}
            
            # Get data for all variants
            variant_data = {}
            for variant in variants:
                values = []
                for record in self.metrics_data[experiment_name][variant]:
                    if metric_name in record['metrics']:
                        values.append(record['metrics'][metric_name])
                variant_data[variant] = values
            
            # Perform pairwise t-tests (simplified - using first two variants)
            variant_names = list(variant_data.keys())
            if len(variant_names) >= 2:
                group1 = variant_data[variant_names[0]]
                group2 = variant_data[variant_names[1]]
                
                if len(group1) >= 2 and len(group2) >= 2:
                    try:
                        t_stat, p_value = stats.ttest_ind(group1, group2)
                        
                        # Calculate confidence interval for difference in means
                        mean1, mean2 = np.mean(group1), np.mean(group2)
                        diff = mean2 - mean1
                        pooled_se = np.sqrt(np.var(group1)/len(group1) + np.var(group2)/len(group2))
                        
                        # 95% confidence interval
                        t_crit = stats.t.ppf(0.975, len(group1) + len(group2) - 2)
                        margin_error = t_crit * pooled_se
                        conf_interval = (diff - margin_error, diff + margin_error)
                        
                        is_significant = p_value < config.significance_level
                        
                        return {
                            'p_value': p_value,
                            'confidence_interval': conf_interval,
                            'is_significant': is_significant,
                            't_statistic': t_stat,
                            'mean_difference': diff
                        }
                    except Exception:
                        pass
            
            return {'p_value': 1.0, 'confidence_interval': (0, 0), 'is_significant': False}
    
    def should_stop_early(self, experiment_name: str) -> bool:
        """Check if experiment should stop early."""
        if not self.early_stopping or experiment_name not in self.experiments:
            return False
            
        with self._lock:
            config = self.experiments[experiment_name]
            
            # Check if we have minimum sample size
            total_samples = sum(
                len(self.metrics_data[experiment_name][variant]) 
                for variant in config.variants.keys()
            )
            
            if total_samples < self.min_sample_size:
                return False
            
            # Check primary metric for significance
            if config.metrics and len(config.metrics) > 0:
                primary_metric = config.metrics[0]
                sig_result = self.test_significance(experiment_name, primary_metric)
                
                if sig_result['is_significant']:
                    # Check if confidence level is high enough
                    if config.early_stop_confidence:
                        confidence = 1 - sig_result['p_value']
                        if confidence >= config.early_stop_confidence:
                            self.experiment_status[experiment_name]['stopped_early'] = True
                            self.experiment_status[experiment_name]['stop_reason'] = 'early_significance'
                            return True
                    else:
                        self.experiment_status[experiment_name]['stopped_early'] = True
                        self.experiment_status[experiment_name]['stop_reason'] = 'significance_detected'
                        return True
            
            return False
    
    def get_experiment_status(self, experiment_name: str) -> Dict[str, Any]:
        """Get experiment status."""
        if experiment_name not in self.experiments:
            return {'stopped_early': False, 'is_significant': False}
            
        with self._lock:
            status = self.experiment_status[experiment_name].copy()
            
            # Check if experiment is significant
            if not status['is_significant']:
                config = self.experiments[experiment_name]
                if config.metrics and len(config.metrics) > 0:
                    sig_result = self.test_significance(experiment_name, config.metrics[0])
                    status['is_significant'] = sig_result['is_significant']
            
            return status


class FeatureGate:
    """Feature gate for conditional feature access with performance-based gating."""
    
    def __init__(self, feature_name: str, performance_based: bool = False):
        self.feature_name = feature_name
        self.performance_based = performance_based
        self.variants: Dict[str, float] = {}
        self.rollout_percentage: int = 0
        self.user_overrides: Dict[str, bool] = {}
        self.performance_thresholds: Dict[str, float] = {}
        self.prerequisite_gate: Optional['FeatureGate'] = None
        self._lock = threading.Lock()
    
    def _hash_user(self, user_id: str) -> float:
        """Create deterministic hash for user assignment."""
        combined = f"{self.feature_name}:{user_id}"
        hash_obj = hashlib.md5(combined.encode('utf-8'))
        hash_int = int(hash_obj.hexdigest(), 16)
        return (hash_int % 10000) / 100.0  # Return percentage 0-99.99
    
    def set_variants(self, variants: Dict[str, float]) -> None:
        """Set feature variants with their probabilities."""
        with self._lock:
            # Normalize probabilities
            total = sum(variants.values())
            if total > 0:
                self.variants = {k: v/total for k, v in variants.items()}
            else:
                self.variants = variants
    
    def get_variant(self, user_id: str) -> str:
        """Get user's feature variant."""
        with self._lock:
            if not self.variants:
                return 'default'
            
            # Check prerequisites first
            if self.prerequisite_gate and not self.prerequisite_gate.is_enabled(user_id):
                return list(self.variants.keys())[0]  # Return first variant but user won't be enabled
            
            variants = list(self.variants.keys())
            probabilities = list(self.variants.values())
            
            # Create deterministic assignment
            user_hash = self._hash_user(user_id)
            
            cumulative = 0
            for variant, prob in zip(variants, probabilities):
                cumulative += prob * 100  # Convert to percentage
                if user_hash <= cumulative:
                    return variant
                    
            return variants[-1]  # Fallback to last variant
    
    def set_rollout_percentage(self, percentage: int) -> None:
        """Set rollout percentage."""
        with self._lock:
            self.rollout_percentage = max(0, min(100, percentage))
    
    def is_enabled(self, user_id: str) -> bool:
        """Check if feature is enabled for user."""
        with self._lock:
            # Check user overrides first
            if user_id in self.user_overrides:
                return self.user_overrides[user_id]
            
            # Check prerequisites
            if self.prerequisite_gate and not self.prerequisite_gate.is_enabled(user_id):
                return False
            
            # Check rollout percentage
            if self.rollout_percentage > 0:
                user_hash = self._hash_user(user_id)
                return user_hash < self.rollout_percentage
                
            return False
    
    def enable_for_user(self, user_id: str) -> None:
        """Enable feature for specific user."""
        with self._lock:
            self.user_overrides[user_id] = True
    
    def disable_for_user(self, user_id: str) -> None:
        """Disable feature for specific user."""
        with self._lock:
            self.user_overrides[user_id] = False
    
    def set_performance_thresholds(self, thresholds: Dict[str, float]) -> None:
        """Set performance-based thresholds."""
        with self._lock:
            self.performance_thresholds = thresholds.copy()
            self.performance_based = True
    
    def is_enabled_with_performance(self, user_id: str, performance: Dict[str, float]) -> bool:
        """Check if enabled based on performance."""
        with self._lock:
            # First check basic enablement
            if not self.is_enabled(user_id):
                return False
            
            # If not performance-based, return basic result
            if not self.performance_based or not self.performance_thresholds:
                return True
            
            # Check all performance thresholds
            for metric, threshold in self.performance_thresholds.items():
                if metric not in performance:
                    continue
                
                value = performance[metric]
                
                # Handle different threshold types based on metric name
                if 'min_' in metric.lower() or 'success' in metric.lower():
                    # Higher is better
                    if value < threshold:
                        return False
                elif 'max_' in metric.lower() or 'drawdown' in metric.lower() or 'error' in metric.lower():
                    # Lower is better (note: max_drawdown is typically negative)
                    if metric.lower() == 'max_drawdown_threshold' and value < threshold:
                        return False
                    elif 'max_' in metric.lower() and value > threshold:
                        return False
                else:
                    # Default: assume higher is better
                    if value < threshold:
                        return False
            
            return True
    
    def set_prerequisite(self, prerequisite_gate: 'FeatureGate') -> None:
        """Set prerequisite gate."""
        with self._lock:
            self.prerequisite_gate = prerequisite_gate


class RolloutController:
    """Safe rollout controller with circuit breakers and automatic rollback."""
    
    def __init__(self, feature: str, stages: List[int] = None, stage_duration: timedelta = None,
                 success_threshold: float = 0.95, circuit_breaker_threshold: float = 0.05,
                 auto_rollback: bool = False, rollback_threshold: Dict[str, float] = None,
                 monitoring_interval: timedelta = None):
        self.feature = feature
        self.stages = stages or [10, 25, 50, 100]
        self.stage_duration = stage_duration
        self.success_threshold = success_threshold
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.auto_rollback = auto_rollback
        self.rollback_threshold = rollback_threshold or {'error_rate': 0.15, 'performance_impact': 0.5}
        self.monitoring_interval = monitoring_interval
        
        self.current_stage_index = 0
        self.stage_start_time: Optional[datetime] = None
        self.rollout_started = False
        self.is_halted = False
        self.halt_reason: Optional[str] = None
        self.is_rolled_back = False
        
        # Metrics tracking
        self.current_metrics: Dict[str, float] = {}
        self.time_series_data: List[Dict[str, Any]] = []
        self.circuit_breaker_triggered = False
        
        self._lock = threading.Lock()
    
    def start_rollout(self) -> None:
        """Start gradual rollout."""
        with self._lock:
            self.rollout_started = True
            self.stage_start_time = datetime.now()
            self.current_stage_index = 0
            self.is_halted = False
            self.halt_reason = None
            self.is_rolled_back = False
            self.circuit_breaker_triggered = False
    
    def get_current_stage(self) -> Dict[str, Any]:
        """Get current rollout stage."""
        with self._lock:
            if not self.rollout_started or self.current_stage_index >= len(self.stages):
                return {'percentage': 0, 'stage_index': 0}
                
            return {
                'percentage': self.stages[self.current_stage_index],
                'stage_index': self.current_stage_index,
                'stage_start_time': self.stage_start_time
            }
    
    def record_stage_metrics(self, metrics: Dict[str, float]) -> None:
        """Record metrics for current stage."""
        with self._lock:
            self.current_metrics.update(metrics)
            
            # Check circuit breaker conditions
            error_rate = metrics.get('error_rate', 0)
            if error_rate > self.circuit_breaker_threshold:
                self.circuit_breaker_triggered = True
                self.is_halted = True
                self.halt_reason = 'circuit_breaker'
            
            # Check rollback conditions if auto-rollback enabled
            if self.auto_rollback:
                should_rollback = False
                for metric, threshold in self.rollback_threshold.items():
                    if metric in metrics and metrics[metric] > threshold:
                        should_rollback = True
                        break
                
                if should_rollback:
                    self.execute_rollback()
    
    def can_advance_stage(self) -> bool:
        """Check if can advance to next stage."""
        with self._lock:
            if not self.rollout_started or self.is_halted or self.is_rolled_back:
                return False
                
            if self.current_stage_index >= len(self.stages) - 1:
                return False
            
            # Check success threshold
            success_rate = self.current_metrics.get('success_rate', 0)
            if success_rate < self.success_threshold:
                return False
            
            # Check stage duration if specified
            if self.stage_duration and self.stage_start_time:
                elapsed = datetime.now() - self.stage_start_time
                if elapsed < self.stage_duration:
                    return False
            
            return True
    
    def advance_stage(self) -> None:
        """Advance to next rollout stage."""
        with self._lock:
            if self.can_advance_stage():
                self.current_stage_index += 1
                self.stage_start_time = datetime.now()
                self.current_metrics = {}  # Reset metrics for new stage
    
    def advance_to_stage(self, stage_index: int) -> None:
        """Advance to specific stage."""
        with self._lock:
            if not self.is_halted and not self.is_rolled_back:
                self.current_stage_index = min(stage_index, len(self.stages) - 1)
                self.stage_start_time = datetime.now()
                self.current_metrics = {}
    
    def is_circuit_breaker_triggered(self) -> bool:
        """Check if circuit breaker is triggered."""
        with self._lock:
            return self.circuit_breaker_triggered
    
    def get_rollout_status(self) -> Dict[str, Any]:
        """Get rollout status."""
        with self._lock:
            return {
                'started': self.rollout_started,
                'halted': self.is_halted,
                'reason': self.halt_reason,
                'rolled_back': self.is_rolled_back,
                'current_stage': self.current_stage_index,
                'current_percentage': self.get_current_rollout_percentage(),
                'circuit_breaker_triggered': self.circuit_breaker_triggered
            }
    
    def should_rollback(self) -> bool:
        """Check if should rollback."""
        with self._lock:
            if not self.auto_rollback or not self.current_metrics:
                return False
            
            for metric, threshold in self.rollback_threshold.items():
                if metric in self.current_metrics:
                    if self.current_metrics[metric] > threshold:
                        return True
            
            return False
    
    def execute_rollback(self) -> None:
        """Execute rollback."""
        with self._lock:
            self.is_rolled_back = True
            self.current_stage_index = 0
            self.is_halted = True
            self.halt_reason = 'rollback'
    
    def get_current_rollout_percentage(self) -> int:
        """Get current rollout percentage."""
        with self._lock:
            if self.is_rolled_back:
                return 0
            if not self.rollout_started or self.current_stage_index >= len(self.stages):
                return 0
            return self.stages[self.current_stage_index]
    
    def record_time_series_metrics(self, metrics: Dict[str, Any]) -> None:
        """Record time series metrics."""
        with self._lock:
            self.time_series_data.append(metrics.copy())
            
            # Keep only recent data (last 100 entries)
            if len(self.time_series_data) > 100:
                self.time_series_data = self.time_series_data[-100:]
    
    def analyze_trends(self) -> Dict[str, float]:
        """Analyze metric trends."""
        with self._lock:
            if len(self.time_series_data) < 2:
                return {}
            
            trends = {}
            
            # Analyze trends for numeric metrics
            for key in ['success_rate', 'latency', 'error_rate']:
                values = []
                for data_point in self.time_series_data:
                    if key in data_point and isinstance(data_point[key], (int, float)):
                        values.append(data_point[key])
                
                if len(values) >= 2:
                    # Simple linear trend calculation
                    x = np.arange(len(values))
                    if len(values) > 1 and np.std(x) > 0:
                        slope, _ = np.polyfit(x, values, 1)
                        trends[f'{key}_trend'] = slope
                    else:
                        trends[f'{key}_trend'] = 0.0
            
            return trends


class PerformanceIsolator:
    """Performance isolation during experiments with resource limits and quality gates."""
    
    def __init__(self, baseline_variant: str = 'control'):
        self.baseline_variant = baseline_variant
        self.resource_limits: Dict[str, Dict[str, int]] = {}
        self.performance_data: Dict[str, List[Dict[str, float]]] = defaultdict(list)
        self.experiments: Dict[str, List[str]] = {}
        self.quality_gates: Dict[str, float] = {}
        self._lock = threading.Lock()
    
    def set_resource_limits(self, limits: Dict[str, Dict[str, int]]) -> None:
        """Set resource limits for each variant."""
        with self._lock:
            self.resource_limits = limits.copy()
    
    def get_allocated_resources(self, variant: str) -> Dict[str, int]:
        """Get allocated resources for variant."""
        with self._lock:
            return self.resource_limits.get(variant, {})
    
    def is_within_limits(self, variant: str, usage: Dict[str, float]) -> bool:
        """Check if usage is within limits."""
        with self._lock:
            if variant not in self.resource_limits:
                return True  # No limits set
            
            limits = self.resource_limits[variant]
            
            for resource, limit in limits.items():
                usage_key = f'{resource.replace("_cores", "_usage").replace("_gb", "_usage")}'
                if usage_key in usage:
                    if usage[usage_key] > limit:
                        return False
            
            return True
    
    def record_performance(self, variant: str, metrics: Dict[str, float]) -> None:
        """Record performance metrics for a variant."""
        with self._lock:
            performance_record = {
                'timestamp': datetime.now(),
                **metrics
            }
            self.performance_data[variant].append(performance_record)
            
            # Keep only recent data (last 1000 entries per variant)
            if len(self.performance_data[variant]) > 1000:
                self.performance_data[variant] = self.performance_data[variant][-1000:]
    
    def calculate_performance_impact(self) -> Dict[str, float]:
        """Calculate performance impact relative to baseline."""
        with self._lock:
            if self.baseline_variant not in self.performance_data:
                return {}
            
            baseline_data = self.performance_data[self.baseline_variant]
            if not baseline_data:
                return {}
            
            # Calculate baseline averages
            baseline_metrics = {}
            for metric in ['latency_ms', 'throughput_rps', 'cpu_utilization', 'memory_usage_mb']:
                values = [record[metric] for record in baseline_data if metric in record]
                if values:
                    baseline_metrics[metric] = np.mean(values)
            
            # Calculate impacts for all variants
            impacts = {}
            for variant, data in self.performance_data.items():
                if variant == self.baseline_variant or not data:
                    continue
                
                variant_impacts = {}
                for metric, baseline_value in baseline_metrics.items():
                    variant_values = [record[metric] for record in data if metric in record]
                    if variant_values and baseline_value > 0:
                        variant_avg = np.mean(variant_values)
                        
                        # Calculate percentage impact
                        if metric in ['latency_ms', 'cpu_utilization', 'memory_usage_mb']:
                            # Higher is worse
                            impact = (variant_avg - baseline_value) / baseline_value
                        else:
                            # Higher is better (throughput)
                            impact = (variant_avg - baseline_value) / baseline_value
                        
                        variant_impacts[f'{metric.replace("_ms", "").replace("_rps", "").replace("_mb", "")}_impact'] = impact
                
                impacts[variant] = variant_impacts
            
            # Flatten for easier access
            flattened = {}
            for variant, variant_impacts in impacts.items():
                for metric, impact in variant_impacts.items():
                    flattened[f'{variant}_{metric}'] = impact
            
            # Also add aggregate impacts
            if len(impacts) > 0:
                # Get impacts for the first non-baseline variant
                first_variant = next(iter(impacts.keys()))
                for metric, impact in impacts[first_variant].items():
                    flattened[metric] = impact
            
            return flattened
    
    def register_experiment(self, experiment_name: str, variants: List[str]) -> None:
        """Register experiment for monitoring."""
        with self._lock:
            self.experiments[experiment_name] = variants.copy()
    
    def detect_interference(self) -> Dict[str, Any]:
        """Detect interference between experiments."""
        with self._lock:
            # Simple interference detection based on performance degradation
            report = {
                'interference_detected': False,
                'affected_experiments': [],
                'performance_anomalies': []
            }
            
            # Check if multiple experiments are running simultaneously
            active_experiments = len(self.experiments)
            if active_experiments <= 1:
                return report
            
            # Look for performance anomalies across variants
            all_variants = set()
            for variants in self.experiments.values():
                all_variants.update(variants)
            
            # Calculate performance statistics for each variant
            for variant in all_variants:
                if variant in self.performance_data and len(self.performance_data[variant]) > 10:
                    # Simple check: if recent performance is significantly worse than earlier
                    recent_data = self.performance_data[variant][-10:]
                    earlier_data = self.performance_data[variant][-50:-10] if len(self.performance_data[variant]) >= 50 else []
                    
                    if earlier_data:
                        for metric in ['latency_ms', 'cpu_utilization']:
                            recent_values = [r[metric] for r in recent_data if metric in r]
                            earlier_values = [r[metric] for r in earlier_data if metric in r]
                            
                            if recent_values and earlier_values:
                                recent_avg = np.mean(recent_values)
                                earlier_avg = np.mean(earlier_values)
                                
                                # If recent performance is >50% worse
                                if recent_avg > earlier_avg * 1.5:
                                    report['interference_detected'] = True
                                    report['performance_anomalies'].append({
                                        'variant': variant,
                                        'metric': metric,
                                        'degradation_pct': ((recent_avg - earlier_avg) / earlier_avg) * 100
                                    })
            
            return report
    
    def set_quality_gates(self, gates: Dict[str, float]) -> None:
        """Set quality gates for experiments."""
        with self._lock:
            self.quality_gates = gates.copy()
    
    def passes_quality_gates(self, metrics: Dict[str, float]) -> bool:
        """Check if metrics pass quality gates."""
        with self._lock:
            if not self.quality_gates:
                return True
            
            for gate_name, threshold in self.quality_gates.items():
                if gate_name in metrics:
                    value = metrics[gate_name]
                    
                    # Handle different types of gates
                    if 'max_' in gate_name.lower():
                        if value > threshold:
                            return False
                    elif 'min_' in gate_name.lower():
                        if value < threshold:
                            return False
                    else:
                        # Default behavior based on metric name
                        if ('increase' in gate_name.lower() or 
                            'error' in gate_name.lower()):
                            if value > threshold:
                                return False
                        elif ('ratio' in gate_name.lower() or 
                              'throughput' in gate_name.lower()):
                            if value < threshold:
                                return False
            
            return True
    
    def evaluate_quality_gates(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Evaluate quality gates and return detailed results."""
        with self._lock:
            result = {
                'passed': True,
                'failures': [],
                'gate_results': {}
            }
            
            if not self.quality_gates:
                return result
            
            for gate_name, threshold in self.quality_gates.items():
                if gate_name in metrics:
                    value = metrics[gate_name]
                    gate_passed = True
                    
                    # Handle different types of gates
                    if 'max_' in gate_name.lower():
                        gate_passed = value <= threshold
                    elif 'min_' in gate_name.lower():
                        gate_passed = value >= threshold
                    else:
                        # Default behavior based on metric name
                        if ('increase' in gate_name.lower() or 
                            'error' in gate_name.lower()):
                            gate_passed = value <= threshold
                        elif ('ratio' in gate_name.lower() or 
                              'throughput' in gate_name.lower()):
                            gate_passed = value >= threshold
                    
                    result['gate_results'][gate_name] = {
                        'passed': gate_passed,
                        'value': value,
                        'threshold': threshold
                    }
                    
                    if not gate_passed:
                        result['passed'] = False
                        result['failures'].append({
                            'gate': gate_name,
                            'value': value,
                            'threshold': threshold,
                            'reason': f'{gate_name} = {value} violates threshold {threshold}'
                        })
                else:
                    # Missing metric
                    result['gate_results'][gate_name] = {
                        'passed': False,
                        'value': None,
                        'threshold': threshold,
                        'reason': 'Metric not provided'
                    }
                    result['passed'] = False
                    result['failures'].append({
                        'gate': gate_name,
                        'value': None,
                        'threshold': threshold,
                        'reason': f'Metric {gate_name} not provided'
                    })
            
            return result


class FeatureFlags:
    """
    Main Feature Flags interface combining all functionality.
    
    This is the primary class for DualHRQ 2.0 feature flag management,
    integrating A/B testing, rollout control, and performance isolation.
    """
    
    def __init__(self):
        self.flag_manager = FeatureFlagManager()
        self.ab_test_manager = ABTestManager()
        self.rollout_controller = RolloutController()
        self.performance_isolator = PerformanceIsolator()
        
        # Initialize core DualHRQ feature flags
        self._init_dualhrq_flags()
    
    def _init_dualhrq_flags(self):
        """Initialize DualHRQ-specific feature flags."""
        dualhrq_flags = [
            'pattern_library_enabled',
            'rag_system_enabled', 
            'hrm_integration_enabled',
            'dynamic_conditioning_enabled',
            'leakage_validation_enabled',
            'statistical_validation_enabled',
            'regulatory_compliance_enabled',
            'paper_trading_enabled',
            'kill_switch_enabled'
        ]
        
        # Enable all core features by default
        for flag in dualhrq_flags:
            self.flag_manager.enable_feature(flag)
    
    # Delegate core functionality
    def is_enabled(self, feature_name: str) -> bool:
        return self.flag_manager.is_enabled(feature_name)
    
    def is_enabled_for_user(self, feature_name: str, user_id: str) -> bool:
        return self.flag_manager.is_enabled_for_user(feature_name, user_id)
    
    def enable_feature(self, feature_name: str) -> None:
        self.flag_manager.enable_feature(feature_name)
    
    def disable_feature(self, feature_name: str) -> None:
        self.flag_manager.disable_feature(feature_name)
    
    def set_rollout_percentage(self, feature_name: str, percentage: int) -> None:
        self.flag_manager.set_rollout_percentage(feature_name, percentage)
    
    def emergency_disable_all(self) -> None:
        """Emergency disable all non-critical features."""
        critical_features = ['kill_switch_enabled', 'regulatory_compliance_enabled']
        
        for flag_name in self.flag_manager.flags:
            if flag_name not in critical_features:
                self.flag_manager.disable_feature(flag_name)
    
    def get_all_flags(self) -> Dict[str, bool]:
        """Get all current feature flag states."""
        return self.flag_manager.flags.copy()
    
    def start_ab_test(self, experiment_name: str, variants: List[str], 
                     traffic_allocation: Dict[str, float]) -> str:
        """Start A/B test."""
        return self.ab_test_manager.create_experiment(
            experiment_name, variants, traffic_allocation
        )
    
    def check_performance_violations(self) -> Dict[str, Any]:
        """Check for performance violations."""
        return self.performance_isolator.check_violations()


# Export all classes
__all__ = ['FeatureFlags', 'FeatureFlagManager', 'ABTestManager', 'RolloutController', 
           'PerformanceIsolator', 'FeatureGate', 'ExperimentConfig']