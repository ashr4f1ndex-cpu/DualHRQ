"""
Adaptive Learning System for DualHRQ

This module implements continuous learning loops that track:
- Import resolution patterns and failures  
- Computational efficiency over time
- Feature engineering effectiveness
- Model performance adaptation
- Error recovery strategies

The system learns from each execution and adapts its strategies
for maximum resilience and performance.
"""

import json
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class LearningEvent:
    """Single learning event record."""
    timestamp: datetime
    event_type: str  # 'import_failure', 'performance_metric', 'adaptation', 'error'
    context: Dict[str, Any]
    resolution: Optional[str] = None
    success: bool = False
    
class AdaptiveLearningSystem:
    """
    Continuous learning system that adapts strategies based on experience.
    
    This system tracks:
    1. Import resolution patterns - learns which imports work in different environments
    2. Performance patterns - tracks computational efficiency over time  
    3. Error recovery - learns from failures and builds resilience
    4. Feature effectiveness - adapts feature engineering based on results
    """
    
    def __init__(self, learning_dir: str = "adaptive_learning", max_events: int = 10000):
        self.learning_dir = Path(learning_dir)
        self.learning_dir.mkdir(exist_ok=True)
        self.max_events = max_events
        
        # In-memory learning state
        self.events: List[LearningEvent] = []
        self.import_patterns: Dict[str, List[str]] = {}
        self.performance_history: List[Dict] = []
        self.adaptation_strategies: Dict[str, Any] = {}
        
        # Learning parameters
        self.learning_rate = 0.1
        self.adaptation_threshold = 0.05
        self.confidence_threshold = 0.7
        
        # Load existing learning state
        self._load_learning_state()
        
        logger.info(f"Adaptive Learning System initialized with {len(self.events)} historical events")
    
    def record_import_attempt(self, module_name: str, import_path: str, success: bool, 
                            error_msg: Optional[str] = None):
        """Record an import attempt and its outcome."""
        event = LearningEvent(
            timestamp=datetime.now(),
            event_type='import_attempt',
            context={
                'module_name': module_name,
                'import_path': import_path,
                'error_message': error_msg
            },
            success=success
        )
        
        self._add_event(event)
        
        # Update import patterns
        if module_name not in self.import_patterns:
            self.import_patterns[module_name] = []
        
        if success and import_path not in self.import_patterns[module_name]:
            self.import_patterns[module_name].insert(0, import_path)  # Prioritize successful paths
        
        # Learn from failure patterns
        if not success:
            self._learn_from_import_failure(module_name, import_path, error_msg)
    
    def get_suggested_import_order(self, module_name: str) -> List[str]:
        """Get suggested import paths in order of likelihood to succeed."""
        if module_name in self.import_patterns:
            # Return paths ordered by historical success
            return self.import_patterns[module_name].copy()
        
        # Fallback to common patterns
        return self._generate_fallback_import_patterns(module_name)
    
    def record_performance_metric(self, metric_name: str, value: float, context: Dict[str, Any]):
        """Record a performance metric for adaptive optimization."""
        event = LearningEvent(
            timestamp=datetime.now(),
            event_type='performance_metric',
            context={
                'metric_name': metric_name,
                'value': value,
                **context
            },
            success=True
        )
        
        self._add_event(event)
        
        # Track performance trends
        self.performance_history.append({
            'timestamp': datetime.now(),
            'metric_name': metric_name,
            'value': value,
            **context
        })
        
        # Trigger adaptation if needed
        self._check_performance_adaptation(metric_name, value, context)
    
    def record_adaptation(self, adaptation_type: str, old_value: Any, new_value: Any, 
                         reasoning: str):
        """Record an adaptation made by the system."""
        event = LearningEvent(
            timestamp=datetime.now(),
            event_type='adaptation',
            context={
                'adaptation_type': adaptation_type,
                'old_value': str(old_value),
                'new_value': str(new_value),
                'reasoning': reasoning
            },
            success=True
        )
        
        self._add_event(event)
        
        # Update adaptation strategies
        self.adaptation_strategies[adaptation_type] = {
            'last_adaptation': datetime.now(),
            'old_value': old_value,
            'new_value': new_value,
            'reasoning': reasoning
        }
    
    def record_error_recovery(self, error_type: str, recovery_strategy: str, success: bool):
        """Record an error recovery attempt."""
        event = LearningEvent(
            timestamp=datetime.now(),
            event_type='error_recovery',
            context={
                'error_type': error_type,
                'recovery_strategy': recovery_strategy
            },
            success=success
        )
        
        self._add_event(event)
        
        if success:
            self._learn_successful_recovery(error_type, recovery_strategy)
    
    def get_error_recovery_strategy(self, error_type: str) -> Optional[str]:
        """Get the best known recovery strategy for an error type."""
        # Find successful recovery strategies for this error type
        successful_recoveries = [
            event for event in self.events 
            if event.event_type == 'error_recovery' 
            and event.context.get('error_type') == error_type 
            and event.success
        ]
        
        if successful_recoveries:
            # Return most recent successful strategy
            latest = max(successful_recoveries, key=lambda e: e.timestamp)
            return latest.context.get('recovery_strategy')
        
        return None
    
    def adapt_computation_strategy(self, current_efficiency: float, target_efficiency: float) -> Dict[str, Any]:
        """Adapt computational strategy based on efficiency metrics."""
        adaptation_needed = abs(current_efficiency - target_efficiency) > self.adaptation_threshold
        
        if not adaptation_needed:
            return {'adapted': False, 'reason': 'Efficiency within acceptable range'}
        
        # Analyze historical performance
        recent_performance = self._get_recent_performance('computational_efficiency')
        
        adaptations = {}
        
        if current_efficiency < target_efficiency:
            # Need to improve efficiency
            if self._performance_trending_down(recent_performance):
                # Reduce computational complexity
                adaptations['act_threshold'] = 'increase'
                adaptations['max_segments'] = 'decrease'
                adaptations['batch_size'] = 'decrease'
            else:
                # Fine-tune parameters
                adaptations['learning_rate'] = 'decrease'
                adaptations['early_stopping'] = 'enable'
        else:
            # Efficiency is good, can be more aggressive
            adaptations['act_threshold'] = 'decrease'
            adaptations['max_segments'] = 'increase' 
            adaptations['exploration_rate'] = 'increase'
        
        # Record the adaptation
        self.record_adaptation(
            'computational_strategy',
            {'efficiency': current_efficiency},
            adaptations,
            f"Adapting to achieve target efficiency {target_efficiency}"
        )
        
        return {'adapted': True, 'adaptations': adaptations, 'reason': 'Efficiency optimization'}
    
    def get_learning_insights(self) -> Dict[str, Any]:
        """Get insights from the learning system."""
        insights = {
            'total_events': len(self.events),
            'import_success_rate': self._calculate_import_success_rate(),
            'adaptation_frequency': self._calculate_adaptation_frequency(),
            'top_error_types': self._get_top_error_types(),
            'performance_trends': self._analyze_performance_trends(),
            'confidence_score': self._calculate_confidence_score()
        }
        
        return insights
    
    def save_learning_state(self):
        """Save current learning state to disk."""
        try:
            # Save events
            events_file = self.learning_dir / "events.json"
            with open(events_file, 'w') as f:
                json.dump([asdict(event) for event in self.events], f, default=str, indent=2)
            
            # Save patterns
            patterns_file = self.learning_dir / "import_patterns.json"
            with open(patterns_file, 'w') as f:
                json.dump(self.import_patterns, f, indent=2)
            
            # Save performance history
            perf_file = self.learning_dir / "performance_history.json"
            with open(perf_file, 'w') as f:
                json.dump(self.performance_history, f, default=str, indent=2)
            
            # Save adaptation strategies
            adapt_file = self.learning_dir / "adaptation_strategies.json"  
            with open(adapt_file, 'w') as f:
                json.dump(self.adaptation_strategies, f, default=str, indent=2)
            
            logger.info(f"Learning state saved to {self.learning_dir}")
            
        except Exception as e:
            logger.warning(f"Failed to save learning state: {e}")
    
    def _load_learning_state(self):
        """Load existing learning state from disk."""
        try:
            # Load events
            events_file = self.learning_dir / "events.json" 
            if events_file.exists():
                with open(events_file, 'r') as f:
                    events_data = json.load(f)
                    self.events = [LearningEvent(**event) for event in events_data[-self.max_events:]]
            
            # Load import patterns
            patterns_file = self.learning_dir / "import_patterns.json"
            if patterns_file.exists():
                with open(patterns_file, 'r') as f:
                    self.import_patterns = json.load(f)
            
            # Load performance history
            perf_file = self.learning_dir / "performance_history.json"
            if perf_file.exists():
                with open(perf_file, 'r') as f:
                    self.performance_history = json.load(f)
            
            # Load adaptation strategies
            adapt_file = self.learning_dir / "adaptation_strategies.json"
            if adapt_file.exists():
                with open(adapt_file, 'r') as f:
                    self.adaptation_strategies = json.load(f)
            
            logger.info("Learning state loaded successfully")
            
        except Exception as e:
            logger.warning(f"Failed to load learning state: {e}")
    
    def _add_event(self, event: LearningEvent):
        """Add event to learning history."""
        self.events.append(event)
        
        # Keep only recent events
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events:]
    
    def _learn_from_import_failure(self, module_name: str, import_path: str, error_msg: Optional[str]):
        """Learn from import failure patterns."""
        # Common failure patterns and their solutions
        if error_msg and "No module named" in error_msg:
            # Module path issue - suggest alternatives
            alternatives = self._generate_alternative_import_paths(module_name, import_path)
            for alt in alternatives:
                if alt not in self.import_patterns.get(module_name, []):
                    self.import_patterns.setdefault(module_name, []).append(alt)
    
    def _generate_fallback_import_patterns(self, module_name: str) -> List[str]:
        """Generate fallback import patterns for unknown modules."""
        patterns = [
            f"src.{module_name}",
            f"lab_v10.src.{module_name}", 
            f"{module_name}",
            f"common.{module_name}",
            f"options.{module_name}"
        ]
        return patterns
    
    def _generate_alternative_import_paths(self, module_name: str, failed_path: str) -> List[str]:
        """Generate alternative import paths based on failure analysis."""
        alternatives = []
        
        # If absolute path failed, try relative
        if failed_path.startswith('lab_v10.'):
            alternatives.append(failed_path.replace('lab_v10.', ''))
        
        # Try different base paths
        base_name = module_name.split('.')[-1]
        alternatives.extend([
            f"src.{base_name}",
            f"common.{base_name}",
            f"options.{base_name}",
            f"portfolio.{base_name}",
            f"models.{base_name}"
        ])
        
        return alternatives
    
    def _check_performance_adaptation(self, metric_name: str, value: float, context: Dict):
        """Check if performance adaptation is needed."""
        recent_values = [
            p['value'] for p in self.performance_history[-10:] 
            if p['metric_name'] == metric_name
        ]
        
        if len(recent_values) >= 5:
            trend = np.polyfit(range(len(recent_values)), recent_values, 1)[0]
            
            # If performance is declining, trigger adaptation
            if trend < -self.adaptation_threshold:
                self._trigger_performance_adaptation(metric_name, trend, context)
    
    def _trigger_performance_adaptation(self, metric_name: str, trend: float, context: Dict):
        """Trigger performance-based adaptation."""
        adaptation_strategy = f"Declining {metric_name} trend detected: {trend:.4f}"
        
        # Record adaptation trigger
        self.record_adaptation(
            f'{metric_name}_optimization',
            'declining_performance',
            'adaptive_tuning', 
            adaptation_strategy
        )
    
    def _get_recent_performance(self, metric_name: str, window: int = 20) -> List[Dict]:
        """Get recent performance data for a metric."""
        return [
            p for p in self.performance_history[-window:] 
            if p['metric_name'] == metric_name
        ]
    
    def _performance_trending_down(self, performance_data: List[Dict]) -> bool:
        """Check if performance is trending downward."""
        if len(performance_data) < 3:
            return False
        
        values = [p['value'] for p in performance_data]
        trend = np.polyfit(range(len(values)), values, 1)[0]
        return trend < -0.01  # Threshold for downward trend
    
    def _calculate_import_success_rate(self) -> float:
        """Calculate overall import success rate."""
        import_events = [e for e in self.events if e.event_type == 'import_attempt']
        if not import_events:
            return 1.0
        
        successful = sum(1 for e in import_events if e.success)
        return successful / len(import_events)
    
    def _calculate_adaptation_frequency(self) -> float:
        """Calculate adaptation frequency (adaptations per day)."""
        adaptation_events = [e for e in self.events if e.event_type == 'adaptation']
        if not adaptation_events:
            return 0.0
        
        if len(self.events) == 0:
            return 0.0
        
        time_span = (self.events[-1].timestamp - self.events[0].timestamp).days or 1
        return len(adaptation_events) / time_span
    
    def _get_top_error_types(self, limit: int = 5) -> List[Tuple[str, int]]:
        """Get most common error types."""
        error_events = [e for e in self.events if e.event_type == 'error_recovery']
        error_counts = {}
        
        for event in error_events:
            error_type = event.context.get('error_type', 'unknown')
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        return sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:limit]
    
    def _analyze_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends across metrics."""
        trends = {}
        
        # Group by metric name
        metrics = {}
        for perf in self.performance_history:
            metric_name = perf['metric_name']
            if metric_name not in metrics:
                metrics[metric_name] = []
            metrics[metric_name].append(perf['value'])
        
        # Calculate trends
        for metric_name, values in metrics.items():
            if len(values) >= 3:
                trend = np.polyfit(range(len(values)), values, 1)[0]
                trends[metric_name] = {
                    'trend': trend,
                    'direction': 'improving' if trend > 0 else 'declining',
                    'latest_value': values[-1],
                    'sample_count': len(values)
                }
        
        return trends
    
    def _calculate_confidence_score(self) -> float:
        """Calculate overall system confidence score."""
        factors = []
        
        # Import success rate factor
        import_success = self._calculate_import_success_rate()
        factors.append(import_success * 0.3)
        
        # Adaptation effectiveness factor
        adaptations = [e for e in self.events if e.event_type == 'adaptation']
        if adaptations:
            adaptation_score = len(adaptations) / (len(adaptations) + 1)  # More adaptations = higher score
            factors.append(min(adaptation_score, 1.0) * 0.2)
        
        # Performance stability factor
        performance_variance = self._calculate_performance_variance()
        stability_score = 1.0 / (1.0 + performance_variance)  # Lower variance = higher score
        factors.append(stability_score * 0.3)
        
        # Recovery success rate
        recovery_events = [e for e in self.events if e.event_type == 'error_recovery']
        if recovery_events:
            recovery_success = sum(1 for e in recovery_events if e.success) / len(recovery_events)
            factors.append(recovery_success * 0.2)
        else:
            factors.append(0.8)  # No errors is good
        
        return sum(factors)
    
    def _calculate_performance_variance(self) -> float:
        """Calculate variance in performance metrics."""
        if not self.performance_history:
            return 0.0
        
        values = [p['value'] for p in self.performance_history]
        if len(values) < 2:
            return 0.0
        
        return float(np.var(values))
    
    def _learn_successful_recovery(self, error_type: str, recovery_strategy: str):
        """Learn from successful error recovery."""
        # Update success strategies
        success_key = f"recovery_{error_type}"
        if success_key not in self.adaptation_strategies:
            self.adaptation_strategies[success_key] = []
        
        strategy_info = {
            'strategy': recovery_strategy,
            'timestamp': datetime.now(),
            'success': True
        }
        
        self.adaptation_strategies[success_key].append(strategy_info)

# Global learning system instance
_learning_system = None

def get_learning_system() -> AdaptiveLearningSystem:
    """Get the global adaptive learning system instance."""
    global _learning_system
    if _learning_system is None:
        _learning_system = AdaptiveLearningSystem()
    return _learning_system