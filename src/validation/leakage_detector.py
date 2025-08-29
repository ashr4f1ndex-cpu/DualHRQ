"""
leakage_detector.py
==================

Implementation of mutual information leakage detection for DRQ-105.

This module implements:
- KSG mutual information estimator
- Feature leakage detection
- Conditioning output validation
- Statistical significance testing
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from scipy.spatial.distance import cdist
from scipy.special import digamma
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')


class KSGMutualInformation:
    """
    Kraskov-Stögbauer-Grassberger mutual information estimator.
    
    This is the gold standard for MI estimation between continuous and discrete variables.
    """
    
    def __init__(self, k: int = 3):
        """
        Initialize KSG estimator.
        
        Args:
            k: Number of nearest neighbors (typically 3-5)
        """
        self.k = k
    
    def estimate_mi(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        Estimate mutual information I(X; Y) using KSG estimator.
        
        Args:
            X: Continuous features [n_samples, n_features] 
            Y: Discrete labels [n_samples]
            
        Returns:
            Mutual information in bits
        """
        if len(X) != len(Y):
            raise ValueError("X and Y must have same number of samples")
            
        if len(X) < self.k + 1:
            return 0.0  # Not enough samples
            
        # Handle edge cases
        if np.var(Y) == 0:  # Y is constant
            return 0.0
            
        # Ensure X is 2D
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        n_samples, n_features = X.shape
        
        # Convert Y to discrete labels if needed
        if not np.issubdtype(Y.dtype, np.integer):
            label_encoder = LabelEncoder()
            Y = label_encoder.fit_transform(Y)
            
        unique_labels = np.unique(Y)
        n_classes = len(unique_labels)
        
        if n_classes <= 1:
            return 0.0
            
        # Improved KSG Algorithm Implementation  
        # Step 1: For each point, find k-th nearest neighbor distance in joint space
        joint_space = np.column_stack([X, Y.reshape(-1, 1)])
        
        # Use Euclidean distance for continuous features, modified for discrete Y
        nbrs_joint = NearestNeighbors(n_neighbors=min(self.k+1, n_samples), metric='euclidean')
        nbrs_joint.fit(joint_space)
        distances_joint, _ = nbrs_joint.kneighbors(joint_space)
        
        # Get k-th nearest neighbor distances (excluding self)  
        k_actual = min(self.k, distances_joint.shape[1] - 1)
        epsilon = distances_joint[:, k_actual] if distances_joint.shape[1] > k_actual else distances_joint[:, -1]
        
        # Step 2: Count neighbors within epsilon distance in marginal spaces
        mi_sum = 0.0
        valid_points = 0
        
        for i in range(n_samples):
            # Count neighbors in X space within epsilon[i] using Euclidean distance
            x_distances = np.linalg.norm(X - X[i], axis=1)
            x_neighbors = np.sum(x_distances < epsilon[i] + 1e-15) - 1  # Exclude self, add small epsilon
            
            # Count neighbors in Y space (discrete)
            y_neighbors = np.sum(Y == Y[i]) - 1
            
            # Only include points with sufficient neighbors
            if x_neighbors > 0 and y_neighbors > 0:
                mi_sum += digamma(x_neighbors) + digamma(y_neighbors) 
                valid_points += 1
        
        if valid_points == 0:
            return 0.0
            
        # Bias-corrected KSG formula
        mi_nats = digamma(k_actual) - mi_sum / valid_points + digamma(n_samples)
        
        # Apply bias correction for small samples
        bias_correction = 0.0
        if n_samples < 1000:
            # Empirical bias correction based on sample size
            bias_correction = (k_actual - 1) / (2.0 * n_samples)
        
        mi_nats = max(0.0, mi_nats - bias_correction)
        
        # Convert from nats to bits  
        mi_bits = mi_nats / np.log(2)
        
        return max(0.0, mi_bits)  # MI is non-negative


class MutualInformationTester:
    """
    Comprehensive mutual information tester for leakage detection.
    """
    
    def __init__(self, mi_threshold: float = 0.1, k: int = 3, n_bootstrap: int = 100):
        """
        Initialize MI tester.
        
        Args:
            mi_threshold: Threshold for leakage detection (bits)
            k: Number of neighbors for KSG estimator
            n_bootstrap: Bootstrap samples for confidence intervals
        """
        self.mi_threshold = mi_threshold
        self.ksg = KSGMutualInformation(k=k)
        self.n_bootstrap = n_bootstrap
    
    def compute_mutual_information(self, conditioning_output: np.ndarray, puzzle_ids: np.ndarray) -> float:
        """
        Compute mutual information between conditioning output and puzzle IDs.
        
        Args:
            conditioning_output: Model conditioning output [n_samples, n_features]
            puzzle_ids: Puzzle identifiers [n_samples]
            
        Returns:
            Mutual information score in bits
        """
        return self.ksg.estimate_mi(conditioning_output, puzzle_ids)
    
    def test_feature_leakage(self, features: np.ndarray, puzzle_ids: np.ndarray) -> Dict[str, Any]:
        """
        Test feature matrix for puzzle_id leakage.
        
        Args:
            features: Feature matrix [n_samples, n_features]
            puzzle_ids: Puzzle IDs [n_samples]
            
        Returns:
            Dictionary with leakage test results
        """
        if features.ndim == 1:
            features = features.reshape(-1, 1)
            
        n_samples, n_features = features.shape
        mi_scores = []
        
        # Test each feature individually
        for i in range(n_features):
            feature = features[:, i]
            mi_score = self.ksg.estimate_mi(feature.reshape(-1, 1), puzzle_ids)
            mi_scores.append(mi_score)
        
        mi_scores = np.array(mi_scores)
        max_mi_score = np.max(mi_scores)
        features_above_threshold = np.sum(mi_scores > self.mi_threshold)
        
        # Test joint MI of all features
        joint_mi = self.ksg.estimate_mi(features, puzzle_ids)
        
        return {
            'leakage_detected': max_mi_score > self.mi_threshold,
            'max_mi_score': float(max_mi_score),
            'joint_mi_score': float(joint_mi),
            'features_above_threshold': int(features_above_threshold),
            'mi_scores': mi_scores.tolist(),
            'threshold_used': self.mi_threshold,
            'leaky_feature_indices': np.where(mi_scores > self.mi_threshold)[0].tolist()
        }
    
    def test_prediction_leakage(self, predictions: np.ndarray, puzzle_ids: np.ndarray) -> Dict[str, Any]:
        """
        Test model predictions for puzzle_id leakage.
        
        Args:
            predictions: Model predictions [n_samples, n_outputs]
            puzzle_ids: Puzzle IDs [n_samples]
            
        Returns:
            Dictionary with prediction leakage results
        """
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)
            
        mi_score = self.ksg.estimate_mi(predictions, puzzle_ids)
        
        # Bootstrap confidence interval
        bootstrap_scores = []
        for _ in range(min(self.n_bootstrap, 50)):  # Limit for performance
            # Bootstrap sample
            indices = np.random.choice(len(predictions), size=len(predictions), replace=True)
            boot_pred = predictions[indices]
            boot_ids = puzzle_ids[indices]
            
            boot_mi = self.ksg.estimate_mi(boot_pred, boot_ids)
            bootstrap_scores.append(boot_mi)
        
        bootstrap_scores = np.array(bootstrap_scores)
        ci_lower = np.percentile(bootstrap_scores, 5)
        ci_upper = np.percentile(bootstrap_scores, 95)
        
        return {
            'leakage_detected': mi_score > self.mi_threshold,
            'max_mi_score': float(mi_score),
            'mi_score': float(mi_score),
            'threshold_used': self.mi_threshold,
            'bootstrap_ci_lower': float(ci_lower),
            'bootstrap_ci_upper': float(ci_upper),
            'bootstrap_mean': float(np.mean(bootstrap_scores)),
            'bootstrap_std': float(np.std(bootstrap_scores))
        }
    
    def test_conditioning_independence(self, conditioning_output: np.ndarray, puzzle_ids: np.ndarray) -> Dict[str, Any]:
        """
        Test that conditioning system output is independent of puzzle_id.
        
        This is the core test for DRQ-105.
        
        Args:
            conditioning_output: Conditioning system output
            puzzle_ids: Puzzle identifiers
            
        Returns:
            Test results with pass/fail status
        """
        mi_score = self.compute_mutual_information(conditioning_output, puzzle_ids)
        
        # Additional statistical tests
        n_unique_puzzles = len(np.unique(puzzle_ids))
        sample_size = len(puzzle_ids)
        
        # Theoretical maximum MI for uniform distribution
        max_possible_mi = np.log2(n_unique_puzzles)
        
        # Normalized MI score
        normalized_mi = mi_score / max_possible_mi if max_possible_mi > 0 else 0
        
        test_passed = mi_score < self.mi_threshold
        
        return {
            'test_passed': test_passed,
            'test_failed': not test_passed,
            'mi_score': float(mi_score),
            'mi_threshold': self.mi_threshold,
            'normalized_mi': float(normalized_mi),
            'max_possible_mi': float(max_possible_mi),
            'n_unique_puzzles': int(n_unique_puzzles),
            'sample_size': int(sample_size),
            'violation_severity': 'CRITICAL' if mi_score > 0.2 else 'HIGH' if mi_score > 0.1 else 'LOW'
        }


class LeakageValidator:
    """
    High-level leakage validation orchestrator.
    """
    
    def __init__(self, mi_threshold: float = 0.1):
        """
        Initialize leakage validator.
        
        Args:
            mi_threshold: MI threshold for leakage detection
        """
        self.mi_tester = MutualInformationTester(mi_threshold=mi_threshold)
        self.mi_threshold = mi_threshold
    
    def test_conditioning_independence(self, conditioning_output: np.ndarray, puzzle_ids: np.ndarray) -> Dict[str, Any]:
        """
        Test conditioning system independence from puzzle_id.
        
        Returns:
            Complete validation results
        """
        return self.mi_tester.test_conditioning_independence(conditioning_output, puzzle_ids)
    
    def validate_system_integrity(self, 
                                 features: np.ndarray, 
                                 conditioning_output: np.ndarray,
                                 predictions: np.ndarray,
                                 puzzle_ids: np.ndarray) -> Dict[str, Any]:
        """
        Comprehensive system validation for information leakage.
        
        Args:
            features: Input features
            conditioning_output: Conditioning system output  
            predictions: Model predictions
            puzzle_ids: Puzzle identifiers
            
        Returns:
            Complete validation report
        """
        # Test feature leakage
        feature_results = self.mi_tester.test_feature_leakage(features, puzzle_ids)
        
        # Test conditioning leakage
        conditioning_results = self.mi_tester.test_conditioning_independence(conditioning_output, puzzle_ids)
        
        # Test prediction leakage
        prediction_results = self.mi_tester.test_prediction_leakage(predictions, puzzle_ids)
        
        # Overall assessment
        any_leakage = (feature_results['leakage_detected'] or 
                      not conditioning_results['test_passed'] or
                      prediction_results['leakage_detected'])
        
        max_mi_found = max(
            feature_results['max_mi_score'],
            conditioning_results['mi_score'],
            prediction_results['mi_score']
        )
        
        return {
            'system_integrity_passed': not any_leakage,
            'system_integrity_failed': any_leakage,
            'max_mi_score_found': float(max_mi_found),
            'mi_threshold_used': self.mi_threshold,
            'feature_leakage_results': feature_results,
            'conditioning_independence_results': conditioning_results,
            'prediction_leakage_results': prediction_results,
            'critical_violations': max_mi_found > 0.2,
            'recommendations': self._generate_recommendations(feature_results, conditioning_results, prediction_results)
        }
    
    def _generate_recommendations(self, feature_results, conditioning_results, prediction_results) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        if feature_results['leakage_detected']:
            recommendations.append(
                f"FEATURE LEAKAGE: Remove puzzle_id information from features "
                f"(max MI: {feature_results['max_mi_score']:.3f})"
            )
            
        if not conditioning_results['test_passed']:
            recommendations.append(
                f"CONDITIONING LEAKAGE: Conditioning system leaks puzzle_id information "
                f"(MI: {conditioning_results['mi_score']:.3f})"
            )
            
        if prediction_results['leakage_detected']:
            recommendations.append(
                f"PREDICTION LEAKAGE: Model predictions leak puzzle_id information "
                f"(MI: {prediction_results['mi_score']:.3f})"
            )
            
        if not recommendations:
            recommendations.append("✅ No information leakage detected - system integrity validated")
            
        return recommendations


# Export compatibility functions for existing tests
def compute_mutual_information(x: np.ndarray, y: np.ndarray) -> float:
    """Compatibility function for existing tests."""
    ksg = KSGMutualInformation()
    return ksg.estimate_mi(x.reshape(-1, 1) if x.ndim == 1 else x, y)