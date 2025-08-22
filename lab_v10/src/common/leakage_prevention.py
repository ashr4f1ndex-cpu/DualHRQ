"""
Comprehensive leakage prevention framework for dual-book trading strategies.

This module implements advanced cross-validation techniques specifically designed
for financial time series data, including:
- Combinatorial Purged Cross-Validation (CPCV)
- Walk-forward analysis with proper purging
- Embargo periods to prevent look-ahead bias
- Corporate action awareness
- Path-dependent backtesting

The framework ensures that model training and validation respect the temporal
nature of financial data and regulatory constraints.

References:
- López de Prado, M. Advances in Financial Machine Learning
- Bailey, D. H., et al. The Probability of Backtest Overfitting
- Harvey, C. R., et al. ... and the Cross-Section of Expected Returns
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Iterator, Callable
from itertools import combinations
from sklearn.model_selection import BaseCrossValidator
import warnings
from dataclasses import dataclass
from abc import ABC, abstractmethod

warnings.filterwarnings('ignore', category=RuntimeWarning)


@dataclass
class ValidationFold:
    """Container for a single validation fold."""
    train_indices: np.ndarray
    test_indices: np.ndarray
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    purge_period: pd.Timedelta
    embargo_period: pd.Timedelta


class BaseFinancialCV(BaseCrossValidator, ABC):
    """
    Base class for financial cross-validation with leakage prevention.
    """
    
    def __init__(self, 
                 time_col: str = 'timestamp',
                 purge: pd.Timedelta = pd.Timedelta(hours=1),
                 embargo: pd.Timedelta = pd.Timedelta(hours=2),
                 min_train_size: int = 1000,
                 min_test_size: int = 100):
        """
        Initialize base financial cross-validator.
        
        Parameters
        ----------
        time_col : str
            Name of timestamp column
        purge : pd.Timedelta
            Purge period before test set
        embargo : pd.Timedelta
            Embargo period after test set
        min_train_size : int
            Minimum training set size
        min_test_size : int
            Minimum test set size
        """
        self.time_col = time_col
        self.purge = purge
        self.embargo = embargo
        self.min_train_size = min_train_size
        self.min_test_size = min_test_size
    
    def _validate_data(self, X: pd.DataFrame) -> None:
        """Validate input data format."""
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")
        
        if self.time_col not in X.columns:
            raise ValueError(f"Time column '{self.time_col}' not found in DataFrame")
        
        if not pd.api.types.is_datetime64_any_dtype(X[self.time_col]):
            raise ValueError(f"Time column '{self.time_col}' must be datetime type")
    
    @abstractmethod
    def split(self, X: pd.DataFrame, y: Optional[pd.Series] = None, 
              groups: Optional[pd.Series] = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate train/test splits."""
        pass


class CombinatorialPurgedCV(BaseFinancialCV):
    """
    Combinatorial Purged Cross-Validation (CPCV) implementation.
    
    This technique addresses the challenges of traditional k-fold CV in financial
    data by:
    1. Using combinatorial splits to reduce path dependency
    2. Purging training data that temporally overlaps with test data
    3. Adding embargo periods to prevent information leakage
    """
    
    def __init__(self,
                 n_splits: int = 6,
                 n_test_groups: int = 2,
                 time_col: str = 'timestamp',
                 purge: pd.Timedelta = pd.Timedelta(hours=1),
                 embargo: pd.Timedelta = pd.Timedelta(hours=2),
                 min_train_size: int = 1000,
                 min_test_size: int = 100):
        """
        Initialize CPCV.
        
        Parameters
        ----------
        n_splits : int
            Number of groups to create from data
        n_test_groups : int
            Number of groups to use for testing in each fold
        time_col : str
            Name of timestamp column
        purge : pd.Timedelta
            Purge period before test set
        embargo : pd.Timedelta
            Embargo period after test set
        min_train_size : int
            Minimum training set size
        min_test_size : int
            Minimum test set size
        """
        super().__init__(time_col, purge, embargo, min_train_size, min_test_size)
        self.n_splits = n_splits
        self.n_test_groups = n_test_groups
        
        if n_test_groups >= n_splits:
            raise ValueError("n_test_groups must be less than n_splits")
    
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Return number of splitting iterations."""
        from math import comb
        return comb(self.n_splits, self.n_test_groups)
    
    def split(self, X: pd.DataFrame, y: Optional[pd.Series] = None, 
              groups: Optional[pd.Series] = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate CPCV splits with purging and embargo.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix with timestamp column
        y : pd.Series, optional
            Target variable (not used in splitting logic)
        groups : pd.Series, optional
            Group labels (not used)
            
        Yields
        ------
        train_indices, test_indices : tuple of arrays
            Training and testing indices for each fold
        """
        self._validate_data(X)
        
        # Sort by time
        X_sorted = X.sort_values(self.time_col).reset_index(drop=True)
        n_samples = len(X_sorted)
        
        # Create time-based groups
        group_size = n_samples // self.n_splits
        groups_list = []
        
        for i in range(self.n_splits):
            start_idx = i * group_size
            end_idx = (i + 1) * group_size if i < self.n_splits - 1 else n_samples
            groups_list.append(np.arange(start_idx, end_idx))
        
        # Generate combinatorial splits
        group_indices = np.arange(self.n_splits)
        
        for test_group_combo in combinations(group_indices, self.n_test_groups):
            # Create test set from selected groups
            test_indices = np.concatenate([groups_list[i] for i in test_group_combo])
            
            # Create train set from remaining groups
            train_group_indices = [i for i in group_indices if i not in test_group_combo]
            train_indices = np.concatenate([groups_list[i] for i in train_group_indices])
            
            # Apply purging and embargo
            train_indices, test_indices = self._apply_purge_embargo(
                X_sorted, train_indices, test_indices
            )
            
            # Check minimum sizes
            if len(train_indices) >= self.min_train_size and len(test_indices) >= self.min_test_size:
                yield train_indices, test_indices
    
    def _apply_purge_embargo(self, 
                            X: pd.DataFrame, 
                            train_indices: np.ndarray, 
                            test_indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply purging and embargo to training indices.
        
        Parameters
        ----------
        X : pd.DataFrame
            Sorted DataFrame with timestamp column
        train_indices : np.ndarray
            Initial training indices
        test_indices : np.ndarray
            Test indices
            
        Returns
        -------
        purged_train_indices, test_indices : tuple of arrays
            Purged training indices and unchanged test indices
        """
        if len(test_indices) == 0:
            return train_indices, test_indices
        
        # Get test period boundaries
        test_start = X.iloc[test_indices[0]][self.time_col]
        test_end = X.iloc[test_indices[-1]][self.time_col]
        
        # Calculate purge and embargo periods
        purge_start = test_start - self.purge
        embargo_end = test_end + self.embargo
        
        # Filter training indices
        train_times = X.iloc[train_indices][self.time_col]
        
        # Remove training samples that fall within purge/embargo periods
        valid_train_mask = (train_times < purge_start) | (train_times > embargo_end)
        purged_train_indices = train_indices[valid_train_mask]
        
        return purged_train_indices, test_indices


class WalkForwardCV(BaseFinancialCV):
    """
    Walk-forward cross-validation with expanding or rolling window.
    
    This technique simulates realistic trading conditions by:
    1. Training on historical data only
    2. Testing on future periods
    3. Optionally expanding the training window over time
    """
    
    def __init__(self,
                 initial_train_size: Union[int, pd.Timedelta],
                 test_size: Union[int, pd.Timedelta],
                 step_size: Union[int, pd.Timedelta] = None,
                 expanding_window: bool = True,
                 max_train_size: Union[int, pd.Timedelta] = None,
                 time_col: str = 'timestamp',
                 purge: pd.Timedelta = pd.Timedelta(hours=1),
                 embargo: pd.Timedelta = pd.Timedelta(hours=2)):
        """
        Initialize walk-forward CV.
        
        Parameters
        ----------
        initial_train_size : int or pd.Timedelta
            Initial training window size
        test_size : int or pd.Timedelta
            Test window size
        step_size : int or pd.Timedelta, optional
            Step size between folds (defaults to test_size)
        expanding_window : bool
            If True, expand training window; if False, use rolling window
        max_train_size : int or pd.Timedelta, optional
            Maximum training window size (for expanding window)
        time_col : str
            Name of timestamp column
        purge : pd.Timedelta
            Purge period before test set
        embargo : pd.Timedelta
            Embargo period after test set
        """
        super().__init__(time_col, purge, embargo)
        self.initial_train_size = initial_train_size
        self.test_size = test_size
        self.step_size = step_size or test_size
        self.expanding_window = expanding_window
        self.max_train_size = max_train_size
    
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Estimate number of splits (requires data to be exact)."""
        if X is None:
            return 10  # Default estimate
        
        self._validate_data(X)
        X_sorted = X.sort_values(self.time_col)
        
        # Calculate based on time periods if using time-based windows
        if isinstance(self.initial_train_size, pd.Timedelta):
            total_time = X_sorted[self.time_col].iloc[-1] - X_sorted[self.time_col].iloc[0]
            available_time = total_time - self.initial_train_size
            n_splits = int(available_time / self.step_size)
        else:
            # Index-based calculation
            n_samples = len(X_sorted)
            available_samples = n_samples - self.initial_train_size
            n_splits = available_samples // self.step_size
        
        return max(0, n_splits)
    
    def split(self, X: pd.DataFrame, y: Optional[pd.Series] = None, 
              groups: Optional[pd.Series] = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate walk-forward splits.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix with timestamp column
        y : pd.Series, optional
            Target variable (not used in splitting logic)
        groups : pd.Series, optional
            Group labels (not used)
            
        Yields
        ------
        train_indices, test_indices : tuple of arrays
            Training and testing indices for each fold
        """
        self._validate_data(X)
        
        # Sort by time
        X_sorted = X.sort_values(self.time_col).reset_index(drop=True)
        n_samples = len(X_sorted)
        
        if isinstance(self.initial_train_size, pd.Timedelta):
            yield from self._time_based_split(X_sorted)
        else:
            yield from self._index_based_split(X_sorted, n_samples)
    
    def _time_based_split(self, X: pd.DataFrame) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate time-based walk-forward splits."""
        start_time = X[self.time_col].iloc[0]
        end_time = X[self.time_col].iloc[-1]
        
        current_train_start = start_time
        current_test_start = start_time + self.initial_train_size + self.purge
        
        while current_test_start + self.test_size <= end_time:
            # Define test period
            test_end = current_test_start + self.test_size
            
            # Define training period
            if self.expanding_window:
                train_start = current_train_start
                train_end = current_test_start - self.purge
                
                # Apply max training size if specified
                if self.max_train_size:
                    train_start = max(train_start, train_end - self.max_train_size)
            else:
                # Rolling window
                train_end = current_test_start - self.purge
                train_start = train_end - self.initial_train_size
            
            # Get indices
            train_mask = (X[self.time_col] >= train_start) & (X[self.time_col] < train_end)
            test_mask = (X[self.time_col] >= current_test_start) & (X[self.time_col] < test_end)
            
            train_indices = np.where(train_mask)[0]
            test_indices = np.where(test_mask)[0]
            
            if len(train_indices) >= self.min_train_size and len(test_indices) >= self.min_test_size:
                yield train_indices, test_indices
            
            # Move to next period
            current_test_start += self.step_size
    
    def _index_based_split(self, X: pd.DataFrame, n_samples: int) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate index-based walk-forward splits."""
        current_train_start = 0
        current_test_start = self.initial_train_size
        
        while current_test_start + self.test_size <= n_samples:
            # Define test indices
            test_indices = np.arange(current_test_start, current_test_start + self.test_size)
            
            # Define training indices
            if self.expanding_window:
                train_start = current_train_start
                train_end = current_test_start
                
                # Apply max training size if specified
                if self.max_train_size:
                    train_start = max(train_start, train_end - self.max_train_size)
            else:
                # Rolling window
                train_end = current_test_start
                train_start = train_end - self.initial_train_size
            
            train_indices = np.arange(max(0, train_start), train_end)
            
            # Apply purging based on time
            train_indices, test_indices = self._apply_purge_embargo(X, train_indices, test_indices)
            
            if len(train_indices) >= self.min_train_size and len(test_indices) >= self.min_test_size:
                yield train_indices, test_indices
            
            # Move to next period
            current_test_start += self.step_size


class LeakageAuditor:
    """
    Comprehensive leakage auditing for financial ML pipelines.
    
    This class provides tools to detect various forms of data leakage:
    - Temporal leakage (look-ahead bias)
    - Feature leakage (using future information)
    - Label leakage (target in features)
    - Group leakage (information from test groups in training)
    """
    
    def __init__(self, 
                 time_col: str = 'timestamp',
                 id_col: str = None,
                 label_horizon: pd.Timedelta = pd.Timedelta(minutes=30)):
        """
        Initialize leakage auditor.
        
        Parameters
        ----------
        time_col : str
            Name of timestamp column
        id_col : str, optional
            Name of ID column for group-based checks
        label_horizon : pd.Timedelta
            Forward-looking period for label calculation
        """
        self.time_col = time_col
        self.id_col = id_col
        self.label_horizon = label_horizon
    
    def audit_temporal_leakage(self, 
                              train_data: pd.DataFrame, 
                              test_data: pd.DataFrame,
                              purge_period: pd.Timedelta = pd.Timedelta(0),
                              embargo_period: pd.Timedelta = pd.Timedelta(0)) -> Dict[str, bool]:
        """
        Audit for temporal leakage between train and test sets.
        
        Parameters
        ----------
        train_data : pd.DataFrame
            Training data
        test_data : pd.DataFrame
            Test data
        purge_period : pd.Timedelta
            Expected purge period
        embargo_period : pd.Timedelta
            Expected embargo period
            
        Returns
        -------
        Dict[str, bool]
            Audit results
        """
        results = {}
        
        if len(train_data) == 0 or len(test_data) == 0:
            return {"error": True, "message": "Empty train or test data"}
        
        # Check basic temporal ordering
        train_max_time = train_data[self.time_col].max()
        test_min_time = test_data[self.time_col].min()
        test_max_time = test_data[self.time_col].max()
        
        results['temporal_order_ok'] = train_max_time <= test_min_time
        
        # Check purge period compliance
        if purge_period > pd.Timedelta(0):
            expected_gap = test_min_time - train_max_time
            results['purge_compliant'] = expected_gap >= purge_period
        else:
            results['purge_compliant'] = True
        
        # Check for label leakage (training labels using future information)
        train_label_end_times = train_data[self.time_col] + self.label_horizon
        label_leakage = (train_label_end_times >= test_min_time).any()
        results['no_label_leakage'] = not label_leakage
        
        # Check for overlap in time periods
        train_time_range = set(pd.date_range(
            train_data[self.time_col].min(),
            train_data[self.time_col].max(),
            freq='1min'
        ))
        test_time_range = set(pd.date_range(
            test_data[self.time_col].min(), 
            test_data[self.time_col].max(),
            freq='1min'
        ))
        
        time_overlap = len(train_time_range.intersection(test_time_range))
        results['no_time_overlap'] = time_overlap == 0
        
        return results
    
    def audit_feature_leakage(self, 
                             features: pd.DataFrame,
                             target: pd.Series,
                             correlation_threshold: float = 0.95) -> Dict[str, Union[bool, List[str]]]:
        """
        Audit for feature leakage (features that are too predictive).
        
        Parameters
        ----------
        features : pd.DataFrame
            Feature matrix
        target : pd.Series
            Target variable
        correlation_threshold : float
            Correlation threshold for flagging features
            
        Returns
        -------
        Dict[str, Union[bool, List[str]]]
            Audit results including suspicious features
        """
        results = {}
        suspicious_features = []
        
        # Calculate correlations
        correlations = features.corrwith(target).abs()
        
        # Find features with suspiciously high correlation
        high_corr_features = correlations[correlations > correlation_threshold].index.tolist()
        suspicious_features.extend(high_corr_features)
        
        # Check for features that perfectly predict target
        for col in features.select_dtypes(include=[np.number]).columns:
            if (features[col] == target).all():
                suspicious_features.append(col)
        
        # Check for features with future information (if time column available)
        if self.time_col in features.columns:
            future_features = []
            for col in features.columns:
                if 'future' in col.lower() or 'next' in col.lower() or 'forward' in col.lower():
                    future_features.append(col)
            suspicious_features.extend(future_features)
        
        results['no_feature_leakage'] = len(suspicious_features) == 0
        results['suspicious_features'] = list(set(suspicious_features))
        results['max_correlation'] = float(correlations.max())
        
        return results
    
    def audit_group_leakage(self, 
                           train_data: pd.DataFrame, 
                           test_data: pd.DataFrame) -> Dict[str, bool]:
        """
        Audit for group leakage (same entities in train and test).
        
        Parameters
        ----------
        train_data : pd.DataFrame
            Training data
        test_data : pd.DataFrame
            Test data
            
        Returns
        -------
        Dict[str, bool]
            Audit results
        """
        results = {}
        
        if self.id_col is None or self.id_col not in train_data.columns:
            results['group_leakage_check'] = False
            results['message'] = "No ID column specified for group leakage check"
            return results
        
        train_ids = set(train_data[self.id_col].unique())
        test_ids = set(test_data[self.id_col].unique())
        
        overlapping_ids = train_ids.intersection(test_ids)
        results['no_group_leakage'] = len(overlapping_ids) == 0
        results['overlapping_groups'] = len(overlapping_ids)
        results['group_leakage_check'] = True
        
        return results
    
    def comprehensive_audit(self, 
                           train_data: pd.DataFrame,
                           test_data: pd.DataFrame,
                           train_features: pd.DataFrame,
                           train_target: pd.Series,
                           purge_period: pd.Timedelta = pd.Timedelta(0),
                           embargo_period: pd.Timedelta = pd.Timedelta(0)) -> Dict[str, Dict]:
        """
        Run comprehensive leakage audit.
        
        Parameters
        ----------
        train_data : pd.DataFrame
            Training data with timestamps
        test_data : pd.DataFrame  
            Test data with timestamps
        train_features : pd.DataFrame
            Training features
        train_target : pd.Series
            Training target
        purge_period : pd.Timedelta
            Purge period
        embargo_period : pd.Timedelta
            Embargo period
            
        Returns
        -------
        Dict[str, Dict]
            Comprehensive audit results
        """
        audit_results = {}
        
        # Temporal leakage audit
        audit_results['temporal'] = self.audit_temporal_leakage(
            train_data, test_data, purge_period, embargo_period
        )
        
        # Feature leakage audit
        audit_results['feature'] = self.audit_feature_leakage(
            train_features, train_target
        )
        
        # Group leakage audit
        audit_results['group'] = self.audit_group_leakage(
            train_data, test_data
        )
        
        # Overall assessment
        all_checks_passed = (
            audit_results['temporal'].get('temporal_order_ok', False) and
            audit_results['temporal'].get('purge_compliant', False) and
            audit_results['temporal'].get('no_label_leakage', False) and
            audit_results['feature'].get('no_feature_leakage', False) and
            audit_results['group'].get('no_group_leakage', True)  # Default True if no ID col
        )
        
        audit_results['summary'] = {
            'all_checks_passed': all_checks_passed,
            'audit_timestamp': pd.Timestamp.now(),
            'recommendations': self._generate_recommendations(audit_results)
        }
        
        return audit_results
    
    def _generate_recommendations(self, audit_results: Dict) -> List[str]:
        """Generate recommendations based on audit results."""
        recommendations = []
        
        temporal = audit_results.get('temporal', {})
        feature = audit_results.get('feature', {})
        group = audit_results.get('group', {})
        
        if not temporal.get('temporal_order_ok', True):
            recommendations.append("Ensure training data comes before test data chronologically")
        
        if not temporal.get('purge_compliant', True):
            recommendations.append("Increase purge period to prevent temporal leakage")
        
        if not temporal.get('no_label_leakage', True):
            recommendations.append("Reduce label horizon or increase purge period")
        
        if not feature.get('no_feature_leakage', True):
            suspicious = feature.get('suspicious_features', [])
            recommendations.append(f"Review suspicious features: {suspicious}")
        
        if not group.get('no_group_leakage', True):
            recommendations.append("Ensure no overlapping entities between train/test sets")
        
        if not recommendations:
            recommendations.append("All leakage checks passed - good job!")
        
        return recommendations


def assert_no_leakage_enhanced(df: pd.DataFrame, 
                              time_col: str, 
                              fold_col: str,
                              label_horizon: pd.Timedelta,
                              purge: pd.Timedelta, 
                              embargo: pd.Timedelta,
                              raise_on_violation: bool = True) -> Dict[str, bool]:
    """
    Enhanced version of the existing leakage assertion with detailed reporting.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with time and fold columns
    time_col : str
        Time column name
    fold_col : str
        Fold column name
    label_horizon : pd.Timedelta
        Forward horizon for label calculation
    purge : pd.Timedelta
        Purge period
    embargo : pd.Timedelta
        Embargo period
    raise_on_violation : bool
        Whether to raise exception on violation
        
    Returns
    -------
    Dict[str, bool]
        Detailed check results
    """
    auditor = LeakageAuditor(time_col=time_col, label_horizon=label_horizon)
    
    d = df.sort_values(time_col).copy()
    folds = d[fold_col].unique()
    
    results = {
        'overall_pass': True,
        'fold_results': {},
        'violations': []
    }
    
    for k in folds:
        val = d[d[fold_col] == k]
        trn = d[d[fold_col] != k]
        
        if val.empty or trn.empty:
            continue
        
        fold_results = auditor.audit_temporal_leakage(
            trn, val, purge, embargo
        )
        
        results['fold_results'][k] = fold_results
        
        # Check for violations
        if not all(fold_results.values()):
            results['overall_pass'] = False
            results['violations'].append(f"Fold {k}: {fold_results}")
    
    if not results['overall_pass'] and raise_on_violation:
        raise AssertionError(f"Leakage detected: {results['violations']}")
    
    return results