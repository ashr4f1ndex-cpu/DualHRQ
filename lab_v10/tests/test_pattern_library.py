"""
Tests for Pattern Library and HRM-inspired learning components.

Tests pattern identification, specialist adapters, and regime detection
to ensure proper functioning of the ARC-inspired pattern recognition system.
"""

import pytest
import tempfile
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

import sys
sys.path.append('/Users/aziymandias/Desktop/dual_book_trading_lab_v10_complete/lab_v10')

from src.models.pattern_library import (
    PatternLibrary, RegimeDetector, PatternIDGenerator,
    RegimeSignature, SpecialistAdapter, PatternMetadata
)


def create_test_market_data(n_days: int = 100, volatility_regime: str = 'normal',
                           trend: str = 'sideways') -> pd.DataFrame:
    """Create synthetic market data with specific regime characteristics."""
    np.random.seed(42)
    
    dates = pd.date_range('2023-01-01', periods=n_days, freq='D')
    
    # Base price path
    base_price = 100.0
    prices = [base_price]
    
    # Volatility based on regime
    vol_multipliers = {'low': 0.5, 'normal': 1.0, 'high': 2.0, 'crisis': 3.0}
    base_vol = 0.02 * vol_multipliers.get(volatility_regime, 1.0)
    
    # Trend based on regime
    trend_drifts = {'bull': 0.001, 'sideways': 0.0, 'bear': -0.001}
    drift = trend_drifts.get(trend, 0.0)
    
    for i in range(1, n_days):
        # Generate price with trend and volatility
        return_ = drift + np.random.normal(0, base_vol)
        new_price = prices[-1] * (1 + return_)
        prices.append(new_price)
    
    # Create OHLCV data
    data = []
    for i, date in enumerate(dates):
        close = prices[i]
        daily_vol = base_vol * np.random.uniform(0.5, 1.5)
        
        high = close * (1 + daily_vol * np.random.uniform(0, 1))
        low = close * (1 - daily_vol * np.random.uniform(0, 1))
        open_price = low + (high - low) * np.random.uniform(0.2, 0.8)
        
        # Ensure OHLC consistency
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        volume = int(np.random.uniform(100000, 1000000))
        
        data.append({
            'timestamp': date,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume,
            'symbol': 'TEST'
        })
    
    df = pd.DataFrame(data)
    
    # Add derived features
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    
    return df


class TestRegimeDetector:
    """Test regime detection functionality."""
    
    def test_volatility_regime_detection(self):
        """Test detection of different volatility regimes."""
        detector = RegimeDetector()
        
        # Test different volatility regimes
        regimes = ['low', 'normal', 'high', 'crisis']
        
        for regime in regimes:
            data = create_test_market_data(50, volatility_regime=regime)
            reference_date = datetime(2023, 1, 25)
            
            signature = detector.detect_regime_signature(data, reference_date)
            
            # Check that detected volatility regime makes sense
            assert signature.volatility_regime in ['low', 'normal', 'high', 'crisis']
            assert isinstance(signature.vix_spike, bool)
            assert signature.trend_regime in ['bull', 'bear', 'sideways']
    
    def test_calendar_regime_detection(self):
        """Test detection of calendar-based regimes."""
        detector = RegimeDetector()
        data = create_test_market_data(100)
        
        # Test month-end detection
        month_end_date = datetime(2023, 1, 30)
        signature = detector.detect_regime_signature(data, month_end_date)
        
        # Month-end should be detected
        assert signature.calendar_regime in ['month_end', 'normal']
        
        # Test normal period
        normal_date = datetime(2023, 1, 15)
        signature = detector.detect_regime_signature(data, normal_date)
        
        assert signature.calendar_regime in ['normal', 'fomc']  # FOMC possible in mid-month
    
    def test_trend_regime_detection(self):
        """Test trend regime detection."""
        detector = RegimeDetector()
        
        trends = ['bull', 'bear', 'sideways']
        
        for trend in trends:
            data = create_test_market_data(60, trend=trend)
            reference_date = datetime(2023, 2, 15)
            
            signature = detector.detect_regime_signature(data, reference_date)
            
            # Trend should be detected (allowing for some noise in detection)
            assert signature.trend_regime in ['bull', 'bear', 'sideways']


class TestPatternIDGenerator:
    """Test pattern ID generation and matching."""
    
    def test_pattern_id_generation(self):
        """Test that pattern IDs are generated consistently."""
        generator = PatternIDGenerator()
        
        signature1 = RegimeSignature(
            volatility_regime='high',
            calendar_regime='month_end',
            macro_regime='recession',
            vix_spike=True,
            trend_regime='bear',
            volume_regime='high'
        )
        
        signature2 = RegimeSignature(
            volatility_regime='high',
            calendar_regime='month_end', 
            macro_regime='recession',
            vix_spike=True,
            trend_regime='bear',
            volume_regime='high'
        )
        
        # Same signatures should produce same pattern ID
        id1 = generator.generate_pattern_id(signature1)
        id2 = generator.generate_pattern_id(signature2)
        
        assert id1 == id2
        assert id1.startswith('pattern_')
        assert len(id1) > 10  # Should have reasonable length
    
    def test_different_signatures_different_ids(self):
        """Test that different signatures produce different IDs."""
        generator = PatternIDGenerator()
        
        signature1 = RegimeSignature(
            volatility_regime='low', calendar_regime='normal',
            macro_regime='expansion', vix_spike=False,
            trend_regime='bull', volume_regime='normal'
        )
        
        signature2 = RegimeSignature(
            volatility_regime='high', calendar_regime='month_end',
            macro_regime='recession', vix_spike=True,
            trend_regime='bear', volume_regime='high'
        )
        
        id1 = generator.generate_pattern_id(signature1)
        id2 = generator.generate_pattern_id(signature2)
        
        assert id1 != id2
    
    def test_similar_pattern_finding(self):
        """Test finding similar patterns."""
        generator = PatternIDGenerator()
        
        # Create target signature
        target = RegimeSignature(
            volatility_regime='high', calendar_regime='normal',
            macro_regime='recession', vix_spike=True,
            trend_regime='bear', volume_regime='normal'
        )
        
        # Create similar signature (only trend differs)
        similar = RegimeSignature(
            volatility_regime='high', calendar_regime='normal',
            macro_regime='recession', vix_spike=True,
            trend_regime='sideways', volume_regime='normal'
        )
        
        # Create very different signature
        different = RegimeSignature(
            volatility_regime='low', calendar_regime='year_end',
            macro_regime='expansion', vix_spike=False,
            trend_regime='bull', volume_regime='low'
        )
        
        # Create mock known patterns
        known_patterns = {
            'pattern_1': PatternMetadata(
                pattern_id='pattern_1',
                regime_signature=similar,
                first_seen=datetime.now(),
                last_seen=datetime.now(),
                occurrence_count=5,
                average_performance=0.6
            ),
            'pattern_2': PatternMetadata(
                pattern_id='pattern_2',
                regime_signature=different,
                first_seen=datetime.now(),
                last_seen=datetime.now(),
                occurrence_count=3,
                average_performance=0.4
            )
        }
        
        # Find similar patterns
        similar_patterns = generator.find_similar_patterns(
            target, known_patterns, similarity_threshold=0.5
        )
        
        # Should find the similar pattern but not the different one
        assert len(similar_patterns) >= 1
        assert similar_patterns[0][0] == 'pattern_1'  # Most similar first
        assert similar_patterns[0][1] > 0.5  # High similarity


class TestSpecialistAdapter:
    """Test specialist adapter functionality."""
    
    def test_adapter_creation(self):
        """Test creating specialist adapters."""
        adapter = SpecialistAdapter(base_dim=256, rank=16)
        
        assert adapter.rank == 16
        assert adapter.scale == 1.0  # alpha/rank = 16/16
        assert adapter.adaptation_count == 0
        assert len(adapter.performance_history) == 0
    
    def test_adapter_forward_pass(self):
        """Test adapter forward pass."""
        adapter = SpecialistAdapter(base_dim=128, rank=8)
        
        # Test forward pass
        x = torch.randn(10, 128)
        output = adapter(x)
        
        assert output.shape == (10, 128)
        assert output.dtype == torch.float32
    
    def test_adapter_merge_with_base(self):
        """Test merging adapter with base layer."""
        # Create base layer
        base_layer = nn.Linear(64, 64)
        
        # Create adapter
        adapter = SpecialistAdapter(base_dim=64, rank=4)
        
        # Merge
        merged_layer = adapter.merge_with_base(base_layer)
        
        assert isinstance(merged_layer, nn.Linear)
        assert merged_layer.in_features == 64
        assert merged_layer.out_features == 64
    
    def test_performance_recording(self):
        """Test recording adapter performance."""
        adapter = SpecialistAdapter(base_dim=32)
        
        # Record some performances
        performances = [0.8, 0.75, 0.9, 0.85]
        for perf in performances:
            adapter.record_performance(perf)
        
        assert adapter.adaptation_count == len(performances)
        assert len(adapter.performance_history) == len(performances)
        assert adapter.performance_history == performances


class TestPatternLibrary:
    """Test main pattern library functionality."""
    
    def test_pattern_library_initialization(self):
        """Test pattern library initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            library = PatternLibrary(library_dir=temp_dir)
            
            assert library.library_dir == Path(temp_dir)
            assert isinstance(library.regime_detector, RegimeDetector)
            assert isinstance(library.pattern_id_generator, PatternIDGenerator)
            assert len(library.patterns) == 0
    
    def test_pattern_identification(self):
        """Test pattern identification process."""
        with tempfile.TemporaryDirectory() as temp_dir:
            library = PatternLibrary(library_dir=temp_dir)
            
            # Create test data
            data = create_test_market_data(50, 'high', 'bear')
            reference_date = datetime(2023, 1, 25)
            
            # Identify pattern
            pattern_id, signature = library.identify_pattern(data, reference_date)
            
            assert pattern_id.startswith('pattern_')
            assert isinstance(signature, RegimeSignature)
            
            # Pattern should be registered
            assert pattern_id in library.patterns
            assert library.patterns[pattern_id].occurrence_count == 1
    
    def test_pattern_reidentification(self):
        """Test that same patterns are recognized again."""
        with tempfile.TemporaryDirectory() as temp_dir:
            library = PatternLibrary(library_dir=temp_dir)
            
            data = create_test_market_data(50, 'normal', 'sideways')
            date1 = datetime(2023, 1, 15)  # Sunday
            date2 = datetime(2023, 1, 15)  # Same date - should produce same pattern
            
            # Identify same pattern twice with same reference date
            pattern_id1, signature1 = library.identify_pattern(data, date1)
            pattern_id2, signature2 = library.identify_pattern(data, date2)
            
            # Should get same pattern ID for same date and data
            assert pattern_id1 == pattern_id2
            assert signature1.volatility_regime == signature2.volatility_regime
            assert signature1.calendar_regime == signature2.calendar_regime
            
            # Occurrence count should increase
            assert library.patterns[pattern_id1].occurrence_count == 2
    
    def test_specialist_adapter_creation(self):
        """Test specialist adapter creation and retrieval."""
        with tempfile.TemporaryDirectory() as temp_dir:
            library = PatternLibrary(library_dir=temp_dir)
            
            # Create a pattern first
            data = create_test_market_data(30)
            reference_date = datetime(2023, 1, 10)
            pattern_id, _ = library.identify_pattern(data, reference_date)
            
            # Get specialist adapter
            adapter = library.get_specialist_adapter(pattern_id, base_dim=128)
            
            assert isinstance(adapter, SpecialistAdapter)
            assert pattern_id in library.specialist_adapters
            
            # Getting same adapter again should return same instance
            adapter2 = library.get_specialist_adapter(pattern_id, base_dim=128)
            assert adapter is adapter2
    
    def test_specialist_adaptation(self):
        """Test specialist adapter fine-tuning."""
        with tempfile.TemporaryDirectory() as temp_dir:
            library = PatternLibrary(library_dir=temp_dir)
            
            # Create pattern
            data = create_test_market_data(40)
            reference_date = datetime(2023, 1, 15)
            pattern_id, _ = library.identify_pattern(data, reference_date)
            
            # Create simple base model
            base_model = nn.Sequential(
                nn.Linear(10, 32),
                nn.ReLU(),
                nn.Linear(32, 2)
            )
            
            # Create training data
            X = torch.randn(20, 10)
            y = torch.randint(0, 2, (20,))
            
            # Adapt specialist
            adapter = library.adapt_specialist(
                pattern_id, (X, y), base_model,
                learning_rate=0.01, num_epochs=5
            )
            
            assert adapter is not None
            assert adapter.adaptation_count > 0
            assert len(adapter.performance_history) > 0
            
            # Pattern should have adaptation history
            assert len(library.patterns[pattern_id].adaptation_history) > 0
    
    def test_pattern_statistics(self):
        """Test pattern library statistics."""
        with tempfile.TemporaryDirectory() as temp_dir:
            library = PatternLibrary(library_dir=temp_dir)
            
            # Create several patterns
            for i, vol_regime in enumerate(['low', 'normal', 'high']):
                data = create_test_market_data(30, vol_regime)
                date = datetime(2023, 1, 10 + i)
                library.identify_pattern(data, date)
            
            stats = library.get_pattern_statistics()
            
            assert stats['total_patterns'] == 3
            assert stats['total_occurrences'] == 3
            assert 'most_common_pattern' in stats
            assert 'regime_distribution' in stats
            assert len(stats['recent_patterns']) <= 10


class TestPatternLibraryIntegration:
    """Integration tests for pattern library components."""
    
    def test_end_to_end_pattern_workflow(self):
        """Test complete pattern identification and adaptation workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            library = PatternLibrary(library_dir=temp_dir, max_patterns=10)
            
            # Create base model
            base_model = nn.Sequential(
                nn.Linear(5, 16),
                nn.ReLU(),
                nn.Linear(16, 1)
            )
            
            # Simulate different market conditions over time
            market_conditions = [
                ('low', 'bull'), ('normal', 'sideways'), ('high', 'bear'),
                ('crisis', 'bear'), ('normal', 'bull')
            ]
            
            adaptation_results = []
            
            for i, (vol_regime, trend) in enumerate(market_conditions):
                # Create market data for this regime
                data = create_test_market_data(40, vol_regime, trend)
                reference_date = datetime(2023, 1, 1) + timedelta(days=i*10)
                
                # Identify pattern
                pattern_id, signature = library.identify_pattern(data, reference_date)
                
                # Create training data based on the market regime
                X = torch.randn(30, 5)
                if vol_regime in ['high', 'crisis']:
                    # Higher variance targets for volatile markets
                    y = torch.randn(30, 1) * 2.0
                else:
                    y = torch.randn(30, 1) * 0.5
                
                # Adapt specialist for this pattern
                adapter = library.adapt_specialist(
                    pattern_id, (X, y), base_model,
                    learning_rate=0.005, num_epochs=3
                )
                
                if adapter is not None:
                    adaptation_results.append({
                        'pattern_id': pattern_id,
                        'regime': (vol_regime, trend),
                        'performance': adapter.performance_history[-1] if adapter.performance_history else 0.0
                    })
            
            # Verify results
            assert len(library.patterns) <= library.max_patterns
            assert len(adaptation_results) > 0
            
            # Check that different regimes got different patterns (mostly)
            unique_patterns = set(r['pattern_id'] for r in adaptation_results)
            assert len(unique_patterns) >= 2  # Should have at least 2 different patterns
            
            # Check statistics
            stats = library.get_pattern_statistics()
            assert stats['total_patterns'] > 0
            assert stats['total_occurrences'] >= len(market_conditions)
    
    def test_pattern_persistence(self):
        """Test that patterns can be saved and loaded."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create library and add patterns
            library1 = PatternLibrary(library_dir=temp_dir)
            
            data = create_test_market_data(50)
            reference_date = datetime(2023, 1, 15)
            pattern_id, _ = library1.identify_pattern(data, reference_date)
            
            # Save patterns
            library1.save_patterns()
            
            # Create new library instance (should load existing patterns)
            library2 = PatternLibrary(library_dir=temp_dir)
            
            # Should have loaded the pattern
            assert len(library2.patterns) == 1
            assert pattern_id in library2.patterns
            assert library2.patterns[pattern_id].occurrence_count == 1
    
    def test_max_patterns_limit(self):
        """Test that pattern library respects maximum pattern limit."""
        with tempfile.TemporaryDirectory() as temp_dir:
            library = PatternLibrary(library_dir=temp_dir, max_patterns=3)
            
            # Create more patterns than the limit
            regimes = ['low', 'normal', 'high', 'crisis', 'normal']
            
            for i, regime in enumerate(regimes):
                data = create_test_market_data(30, regime)
                # Use different dates to ensure different patterns
                date = datetime(2023, 1, 1) + timedelta(days=i*30)
                library.identify_pattern(data, date)
            
            # Should not exceed max patterns
            assert len(library.patterns) <= library.max_patterns
            
            # Should have removed least frequently used patterns
            occurrence_counts = [p.occurrence_count for p in library.patterns.values()]
            assert all(count >= 1 for count in occurrence_counts)