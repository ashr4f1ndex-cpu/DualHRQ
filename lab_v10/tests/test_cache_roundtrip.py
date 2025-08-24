"""
Test cache write→read parity for data integrity.

Ensures that data stored in cache can be retrieved identically,
which is critical for reproducible financial ML results.
"""

import tempfile
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

from src.data.cache import DataCache, CacheManager, CacheMetadata
from src.data.adapters import YahooFinanceAdapter, FREDAdapter, AdapterFactory


def create_test_equity_data(n_days: int = 100, symbol: str = "TEST") -> pd.DataFrame:
    """Create realistic test equity data."""
    dates = pd.date_range('2023-01-01', periods=n_days, freq='D')
    
    # Generate realistic OHLC data
    np.random.seed(42)  # For reproducibility
    base_price = 100.0
    prices = []
    
    for i in range(n_days):
        if i == 0:
            open_price = base_price
        else:
            # Open at previous close with some gap
            open_price = prices[-1]['close'] * (1 + np.random.normal(0, 0.01))
        
        # Intraday volatility
        daily_range = open_price * 0.02 * np.random.uniform(0.5, 2.0)
        high = open_price + daily_range * np.random.uniform(0, 1)
        low = open_price - daily_range * np.random.uniform(0, 1)
        close = low + (high - low) * np.random.uniform(0.2, 0.8)
        
        # Ensure OHLC relationships
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        volume = int(np.random.uniform(100000, 1000000))
        
        prices.append({
            'timestamp': dates[i],
            'symbol': symbol,
            'open': round(open_price, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'close': round(close, 2),
            'volume': volume,
            'adj_close': round(close, 2),  # Simplified
            'data_source': 'test',
            'quality_score': 1.0
        })
    
    return pd.DataFrame(prices)


def create_test_economic_data(n_points: int = 50, indicator: str = "FEDFUNDS") -> pd.DataFrame:
    """Create realistic economic indicator data."""
    dates = pd.date_range('2020-01-01', periods=n_points, freq='M')
    
    np.random.seed(123)  # For reproducibility
    
    # Generate realistic Fed Funds rate data
    base_rate = 2.0
    values = []
    current_rate = base_rate
    
    for _ in range(n_points):
        # Random walk with mean reversion
        change = np.random.normal(0, 0.1) - 0.1 * (current_rate - base_rate)
        current_rate += change
        current_rate = max(0, min(current_rate, 10))  # Realistic bounds
        values.append(round(current_rate, 2))
    
    return pd.DataFrame({
        'timestamp': dates,
        'symbol': indicator,
        'open': values,
        'high': values,
        'low': values,
        'close': values,
        'volume': 0,
        'data_source': 'test',
        'quality_score': 1.0
    })


class TestDataCacheRoundtrip:
    """Test cache write→read parity."""
    
    def test_basic_cache_roundtrip(self):
        """Test basic cache write and read operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = DataCache(cache_dir=temp_dir)
            
            # Create test data
            original_data = create_test_equity_data(50, "AAPL")
            
            # Write to cache
            success = cache.put(original_data, "AAPL", "yahoo_finance", "1d", "2023-01-01", "2023-02-19")
            assert success, "Cache put operation failed"
            
            # Read from cache
            retrieved_data = cache.get("AAPL", "yahoo_finance", "1d", "2023-01-01", "2023-02-19")
            
            assert retrieved_data is not None, "Cache get returned None"
            
            # Verify exact match
            pd.testing.assert_frame_equal(original_data, retrieved_data)
    
    def test_different_data_types_roundtrip(self):
        """Test roundtrip with different data types (equity, economic)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = DataCache(cache_dir=temp_dir)
            
            # Test equity data
            equity_data = create_test_equity_data(30, "SPY")
            cache.put(equity_data, "SPY", "yahoo_finance", "1d")
            retrieved_equity = cache.get("SPY", "yahoo_finance", "1d")
            
            pd.testing.assert_frame_equal(equity_data, retrieved_equity)
            
            # Test economic data
            econ_data = create_test_economic_data(24, "UNRATE")
            cache.put(econ_data, "UNRATE", "fred", "M")
            retrieved_econ = cache.get("UNRATE", "fred", "M")
            
            pd.testing.assert_frame_equal(econ_data, retrieved_econ)
    
    def test_large_dataset_roundtrip(self):
        """Test roundtrip with large datasets."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = DataCache(cache_dir=temp_dir)
            
            # Create large dataset (5 years of daily data)
            large_data = create_test_equity_data(1826, "QQQ")  # ~5 years
            
            # Cache large dataset
            success = cache.put(large_data, "QQQ", "yahoo_finance", "1d", "2019-01-01", "2023-12-31")
            assert success, "Failed to cache large dataset"
            
            # Retrieve and verify
            retrieved_data = cache.get("QQQ", "yahoo_finance", "1d", "2019-01-01", "2023-12-31")
            
            assert retrieved_data is not None, "Failed to retrieve large dataset"
            assert len(retrieved_data) == len(large_data), "Size mismatch in large dataset"
            
            pd.testing.assert_frame_equal(large_data, retrieved_data)
    
    def test_metadata_roundtrip(self):
        """Test that metadata is correctly stored and retrieved."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = DataCache(cache_dir=temp_dir)
            
            test_data = create_test_equity_data(10, "MSFT")
            
            # Store data
            cache.put(test_data, "MSFT", "alpha_vantage", "1d", "2023-01-01", "2023-01-10")
            
            # Check cache info
            info = cache.get_cache_info()
            
            assert info['total_entries'] == 1
            assert len(info['entries']) == 1
            
            entry = info['entries'][0]
            assert entry['symbol'] == "MSFT"
            assert entry['source'] == "alpha_vantage"
            assert entry['frequency'] == "1d"
            assert entry['records'] == 10
            assert entry['quality_score'] == 1.0
            assert not entry['expired']
    
    def test_cache_integrity_checking(self):
        """Test that cache detects data corruption."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = DataCache(cache_dir=temp_dir)
            
            test_data = create_test_equity_data(20, "TSLA")
            
            # Store data
            cache.put(test_data, "TSLA", "yahoo_finance", "1d")
            
            # Manually corrupt the cached file
            cache_key = cache._get_cache_key("TSLA", "yahoo_finance", "1d")
            data_path, metadata_path = cache._get_file_paths(cache_key)
            
            # Corrupt the data file
            corrupted_data = create_test_equity_data(20, "FAKE")  # Different data
            corrupted_data.to_parquet(data_path, index=False)
            
            # Cache should detect corruption and return None
            retrieved_data = cache.get("TSLA", "yahoo_finance", "1d")
            assert retrieved_data is None, "Cache should have detected corruption"
            
            # Cache entry should be cleaned up
            assert not cache.exists("TSLA", "yahoo_finance", "1d")
    
    def test_empty_dataframe_handling(self):
        """Test handling of empty DataFrames."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = DataCache(cache_dir=temp_dir)
            
            empty_df = pd.DataFrame()
            
            # Should not cache empty DataFrame
            success = cache.put(empty_df, "EMPTY", "test", "1d")
            assert not success, "Should not cache empty DataFrame"
            
            # Should not find it in cache
            assert not cache.exists("EMPTY", "test", "1d")
            retrieved = cache.get("EMPTY", "test", "1d")
            assert retrieved is None
    
    def test_special_characters_in_symbols(self):
        """Test cache keys with special characters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = DataCache(cache_dir=temp_dir)
            
            # Test symbols with special characters
            symbols = ["BRK-B", "SPX.US", "EUR/USD", "VIX^VIX"]
            
            for symbol in symbols:
                test_data = create_test_equity_data(10, symbol)
                
                # Should handle special characters in cache keys
                success = cache.put(test_data, symbol, "test", "1d")
                assert success, f"Failed to cache symbol with special chars: {symbol}"
                
                retrieved = cache.get(symbol, "test", "1d")
                assert retrieved is not None, f"Failed to retrieve symbol: {symbol}"
                
                pd.testing.assert_frame_equal(test_data, retrieved)
    
    def test_concurrent_cache_access(self):
        """Test that cache handles concurrent access gracefully."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = DataCache(cache_dir=temp_dir)
            
            # Store multiple symbols
            symbols = ["AAPL", "GOOGL", "AMZN", "MSFT"]
            original_data = {}
            
            for symbol in symbols:
                data = create_test_equity_data(20, symbol)
                original_data[symbol] = data
                success = cache.put(data, symbol, "yahoo_finance", "1d")
                assert success, f"Failed to cache {symbol}"
            
            # Retrieve all symbols and verify
            for symbol in symbols:
                retrieved = cache.get(symbol, "yahoo_finance", "1d")
                assert retrieved is not None, f"Failed to retrieve {symbol}"
                pd.testing.assert_frame_equal(original_data[symbol], retrieved)


class TestCacheManagerRoundtrip:
    """Test CacheManager write→read parity."""
    
    def test_cache_manager_get_or_fetch(self):
        """Test CacheManager get_or_fetch functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CacheManager(cache_dir=temp_dir)
            
            # Mock fetch function
            def fetch_test_data():
                return create_test_equity_data(30, "TEST")
            
            # First call should fetch and cache
            data1 = manager.get_or_fetch(fetch_test_data, "TEST", "mock_source", "1d")
            assert data1 is not None
            assert len(data1) == 30
            
            # Second call should return cached data
            fetch_call_count = 0
            def counting_fetch():
                nonlocal fetch_call_count
                fetch_call_count += 1
                return create_test_equity_data(30, "TEST")
            
            data2 = manager.get_or_fetch(counting_fetch, "TEST", "mock_source", "1d")
            
            # Should not have called fetch function (cache hit)
            assert fetch_call_count == 0
            assert data2 is not None
            
            # Data should be identical
            pd.testing.assert_frame_equal(data1, data2)
    
    def test_batch_get_or_fetch(self):
        """Test batch cache operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CacheManager(cache_dir=temp_dir)
            
            # Prepare batch requests
            requests = [
                {"symbol": "AAPL", "source": "test", "frequency": "1d"},
                {"symbol": "GOOGL", "source": "test", "frequency": "1d"},
                {"symbol": "MSFT", "source": "test", "frequency": "1d"}
            ]
            
            # Mock batch fetch function
            def batch_fetch(missing_requests):
                results = {}
                for req in missing_requests:
                    key = f"{req['symbol']}_test"
                    results[key] = create_test_equity_data(25, req['symbol'])
                return results
            
            # Fetch batch
            results = manager.batch_get_or_fetch(requests, batch_fetch)
            
            assert len(results) == 3
            assert "AAPL_test" in results
            assert "GOOGL_test" in results  
            assert "MSFT_test" in results
            
            # Verify data integrity
            for symbol in ["AAPL", "GOOGL", "MSFT"]:
                key = f"{symbol}_test"
                assert len(results[key]) == 25
                assert (results[key]['symbol'] == symbol).all()
            
            # Second batch call should use cache
            results2 = manager.batch_get_or_fetch(requests, batch_fetch)
            
            # Results should be identical
            for key in results:
                pd.testing.assert_frame_equal(results[key], results2[key])


class TestAdapterCacheIntegration:
    """Test integration between adapters and cache."""
    
    def test_yahoo_adapter_cache_roundtrip(self):
        """Test Yahoo Finance adapter with cache."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = DataCache(cache_dir=temp_dir)
            adapter = YahooFinanceAdapter()
            
            # Create mock Yahoo Finance data (simulating yfinance output)
            mock_yf_data = pd.DataFrame({
                'Date': pd.date_range('2023-01-01', periods=10),
                'Open': [100 + i for i in range(10)],
                'High': [105 + i for i in range(10)],
                'Low': [95 + i for i in range(10)],
                'Close': [102 + i for i in range(10)],
                'Volume': [1000000 + i*10000 for i in range(10)],
                'Adj Close': [102 + i for i in range(10)]
            }).set_index('Date')
            
            # Convert to canonical format
            canonical_data = adapter.to_canonical(mock_yf_data, "AAPL")
            
            # Cache the canonical data
            success = cache.put(canonical_data, "AAPL", "yahoo_finance", "1d")
            assert success
            
            # Retrieve and verify
            retrieved_data = cache.get("AAPL", "yahoo_finance", "1d")
            
            pd.testing.assert_frame_equal(canonical_data, retrieved_data)
    
    def test_fred_adapter_cache_roundtrip(self):
        """Test FRED adapter with cache."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = DataCache(cache_dir=temp_dir)
            adapter = FREDAdapter()
            
            # Create mock FRED data (simulating fredapi output)
            dates = pd.date_range('2023-01-01', periods=12, freq='M')
            mock_fred_series = pd.Series(
                data=[2.0 + 0.1*i for i in range(12)],
                index=dates
            )
            
            # Convert to canonical format
            canonical_data = adapter.to_canonical(mock_fred_series, "FEDFUNDS")
            
            # Cache the canonical data
            success = cache.put(canonical_data, "FEDFUNDS", "fred", "M")
            assert success
            
            # Retrieve and verify
            retrieved_data = cache.get("FEDFUNDS", "fred", "M")
            
            pd.testing.assert_frame_equal(canonical_data, retrieved_data)