"""
benchmark_datasets.py
=====================

Benchmark datasets and synthetic data generators for HRM training.
Provides standardized datasets for comparing model performance and
synthetic data generators for testing and validation.

Features:
- Financial time series benchmarks
- Synthetic options data generators
- Regime-aware synthetic data
- Stress test scenarios
- Cross-validation datasets
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from scipy import stats
from datetime import datetime, timedelta
import warnings

class FinancialDataGenerator:
    """Generate synthetic financial data for HRM training and testing."""

    def __init__(self, seed: int = 42):
        """Initialize with random seed for reproducibility."""
        np.random.seed(seed)
        self.seed = seed

    def generate_gbm_prices(self, S0: float = 100, mu: float = 0.05,
                           sigma: float = 0.2, T: float = 1.0,
                           steps: int = 252) -> pd.Series:
        """
        Generate prices using Geometric Brownian Motion.

        Args:
            S0: Initial price
            mu: Drift parameter
            sigma: Volatility parameter
            T: Time horizon in years
            steps: Number of time steps

        Returns:
            Price series with daily frequency
        """
        dt = T / steps
        dates = pd.date_range(start='2020-01-01', periods=steps+1, freq='D')

        # Generate random shocks
        dW = np.random.normal(0, np.sqrt(dt), steps)

        # Calculate price path
        log_returns = (mu - 0.5 * sigma**2) * dt + sigma * dW
        log_prices = np.log(S0) + np.cumsum(log_returns)
        prices = np.exp(log_prices)

        # Add initial price
        all_prices = np.concatenate([[S0], prices])

        return pd.Series(all_prices, index=dates, name='price')

    def generate_regime_switching_prices(self, S0: float = 100, T: float = 2.0,
                                       steps: int = 504) -> tuple[pd.Series, pd.Series]:
        """
        Generate prices with regime switching (low vol vs high vol).

        Useful for testing HRM's ability to adapt to changing market conditions.

        Args:
            S0: Initial price
            T: Time horizon in years
            steps: Number of time steps

        Returns:
            Tuple of (prices, regime_indicator)
        """
        dt = T / steps
        dates = pd.date_range(start='2020-01-01', periods=steps+1, freq='D')

        # Regime parameters
        mu_low, sigma_low = 0.05, 0.15    # Low volatility regime
        mu_high, sigma_high = 0.02, 0.35  # High volatility regime

        # Transition probabilities
        p_low_to_high = 0.05   # 5% chance of switching from low to high vol
        p_high_to_low = 0.10   # 10% chance of switching from high to low vol

        # Initialize
        regimes = np.zeros(steps)
        current_regime = 0  # Start in low vol regime

        log_prices = [np.log(S0)]

        for _t in range(steps):
            # Regime switching logic
            if current_regime == 0:  # Low vol regime
                if np.random.random() < p_low_to_high:
                    current_regime = 1
            else:  # High vol regime
                if np.random.random() < p_high_to_low:
                    current_regime = 0

            regimes[_t] = current_regime

            # Generate return based on current regime
            if current_regime == 0:
                mu, sigma = mu_low, sigma_low
            else:
                mu, sigma = mu_high, sigma_high

            dW = np.random.normal(0, np.sqrt(dt))
            log_return = (mu - 0.5 * sigma**2) * dt + sigma * dW
            log_prices.append(log_prices[-1] + log_return)

        prices = np.exp(log_prices)
        price_series = pd.Series(prices, index=dates, name='price')
        regime_series = pd.Series(np.concatenate([[0], regimes]), index=dates, name='regime')

        return price_series, regime_series

    def generate_jump_diffusion_prices(self, S0: float = 100, mu: float = 0.05,
                                     sigma: float = 0.2, lambda_jump: float = 0.1,
                                     mu_jump: float = -0.05, sigma_jump: float = 0.1,
                                     T: float = 1.0, steps: int = 252) -> pd.Series:
        """
        Generate prices using Merton jump-diffusion model.

        Includes rare but large price jumps, useful for stress testing.

        Args:
            S0: Initial price
            mu: Drift parameter
            sigma: Volatility parameter
            lambda_jump: Jump frequency (jumps per year)
            mu_jump: Mean jump size
            sigma_jump: Jump size volatility
            T: Time horizon in years
            steps: Number of time steps

        Returns:
            Price series with jumps
        """
        dt = T / steps
        dates = pd.date_range(start='2020-01-01', periods=steps+1, freq='D')

        log_prices = [np.log(S0)]

        for _t in range(steps):
            # Normal diffusion component
            dW = np.random.normal(0, np.sqrt(dt))
            diffusion = (mu - 0.5 * sigma**2) * dt + sigma * dW

            # Jump component
            jump = 0
            if np.random.poisson(lambda_jump * dt) > 0:
                jump = np.random.normal(mu_jump, sigma_jump)

            log_return = diffusion + jump
            log_prices.append(log_prices[-1] + log_return)

        prices = np.exp(log_prices)
        return pd.Series(prices, index=dates, name='price')

    def generate_options_iv_surface(self, S: float = 100, T_max: float = 1.0,
                                   K_range: tuple[float, float] = (0.8, 1.2),
                                   n_strikes: int = 21, n_expiries: int = 10) -> pd.DataFrame:
        """
        Generate a realistic implied volatility surface.

        Args:
            S: Current underlying price
            T_max: Maximum time to expiration
            K_range: Strike range as (min_moneyness, max_moneyness)
            n_strikes: Number of strike prices
            n_expiries: Number of expiration dates

        Returns:
            DataFrame with IV surface (rows=strikes, cols=expiries)
        """
        # Create strikes and expiries
        strikes = np.linspace(K_range[0] * S, K_range[1] * S, n_strikes)
        expiries = np.linspace(0.01, T_max, n_expiries)

        # Base volatility parameters
        atm_vol = 0.20
        vol_of_vol = 0.02
        skew_strength = -0.1
        term_structure_slope = 0.05

        iv_surface = np.zeros((n_strikes, n_expiries))

        for i, K in enumerate(strikes):
            for j, T in enumerate(expiries):
                # Moneyness effect (volatility skew)
                moneyness = np.log(K / S)
                skew_effect = skew_strength * moneyness

                # Term structure effect
                term_effect = term_structure_slope * np.sqrt(T)

                # Add some noise
                noise = np.random.normal(0, vol_of_vol)

                # Combine effects
                iv = atm_vol + skew_effect + term_effect + noise
                iv = max(iv, 0.05)  # Floor at 5%

                iv_surface[i, j] = iv

        # Create DataFrame
        strike_labels = [f"K_{K:.0f}" for K in strikes]
        expiry_labels = [f"T_{T:.2f}" for T in expiries]

        return pd.DataFrame(iv_surface, index=strike_labels, columns=expiry_labels)

    def generate_intraday_bars(self, daily_price: float, volatility: float = 0.02,
                              n_bars: int = 390, volume_pattern: str = 'u_shaped') -> pd.DataFrame:
        """
        Generate intraday minute bars from daily price.

        Args:
            daily_price: End-of-day price
            volatility: Intraday volatility
            n_bars: Number of minute bars (default 390 = 6.5 hours)
            volume_pattern: Volume pattern ('u_shaped', 'flat', 'morning_spike')

        Returns:
            DataFrame with OHLCV minute bars
        """
        # Generate minute returns
        returns = np.random.normal(0, volatility / np.sqrt(n_bars), n_bars)

        # Ensure returns sum to daily return (price conservation)
        returns = returns - returns.mean()  # Center around zero

        # Calculate minute prices
        prices = [daily_price]
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))

        # Create OHLC bars
        bars = []
        for i in range(n_bars):
            open_price = prices[i]
            close_price = prices[i + 1]

            # Generate high/low with some randomness
            high_low_spread = abs(close_price - open_price) * (1 + np.random.exponential(0.5))
            high = max(open_price, close_price) + high_low_spread * np.random.random()
            low = min(open_price, close_price) - high_low_spread * np.random.random()

            # Generate volume based on pattern
            if volume_pattern == 'u_shaped':
                # Higher volume at open and close
                volume_factor = 1 + 2 * (abs(i - n_bars/2) / (n_bars/2))
            elif volume_pattern == 'morning_spike':
                # Higher volume in morning
                volume_factor = np.exp(-i / (n_bars/4))
            else:  # flat
                volume_factor = 1

            base_volume = 10000
            volume = int(base_volume * volume_factor * (1 + np.random.normal(0, 0.3)))
            volume = max(volume, 100)  # Minimum volume

            bars.append({
                'open': open_price,
                'high': high,
                'low': low,
                'close': close_price,
                'volume': volume
            })

        # Create timestamps
        start_time = pd.Timestamp('2023-01-01 09:30:00')
        timestamps = pd.date_range(start=start_time, periods=n_bars, freq='T')

        return pd.DataFrame(bars, index=timestamps)

class BenchmarkDatasets:
    """Curated benchmark datasets for HRM evaluation."""

    @staticmethod
    def load_sp500_crisis_periods() -> dict[str, tuple[str, str]]:
        """
        Get date ranges for major market crisis periods.

        Useful for testing HRM performance during stressed markets.

        Returns:
            Dictionary mapping crisis names to (start_date, end_date) tuples
        """
        return {
            'dot_com_crash': ('2000-03-01', '2002-10-01'),
            'financial_crisis': ('2007-10-01', '2009-03-01'),
            'flash_crash': ('2010-05-01', '2010-06-01'),
            'covid_crash': ('2020-02-01', '2020-04-01'),
            'volatility_spike_2018': ('2018-01-01', '2018-03-01'),
            'rate_hike_2022': ('2022-01-01', '2022-12-31')
        }

    @staticmethod
    def generate_stress_test_scenarios(base_price: float = 100,
                                     n_scenarios: int = 1000) -> pd.DataFrame:
        """
        Generate stress test scenarios for robust HRM evaluation.

        Args:
            base_price: Starting price for scenarios
            n_scenarios: Number of scenarios to generate

        Returns:
            DataFrame with price paths for different stress scenarios
        """
        generator = FinancialDataGenerator()
        scenarios = {}

        for i in range(n_scenarios):
            # Randomize parameters for stress testing
            mu = np.random.uniform(-0.3, 0.3)        # Wide range of drifts
            sigma = np.random.uniform(0.1, 0.8)      # Wide range of volatilities

            # Generate scenario
            if i % 3 == 0:  # Jump diffusion scenarios
                scenario = generator.generate_jump_diffusion_prices(
                    S0=base_price, mu=mu, sigma=sigma,
                    lambda_jump=np.random.uniform(0, 0.5),
                    mu_jump=np.random.uniform(-0.2, 0.1),
                    sigma_jump=np.random.uniform(0.05, 0.3),
                    T=1.0, steps=252
                )
            else:  # Regular GBM scenarios
                scenario = generator.generate_gbm_prices(
                    S0=base_price, mu=mu, sigma=sigma, T=1.0, steps=252
                )

            scenarios[f'scenario_{i}'] = scenario

        return pd.DataFrame(scenarios)

    @staticmethod
    def create_cross_validation_splits(data: pd.DataFrame,
                                     n_folds: int = 5,
                                     test_size: float = 0.2,
                                     embargo_days: int = 21) -> list[dict]:
        """
        Create time-aware cross-validation splits for financial data.

        Implements embargo periods to prevent data leakage.

        Args:
            data: Time series data to split
            n_folds: Number of CV folds
            test_size: Fraction of data for testing
            embargo_days: Embargo period in days

        Returns:
            List of fold dictionaries with train/test indices
        """
        n_samples = len(data)
        test_samples = int(n_samples * test_size)

        folds = []

        for fold in range(n_folds):
            # Calculate test start/end for this fold
            test_start = fold * (n_samples // n_folds)
            test_end = min(test_start + test_samples, n_samples)

            # Apply embargo
            train_end = test_start - embargo_days
            train_start = max(0, train_end - test_samples * 2)  # Use 2x test size for training

            if train_start >= 0 and train_end > train_start and test_end > test_start:
                folds.append({
                    'fold': fold,
                    'train_start': train_start,
                    'train_end': train_end,
                    'test_start': test_start,
                    'test_end': test_end,
                    'train_dates': data.index[train_start:train_end],
                    'test_dates': data.index[test_start:test_end]
                })

        return folds

class OptionsDatasetGenerator:
    """Generate realistic options datasets for HRM training."""

    def __init__(self, seed: int = 42):
        """Initialize with random seed."""
        self.generator = FinancialDataGenerator(seed)
        self.seed = seed

    def generate_straddle_dataset(self, n_samples: int = 1000,
                                 start_date: str = '2018-01-01',
                                 end_date: str = '2023-12-31') -> pd.DataFrame:
        """
        Generate a complete dataset for ATM straddle backtesting.

        Args:
            n_samples: Number of trading days to generate
            start_date: Start date for dataset
            end_date: End date for dataset

        Returns:
            DataFrame with all required columns for straddle backtesting
        """
        # Generate underlying price path
        dates = pd.date_range(start=start_date, end=end_date, freq='B')[:n_samples]

        # Use regime-switching model for more realistic price dynamics
        prices, regimes = self.generator.generate_regime_switching_prices(
            S0=100, T=len(dates)/252, steps=len(dates)
        )
        prices.index = dates
        regimes.index = dates

        # Calculate returns and realized volatility
        returns = np.log(prices / prices.shift(1)).dropna()
        rv_5d = returns.rolling(5).std() * np.sqrt(252)
        rv_21d = returns.rolling(21).std() * np.sqrt(252)

        # Generate implied volatility with realistic features
        base_iv = 0.20
        iv_data = []

        for i, (date, price) in enumerate(prices.items()):
            if i == 0:
                iv_data.append(base_iv)
                continue

            # IV responds to price moves and regime
            price_shock = abs(returns.iloc[i-1]) if i-1 < len(returns) else 0
            regime_effect = 0.10 if regimes.iloc[i] == 1 else 0  # High vol regime
            rv_effect = (rv_21d.iloc[i-1] - base_iv) * 0.3 if i-1 < len(rv_21d) and not pd.isna(rv_21d.iloc[i-1]) else 0

            # Mean reversion in IV
            prev_iv = iv_data[-1]
            mean_reversion = (base_iv - prev_iv) * 0.05

            # Noise
            noise = np.random.normal(0, 0.02)

            new_iv = prev_iv + price_shock * 0.5 + regime_effect + rv_effect + mean_reversion + noise
            new_iv = max(0.05, min(1.0, new_iv))  # Clamp between 5% and 100%

            iv_data.append(new_iv)

        iv_series = pd.Series(iv_data, index=dates, name='iv_entry')

        # Create vol-gap target (key label for HRM)
        iv_exit = iv_series.shift(-5).fillna(method='ffill')  # 5-day forward IV
        vol_gap = rv_5d.shift(-5) - iv_series  # Future RV - Current IV

        # Create expiry dates (30 DTE target)
        expiry = dates + pd.Timedelta(days=30)

        # VIX proxy (market fear gauge)
        vix = iv_series * 100 + np.random.normal(0, 2, len(iv_series))  # Scale to VIX-like levels
        vix = vix.clip(10, 80)  # Realistic VIX range

        # Combine into dataset
        dataset = pd.DataFrame({
            'underlying_price': prices,
            'iv_entry': iv_series,
            'iv_exit': iv_exit,
            'expiry': expiry,
            'vol_gap': vol_gap,
            'realized_vol_5d': rv_5d,
            'realized_vol_21d': rv_21d,
            'returns': returns,
            'regime': regimes,
            'vix_proxy': vix,
            'price_momentum_5d': prices.pct_change(5),
            'price_momentum_21d': prices.pct_change(21)
        })

        return dataset.dropna()

    def generate_intraday_signals_dataset(self, n_days: int = 500,
                                         signals_per_day: int = 5) -> pd.DataFrame:
        """
        Generate intraday signals dataset for HRM training.

        Args:
            n_days: Number of trading days
            signals_per_day: Average number of signals per day

        Returns:
            DataFrame with intraday signal features and labels
        """
        all_signals = []

        for day in range(n_days):
            # Generate daily price for this day
            daily_price = 100 * (1 + np.random.normal(0, 0.02))**day

            # Generate intraday bars
            bars = self.generator.generate_intraday_bars(
                daily_price=daily_price,
                volatility=np.random.uniform(0.01, 0.05)
            )

            # Calculate intraday features
            bars['vwap'] = (bars['close'] * bars['volume']).cumsum() / bars['volume'].cumsum()
            bars['atr'] = bars[['high', 'low']].max(axis=1) - bars[['high', 'low']].min(axis=1)
            bars['stretch'] = abs(bars['close'] - bars['vwap']) / bars['atr'].rolling(14).mean()
            bars['volume_ratio'] = bars['volume'] / bars['volume'].rolling(20).mean()
            bars['time_to_close'] = (bars.index.hour * 60 + bars.index.minute - 9*60 - 30) / (6.5 * 60)

            # Generate signals based on realistic patterns
            n_signals_today = np.random.poisson(signals_per_day)

            for _ in range(n_signals_today):
                signal_idx = np.random.randint(50, len(bars)-50)  # Avoid start/end of day
                signal_time = bars.index[signal_idx]

                # Extract features at signal time
                features = {
                    'timestamp': signal_time,
                    'day': day,
                    'underlying_price': bars.iloc[signal_idx]['close'],
                    'vwap': bars.iloc[signal_idx]['vwap'],
                    'stretch': bars.iloc[signal_idx]['stretch'],
                    'volume_ratio': bars.iloc[signal_idx]['volume_ratio'],
                    'time_to_close': bars.iloc[signal_idx]['time_to_close'],
                    'atr_normalized': bars.iloc[signal_idx]['atr'] / bars.iloc[signal_idx]['close'],
                    'price_momentum_5min': bars.iloc[signal_idx]['close'] / bars.iloc[max(0, signal_idx-5)]['close'] - 1,
                    'volume_5min_avg': bars.iloc[max(0, signal_idx-5):signal_idx+1]['volume'].mean()
                }

                # Generate realistic signal outcome (probability of profitable short)
                # Higher stretch and volume ratio should increase probability
                base_prob = 0.35  # Base success rate
                stretch_effect = min(0.3, features['stretch'] * 0.1)  # Up to 30% boost from stretch
                volume_effect = min(0.15, (features['volume_ratio'] - 1) * 0.05)  # Volume boost
                time_effect = abs(features['time_to_close'] - 0.5) * 0.1  # Avoid lunch time

                signal_prob = base_prob + stretch_effect + volume_effect + time_effect
                signal_prob = max(0.1, min(0.8, signal_prob))  # Clamp realistic range

                # Binary outcome
                features['signal_success'] = np.random.random() < signal_prob
                features['signal_probability'] = signal_prob

                # Expected return (for regression target)
                if features['signal_success']:
                    expected_return = np.random.lognormal(-1, 0.5) * features['stretch'] * 0.01
                else:
                    expected_return = -np.random.exponential(0.005)  # Small loss

                features['expected_return'] = expected_return

                all_signals.append(features)

        return pd.DataFrame(all_signals)

def create_hrm_training_package() -> dict[str, pd.DataFrame]:
    """
    Create a complete training package for HRM with multiple datasets.

    Returns:
        Dictionary containing various datasets for comprehensive HRM training
    """
    print("Generating HRM training datasets...")

    # Initialize generators
    options_gen = OptionsDatasetGenerator(seed=42)
    benchmark = BenchmarkDatasets()

    datasets = {}

    # 1. Main straddle training dataset
    print("  - Generating straddle dataset...")
    datasets['straddle_train'] = options_gen.generate_straddle_dataset(
        n_samples=2000, start_date='2018-01-01', end_date='2023-12-31'
    )

    # 2. Intraday signals dataset
    print("  - Generating intraday signals dataset...")
    datasets['intraday_signals'] = options_gen.generate_intraday_signals_dataset(
        n_days=1000, signals_per_day=8
    )

    # 3. Stress test scenarios
    print("  - Generating stress test scenarios...")
    datasets['stress_scenarios'] = benchmark.generate_stress_test_scenarios(
        base_price=100, n_scenarios=500
    )

    # 4. Regime-aware validation set
    print("  - Generating regime validation set...")
    fin_gen = FinancialDataGenerator(seed=123)
    regime_prices, regime_labels = fin_gen.generate_regime_switching_prices(
        S0=100, T=3, steps=750
    )
    datasets['regime_validation'] = pd.DataFrame({
        'price': regime_prices,
        'regime': regime_labels
    })

    print("HRM training package created successfully!")
    return datasets

if __name__ == "__main__":
    # Example usage
    datasets = create_hrm_training_package()

    print("\nDataset summary:")
    for name, data in datasets.items():
        print(f"  {name}: {data.shape}")

    # Save datasets
    print("\nSaving datasets...")
    for name, data in datasets.items():
        filename = f"benchmark_{name}.csv"
        data.to_csv(filename)
        print(f"  Saved {filename}")