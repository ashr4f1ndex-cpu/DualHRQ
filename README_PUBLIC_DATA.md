# HRM Public Data Training Enhancement

## 🚀 Complete Public Data Integration for Hierarchical Reasoning Model Training

This comprehensive enhancement adds extensive public data sources and training capabilities to the HRM (Hierarchical Reasoning Model) trading lab, enabling world-class training without expensive data subscriptions.

### 📋 Table of Contents

- [Quick Start](#quick-start)
- [Features Overview](#features-overview)
- [Data Sources](#data-sources)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

---

## ⚡ Quick Start

### 1. Automated Setup (Recommended)

```bash
# Clone and setup
git clone <repository>
cd dual_book_trading_lab_v10_complete

# Run automated setup
python setup_public_data.py --interactive

# Start training with public data
python example_public_data_training.py
```

### 2. Manual Setup

```bash
# Install dependencies
pip install -r requirements_public_data.txt

# Configure data sources
cp config/public_data_config.yaml config/my_config.yaml
# Edit my_config.yaml with your API keys

# Generate benchmark datasets
python -c "from lab_v10.src.common.benchmark_datasets import create_hrm_training_package; create_hrm_training_package()"
```

---

## 🎯 Features Overview

### 🆓 Completely Free Data Sources
- **Yahoo Finance**: Stock prices, options data, historical quotes
- **FRED Economic Data**: VIX, interest rates, economic indicators
- **DoltHub Options**: Community-maintained options database
- **Crypto Options**: Deribit cryptocurrency options (free API)

### 💰 Freemium Sources (Free Tiers Available)
- **Alpha Vantage**: 25 API calls/day (historical data, fundamentals)
- **IEX Cloud**: 50k credits/month (high-quality market data)
- **NewsAPI**: 1000 requests/day (financial news and sentiment)
- **Polygon.io**: Limited free tier (real-time market data)

### 🔬 Benchmark Datasets
- **Synthetic Options Data**: Realistic ATM straddle scenarios
- **Regime-Switching Models**: Bull/bear market simulations
- **Jump-Diffusion Processes**: Crisis scenario modeling
- **Intraday Signal Patterns**: Backside short signal generation

### 🧠 Pre-trained Models
- **HRM Base Financial**: Pre-trained on synthetic financial data
- **Transformer Encoders**: General-purpose financial sequence models
- **Volatility Predictors**: Specialized IV forecasting models
- **Transfer Learning**: Fine-tune pre-trained components

### 📡 Real-time Data Feeds
- **WebSocket Connections**: Live market data streaming
- **Feature Engineering**: Real-time technical indicators
- **Sentiment Analysis**: Live news and social media sentiment
- **Multi-source Aggregation**: Combine multiple data streams

---

## 📊 Data Sources

### Free Sources (No API Key Required)

#### Yahoo Finance
```python
from lab_v10.src.options.data.public_data_loaders import load_yahoo_finance_options

# Load options data
S, iv_entry, iv_exit, expiry = load_yahoo_finance_options(
    symbol='SPY',
    start_date='2023-01-01',
    end_date='2023-12-31'
)
```

#### FRED Economic Data
```python
from lab_v10.src.options.data.public_data_loaders import load_fred_economic_data

# Load VIX data
vix_data = load_fred_economic_data('VIXCLS', '2023-01-01', '2023-12-31')
```

### Freemium Sources (Free Tier)

#### Alpha Vantage (25 calls/day)
```python
# Get free API key: https://www.alphavantage.co/support/#api-key
S, iv_entry, iv_exit, expiry = load_alpha_vantage_data(
    symbol='AAPL',
    api_key='your_api_key',
    start_date='2023-01-01',
    end_date='2023-12-31'
)
```

#### IEX Cloud (50k credits/month)
```python
# Get free token: https://iexcloud.io/cloud-login#/register
S, iv_entry, iv_exit, expiry = load_iex_cloud_data(
    symbol='SPY',
    token='your_token',
    start_date='2023-01-01',
    end_date='2023-12-31'
)
```

### Alternative Data Sources

#### Reddit Sentiment
```python
from lab_v10.src.options.data.public_data_loaders import load_reddit_sentiment

sentiment_data = load_reddit_sentiment(
    subreddit='wallstreetbets',
    symbol='TSLA',
    days_back=30
)
```

#### Cryptocurrency Options
```python
# Deribit crypto options (free API)
S, iv_entry, iv_exit, expiry = load_deribit_options(
    symbol='BTC',
    start_date='2023-01-01',
    end_date='2023-12-31'
)
```

---

## 💻 Installation

### System Requirements
- Python 3.8+
- 8GB+ RAM (16GB recommended)
- 10GB+ free disk space
- Internet connection for data downloads

### Core Installation
```bash
# Install core dependencies
pip install -r requirements_public_data.txt

# For development
pip install -r requirements_dev.txt
```

### Optional Enhancements
```bash
# Technical analysis
pip install ta-lib

# Advanced NLP for sentiment
pip install spacy transformers
python -m spacy download en_core_web_sm

# Distributed computing
pip install dask ray

# Cloud storage
pip install boto3 google-cloud-storage
```

---

## ⚙️ Configuration

### Basic Configuration

Edit `config/public_data_config.yaml`:

```yaml
data_sources:
  free:
    yahoo_finance:
      enabled: true
      symbols: ["SPY", "QQQ", "AAPL", "GOOGL"]
      update_interval: 30
  
  freemium:
    alpha_vantage:
      enabled: true
      api_key: "your_api_key_here"
      rate_limit: 25

benchmark_datasets:
  enabled: true
  synthetic:
    straddle_dataset:
      n_samples: 2000
      regime_switching: true

pretrained_models:
  enabled: true
  models:
    hrm_base_financial:
      enabled: true
      auto_download: true
```

### API Key Setup

1. **Alpha Vantage** (Free - 25 calls/day)
   - Visit: https://www.alphavantage.co/support/#api-key
   - Enter email to get free API key

2. **IEX Cloud** (Free - 50k credits/month)
   - Visit: https://iexcloud.io/cloud-login#/register
   - Create account and copy publishable token

3. **NewsAPI** (Free - 1000 requests/day)
   - Visit: https://newsapi.org/register
   - Create account and copy API key

---

## 🏃 Usage Examples

### Example 1: Basic HRM Training with Yahoo Finance

```python
import pandas as pd
from lab_v10.src.options.data.public_data_loaders import load_yahoo_finance_options
from lab_v10.src.options.hrm_adapter import HRMAdapter
from lab_v10.src.options.pretrained_models import load_pretrained_hrm

# Load data
symbols = ['SPY', 'QQQ', 'AAPL']
all_data = {}

for symbol in symbols:
    S, iv_entry, iv_exit, expiry = load_yahoo_finance_options(
        symbol=symbol,
        start_date='2020-01-01',
        end_date='2023-12-31'
    )
    all_data[symbol] = pd.DataFrame({
        'underlying_price': S,
        'iv_entry': iv_entry,
        'iv_exit': iv_exit,
        'expiry': expiry
    })

# Load pre-trained model with transfer learning
model, transfer_learning = load_pretrained_hrm('hrm_base_financial')

# Setup training
config = {
    'hrm': {
        'h': {'d_model': 256, 'n_layers': 2, 'n_heads': 4},
        'l': {'d_model': 384, 'n_layers': 3, 'n_heads': 6}
    }
}

adapter = HRMAdapter(config)

# Train model (simplified)
# adapter.fit(daily_features, intraday_features, targets_A, targets_B)
```

### Example 2: Multi-Source Data Aggregation

```python
from lab_v10.src.options.data.public_data_loaders import combine_data_sources

# Configure multiple data sources
sources = [
    {'type': 'yahoo', 'name': 'yahoo_primary'},
    {'type': 'alpha_vantage', 'name': 'alpha_secondary', 'api_key': 'your_key'},
    {'type': 'iex', 'name': 'iex_validation', 'token': 'your_token'}
]

# Combine data from multiple sources
combined_data = combine_data_sources(
    sources=sources,
    symbol='SPY',
    start_date='2023-01-01',
    end_date='2023-12-31'
)

# Create ensemble features
from lab_v10.src.options.data.public_data_loaders import create_ensemble_features
ensemble_features = create_ensemble_features(combined_data)
```

### Example 3: Real-time Data Streaming

```python
import asyncio
from lab_v10.src.common.realtime_feeds import RealTimeDataManager, YahooFinanceFeed

async def main():
    # Setup real-time data manager
    symbols = ['SPY', 'AAPL', 'GOOGL']
    data_manager = RealTimeDataManager(symbols)
    
    # Add callback for features
    def on_features(symbol, features):
        print(f"{symbol}: Price={features.get('price_current', 0):.2f}")
    
    data_manager.add_callback(on_features)
    
    # Add Yahoo Finance feed
    yahoo_feed = YahooFinanceFeed(symbols)
    data_manager.add_feed('yahoo', yahoo_feed)
    
    # Start streaming
    await data_manager.start_all_feeds()

# Run the streaming example
asyncio.run(main())
```

### Example 4: Benchmark Dataset Generation

```python
from lab_v10.src.common.benchmark_datasets import create_hrm_training_package

# Generate comprehensive training datasets
datasets = create_hrm_training_package()

# Available datasets:
# - straddle_train: Options straddle backtesting data
# - intraday_signals: Intraday trading signals
# - stress_scenarios: Market crisis scenarios
# - regime_validation: Bull/bear market regimes

# Use for training
straddle_data = datasets['straddle_train']
print(f"Generated {len(straddle_data)} straddle training samples")
```

### Example 5: Transfer Learning with Pre-trained Models

```python
from lab_v10.src.options.pretrained_models import HRMTransferLearning, HRMNet, HRMConfig

# Create target model
config = HRMConfig(
    h_config={'d_model': 512, 'n_layers': 4, 'n_heads': 8},
    l_config={'d_model': 768, 'n_layers': 6, 'n_heads': 12}
)
model = HRMNet(config)

# Setup transfer learning
transfer_learning = HRMTransferLearning(model)
status = transfer_learning.setup_transfer_learning()

if status['status'] == 'success':
    print(f"Transfer learning ready: {status['trainable_params']:,} trainable parameters")
    
    # Get parameter groups for optimizer
    param_groups = transfer_learning.get_parameter_groups()
    
    # Use with PyTorch optimizer
    import torch.optim as optim
    optimizer = optim.AdamW(param_groups)
```

---

## 📚 API Reference

### Core Modules

#### `public_data_loaders.py`
- `load_yahoo_finance_options()`: Yahoo Finance options data
- `load_alpha_vantage_data()`: Alpha Vantage API integration
- `load_iex_cloud_data()`: IEX Cloud API integration
- `load_fred_economic_data()`: Federal Reserve economic data
- `combine_data_sources()`: Multi-source data aggregation

#### `benchmark_datasets.py`
- `FinancialDataGenerator`: Synthetic financial data generation
- `BenchmarkDatasets`: Curated benchmark datasets
- `OptionsDatasetGenerator`: Options-specific synthetic data
- `create_hrm_training_package()`: Complete training dataset package

#### `pretrained_models.py`
- `PretrainedModelRegistry`: Model zoo management
- `TransferLearningManager`: Transfer learning utilities
- `HRMTransferLearning`: HRM-specific transfer learning
- `load_pretrained_hrm()`: Convenience function for pre-trained models

#### `realtime_feeds.py`
- `DataFeed`: Base class for real-time feeds
- `YahooFinanceFeed`: Yahoo Finance real-time data
- `AlphaVantageFeed`: Alpha Vantage real-time API
- `RealTimeDataManager`: Multi-feed management
- `RealTimeFeatureEngine`: Real-time feature engineering

### Configuration

#### `public_data_config.yaml`
Complete configuration file with all available options:
- Data source configurations
- API key settings
- Benchmark dataset parameters
- Pre-trained model settings
- Real-time feed configurations
- Feature engineering options

---

## 🔧 Troubleshooting

### Common Issues

#### Installation Problems
```bash
# If ta-lib installation fails:
# On Ubuntu/Debian:
sudo apt-get install build-essential
pip install ta-lib

# On macOS:
brew install ta-lib
pip install ta-lib

# On Windows:
# Download wheel from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
pip install TA_Lib-0.4.25-cp39-cp39-win_amd64.whl
```

#### API Rate Limits
```python
# Handle rate limits gracefully
import time
from requests.exceptions import HTTPError

def safe_api_call(api_func, *args, **kwargs):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return api_func(*args, **kwargs)
        except HTTPError as e:
            if e.response.status_code == 429:  # Rate limit
                wait_time = 2 ** attempt  # Exponential backoff
                time.sleep(wait_time)
            else:
                raise
    raise Exception("Max retries exceeded")
```

#### Memory Issues with Large Datasets
```python
# Use chunked processing for large datasets
def process_large_dataset(data_source, chunk_size=10000):
    for chunk in pd.read_csv(data_source, chunksize=chunk_size):
        # Process chunk
        processed_chunk = process_chunk(chunk)
        yield processed_chunk
```

#### Data Quality Issues
```python
# Validate data quality
def validate_data_quality(df):
    issues = []
    
    # Check for missing data
    missing_pct = df.isnull().sum() / len(df)
    if missing_pct.max() > 0.05:
        issues.append(f"High missing data: {missing_pct.max():.2%}")
    
    # Check for outliers
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        q99 = df[col].quantile(0.99)
        q01 = df[col].quantile(0.01)
        outliers = ((df[col] > q99) | (df[col] < q01)).sum()
        if outliers / len(df) > 0.02:
            issues.append(f"High outliers in {col}: {outliers/len(df):.2%}")
    
    return issues
```

### Performance Optimization

#### Parallel Data Loading
```python
import concurrent.futures
from functools import partial

def parallel_data_loading(symbols, data_loader_func, max_workers=4):
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(data_loader_func, symbol): symbol 
            for symbol in symbols
        }
        
        # Collect results
        results = {}
        for future in concurrent.futures.as_completed(futures):
            symbol = futures[future]
            try:
                results[symbol] = future.result()
            except Exception as e:
                print(f"Error loading {symbol}: {e}")
        
        return results
```

#### Memory-Efficient Feature Engineering
```python
# Use generators for memory efficiency
def generate_features(data_stream, window_size=20):
    buffer = []
    
    for data_point in data_stream:
        buffer.append(data_point)
        
        if len(buffer) >= window_size:
            # Compute features on window
            features = compute_technical_indicators(buffer)
            yield features
            
            # Slide window
            buffer = buffer[1:]
```

---

## 🤝 Contributing

### Development Setup
```bash
# Clone repository
git clone <repository>
cd dual_book_trading_lab_v10_complete

# Create development environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install development dependencies
pip install -r requirements_dev.txt

# Install pre-commit hooks
pre-commit install
```

### Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=lab_v10 --cov-report=html

# Run specific test module
pytest tests/test_public_data_loaders.py
```

### Code Quality
```bash
# Format code
black lab_v10/

# Lint code
flake8 lab_v10/

# Type checking
mypy lab_v10/
```

### Adding New Data Sources

1. **Create loader function** in `public_data_loaders.py`:
```python
def load_new_data_source(symbol: str, api_key: str, **kwargs) -> Tuple[pd.Series, ...]:
    # Implement data loading logic
    pass
```

2. **Add configuration** in `public_data_config.yaml`:
```yaml
data_sources:
  freemium:
    new_source:
      enabled: false
      api_key: "your_api_key"
      rate_limit: 1000
```

3. **Update setup script** in `setup_public_data.py`:
```python
def setup_new_source(self, config: Dict):
    # Add setup logic
    pass
```

4. **Add tests** in `tests/test_public_data_loaders.py`:
```python
def test_load_new_data_source():
    # Test the new data source
    pass
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Yahoo Finance** for providing free financial data
- **Federal Reserve** for FRED economic data
- **Alpha Vantage** for freemium financial APIs
- **IEX Cloud** for high-quality market data
- **DoltHub** for community options data
- **Deribit** for cryptocurrency options data
- **HRM Research Team** for the original Hierarchical Reasoning Model

---

## 📞 Support

For questions, issues, or feature requests:

1. **Check the documentation** in this README
2. **Search existing issues** on GitHub
3. **Create a new issue** with detailed information
4. **Join the community** discussions

---

## 🔮 Roadmap

### Upcoming Features

- [ ] **More Data Sources**: Additional free and freemium APIs
- [ ] **Advanced NLP**: Better sentiment analysis and news processing
- [ ] **Real-time Alerts**: Market event detection and notifications
- [ ] **Cloud Integration**: AWS/GCP/Azure deployment templates
- [ ] **Mobile Interface**: React Native app for monitoring
- [ ] **API Gateway**: RESTful API for external integrations
- [ ] **Kubernetes Deployment**: Container orchestration templates
- [ ] **Edge Computing**: Raspberry Pi and edge device support

### Long-term Vision

- **Community Data Marketplace**: User-contributed datasets
- **Federated Learning**: Privacy-preserving collaborative training
- **AutoML Integration**: Automated model selection and tuning
- **Explainable AI**: Model interpretability for regulatory compliance
- **Quantum Computing**: Quantum-enhanced optimization algorithms

---

*Made with ❤️ for the quantitative finance community*