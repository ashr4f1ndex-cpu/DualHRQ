# Phase H: Paper Trading with Alpaca Integration

## Overview

Phase H implements comprehensive paper trading capabilities with real-time execution, kill switches, and risk management controls through Alpaca Markets integration.

## Key Components

### 1. Paper Trading Engine (`paper_trading.py`)

**Core Features:**
- Alpaca Markets API integration for realistic paper trading
- Real-time portfolio monitoring and metrics tracking
- Comprehensive kill switch system for risk management
- Multi-threaded monitoring with configurable update frequencies
- Emergency stop capabilities with order cancellation

**Configuration:**
```python
config = PaperTradingConfig(
    initial_capital=100000.0,
    alpaca_api_key="your_key",
    alpaca_secret_key="your_secret",
    kill_switches=KillSwitchConfig(
        max_drawdown_threshold=0.15,
        max_daily_loss_threshold=0.05,
        enable_all=True
    ),
    update_frequency=60
)
```

### 2. Kill Switch System

**Supported Kill Switches:**
- **Max Drawdown**: Emergency stop if portfolio drops beyond threshold
- **Daily Loss Limit**: Halt trading on excessive single-day losses
- **Position Size Control**: Prevent over-concentration in single positions
- **Market Volatility**: Pause during extreme market conditions
- **Connection Failures**: Handle API disconnections gracefully
- **Strategy Divergence**: Stop if performance deviates significantly from expectations

**Kill Switch Actions:**
- `EMERGENCY_STOP`: Immediate halt with order cancellation
- `PAUSE_TRADING`: Temporary suspension, can be resumed
- Configurable thresholds and sensitivity levels

### 3. Strategy Executor (`strategy_executor.py`)

**Capabilities:**
- Real-time signal generation with configurable strategies
- Portfolio rebalancing with risk-based position sizing
- Order execution with comprehensive logging
- Performance tracking and execution analytics
- Adaptive position sizing based on portfolio volatility

**Signal Generation:**
- Moving average crossover strategy (sample implementation)
- Confidence-based filtering (minimum confidence thresholds)
- Risk-adjusted position sizing
- Pattern recognition integration (when HRM available)

### 4. Position Management

**Features:**
- Dynamic position sizing based on signal confidence
- Portfolio weight limits and diversification controls
- Rebalancing frequency management
- Volatility targeting for risk management
- Real-time position tracking and PnL calculation

## Usage Examples

### Basic Setup
```python
from src.trading.paper_trading import PaperTradingEngine, PaperTradingConfig
from src.trading.strategy_executor import StrategyExecutor, StrategyConfig

# Configure paper trading
config = PaperTradingConfig(
    initial_capital=100000.0,
    alpaca_api_key="your_alpaca_key",
    alpaca_secret_key="your_alpaca_secret"
)

# Create engine and executor
engine = PaperTradingEngine(config)
executor = StrategyExecutor(engine, StrategyConfig())

# Initialize and start
executor.initialize_strategy(['AAPL', 'MSFT', 'GOOGL'])
engine.start_trading()
executor.start_execution()
```

### Kill Switch Monitoring
```python
def on_kill_switch_triggered(events):
    for event in events:
        print(f"Kill switch activated: {event.switch_type}")
        print(f"Reason: {event.message}")

engine.on_kill_switch_triggered = on_kill_switch_triggered
```

### Performance Tracking
```python
def on_performance_update(metrics):
    print(f"Portfolio: ${metrics.portfolio_value:,.2f}")
    print(f"Return: {metrics.total_return:+.2%}")
    print(f"Drawdown: {metrics.drawdown:.2%}")

engine.on_performance_update = on_performance_update
```

## Testing Framework

**Test Coverage:**
- Kill switch functionality with various trigger conditions
- Alpaca API integration with comprehensive mocking
- Strategy execution and signal generation
- Portfolio management and rebalancing
- Error handling and recovery scenarios
- End-to-end integration testing

**Running Tests:**
```bash
# Run all paper trading tests
python -m pytest tests/test_paper_trading.py -v

# Run specific test categories
python -m pytest tests/test_paper_trading.py::TestKillSwitchManager -v
python -m pytest tests/test_paper_trading.py::TestAlpacaPaperTrader -v
python -m pytest tests/test_paper_trading.py::TestStrategyExecutor -v
```

## Risk Management Features

### 1. Multi-Layer Protection
- **Kill Switches**: Automatic trading halts on risk breaches
- **Position Limits**: Maximum exposure per symbol and total
- **Volatility Controls**: Dynamic sizing based on market conditions
- **Connection Monitoring**: Automatic reconnection and failsafe

### 2. Real-Time Monitoring
- Continuous portfolio value tracking
- Drawdown measurement and alerting
- Performance metrics calculation (Sharpe ratio, etc.)
- Trade execution logging and analysis

### 3. Emergency Procedures
- Immediate order cancellation on emergency stop
- Position preservation (no forced liquidation in paper trading)
- Comprehensive event logging for post-analysis
- Manual override capabilities

## Integration Points

### 1. Alpaca Markets API
- Paper trading environment with realistic execution
- Real-time market data feeds
- Order management and position tracking
- Account information and buying power monitoring

### 2. Validation System Integration
- Strategy confidence requirements from Phase G results
- Performance validation against backtesting benchmarks
- Statistical significance monitoring during live trading

### 3. HRM Loop Integration (Optional)
- Pattern recognition for signal generation
- Adaptive learning from live trading results
- Dynamic strategy parameter adjustment

## Configuration Options

### Kill Switch Settings
```python
KillSwitchConfig(
    max_drawdown_threshold=0.15,        # 15% max drawdown
    max_daily_loss_threshold=0.05,      # 5% max daily loss
    max_position_size=0.10,             # 10% max position size
    market_vol_threshold=0.40,          # 40% volatility threshold
    connection_timeout=30,              # 30s connection timeout
    enable_all=True                     # Enable all switches
)
```

### Strategy Settings
```python
StrategyConfig(
    max_positions=10,                   # Maximum positions
    max_position_weight=0.10,           # 10% max per position
    min_signal_confidence=0.60,         # 60% min confidence
    rebalance_frequency=300,            # 5-minute rebalancing
    enable_adaptive_sizing=True,        # Dynamic sizing
    target_volatility=0.15             # 15% target volatility
)
```

## Deployment Considerations

### 1. Prerequisites
- Alpaca Markets paper trading account
- API keys with appropriate permissions
- Stable internet connection for real-time data
- Sufficient computational resources for monitoring

### 2. Production Setup
- Environment variable management for API keys
- Logging configuration for audit trails
- Monitoring alerts and notifications
- Backup and recovery procedures

### 3. Scaling Considerations
- Multiple strategy deployment
- Resource usage monitoring
- Performance optimization
- Error handling and recovery

## Future Enhancements

### 1. Advanced Features
- Multi-broker support beyond Alpaca
- Options and futures trading capabilities
- Advanced order types (stop-loss, take-profit)
- Portfolio optimization algorithms

### 2. Monitoring Improvements
- Web dashboard for real-time monitoring
- Mobile alerts and notifications
- Advanced analytics and reporting
- Machine learning integration for kill switch optimization

### 3. Integration Expansions
- Slack/Discord bot integration
- Email/SMS alerting systems
- Database integration for historical analysis
- Cloud deployment capabilities

## Example Demo Script

A comprehensive example is available at `examples/paper_trading_example.py` demonstrating:
- Complete system setup and configuration
- Real-time monitoring and alerting
- Kill switch handling and emergency procedures
- Performance tracking and reporting

Run with:
```bash
cd examples
python paper_trading_example.py
```

## Security Notes

- **Never commit API keys to version control**
- Use environment variables for sensitive configuration
- Enable paper trading mode only (never live trading without explicit safeguards)
- Implement proper logging without exposing credentials
- Regular security audits of API permissions and access

## Compliance and Risk Disclosure

This system is designed for paper trading and educational purposes. Before any live trading:
- Understand all regulatory requirements
- Implement proper risk management
- Test thoroughly in paper trading environment
- Ensure adequate capital and risk tolerance
- Consider professional financial advice