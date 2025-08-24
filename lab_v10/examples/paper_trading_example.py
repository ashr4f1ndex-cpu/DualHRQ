#!/usr/bin/env python3
"""
Example usage of DualHRQ Paper Trading System.

This script demonstrates how to set up and run paper trading
with Alpaca Markets integration and kill switches.
"""

import os
import sys
import time
import logging
from datetime import datetime

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.trading.paper_trading import (
    PaperTradingEngine, PaperTradingConfig, KillSwitchConfig
)
from src.trading.strategy_executor import StrategyExecutor, StrategyConfig

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_sample_config() -> PaperTradingConfig:
    """Create sample paper trading configuration."""
    
    # NOTE: You need to replace these with your actual Alpaca API credentials
    # Get them from: https://app.alpaca.markets/paper/dashboard/overview
    alpaca_key = os.getenv("ALPACA_API_KEY", "YOUR_API_KEY_HERE")
    alpaca_secret = os.getenv("ALPACA_SECRET_KEY", "YOUR_SECRET_KEY_HERE")
    
    if alpaca_key == "YOUR_API_KEY_HERE" or alpaca_secret == "YOUR_SECRET_KEY_HERE":
        print("\n⚠️  WARNING: Using placeholder API keys!")
        print("Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables")
        print("or edit this script with your actual Alpaca paper trading credentials.")
        print("\nFor testing purposes, we'll proceed with mock credentials...")
    
    kill_switch_config = KillSwitchConfig(
        max_drawdown_threshold=0.15,      # 15% maximum drawdown
        max_daily_loss_threshold=0.05,    # 5% maximum daily loss
        max_position_size=0.10,           # 10% maximum position size
        market_vol_threshold=0.40,        # 40% market volatility threshold
        enable_all=True                   # Enable all kill switches
    )
    
    config = PaperTradingConfig(
        initial_capital=100000.0,         # $100k starting capital
        alpaca_api_key=alpaca_key,
        alpaca_secret_key=alpaca_secret,
        alpaca_base_url="https://paper-api.alpaca.markets",  # Paper trading URL
        kill_switches=kill_switch_config,
        update_frequency=30,              # Update every 30 seconds
        max_positions=10,                 # Maximum 10 positions
        enable_logging=True,
        enable_alerts=True
    )
    
    return config


def create_strategy_config() -> StrategyConfig:
    """Create strategy configuration."""
    return StrategyConfig(
        max_positions=8,                  # Maximum 8 positions
        max_position_weight=0.12,         # 12% maximum per position
        min_signal_confidence=0.60,       # 60% minimum confidence
        rebalance_frequency=300,          # Rebalance every 5 minutes
        enable_adaptive_sizing=True,      # Enable adaptive position sizing
        target_volatility=0.15           # Target 15% annual volatility
    )


def main():
    """Main paper trading demo."""
    
    print("🤖 DualHRQ Paper Trading System Demo")
    print("=" * 50)
    
    try:
        # Create configurations
        paper_config = create_sample_config()
        strategy_config = create_strategy_config()
        
        print(f"💰 Initial Capital: ${paper_config.initial_capital:,.2f}")
        print(f"🎯 Max Drawdown Threshold: {paper_config.kill_switches.max_drawdown_threshold:.1%}")
        print(f"⚠️  Daily Loss Threshold: {paper_config.kill_switches.max_daily_loss_threshold:.1%}")
        print(f"📊 Update Frequency: {paper_config.update_frequency}s")
        print()
        
        # Create paper trading engine
        print("🚀 Initializing paper trading engine...")
        engine = PaperTradingEngine(paper_config)
        
        # Create strategy executor
        print("📈 Setting up strategy executor...")
        executor = StrategyExecutor(engine, strategy_config)
        
        # Define trading universe
        universe = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']
        print(f"🌍 Trading Universe: {', '.join(universe)}")
        
        # Initialize strategy
        print("⚙️  Initializing trading strategy...")
        executor.initialize_strategy(universe)
        
        # Set up callbacks for monitoring
        def on_performance_update(metrics):
            """Handle performance updates."""
            print(f"💼 Portfolio: ${metrics.portfolio_value:,.2f} "
                  f"(Return: {metrics.total_return:+.2%}, "
                  f"Positions: {metrics.active_positions})")
        
        def on_kill_switch(events):
            """Handle kill switch events."""
            for event in events:
                print(f"🛑 KILL SWITCH: {event.switch_type.value} - {event.message}")
        
        engine.on_performance_update = on_performance_update
        engine.on_kill_switch_triggered = on_kill_switch
        
        print("\n✅ Setup complete! Starting paper trading...")
        print("   Press Ctrl+C to stop trading")
        print("-" * 50)
        
        # Start paper trading
        if engine.start_trading():
            executor.start_execution()
            
            # Get initial metrics
            initial_metrics = engine.get_current_metrics()
            if initial_metrics:
                print(f"🏁 Started with ${initial_metrics.portfolio_value:,.2f}")
            
            # Main trading loop
            cycle_count = 0
            try:
                while True:
                    # Run strategy cycle
                    cycle_result = executor.run_strategy_cycle(universe)
                    cycle_count += 1
                    
                    if cycle_result['status'] == 'success':
                        signals_count = cycle_result.get('signals_generated', 0)
                        orders_executed = cycle_result.get('orders_executed', False)
                        
                        print(f"📊 Cycle #{cycle_count}: "
                              f"{signals_count} signals, "
                              f"Orders: {'✅' if orders_executed else '❌'}")
                    else:
                        print(f"❌ Cycle #{cycle_count} failed: {cycle_result.get('error', 'Unknown error')}")
                    
                    # Show execution summary every 10 cycles
                    if cycle_count % 10 == 0:
                        summary = executor.get_execution_summary()
                        if 'no_data' not in summary:
                            print(f"📈 Execution Summary: "
                                  f"{summary['successful_orders']}/{summary['total_orders']} orders successful "
                                  f"({summary['success_rate']:.1%} success rate)")
                    
                    # Wait before next cycle
                    time.sleep(60)  # Run cycle every minute
                    
            except KeyboardInterrupt:
                print("\n\n🛑 Stopping paper trading...")
                
        else:
            print("❌ Failed to start paper trading!")
            return
            
    except Exception as e:
        print(f"💥 Error: {e}")
        logger.exception("Paper trading demo failed")
        
    finally:
        # Clean shutdown
        try:
            executor.stop_execution()
            engine.stop_trading()
            
            # Final summary
            final_metrics = engine.get_current_metrics()
            if final_metrics:
                print(f"\n📊 Final Results:")
                print(f"   Portfolio Value: ${final_metrics.portfolio_value:,.2f}")
                print(f"   Total Return: {final_metrics.total_return:+.2%}")
                print(f"   Max Drawdown: {final_metrics.drawdown:.2%}")
                print(f"   Active Positions: {final_metrics.active_positions}")
                
                if final_metrics.total_return > 0:
                    print("🎉 Profitable session!")
                else:
                    print("📉 Loss this session - review strategy")
            
            execution_summary = executor.get_execution_summary()
            if 'no_data' not in execution_summary:
                print(f"\n🔄 Trading Activity:")
                print(f"   Total Orders: {execution_summary['total_orders']}")
                print(f"   Success Rate: {execution_summary['success_rate']:.1%}")
                print(f"   Active Positions: {execution_summary['positions_count']}")
            
            print("\n✅ Paper trading stopped successfully")
            
        except Exception as cleanup_error:
            print(f"⚠️  Cleanup error: {cleanup_error}")


if __name__ == "__main__":
    main()