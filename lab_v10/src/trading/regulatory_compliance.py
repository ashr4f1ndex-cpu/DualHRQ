"""
Regulatory Compliance System for DualHRQ Backtesting.

Implements realistic regulatory constraints for backtesting including:
- SSR (Short Sale Restriction): Next-day persistence, uptick-only execution
- LULD (Limit Up/Limit Down): Circuit breakers with last-25-min band doubling
- Trading halts and execution delays
- Compliance validation and reporting

Critical for ensuring backtest realism matches live trading constraints.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta, time
from decimal import Decimal, ROUND_HALF_UP
import warnings

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class RegulatoryRule(Enum):
    """Types of regulatory rules."""
    SSR = "ssr"  # Short Sale Restriction
    LULD = "luld"  # Limit Up/Limit Down
    TRADING_HALT = "halt"
    CIRCUIT_BREAKER = "circuit_breaker"


@dataclass
class SSRStatus:
    """Short Sale Restriction status for a symbol."""
    symbol: str
    is_restricted: bool
    restriction_date: datetime
    trigger_price: float
    restriction_expires: datetime
    last_uptick_price: Optional[float] = None
    last_price_direction: Optional[str] = None  # 'up', 'down', 'flat'


@dataclass
class LULDBands:
    """LULD price bands for a symbol."""
    symbol: str
    reference_price: float
    lower_band: float
    upper_band: float
    band_percentage: float
    is_doubled: bool  # True if last 25 minutes doubling is active
    last_updated: datetime
    

@dataclass
class TradingHalt:
    """Trading halt information."""
    symbol: str
    halt_start: datetime
    halt_end: Optional[datetime]
    halt_reason: str
    halt_type: RegulatoryRule


@dataclass
class ComplianceViolation:
    """Record of compliance violation."""
    timestamp: datetime
    symbol: str
    violation_type: RegulatoryRule
    description: str
    attempted_action: str
    severity: str  # 'warning', 'error', 'critical'


class RegulatoryComplianceEngine:
    """
    Engine for enforcing trading regulations in backtests.
    
    Implements SSR, LULD, and other regulatory constraints to ensure
    backtesting reflects real market conditions and execution limitations.
    """
    
    def __init__(self, enable_ssr: bool = True, enable_luld: bool = True,
                 market_hours: Tuple[time, time] = (time(9, 30), time(16, 0))):
        """
        Initialize regulatory compliance engine.
        
        Args:
            enable_ssr: Enable Short Sale Restriction enforcement
            enable_luld: Enable LULD circuit breaker enforcement  
            market_hours: Regular market hours (start, end)
        """
        self.enable_ssr = enable_ssr
        self.enable_luld = enable_luld
        self.market_hours = market_hours
        
        # SSR tracking
        self.ssr_status: Dict[str, SSRStatus] = {}
        self.ssr_triggers: List[Dict] = []
        
        # LULD tracking
        self.luld_bands: Dict[str, LULDBands] = {}
        self.luld_violations: List[Dict] = []
        
        # Trading halts
        self.active_halts: Dict[str, TradingHalt] = {}
        self.halt_history: List[TradingHalt] = []
        
        # Violations log
        self.violations: List[ComplianceViolation] = []
        
        # Configuration
        self.ssr_threshold = 0.10  # 10% decline triggers SSR
        self.luld_tiers = self._initialize_luld_tiers()
        
        logger.info(f"Regulatory compliance engine initialized (SSR: {enable_ssr}, LULD: {enable_luld})")
    
    def _initialize_luld_tiers(self) -> Dict[str, float]:
        """Initialize LULD band percentages by stock tier."""
        return {
            'tier1': 0.05,    # 5% bands for Tier 1 NMS stocks (>$3, S&P 500/Russell 1000)
            'tier2': 0.10,    # 10% bands for other NMS stocks
            'etf': 0.05,      # 5% bands for most ETFs
            'otc': 0.20       # 20% bands for OTC securities
        }
    
    def update_market_data(self, symbol: str, timestamp: datetime, 
                          price: float, volume: int, 
                          high: float = None, low: float = None) -> Dict[str, Any]:
        """
        Update market data and check for regulatory triggers.
        
        Args:
            symbol: Stock symbol
            timestamp: Current timestamp
            price: Current price
            volume: Current volume
            high: Session high price
            low: Session low price
            
        Returns:
            Dictionary with compliance status and any violations
        """
        compliance_status = {
            'symbol': symbol,
            'timestamp': timestamp,
            'price': price,
            'ssr_restricted': False,
            'luld_violation': False,
            'trading_halted': False,
            'violations': []
        }
        
        # Update SSR status
        if self.enable_ssr:
            ssr_result = self._update_ssr_status(symbol, timestamp, price, high, low)
            compliance_status.update(ssr_result)
        
        # Update LULD bands and check violations
        if self.enable_luld:
            luld_result = self._update_luld_status(symbol, timestamp, price)
            compliance_status.update(luld_result)
        
        # Check for active trading halts
        halt_status = self._check_trading_halts(symbol, timestamp)
        compliance_status.update(halt_status)
        
        return compliance_status
    
    def _update_ssr_status(self, symbol: str, timestamp: datetime, 
                          price: float, high: float = None, low: float = None) -> Dict[str, Any]:
        """Update SSR status for symbol."""
        result = {'ssr_restricted': False, 'ssr_info': {}}
        
        # Get or create SSR status
        if symbol not in self.ssr_status:
            self.ssr_status[symbol] = SSRStatus(
                symbol=symbol,
                is_restricted=False,
                restriction_date=timestamp,
                trigger_price=price,
                restriction_expires=timestamp + timedelta(days=1),
                last_uptick_price=price
            )
        
        ssr = self.ssr_status[symbol]
        
        # Check if current restriction has expired
        if ssr.is_restricted and timestamp >= ssr.restriction_expires:
            ssr.is_restricted = False
            logger.info(f"SSR restriction expired for {symbol} at {timestamp}")
        
        # Check for new SSR trigger (10% decline from previous close)
        if not ssr.is_restricted and self._is_trading_day_start(timestamp):
            # Reset for new trading day
            ssr.trigger_price = price
            ssr.last_uptick_price = price
        
        # Calculate intraday decline if we have high/low
        if high is not None and low is not None:
            daily_decline = (ssr.trigger_price - low) / ssr.trigger_price
            
            if daily_decline >= self.ssr_threshold and not ssr.is_restricted:
                # Trigger SSR
                ssr.is_restricted = True
                ssr.restriction_date = timestamp
                ssr.restriction_expires = self._get_next_trading_day(timestamp) + timedelta(hours=16)
                
                self.ssr_triggers.append({
                    'symbol': symbol,
                    'trigger_time': timestamp,
                    'trigger_price': ssr.trigger_price,
                    'decline_price': low,
                    'decline_percent': daily_decline * 100
                })
                
                logger.warning(f"SSR triggered for {symbol}: {daily_decline:.1%} decline at {timestamp}")
        
        # Update price direction for uptick rule
        if ssr.last_uptick_price is not None:
            if price > ssr.last_uptick_price:
                ssr.last_price_direction = 'up'
                ssr.last_uptick_price = price
            elif price < ssr.last_uptick_price:
                ssr.last_price_direction = 'down'
            else:
                ssr.last_price_direction = 'flat'
        
        result['ssr_restricted'] = ssr.is_restricted
        result['ssr_info'] = {
            'can_short_sell': not ssr.is_restricted or ssr.last_price_direction == 'up',
            'last_direction': ssr.last_price_direction,
            'restriction_expires': ssr.restriction_expires,
            'trigger_price': ssr.trigger_price
        }
        
        return result
    
    def _update_luld_status(self, symbol: str, timestamp: datetime, price: float) -> Dict[str, Any]:
        """Update LULD bands and check for violations."""
        result = {'luld_violation': False, 'luld_info': {}}
        
        # Get or create LULD bands
        if symbol not in self.luld_bands:
            tier = self._get_stock_tier(symbol, price)
            band_pct = self.luld_tiers[tier]
            
            self.luld_bands[symbol] = LULDBands(
                symbol=symbol,
                reference_price=price,
                lower_band=price * (1 - band_pct),
                upper_band=price * (1 + band_pct),
                band_percentage=band_pct,
                is_doubled=False,
                last_updated=timestamp
            )
        
        bands = self.luld_bands[symbol]
        
        # Check if we need to double bands (last 25 minutes of trading)
        is_last_25_min = self._is_last_25_minutes(timestamp)
        if is_last_25_min and not bands.is_doubled:
            bands.band_percentage *= 2
            bands.lower_band = bands.reference_price * (1 - bands.band_percentage)
            bands.upper_band = bands.reference_price * (1 + bands.band_percentage)
            bands.is_doubled = True
            bands.last_updated = timestamp
            
            logger.info(f"LULD bands doubled for {symbol} - last 25 minutes of trading")
        
        # Reset doubled bands for new trading day
        elif not is_last_25_min and bands.is_doubled and self._is_trading_day_start(timestamp):
            bands.band_percentage /= 2
            bands.lower_band = bands.reference_price * (1 - bands.band_percentage)
            bands.upper_band = bands.reference_price * (1 + bands.band_percentage)
            bands.is_doubled = False
            bands.last_updated = timestamp
        
        # Check for LULD violations
        luld_violation = False
        violation_type = None
        
        if price <= bands.lower_band:
            luld_violation = True
            violation_type = 'limit_down'
            self._trigger_luld_halt(symbol, timestamp, price, 'limit_down', bands)
            
        elif price >= bands.upper_band:
            luld_violation = True
            violation_type = 'limit_up'
            self._trigger_luld_halt(symbol, timestamp, price, 'limit_up', bands)
        
        result['luld_violation'] = luld_violation
        result['luld_info'] = {
            'lower_band': bands.lower_band,
            'upper_band': bands.upper_band,
            'reference_price': bands.reference_price,
            'band_percentage': bands.band_percentage,
            'is_doubled': bands.is_doubled,
            'violation_type': violation_type
        }
        
        return result
    
    def _trigger_luld_halt(self, symbol: str, timestamp: datetime, 
                          price: float, violation_type: str, bands: LULDBands):
        """Trigger LULD trading halt."""
        halt_duration = timedelta(minutes=5)  # Standard 5-minute halt
        
        halt = TradingHalt(
            symbol=symbol,
            halt_start=timestamp,
            halt_end=timestamp + halt_duration,
            halt_reason=f"LULD {violation_type}: price {price:.2f} outside band [{bands.lower_band:.2f}, {bands.upper_band:.2f}]",
            halt_type=RegulatoryRule.LULD
        )
        
        self.active_halts[symbol] = halt
        self.halt_history.append(halt)
        
        self.luld_violations.append({
            'symbol': symbol,
            'timestamp': timestamp,
            'price': price,
            'violation_type': violation_type,
            'lower_band': bands.lower_band,
            'upper_band': bands.upper_band,
            'halt_duration': halt_duration.total_seconds()
        })
        
        logger.warning(f"LULD halt triggered for {symbol}: {violation_type} at {price:.2f}")
    
    def _check_trading_halts(self, symbol: str, timestamp: datetime) -> Dict[str, Any]:
        """Check if symbol is currently halted."""
        result = {'trading_halted': False, 'halt_info': {}}
        
        if symbol in self.active_halts:
            halt = self.active_halts[symbol]
            
            if halt.halt_end and timestamp >= halt.halt_end:
                # Halt has ended
                del self.active_halts[symbol]
                logger.info(f"Trading halt ended for {symbol} at {timestamp}")
                result['trading_halted'] = False  # Explicitly set to false
            else:
                # Still halted
                result['trading_halted'] = True
                result['halt_info'] = {
                    'halt_start': halt.halt_start,
                    'halt_end': halt.halt_end,
                    'halt_reason': halt.halt_reason,
                    'halt_type': halt.halt_type.value
                }
        
        return result
    
    def validate_order(self, symbol: str, side: str, quantity: int, 
                      price: float, timestamp: datetime) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Validate order against regulatory constraints.
        
        Args:
            symbol: Stock symbol
            side: 'buy' or 'sell' or 'short'
            quantity: Order quantity
            price: Order price
            timestamp: Order timestamp
            
        Returns:
            Tuple of (is_valid, rejection_reason, compliance_info)
        """
        compliance_info = {'violations': [], 'warnings': []}
        
        # Check if trading is halted (and clear expired halts)
        halt_status = self._check_trading_halts(symbol, timestamp)
        if halt_status['trading_halted']:
            halt = self.active_halts[symbol]
            violation = ComplianceViolation(
                timestamp=timestamp,
                symbol=symbol,
                violation_type=halt.halt_type,
                description=f"Order rejected: {halt.halt_reason}",
                attempted_action=f"{side} {quantity} shares at {price}",
                severity='error'
            )
            self.violations.append(violation)
            compliance_info['violations'].append(violation)
            
            return False, f"Trading halted: {halt.halt_reason}", compliance_info
        
        # Check SSR constraints for short sales
        if side.lower() == 'short' and self.enable_ssr and symbol in self.ssr_status:
            ssr = self.ssr_status[symbol]
            
            if ssr.is_restricted and ssr.last_price_direction != 'up':
                violation = ComplianceViolation(
                    timestamp=timestamp,
                    symbol=symbol,
                    violation_type=RegulatoryRule.SSR,
                    description="Short sale rejected: SSR active, no uptick",
                    attempted_action=f"short {quantity} shares at {price}",
                    severity='error'
                )
                self.violations.append(violation)
                compliance_info['violations'].append(violation)
                
                return False, "SSR violation: Short sales allowed only on upticks", compliance_info
        
        # Check LULD band violations
        if self.enable_luld and symbol in self.luld_bands:
            bands = self.luld_bands[symbol]
            
            # Check if order price would violate LULD bands
            if side.lower() == 'buy' and price > bands.upper_band:
                violation = ComplianceViolation(
                    timestamp=timestamp,
                    symbol=symbol,
                    violation_type=RegulatoryRule.LULD,
                    description=f"Buy order above upper LULD band: {price} > {bands.upper_band}",
                    attempted_action=f"buy {quantity} shares at {price}",
                    severity='warning'
                )
                self.violations.append(violation)
                compliance_info['warnings'].append(violation)
                
            elif side.lower() in ['sell', 'short'] and price < bands.lower_band:
                violation = ComplianceViolation(
                    timestamp=timestamp,
                    symbol=symbol,
                    violation_type=RegulatoryRule.LULD,
                    description=f"Sell order below lower LULD band: {price} < {bands.lower_band}",
                    attempted_action=f"{side} {quantity} shares at {price}",
                    severity='warning'
                )
                self.violations.append(violation)
                compliance_info['warnings'].append(violation)
        
        # Check market hours
        if not self._is_market_hours(timestamp):
            violation = ComplianceViolation(
                timestamp=timestamp,
                symbol=symbol,
                violation_type=RegulatoryRule.TRADING_HALT,
                description="Order outside market hours",
                attempted_action=f"{side} {quantity} shares at {price}",
                severity='warning'
            )
            self.violations.append(violation)
            compliance_info['warnings'].append(violation)
        
        return True, "Order validated", compliance_info
    
    def _get_stock_tier(self, symbol: str, price: float) -> str:
        """Determine LULD tier for stock (simplified)."""
        # In production, this would check against actual tier lists
        if price >= 3.0:
            # Assume Tier 1 for stocks >= $3
            return 'tier1'
        else:
            # Tier 2 for stocks < $3
            return 'tier2'
    
    def _is_trading_day_start(self, timestamp: datetime) -> bool:
        """Check if timestamp is start of trading day."""
        market_open = self.market_hours[0]
        current_time = timestamp.time()
        
        # Consider first 30 minutes as "start"
        start_window = timedelta(minutes=30)
        market_open_dt = datetime.combine(timestamp.date(), market_open)
        
        return market_open_dt <= timestamp <= market_open_dt + start_window
    
    def _is_last_25_minutes(self, timestamp: datetime) -> bool:
        """Check if timestamp is in last 25 minutes of trading."""
        market_close = self.market_hours[1]
        current_time = timestamp.time()
        
        # Last 25 minutes before close
        close_window = timedelta(minutes=25)
        market_close_dt = datetime.combine(timestamp.date(), market_close)
        
        return market_close_dt - close_window <= timestamp <= market_close_dt
    
    def _is_market_hours(self, timestamp: datetime) -> bool:
        """Check if timestamp is during regular market hours."""
        current_time = timestamp.time()
        return self.market_hours[0] <= current_time <= self.market_hours[1]
    
    def _get_next_trading_day(self, current_date: datetime) -> datetime:
        """Get next trading day (simplified - excludes holidays)."""
        next_day = current_date + timedelta(days=1)
        
        # Skip weekends
        while next_day.weekday() >= 5:  # Saturday = 5, Sunday = 6
            next_day += timedelta(days=1)
        
        return next_day
    
    def get_compliance_report(self) -> Dict[str, Any]:
        """Generate comprehensive compliance report."""
        total_violations = len(self.violations)
        
        violations_by_type = {}
        for violation in self.violations:
            rule_type = violation.violation_type.value
            if rule_type not in violations_by_type:
                violations_by_type[rule_type] = {'error': 0, 'warning': 0, 'critical': 0}
            violations_by_type[rule_type][violation.severity] += 1
        
        # SSR statistics
        ssr_stats = {
            'total_triggers': len(self.ssr_triggers),
            'currently_restricted': sum(1 for ssr in self.ssr_status.values() if ssr.is_restricted),
            'symbols_tracked': len(self.ssr_status)
        }
        
        # LULD statistics  
        luld_stats = {
            'total_violations': len(self.luld_violations),
            'active_halts': len(self.active_halts),
            'total_halts': len(self.halt_history),
            'symbols_tracked': len(self.luld_bands)
        }
        
        return {
            'total_violations': total_violations,
            'violations_by_type': violations_by_type,
            'ssr_statistics': ssr_stats,
            'luld_statistics': luld_stats,
            'active_halts': list(self.active_halts.keys()),
            'recent_violations': self.violations[-10:] if self.violations else []
        }
    
    def reset(self):
        """Reset compliance engine state."""
        self.ssr_status.clear()
        self.ssr_triggers.clear()
        self.luld_bands.clear()
        self.luld_violations.clear()
        self.active_halts.clear()
        self.halt_history.clear()
        self.violations.clear()
        
        logger.info("Regulatory compliance engine reset")


class BacktestComplianceValidator:
    """
    Validator for ensuring backtest compliance with regulations.
    
    Provides post-backtest analysis to verify that simulated trades
    would have been executed under real market conditions.
    """
    
    def __init__(self, compliance_engine: RegulatoryComplianceEngine):
        self.compliance_engine = compliance_engine
        self.trade_validations: List[Dict] = []
        
    def validate_backtest_trades(self, trades_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate all trades from backtest against regulatory rules.
        
        Args:
            trades_df: DataFrame with columns: timestamp, symbol, side, quantity, price
            
        Returns:
            Validation report with compliance statistics
        """
        invalid_trades = []
        valid_trades = 0
        warnings = []
        
        for idx, trade in trades_df.iterrows():
            is_valid, reason, compliance_info = self.compliance_engine.validate_order(
                symbol=trade['symbol'],
                side=trade['side'],
                quantity=trade['quantity'], 
                price=trade['price'],
                timestamp=trade['timestamp']
            )
            
            validation_result = {
                'trade_id': idx,
                'is_valid': is_valid,
                'reason': reason,
                'compliance_info': compliance_info,
                'trade': trade.to_dict()
            }
            
            self.trade_validations.append(validation_result)
            
            if is_valid:
                valid_trades += 1
            else:
                invalid_trades.append(validation_result)
            
            # Count warnings
            if compliance_info.get('warnings'):
                warnings.extend(compliance_info['warnings'])
        
        validation_report = {
            'total_trades': len(trades_df),
            'valid_trades': valid_trades,
            'invalid_trades': len(invalid_trades),
            'warnings': len(warnings),
            'compliance_rate': valid_trades / len(trades_df) if len(trades_df) > 0 else 0,
            'invalid_trade_details': invalid_trades,
            'warning_details': warnings,
            'compliance_engine_report': self.compliance_engine.get_compliance_report()
        }
        
        return validation_report