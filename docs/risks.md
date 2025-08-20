# Risk Management

## Monitoring
- Deflated Sharpe ratio degradation
- Portfolio turnover spikes  
- Borrow availability shocks
- Model prediction drift

## Playbooks
- SSR lockouts: switch to long-only mode
- LULD halts: pause new entries, monitor reopening
- API degradation: fallback to cached data
- Model failures: revert to baseline strategies

## Fallbacks
- Baseline mean-reversion model
- Circuit breakers for drawdown limits
- Emergency liquidation procedures
- Manual override protocols
