#!/usr/bin/env python3
"""
DualHRQ Production Deployment Script

Complete deployment and execution of the DualHRQ system:
- Validates environment and dependencies
- Configures production-grade HRM model (27M parameters)
- Executes complete trading pipeline
- Generates comprehensive performance reports
- Validates against institutional benchmarks
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
import warnings

# Add lab_v10/src to Python path
sys.path.insert(0, str(Path(__file__).parent / "lab_v10" / "src"))

try:
    import torch
    import numpy as np
    import pandas as pd
    from main_orchestrator import DualHRQOrchestrator, DualHRQConfig
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure all dependencies are installed:")
    print("pip install torch pandas numpy scipy scikit-learn")
    sys.exit(1)

warnings.filterwarnings('ignore')

def setup_logging(log_level: str = "INFO", log_file: str = None) -> None:
    """Setup comprehensive logging."""
    
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Configure logging format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

def validate_environment() -> bool:
    """Validate deployment environment."""
    
    logger = logging.getLogger(__name__)
    logger.info("🔍 Validating deployment environment...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        logger.error("❌ Python 3.8+ required")
        return False
    
    # Check dependencies
    required_packages = [
        'torch', 'numpy', 'pandas', 'scipy', 'sklearn'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"❌ Missing packages: {missing_packages}")
        return False
    
    # Check PyTorch device availability
    logger.info(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        logger.info(f"✅ CUDA available: {torch.cuda.get_device_name()}")
    else:
        logger.info("⚠️  CUDA not available, using CPU")
    
    # Check memory availability
    import psutil
    available_memory_gb = psutil.virtual_memory().available / (1024**3)
    if available_memory_gb < 4:
        logger.warning(f"⚠️  Low memory: {available_memory_gb:.1f}GB available")
    else:
        logger.info(f"✅ Memory available: {available_memory_gb:.1f}GB")
    
    logger.info("✅ Environment validation passed")
    return True

def create_production_config(args) -> DualHRQConfig:
    """Create production-grade DualHRQ configuration."""
    
    # Production HRM configuration (27M parameters)
    hrm_config = {
        'h_dim': 512,
        'l_dim': 256,
        'num_h_layers': 12,
        'num_l_layers': 8,
        'num_heads': 8,
        'dropout': 0.1,
        'max_sequence_length': 256,
        'deq_threshold': 1e-3,
        'max_deq_iterations': 50,
        'use_layer_norm': True,
        'activation_function': 'gelu'
    }
    
    # Calculate date range
    if args.end_date == "today":
        end_date = datetime.now().strftime("%Y-%m-%d")
    else:
        end_date = args.end_date
    
    if args.start_date == "auto":
        # Default to 4 years of history
        start_date = (datetime.now() - timedelta(days=4*365)).strftime("%Y-%m-%d")
    else:
        start_date = args.start_date
    
    return DualHRQConfig(
        hrm_config=hrm_config,
        model_path=args.model_path,
        data_path=args.data_path,
        start_date=start_date,
        end_date=end_date,
        initial_capital=args.capital,
        options_allocation_target=args.options_weight,
        intraday_allocation_target=args.intraday_weight,
        cash_allocation_target=1.0 - args.options_weight - args.intraday_weight,
        number_of_trials=args.trials,
        confidence_level=args.confidence,
        deterministic_seed=args.seed,
        enable_mlops_tracking=args.enable_mlops
    )

def validate_model_parameters(orchestrator: DualHRQOrchestrator) -> bool:
    """Validate HRM model meets parameter requirements."""
    
    logger = logging.getLogger(__name__)
    
    if not orchestrator.hrm_model:
        logger.error("❌ HRM model not initialized")
        return False
    
    total_params = sum(p.numel() for p in orchestrator.hrm_model.parameters())
    
    # Check parameter count constraint from CLAUDE.md
    if not (26_500_000 <= total_params <= 27_500_000):
        logger.error(f"❌ Parameter count {total_params:,} outside required range [26.5M, 27.5M]")
        return False
    
    logger.info(f"✅ HRM model validated: {total_params:,} parameters")
    return True

def generate_performance_report(results: dict, output_dir: Path) -> Path:
    """Generate comprehensive performance report."""
    
    logger = logging.getLogger(__name__)
    
    if not results['success']:
        logger.error("Cannot generate report for failed pipeline")
        return None
    
    report = results['final_report']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create detailed report
    detailed_report = {
        'executive_summary': generate_executive_summary(report),
        'system_configuration': report['system_info'],
        'performance_metrics': extract_performance_metrics(report),
        'risk_analysis': generate_risk_analysis(report),
        'statistical_validation': report['validation_results'],
        'benchmark_comparison': generate_benchmark_comparison(report),
        'deployment_info': {
            'timestamp': timestamp,
            'python_version': sys.version,
            'environment': 'production'
        }
    }
    
    # Save detailed JSON report
    json_path = output_dir / f"dualhrq_report_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(detailed_report, f, indent=2, default=str)
    
    # Generate executive summary
    summary_path = output_dir / f"executive_summary_{timestamp}.txt"
    with open(summary_path, 'w') as f:
        f.write(generate_text_summary(detailed_report))
    
    logger.info(f"📊 Performance reports generated:")
    logger.info(f"   Detailed: {json_path}")
    logger.info(f"   Summary: {summary_path}")
    
    return json_path

def generate_executive_summary(report: dict) -> dict:
    """Generate executive summary."""
    
    backtest_results = report.get('backtest_results', {})
    combined = backtest_results.get('combined_portfolio', {})
    metrics = combined.get('metrics', {})
    validation = report.get('validation_results', {})
    
    # Key performance indicators
    total_return = metrics.get('total_return', 0)
    annual_return = metrics.get('annualized_return', 0) 
    sharpe_ratio = metrics.get('sharpe_ratio', 0)
    max_drawdown = metrics.get('max_drawdown', 0)
    
    # Validation summary
    overall_assessment = validation.get('overall_assessment', {})
    confidence_score = overall_assessment.get('confidence_score', 0)
    
    return {
        'total_return_pct': round(total_return * 100, 2),
        'annual_return_pct': round(annual_return * 100, 2),
        'sharpe_ratio': round(sharpe_ratio, 3),
        'max_drawdown_pct': round(max_drawdown * 100, 2),
        'statistical_confidence': round(confidence_score, 3),
        'model_parameters': report['system_info'].get('model_parameters', 0),
        'validation_status': overall_assessment.get('recommendation', 'Unknown')
    }

def extract_performance_metrics(report: dict) -> dict:
    """Extract detailed performance metrics."""
    
    backtest_results = report.get('backtest_results', {})
    
    return {
        'equity_strategies': backtest_results.get('equity_strategies', {}),
        'options_strategies': backtest_results.get('options_strategies', {}),
        'combined_portfolio': backtest_results.get('combined_portfolio', {})
    }

def generate_risk_analysis(report: dict) -> dict:
    """Generate risk analysis."""
    
    combined = report.get('backtest_results', {}).get('combined_portfolio', {})
    metrics = combined.get('metrics', {})
    
    # Basic risk metrics
    volatility = metrics.get('annualized_volatility', 0)
    max_dd = metrics.get('max_drawdown', 0)
    sharpe = metrics.get('sharpe_ratio', 0)
    
    # Risk assessment
    risk_level = "Low"
    if volatility > 0.25 or abs(max_dd) > 0.20:
        risk_level = "High"
    elif volatility > 0.15 or abs(max_dd) > 0.10:
        risk_level = "Medium"
    
    return {
        'risk_level': risk_level,
        'annualized_volatility': round(volatility * 100, 2),
        'maximum_drawdown_pct': round(max_dd * 100, 2),
        'risk_adjusted_return': round(sharpe, 3),
        'risk_metrics': metrics
    }

def generate_benchmark_comparison(report: dict) -> dict:
    """Generate benchmark comparison."""
    
    metrics = report.get('backtest_results', {}).get('combined_portfolio', {}).get('metrics', {})
    
    annual_return = metrics.get('annualized_return', 0)
    sharpe_ratio = metrics.get('sharpe_ratio', 0)
    
    # Compare against common benchmarks
    benchmarks = {
        'SP500_historical': {'return': 0.10, 'sharpe': 0.8},  # Historical S&P 500
        'hedge_fund_average': {'return': 0.08, 'sharpe': 0.6},  # Hedge fund average
        'quantitative_funds': {'return': 0.12, 'sharpe': 1.0}  # Quant fund benchmark
    }
    
    comparison = {}
    for benchmark, stats in benchmarks.items():
        excess_return = annual_return - stats['return']
        sharpe_diff = sharpe_ratio - stats['sharpe']
        
        comparison[benchmark] = {
            'excess_return_pct': round(excess_return * 100, 2),
            'sharpe_difference': round(sharpe_diff, 3),
            'outperformed': excess_return > 0 and sharpe_diff > 0
        }
    
    return comparison

def generate_text_summary(report: dict) -> str:
    """Generate human-readable text summary."""
    
    summary = report['executive_summary']
    risk = report['risk_analysis']
    benchmarks = report['benchmark_comparison']
    
    text = f"""
{'='*80}
DUALHRQ QUANTITATIVE TRADING SYSTEM - EXECUTIVE SUMMARY
{'='*80}

PERFORMANCE OVERVIEW
--------------------
Total Return:           {summary['total_return_pct']:>8.2f}%
Annualized Return:      {summary['annual_return_pct']:>8.2f}%
Sharpe Ratio:           {summary['sharpe_ratio']:>8.3f}
Maximum Drawdown:       {summary['max_drawdown_pct']:>8.2f}%

SYSTEM SPECIFICATIONS  
---------------------
Model Parameters:       {summary['model_parameters']:>8,}
Risk Level:             {risk['risk_level']:>8}
Validation Confidence:  {summary['statistical_confidence']:>8.3f}
Validation Status:      {summary['validation_status']}

BENCHMARK COMPARISON
--------------------"""
    
    for benchmark, stats in benchmarks.items():
        status = "✅ OUTPERFORMED" if stats['outperformed'] else "❌ UNDERPERFORMED"
        text += f"\n{benchmark:20}: {status} (Excess Return: {stats['excess_return_pct']:+6.2f}%)"
    
    text += f"""

RISK ANALYSIS
-------------
Annualized Volatility:  {risk['annualized_volatility']:>8.2f}%
Risk-Adjusted Return:   {risk['risk_adjusted_return']:>8.3f}
Risk Level:             {risk['risk_level']}

RECOMMENDATION
--------------"""
    
    if summary['statistical_confidence'] > 0.7 and summary['sharpe_ratio'] > 1.0:
        text += "\n✅ APPROVED FOR PRODUCTION DEPLOYMENT"
        text += "\n   Strong statistical evidence of alpha generation."
    elif summary['statistical_confidence'] > 0.5:
        text += "\n⚠️  CONDITIONAL APPROVAL"
        text += "\n   Moderate evidence of skill, monitor closely."
    else:
        text += "\n❌ NOT RECOMMENDED FOR PRODUCTION"
        text += "\n   Insufficient evidence of consistent alpha generation."
    
    text += f"\n\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    text += "\n" + "="*80
    
    return text

def main():
    """Main deployment function."""
    
    parser = argparse.ArgumentParser(
        description="Deploy DualHRQ Quantitative Trading System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--capital", type=float, default=10_000_000,
                       help="Initial capital in USD")
    parser.add_argument("--start-date", default="auto",
                       help="Start date (YYYY-MM-DD) or 'auto' for 4 years")
    parser.add_argument("--end-date", default="today", 
                       help="End date (YYYY-MM-DD) or 'today'")
    parser.add_argument("--data-path", default="data",
                       help="Path to market data directory")
    parser.add_argument("--model-path", 
                       help="Path to pre-trained HRM model weights")
    parser.add_argument("--output-dir", default="results",
                       help="Output directory for reports")
    parser.add_argument("--options-weight", type=float, default=0.4,
                       help="Target allocation to options strategies")
    parser.add_argument("--intraday-weight", type=float, default=0.4,
                       help="Target allocation to intraday strategies")
    parser.add_argument("--trials", type=int, default=100,
                       help="Number of trials for statistical validation")
    parser.add_argument("--confidence", type=float, default=0.95,
                       help="Statistical confidence level")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for deterministic execution")
    parser.add_argument("--log-level", default="INFO",
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help="Logging level")
    parser.add_argument("--log-file", 
                       help="Log file path (optional)")
    parser.add_argument("--enable-mlops", action="store_true",
                       help="Enable MLOps tracking")
    parser.add_argument("--dry-run", action="store_true",
                       help="Validate configuration without running")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    logger = logging.getLogger(__name__)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    logger.info("🚀 DualHRQ Production Deployment Starting...")
    logger.info(f"   Capital: ${args.capital:,.0f}")
    logger.info(f"   Period: {args.start_date} to {args.end_date}")
    logger.info(f"   Allocation: Options {args.options_weight:.0%}, Intraday {args.intraday_weight:.0%}")
    
    # Validate environment
    if not validate_environment():
        logger.error("❌ Environment validation failed")
        return 1
    
    # Create configuration
    try:
        config = create_production_config(args)
        logger.info("✅ Production configuration created")
    except Exception as e:
        logger.error(f"❌ Configuration creation failed: {e}")
        return 1
    
    # Initialize orchestrator
    try:
        orchestrator = DualHRQOrchestrator(config)
        logger.info("✅ DualHRQ orchestrator initialized")
    except Exception as e:
        logger.error(f"❌ Orchestrator initialization failed: {e}")
        return 1
    
    if args.dry_run:
        logger.info("✅ Dry run completed successfully")
        return 0
    
    # Execute pipeline
    try:
        logger.info("🔄 Starting production pipeline execution...")
        results = orchestrator.run_complete_pipeline(args.data_path)
        
        if not results['success']:
            logger.error(f"❌ Pipeline execution failed: {results.get('error_message', 'Unknown error')}")
            return 1
        
        logger.info("✅ Pipeline execution completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Pipeline execution failed: {e}")
        return 1
    
    # Validate model parameters
    if not validate_model_parameters(orchestrator):
        return 1
    
    # Generate performance report
    try:
        report_path = generate_performance_report(results, output_dir)
        if report_path:
            logger.info(f"✅ Performance report generated: {report_path}")
        else:
            logger.warning("⚠️  Report generation failed")
            
    except Exception as e:
        logger.error(f"❌ Report generation failed: {e}")
        return 1
    
    # Final summary
    if results['success']:
        logger.info("🎉 DualHRQ deployment completed successfully!")
        
        # Print key metrics
        final_report = results['final_report']
        combined = final_report.get('backtest_results', {}).get('combined_portfolio', {})
        metrics = combined.get('metrics', {})
        
        if metrics:
            logger.info("📊 Key Performance Metrics:")
            logger.info(f"   Total Return: {metrics.get('total_return', 0):.2%}")
            logger.info(f"   Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
            logger.info(f"   Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
        
        return 0
    else:
        logger.error("❌ DualHRQ deployment failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())