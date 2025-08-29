#!/usr/bin/env bash
# Quick test of the DualHRQ environment setup
set -euo pipefail

echo "🧪 Testing DualHRQ Environment Setup"
echo "===================================="

# Check if we're already in virtual environment
if [[ -z "$VIRTUAL_ENV" ]]; then
    ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    exec "$ROOT/scripts/run.sh" "$ROOT/scripts/test-setup.sh"
fi

echo "🐍 Python Environment:"
python --version
echo "   Virtual env: $(python -c 'import sys; print(sys.prefix)')"
echo "   Packages: $(python -m pip list --format=freeze | wc -l | tr -d ' ') installed"

echo ""
echo "📊 Core Libraries:"
python -c "import numpy as np; print(f'   NumPy: {np.__version__}')"
python -c "import pandas as pd; print(f'   Pandas: {pd.__version__}')"
python -c "import sklearn; print(f'   Scikit-learn: {sklearn.__version__}')"
python -c "import torch; print(f'   PyTorch: {torch.__version__}')"

echo ""
echo "💼 Trading Libraries:"
python -c "import alpaca_trade_api as tradeapi; print(f'   Alpaca API: {tradeapi.__version__}')"
python -c "import fredapi; print('   ✅ FRED API')"
python -c "import sqlalchemy; print(f'   SQLAlchemy: {sqlalchemy.__version__}')"
python -c "import statsmodels; print(f'   Statsmodels: {statsmodels.__version__}')"

echo ""
echo "🔬 DualHRQ Modules:"
cd lab_v10
python -c "from src.trading.regulatory_compliance import RegulatoryComplianceEngine; print('   ✅ Regulatory Compliance')"
python -c "from src.trading.paper_trading import AlpacaPaperTrader; print('   ✅ Paper Trading')"
python -c "from src.trading.realistic_backtester import RealisticBacktester; print('   ✅ Realistic Backtester')"
python -c "from src.validation import statistical_validity; print('   ✅ Statistical Validity')"
python -c "from src.validation.backtest_validation import BacktestValidator; print('   ✅ Backtest Validation')"

echo ""
echo "🧪 Running Key Tests:"
python -m pytest tests/test_determinism.py::test_numpy_repeatability -v --tb=short
echo ""

echo "✨ Environment setup successful!"
echo "🎯 Usage: Always run commands through ./scripts/run.sh"
echo "   Example: ./scripts/run.sh python your_script.py"