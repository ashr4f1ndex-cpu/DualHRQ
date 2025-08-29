#!/usr/bin/env bash
set -euo pipefail

echo "🚀 DualHRQ Environment Bootstrap"
echo "================================"

# 1) Use Homebrew Python 3.11 explicitly (Apple Silicon)
PY="/opt/homebrew/opt/python@3.11/bin/python3.11"
if ! [ -x "$PY" ]; then
  echo "❌ ERROR: Homebrew python@3.11 not found at $PY"
  echo "   Install with: brew install python@3.11"
  exit 1
fi

echo "✅ Found Python 3.11: $PY"
$PY --version

# 2) Create venv if missing
if [ ! -d ".venv" ]; then
  echo "📦 Creating virtual environment..."
  "$PY" -m venv .venv
else
  echo "✅ Virtual environment already exists"
fi

# 3) Activate venv
echo "🔌 Activating virtual environment..."
source .venv/bin/activate

# 4) Upgrade pip tooling
echo "📋 Upgrading pip and build tools..."
python -m ensurepip --upgrade || true
python -m pip install --upgrade pip setuptools wheel

# 5) Install core dependencies for DualHRQ trading system
echo "📊 Installing core scientific/ML stack..."
python -m pip install --upgrade \
  numpy==1.26.4 \
  scipy==1.13.1 \
  pandas==2.2.2 \
  scikit-learn==1.5.1 \
  statsmodels \
  numba \
  joblib \
  matplotlib==3.9.0

echo "🧠 Installing PyTorch (CPU)..."
python -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cpu

echo "💼 Installing trading/financial libraries..."
python -m pip install --upgrade \
  alpaca-trade-api \
  fredapi \
  sqlalchemy \
  pyportfolioopt

echo "🔧 Installing utilities and validation..."
python -m pip install --upgrade \
  pydantic \
  jsonschema \
  rich \
  tqdm \
  typer \
  pyyaml==6.0.2

echo "🧪 Installing testing and quality tools..."
python -m pip install --upgrade \
  pytest==8.2.2 \
  pytest-cov \
  pytest-xdist \
  pytest-socket \
  ruff \
  mypy \
  bandit

# 6) Save current state
echo "💾 Saving requirements..."
python -m pip freeze > requirements_frozen.txt

echo "✨ Bootstrap complete!"
echo "   Python: $(python -c 'import sys;print(sys.executable)')"
echo "   Packages: $(python -m pip list --format=freeze | wc -l | tr -d ' ') installed"
echo ""
echo "🎯 Usage: Always run commands through ./scripts/run.sh"
echo "   Example: ./scripts/run.sh python -m pytest"