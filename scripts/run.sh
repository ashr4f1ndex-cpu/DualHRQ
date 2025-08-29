#!/usr/bin/env bash
# Ensures the DualHRQ venv is active, then runs whatever command you pass
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Use the project venv; create if missing (failsafe)
if [ ! -d "$ROOT/.venv" ]; then
  echo "⚠️  Virtual environment not found. Running bootstrap first..."
  "$ROOT/scripts/bootstrap.sh"
fi

# Activate the venv
source "$ROOT/.venv/bin/activate"

# Safety: prevent accidental global pip installs
python -m pip config set global.require-virtualenv true >/dev/null 2>&1 || true

# Verify we're in the right environment
if [[ "$VIRTUAL_ENV" != "$ROOT/.venv" ]]; then
  echo "❌ ERROR: Wrong virtual environment active"
  echo "   Expected: $ROOT/.venv"
  echo "   Got: $VIRTUAL_ENV"
  exit 1
fi

# Hand off to the requested command, inside the venv
exec "$@"