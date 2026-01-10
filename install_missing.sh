#!/usr/bin/env bash
set -euo pipefail

echo "🔧 Installerer manglende pakker for Ryfast..."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  for candidate in python3.12 python3.11 python3.10 python3 python; do
    if ! command -v "$candidate" >/dev/null 2>&1; then
      continue
    fi
    if ! "$candidate" -c "import sys" >/dev/null 2>&1; then
      continue
    fi
    if "$candidate" -c "import sys; raise SystemExit(0 if sys.version_info[:2] <= (3, 12) else 1)" >/dev/null 2>&1; then
      PYTHON_BIN="$candidate"
      break
    fi
  done
fi

if [[ -z "$PYTHON_BIN" ]]; then
  echo "❌ Fant ikke en fungerende Python (3.12 eller eldre) i PATH"
  echo "💡 Tips: installer Python 3.12 og kjør: PYTHON_BIN=python3.12 bash install_missing.sh"
  exit 1
fi

VENV_ACTIVATE=""
if [[ -f ".venv/bin/activate" ]]; then
  VENV_ACTIVATE=".venv/bin/activate"
fi

if [[ -z "$VENV_ACTIVATE" ]]; then
  echo "ℹ️ Fant ikke virtuelt miljø i ./.venv"
  echo "🔧 Oppretter .venv..."
  "$PYTHON_BIN" -m venv .venv
  VENV_ACTIVATE=".venv/bin/activate"
fi

# Aktiver virtuelt miljø
source "$VENV_ACTIVATE"

echo "📦 Installerer fra requirements.txt..."
python -m pip install -U pip
python -m pip install -r requirements.txt

echo "✅ Kontrollerer installerte pakker..."
python -m pip list | grep -E "(openpyxl|watchdog|fpdf2|streamlit)" || true

echo "🎯 Test import av openpyxl..."
python -c "import openpyxl; print('✅ openpyxl importerer OK')"

echo "🚀 Nå kan du kjøre: streamlit run ryfast_app/app.py"
