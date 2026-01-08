#!/usr/bin/env bash
set -euo pipefail

echo "🔧 Installerer manglende pakker for Ryfast..."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_ACTIVATE=""
if [[ -f ".venv/bin/activate" ]]; then
  VENV_ACTIVATE=".venv/bin/activate"
elif [[ -f "myenv/bin/activate" ]]; then
  VENV_ACTIVATE="myenv/bin/activate"
fi

if [[ -z "$VENV_ACTIVATE" ]]; then
  echo "ℹ️ Fant ikke virtuelt miljø i ./.venv eller ./myenv"
  echo "🔧 Oppretter .venv..."
  python -m venv .venv
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
