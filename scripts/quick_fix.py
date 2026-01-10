import logging
import numpy as np
import requests
import pandas as pd
import plotly.express as px
import streamlit as st
import time
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import io

# QUICK FIX: Alternative export function without openpyxl
def export_to_csv_alternative(df: pd.DataFrame, filename: str) -> str:
    """Export DataFrame to CSV as alternative to Excel"""
    csv_string = df.to_csv(index=False, sep=';')
    return csv_string

def create_export_section_alternative(df: pd.DataFrame, point: str):
    """Create export functionality section without Excel dependency"""
    st.subheader("📊 Eksporter data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv_data = export_to_csv_alternative(df, f"{point}_trafikkdata.csv")
        st.download_button(
            label="📁 Last ned CSV",
            data=csv_data,
            file_name=f"{point}_trafikkdata_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    with col2:
        json_data = df.to_json(orient='records', indent=2)
        st.download_button(
            label="🔗 Last ned JSON",
            data=json_data,
            file_name=f"{point}_trafikkdata_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json"
        )
    
    # Show CSV preview
    if st.button("👀 Forhåndsvis CSV"):
        st.text_area("CSV Data:", csv_data, height=200)

# INSTRUKS FOR Å FIKSE OPENPYXL-PROBLEMET:
def show_fix_instructions():
    st.error("❌ openpyxl mangler i det virtuelle miljøet")
    
    with st.expander("🔧 Slik fikser du det:"):
        st.code("""
# I terminal:
cd /Users/tor.inge.jossang@aftenbladet.no/dev/sa-ryfast
source .venv/bin/activate
python -m pip install openpyxl==3.1.5

# Eller kjør install-scriptet:
bash install_missing.sh
        """)
        
        st.info("💡 Alternativt kan du bruke CSV-export i stedet for Excel")

# Test om openpyxl er tilgjengelig
try:
    import openpyxl
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False
    st.warning("⚠️ openpyxl ikke tilgjengelig - bruker alternativ export")

# I din main() funksjon, erstatt export-delen med:
if OPENPYXL_AVAILABLE:
    # Bruk original export_to_excel funksjon
    pass
else:
    # Bruk alternativ export
    create_export_section_alternative(df, point)
    show_fix_instructions()
