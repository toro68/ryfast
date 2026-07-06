"""Eksportseksjonen: nedlasting av CSV, Excel og PDF."""

from datetime import datetime
from typing import Optional

import pandas as pd
import streamlit as st

from ryfast_app.exports.excel import OPENPYXL_AVAILABLE, build_excel_report, export_to_excel
from ryfast_app.exports.pdf import build_pdf_report


def create_export_section(df: pd.DataFrame, point: str, coverage_summary: Optional[pd.DataFrame] = None):
    st.subheader("📊 Eksporter data")
    col1, col2, col3 = st.columns(3)

    with col1:
        csv_data = df.to_csv(index=False, sep=";", encoding="utf-8")
        st.download_button(
            label="📁 Last ned CSV",
            data=csv_data,
            file_name=f"{point}_trafikkdata_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )

    with col2:
        json_data = df.to_json(orient="records", indent=2)
        st.download_button(
            label="🔗 Last ned JSON",
            data=json_data,
            file_name=f"{point}_trafikkdata_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json",
        )

    with col3:
        if OPENPYXL_AVAILABLE:
            st.download_button(
                label="📄 Last ned Excel",
                data=export_to_excel(df),
                file_name=f"{point}_trafikkdata_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        else:
            st.caption("Excel-export krever `openpyxl`.")

    if coverage_summary is not None and not coverage_summary.empty:
        st.markdown("#### 🛡️ Eksporter dekning")
        c1, c2 = st.columns(2)
        with c1:
            cov_csv = coverage_summary.to_csv(index=False, sep=";", encoding="utf-8")
            st.download_button(
                label="📁 Last ned dekning (CSV)",
                data=cov_csv,
                file_name=f"{point}_dekning_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
            )
        with c2:
            cov_json = coverage_summary.to_json(orient="records", indent=2)
            st.download_button(
                label="🔗 Last ned dekning (JSON)",
                data=cov_json,
                file_name=f"{point}_dekning_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json",
            )

    st.markdown("### 📦 Ferdig rapport (anbefalt)")
    st.caption("Inkluderer metadata, data, dekning og (der mulig) grafer i Excel/PDF.")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("📊 Lag Excel-rapport", type="secondary"):
            result = st.session_state.get("last_result") or {}
            report_bytes = build_excel_report(
                df=df,
                point=point,
                comparison_mode=result.get("comparison_mode", ""),
                year_list=result.get("year_list", []),
                year=int(result.get("year", 0) or 0),
                point_ids=result.get("point_ids", []),
                timeout_s=int(result.get("timeout_s", 60)),
                use_cache=bool(result.get("use_cache", True)),
                coverage_threshold=float(result.get("coverage_threshold", 90)),
                ryfast_include_ramp=bool(st.session_state.get("ryfast_include_ramp", True)),
                estimate_missing_points=bool(result.get("estimate_missing_points", False)),
            )
            st.download_button(
                "⬇️ Last ned Excel-rapport",
                data=report_bytes,
                file_name=f"ryfast_rapport_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
    with c2:
        if st.button("📄 Lag PDF-rapport", type="secondary"):
            result = st.session_state.get("last_result") or {}
            report_bytes = build_pdf_report(
                df=df,
                point=point,
                comparison_mode=result.get("comparison_mode", ""),
                year_list=result.get("year_list", []),
                year=int(result.get("year", 0) or 0),
                point_ids=result.get("point_ids", []),
                timeout_s=int(result.get("timeout_s", 60)),
                use_cache=bool(result.get("use_cache", True)),
                coverage_threshold=float(result.get("coverage_threshold", 90)),
                ryfast_include_ramp=bool(st.session_state.get("ryfast_include_ramp", True)),
                estimate_missing_points=bool(result.get("estimate_missing_points", False)),
            )
            st.download_button(
                "⬇️ Last ned PDF-rapport",
                data=report_bytes,
                file_name=f"ryfast_rapport_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
            )
