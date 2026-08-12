"""Streamlit app for fetching, comparing, and exporting Ryfast traffic data."""

import logging
from typing import List

import numpy as np
import pandas as pd
import streamlit as st

from ryfast_app.config import (
    COMPARE_MONTHS,
    COMPARE_WEEKS,
    COMPARE_YEARS,
    DEFAULT_YEARS,
    HUNDVAG_TUNNEL_IDS_UTEN_PÅRAMPE,
    HUNDVAG_TUNNEL_RAMP_IDS,
    MONTH_NAMES,
    TRAFFIC_POINTS,
    YEAR_RANGE,
)
from ryfast_app.processing import format_number
from ryfast_app.ui.banners import (
    render_anomaly_banner,
    render_api_status_sidebar,
    render_data_coverage_banner,
    render_point_basis_note,
    reserve_api_status_sidebar,
)
from ryfast_app.ui.bicycle_tab import render_bicycle_tab
from ryfast_app.ui.comparisons import create_comparison_dashboard
from ryfast_app.ui.export_section import create_export_section
from ryfast_app.ui.tabs import render_data_quality_tab, render_totals_tab
from ryfast_app.workflows import (
    process_data_for_months,
    process_data_for_weeks,
    process_data_for_years,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def init_session_state():
    if "comparison_history" not in st.session_state:
        st.session_state.comparison_history = []
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "api_errors" not in st.session_state:
        st.session_state.api_errors = []
    if "bicycle_result" not in st.session_state:
        st.session_state.bicycle_result = None


def main():
    init_session_state()

    st.set_page_config(page_title="Trafikkdata Visualisering - Ryfast", page_icon="🚗", layout="wide", initial_sidebar_state="expanded")
    st.markdown(
        """
    <style>
    .main-header { background: linear-gradient(90deg, #1f77b4, #ff7f0e); color: white; padding: 1rem;
      border-radius: 10px; text-align: center; margin-bottom: 2rem; }
    </style>
    """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
    <div class="main-header">
        <h1>🚗 Trafikkdata Visualisering - Ryfast</h1>
        <p>Analyse av trafikkmønstre for Ryfylketunnelen, Hundvågtunnelen og Bybrua</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.sidebar.header("⚙️ Innstillinger")
    point_options = list(TRAFFIC_POINTS.keys())
    point_descriptions = [f"{p} - {TRAFFIC_POINTS[p]['description']}" for p in point_options]

    with st.sidebar.form("controls", clear_on_submit=False):
        selected_index = st.selectbox(
            "Velg målepunkt",
            range(len(point_options)),
            format_func=lambda x: point_descriptions[x],
            key="point_selector",
        )
        point = point_options[selected_index]

        with st.expander("ℹ️ Om valgt målepunkt"):
            st.write(f"**Beskrivelse:** {TRAFFIC_POINTS[point]['description']}")
            st.write(f"**Åpnet:** {TRAFFIC_POINTS[point]['opened']}")

        comparison_mode = st.radio(
            "Velg analysetype",
            [COMPARE_YEARS, COMPARE_MONTHS, COMPARE_WEEKS],
            key="comparison_mode",
        )

        with st.expander("🔧 Avanserte innstillinger"):
            use_cache = st.checkbox("Bruk hurtigbuffer", value=True, key="use_cache")
            timeout_s = st.slider("API timeout (sekunder)", 10, 90, 60, key="timeout_s")
            coverage_threshold = st.slider("Min. dekning (%)", 50, 100, 90, key="coverage_threshold")
            estimate_missing_points = st.checkbox(
                "Estimer manglende målepunkter (pro-rata)",
                value=False,
                key="estimate_missing_points",
                help="Når noen (men ikke alle) målepunkter mangler data i en måned, skaleres totalen opp basert på antall punkter med data. Bruk med varsomhet.",
            )

        direction = "Begge retninger"
        has_optional_points = point in {"Ryfast (sum tunneler)", "Hundvågtunnelen"}
        include_ramp = st.checkbox(
            "Inkluder pårampe/tilleggspunkt (der definert)",
            value=bool(st.session_state.get("ryfast_include_ramp", True)),
            key="ryfast_include_ramp",
            disabled=not has_optional_points,
            help=(
                "Når av (der tilgjengelig), ekskluderes pårampe/tilleggspunkt for Hundvågtunnelen "
                f"({', '.join(HUNDVAG_TUNNEL_RAMP_IDS)})."
            ),
        )
        if point == "Bybrua":
            direction = st.selectbox("Velg retning", ["Begge retninger", "Mot nord", "Mot sør"], key="bybrua_direction")

        year_input = DEFAULT_YEARS
        year_list: List[int] = []
        year = 2025
        months: List[int] = []
        weeks: List[int] = []

        if comparison_mode == COMPARE_YEARS:
            year_input = st.text_input("År (komma-separert)", value=DEFAULT_YEARS, key="years_input")
        elif comparison_mode == COMPARE_MONTHS:
            year = st.selectbox("År", list(YEAR_RANGE), index=list(YEAR_RANGE).index(2025) if 2025 in YEAR_RANGE else 0, key="months_year")
            months = st.multiselect(
                "Velg måneder",
                list(range(1, 13)),
                default=list(range(1, 13)),
                format_func=lambda m: MONTH_NAMES[m - 1],
                key="months_selected",
            )
        else:
            year = st.selectbox("År", list(YEAR_RANGE), index=list(YEAR_RANGE).index(2025) if 2025 in YEAR_RANGE else 0, key="weeks_year")
            weeks = st.multiselect("Velg uker", list(range(1, 53)), default=list(range(1, 11)), key="weeks_selected")

        submitted = st.form_submit_button("📊 Analyser data", type="primary")

    api_status_slot = reserve_api_status_sidebar()

    if st.sidebar.button("🗑️ Tøm cache"):
        st.cache_data.clear()
        st.session_state.last_result = None
        st.session_state.bicycle_result = None
        st.sidebar.success("Cache tømt!")

    def resolve_point_ids() -> List[str]:
        if point == "Ryfast (sum tunneler)":
            ryfylke_ids = TRAFFIC_POINTS["Ryfylketunnelen"]["ids"]
            hundvag_ids = TRAFFIC_POINTS["Hundvågtunnelen"]["ids"] if include_ramp else HUNDVAG_TUNNEL_IDS_UTEN_PÅRAMPE
            return ryfylke_ids + hundvag_ids
        if point == "Ryfylketunnelen":
            return TRAFFIC_POINTS["Ryfylketunnelen"]["ids"]
        if point == "Hundvågtunnelen":
            return TRAFFIC_POINTS["Hundvågtunnelen"]["ids"] if include_ramp else HUNDVAG_TUNNEL_IDS_UTEN_PÅRAMPE
        if direction == "Begge retninger":
            return TRAFFIC_POINTS["Bybrua"]["ids"]["Mot nord"] + TRAFFIC_POINTS["Bybrua"]["ids"]["Mot sør"]
        return TRAFFIC_POINTS["Bybrua"]["ids"][direction]

    if submitted:
        # Valider før henting: uten gyldig utvalg skal vi ikke kalle API-et og
        # ende opp med et generisk «ingen data»-varsel.
        selection_ok = True
        if comparison_mode == COMPARE_YEARS:
            try:
                year_list = [int(y.strip()) for y in year_input.split(",") if y.strip()]
            except ValueError:
                st.sidebar.error("Ugyldig format. Bruk f.eks. 2024,2025")
                year_list = []
            if not year_list:
                st.sidebar.warning("Velg minst ett år")
                selection_ok = False
            else:
                year = year_list[-1]
        elif comparison_mode == COMPARE_MONTHS and not months:
            st.sidebar.warning("Velg minst én måned")
            selection_ok = False
        elif comparison_mode == COMPARE_WEEKS and not weeks:
            st.sidebar.warning("Velg minst én uke")
            selection_ok = False

        if not selection_ok:
            st.session_state.last_result = None

        point_ids = resolve_point_ids()

        if selection_ok:
            with st.spinner("🔄 Behandler data..."):
                if comparison_mode == COMPARE_YEARS:
                    result_tuple = process_data_for_years(point_ids, year_list, timeout_s, use_cache, estimate_missing_points)
                    df, coverage_summary = result_tuple
                    title = f"Årlig sammenligning for {point}"
                elif comparison_mode == COMPARE_MONTHS:
                    result_tuple = process_data_for_months(point_ids, year, months, timeout_s, use_cache, estimate_missing_points)
                    if result_tuple is None:
                        df, coverage_summary = None, pd.DataFrame()
                    else:
                        df, coverage_summary = result_tuple
                    title = f"Månedlig analyse for {point} i {year}"
                else:
                    result_tuple = process_data_for_weeks(point_ids, year, weeks, timeout_s, use_cache)
                    if result_tuple is None:
                        df, coverage_summary = None, pd.DataFrame()
                    else:
                        df, coverage_summary = result_tuple
                    title = f"Ukentlig analyse for {point} i {year}"

            if df is None or df.empty:
                st.error("❌ Ingen data tilgjengelig for valgte kriterier")
                st.session_state.last_result = None
            else:
                st.session_state.last_result = {
                    "df": df,
                    "coverage_summary": coverage_summary,
                    "title": title,
                    "point": point,
                    "comparison_mode": comparison_mode,
                    "year_list": year_list,
                    "year": year,
                    "point_ids": point_ids,
                    "timeout_s": timeout_s,
                    "use_cache": use_cache,
                    "coverage_threshold": coverage_threshold,
                    "estimate_missing_points": bool(estimate_missing_points),
                }

    try:
        _render_tabs()
    finally:
        # Til slutt, uansett utfall: fanene over rekker å registrere API-feil
        # etter at sidebaren er tegnet, og de skal med i samme rerun.
        render_api_status_sidebar(api_status_slot)


def _render_tabs() -> None:
    """Tegn de seks fanene. Alt de trenger ligger i `st.session_state`.

    Egen funksjon slik at `main()` kan flushe API-feil til sidebaren *etter*
    at fanene er kjørt, uten at en tidlig `return` her hopper over flushen.
    """
    tab_labels = [
        "📈 Visualisering", "📊 Data", "🧮 Totaltall", "🛡️ Datakvalitet", "📄 Rapport", "🚲 Sykkel"
    ]
    default_tab = "🚲 Sykkel" if st.session_state.pop("bicycle_tab_requested", False) else None
    tabs = st.tabs(tab_labels, default=default_tab)
    tab1, tab2, tab3, tab4, tab5, tab6 = tabs

    # Sykkelfanen har eget punkt- og årsvalg, så den ligger utenfor sjekken på
    # fullført bilanalyse: den skal virke også før «Analyser data» er trykket.
    with tab6:
        render_bicycle_tab()

    if not st.session_state.last_result:
        with tab1:
            st.info("Velg målepunkt og analysetype i sidebaren, og trykk «Analyser data».")
        return

    result = st.session_state.last_result
    df = result["df"]
    coverage_summary = result.get("coverage_summary", pd.DataFrame())
    title = result["title"]
    point = result["point"]
    comparison_mode = result["comparison_mode"]
    year_list = result["year_list"]
    year = result["year"]
    point_ids = result["point_ids"]
    timeout_s = result["timeout_s"]
    use_cache = result["use_cache"]
    coverage_threshold = result.get("coverage_threshold", 90)

    with tab1:
        st.subheader(title)
        render_data_coverage_banner(coverage_summary)
        render_anomaly_banner(df)
        create_comparison_dashboard(df, point)
        render_point_basis_note(coverage_summary)

    with tab2:
        st.subheader("📊 Rådata")
        formatted_df = df.copy()
        numeric_cols = formatted_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            formatted_df[col] = formatted_df[col].map(format_number)
        st.dataframe(formatted_df, width="stretch", hide_index=True)
        if coverage_summary is not None and not coverage_summary.empty:
            with st.expander("🛡️ Datadekning (for perioden som vises)"):
                cov_view = coverage_summary.copy()
                for c in ["points_present_pct", "mean_coverage_pct", "min_coverage_pct"]:
                    if c in cov_view.columns:
                        cov_view[c] = pd.to_numeric(cov_view[c], errors="coerce").round(1)
                st.dataframe(cov_view, width="stretch", hide_index=True)

    with tab3:
        render_totals_tab(df, point, comparison_mode, year_list, year, point_ids, timeout_s, use_cache)

    with tab4:
        render_data_quality_tab(
            point=point,
            comparison_mode=comparison_mode,
            year_list=year_list,
            year=year,
            point_ids=point_ids,
            timeout_s=timeout_s,
            use_cache=use_cache,
            coverage_threshold=float(coverage_threshold),
            df=df,
            coverage_summary=coverage_summary,
        )

    with tab5:
        st.subheader("📄 Rapport / eksport")
        create_export_section(df, point, coverage_summary=coverage_summary)


if __name__ == "__main__":
    main()
