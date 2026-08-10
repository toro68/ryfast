"""Fanene Totaltall og Datakvalitet."""

import logging
from typing import List, Optional

import altair as alt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from ryfast_app.api import fetch_batch_traffic_data
from ryfast_app.config import (
    ANOMALY_THRESHOLD_PCT,
    COMPARE_WEEKS,
    COMPARE_YEARS,
    HUNDVAG_TUNNEL_IDS_UTEN_PÅRAMPE,
    MONTH_NAMES,
    TRAFFIC_POINTS,
)
from ryfast_app.metrics import (
    aggregate_monthly_totals_by_group,
    compute_monthly_totals_table,
    coverage_pivot,
    group_coverage_by_month,
    totals_with_uncertainty_from_metrics,
)
from ryfast_app.processing import (
    calculate_yearly_total_from_monthly_averages,
    days_in_year,
    detect_monthly_anomalies,
    extract_point_monthly_metrics,
    format_number,
    overlapping_months,
)

logger = logging.getLogger(__name__)


def render_totals_tab(df: pd.DataFrame, point: str, comparison_mode: str, year_list: List[int], year: int, point_ids: List[str], timeout_s: int, use_cache: bool):
    st.subheader("🧮 Totale passeringer (samlet trafikk)")
    if comparison_mode == COMPARE_WEEKS:
        st.info(
            "Ukesvisning viser per nå gjennomsnitt per døgn (sum av punkter). "
            "Totale uketall krever at vi vet hvor mange gyldige dager som inngår i snittet."
        )
        return

    years_for_totals = year_list if comparison_mode == COMPARE_YEARS else [year]
    totals_df = compute_monthly_totals_table(df, years_for_totals)
    st.caption("Beregning: (månedsvis gjennomsnitt per døgn) × (antall dager i måneden), summert per måned.")

    year_cols = [y for y in years_for_totals if str(y) in totals_df.columns]
    if year_cols:
        latest_year = max(year_cols)
        total, months_present, days_covered = calculate_yearly_total_from_monthly_averages(df, int(latest_year))
        avg_per_day = (total / days_covered) if days_covered else None
        full_year_estimate = (avg_per_day * days_in_year(int(latest_year))) if avg_per_day is not None else None

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric(
                f"Totalt {latest_year} ({'hittil' if months_present < 12 else 'helår'})",
                format_number(total),
                help="Totalt antall passeringer beregnet fra Vegvesen sine måneds-ÅDT.",
            )
        with m2:
            st.metric("Snitt per døgn (basert på tilgjengelige måneder)", format_number(avg_per_day) if avg_per_day is not None else "N/A")
        with m3:
            if months_present < 12 and full_year_estimate is not None:
                st.metric("Estimert helår", format_number(full_year_estimate))
            else:
                st.metric("Måneder med data", f"{months_present}/12")

        if comparison_mode == COMPARE_YEARS and len(year_cols) >= 2:
            prev_year = sorted(year_cols)[-2]
            # Like perioder: bare måneder der begge år har tall, ellers måles
            # et delår mot et helår.
            shared = overlapping_months(df, [str(prev_year), str(latest_year)])
            df_shared = df[pd.to_numeric(df["Month"], errors="coerce").isin(shared)] if "Month" in df.columns else df
            prev_total, _, _ = calculate_yearly_total_from_monthly_averages(df_shared, int(prev_year))
            latest_shared_total, shared_months, _ = calculate_yearly_total_from_monthly_averages(df_shared, int(latest_year))
            if prev_total:
                st.metric(
                    f"Endring vs {prev_year}",
                    f"{((latest_shared_total - prev_total) / prev_total * 100):+.1f}%",
                    help=(
                        f"Sammenligner de {shared_months} månedene der både {prev_year} og "
                        f"{latest_year} har data."
                    ),
                )

    melt_cols = [str(y) for y in years_for_totals if str(y) in totals_df.columns]
    if melt_cols:
        melted = totals_df.melt(id_vars=["Month", "Month Name"], value_vars=melt_cols, var_name="År", value_name="Passeringer")
        fig_totals = px.bar(
            melted, x="Month Name", y="Passeringer", color="År", barmode="group",
            title="Totale passeringer per måned",
            labels={"Passeringer": "Passeringer", "Month Name": "Måned"},
        )
        fig_totals.update_traces(hovertemplate="%{x}: <b>%{y:,.0f}</b><extra>%{fullData.name}</extra>")
        fig_totals.update_layout(
            yaxis=dict(tickformat=","),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(t=60, b=40),
        )
        st.plotly_chart(fig_totals, width="stretch")

    if point == "Ryfast (sum tunneler)":
        with st.expander("🔎 Fordeling mellom tunneler (Ryfylketunnelen vs Hundvågtunnelen)"):
            breakdown_year = st.selectbox("Velg år for fordeling", options=years_for_totals, index=len(years_for_totals) - 1)
            traffic_by_point = fetch_batch_traffic_data(point_ids, int(breakdown_year), timeout_s, use_cache)
            point_ids_by_group = {
                "Ryfylketunnelen": TRAFFIC_POINTS["Ryfylketunnelen"]["ids"],
                "Hundvågtunnelen": (
                    TRAFFIC_POINTS["Hundvågtunnelen"]["ids"]
                    if st.session_state.get("ryfast_include_ramp", True)
                    else HUNDVAG_TUNNEL_IDS_UTEN_PÅRAMPE
                ),
            }
            totals_by_group_df, coverage_by_group_df = aggregate_monthly_totals_by_group(traffic_by_point, point_ids_by_group, int(breakdown_year))
            melted_groups = totals_by_group_df.melt(id_vars=["Month", "Month Name"], value_vars=list(point_ids_by_group.keys()), var_name="Tunnel", value_name="Passeringer")
            fig_stack = px.bar(melted_groups, x="Month Name", y="Passeringer", color="Tunnel", title=f"Samlet trafikk {breakdown_year} (stacked per tunnel)")
            fig_stack.update_yaxes(tickformat=",")
            st.plotly_chart(fig_stack, width="stretch")
            st.caption("Dekning (%) er snitt av rapportert dekning på målepunktene per måned.")
            st.dataframe(coverage_by_group_df.round(1), width="stretch", hide_index=True)

    with st.expander("📋 Totaltabell"):
        formatted_totals = totals_df.copy()
        numeric_cols = formatted_totals.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            formatted_totals[col] = formatted_totals[col].map(format_number)
        st.dataframe(formatted_totals, width="stretch", hide_index=True)


def render_data_quality_tab(
    point: str,
    comparison_mode: str,
    year_list: List[int],
    year: int,
    point_ids: List[str],
    timeout_s: int,
    use_cache: bool,
    coverage_threshold: float,
    df: Optional[pd.DataFrame] = None,
    coverage_summary: Optional[pd.DataFrame] = None,
):
    st.subheader("🛡️ Dekning og datakvalitet")
    st.caption(
        "Dekning (%) er rapportert datadekning fra Vegvesen. "
        "Lav dekning kan gi skjevheter. Usikkerhet er indikativ og basert på oppgitte konfidensintervaller."
    )

    if comparison_mode == COMPARE_WEEKS:
        cov = coverage_summary if coverage_summary is not None else pd.DataFrame()
        if cov is None or cov.empty:
            st.warning("Fant ingen dekningsdata for ukesvisning.")
            return

        below_cov = cov[(cov["mean_coverage_pct"].notna()) & (cov["mean_coverage_pct"] < float(coverage_threshold))]
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Dekningsterskel", f"{coverage_threshold:.0f}%")
        with m2:
            st.metric("Uker under terskel", str(len(below_cov)))
        with m3:
            st.metric(
                "Snitt dekning",
                f"{cov['mean_coverage_pct'].mean():.1f}%" if cov["mean_coverage_pct"].notna().any() else "N/A",
            )

        st.markdown("### Ukentlig dekning")
        try:
            plot_df = cov.dropna(subset=["mean_coverage_pct"]).copy()
            if not plot_df.empty:
                week_order = sorted(
                    plot_df["week"].astype(str).unique(),
                    key=lambda s: int("".join([c for c in s if c.isdigit()]) or "0"),
                )
                chart = (
                    alt.Chart(plot_df)
                    .mark_line(point=True, color="#1f77b4")
                    .encode(
                        x=alt.X("week:N", sort=week_order, title="Uke"),
                        y=alt.Y("mean_coverage_pct:Q", title="Snitt dekning (%)", scale=alt.Scale(domain=[0, 100])),
                        tooltip=[
                            alt.Tooltip("week:N", title="Uke"),
                            alt.Tooltip("points_present:Q", title="Punkter m/data"),
                            alt.Tooltip("points_expected:Q", title="Punkter forventet"),
                            alt.Tooltip("mean_coverage_pct:Q", format=".1f", title="Snitt dekning (%)"),
                            alt.Tooltip("min_coverage_pct:Q", format=".1f", title="Min dekning (%)"),
                        ],
                    )
                    .properties(height=320)
                )
                st.altair_chart(chart, width="stretch")
        except Exception as exc:
            logger.warning("Datakvalitet (uker): klarte ikke vise dekningsgraf: %s", exc)

        st.dataframe(cov.copy(), width="stretch", hide_index=True)
        return

    years = year_list if comparison_mode == COMPARE_YEARS else [year]
    selected_year = st.selectbox("År", options=years, index=len(years) - 1, key="dq_year")

    traffic_by_point = fetch_batch_traffic_data(point_ids, int(selected_year), timeout_s, use_cache)
    metrics = extract_point_monthly_metrics(traffic_by_point, int(selected_year))
    if metrics.empty:
        st.warning("Fant ingen dekning/data for valgt år.")
        return

    below = metrics[(metrics["coverage_pct"].notna()) & (metrics["coverage_pct"] < coverage_threshold)]
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Dekningsterskel", f"{coverage_threshold:.0f}%")
    with m2:
        st.metric("Observasjoner under terskel", str(len(below)))
    with m3:
        st.metric("Snitt dekning", f"{metrics['coverage_pct'].mean():.1f}%" if metrics["coverage_pct"].notna().any() else "N/A")

    if not below.empty:
        bad_months = (
            below.groupby("month_name")["point_label"]
            .apply(lambda s: ", ".join(sorted(set(s))[:4]) + (" …" if len(set(s)) > 4 else ""))
            .reindex(MONTH_NAMES)
            .dropna()
        )
        st.warning("Måneder med lav dekning: " + ", ".join([m for m in bad_months.index.tolist() if m]))

    # Coverage per point heatmap/table
    st.markdown("### Dekning per målepunkt")
    cov_piv = coverage_pivot(metrics)
    if not cov_piv.empty:
        cov_long = cov_piv.reset_index().melt(id_vars=["month_name"], var_name="Målepunkt", value_name="Dekning")
        cov_long = cov_long.dropna(subset=["Dekning"])
        chart = (
            alt.Chart(cov_long)
            .mark_rect()
            .encode(
                x=alt.X("month_name:N", sort=MONTH_NAMES, title="Måned"),
                y=alt.Y("Målepunkt:N", title="Målepunkt"),
                color=alt.Color("Dekning:Q", scale=alt.Scale(domain=[0, 100], scheme="yellowgreenblue"), title="Dekning (%)"),
                tooltip=[
                    alt.Tooltip("Målepunkt:N"),
                    alt.Tooltip("month_name:N", title="Måned"),
                    alt.Tooltip("Dekning:Q", format=".1f", title="Dekning (%)"),
                ],
            )
            .properties(height=min(420, 24 * max(1, len(cov_piv.columns))), title="Dekning per måned og målepunkt")
        )
        st.altair_chart(chart, width="stretch")

        with st.expander("📋 Dekningstabell"):
            st.dataframe(cov_piv.round(1), width="stretch")

    # Group coverage (Ryfast breakdown)
    if point == "Ryfast (sum tunneler)":
        st.markdown("### Dekning per tunnel")
        point_ids_by_group = {
            "Ryfylketunnelen": TRAFFIC_POINTS["Ryfylketunnelen"]["ids"],
            "Hundvågtunnelen": (
                TRAFFIC_POINTS["Hundvågtunnelen"]["ids"]
                if st.session_state.get("ryfast_include_ramp", True)
                else HUNDVAG_TUNNEL_IDS_UTEN_PÅRAMPE
            ),
        }
        grouped = group_coverage_by_month(metrics, point_ids_by_group)
        if not grouped.empty:
            chart = (
                alt.Chart(grouped)
                .mark_line(point=True)
                .encode(
                    x=alt.X("month_name:N", sort=MONTH_NAMES, title="Måned"),
                    y=alt.Y("coverage_pct:Q", title="Dekning (%)", scale=alt.Scale(domain=[0, 100])),
                    color=alt.Color("group:N", title="Tunnel"),
                    tooltip=[
                        alt.Tooltip("group:N", title="Tunnel"),
                        alt.Tooltip("month_name:N", title="Måned"),
                        alt.Tooltip("coverage_pct:Q", format=".1f", title="Dekning (%)"),
                    ],
                )
                .properties(height=320)
            )
            st.altair_chart(chart, width="stretch")

    st.markdown("### Usikkerhet (indikativ)")
    st.caption(
        "Konfidensintervaller er oppgitt per målepunkt. Vi summerer bounds på tvers av punkt for å gi et "
        "indikativt spenn (ikke et strengt statistisk intervall)."
    )
    totals_ci = totals_with_uncertainty_from_metrics(metrics)
    if not totals_ci.empty:
        totals_ci["total"] = totals_ci["total"].round(0).astype("Int64")
        totals_ci["total_lower"] = totals_ci["total_lower"].round(0).astype("Int64")
        totals_ci["total_upper"] = totals_ci["total_upper"].round(0).astype("Int64")
        display = totals_ci[["month_name", "total", "total_lower", "total_upper", "coverage_pct"]].copy()
        display["coverage_pct"] = display["coverage_pct"].round(1)
        display.columns = ["Måned", "Totalt", "Nedre", "Øvre", "Dekning (%)"]
        st.dataframe(display, width="stretch", hide_index=True)

    if df is not None and not df.empty:
        st.markdown("### Anomali-varsler (indikativ)")
        st.caption(
            f"Flagger måneder der nivået avviker mer enn ±{ANOMALY_THRESHOLD_PCT:.0f}% fra forventet basert på de andre valgte årene."
        )
        anomalies = detect_monthly_anomalies(df, threshold_pct=float(ANOMALY_THRESHOLD_PCT))
        if anomalies.empty:
            st.caption("Ingen anomalier flagget.")
        else:
            st.warning(f"Flagget {len(anomalies)} anomalier.")
            st.dataframe(
                anomalies.assign(deviation_pct=anomalies["deviation_pct"].round(1)),
                width="stretch",
                hide_index=True,
            )
