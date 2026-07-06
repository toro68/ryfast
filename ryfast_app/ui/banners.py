"""Bannere og sidebar-status for datadekning, anomalier og API-feil."""

import logging
from typing import List

import altair as alt
import pandas as pd
import streamlit as st

from ryfast_app.api import clear_api_errors, flush_api_error_buffer_to_session_state
from ryfast_app.config import ANOMALY_THRESHOLD_PCT, FULL_COVERAGE_TOL_PCT, MONTH_NAMES
from ryfast_app.processing import detect_monthly_anomalies

logger = logging.getLogger(__name__)


def render_api_status_sidebar() -> None:
    flush_api_error_buffer_to_session_state()
    errors = st.session_state.get("api_errors", []) or []
    with st.sidebar.expander("🧾 API-feil / status", expanded=False):
        if not errors:
            st.caption("Ingen registrerte API-feil i denne sesjonen.")
            return
        st.warning(f"{len(errors)} API-feil registrert i denne sesjonen.")
        if st.button("Tøm API-feil", type="secondary"):
            clear_api_errors()
            st.rerun()
        st.dataframe(pd.DataFrame(errors).iloc[::-1], use_container_width=True, hide_index=True)


def render_data_coverage_banner(coverage_df: pd.DataFrame) -> None:
    if coverage_df is None or coverage_df.empty:
        return

    tmp = coverage_df.copy()
    tmp["has_issue"] = False
    if "points_expected" in tmp.columns and "points_present" in tmp.columns:
        tmp["has_issue"] |= tmp["points_present"] < tmp["points_expected"]
    if "mean_coverage_pct" in tmp.columns:
        tmp["has_issue"] |= tmp["mean_coverage_pct"].notna() & (
            tmp["mean_coverage_pct"] < (100.0 - FULL_COVERAGE_TOL_PCT)
        )
    if "min_coverage_pct" in tmp.columns:
        tmp["has_issue"] |= tmp["min_coverage_pct"].notna() & (tmp["min_coverage_pct"] < (100.0 - FULL_COVERAGE_TOL_PCT))

    issues = tmp[tmp["has_issue"]].copy()

    expected = int(tmp["points_expected"].dropna().iloc[0]) if "points_expected" in tmp.columns and tmp["points_expected"].notna().any() else 0
    present_min = int(tmp["points_present"].min()) if "points_present" in tmp.columns and tmp["points_present"].notna().any() else 0
    present_avg = float(tmp["points_present"].mean()) if "points_present" in tmp.columns and tmp["points_present"].notna().any() else float("nan")
    missing_periods = int((tmp["points_present"] < tmp["points_expected"]).sum()) if {"points_present", "points_expected"} <= set(tmp.columns) else 0
    periods_total = int(len(tmp))

    mean_cov = float(tmp["mean_coverage_pct"].mean()) if "mean_coverage_pct" in tmp.columns and tmp["mean_coverage_pct"].notna().any() else float("nan")
    min_cov = float(tmp["min_coverage_pct"].min()) if "min_coverage_pct" in tmp.columns and tmp["min_coverage_pct"].notna().any() else float("nan")

    point_cov_text = (
        f"**Punktdekning:** min {present_min}/{expected} punkter rapporterer"
        + (f" (snitt {present_avg:.1f}/{expected})" if expected and not pd.isna(present_avg) else "")
        + (f" — mangler i {missing_periods}/{periods_total} perioder" if periods_total else "")
    )
    data_cov_text = "**Datadekning:** "
    if not pd.isna(mean_cov):
        data_cov_text += f"snitt {mean_cov:.1f}%"
        if not pd.isna(min_cov):
            data_cov_text += f", min {min_cov:.1f}%"
        data_cov_text += " (på rapporterende punkter)"
    else:
        data_cov_text += "N/A"

    if issues.empty:
        st.success("✅ 100% i perioden som vises\n\n" + point_cov_text + "\n\n" + data_cov_text)
        return

    def _labels(sub: pd.DataFrame) -> List[str]:
        if "month_name" in sub.columns:
            months = [m for m in MONTH_NAMES if m in set(sub["month_name"].astype(str))]
            return months
        if "week" in sub.columns:
            return sub["week"].astype(str).tolist()
        return []

    if "year" in issues.columns and issues["year"].nunique() > 1:
        parts = [f"{int(y)}: " + ", ".join(_labels(sub)) for y, sub in issues.groupby("year")]
        where = " | ".join(parts)
    else:
        where = ", ".join(_labels(issues))

    st.warning("⚠️ Ikke 100% i perioden som vises\n\n" + point_cov_text + "\n\n" + data_cov_text + "\n\n" + f"**Perioder:** {where}")

    with st.expander("🔎 Detaljer (dekning)", expanded=False):
        try:
            if "month_name" in coverage_df.columns and "mean_coverage_pct" in coverage_df.columns:
                plot_df = coverage_df.dropna(subset=["mean_coverage_pct"]).copy()
                if not plot_df.empty:
                    plot_df["month_name"] = pd.Categorical(plot_df["month_name"], categories=MONTH_NAMES, ordered=True)
                    color = alt.Color("year:N", title="År") if "year" in plot_df.columns and plot_df["year"].nunique() > 1 else alt.value("#1f77b4")
                    tooltips = [
                        alt.Tooltip("month_name:N", title="Måned"),
                        alt.Tooltip("mean_coverage_pct:Q", format=".1f", title="Snitt dekning (%)"),
                    ]
                    if "year" in plot_df.columns:
                        tooltips.insert(0, alt.Tooltip("year:N", title="År"))
                    chart = (
                        alt.Chart(plot_df)
                        .mark_line(point=alt.OverlayMarkDef(size=50))
                        .encode(
                            x=alt.X("month_name:N", sort=MONTH_NAMES, title="Måned"),
                            y=alt.Y("mean_coverage_pct:Q", title="Snitt dekning (%)", scale=alt.Scale(domain=[0, 100])),
                            color=color,
                            tooltip=tooltips,
                        )
                        .properties(height=220)
                    )
                    st.altair_chart(chart, use_container_width=True)
            elif "week" in coverage_df.columns and "mean_coverage_pct" in coverage_df.columns:
                plot_df = coverage_df.dropna(subset=["mean_coverage_pct"]).copy()
                if not plot_df.empty:
                    week_order = sorted(plot_df["week"].astype(str).unique(), key=lambda s: int("".join([c for c in s if c.isdigit()]) or "0"))
                    chart = (
                        alt.Chart(plot_df)
                        .mark_line(point=alt.OverlayMarkDef(size=50), color="#1f77b4")
                        .encode(
                            x=alt.X("week:N", sort=week_order, title="Uke"),
                            y=alt.Y("mean_coverage_pct:Q", title="Snitt dekning (%)", scale=alt.Scale(domain=[0, 100])),
                            tooltip=[
                                alt.Tooltip("week:N", title="Uke"),
                                alt.Tooltip("mean_coverage_pct:Q", format=".1f", title="Snitt dekning (%)"),
                            ],
                        )
                        .properties(height=220)
                    )
                    st.altair_chart(chart, use_container_width=True)
        except Exception as exc:
            logger.warning("Dekning: klarte ikke vise detalj-graf: %s", exc)

        cols: List[str] = []
        if "year" in coverage_df.columns:
            cols.append("year")
        if "month_name" in coverage_df.columns:
            cols.append("month_name")
        if "week" in coverage_df.columns:
            cols.append("week")
        cols += [c for c in ["points_present", "points_expected", "mean_coverage_pct", "min_coverage_pct"] if c in coverage_df.columns]
        view = coverage_df[cols].copy()
        for c in ["mean_coverage_pct", "min_coverage_pct", "points_present_pct"]:
            if c in view.columns:
                view[c] = pd.to_numeric(view[c], errors="coerce").round(1)
        st.dataframe(view, use_container_width=True, hide_index=True)


def render_anomaly_banner(df: pd.DataFrame) -> None:
    anomalies = detect_monthly_anomalies(df, threshold_pct=float(ANOMALY_THRESHOLD_PCT))
    if anomalies is None or anomalies.empty:
        return
    preview = anomalies.sort_values(["year", "month"]).head(6)
    items = [f"{int(r.year)} {str(r.month_name)} ({float(r.deviation_pct):+.0f}%)" for r in preview.itertuples(index=False)]
    st.warning("⚠️ Mulige anomalier: " + ", ".join(items) + (" …" if len(anomalies) > len(preview) else ""))


def render_point_basis_note(coverage_df: pd.DataFrame) -> None:
    if coverage_df is None or coverage_df.empty:
        return
    if not {"points_present", "points_expected"} <= set(coverage_df.columns):
        return
    expected = int(coverage_df["points_expected"].dropna().iloc[0]) if coverage_df["points_expected"].notna().any() else 0
    if not expected:
        return
    present_min = int(coverage_df["points_present"].min()) if coverage_df["points_present"].notna().any() else 0
    if present_min < expected:
        st.caption(f"⚠️ Grafene kan være basert på ufullstendige målepunkter (min {present_min}/{expected} rapporterer i perioden).")
