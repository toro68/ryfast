"""PDF-eksport av trafikkdata-rapporter (fpdf2 + plotly/kaleido)."""

import logging
import os
import tempfile
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from fpdf import FPDF
from fpdf.enums import XPos, YPos

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
    compute_monthly_coverage_summary,
    totals_with_uncertainty_from_metrics,
)
from ryfast_app.processing import (
    calculate_yearly_total_from_monthly_averages,
    days_in_year,
    detect_monthly_anomalies,
    extract_point_monthly_metrics,
    format_number,
)

logger = logging.getLogger(__name__)

_PDF_TEXT_REPLACEMENTS = {
    "–": "-",  # en dash
    "—": "-",  # em dash
    "−": "-",  # minus sign
    " ": " ",  # nbsp
    "‘": "'",  # left single quote
    "’": "'",  # right single quote
    "“": '"',  # left double quote
    "”": '"',  # right double quote
    "…": "...",  # ellipsis
}


def pdf_safe_text(text: str) -> str:
    """Erstatt typografiske tegn og fall tilbake til latin-1 (kjernefontene)."""
    for src, dst in _PDF_TEXT_REPLACEMENTS.items():
        text = text.replace(src, dst)
    return text.encode("latin-1", errors="replace").decode("latin-1")


def _pdf_line(pdf: "FPDF", text: str, height: float = 6) -> None:
    """Skriv én linje og flytt til neste (erstatter deprecated ln=True)."""
    pdf.cell(0, height, pdf_safe_text(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)


def _pdf_embed_figure(pdf: "FPDF", fig: "go.Figure") -> None:
    """Render a Plotly figure as PNG and embed it in the PDF, cleaning up the temp file."""
    img_path: Optional[str] = None
    try:
        img_bytes = fig.to_image(format="png", scale=2)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(img_bytes)
            img_path = tmp.name
        pdf.ln(2)
        pdf.image(img_path, w=190)
    except Exception as exc:
        logger.warning("PDF: klarte ikke legge til figur: %s", exc)
    finally:
        if img_path:
            try:
                os.remove(img_path)
            except OSError:
                pass

def build_pdf_report(
    df: pd.DataFrame,
    point: str,
    comparison_mode: str,
    year_list: List[int],
    year: int,
    point_ids: List[str],
    timeout_s: int,
    use_cache: bool,
    coverage_threshold: float,
    ryfast_include_ramp: Optional[bool] = None,
    estimate_missing_points: bool = False,
) -> bytes:
    include_ramp = (
        bool(st.session_state.get("ryfast_include_ramp", True)) if ryfast_include_ramp is None else bool(ryfast_include_ramp)
    )

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    _pdf_line(pdf, "Ryfast - rapport", height=10)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(
        0,
        6,
        pdf_safe_text(
            f"Generert: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Målepunkt: {point}\n"
            f"Analysetype: {comparison_mode}\n"
            f"År: {','.join(map(str, year_list)) if comparison_mode == COMPARE_YEARS else str(year)}\n"
            f"Min. dekning: {coverage_threshold:.0f}%\n"
            f"Estimer manglende punkt: {'Ja' if estimate_missing_points else 'Nei'}\n"
            f"Punkt-IDer: {', '.join(point_ids)}"
        ),
    )

    # Year summary (derived from df; no extra API calls)
    year_cols_in_df = sorted([int(c) for c in df.columns if str(c).isdigit()])
    if comparison_mode != COMPARE_WEEKS and year_cols_in_df:
        pdf.ln(2)
        pdf.set_font("Helvetica", "B", 12)
        _pdf_line(pdf, "Årsoppsummering (beregnet fra månedsdata)", height=8)
        pdf.set_font("Helvetica", "", 10)
        estimates: List[Tuple[int, float]] = []
        prev_estimate: Optional[float] = None
        for y in year_cols_in_df:
            total, months_present, days_covered = calculate_yearly_total_from_monthly_averages(df, int(y))
            avg_per_day = (total / days_covered) if days_covered else None
            estimate = None
            if avg_per_day is not None and not pd.isna(avg_per_day):
                estimate = float(avg_per_day) * days_in_year(int(y))
                estimates.append((int(y), estimate))
            yoy_txt = ""
            if prev_estimate and estimate and prev_estimate != 0:
                yoy = (estimate - prev_estimate) / prev_estimate * 100
                yoy_txt = f"  YoY {yoy:+.1f}%"
            prev_estimate = estimate if estimate is not None else prev_estimate
            _pdf_line(
                pdf,
                f"{int(y)}: sum {int(round(total)):,} (mnd={int(months_present)}, dager={int(days_covered)})"
                + (f"  helår {int(round(estimate)):,}" if estimate is not None else "")
                + yoy_txt,
            )

        if len(estimates) >= 1:
            try:
                fig = go.Figure()
                fig.add_trace(go.Bar(x=[str(y) for y, _ in estimates], y=[v for _, v in estimates], name="Helårsestimat"))
                fig.update_layout(
                    title="Helårsestimat (beregnet)",
                    yaxis_title="Passeringer",
                    xaxis_title="År",
                    template="plotly_white",
                    height=320,
                    margin=dict(l=10, r=10, t=40, b=10),
                )
                _pdf_embed_figure(pdf, fig)
            except Exception as exc:
                logger.warning("PDF: helårsestimat-graf feilet: %s", exc)

    if comparison_mode != COMPARE_WEEKS:
        selected_year = max(year_list) if comparison_mode == COMPARE_YEARS else year
        traffic_by_point = fetch_batch_traffic_data(point_ids, int(selected_year), timeout_s, use_cache)
        metrics = extract_point_monthly_metrics(traffic_by_point, int(selected_year))
        totals_ci = totals_with_uncertainty_from_metrics(metrics)
        if not totals_ci.empty:
            try:
                totals_ci_sorted = totals_ci.sort_values("month")
                fig = go.Figure()
                fig.add_trace(go.Bar(x=totals_ci_sorted["month_name"], y=totals_ci_sorted["total"], name="Totalt"))
                fig.add_trace(
                    go.Scatter(
                        x=totals_ci_sorted["month_name"],
                        y=totals_ci_sorted["total_lower"],
                        mode="lines",
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=totals_ci_sorted["month_name"],
                        y=totals_ci_sorted["total_upper"],
                        mode="lines",
                        line=dict(width=0),
                        fill="tonexty",
                        fillcolor="rgba(31,119,180,0.18)",
                        name="Usikkerhet (indikativ)",
                    )
                )
                fig.update_layout(
                    title=f"Totale passeringer per måned ({selected_year})",
                    yaxis_title="Passeringer",
                    xaxis_title="Måned",
                    template="plotly_white",
                    height=360,
                    margin=dict(l=10, r=10, t=40, b=10),
                )
                _pdf_embed_figure(pdf, fig)
            except Exception as exc:
                logger.warning("PDF: totale passeringer-graf feilet: %s", exc)
        if not totals_ci.empty:
            pdf.ln(2)
            pdf.set_font("Helvetica", "B", 12)
            _pdf_line(pdf, f"Totale passeringer (indikativ usikkerhet) - {selected_year}", height=8)
            pdf.set_font("Helvetica", "", 10)
            for _, r in totals_ci.iterrows():
                if pd.isna(r["total"]):
                    continue
                coverage_text = f"dekning {float(r['coverage_pct']):.1f}%" if pd.notna(r["coverage_pct"]) else "dekning N/A"
                lower_text = int(round(r["total_lower"])) if pd.notna(r["total_lower"]) else None
                upper_text = int(round(r["total_upper"])) if pd.notna(r["total_upper"]) else None
                _pdf_line(
                    pdf,
                    f"{r['month_name']}: {int(round(r['total'])):,}  "
                    f"[{format_number(lower_text) if lower_text is not None else 'N/A'} – {format_number(upper_text) if upper_text is not None else 'N/A'}]  "
                    f"{coverage_text}",
                )

        # Coverage summary (uses already-fetched metrics)
        if metrics is not None and not metrics.empty and metrics["coverage_pct"].notna().any():
            pdf.ln(2)
            pdf.set_font("Helvetica", "B", 12)
            _pdf_line(pdf, f"Dekning og datakvalitet - {selected_year}", height=8)
            pdf.set_font("Helvetica", "", 10)
            below = metrics[(metrics["coverage_pct"].notna()) & (metrics["coverage_pct"] < float(coverage_threshold))].copy()
            _pdf_line(pdf, f"Snitt dekning: {metrics['coverage_pct'].mean():.1f}%")
            _pdf_line(pdf, f"Observasjoner under terskel ({coverage_threshold:.0f}%): {len(below)}")

            if not below.empty:
                worst = below.sort_values("coverage_pct", ascending=True).head(8)
                pdf.ln(1)
                _pdf_line(pdf, "Lavest dekning (utvalg):")
                for r in worst.itertuples(index=False):
                    _pdf_line(pdf, f"- {r.month_name}: {r.point_label} {float(r.coverage_pct):.1f}%")

            if not totals_ci.empty and totals_ci["coverage_pct"].notna().any():
                try:
                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(
                            x=totals_ci["month_name"],
                            y=totals_ci["coverage_pct"],
                            mode="lines+markers",
                            name="Dekning (%)",
                        )
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=totals_ci["month_name"],
                            y=[float(coverage_threshold)] * len(totals_ci),
                            mode="lines",
                            name="Terskel",
                            line=dict(dash="dash"),
                        )
                    )
                    fig.update_layout(
                        title=f"Dekning per måned ({selected_year})",
                        yaxis_title="Dekning (%)",
                        xaxis_title="Måned",
                        template="plotly_white",
                        height=320,
                        margin=dict(l=10, r=10, t=40, b=10),
                    )
                    _pdf_embed_figure(pdf, fig)
                except Exception as exc:
                    logger.warning("PDF: dekning-graf feilet: %s", exc)

        # Ryfast tunnel breakdown (optional)
        if point == "Ryfast (sum tunneler)":
            try:
                point_ids_by_group = {
                    "Ryfylketunnelen": TRAFFIC_POINTS["Ryfylketunnelen"]["ids"],
                    "Hundvågtunnelen": (
                        TRAFFIC_POINTS["Hundvågtunnelen"]["ids"] if include_ramp else HUNDVAG_TUNNEL_IDS_UTEN_PÅRAMPE
                    ),
                }
                totals_by_group_df, cov_by_group_df = aggregate_monthly_totals_by_group(
                    traffic_by_point, point_ids_by_group, int(selected_year)
                )
                if totals_by_group_df is not None and not totals_by_group_df.empty:
                    try:
                        fig = px.bar(
                            totals_by_group_df.melt(
                                id_vars=["Month", "Month Name"],
                                value_vars=list(point_ids_by_group.keys()),
                                var_name="Tunnel",
                                value_name="Passeringer",
                            ),
                            x="Month Name",
                            y="Passeringer",
                            color="Tunnel",
                            barmode="stack",
                            title=f"Samlet trafikk per måned ({selected_year}) - tunnel-fordeling",
                        )
                        fig.update_yaxes(tickformat=",")
                        fig.update_layout(template="plotly_white", height=340, margin=dict(l=10, r=10, t=40, b=10))
                        _pdf_embed_figure(pdf, fig)
                    except Exception as exc:
                        logger.warning("PDF: tunnelfordeling-graf feilet: %s", exc)
                if cov_by_group_df is not None and not cov_by_group_df.empty:
                    pdf.ln(2)
                    pdf.set_font("Helvetica", "B", 12)
                    _pdf_line(pdf, f"Dekning per tunnel (snitt) - {selected_year}", height=8)
                    pdf.set_font("Helvetica", "", 10)
                    # iterrows() framfor itertuples(): "Month Name" er ikke en gyldig
                    # Python-identifikator og blir omdøpt til "_1" i namedtuples.
                    for _, row in cov_by_group_df.iterrows():
                        try:
                            month_name = row.get("Month Name", "")
                            ryf = row.get("Ryfylketunnelen", np.nan)
                            hund = row.get("Hundvågtunnelen", np.nan)
                            if pd.isna(ryf) and pd.isna(hund):
                                continue
                            ryf_txt = f"{float(ryf):.1f}%" if pd.notna(ryf) else "N/A"
                            hund_txt = f"{float(hund):.1f}%" if pd.notna(hund) else "N/A"
                            _pdf_line(pdf, f"{month_name}: Ryfylke {ryf_txt}  Hundvåg {hund_txt}")
                        except Exception as exc:
                            logger.warning("PDF: tunnel-dekning per måned feilet: %s", exc)
                            continue
            except Exception as exc:
                logger.warning("PDF: Ryfast tunnel-fordeling feilet: %s", exc)

        # Data quality warnings
        try:
            warnings: List[str] = []
            cov_month = compute_monthly_coverage_summary(traffic_by_point, int(selected_year), point_ids)
            missing_months = cov_month[(cov_month["points_present"] < cov_month["points_expected"]) & (cov_month["points_present"] > 0)][
                "month_name"
            ].tolist()
            if missing_months:
                warnings.append("Manglende målepunkter: " + ", ".join([m for m in MONTH_NAMES if m in set(missing_months)]))

            none_months = cov_month[cov_month["points_present"] == 0]["month_name"].tolist()
            if none_months:
                warnings.append("Ingen data: " + ", ".join([m for m in MONTH_NAMES if m in set(none_months)]))

            anomalies = detect_monthly_anomalies(df, threshold_pct=float(ANOMALY_THRESHOLD_PCT))
            if not anomalies.empty:
                for r in anomalies.itertuples(index=False):
                    warnings.append(f"Anomali {int(r.year)} {str(r.month_name)}: {float(r.deviation_pct):+.1f}%")

            if warnings:
                pdf.ln(2)
                pdf.set_font("Helvetica", "B", 12)
                _pdf_line(pdf, "Data Quality Warnings", height=8)
                pdf.set_font("Helvetica", "", 10)
                for w in warnings[:15]:
                    pdf.multi_cell(0, 5, pdf_safe_text(f"- {w}"))
        except Exception as exc:
            logger.warning("PDF: feil ved generering av varselseksjon: %s", exc)

    output = pdf.output()
    return bytes(output) if isinstance(output, (bytes, bytearray)) else output.encode("latin-1")
