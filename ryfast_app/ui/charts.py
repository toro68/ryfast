"""Grafer og dataframe-hjelpere for visualisering."""

from typing import List, Optional

import altair as alt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ryfast_app.config import MONTH_NAMES


def create_advanced_visualization(df: pd.DataFrame, point: str, chart_type: str) -> go.Figure:
    template = "plotly_white"

    if chart_type == "line_with_confidence":
        fig = go.Figure()
        year_columns = [col for col in df.columns if col not in ["Month", "Month Name", "Week", "Volume"]]
        colors = px.colors.qualitative.Set1
        x_vals = df["Month Name"].tolist() if "Month Name" in df.columns else list(df.index)
        for i, year in enumerate(year_columns):
            y = pd.to_numeric(df[year], errors="coerce")
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=y,
                    mode="lines+markers",
                    name=str(year),
                    line=dict(color=colors[i % len(colors)], width=2.5),
                    marker=dict(size=7, symbol="circle"),
                    hovertemplate="%{x}: <b>%{y:,.0f}</b> ÅDT<extra>%{fullData.name}</extra>",
                )
            )
        fig.update_layout(
            title=dict(text=f"Trafikkutvikling for {point}", font=dict(size=15)),
            xaxis_title="Måned",
            yaxis=dict(title="Gjennomsnittlig døgntrafikk (ÅDT)", tickformat=","),
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            template=template,
            height=520,
            margin=dict(t=60, b=40),
        )
        return fig

    if chart_type == "heatmap":
        year_columns = [col for col in df.columns if col not in ["Month", "Month Name", "Week", "Volume"]]
        if len(year_columns) > 1 and "Month Name" in df.columns:
            table = df[["Month Name"] + year_columns].copy()
            z = table[year_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float).T
            x = table["Month Name"].tolist()
            y = [str(c) for c in year_columns]

            fig = go.Figure(
                data=go.Heatmap(
                    z=z,
                    x=x,
                    y=y,
                    colorscale="RdYlBu_r",
                    colorbar=dict(title="ÅDT"),
                    hoverongaps=False,
                    text=[[f"{v:,.0f}" if not np.isnan(v) else "" for v in row] for row in z],
                    texttemplate="%{text}",
                    textfont=dict(size=10),
                    hovertemplate="%{y} – %{x}: <b>%{z:,.0f}</b> ÅDT<extra></extra>",
                )
            )
            fig.update_layout(
                title=dict(text=f"Sesongmønster for {point}", font=dict(size=15)),
                xaxis_title="Måned",
                yaxis_title="År",
                template=template,
                height=max(320, 120 * len(y) + 80),
            )
            return fig
        return create_advanced_visualization(df, point, "line")

    if chart_type == "box":
        year_columns = [col for col in df.columns if col not in ["Month", "Month Name", "Week", "Volume"]]
        if not year_columns:
            return create_advanced_visualization(df, point, "line")

        fig = go.Figure()
        colors = px.colors.qualitative.Set1
        for i, year in enumerate(year_columns):
            yvals = pd.to_numeric(df[year], errors="coerce").to_numpy(dtype=float)
            yvals = [v for v in yvals.tolist() if not np.isnan(v)]
            if not yvals:
                continue
            fig.add_trace(
                go.Box(
                    y=yvals,
                    name=str(year),
                    boxpoints="all",
                    jitter=0.3,
                    marker=dict(color=colors[i % len(colors)], size=5, opacity=0.55),
                    line=dict(color=colors[i % len(colors)], width=2),
                    hovertemplate="%{y:,.0f} ÅDT<extra>%{fullData.name}</extra>",
                )
            )
        if not fig.data:
            return create_advanced_visualization(df, point, "line")
        fig.update_layout(
            title=dict(text=f"Trafikkfordeling for {point}", font=dict(size=15)),
            yaxis=dict(title="Gjennomsnittlig døgntrafikk (ÅDT)", tickformat=","),
            xaxis_title="År",
            template=template,
            height=520,
            margin=dict(t=60, b=40),
        )
        return fig

    # Default line chart
    if "Month Name" in df.columns:
        df_melted = df.melt(id_vars=["Month", "Month Name"], var_name="År", value_name="Trafikk")
        x_col = "Month Name"
    elif "Week" in df.columns:
        df_melted = df.melt(id_vars=["Week"], var_name="År", value_name="Trafikk")
        x_col = "Week"
    else:
        df_melted = df.melt(var_name="År", value_name="Trafikk")
        x_col = df_melted.index

    fig = px.line(
        df_melted,
        x=x_col,
        y="Trafikk",
        color="År",
        markers=True,
        title=f"Trafikkutvikling for {point}",
        labels={"Trafikk": "Gjennomsnittlig døgntrafikk (ÅDT)", x_col: "Periode"},
    )
    fig.update_traces(hovertemplate="%{x}: <b>%{y:,.0f}</b> ÅDT<extra>%{fullData.name}</extra>")
    fig.update_layout(
        template=template,
        height=520,
        hovermode="x unified",
        yaxis=dict(tickformat=","),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=60, b=40),
    )
    return fig


def _render_altair_chart(df: pd.DataFrame, point: str, chart_type: str):
    years = _year_columns(df)
    if not years:
        st.warning("Fant ingen årskolonner i datasettet. Viser linjediagram.")
        st.plotly_chart(create_advanced_visualization(df, point, "line"), width="stretch", key="fallback_line_no_years")
        return

    long_df = _long_year_df(df)
    if long_df.empty:
        st.warning("Fant ingen plottbare data. Viser linjediagram.")
        st.plotly_chart(create_advanced_visualization(df, point, "line"), width="stretch", key="fallback_line_empty")
        return

    if chart_type == "heatmap":
        if "Month Name" not in long_df.columns or len(years) < 2:
            st.info("Varmekart krever minst 2 år og månedsdata.")
            st.plotly_chart(create_advanced_visualization(df, point, "line"), width="stretch", key="fallback_line_no_heatmap")
            return

        month_sort = MONTH_NAMES
        rect = (
            alt.Chart(long_df)
            .mark_rect()
            .encode(
                x=alt.X("Month Name:N", sort=month_sort, title="Måned"),
                y=alt.Y("År:N", sort=sorted([str(y) for y in years]), title="År"),
                color=alt.Color("Trafikk:Q", scale=alt.Scale(scheme="redyellowblue", reverse=True), title="ÅDT"),
                tooltip=[
                    alt.Tooltip("År:N"),
                    alt.Tooltip("Month Name:N", title="Måned"),
                    alt.Tooltip("Trafikk:Q", format=",.0f", title="ÅDT"),
                ],
            )
        )
        text = (
            alt.Chart(long_df)
            .mark_text(fontSize=9)
            .encode(
                x=alt.X("Month Name:N", sort=month_sort),
                y=alt.Y("År:N", sort=sorted([str(y) for y in years])),
                text=alt.Text("Trafikk:Q", format=",.0f"),
                color=alt.condition(
                    alt.datum.Trafikk > long_df["Trafikk"].median(),
                    alt.value("white"),
                    alt.value("#333"),
                ),
            )
        )
        chart = (rect + text).properties(title=f"Sesongmønster for {point}", height=420)
        st.altair_chart(chart, width="stretch")
        return

    if chart_type == "box":
        chart = (
            alt.Chart(long_df)
            .mark_boxplot(extent="min-max")
            .encode(
                x=alt.X("År:N", sort=sorted([str(y) for y in years]), title="År"),
                y=alt.Y("Trafikk:Q", title="Gjennomsnittlig døgntrafikk"),
                color=alt.Color("År:N", legend=None),
                tooltip=[alt.Tooltip("År:N"), alt.Tooltip("Trafikk:Q", format=",.0f")],
            )
            .properties(title=f"Trafikkfordeling for {point}", height=420)
        )
        st.altair_chart(chart, width="stretch")
        return

    if chart_type == "line_with_confidence":
        if "Month Name" in long_df.columns:
            x = alt.X("Month Name:N", sort=MONTH_NAMES, title="Måned")
        elif "Week" in long_df.columns:
            x = alt.X("Week:N", title="Uke")
        else:
            x = alt.X("index:O", title="Periode")

        chart = (
            alt.Chart(long_df)
            .mark_line(point=alt.OverlayMarkDef(size=60))
            .encode(
                x=x,
                y=alt.Y("Trafikk:Q", title="Gjennomsnittlig døgntrafikk (ÅDT)", axis=alt.Axis(format=",d")),
                color=alt.Color("År:N", sort=sorted([str(y) for y in years]), title="År"),
                tooltip=[
                    alt.Tooltip("År:N", title="År"),
                    alt.Tooltip("Month Name:N", title="Måned") if "Month Name" in long_df.columns else alt.Tooltip("Week:N", title="Uke"),
                    alt.Tooltip("Trafikk:Q", format=",.0f", title="ÅDT"),
                ],
            )
            .properties(title=f"Trafikkutvikling for {point}", height=460)
        )
        st.altair_chart(chart, width="stretch")
        return

    st.plotly_chart(create_advanced_visualization(df, point, "line"), width="stretch", key="fallback_line_default")


def _year_columns(df: pd.DataFrame) -> List[str]:
    return [col for col in df.columns if col not in ["Month", "Month Name", "Week", "Volume"]]


def _long_year_df(df: pd.DataFrame) -> pd.DataFrame:
    years = _year_columns(df)
    if "Month Name" in df.columns:
        id_vars = ["Month", "Month Name"]
    elif "Week" in df.columns:
        id_vars = ["Week"]
    else:
        id_vars = []
    melted = df.melt(id_vars=id_vars, value_vars=years, var_name="År", value_name="Trafikk")
    melted["Trafikk"] = pd.to_numeric(melted["Trafikk"], errors="coerce")
    return melted.dropna(subset=["Trafikk"])


def _period_label_column(df: pd.DataFrame) -> Optional[str]:
    if "Month Name" in df.columns:
        return "Month Name"
    if "Week" in df.columns:
        return "Week"
    return None


def _pairwise_period_comparison(df: pd.DataFrame, baseline_year: str, compare_year: str) -> pd.DataFrame:
    label_col = _period_label_column(df)
    if label_col is None or baseline_year not in df.columns or compare_year not in df.columns:
        return pd.DataFrame()

    compare_df = df[[label_col, baseline_year, compare_year]].copy()
    compare_df[baseline_year] = pd.to_numeric(compare_df[baseline_year], errors="coerce")
    compare_df[compare_year] = pd.to_numeric(compare_df[compare_year], errors="coerce")
    compare_df = compare_df.dropna(subset=[baseline_year, compare_year], how="any")
    compare_df = compare_df.rename(columns={baseline_year: "Baseline", compare_year: "Sammenligning"})
    compare_df["Endring"] = compare_df["Sammenligning"] - compare_df["Baseline"]
    compare_df["Endring (%)"] = np.where(
        compare_df["Baseline"].notna() & (compare_df["Baseline"] != 0),
        compare_df["Endring"] / compare_df["Baseline"] * 100.0,
        np.nan,
    )
    compare_df["Retning"] = np.where(compare_df["Endring"].ge(0), "Opp", "Ned")
    return compare_df
