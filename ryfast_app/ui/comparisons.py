"""Sammenligningsvisninger: parvise år, måneder og uker."""

import altair as alt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from ryfast_app.metrics import calculate_growth_rates
from ryfast_app.processing import (
    calculate_yearly_total_from_monthly_averages,
    format_number,
    overlapping_months,
)
from ryfast_app.ui.charts import (
    _pairwise_period_comparison,
    _period_label_column,
    _render_altair_chart,
    _year_columns,
    create_advanced_visualization,
)


def _render_pairwise_year_comparison(df: pd.DataFrame) -> None:
    year_columns = sorted([str(col) for col in _year_columns(df) if str(col).isdigit()], key=int)
    if len(year_columns) < 2:
        return

    st.markdown("### År-mot-år")
    baseline_default = max(0, len(year_columns) - 2)
    compare_default = len(year_columns) - 1
    col1, col2 = st.columns(2)
    with col1:
        baseline_year = st.selectbox("Basisår", year_columns, index=baseline_default, key="pairwise_baseline_year")
    with col2:
        compare_year = st.selectbox("Sammenlign med", year_columns, index=compare_default, key="pairwise_compare_year")

    if baseline_year == compare_year:
        st.info("Velg to ulike år for å se differanser.")
        return

    compare_df = _pairwise_period_comparison(df, baseline_year, compare_year)
    if compare_df.empty:
        st.info("Fant ingen perioder med sammenlignbare verdier for de valgte årene.")
        return

    # Sammenlign bare måneder der begge år har tall. Ellers måles et delår mot
    # et helår, og et år med vekst framstår som kraftig nedgang.
    shared_months = overlapping_months(df, [baseline_year, compare_year])
    df_shared = df[pd.to_numeric(df["Month"], errors="coerce").isin(shared_months)] if "Month" in df.columns else df
    total_baseline, _, _ = calculate_yearly_total_from_monthly_averages(df_shared, int(baseline_year))
    total_compare, compare_months, _ = calculate_yearly_total_from_monthly_averages(df_shared, int(compare_year))
    total_delta = total_compare - total_baseline
    total_delta_pct = (total_delta / total_baseline * 100.0) if total_baseline else np.nan
    best_row = compare_df.loc[compare_df["Endring"].idxmax()]
    worst_row = compare_df.loc[compare_df["Endring"].idxmin()]
    label_col = _period_label_column(df) or "Periode"
    period_label = "Måned" if label_col == "Month Name" else ("Uke" if label_col == "Week" else "Periode")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(
            f"Totalt {compare_year} ({compare_months} mnd)",
            format_number(total_compare),
            None if pd.isna(total_delta_pct) else f"{total_delta_pct:+.1f}% vs {baseline_year}",
        )
    with m2:
        st.metric("Differanse i totaltrafikk", format_number(total_delta) if pd.notna(total_delta) else "N/A")
    with m3:
        st.metric(f"Sterkeste {period_label.lower()}", str(best_row[label_col]), f"{float(best_row['Endring']):+,.0f}".replace(",", " "))
    with m4:
        st.metric(f"Svakeste {period_label.lower()}", str(worst_row[label_col]), f"{float(worst_row['Endring']):+,.0f}".replace(",", " "))

    if compare_months and compare_months < 12:
        st.caption(
            f"⚖️ Sammenligningen bruker de {compare_months} månedene der begge år har data, "
            "slik at like perioder måles mot hverandre."
        )
    st.caption(
        "Kortene over viser totaltrafikk beregnet fra måneds-ÅDT. Grafen under viser periodevis endring i ÅDT mellom årene."
    )

    fig = px.bar(
        compare_df,
        x=label_col,
        y="Endring",
        color="Retning",
        color_discrete_map={"Opp": "#2ca02c", "Ned": "#d62728"},
        title=f"Differanse per periode: {compare_year} minus {baseline_year}",
        labels={label_col: "Periode", "Endring": "Endring i ÅDT"},
        custom_data=["Baseline", "Sammenligning", "Endring (%)"],
    )
    fig.update_traces(
        hovertemplate=(
            "%{x}<br>"
            f"{baseline_year}: <b>%{{customdata[0]:,.0f}}</b><br>"
            f"{compare_year}: <b>%{{customdata[1]:,.0f}}</b><br>"
            "Endring: <b>%{y:+,.0f}</b><br>"
            "Endring (%): <b>%{customdata[2]:+.1f}%</b><extra></extra>"
        )
    )
    fig.update_layout(
        template="plotly_white",
        height=380,
        yaxis=dict(tickformat=","),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=60, b=40),
    )
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.6)
    st.plotly_chart(fig, width="stretch", key="pairwise_year_delta")


def _render_single_year_month_comparison(df: pd.DataFrame) -> None:
    year_columns = [col for col in _year_columns(df) if str(col).isdigit()]
    if len(year_columns) != 1 or "Month Name" not in df.columns:
        return

    year_col = str(year_columns[0])
    compare_df = df[["Month Name", year_col]].copy()
    compare_df[year_col] = pd.to_numeric(compare_df[year_col], errors="coerce")
    compare_df = compare_df.dropna(subset=[year_col])
    if compare_df.empty:
        return

    best_row = compare_df.loc[compare_df[year_col].idxmax()]
    worst_row = compare_df.loc[compare_df[year_col].idxmin()]
    spread = float(best_row[year_col]) - float(worst_row[year_col])
    average = float(compare_df[year_col].mean())

    st.markdown("### Månedssammenligning")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Høyeste måned", str(best_row["Month Name"]), format_number(best_row[year_col]))
    with m2:
        st.metric("Laveste måned", str(worst_row["Month Name"]), format_number(worst_row[year_col]))
    with m3:
        st.metric("Spenn", format_number(spread))
    with m4:
        st.metric("Snitt valgte måneder", format_number(average))

    ranked = compare_df.sort_values(year_col, ascending=False).copy()
    fig = px.bar(
        ranked,
        x=year_col,
        y="Month Name",
        orientation="h",
        title=f"Rangering av måneder i {year_col}",
        labels={year_col: "ÅDT", "Month Name": "Måned"},
    )
    fig.update_traces(hovertemplate="%{y}: <b>%{x:,.0f}</b> ÅDT<extra></extra>")
    fig.update_layout(template="plotly_white", height=420, xaxis=dict(tickformat=","), margin=dict(t=60, b=40))
    st.plotly_chart(fig, width="stretch", key="month_ranking")


def _render_weekly_change_summary(df: pd.DataFrame) -> None:
    if "Week" not in df.columns or "Volume" not in df.columns:
        return

    weekly = df[["Week", "Volume"]].copy()
    # astype(float) unngår nullable Int64: diff() på Int64 gir pd.NA, som np.where ikke tåler
    weekly["Volume"] = pd.to_numeric(weekly["Volume"], errors="coerce").astype(float)
    weekly = weekly.dropna(subset=["Volume"])
    if len(weekly) < 2:
        return

    weekly["Endring"] = weekly["Volume"].diff()
    weekly["Endring (%)"] = weekly["Volume"].pct_change() * 100.0
    weekly["Retning"] = np.where(weekly["Endring"] >= 0, "Opp", "Ned")

    latest_change = weekly.iloc[-1]
    best_row = weekly.loc[weekly["Volume"].idxmax()]
    worst_row = weekly.loc[weekly["Volume"].idxmin()]

    st.markdown("### Uke-til-uke")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Siste uke", str(latest_change["Week"]), format_number(latest_change["Volume"]))
    with m2:
        delta_txt = "N/A" if pd.isna(latest_change["Endring"]) else f"{float(latest_change['Endring']):+,.0f}".replace(",", " ")
        st.metric("Endring fra forrige uke", delta_txt)
    with m3:
        st.metric("Høyeste uke", str(best_row["Week"]), format_number(best_row["Volume"]))
    with m4:
        st.metric("Laveste uke", str(worst_row["Week"]), format_number(worst_row["Volume"]))

    weekly_delta = weekly.dropna(subset=["Endring"]).copy()
    if weekly_delta.empty:
        return
    fig = px.bar(
        weekly_delta,
        x="Week",
        y="Endring",
        color="Retning",
        color_discrete_map={"Opp": "#2ca02c", "Ned": "#d62728"},
        title="Endring fra uke til uke",
        labels={"Week": "Uke", "Endring": "Endring i ÅDT"},
        custom_data=["Volume", "Endring (%)"],
    )
    fig.update_traces(
        hovertemplate=(
            "%{x}<br>"
            "Volum: <b>%{customdata[0]:,.0f}</b><br>"
            "Endring: <b>%{y:+,.0f}</b><br>"
            "Endring (%): <b>%{customdata[1]:+.1f}%</b><extra></extra>"
        )
    )
    fig.update_layout(template="plotly_white", height=360, yaxis=dict(tickformat=","), margin=dict(t=60, b=40))
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.6)
    st.plotly_chart(fig, width="stretch", key="weekly_delta")


def create_comparison_dashboard(df: pd.DataFrame, point: str):
    col1, col2 = st.columns(2)
    with col1:
        is_weekly = "Volume" in df.columns
        year_columns = _year_columns(df)
        can_heatmap = (not is_weekly) and ("Month Name" in df.columns) and (len(year_columns) > 1)

        if is_weekly:
            chart_options = ["line"]
        else:
            chart_options = ["line", "box", "line_with_confidence"]
            if can_heatmap:
                chart_options.insert(1, "heatmap")

        chart_type = st.selectbox(
            "Velg diagramtype",
            chart_options,
            key="chart_type_selector",
            format_func=lambda x: {
                "line": "Linjediagram",
                "heatmap": "Varmekart",
                "box": "Boksplot",
                "line_with_confidence": "Linje med konfidensintervall",
            }[x],
        )
        if not can_heatmap and not is_weekly:
            st.caption("Varmekart vises kun når du sammenligner minst 2 år (sesongmønster på tvers av år).")

    with col2:
        show_growth = st.checkbox("Vis vekstrater", value=False)

    if chart_type in {"heatmap", "box", "line_with_confidence"}:
        _render_altair_chart(df, point, chart_type)
    elif is_weekly:
        # Dedicated bar chart for weekly volume
        week_order = sorted(
            df["Week"].astype(str).tolist(),
            key=lambda s: int("".join([c for c in s if c.isdigit()]) or "0"),
        )
        fig = px.bar(
            df, x="Week", y="Volume",
            title=f"Ukentlig trafikkvolum for {point}",
            labels={"Volume": "Gjennomsnittlig døgntrafikk (ÅDT)", "Week": "Uke"},
            category_orders={"Week": week_order},
        )
        fig.update_traces(
            marker_color="#1f77b4",
            hovertemplate="%{x}: <b>%{y:,.0f}</b> ÅDT<extra></extra>",
        )
        fig.update_layout(
            template="plotly_white",
            height=480,
            yaxis=dict(tickformat=","),
            margin=dict(t=60, b=40),
        )
        st.plotly_chart(fig, width="stretch", key="weekly_bar", config={"displayModeBar": True})
    else:
        fig = create_advanced_visualization(df, point, chart_type)
        st.plotly_chart(
            fig,
            width="stretch",
            key=f"main_chart_{chart_type}",
            config={"displayModeBar": True, "responsive": True},
        )

    if is_weekly:
        _render_weekly_change_summary(df)
    elif len(year_columns) >= 2:
        _render_pairwise_year_comparison(df)
    else:
        _render_single_year_month_comparison(df)

    if show_growth:
        growth_df = calculate_growth_rates(df)
        growth_columns = [col for col in growth_df.columns if "Vekst" in col]
        if not growth_columns:
            st.info("Vekstrater krever minst 2 år i samme visning (bruk «Sammenlign år» og velg flere år).")
        else:
            st.subheader("Vekstrater (år-til-år)")
            if "Month Name" in growth_df.columns:
                id_vars = ["Month", "Month Name"]
                x_col = "Month Name"
            elif "Week" in growth_df.columns:
                id_vars = ["Week"]
                x_col = "Week"
            else:
                id_vars = []
                x_col = growth_df.index

            growth_melted = growth_df.melt(
                id_vars=id_vars,
                value_vars=growth_columns,
                var_name="Periode",
                value_name="Vekst (%)",
            )
            growth_melted["Vekst (%)"] = pd.to_numeric(growth_melted["Vekst (%)"], errors="coerce")
            growth_melted = growth_melted.dropna(subset=["Vekst (%)"])
            if growth_melted.empty:
                st.warning("Fant ingen gyldige vekstrater i datasettet (mangler baseline eller nullverdier).")
            else:
                zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="black", strokeDash=[4, 4]).encode(y="y:Q")
                bars = (
                    alt.Chart(growth_melted)
                    .mark_bar()
                    .encode(
                        x=alt.X(f"{x_col}:N", title="Periode"),
                        y=alt.Y("Vekst (%):Q", title="Vekst (%)"),
                        color=alt.condition(
                            alt.datum["Vekst (%)"] >= 0,
                            alt.value("#2ca02c"),
                            alt.value("#d62728"),
                        ),
                        tooltip=[alt.Tooltip("Periode:N"), alt.Tooltip("Vekst (%):Q", format="+.1f")],
                    )
                    .properties(height=320)
                )
                text = (
                    alt.Chart(growth_melted)
                    .mark_text(dy=alt.ExprRef("datum['Vekst (%)'] >= 0 ? -8 : 8"), fontSize=10)
                    .encode(
                        x=alt.X(f"{x_col}:N"),
                        y=alt.Y("Vekst (%):Q"),
                        text=alt.Text("Vekst (%):Q", format="+.1f"),
                        color=alt.condition(
                            alt.datum["Vekst (%)"] >= 0,
                            alt.value("#1a7a1a"),
                            alt.value("#a01a1a"),
                        ),
                    )
                )
                st.altair_chart((bars + zero + text).properties(title="År-til-år vekstrater"), width="stretch")
