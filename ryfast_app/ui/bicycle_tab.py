"""Sykkelfane: døgnvolum for ett registreringspunkt på Nord-Jæren.

Fanen har eget punktvalg og henter data på klikk, uavhengig av bilanalysen i
de andre fanene. Døgn er hovedvisningen fordi sykling er værstyrt og har
markert ukesrytme; et månedssnitt skjuler nettopp det som er interessant.
"""

import logging
from datetime import date
from typing import Dict, List, Optional

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from ryfast_app.api import fetch_bicycle_year
from ryfast_app.bicycle import (
    WEEKDAY_NAMES,
    bicycle_point_options,
    coverage_summary,
    monthly_profile,
    parse_daily_volumes,
    weekday_profile,
    weekend_vs_weekday,
)
from ryfast_app.config import (
    BICYCLE_DATA_START_YEAR,
    BICYCLE_MIN_COVERAGE_PCT,
    BICYCLE_POINTS,
    MONTH_NAMES,
)
from ryfast_app.processing import format_number

logger = logging.getLogger(__name__)

SEASON_ORDER = ["Vinter", "Vår", "Sommer", "Høst"]


def _available_years(today: Optional[date] = None) -> List[int]:
    """År det finnes sykkeldata for, nyeste først."""
    today = today or date.today()
    return list(range(today.year, BICYCLE_DATA_START_YEAR - 1, -1))


def _render_coverage_banner(summary: Dict[str, object]) -> None:
    """Meld om datagrunnlaget før tallene tolkes."""
    days_total = int(summary.get("days_total") or 0)
    if not days_total:
        return
    days_reliable = int(summary.get("days_reliable") or 0)
    days_missing = int(summary.get("days_missing") or 0)
    mean_cov = summary.get("mean_coverage_pct")
    cov_text = f"{float(mean_cov):.0f} %" if mean_cov is not None and pd.notna(mean_cov) else "ukjent"

    melding = (
        f"{days_reliable} av {days_total} døgn har dekning over {BICYCLE_MIN_COVERAGE_PCT:.0f} % "
        f"(snittdekning {cov_text})."
    )
    if days_missing:
        melding += f" {days_missing} døgn mangler tall helt."

    # Sykkeltall er små, så lav dekning slår kraftigere ut enn for bil.
    if days_reliable == days_total:
        st.success(f"✅ {melding}")
    elif days_reliable >= days_total * 0.8:
        st.info(f"ℹ️ {melding} Døgn under terskel er utelatt fra snittene.")
    else:
        st.warning(f"⚠️ {melding} Snittene bygger på et tynt grunnlag — tolk dem med varsomhet.")


def _render_daily_chart(daily: pd.DataFrame, label: str) -> None:
    """Døgnkurve med 7-dagers glidende snitt, som demper ukesrytmen."""
    plot_df = daily.dropna(subset=["volume"]).copy()
    if plot_df.empty:
        st.info("Ingen døgn med tall i perioden.")
        return

    plot_df["date"] = pd.to_datetime(plot_df["date"])
    # Glidende snitt bare over pålitelige døgn, ellers trekker hull kurven ned.
    reliable = plot_df["volume"].where(plot_df["reliable"])
    plot_df["rolling"] = reliable.rolling(7, min_periods=4).mean()
    plot_df["Dekning"] = np.where(plot_df["reliable"], "God dekning", "Lav dekning")

    base = alt.Chart(plot_df).encode(
        x=alt.X("date:T", title="Dato"),
        tooltip=[
            alt.Tooltip("date:T", title="Dato"),
            alt.Tooltip("weekday_name:N", title="Ukedag"),
            alt.Tooltip("volume:Q", format=",.0f", title="Syklister"),
            alt.Tooltip("coverage_pct:Q", format=".0f", title="Dekning (%)"),
        ],
    )
    punkter = base.mark_circle(size=18, opacity=0.55).encode(
        y=alt.Y("volume:Q", title="Syklister per døgn"),
        color=alt.Color(
            "Dekning:N",
            scale=alt.Scale(domain=["God dekning", "Lav dekning"], range=["#1f77b4", "#c7c7c7"]),
            legend=alt.Legend(title=None, orient="top"),
        ),
    )
    linje = base.mark_line(color="#d62728", strokeWidth=2).encode(
        y=alt.Y("rolling:Q", title="Syklister per døgn")
    )
    st.altair_chart((punkter + linje).properties(height=340), width="stretch")
    st.caption(
        f"Punkter er enkeltdøgn ved {label}. Rød linje er 7-dagers glidende snitt "
        "over døgn med god dekning."
    )


def _render_weekday_chart(daily: pd.DataFrame) -> None:
    profile = weekday_profile(daily)
    if profile.empty:
        st.info("Ingen døgn med god nok dekning til å vise ukesprofil.")
        return
    chart = (
        alt.Chart(profile)
        .mark_bar()
        .encode(
            x=alt.X("weekday_name:N", sort=WEEKDAY_NAMES, title="Ukedag"),
            y=alt.Y("mean_volume:Q", title="Snitt per døgn"),
            color=alt.Color("weekday_name:N", sort=WEEKDAY_NAMES, legend=None),
            tooltip=[
                alt.Tooltip("weekday_name:N", title="Ukedag"),
                alt.Tooltip("mean_volume:Q", format=",.0f", title="Snitt"),
                alt.Tooltip("days:Q", title="Antall døgn"),
            ],
        )
        .properties(height=280)
    )
    st.altair_chart(chart, width="stretch")


def _render_season_chart(daily: pd.DataFrame) -> None:
    """Månedsprofil: sesongvariasjonen gjennom året."""
    profile = monthly_profile(daily)
    if profile.empty:
        st.info("Ingen døgn med god nok dekning til å vise sesongprofil.")
        return
    profile = profile.copy()
    profile["Måned"] = profile["month"].map(lambda m: MONTH_NAMES[int(m) - 1])
    chart = (
        alt.Chart(profile)
        .mark_bar()
        .encode(
            x=alt.X("Måned:N", sort=MONTH_NAMES, title="Måned"),
            y=alt.Y("mean_volume:Q", title="Snitt per døgn"),
            color=alt.Color(
                "mean_volume:Q",
                scale=alt.Scale(scheme="viridis"),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("Måned:N"),
                alt.Tooltip("mean_volume:Q", format=",.0f", title="Snitt"),
                alt.Tooltip("days:Q", title="Antall døgn"),
            ],
        )
        .properties(height=280)
    )
    st.altair_chart(chart, width="stretch")


def _render_metrics(daily: pd.DataFrame) -> None:
    reliable = daily[daily["reliable"]].dropna(subset=["volume"])
    split = weekend_vs_weekday(daily)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        snitt = reliable["volume"].mean() if not reliable.empty else None
        st.metric("Snitt per døgn", format_number(snitt) if snitt is not None else "N/A")
    with c2:
        if reliable.empty:
            st.metric("Travleste døgn", "N/A")
        else:
            top = reliable.loc[reliable["volume"].idxmax()]
            st.metric(
                "Travleste døgn",
                format_number(top["volume"]),
                help=f"{top['weekday_name']} {top['date']:%d.%m.%Y}",
            )
    with c3:
        st.metric(
            "Hverdag vs helg",
            f"{split['weekend_share_pct']:.0f} %" if split["weekend_share_pct"] is not None else "N/A",
            help=(
                "Helgesnittet som andel av hverdagssnittet. Lav andel tyder på "
                "arbeidsreiser, høy andel på tur- og fritidssykling."
            ),
        )
    with c4:
        total = reliable["volume"].sum() if not reliable.empty else 0
        st.metric(
            "Sum passeringer",
            format_number(total),
            help="Sum over døgn med god dekning; ikke et fullstendig årstall når døgn mangler.",
        )


def _render_map(point_id: str) -> None:
    meta = BICYCLE_POINTS.get(point_id)
    if not meta:
        return
    st.map(
        pd.DataFrame({"lat": [float(meta["lat"])], "lon": [float(meta["lon"])]}),
        zoom=12,
        size=60,
    )


def render_bicycle_tab() -> None:
    """Sykkelfanen: eget punkt- og årsvalg, henter data på klikk."""
    st.subheader("🚲 Sykkeltrafikk på Nord-Jæren")
    st.caption(
        "Døgntall fra Statens vegvesens sykkelregistreringspunkter. Døgn er "
        "hovedvisningen fordi sykling er værstyrt og varierer kraftig gjennom uka."
    )

    options = bicycle_point_options(include_retired=True)
    if not options:
        st.info("Ingen sykkelpunkter er konfigurert.")
        return

    years = _available_years()
    with st.form("bicycle_controls"):
        c1, c2 = st.columns([3, 1])
        with c1:
            label = st.selectbox(
                "Velg sykkelpunkt",
                list(options.keys()),
                key="bicycle_point",
                help="Punkter merket «nedlagt» har historikk, men gir ingen nye tall.",
            )
        with c2:
            year = st.selectbox("År", years, key="bicycle_year")
        submitted = st.form_submit_button("🚲 Hent sykkeldata", type="primary")

    if submitted:
        point_id = options[label]
        meta = BICYCLE_POINTS.get(point_id, {})
        retired = not bool(meta.get("operational", True))
        with st.spinner(f"Henter døgndata for {label} i {year} …"):
            payload = fetch_bicycle_year(
                point_id,
                int(year),
                timeout_s=int(st.session_state.get("timeout_s", 60)),
                use_cache=bool(st.session_state.get("use_cache", True)),
            )
        daily = parse_daily_volumes(payload)
        st.session_state.bicycle_result = {
            "daily": daily,
            "label": label,
            "point_id": point_id,
            "year": int(year),
            "retired": retired,
        }

    result = st.session_state.get("bicycle_result")
    if not result:
        st.info("Velg sykkelpunkt og år, og trykk «Hent sykkeldata».")
        return

    daily = result["daily"]
    label = result["label"]
    year = result["year"]

    if result["retired"]:
        st.warning(
            f"⚠️ {label} er nedlagt. Punktet har historikk, men gir ingen tall for "
            "inneværende år — manglende data her er ikke et datahull."
        )

    if daily is None or daily.empty:
        st.error(
            f"❌ Ingen sykkeldata for {label} i {year}. "
            "Punktet kan mangle registreringer for dette året."
        )
        return

    st.markdown(f"**{label} — {year}**")
    _render_coverage_banner(coverage_summary(daily))
    _render_metrics(daily)

    _render_daily_chart(daily, label)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Ukesprofil**")
        _render_weekday_chart(daily)
    with c2:
        st.markdown("**Sesongprofil**")
        _render_season_chart(daily)

    with st.expander("🗺️ Hvor står punktet?"):
        _render_map(result["point_id"])

    with st.expander("📊 Døgndata"):
        view = daily.copy()
        view["date"] = pd.to_datetime(view["date"]).dt.strftime("%d.%m.%Y")
        view = view.rename(
            columns={
                "date": "Dato",
                "volume": "Syklister",
                "coverage_pct": "Dekning (%)",
                "weekday_name": "Ukedag",
                "season": "Sesong",
                "reliable": "God dekning",
            }
        )
        st.dataframe(
            view[["Dato", "Ukedag", "Syklister", "Dekning (%)", "Sesong", "God dekning"]],
            width="stretch",
            hide_index=True,
        )
        st.download_button(
            "⬇️ Last ned som CSV",
            data=daily.to_csv(index=False).encode("utf-8"),
            file_name=f"sykkel_{result['point_id']}_{year}.csv",
            mime="text/csv",
        )
