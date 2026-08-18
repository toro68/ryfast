"""Sykkelside: døgnvolum for sykkelregistreringspunkter på Nord-Jæren.

Siden har egne innstillinger i sidebaren (punkter, år, aggregering) og henter
data på klikk, uavhengig av Ryfast-analysen.

Døgn er hovedvisningen fordi sykling er værstyrt og har markert ukesrytme; et
månedssnitt skjuler nettopp det som er interessant.

Tre fallgruver som styrer visningen, alle synlige i ekte data:
- Summerer man flere punkter, faller summen når ett punkt mangler data. Det
  ser ut som nedgang i sykling. Derfor krever et pålitelig døgn at alle valgte
  punkter har tall, og `points_present` gjør grunnlaget synlig.
- Snittet per punkt faller ikke ved datahull, men skifter nivå når utvalget
  endres: punktene har svært ulikt volum, så døgn med få punkter ligger høyere
  enn døgn med alle. Begge aggregeringene bruker derfor et balansert panel når
  flere år sammenlignes.
- Dekningen varierer mellom år. Et år kan ha mistet hele vinteren, og siden
  sommertrafikken er 3-4x vintertrafikken, ville et årssnitt sammenlignet
  sommer mot helår. Årssammenligningen bruker derfor bare felles måneder.
"""

import logging
from datetime import date
from typing import Dict, List, Optional

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from ryfast_app.api import fetch_bicycle_points_years
from ryfast_app.bicycle import (
    WEEKDAY_NAMES,
    bicycle_point_options,
    comparable_months,
    coverage_summary,
    days_expected_in_year,
    mean_points_daily,
    monthly_profile,
    panel_point_ids,
    parse_daily_volumes,
    restrict_to_common_period,
    restrict_to_common_period_from,
    restrict_to_common_calendar_days,
    restrict_to_comparable_months,
    sum_points_daily,
    weekday_profile,
    year_comparison_summary,
)
from ryfast_app.config import (
    BICYCLE_DATA_START_YEAR,
    BICYCLE_DEFAULT_OPENING_DATE,
    BICYCLE_DEFAULT_POINT_ID,
    BICYCLE_MIN_COVERAGE_PCT,
    BICYCLE_POINTS,
    MONTH_NAMES,
)
from ryfast_app.processing import format_number

logger = logging.getLogger(__name__)

# Utvalgsmåter for punkter
SCOPE_SELECTED = "Valgte punkter"
SCOPE_ALL_OPERATIONAL = "Alle punkter i drift"
SCOPE_ALL = "Alle punkter"
SCOPE_MUNICIPALITY = "Alle i én kommune"

# Hvordan flere punkter slås sammen
AGG_SUM = "Sum (samlet antall)"
AGG_MEAN = "Snitt per punkt"


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
    days_absent = int(summary.get("days_absent") or 0)
    mean_cov = summary.get("mean_coverage_pct")
    cov_text = f"{float(mean_cov):.0f} %" if mean_cov is not None and pd.notna(mean_cov) else "ukjent"

    melding = (
        f"{days_reliable} av {days_total} døgn har dekning over {BICYCLE_MIN_COVERAGE_PCT:.0f} % "
        f"(snittdekning {cov_text})."
    )
    if days_missing:
        melding += f" {days_missing} døgn mangler tall helt."
    # Snittdekningen måles bare på døgn som finnes som rad, så den kan stå på
    # 100 % samtidig som store deler av perioden mangler. Si det eksplisitt.
    if days_absent:
        melding += (
            f" Av disse finnes {days_absent} døgn ikke i datagrunnlaget i det hele tatt, "
            "så snittdekningen over gjelder bare de øvrige døgnene."
        )

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


def _render_metrics(daily: pd.DataFrame, aggregation: str = AGG_SUM) -> None:
    reliable = daily[daily["reliable"]].dropna(subset=["volume"])
    er_snitt = aggregation == AGG_MEAN and "points_present" in daily.columns

    c1, c2, c3 = st.columns(3)
    with c1:
        if reliable.empty:
            st.metric("Siste registrerte døgn", "N/A")
        else:
            latest = reliable.sort_values("date").iloc[-1]
            st.metric(
                "Siste døgn per punkt" if er_snitt else "Siste registrerte døgn",
                format_number(latest["volume"]),
                help=f"{latest['weekday_name']} {latest['date']:%d.%m.%Y}",
            )
    with c2:
        if reliable.empty:
            st.metric("Travleste døgn", "N/A")
        else:
            top = reliable.loc[reliable["volume"].idxmax()]
            hjelp = f"{top['weekday_name']} {top['date']:%d.%m.%Y}"
            if er_snitt:
                # Uten dette leses ett punkts døgntall som utvalgets travleste.
                hjelp += f" — snitt over {int(top['points_present'])} punkter, ikke et samlet døgntall."
            st.metric("Travleste døgn", format_number(top["volume"]), help=hjelp)
    with c3:
        st.metric(
            "Døgn med god dekning",
            format_number(len(reliable)),
            help="Antall døgn som inngår i visningen og sammenligningene.",
        )
def _render_year_comparison(
    frames: Dict[int, pd.DataFrame], opening_date: Optional[date] = None
) -> None:
    """Sammenlign år på lik periode og like måneder.

    To korreksjoner, som begge er nødvendige for at prosenttallet skal bety
    noe: inneværende år er avkortet mot dagens dato, og dekningen varierer
    mellom år slik at et år kan mangle hele vinteren.
    """
    common = (
        restrict_to_common_period_from(frames, (opening_date.month, opening_date.day))
        if opening_date is not None
        else restrict_to_common_period(frames)
    )
    if len(common) < 2:
        st.info("Velg minst to år med data for å sammenligne.")
        return

    cutoff_note = ""
    any_frame = next((f for f in common.values() if not f.empty), None)
    if any_frame is not None:
        siste = pd.to_datetime(any_frame["date"]).max()
        # Bare verdt å nevne når vi faktisk har kuttet noe.
        if (siste.month, siste.day) != (12, 31):
            cutoff_note = f" Alle år er kuttet ved {siste:%d.%m} for å måle like perioder."

    # Sykkeltrafikken er 3-4x høyere om sommeren, så måneder som mangler i ett
    # år ville alene gitt en «endring». Sammenlign derfor bare felles måneder.
    months = comparable_months(common)
    comparable = restrict_to_comparable_months(common)
    if len(comparable) < 2 or not months:
        st.warning(
            "⚠️ Årene har ingen måneder med god dekning til felles, så et "
            "prosenttall ville sammenlignet ulike deler av året. Se månedsgrafen "
            "under for det som finnes."
        )
        comparable = common
        months = []

    comparable = restrict_to_common_calendar_days(comparable)

    summary = year_comparison_summary(comparable)
    if summary.empty:
        st.info("Ingen døgn med god nok dekning i de valgte årene.")
        return

    if opening_date is not None:
        note = (
            f"Sammenligningen starter {opening_date:%d.%m} i hvert år, fordi "
            f"Sykkelstamvegen åpnet {opening_date:%d.%m.%Y}." + cutoff_note
        )
    else:
        note = (
            "Sammenligningen bruker samme del av kalenderåret for alle år, siden "
            "inneværende år ikke er ferdig." + cutoff_note
        )
    if months and len(months) < 12:
        navn = ", ".join(MONTH_NAMES[m - 1] for m in months)
        note += (
            f" Snittene bygger bare på månedene alle årene har god dekning for "
            f"({navn}), slik at ulik dekning ikke leses som endring."
        )
    st.caption(note)

    baseline_year = int(summary.iloc[0]["year"])
    baseline_total = float(summary.iloc[0]["total_volume"])
    cols = st.columns(len(summary))
    for col, (_, row) in zip(cols, summary.iterrows()):
        with col:
            total = float(row["total_volume"])
            change = None
            if int(row["year"]) != baseline_year and baseline_total:
                change = (total - baseline_total) / baseline_total * 100.0
            st.metric(
                f"{int(row['year'])}",
                f"{format_number(total)} passeringer",
                delta=None if change is None else f"{change:+.1f} %",
                help=(
                    f"Registrerte passeringer over {int(row['days'])} identiske kalenderdøgn."
                    + ("" if int(row["year"]) == baseline_year else f" Endring målt mot {baseline_year}.")
                ),
            )

    # Rå døgntall, lagt på samme kalenderakse slik at år kan leses direkte mot
    # hverandre uten at et månedssnitt skjuler variasjonen.
    daily_rows = []
    for year, daily in comparable.items():
        part = daily.copy()
        dates = pd.to_datetime(part["date"])
        part["Sammenligningsdato"] = pd.to_datetime(
            {"year": 2000, "month": dates.dt.month, "day": dates.dt.day}
        )
        part["År"] = str(year)
        daily_rows.append(part)
    if daily_rows:
        plot_df = pd.concat(daily_rows, ignore_index=True)
        chart = (
            alt.Chart(plot_df)
            .mark_line(point=alt.OverlayMarkDef(size=24), strokeWidth=1.5)
            .encode(
                x=alt.X("Sammenligningsdato:T", title="Dato", axis=alt.Axis(format="%d.%m")),
                y=alt.Y("volume:Q", title="Registrerte passeringer per døgn"),
                color=alt.Color("År:N", legend=alt.Legend(title="År", orient="top")),
                tooltip=[
                    alt.Tooltip("År:N"),
                    alt.Tooltip("date:T", title="Dato"),
                    alt.Tooltip("volume:Q", format=",.0f", title="Passeringer"),
                ],
            )
            .properties(height=320)
        )
        st.altair_chart(chart, width="stretch")

    view = summary.copy()
    view["Registrerte passeringer"] = view["total_volume"].map(format_number)
    view["Endring"] = view["total_volume"].map(
        lambda total: "referanse"
        if not baseline_total or float(total) == baseline_total
        else f"{(float(total) - baseline_total) / baseline_total * 100.0:+.1f} %"
    )
    view = view.rename(columns={"year": "År", "days": "Døgn med god dekning"})
    st.dataframe(
        view[["År", "Registrerte passeringer", "Døgn med god dekning", "Endring"]],
        width="stretch",
        hide_index=True,
    )


def _render_points_map(point_ids: List[str]) -> None:
    """Kart over de valgte punktene."""
    rows = []
    for pid in point_ids:
        meta = BICYCLE_POINTS.get(pid)
        if meta:
            rows.append({"lat": float(meta["lat"]), "lon": float(meta["lon"])})
    if not rows:
        return
    st.map(pd.DataFrame(rows), zoom=10 if len(rows) > 1 else 12, size=60)


def _render_per_point_comparison(frames_by_point: Dict[str, pd.DataFrame]) -> None:
    """Rangér de valgte punktene mot hverandre på snitt per døgn."""
    rows = []
    for pid, daily in frames_by_point.items():
        if daily is None or daily.empty:
            continue
        good = daily[daily["reliable"]].dropna(subset=["volume"])
        if good.empty:
            continue
        meta = BICYCLE_POINTS.get(pid, {})
        rows.append(
            {
                "Punkt": str(meta.get("name", pid)),
                "Kommune": str(meta.get("municipality", "")),
                "mean_volume": float(good["volume"].mean()),
                "days": int(len(good)),
            }
        )
    if not rows:
        st.info("Ingen av punktene har døgn med god nok dekning.")
        return

    df = pd.DataFrame(rows).sort_values("mean_volume", ascending=False).reset_index(drop=True)
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            y=alt.Y("Punkt:N", sort="-x", title=None),
            x=alt.X("mean_volume:Q", title="Snitt syklister per døgn"),
            color=alt.Color("Kommune:N", legend=alt.Legend(title="Kommune", orient="top")),
            tooltip=[
                alt.Tooltip("Punkt:N"),
                alt.Tooltip("Kommune:N"),
                alt.Tooltip("mean_volume:Q", format=",.0f", title="Snitt per døgn"),
                alt.Tooltip("days:Q", title="Døgn med god dekning"),
            ],
        )
        .properties(height=max(220, 22 * len(df)))
    )
    st.altair_chart(chart, width="stretch")

    view = df.copy()
    view["Snitt per døgn"] = view["mean_volume"].map(format_number)
    view = view.rename(columns={"days": "Døgn med god dekning"})
    st.dataframe(
        view[["Punkt", "Kommune", "Snitt per døgn", "Døgn med god dekning"]],
        width="stretch",
        hide_index=True,
    )


def _render_basis_note(daily: pd.DataFrame, point_count: int, aggregation: str) -> None:
    """Si fra når summen bygger på færre punkter enn valgt."""
    if point_count < 2 or "points_present" not in daily.columns:
        return
    present = pd.to_numeric(daily["points_present"], errors="coerce")
    incomplete = int((present < point_count).sum())
    if not incomplete:
        return
    if aggregation == AGG_SUM:
        st.caption(
            f"ℹ️ {incomplete} av {len(daily)} døgn har tall fra færre enn {point_count} punkter. "
            "Disse er markert som lav dekning og utelatt fra snittene, siden en sum "
            "som mangler et punkt ser ut som nedgang."
        )
    else:
        st.caption(
            f"ℹ️ {incomplete} av {len(daily)} døgn har tall fra færre enn {point_count} punkter. "
            "Snittet per punkt tåler dette, men bygger da på et tynnere grunnlag."
        )


def _render_bicycle_sidebar() -> Dict[str, object]:
    """Innstillinger for sykkelvisningen, i sidebaren som for bildataene."""
    st.sidebar.markdown("---")
    st.sidebar.header("🚲 Sykkelinnstillinger")

    options = bicycle_point_options(include_retired=True)
    retired_labels = [lab for lab in options if "nedlagt" in lab]
    operational_labels = [lab for lab in options if "nedlagt" not in lab]
    default_labels = [
        label for label, point_id in options.items() if point_id == BICYCLE_DEFAULT_POINT_ID
    ]
    if not default_labels:
        default_labels = operational_labels[:1]

    municipalities = sorted({str(m.get("municipality", "")) for m in BICYCLE_POINTS.values()})

    with st.sidebar.container(border=True):
        scope = st.radio(
            "Hvilke punkter?",
            [SCOPE_SELECTED, SCOPE_ALL_OPERATIONAL, SCOPE_ALL, SCOPE_MUNICIPALITY],
            key="bicycle_scope",
            help="«Alle punkter» henter mange kall og tar lengre tid første gang.",
        )

        chosen_municipality = st.selectbox(
            "Kommune",
            municipalities,
            key="bicycle_municipality",
            disabled=scope != SCOPE_MUNICIPALITY,
        )

        chosen_labels = st.multiselect(
            "Velg sykkelpunkt",
            list(options.keys()),
            default=default_labels,
            key="bicycle_points",
            disabled=scope != SCOPE_SELECTED,
            help="Punkter merket «nedlagt» har historikk, men gir ingen nye tall.",
        )

        years = _available_years()
        chosen_years = st.multiselect(
            "År (velg flere for å sammenligne)",
            years,
            default=years[:2],
            key="bicycle_years",
        )

        aggregation = st.radio(
            "Når flere punkter er valgt",
            [AGG_SUM, AGG_MEAN],
            key="bicycle_aggregation",
            help=(
                "Sum gir samlet antall syklister, men faller når et punkt mangler data. "
                "Snitt per punkt er robust mot datahull, men er ikke et samlet antall."
            ),
        )

        with st.expander("🔧 Avansert", expanded=True):
            include_retired = st.checkbox(
                "Ta med nedlagte punkter i «alle»",
                value=False,
                key="bicycle_include_retired",
                help=f"{len(retired_labels)} punkter er nedlagt og gir ingen nye tall.",
            )
            min_coverage = st.slider(
                "Min. dekning per døgn (%)",
                0,
                100,
                int(BICYCLE_MIN_COVERAGE_PCT),
                key="bicycle_min_coverage",
                help="Døgn under terskelen beholder tallet, men utelates fra snittene.",
            )

        submitted = st.button(
            "🚲 Hent sykkeldata",
            type="primary",
            key="fetch_bicycle_data",
        )

    # Løs opp utvalget til faktiske ID-er
    if scope == SCOPE_ALL:
        point_ids = [
            pid
            for pid, m in BICYCLE_POINTS.items()
            if include_retired or m.get("operational", True)
        ]
    elif scope == SCOPE_ALL_OPERATIONAL:
        point_ids = [pid for pid, m in BICYCLE_POINTS.items() if m.get("operational", True)]
    elif scope == SCOPE_MUNICIPALITY:
        point_ids = [
            pid
            for pid, m in BICYCLE_POINTS.items()
            if str(m.get("municipality", "")) == chosen_municipality
            and (include_retired or m.get("operational", True))
        ]
    else:
        point_ids = [options[lab] for lab in chosen_labels]

    return {
        "submitted": bool(submitted),
        "point_ids": point_ids,
        "years": [int(y) for y in chosen_years],
        "aggregation": aggregation,
        "min_coverage": float(min_coverage),
        "scope": scope,
    }


def _selection_label(point_ids: List[str], scope: str, aggregation: str) -> str:
    """Kort beskrivelse av utvalget, til overskrifter og filnavn."""
    if len(point_ids) == 1:
        meta = BICYCLE_POINTS.get(point_ids[0], {})
        return f"{meta.get('name', point_ids[0])} ({meta.get('municipality', '')})"
    hva = "sum" if aggregation == AGG_SUM else "snitt per punkt"
    if scope == SCOPE_MUNICIPALITY:
        kommuner = {str(BICYCLE_POINTS[p]["municipality"]) for p in point_ids if p in BICYCLE_POINTS}
        if len(kommuner) == 1:
            return f"{kommuner.pop()} — {len(point_ids)} punkter ({hva})"
    return f"{len(point_ids)} sykkelpunkter ({hva})"


def render_bicycle_tab() -> None:
    """Sykkelsiden: flere punkter, flere år og innstillinger i sidebaren."""
    st.subheader("Utforsk sykkeltellingene")
    st.caption(
        "Døgntall fra Statens vegvesens sykkelregistreringspunkter. Døgn er "
        "hovedvisningen fordi sykling er værstyrt og varierer kraftig gjennom uka."
    )

    if not BICYCLE_POINTS:
        st.info("Ingen sykkelpunkter er konfigurert.")
        return

    settings = _render_bicycle_sidebar()

    if settings["submitted"]:
        point_ids = settings["point_ids"]
        years = settings["years"]
        if not point_ids:
            st.sidebar.warning("Velg minst ett sykkelpunkt")
        elif not years:
            st.sidebar.warning("Velg minst ett år")
        else:
            with st.spinner(
                f"Henter døgndata for {len(point_ids)} punkt(er) × {len(years)} år …"
            ):
                raw, failed_calls = fetch_bicycle_points_years(
                    point_ids,
                    years,
                    timeout_s=int(st.session_state.get("timeout_s", 60)),
                    use_cache=bool(st.session_state.get("use_cache", True)),
                )
            min_cov = settings["min_coverage"]
            # {år: {punkt: døgn-df}}
            parsed = {
                year: {
                    pid: parse_daily_volumes(payload, min_coverage_pct=min_cov)
                    for pid, payload in by_point.items()
                }
                for year, by_point in raw.items()
            }
            st.session_state.bicycle_result = {
                "parsed": parsed,
                "point_ids": point_ids,
                "years": years,
                "aggregation": settings["aggregation"],
                "scope": settings["scope"],
                "min_coverage": min_cov,
                "failed_calls": int(failed_calls),
                "expected_calls": len(point_ids) * len(years),
            }

    result = st.session_state.get("bicycle_result")
    if not result:
        st.info("Velg punkter og år under «🚲 Sykkelinnstillinger» i sidebaren, og trykk «Hent sykkeldata».")
        return

    parsed: Dict[int, Dict[str, pd.DataFrame]] = result["parsed"]
    point_ids: List[str] = result["point_ids"]
    aggregation: str = result["aggregation"]
    label = _selection_label(point_ids, result["scope"], aggregation)
    failed_calls = int(result.get("failed_calls") or 0)
    expected_calls = int(result.get("expected_calls") or 0)

    if not parsed:
        # Skill API-feil fra manglende registreringer: symptomet er det samme
        # (ingen tall), men det brukeren skal gjøre er stikk motsatt.
        if failed_calls:
            st.error(
                f"❌ Klarte ikke hente sykkeldata: {failed_calls} av {expected_calls} "
                "API-kall feilet. Dette er et problem med tilkoblingen til Statens "
                "vegvesen, ikke med punktene — prøv igjen, eller se «🧾 API-feil / "
                "status» i sidebaren for detaljer."
            )
        else:
            st.error(
                "❌ Ingen sykkeldata for utvalget. Punktene kan mangle registreringer "
                "for de valgte årene."
            )
        return

    # Delvis feil: vi har tall, men ikke alle. Uten dette varselet ser et
    # datahull fra en nede-periode ut som nedgang i sykling.
    if failed_calls:
        st.warning(
            f"⚠️ {failed_calls} av {expected_calls} API-kall feilet, så noen punkter "
            "eller år mangler helt. Tallene under bygger bare på det som kom gjennom — "
            "prøv igjen for et komplett grunnlag."
        )

    retired_valgt = [
        str(BICYCLE_POINTS[p]["name"])
        for p in point_ids
        if p in BICYCLE_POINTS and not BICYCLE_POINTS[p].get("operational", True)
    ]
    if retired_valgt:
        st.warning(
            f"⚠️ Nedlagte punkter i utvalget: {', '.join(retired_valgt)}. De har historikk, "
            "men gir ingen tall for inneværende år — manglende data der er ikke et datahull."
        )

    # Begge aggregeringene trenger et balansert panel når flere år sammenlignes,
    # men av to ulike grunner: summen faller ved hvert datahull, og snittet
    # skifter nivå når utvalget endres (punktene har svært ulikt volum). Uten
    # panelet måler vi ulike punkter i ulike år, og et nivåskift leses som
    # endring i sykling.
    panel: List[str] = []
    flere_aar = len(parsed) > 1
    if len(point_ids) > 1 and (aggregation == AGG_SUM or flere_aar):
        panel = panel_point_ids(parsed)
        if not panel:
            if aggregation == AGG_SUM:
                st.error(
                    "❌ Ingen av punktene har god nok dekning i alle de valgte årene til "
                    "å kunne summeres. Velg «Snitt per punkt», færre år eller færre punkter."
                )
                return
            st.warning(
                "⚠️ Ingen av punktene har god nok dekning i alle de valgte årene, så "
                "snittene måler ulike punkter i ulike år. Siden punktene har svært "
                "ulikt volum, kan et nivåskift her komme av utvalget framfor av "
                "sykkeltrafikken. Velg færre år eller færre punkter."
            )
        else:
            parsed = {
                year: {pid: f for pid, f in by_point.items() if pid in panel}
                for year, by_point in parsed.items()
            }
            utelatt = len(point_ids) - len(panel)
            if utelatt:
                hvorfor = (
                    "å ta dem med ville gjort summen lavere i de årene de mangler, "
                    "og det ville lest som nedgang i sykling."
                    if aggregation == AGG_SUM
                    else "med dem ville snittet målt ulike punkter i ulike år, og siden "
                    "punktene har svært ulikt volum ville nivåskiftet lest som endring."
                )
                hva = "Summen" if aggregation == AGG_SUM else "Sammenligningen"
                st.info(
                    f"ℹ️ {hva} bygger på {len(panel)} av {len(point_ids)} punkter. "
                    f"{utelatt} punkter er utelatt fordi de mangler data i minst ett av "
                    f"årene — {hvorfor}"
                )

    # Slå punkter sammen til én serie per år
    aggregate = sum_points_daily if aggregation == AGG_SUM else mean_points_daily
    effective_points = panel or point_ids
    frames: Dict[int, pd.DataFrame] = {}
    for year, by_point in parsed.items():
        if len(effective_points) == 1:
            only = next(iter(by_point.values()), None)
            if only is not None and not only.empty:
                frames[year] = only
        else:
            combined = aggregate(by_point)
            if not combined.empty:
                frames[year] = combined

    if not frames:
        st.error("❌ Ingen døgn med data i utvalget.")
        return

    years_with_data = sorted(frames)
    st.markdown(f"**{label} — {', '.join(str(y) for y in years_with_data)}**")

    opening_date = (
        BICYCLE_DEFAULT_OPENING_DATE
        if len(effective_points) == 1 and effective_points[0] == BICYCLE_DEFAULT_POINT_ID
        else None
    )
    if opening_date is not None:
        st.info(
            f"Sykkelstamvegen åpnet {opening_date:%d.%m.%Y}. Utviklingen under "
            "sammenligner samme kalenderperiode fra 16. juni i hvert år, slik at "
            "tall før åpningen ikke blandes inn i referansen."
        )

    if len(years_with_data) > 1:
        st.markdown(
            "### 📅 Utvikling etter åpningen"
            if opening_date is not None
            else "### 📅 Sammenligning mellom år"
        )
        _render_year_comparison(frames, opening_date=opening_date)
        st.markdown("---")

    # Detaljvisningen gjelder det nyeste året
    latest = years_with_data[-1]
    daily = frames[latest]
    st.markdown(f"### 🔍 Detaljer for {latest}")
    # Nevneren er periodens lengde, ikke antall rader: døgn der ingen punkter
    # har tall finnes ikke som rad i aggregatet.
    _render_coverage_banner(
        coverage_summary(daily, days_expected=days_expected_in_year(latest))
    )
    _render_basis_note(daily, len(effective_points), aggregation)
    _render_metrics(daily, aggregation)
    _render_daily_chart(daily, label)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Ukesprofil**")
        _render_weekday_chart(daily)
    with c2:
        st.markdown("**Sesongprofil**")
        _render_season_chart(daily)

    if len(point_ids) > 1:
        with st.expander(f"🏆 Sammenligning mellom punkter ({latest})", expanded=False):
            _render_per_point_comparison(parsed.get(latest, {}))

    with st.expander("🗺️ Hvor står punktene?"):
        _render_points_map(point_ids)

    with st.expander("📊 Døgndata"):
        view = daily.copy()
        view["date"] = pd.to_datetime(view["date"]).dt.strftime("%d.%m.%Y")
        rename = {
            "date": "Dato",
            "volume": "Syklister",
            "coverage_pct": "Dekning (%)",
            "weekday_name": "Ukedag",
            "season": "Sesong",
            "reliable": "God dekning",
            "points_present": "Punkter med tall",
        }
        view = view.rename(columns=rename)
        vis_kolonner = [c for c in rename.values() if c in view.columns]
        st.dataframe(view[vis_kolonner], width="stretch", hide_index=True)

        # Full CSV med alle år, ikke bare det som vises
        eksport = pd.concat(
            [f.assign(year=y) for y, f in sorted(frames.items())], ignore_index=True
        )
        st.download_button(
            "⬇️ Last ned som CSV",
            data=eksport.to_csv(index=False).encode("utf-8"),
            file_name=f"sykkel_{len(point_ids)}punkt_{years_with_data[0]}-{years_with_data[-1]}.csv",
            mime="text/csv",
        )
