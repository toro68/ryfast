"""Databehandling for sykkelregistreringer på Nord-Jæren.

Streamlit-fri og ren pandas, slik at logikken kan testes uten nettverk.
Sykkeltall skiller seg fra biltall på to måter som styrer modulen:

- Volumene er små (titalls til hundretalls per døgn), så lav dekning slår
  kraftigere ut enn for bil. Dager under terskel skilles derfor ut framfor å
  blandes inn i snittet.
- Sykling er værstyrt og har markert ukesrytme. Døgn er den meningsbærende
  oppløsningen; et månedssnitt skjuler nettopp det som er interessant.
"""

from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ryfast_app.config import BICYCLE_MIN_COVERAGE_PCT, BICYCLE_POINTS

WEEKDAY_NAMES = [
    "Mandag",
    "Tirsdag",
    "Onsdag",
    "Torsdag",
    "Fredag",
    "Lørdag",
    "Søndag",
]

SEASON_BY_MONTH = {
    12: "Vinter", 1: "Vinter", 2: "Vinter",
    3: "Vår", 4: "Vår", 5: "Vår",
    6: "Sommer", 7: "Sommer", 8: "Sommer",
    9: "Høst", 10: "Høst", 11: "Høst",
}


def bicycle_point_options(include_retired: bool = True) -> Dict[str, str]:
    """Etiketter for nedtrekkslisten, sortert på kommune og navn.

    Nedlagte punkter merkes eksplisitt: de har historikk, men gir ingen tall
    for inneværende år, og det skal ikke leses som et datahull.
    """
    items = []
    for pid, meta in BICYCLE_POINTS.items():
        operational = bool(meta.get("operational", True))
        if not operational and not include_retired:
            continue
        label = f"{meta['name']} ({meta['municipality']})"
        if not operational:
            label += " – nedlagt"
        items.append((str(meta["municipality"]), str(meta["name"]), pid, label))
    items.sort()
    return {label: pid for _, _, pid, label in items}


def parse_daily_volumes(
    api_payload: Optional[Dict],
    min_coverage_pct: float = BICYCLE_MIN_COVERAGE_PCT,
) -> pd.DataFrame:
    """Pakk ut byDay-svaret til én rad per døgn.

    Returnerer kolonnene date, volume, coverage_pct, weekday, weekday_name,
    is_weekend, month, season og reliable. `volume` beholdes selv når dekningen
    er lav, men `reliable` er da False slik at kallstedet kan velge å utelate den.
    """
    cols = [
        "date", "volume", "coverage_pct", "weekday", "weekday_name",
        "is_weekend", "month", "season", "reliable",
    ]
    if not api_payload:
        return pd.DataFrame(columns=cols)

    node_list = (
        (api_payload.get("data") or {})
        .get("trafficData", {})
        .get("volume", {})
        .get("byDay", {})
        .get("edges")
    ) or []

    rows = []
    for edge in node_list:
        node = (edge or {}).get("node") or {}
        raw_from = node.get("from")
        if not raw_from:
            continue
        try:
            day = datetime.fromisoformat(raw_from).date()
        except (TypeError, ValueError):
            continue
        total = node.get("total") or {}
        volume_numbers = total.get("volumeNumbers") or {}
        volume = volume_numbers.get("volume")
        coverage = (total.get("coverage") or {}).get("percentage")
        rows.append(
            {
                "date": day,
                "volume": float(volume) if volume is not None else np.nan,
                "coverage_pct": float(coverage) if coverage is not None else np.nan,
            }
        )

    if not rows:
        return pd.DataFrame(columns=cols)

    out = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    weekdays = pd.to_datetime(out["date"])
    out["weekday"] = weekdays.dt.weekday
    out["weekday_name"] = out["weekday"].map(lambda i: WEEKDAY_NAMES[int(i)])
    out["is_weekend"] = out["weekday"] >= 5
    out["month"] = weekdays.dt.month
    out["season"] = out["month"].map(SEASON_BY_MONTH)
    # Manglende dekning regnes som upålitelig: et døgn uten kjent dekning kan
    # like godt være en halv dag med telling.
    out["reliable"] = out["coverage_pct"].notna() & (out["coverage_pct"] >= float(min_coverage_pct))
    return out


def weekday_profile(daily: pd.DataFrame, reliable_only: bool = True) -> pd.DataFrame:
    """Snitt per ukedag, i kalenderrekkefølge fra mandag."""
    if daily is None or daily.empty:
        return pd.DataFrame(columns=["weekday", "weekday_name", "mean_volume", "days"])
    src = daily[daily["reliable"]] if reliable_only else daily
    src = src.dropna(subset=["volume"])
    if src.empty:
        return pd.DataFrame(columns=["weekday", "weekday_name", "mean_volume", "days"])
    grouped = (
        src.groupby("weekday")
        .agg(mean_volume=("volume", "mean"), days=("volume", "size"))
        .reset_index()
        .sort_values("weekday")
    )
    grouped["weekday_name"] = grouped["weekday"].map(lambda i: WEEKDAY_NAMES[int(i)])
    return grouped[["weekday", "weekday_name", "mean_volume", "days"]].reset_index(drop=True)


def monthly_profile(daily: pd.DataFrame, reliable_only: bool = True) -> pd.DataFrame:
    """Snitt per måned, med antall døgn bak hvert snitt."""
    if daily is None or daily.empty:
        return pd.DataFrame(columns=["month", "mean_volume", "days"])
    src = daily[daily["reliable"]] if reliable_only else daily
    src = src.dropna(subset=["volume"])
    if src.empty:
        return pd.DataFrame(columns=["month", "mean_volume", "days"])
    return (
        src.groupby("month")
        .agg(mean_volume=("volume", "mean"), days=("volume", "size"))
        .reset_index()
        .sort_values("month")
        .reset_index(drop=True)
    )


def weekend_vs_weekday(daily: pd.DataFrame, reliable_only: bool = True) -> Dict[str, float]:
    """Snitt for hverdag og helg, og helgens andel av hverdagsnivået.

    `weekend_share_pct` er None når hverdagsnivået mangler eller er null, slik
    at kallstedet ikke får en villedende 0 %.
    """
    empty = {"weekday_mean": np.nan, "weekend_mean": np.nan, "weekend_share_pct": None}
    if daily is None or daily.empty:
        return empty
    src = daily[daily["reliable"]] if reliable_only else daily
    src = src.dropna(subset=["volume"])
    if src.empty:
        return empty
    weekday_mean = src[~src["is_weekend"]]["volume"].mean()
    weekend_mean = src[src["is_weekend"]]["volume"].mean()
    share = None
    if pd.notna(weekday_mean) and weekday_mean and pd.notna(weekend_mean):
        share = float(weekend_mean) / float(weekday_mean) * 100.0
    return {
        "weekday_mean": float(weekday_mean) if pd.notna(weekday_mean) else np.nan,
        "weekend_mean": float(weekend_mean) if pd.notna(weekend_mean) else np.nan,
        "weekend_share_pct": share,
    }


def coverage_summary(
    daily: pd.DataFrame, days_expected: Optional[int] = None
) -> Dict[str, object]:
    """Nøkkeltall om datagrunnlaget, til bruk i et dekningsbanner.

    `days_expected` er antall døgn perioden *skulle* hatt. Uten det måles
    dekningen mot radene som finnes, og for et aggregat over flere punkter er
    det villedende: døgn der ingen punkter har tall finnes ikke som rad, og
    `volume` er aldri NaN etter en groupby-sum. Banneret ville da meldt «alle
    døgn har god dekning» om de døgnene som overlevde, mens resten av året var
    borte. Kallstedet oppgir derfor lengden på perioden.
    """
    empty = {
        "days_total": 0, "days_reliable": 0, "days_missing": 0,
        "days_absent": int(days_expected or 0), "mean_coverage_pct": np.nan,
    }
    if daily is None or daily.empty:
        return empty
    rows = int(len(daily))
    days_total = int(days_expected) if days_expected else rows
    days_reliable = int(daily["reliable"].sum())
    # Døgn uten tall: både rader med NaN og døgn som mangler helt.
    days_absent = max(0, days_total - rows)
    days_missing = int(daily["volume"].isna().sum()) + days_absent
    mean_cov = daily["coverage_pct"].mean()
    return {
        "days_total": days_total,
        "days_reliable": days_reliable,
        "days_missing": days_missing,
        "days_absent": days_absent,
        "mean_coverage_pct": float(mean_cov) if pd.notna(mean_cov) else np.nan,
    }


def sum_points_daily(frames_by_point: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Summer flere punkter til én døgnserie, med antall punkter bak hvert døgn.

    Bare pålitelige døgn summeres. Et døgn regnes som pålitelig for summen
    først når *alle* valgte punkter har tall: ellers ville summen falle når et
    punkt mangler data, og det ville lese som nedgang i sykling framfor et
    datahull. `points_present`/`points_expected` gjør grunnlaget synlig.
    """
    cols = [
        "date", "volume", "coverage_pct", "weekday", "weekday_name",
        "is_weekend", "month", "season", "reliable",
        "points_present", "points_expected",
    ]
    usable = {pid: f for pid, f in frames_by_point.items() if f is not None and not f.empty}
    if not usable:
        return pd.DataFrame(columns=cols)

    expected = len(usable)
    parts = []
    for pid, frame in usable.items():
        part = frame[["date", "volume", "coverage_pct", "reliable"]].copy()
        part["point_id"] = pid
        parts.append(part)
    long_df = pd.concat(parts, ignore_index=True)

    # Bare pålitelige døgn bidrar til summen; resten teller som fraværende.
    good = long_df[long_df["reliable"] & long_df["volume"].notna()]
    if good.empty:
        return pd.DataFrame(columns=cols)

    grouped = (
        good.groupby("date")
        .agg(
            volume=("volume", "sum"),
            coverage_pct=("coverage_pct", "mean"),
            points_present=("point_id", "nunique"),
        )
        .reset_index()
        .sort_values("date")
        .reset_index(drop=True)
    )
    grouped["points_expected"] = expected

    dates = pd.to_datetime(grouped["date"])
    grouped["weekday"] = dates.dt.weekday
    grouped["weekday_name"] = grouped["weekday"].map(lambda i: WEEKDAY_NAMES[int(i)])
    grouped["is_weekend"] = grouped["weekday"] >= 5
    grouped["month"] = dates.dt.month
    grouped["season"] = grouped["month"].map(SEASON_BY_MONTH)
    grouped["reliable"] = grouped["points_present"] >= expected
    return grouped[cols]


def mean_points_daily(
    frames_by_point: Dict[str, pd.DataFrame],
    min_points_share: float = 0.8,
) -> pd.DataFrame:
    """Snitt per punkt per døgn, som tåler at et punkt mangler.

    Snittet er robust mot at summen *faller* ved datahull, men ikke mot at
    nivået skifter når utvalget endres: punktene har svært ulikt volum, så et
    døgn målt på de tre travleste punktene ligger høyere enn et døgn målt på
    alle. I ekte data ga dette 63 % nivåforskjell mellom døgn med ett og to
    punkter — en «endring» som bare gjenspeilte hvilket punkt som var med.

    Derfor kreves `min_points_share` av punktene til stede før et døgn regnes
    som pålitelig. Terskelen er lavere enn summens krav om alle punkter: her
    er poenget å utelukke nivåskift, ikke å unngå at summen faller.
    """
    summed = sum_points_daily(frames_by_point)
    if summed.empty:
        return summed
    out = summed.copy()
    out["volume"] = out["volume"] / out["points_present"]
    required = float(out["points_expected"].iloc[0]) * float(min_points_share)
    out["reliable"] = out["points_present"] >= max(1.0, required)
    return out


def panel_point_ids(
    frames_by_year: Dict[int, Dict[str, pd.DataFrame]],
    min_reliable_share: float = 0.9,
) -> List[str]:
    """Punkter med god nok dekning i *alle* årene — grunnlaget for en sum.

    Nødvendig fordi `sum_points_daily` krever at alle punkter har tall for at
    et døgn skal regnes som pålitelig. Med mange punkter valgt finnes det da
    knapt et slikt døgn: ett punkt uten data hele året nuller ut hele summen.
    Å slippe kravet er ikke et alternativ — da faller summen ved hvert datahull
    og leses som nedgang i sykling.

    Løsningen er et balansert panel: bare punkter som er pålitelige minst
    `min_reliable_share` av døgnene i hvert år blir med. Da er «alle til stede»
    oppnåelig, og summen måler de samme punktene i alle årene, slik at et
    nivåskift ikke kan komme av at utvalget endret seg.
    """
    if not frames_by_year:
        return []

    per_year_ok = []
    for frames in frames_by_year.values():
        usable = {pid: f for pid, f in (frames or {}).items() if f is not None and not f.empty}
        if not usable:
            continue
        # Antall døgn i året måles på det punktet som dekker perioden bredest,
        # slik at terskelen ikke blir lettere for punkter med kort serie.
        days_in_year = max(len(f) for f in usable.values())
        if not days_in_year:
            continue
        ok = {
            pid
            for pid, f in usable.items()
            if int(f["reliable"].sum()) >= days_in_year * float(min_reliable_share)
        }
        per_year_ok.append(ok)

    if not per_year_ok:
        return []
    return sorted(set.intersection(*per_year_ok))


def common_period_cutoff(frames: Dict[int, pd.DataFrame]) -> Optional[tuple]:
    """Siste (måned, dag) som *alle* årene har data til og med.

    Inneværende år er avkortet mot dagens dato. Uten en felles slutt ville et
    delår blitt målt mot et helår, og sammenligningen ville sagt mer om hvor
    langt året er kommet enn om sykkeltrafikken.
    """
    cutoffs = []
    for daily in frames.values():
        if daily is None or daily.empty:
            continue
        dates = pd.to_datetime(daily["date"])
        # Maks framfor siste rad: uavhengig av sortering i input.
        cutoffs.append(max(zip(dates.dt.month, dates.dt.day)))
    return min(cutoffs) if cutoffs else None


def restrict_to_common_period(frames: Dict[int, pd.DataFrame]) -> Dict[int, pd.DataFrame]:
    """Kutt alle årene til samme del av kalenderåret.

    Sammenligningen skjer på (måned, dag) framfor dagnummer, slik at datoene
    stemmer overens på tvers av skuddår. 29. februar blir dermed med i skuddår
    og gir det året én dag mer, men vi sammenligner snitt per døgn, så det
    påvirker ikke nivåene.
    """
    cutoff = common_period_cutoff(frames)
    if cutoff is None:
        return {}
    out = {}
    for year, daily in frames.items():
        if daily is None or daily.empty:
            continue
        dates = pd.to_datetime(daily["date"])
        mask = [(m, d) <= cutoff for m, d in zip(dates.dt.month, dates.dt.day)]
        out[year] = daily[pd.Series(mask, index=daily.index)].copy()
    return out


def restrict_to_common_period_from(
    frames: Dict[int, pd.DataFrame], start_month_day: tuple[int, int]
) -> Dict[int, pd.DataFrame]:
    """Sammenlign samme kalenderperiode fra en gitt måned og dag.

    Brukes for tiltak som åpner midt i året: referanseåret skal ikke få med
    døgn før åpningen, og senere år må starte på samme kalenderdato for at
    sesong og antall døgn skal være sammenlignbare.
    """
    common = restrict_to_common_period(frames)
    out = {}
    for year, daily in common.items():
        dates = pd.to_datetime(daily["date"])
        mask = [(m, d) >= start_month_day for m, d in zip(dates.dt.month, dates.dt.day)]
        filtered = daily[pd.Series(mask, index=daily.index)].copy()
        if not filtered.empty:
            out[year] = filtered
    return out


def comparable_months(frames: Dict[int, pd.DataFrame], reliable_only: bool = True) -> List[int]:
    """Måneder der *alle* årene har pålitelige døgn.

    Nødvendig fordi dekningen varierer kraftig mellom år: et år kan ha mistet
    hele vinteren, og et årssnitt ville da sammenlignet sommer mot helår.
    Sykkeltrafikken er 3–4 ganger høyere om sommeren enn om vinteren, så det
    gir en «endring» som bare gjenspeiler hvilke måneder som overlevde
    dekningsterskelen.
    """
    per_year = []
    for daily in frames.values():
        if daily is None or daily.empty:
            continue
        src = daily[daily["reliable"]] if reliable_only else daily
        src = src.dropna(subset=["volume"])
        if src.empty:
            continue
        per_year.append(set(int(m) for m in src["month"].unique()))
    if not per_year:
        return []
    return sorted(set.intersection(*per_year))


def restrict_to_comparable_months(
    frames: Dict[int, pd.DataFrame], reliable_only: bool = True
) -> Dict[int, pd.DataFrame]:
    """Behold bare måneder alle årene har pålitelige tall for."""
    months = comparable_months(frames, reliable_only=reliable_only)
    if not months:
        return {}
    return {
        year: daily[daily["month"].isin(months)].copy()
        for year, daily in frames.items()
        if daily is not None and not daily.empty
    }


def restrict_to_common_calendar_days(
    frames: Dict[int, pd.DataFrame], reliable_only: bool = True
) -> Dict[int, pd.DataFrame]:
    """Behold bare kalenderdøgn med tall i alle år.

    Reelle summer kan bare sammenlignes når de bygger på nøyaktig de samme
    måned/dag-kombinasjonene. Ellers vil et manglende døgn se ut som lavere
    trafikk. Med `reliable_only` beholdes bare døgn med god dekning og volum.
    """
    usable = {}
    day_sets = []
    for year, daily in sorted(frames.items()):
        if daily is None or daily.empty:
            continue
        src = daily
        if reliable_only:
            src = src[src["reliable"] & src["volume"].notna()]
        if src.empty:
            continue
        dates = pd.to_datetime(src["date"])
        calendar_days = set(zip(dates.dt.month, dates.dt.day))
        usable[year] = src
        day_sets.append(calendar_days)
    if not day_sets:
        return {}

    common_days = set.intersection(*day_sets)
    out = {}
    for year, daily in usable.items():
        dates = pd.to_datetime(daily["date"])
        mask = [(m, d) in common_days for m, d in zip(dates.dt.month, dates.dt.day)]
        filtered = daily[pd.Series(mask, index=daily.index)].copy()
        if not filtered.empty:
            out[year] = filtered
    return out


def compare_years_monthly(frames: Dict[int, pd.DataFrame], reliable_only: bool = True) -> pd.DataFrame:
    """Månedssnitt per år i lang form, til en gruppert søylegraf.

    Kolonner: year, month, mean_volume, days.
    """
    rows = []
    for year, daily in sorted(frames.items()):
        profile = monthly_profile(daily, reliable_only=reliable_only)
        if profile.empty:
            continue
        profile = profile.copy()
        profile["year"] = int(year)
        rows.append(profile)
    if not rows:
        return pd.DataFrame(columns=["year", "month", "mean_volume", "days"])
    return pd.concat(rows, ignore_index=True)[["year", "month", "mean_volume", "days"]]


def year_comparison_summary(
    frames: Dict[int, pd.DataFrame], reliable_only: bool = True
) -> pd.DataFrame:
    """Nøkkeltall per år, med endring mot det eldste året i utvalget.

    Kolonner: year, mean_volume, total_volume, days, change_pct. `change_pct`
    er NaN for referanseåret og for år der referansen mangler eller er null —
    kallstedet må bruke pd.isna(), ikke `is None`.
    """
    cols = ["year", "mean_volume", "total_volume", "days", "change_pct"]
    rows = []
    for year, daily in sorted(frames.items()):
        if daily is None or daily.empty:
            continue
        src = daily[daily["reliable"]] if reliable_only else daily
        src = src.dropna(subset=["volume"])
        if src.empty:
            continue
        rows.append(
            {
                "year": int(year),
                "mean_volume": float(src["volume"].mean()),
                "total_volume": float(src["volume"].sum()),
                "days": int(len(src)),
            }
        )
    if not rows:
        return pd.DataFrame(columns=cols)

    out = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)
    baseline = out.iloc[0]["mean_volume"]
    if baseline:
        out["change_pct"] = (out["mean_volume"] - baseline) / baseline * 100.0
        out.loc[0, "change_pct"] = None
    else:
        out["change_pct"] = None
    return out[cols]


def year_to_date_range(year: int, today: Optional[date] = None) -> tuple:
    """Fra 1. januar til og med i går for inneværende år, ellers hele året.

    I dag utelates fordi døgnet ikke er ferdig telt og ville framstått som et
    kraftig fall på siste punkt i grafen.
    """
    today = today or date.today()
    start = date(year, 1, 1)
    if year < today.year:
        end = date(year, 12, 31)
    elif year > today.year:
        return None
    else:
        end = today - timedelta(days=1)
        if end < start:
            return None
    return start, end


def days_expected_in_year(year: int, today: Optional[date] = None) -> int:
    """Antall døgn året skulle hatt data for, avkortet mot dagens dato.

    Brukes som nevner i dekningsbanneret, slik at døgn som mangler helt —
    og derfor ikke finnes som rad i et aggregat — også blir talt.
    """
    span = year_to_date_range(year, today=today)
    if span is None:
        return 0
    start, end = span
    return (end - start).days + 1
