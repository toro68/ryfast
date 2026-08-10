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


def retired_point_ids() -> List[str]:
    """ID-ene til punkter som er ute av drift."""
    return [pid for pid, m in BICYCLE_POINTS.items() if not m.get("operational", True)]


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


def coverage_summary(daily: pd.DataFrame) -> Dict[str, object]:
    """Nøkkeltall om datagrunnlaget, til bruk i et dekningsbanner."""
    if daily is None or daily.empty:
        return {"days_total": 0, "days_reliable": 0, "days_missing": 0, "mean_coverage_pct": np.nan}
    days_total = int(len(daily))
    days_reliable = int(daily["reliable"].sum())
    days_missing = int(daily["volume"].isna().sum())
    mean_cov = daily["coverage_pct"].mean()
    return {
        "days_total": days_total,
        "days_reliable": days_reliable,
        "days_missing": days_missing,
        "mean_coverage_pct": float(mean_cov) if pd.notna(mean_cov) else np.nan,
    }


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
