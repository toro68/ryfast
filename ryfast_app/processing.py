"""Radnivå-databehandling for trafikkdata (streamlit-fri)."""

import calendar
from datetime import date
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ryfast_app.config import ANOMALY_THRESHOLD_PCT, MONTH_NAMES, POINT_ID_LABELS


def format_number(x):
    """Norsk tallformat: mellomrom som tusenskiller, komma som desimaltegn."""
    if pd.isna(x):
        return "N/A"
    if isinstance(x, (int, float, np.integer, np.floating)):
        if float(x).is_integer():
            return f"{int(x):,}".replace(",", " ")
        return f"{x:,.1f}".replace(",", " ").replace(".", ",")
    if isinstance(x, str):
        try:
            return format_number(float(x))
        except ValueError:
            return x
    return str(x)

def days_in_year(year: int) -> int:
    return 366 if calendar.isleap(year) else 365

def assessable_months(year: int, today: Optional[date] = None) -> List[int]:
    """Måneder som er ferdige, og dermed kan vurderes for datadekning.

    Inneværende måned utelates: den er per definisjon ufullstendig og ville
    ellers flagges som et datahull. Fremtidige år gir ingen måneder.
    """
    today = today or date.today()
    if year < today.year:
        return list(range(1, 13))
    if year > today.year:
        return []
    return list(range(1, today.month))

def overlapping_months(df: pd.DataFrame, year_cols: List[str]) -> List[int]:
    """Månedsnumre der samtlige oppgitte årskolonner har en tallverdi.

    Brukes for å sammenligne like perioder: et delår skal ikke måles mot et helår.
    """
    if df is None or df.empty or "Month" not in df.columns:
        return []
    months = pd.to_numeric(df["Month"], errors="coerce")
    mask = months.between(1, 12)
    for col in year_cols:
        if col not in df.columns:
            return []
        mask &= pd.to_numeric(df[col], errors="coerce").notna()
    return [int(m) for m in months[mask].tolist()]

def sum_traffic_data(
    traffic_data_dict: Dict[str, List[Dict]],
    expected_point_ids: Optional[List[str]] = None,
    estimate_missing_points: bool = False,
) -> Tuple[List[float], List[Dict[str, float]], List[bool], List[int]]:
    monthly_sums = [0.0] * 12
    monthly_confidence = [{"lower": 0.0, "upper": 0.0} for _ in range(12)]
    monthly_has_data = [False] * 12
    monthly_points_present = [0] * 12
    expected_points = len(expected_point_ids or [])

    for point_data in (traffic_data_dict or {}).values():
        for entry in point_data or []:
            month = int(entry.get("month") or 0)
            if not (1 <= month <= 12):
                continue
            volume = entry.get("total", {}).get("volume", {}).get("average")
            if volume is None:
                continue
            monthly_has_data[month - 1] = True
            monthly_sums[month - 1] += float(volume)
            monthly_points_present[month - 1] += 1
            ci = entry.get("total", {}).get("volume", {}).get("confidenceInterval") or {}
            lb = ci.get("lowerBound")
            ub = ci.get("upperBound")
            if lb is not None and ub is not None:
                monthly_confidence[month - 1]["lower"] += float(lb)
                monthly_confidence[month - 1]["upper"] += float(ub)

    if estimate_missing_points and expected_points:
        for i in range(12):
            present = monthly_points_present[i]
            if present and present < expected_points:
                monthly_sums[i] = monthly_sums[i] * (expected_points / present)
                monthly_confidence[i]["lower"] = monthly_confidence[i]["lower"] * (expected_points / present)
                monthly_confidence[i]["upper"] = monthly_confidence[i]["upper"] * (expected_points / present)

    return monthly_sums, monthly_confidence, monthly_has_data, monthly_points_present

def detect_monthly_anomalies(df: pd.DataFrame, threshold_pct: float = ANOMALY_THRESHOLD_PCT) -> pd.DataFrame:
    if df is None or df.empty or "Month" not in df.columns:
        return pd.DataFrame()
    years = [c for c in df.columns if str(c).isdigit()]
    if len(years) < 2:
        return pd.DataFrame()

    # Med kun to år er median-av-øvrige symmetrisk: hver måned ville flagges to
    # ganger med motsatt fortegn. Da vurderer vi bare det seneste året mot det
    # eldre. Med tre eller flere år er medianen et reelt forventningsnivå.
    years_to_check = [max(years, key=int)] if len(years) == 2 else years

    rows: List[Dict[str, object]] = []
    for year_col in years_to_check:
        other_cols = [c for c in years if c != year_col]
        for _, r in df.iterrows():
            month = int(r.get("Month", 0) or 0)
            if not (1 <= month <= 12):
                continue
            actual = pd.to_numeric(r.get(year_col), errors="coerce")
            if pd.isna(actual):
                continue
            expected = pd.to_numeric(pd.Series([r.get(c) for c in other_cols]), errors="coerce").median()
            if pd.isna(expected) or expected == 0:
                continue
            deviation = (float(actual) - float(expected)) / float(expected) * 100.0
            if abs(deviation) >= float(threshold_pct):
                rows.append(
                    {
                        "year": int(year_col),
                        "month": month,
                        "month_name": MONTH_NAMES[month - 1],
                        "actual": float(actual),
                        "expected": float(expected),
                        "deviation_pct": float(deviation),
                    }
                )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["year", "month"]).reset_index(drop=True)

def sum_weekly_traffic_data(weekly_data_dict: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    return {week: float(sum(vals.values())) for week, vals in weekly_data_dict.items()}

def add_month_names(df: pd.DataFrame) -> pd.DataFrame:
    if "Month" in df.columns:
        df["Month Name"] = [MONTH_NAMES[i - 1] for i in df["Month"]]
        return df[["Month", "Month Name"] + [c for c in df.columns if c not in ["Month", "Month Name"]]]
    return df

def calculate_yearly_total_from_monthly_averages(df: pd.DataFrame, year: int) -> Tuple[float, int, int]:
    year_col = str(year)
    if df is None or df.empty or "Month" not in df.columns or year_col not in df.columns:
        return 0.0, 0, 0

    months = pd.to_numeric(df["Month"], errors="coerce")
    avg_vals = pd.to_numeric(df[year_col], errors="coerce")
    valid = avg_vals.notna() & months.between(1, 12)
    if not valid.any():
        return 0.0, 0, 0

    valid_months = months[valid].astype(int)
    dims = valid_months.apply(lambda m: calendar.monthrange(year, m)[1])
    total = float((avg_vals[valid] * dims).sum())
    return total, int(valid.sum()), int(dims.sum())

def extract_point_monthly_metrics(traffic_by_point: Dict[str, List[Dict]], year: int) -> pd.DataFrame:
    """
    Returnerer per-punkt/per-måned:
      - avg_daily (ÅDT)
      - coverage_pct
      - ci_lower/ci_upper (ÅDT), hvis tilgjengelig
    """
    rows: List[Dict[str, object]] = []
    for point_id, entries in (traffic_by_point or {}).items():
        for entry in entries or []:
            month = int(entry.get("month", 0) or 0)
            if not (1 <= month <= 12):
                continue
            total = entry.get("total", {}) or {}
            volume = (total.get("volume", {}) or {}).get("average")
            cov = (total.get("coverage", {}) or {}).get("percentage")
            ci = (total.get("volume", {}) or {}).get("confidenceInterval") or {}
            rows.append(
                {
                    "point_id": point_id,
                    "point_label": POINT_ID_LABELS.get(point_id, point_id),
                    "year": year,
                    "month": month,
                    "month_name": MONTH_NAMES[month - 1],
                    "avg_daily": float(volume) if volume is not None else np.nan,
                    "coverage_pct": float(cov) if cov is not None else np.nan,
                    "ci_lower": float(ci["lowerBound"]) if ci.get("lowerBound") is not None else np.nan,
                    "ci_upper": float(ci["upperBound"]) if ci.get("upperBound") is not None else np.nan,
                }
            )
    return pd.DataFrame(rows)
