"""Dekningssammendrag, totaler, vekst og sesongmønstre (streamlit-fri)."""

import calendar
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ryfast_app.config import MONTH_NAMES, POINT_ID_LABELS
from ryfast_app.processing import add_month_names, assessable_months, extract_point_monthly_metrics


def compute_monthly_coverage_summary(
    traffic_by_point: Dict[str, List[Dict]],
    year: int,
    expected_point_ids: List[str],
) -> pd.DataFrame:
    """Dekning per måned for et år.

    Kolonnen `is_assessable` er False for måneder som ennå ikke er ferdige
    (inneværende og fremtidige måneder). De har ingen data å mangle, og skal
    derfor ikke telles som datahull av bannere eller varsler.
    """
    metrics = extract_point_monthly_metrics(traffic_by_point, year)
    expected_points = len(expected_point_ids)
    assessable = set(assessable_months(year))

    if metrics.empty:
        return pd.DataFrame(
            {
                "year": [year] * 12,
                "month": list(range(1, 13)),
                "month_name": MONTH_NAMES,
                "points_expected": [expected_points] * 12,
                "points_present": [0] * 12,
                "points_present_pct": [0.0] * 12,
                "mean_coverage_pct": [np.nan] * 12,
                "min_coverage_pct": [np.nan] * 12,
                "is_assessable": [m in assessable for m in range(1, 13)],
            }
        )

    tmp = metrics[metrics["avg_daily"].notna()].copy()
    grouped = (
        tmp.groupby(["year", "month", "month_name"], as_index=False)
        .agg(
            points_present=("point_id", "nunique"),
            mean_coverage_pct=("coverage_pct", "mean"),
            min_coverage_pct=("coverage_pct", "min"),
        )
        .sort_values("month")
    )
    base = pd.DataFrame({"year": [year] * 12, "month": list(range(1, 13)), "month_name": MONTH_NAMES})
    out = base.merge(grouped, on=["year", "month", "month_name"], how="left")
    out["points_expected"] = expected_points
    out["points_present"] = out["points_present"].fillna(0).astype(int)
    out["points_present_pct"] = np.where(
        expected_points > 0,
        out["points_present"].astype(float) / float(expected_points) * 100.0,
        np.nan,
    )
    out["is_assessable"] = out["month"].isin(assessable)
    return out

def compute_weekly_coverage_summary(
    weekly_data_by_point: Dict[str, Dict[str, float]],
    weekly_cov_by_point: Dict[str, Dict[str, float]],
    expected_point_ids: List[str],
    year: int,
) -> pd.DataFrame:
    expected_points = len(expected_point_ids)
    weeks = sorted(
        {str(w) for w in (weekly_data_by_point or {}).keys()},
        key=lambda s: int("".join([c for c in s if c.isdigit()]) or "0"),
    )
    rows: List[Dict[str, object]] = []
    for week in weeks:
        vols = (weekly_data_by_point or {}).get(week, {}) or {}
        covs = (weekly_cov_by_point or {}).get(week, {}) or {}
        points_present = len(vols)
        cov_values = [float(v) for v in covs.values() if v is not None and not pd.isna(v)]
        rows.append(
            {
                "year": year,
                "week": week,
                "points_expected": expected_points,
                "points_present": points_present,
                "points_present_pct": (points_present / expected_points * 100.0) if expected_points else np.nan,
                "mean_coverage_pct": (sum(cov_values) / len(cov_values)) if cov_values else np.nan,
                "min_coverage_pct": min(cov_values) if cov_values else np.nan,
                # Radene finnes bare for uker API-et returnerte data for, så alle
                # kan vurderes. Kolonnen holdes for felles bannerlogikk.
                "is_assessable": True,
            }
        )
    return pd.DataFrame(rows)

def coverage_pivot(metrics_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_df is None or metrics_df.empty:
        return pd.DataFrame()
    pivot = metrics_df.pivot_table(
        index="month_name",
        columns="point_label",
        values="coverage_pct",
        aggfunc="mean",
    ).reindex(MONTH_NAMES)
    return pivot

def group_coverage_by_month(metrics_df: pd.DataFrame, point_ids_by_group: Dict[str, List[str]]) -> pd.DataFrame:
    if metrics_df is None or metrics_df.empty:
        return pd.DataFrame()
    group_rows: List[pd.DataFrame] = []
    for group, ids in point_ids_by_group.items():
        labels = {POINT_ID_LABELS.get(pid, pid) for pid in ids}
        sub = metrics_df[metrics_df["point_label"].isin(labels)].copy()
        if sub.empty:
            continue
        grouped = (
            sub.groupby("month_name", as_index=False)
            .agg(coverage_pct=("coverage_pct", "mean"))
            .assign(group=group)
        )
        group_rows.append(grouped)
    if not group_rows:
        return pd.DataFrame()
    out = pd.concat(group_rows, ignore_index=True)
    out["month_name"] = pd.Categorical(out["month_name"], categories=MONTH_NAMES, ordered=True)
    out = out.sort_values(["month_name", "group"])
    return out

def totals_with_uncertainty_from_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Lager per måned totals (passeringer) og et "min/max"-intervall basert på CI.
    Merk: Summert CI på tvers av punkter er en forenkling (indikativ usikkerhet).
    """
    if metrics_df is None or metrics_df.empty:
        return pd.DataFrame()
    tmp = metrics_df.copy()
    tmp["days_in_month"] = tmp.apply(
        lambda row: calendar.monthrange(int(row["year"]), int(row["month"]))[1], axis=1
    )
    tmp["total"] = tmp["avg_daily"] * tmp["days_in_month"]
    tmp["total_lower"] = tmp["ci_lower"] * tmp["days_in_month"]
    tmp["total_upper"] = tmp["ci_upper"] * tmp["days_in_month"]
    out = (
        tmp.groupby(["month", "month_name"], as_index=False)
        .agg(
            total=("total", "sum"),
            total_lower=("total_lower", "sum"),
            total_upper=("total_upper", "sum"),
            coverage_pct=("coverage_pct", "mean"),
        )
        .sort_values("month")
    )
    return out

def monthly_totals_from_monthly_averages(df: pd.DataFrame, year: int) -> pd.Series:
    year_col = str(year)
    if df is None or df.empty or "Month" not in df.columns or year_col not in df.columns:
        return pd.Series([np.nan] * (len(df) if df is not None else 0))

    months = pd.to_numeric(df["Month"], errors="coerce")
    avg_vals = pd.to_numeric(df[year_col], errors="coerce")
    valid = avg_vals.notna() & months.between(1, 12)
    dims = months.where(valid).apply(
        lambda m: calendar.monthrange(year, int(m))[1] if pd.notna(m) else np.nan
    )
    return (avg_vals * dims).where(valid)

def compute_monthly_totals_table(df: pd.DataFrame, years: List[int]) -> pd.DataFrame:
    base_cols = [c for c in ["Month", "Month Name"] if c in df.columns]
    out = df[base_cols].copy()
    for y in years:
        out[str(y)] = monthly_totals_from_monthly_averages(df, y)
    numeric_columns = out.select_dtypes(include=[np.number]).columns
    out[numeric_columns] = out[numeric_columns].round(0).astype("Int64")
    return out

def aggregate_monthly_totals_by_group(
    traffic_data_by_point: Dict[str, List[Dict]],
    point_ids_by_group: Dict[str, List[str]],
    year: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    point_to_group = {pid: group for group, ids in point_ids_by_group.items() for pid in ids}
    groups = list(point_ids_by_group.keys())
    totals = {g: {m: 0.0 for m in range(1, 13)} for g in groups}
    cov_lists: Dict[str, Dict[int, List[float]]] = {g: {m: [] for m in range(1, 13)} for g in groups}

    for pid, entries in traffic_data_by_point.items():
        group = point_to_group.get(pid)
        if not group:
            continue
        for entry in entries or []:
            month = int(entry.get("month", 0) or 0)
            if not (1 <= month <= 12):
                continue
            avg_daily = entry.get("total", {}).get("volume", {}).get("average")
            if avg_daily is None:
                continue
            dim = calendar.monthrange(year, month)[1]
            totals[group][month] += float(avg_daily) * dim
            cov = entry.get("total", {}).get("coverage", {}).get("percentage")
            if cov is not None:
                cov_lists[group][month].append(float(cov))

    totals_df = pd.DataFrame({"Month": list(range(1, 13))})
    for g in groups:
        totals_df[g] = [totals[g][m] for m in range(1, 13)]
    totals_df = add_month_names(totals_df)
    totals_df[groups] = totals_df[groups].round(0).astype("Int64")

    coverage_df = pd.DataFrame({"Month": list(range(1, 13))})
    for g in groups:
        coverage_df[g] = [
            (sum(cov_lists[g][m]) / len(cov_lists[g][m])) if cov_lists[g][m] else np.nan for m in range(1, 13)
        ]
    coverage_df = add_month_names(coverage_df)
    return totals_df, coverage_df

def calculate_growth_rates(df: pd.DataFrame) -> pd.DataFrame:
    growth_df = df.copy()
    year_columns = [col for col in df.columns if col not in ["Month", "Month Name", "Week", "Volume"]]
    if len(year_columns) >= 2:
        for i in range(1, len(year_columns)):
            prev_year = year_columns[i - 1]
            curr_year = year_columns[i]
            growth_col = f"Vekst {prev_year}-{curr_year} (%)"
            curr = pd.to_numeric(df[curr_year], errors="coerce")
            prev_vals = pd.to_numeric(df[prev_year], errors="coerce")
            growth_df[growth_col] = ((curr - prev_vals) / prev_vals * 100).round(1)
    return growth_df

def calculate_seasonal_patterns(df: pd.DataFrame) -> Dict:
    if "Month" not in df.columns:
        return {}
    patterns = {}
    year_columns = [col for col in df.columns if col not in ["Month", "Month Name"]]
    for year in year_columns:
        if year in df.columns:
            yearly = pd.to_numeric(df[year], errors="coerce").to_numpy()
            if len(yearly) == 12 and not np.all(np.isnan(yearly)):
                patterns[year] = {
                    "vinter_snitt": np.nanmean([yearly[11], yearly[0], yearly[1]]),
                    "vår_snitt": np.nanmean(yearly[2:5]),
                    "sommer_snitt": np.nanmean(yearly[5:8]),
                    "høst_snitt": np.nanmean(yearly[8:11]),
                }
    return patterns
