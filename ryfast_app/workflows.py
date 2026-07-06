"""Orkestrering av datahenting og -prosessering per sammenligningsmodus."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from ryfast_app.api import fetch_batch_traffic_data, fetch_weekly_traffic_data
from ryfast_app.metrics import compute_monthly_coverage_summary, compute_weekly_coverage_summary
from ryfast_app.processing import add_month_names, sum_traffic_data, sum_weekly_traffic_data


def process_data_for_years(
    point_ids: List[str], year_list: List[int], timeout_s: int, use_cache: bool, estimate_missing_points: bool
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data: Dict[int, List[float]] = {}
    coverage_rows: List[pd.DataFrame] = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, year in enumerate(year_list):
        status_text.text(f"Henter data for {year}...")
        progress_bar.progress((i + 1) / len(year_list))
        traffic_data_dict = fetch_batch_traffic_data(point_ids, year, timeout_s, use_cache)
        if traffic_data_dict:
            monthly_sums, _, monthly_has_data, _ = sum_traffic_data(
                traffic_data_dict,
                expected_point_ids=point_ids,
                estimate_missing_points=estimate_missing_points,
            )
            data[year] = [v if has else np.nan for v, has in zip(monthly_sums, monthly_has_data)]
            coverage_rows.append(compute_monthly_coverage_summary(traffic_data_dict, year, point_ids))

    status_text.empty()
    progress_bar.empty()

    df = pd.DataFrame({"Month": list(range(1, 13))})
    for y in year_list:
        if y in data:
            df[str(y)] = data[y]
    df = add_month_names(df)
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].round(0).astype("Int64")
    coverage_df = pd.concat(coverage_rows, ignore_index=True) if coverage_rows else pd.DataFrame()
    return df, coverage_df


def process_data_for_months(
    point_ids: List[str], year: int, months: List[int], timeout_s: int, use_cache: bool, estimate_missing_points: bool
) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    traffic_data_dict = fetch_batch_traffic_data(point_ids, year, timeout_s, use_cache)
    if not traffic_data_dict:
        return None
    data, _, monthly_has_data, _ = sum_traffic_data(
        traffic_data_dict,
        expected_point_ids=point_ids,
        estimate_missing_points=estimate_missing_points,
    )
    df = pd.DataFrame({"Month": list(range(1, 13)), str(year): data})
    df.loc[~pd.Series(monthly_has_data), str(year)] = np.nan
    df = df[df["Month"].isin(months)]
    df = add_month_names(df)
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].round(0).astype("Int64")
    coverage_df = compute_monthly_coverage_summary(traffic_data_dict, year, point_ids)
    coverage_df = coverage_df[coverage_df["month"].isin(months)].copy()
    return df, coverage_df


def process_data_for_weeks(
    point_ids: List[str], year: int, weeks: List[int], timeout_s: int, use_cache: bool
) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    weekly_data_dict, weekly_cov_dict = fetch_weekly_traffic_data(point_ids, year, weeks, timeout_s, use_cache)
    if not weekly_data_dict:
        return None
    weekly_sums = sum_weekly_traffic_data(weekly_data_dict)
    df = pd.DataFrame([{"Week": week, "Volume": volume} for week, volume in weekly_sums.items()])
    df["Week_Num"] = df["Week"].str.extract(r"(\d+)").astype(int)
    df = df.sort_values("Week_Num").drop(columns=["Week_Num"]).reset_index(drop=True)
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").round(0).astype("Int64")
    coverage_df = compute_weekly_coverage_summary(weekly_data_dict, weekly_cov_dict, point_ids, year)
    return df, coverage_df
