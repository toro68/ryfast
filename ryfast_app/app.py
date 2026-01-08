import calendar
import io
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

try:
    import openpyxl  # noqa: F401

    OPENPYXL_AVAILABLE = True
except Exception:
    OPENPYXL_AVAILABLE = False


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

URL = "https://trafikkdata-api.atlas.vegvesen.no"

TRAFFIC_POINTS = {
    "Ryfast (sum tunneler)": {
        "ids": [
            "99040V2725982",
            "00911V2725983",
            "10239V2725979",
            "62464V2725991",
            "92743V2726085",
            "25926V2725990",
        ],
        "description": "Sum av Ryfylketunnelen + Hundvågtunnelen (Ryfast totalt)",
        "opened": "2019-12-30 / 2020-04-22",
    },
    "Ryfylketunnelen": {
        "ids": ["99040V2725982", "00911V2725983"],
        "description": "Ryfylketunnelen - hovedforbindelse til Ryfylke",
        "opened": "2019-12-30",
    },
    "Hundvågtunnelen": {
        "ids": ["10239V2725979", "62464V2725991", "92743V2726085", "25926V2725990"],
        "description": "Hundvågtunnelen - forbindelse til Hundvåg og Eiganes",
        "opened": "2020-04-22",
    },
    "Bybrua": {
        "ids": {"Mot nord": ["17949V320695"], "Mot sør": ["54184V320694"]},
        "description": "Bybrua - historisk broforbindelse over Strømsteinsundet",
        "opened": "Historisk",
    },
}

HUNDVAG_TUNNEL_IDS_UTEN_PÅRAMPE = ["10239V2725979", "92743V2726085"]

MONTH_NAMES = [
    "Januar",
    "Februar",
    "Mars",
    "April",
    "Mai",
    "Juni",
    "Juli",
    "August",
    "September",
    "Oktober",
    "November",
    "Desember",
]

DEFAULT_YEARS = "2024,2025"
YEAR_RANGE = range(2019, 2027)
API_MAX_RETRIES = 3
API_RETRY_DELAY = 1
API_CACHE_TTL = 24 * 3600

QUERY_TEMPLATE = """
query {{
  trafficData(trafficRegistrationPointId: "{point_id}") {{
    volume {{
      average {{
        daily {{
          byMonth(year: {year}) {{
            month
            total {{
              volume {{
                average
                confidenceInterval {{
                  lowerBound
                  upperBound
                }}
              }}
              coverage {{
                percentage
              }}
            }}
          }}
        }}
      }}
    }}
  }}
}}
"""

WEEKLY_QUERY_TEMPLATE = """
query {{
  trafficData(trafficRegistrationPointId: "{point_id}") {{
    volume {{
      byDay(from: "{from_date}", to: "{to_date}") {{
        edges {{
          node {{
            from
            to
            total {{
              volumeNumbers {{
                volume
              }}
              coverage {{
                percentage
              }}
            }}
          }}
        }}
      }}
    }}
  }}
}}
"""


def format_number(x):
    if pd.isna(x):
        return "N/A"
    if isinstance(x, (int, float, np.integer, np.floating)):
        if float(x).is_integer():
            return f"{int(x):,}".replace(",", " ")
        return f"{x:,.1f}".replace(",", " ")
    if isinstance(x, str):
        try:
            return format_number(float(x))
        except ValueError:
            return x
    return str(x)


def days_in_year(year: int) -> int:
    return 366 if calendar.isleap(year) else 365


def init_session_state():
    if "comparison_history" not in st.session_state:
        st.session_state.comparison_history = []
    if "last_result" not in st.session_state:
        st.session_state.last_result = None


def _fetch_data_uncached(query: str, timeout_s: int) -> Optional[Dict]:
    for attempt in range(API_MAX_RETRIES):
        try:
            response = requests.post(URL, json={"query": query}, timeout=timeout_s)
            response.raise_for_status()
            data = response.json()
            if "errors" in data:
                logger.error("GraphQL errors: %s", data["errors"])
                return None
            return data
        except requests.Timeout:
            logger.warning("Timeout on attempt %s/%s", attempt + 1, API_MAX_RETRIES)
            if attempt == API_MAX_RETRIES - 1:
                return None
        except requests.RequestException as e:
            logger.warning("Request failed on attempt %s/%s: %s", attempt + 1, API_MAX_RETRIES, str(e))
            if attempt == API_MAX_RETRIES - 1:
                return None
            time.sleep(API_RETRY_DELAY * (attempt + 1))
    return None


@st.cache_data(ttl=API_CACHE_TTL, show_spinner=False)
def _fetch_data_cached(query: str, timeout_s: int) -> Optional[Dict]:
    return _fetch_data_uncached(query, timeout_s)


def fetch_data(query: str, timeout_s: int, use_cache: bool) -> Optional[Dict]:
    return _fetch_data_cached(query, timeout_s) if use_cache else _fetch_data_uncached(query, timeout_s)


def fetch_batch_traffic_data(point_ids: List[str], year: int, timeout_s: int, use_cache: bool) -> Dict[str, List[Dict]]:
    if year < 2019:
        return {}

    result: Dict[str, List[Dict]] = {}
    if use_cache:
        for point_id in point_ids:
            query = QUERY_TEMPLATE.format(point_id=point_id, year=year)
            data = fetch_data(query, timeout_s, use_cache=True)
            if data and data.get("data", {}).get("trafficData"):
                monthly = data["data"]["trafficData"]["volume"]["average"]["daily"]["byMonth"]
                if monthly:
                    result[point_id] = monthly
        return result

    with ThreadPoolExecutor(max_workers=3) as executor:
        future_to_point = {
            executor.submit(_fetch_data_uncached, QUERY_TEMPLATE.format(point_id=pid, year=year), timeout_s): pid
            for pid in point_ids
        }

        for future, point_id in future_to_point.items():
            try:
                data = future.result()
                if data and data.get("data", {}).get("trafficData"):
                    monthly = data["data"]["trafficData"]["volume"]["average"]["daily"]["byMonth"]
                    if monthly:
                        result[point_id] = monthly
            except Exception as e:
                logger.error("Feil ved henting av data for punkt %s, år %s: %s", point_id, year, str(e))

    return result


def fetch_weekly_traffic_data(
    point_ids: List[str], year: int, week_numbers: List[int], timeout_s: int, use_cache: bool
) -> Dict[str, Dict[str, float]]:
    if year < 2019:
        return {}

    result: Dict[str, Dict[str, float]] = {}
    for week_num in week_numbers:
        try:
            jan_1 = datetime(year, 1, 1)
            week_1_start = jan_1 - timedelta(days=jan_1.weekday())
            if week_1_start.year < year:
                week_1_start += timedelta(weeks=1)
            week_start = week_1_start + timedelta(weeks=week_num - 1)
            week_end = week_start + timedelta(days=6)
            if week_start.year != year or week_end.year != year:
                continue

            from_date = week_start.strftime("%Y-%m-%dT00:00:00+01:00")
            to_date = week_end.strftime("%Y-%m-%dT23:59:59+01:00")

            week_data: Dict[str, float] = {}
            for point_id in point_ids:
                query = WEEKLY_QUERY_TEMPLATE.format(point_id=point_id, from_date=from_date, to_date=to_date)
                data = fetch_data(query, timeout_s, use_cache)
                if data and data.get("data", {}).get("trafficData"):
                    edges = data["data"]["trafficData"]["volume"]["byDay"]["edges"] or []
                    total_volume = 0.0
                    valid_days = 0
                    for edge in edges:
                        volume = edge["node"]["total"]["volumeNumbers"]["volume"]
                        if volume is not None:
                            total_volume += float(volume)
                            valid_days += 1
                    if valid_days:
                        week_data[point_id] = total_volume / valid_days
            if week_data:
                result[f"Uke {week_num}"] = week_data
        except Exception as e:
            logger.error("Feil ved henting av ukesdata for uke %s: %s", week_num, str(e))
    return result


def sum_traffic_data(traffic_data_dict: Dict[str, List[Dict]]) -> Tuple[List[float], List[Dict[str, float]]]:
    monthly_sums = [0.0] * 12
    monthly_confidence = [{"lower": 0.0, "upper": 0.0} for _ in range(12)]

    for point_data in traffic_data_dict.values():
        for entry in point_data:
            month = int(entry.get("month") or 0)
            if not (1 <= month <= 12):
                continue
            volume = entry.get("total", {}).get("volume", {}).get("average")
            if volume is None:
                continue
            monthly_sums[month - 1] += float(volume)
            ci = entry.get("total", {}).get("volume", {}).get("confidenceInterval") or {}
            lb = ci.get("lowerBound")
            ub = ci.get("upperBound")
            if lb is not None and ub is not None:
                monthly_confidence[month - 1]["lower"] += float(lb)
                monthly_confidence[month - 1]["upper"] += float(ub)

    return monthly_sums, monthly_confidence


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

    total = 0.0
    months_present = 0
    days_covered = 0
    for _, row in df.iterrows():
        try:
            month = int(row["Month"])
        except Exception:
            continue
        avg_daily = row.get(year_col, None)
        if pd.isna(avg_daily) or avg_daily is None or not (1 <= month <= 12):
            continue
        dim = calendar.monthrange(year, month)[1]
        total += float(avg_daily) * dim
        months_present += 1
        days_covered += dim

    return total, months_present, days_covered


def monthly_totals_from_monthly_averages(df: pd.DataFrame, year: int) -> pd.Series:
    year_col = str(year)
    if df is None or df.empty or "Month" not in df.columns or year_col not in df.columns:
        return pd.Series([pd.NA] * (len(df) if df is not None else 0))

    totals: List[object] = []
    for _, row in df.iterrows():
        try:
            month = int(row["Month"])
        except Exception:
            totals.append(pd.NA)
            continue
        avg_daily = row.get(year_col, None)
        if pd.isna(avg_daily) or avg_daily is None or not (1 <= month <= 12):
            totals.append(pd.NA)
            continue
        dim = calendar.monthrange(year, month)[1]
        totals.append(float(avg_daily) * dim)
    return pd.Series(totals)


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
            growth_df[growth_col] = ((pd.to_numeric(df[curr_year], errors="coerce") - pd.to_numeric(df[prev_year], errors="coerce")) / pd.to_numeric(df[prev_year], errors="coerce") * 100).round(1)
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


def export_to_excel(df: pd.DataFrame) -> bytes:
    if not OPENPYXL_AVAILABLE:
        raise ImportError("openpyxl is not available")
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Trafikkdata", index=False)
        if len([col for col in df.columns if str(col).isdigit()]) >= 2:
            calculate_growth_rates(df).to_excel(writer, sheet_name="Vekstrater", index=False)
        seasonal = calculate_seasonal_patterns(df)
        if seasonal:
            pd.DataFrame(seasonal).T.to_excel(writer, sheet_name="Sesongmønstre")
    return output.getvalue()


def create_export_section(df: pd.DataFrame, point: str):
    st.subheader("📊 Eksporter data")
    col1, col2, col3 = st.columns(3)

    with col1:
        csv_data = df.to_csv(index=False, sep=";", encoding="utf-8")
        st.download_button(
            label="📁 Last ned CSV",
            data=csv_data,
            file_name=f"{point}_trafikkdata_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )

    with col2:
        json_data = df.to_json(orient="records", indent=2)
        st.download_button(
            label="🔗 Last ned JSON",
            data=json_data,
            file_name=f"{point}_trafikkdata_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json",
        )

    with col3:
        if OPENPYXL_AVAILABLE:
            st.download_button(
                label="📄 Last ned Excel",
                data=export_to_excel(df),
                file_name=f"{point}_trafikkdata_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        else:
            st.caption("Excel-export krever `openpyxl`.")


def create_advanced_visualization(df: pd.DataFrame, point: str, chart_type: str) -> go.Figure:
    if chart_type == "line_with_confidence":
        fig = go.Figure()
        year_columns = [col for col in df.columns if col not in ["Month", "Month Name", "Week", "Volume"]]
        colors = px.colors.qualitative.Set1
        for i, year in enumerate(year_columns):
            y = pd.to_numeric(df[year], errors="coerce")
            fig.add_trace(
                go.Scatter(
                    x=df["Month Name"] if "Month Name" in df.columns else df.index,
                    y=y,
                    mode="lines+markers",
                    name=str(year),
                    line=dict(color=colors[i % len(colors)], width=3),
                    marker=dict(size=8),
                )
            )
        fig.update_layout(
            title=f"Trafikkutvikling for {point}",
            xaxis_title="Måned",
            yaxis_title="Gjennomsnittlig døgntrafikk",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        return fig

    if chart_type == "heatmap":
        year_columns = [col for col in df.columns if col not in ["Month", "Month Name", "Week", "Volume"]]
        if len(year_columns) > 1 and "Month Name" in df.columns:
            heatmap_data = df[year_columns].apply(pd.to_numeric, errors="coerce").T
            fig = go.Figure(
                data=go.Heatmap(
                    z=heatmap_data.values,
                    x=df["Month Name"],
                    y=[str(y) for y in year_columns],
                    colorscale="RdYlBu_r",
                    hoverongaps=False,
                )
            )
            fig.update_layout(title=f"Sesongmønster for {point}", xaxis_title="Måned", yaxis_title="År")
            return fig
        return create_advanced_visualization(df, point, "line")

    if chart_type == "box":
        year_columns = [col for col in df.columns if col not in ["Month", "Month Name", "Week", "Volume"]]
        fig = go.Figure()
        for year in year_columns:
            y = pd.to_numeric(df[year], errors="coerce").dropna()
            if y.empty:
                continue
            fig.add_trace(go.Box(y=y, name=str(year), boxpoints="all", jitter=0.3, pointpos=-1.8))
        if not fig.data:
            return create_advanced_visualization(df, point, "line")
        fig.update_layout(title=f"Trafikkfordeling for {point}", yaxis_title="Gjennomsnittlig døgntrafikk", xaxis_title="År")
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
        title=f"Trafikkutvikling for {point}",
        labels={"Trafikk": "Gjennomsnittlig døgntrafikk", x_col: "Periode"},
    )
    return fig


def create_comparison_dashboard(df: pd.DataFrame, point: str):
    col1, col2 = st.columns(2)
    with col1:
        is_weekly = "Volume" in df.columns
        year_columns = [col for col in df.columns if col not in ["Month", "Month Name", "Week", "Volume"]]
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

    st.plotly_chart(create_advanced_visualization(df, point, chart_type), use_container_width=True)

    if show_growth:
        growth_df = calculate_growth_rates(df)
        growth_columns = [col for col in growth_df.columns if "Vekst" in col]
        if growth_columns:
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

            growth_melted = growth_df.melt(id_vars=id_vars, value_vars=growth_columns, var_name="Periode", value_name="Vekst (%)")
            fig_growth = px.bar(growth_melted, x=x_col, y="Vekst (%)", color="Periode", title="År-til-år vekstrater")
            fig_growth.add_hline(y=0, line_dash="dash", line_color="black")
            st.plotly_chart(fig_growth, use_container_width=True)


def process_data_for_years(point_ids: List[str], year_list: List[int], timeout_s: int, use_cache: bool) -> pd.DataFrame:
    data: Dict[int, List[float]] = {}
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, year in enumerate(year_list):
        status_text.text(f"Henter data for {year}...")
        progress_bar.progress((i + 1) / len(year_list))
        traffic_data_dict = fetch_batch_traffic_data(point_ids, year, timeout_s, use_cache)
        if traffic_data_dict:
            monthly_sums, _ = sum_traffic_data(traffic_data_dict)
            data[year] = monthly_sums

    status_text.empty()
    progress_bar.empty()

    df = pd.DataFrame({"Month": list(range(1, 13))})
    for y in year_list:
        if y in data:
            df[str(y)] = data[y]
    df = add_month_names(df)
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].round(0).astype("Int64")
    return df


def process_data_for_months(point_ids: List[str], year: int, months: List[int], timeout_s: int, use_cache: bool) -> Optional[pd.DataFrame]:
    traffic_data_dict = fetch_batch_traffic_data(point_ids, year, timeout_s, use_cache)
    if not traffic_data_dict:
        return None
    data, _ = sum_traffic_data(traffic_data_dict)
    df = pd.DataFrame({"Month": list(range(1, 13)), str(year): data})
    df = df[df["Month"].isin(months)]
    df = add_month_names(df)
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].round(0).astype("Int64")
    return df


def process_data_for_weeks(point_ids: List[str], year: int, weeks: List[int], timeout_s: int, use_cache: bool) -> Optional[pd.DataFrame]:
    weekly_data_dict = fetch_weekly_traffic_data(point_ids, year, weeks, timeout_s, use_cache)
    if not weekly_data_dict:
        return None
    weekly_sums = sum_weekly_traffic_data(weekly_data_dict)
    df = pd.DataFrame([{"Week": week, "Volume": volume} for week, volume in weekly_sums.items()])
    df["Week_Num"] = df["Week"].str.extract(r"(\d+)").astype(int)
    df = df.sort_values("Week_Num").drop(columns=["Week_Num"]).reset_index(drop=True)
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").round(0).astype("Int64")
    return df


def render_totals_tab(df: pd.DataFrame, point: str, comparison_mode: str, year_list: List[int], year: int, point_ids: List[str], timeout_s: int, use_cache: bool):
    st.subheader("🧮 Totale passeringer (samlet trafikk)")
    if comparison_mode == "Sammenlign uker":
        st.info(
            "Ukesvisning viser per nå gjennomsnitt per døgn (sum av punkter). "
            "Totale uketall krever at vi vet hvor mange gyldige dager som inngår i snittet."
        )
        return

    years_for_totals = year_list if comparison_mode == "Sammenlign år" else [year]
    totals_df = compute_monthly_totals_table(df, years_for_totals)
    st.caption("Beregning: (månedsvis gjennomsnitt per døgn) × (antall dager i måneden), summert per måned.")

    year_cols = [y for y in years_for_totals if str(y) in totals_df.columns]
    if year_cols:
        latest_year = max(year_cols)
        total, months_present, days_covered = calculate_yearly_total_from_monthly_averages(df, int(latest_year))
        avg_per_day = (total / days_covered) if days_covered else None
        full_year_estimate = (avg_per_day * days_in_year(int(latest_year))) if avg_per_day is not None else None

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric(
                f"Totalt {latest_year} ({'hittil' if months_present < 12 else 'helår'})",
                format_number(total),
                help="Totalt antall passeringer beregnet fra Vegvesen sine måneds-ÅDT.",
            )
        with m2:
            st.metric("Snitt per døgn (basert på tilgjengelige måneder)", format_number(avg_per_day) if avg_per_day is not None else "N/A")
        with m3:
            if months_present < 12 and full_year_estimate is not None:
                st.metric("Estimert helår", format_number(full_year_estimate))
            else:
                st.metric("Måneder med data", f"{months_present}/12")

        if comparison_mode == "Sammenlign år" and len(year_cols) >= 2:
            prev_year = sorted(year_cols)[-2]
            prev_total, _, _ = calculate_yearly_total_from_monthly_averages(df, int(prev_year))
            if prev_total:
                st.metric(f"Endring vs {prev_year}", f"{((total - prev_total) / prev_total * 100):+.1f}%")

    melt_cols = [str(y) for y in years_for_totals if str(y) in totals_df.columns]
    if melt_cols:
        melted = totals_df.melt(id_vars=["Month", "Month Name"], value_vars=melt_cols, var_name="År", value_name="Passeringer")
        fig_totals = px.bar(melted, x="Month Name", y="Passeringer", color="År", barmode="group", title="Totale passeringer per måned")
        fig_totals.update_yaxes(tickformat=",")
        st.plotly_chart(fig_totals, use_container_width=True)

    if point == "Ryfast (sum tunneler)":
        with st.expander("🔎 Fordeling mellom tunneler (Ryfylketunnelen vs Hundvågtunnelen)"):
            breakdown_year = st.selectbox("Velg år for fordeling", options=years_for_totals, index=len(years_for_totals) - 1)
            traffic_by_point = fetch_batch_traffic_data(point_ids, int(breakdown_year), timeout_s, use_cache)
            point_ids_by_group = {
                "Ryfylketunnelen": TRAFFIC_POINTS["Ryfylketunnelen"]["ids"],
                "Hundvågtunnelen": (
                    TRAFFIC_POINTS["Hundvågtunnelen"]["ids"]
                    if st.session_state.get("ryfast_include_ramp", True)
                    else HUNDVAG_TUNNEL_IDS_UTEN_PÅRAMPE
                ),
            }
            totals_by_group_df, coverage_by_group_df = aggregate_monthly_totals_by_group(traffic_by_point, point_ids_by_group, int(breakdown_year))
            melted_groups = totals_by_group_df.melt(id_vars=["Month", "Month Name"], value_vars=list(point_ids_by_group.keys()), var_name="Tunnel", value_name="Passeringer")
            fig_stack = px.bar(melted_groups, x="Month Name", y="Passeringer", color="Tunnel", title=f"Samlet trafikk {breakdown_year} (stacked per tunnel)")
            fig_stack.update_yaxes(tickformat=",")
            st.plotly_chart(fig_stack, use_container_width=True)
            st.caption("Dekning (%) er snitt av rapportert dekning på målepunktene per måned.")
            st.dataframe(coverage_by_group_df.round(1), use_container_width=True, hide_index=True)

    with st.expander("📋 Totaltabell"):
        formatted_totals = totals_df.copy()
        numeric_cols = formatted_totals.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            formatted_totals[col] = formatted_totals[col].map(format_number)
        st.dataframe(formatted_totals, use_container_width=True, hide_index=True)


def main():
    init_session_state()

    st.set_page_config(page_title="Trafikkdata Visualisering - Ryfast", page_icon="🚗", layout="wide", initial_sidebar_state="expanded")
    st.markdown(
        """
    <style>
    .main-header { background: linear-gradient(90deg, #1f77b4, #ff7f0e); color: white; padding: 1rem;
      border-radius: 10px; text-align: center; margin-bottom: 2rem; }
    </style>
    """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
    <div class="main-header">
        <h1>🚗 Trafikkdata Visualisering - Ryfast</h1>
        <p>Analyse av trafikkmønstre for Ryfylketunnelen, Hundvågtunnelen og Bybrua</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.sidebar.header("⚙️ Innstillinger")
    point_options = list(TRAFFIC_POINTS.keys())
    point_descriptions = [f"{p} - {TRAFFIC_POINTS[p]['description']}" for p in point_options]

    with st.sidebar.form("controls", clear_on_submit=False):
        selected_index = st.selectbox(
            "Velg målepunkt",
            range(len(point_options)),
            format_func=lambda x: point_descriptions[x],
            key="point_selector",
        )
        point = point_options[selected_index]

        with st.expander("ℹ️ Om valgt målepunkt"):
            st.write(f"**Beskrivelse:** {TRAFFIC_POINTS[point]['description']}")
            st.write(f"**Åpnet:** {TRAFFIC_POINTS[point]['opened']}")

        comparison_mode = st.radio(
            "Velg analysetype",
            ["Sammenlign år", "Sammenlign måneder", "Sammenlign uker"],
            key="comparison_mode",
        )

        with st.expander("🔧 Avanserte innstillinger"):
            use_cache = st.checkbox("Bruk hurtigbuffer", value=True, key="use_cache")
            timeout_s = st.slider("API timeout (sekunder)", 10, 90, 60, key="timeout_s")

        include_ramp = True
        direction = "Begge retninger"
        if point == "Ryfast (sum tunneler)":
            include_ramp = st.checkbox(
                "Inkluder pårampe i Hundvågtunnelen",
                value=True,
                key="ryfast_include_ramp",
                help="Når av, summeres kun hovedløpene (uten pårampe) for Hundvågtunnelen.",
            )
        elif point == "Bybrua":
            direction = st.selectbox("Velg retning", ["Begge retninger", "Mot nord", "Mot sør"], key="bybrua_direction")

        year_input = DEFAULT_YEARS
        year_list: List[int] = []
        year = 2025
        months: List[int] = []
        weeks: List[int] = []

        if comparison_mode == "Sammenlign år":
            year_input = st.text_input("År (komma-separert)", value=DEFAULT_YEARS, key="years_input")
        elif comparison_mode == "Sammenlign måneder":
            year = st.selectbox("År", list(YEAR_RANGE), index=list(YEAR_RANGE).index(2025) if 2025 in YEAR_RANGE else 0, key="months_year")
            months = st.multiselect(
                "Velg måneder",
                list(range(1, 13)),
                default=list(range(1, 13)),
                format_func=lambda m: MONTH_NAMES[m - 1],
                key="months_selected",
            )
        else:
            year = st.selectbox("År", list(YEAR_RANGE), index=list(YEAR_RANGE).index(2025) if 2025 in YEAR_RANGE else 0, key="weeks_year")
            weeks = st.multiselect("Velg uker", list(range(1, 53)), default=list(range(1, 11)), key="weeks_selected")

        submitted = st.form_submit_button("📊 Analyser data", type="primary")

    if st.sidebar.button("🗑️ Tøm cache"):
        st.cache_data.clear()
        st.session_state.last_result = None
        st.sidebar.success("Cache tømt!")

    def resolve_point_ids() -> List[str]:
        if point == "Ryfast (sum tunneler)":
            ryfylke_ids = TRAFFIC_POINTS["Ryfylketunnelen"]["ids"]
            hundvag_ids = TRAFFIC_POINTS["Hundvågtunnelen"]["ids"] if include_ramp else HUNDVAG_TUNNEL_IDS_UTEN_PÅRAMPE
            return ryfylke_ids + hundvag_ids
        if point == "Ryfylketunnelen":
            return TRAFFIC_POINTS["Ryfylketunnelen"]["ids"]
        if point == "Hundvågtunnelen":
            return TRAFFIC_POINTS["Hundvågtunnelen"]["ids"]
        if direction == "Begge retninger":
            return TRAFFIC_POINTS["Bybrua"]["ids"]["Mot nord"] + TRAFFIC_POINTS["Bybrua"]["ids"]["Mot sør"]
        return TRAFFIC_POINTS["Bybrua"]["ids"][direction]

    if submitted:
        if comparison_mode == "Sammenlign år":
            try:
                year_list = [int(y.strip()) for y in year_input.split(",") if y.strip()]
            except Exception:
                st.sidebar.error("Ugyldig format. Bruk f.eks. 2024,2025")
                st.session_state.last_result = None
                year_list = []
            if not year_list:
                st.sidebar.warning("Velg minst ett år")
                st.session_state.last_result = None
            else:
                year = year_list[-1]

        if comparison_mode == "Sammenlign måneder" and not months:
            st.sidebar.warning("Velg minst én måned")
            st.session_state.last_result = None
        if comparison_mode == "Sammenlign uker" and not weeks:
            st.sidebar.warning("Velg minst én uke")
            st.session_state.last_result = None

        point_ids = resolve_point_ids()

        if (comparison_mode == "Sammenlign år" and year_list) or (comparison_mode != "Sammenlign år"):
            with st.spinner("🔄 Behandler data..."):
                if comparison_mode == "Sammenlign år":
                    df = process_data_for_years(point_ids, year_list, timeout_s, use_cache)
                    title = f"Årlig sammenligning for {point}"
                elif comparison_mode == "Sammenlign måneder":
                    df = process_data_for_months(point_ids, year, months, timeout_s, use_cache)
                    title = f"Månedlig analyse for {point} i {year}"
                else:
                    df = process_data_for_weeks(point_ids, year, weeks, timeout_s, use_cache)
                    title = f"Ukentlig analyse for {point} i {year}"

            if df is None or df.empty:
                st.error("❌ Ingen data tilgjengelig for valgte kriterier")
                st.session_state.last_result = None
            else:
                st.session_state.last_result = {
                    "df": df,
                    "title": title,
                    "point": point,
                    "comparison_mode": comparison_mode,
                    "year_list": year_list,
                    "year": year,
                    "point_ids": point_ids,
                    "timeout_s": timeout_s,
                    "use_cache": use_cache,
                }

    if not st.session_state.last_result:
        st.info("Velg målepunkt og analysetype i sidebaren, og trykk «Analyser data».")
        return

    result = st.session_state.last_result
    df = result["df"]
    title = result["title"]
    point = result["point"]
    comparison_mode = result["comparison_mode"]
    year_list = result["year_list"]
    year = result["year"]
    point_ids = result["point_ids"]
    timeout_s = result["timeout_s"]
    use_cache = result["use_cache"]

    tab1, tab2, tab3, tab4 = st.tabs(["📈 Visualisering", "📊 Data", "🧮 Totaltall", "📄 Rapport"])

    with tab1:
        st.subheader(title)
        create_comparison_dashboard(df, point)

    with tab2:
        st.subheader("📊 Rådata")
        formatted_df = df.copy()
        numeric_cols = formatted_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            formatted_df[col] = formatted_df[col].map(format_number)
        st.dataframe(formatted_df, use_container_width=True, hide_index=True)

    with tab3:
        render_totals_tab(df, point, comparison_mode, year_list, year, point_ids, timeout_s, use_cache)

    with tab4:
        st.subheader("📄 Rapport / eksport")
        create_export_section(df, point)


if __name__ == "__main__":
    main()
