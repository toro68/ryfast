import logging
import numpy as np
import requests
import pandas as pd
import plotly.express as px
import streamlit as st
import time
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import calendar
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import io

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# API Constants
URL = "https://trafikkdata-api.atlas.vegvesen.no"

# Traffic Point Constants with descriptions
TRAFFIC_POINTS = {
    "Ryfast (sum tunneler)": {
        "ids": [
            # Ryfylketunnelen
            "99040V2725982", "00911V2725983",
            # Hundvågtunnelen
            "10239V2725979", "62464V2725991", "92743V2726085", "25926V2725990",
        ],
        "description": "Sum av Ryfylketunnelen + Hundvågtunnelen (Ryfast totalt)",
        "opened": "2019-12-30 / 2020-04-22"
    },
    "Ryfylketunnelen": {
        "ids": ["99040V2725982", "00911V2725983"],
        "description": "Ryfylketunnelen - hovedforbindelse til Ryfylke",
        "opened": "2019-12-30"
    },
    "Hundvågtunnelen": {
        "ids": ["10239V2725979", "62464V2725991", "92743V2726085", "25926V2725990"],
        "description": "Hundvågtunnelen - forbindelse til Hundvåg og Eiganes",
        "opened": "2020-04-22"
    },
    "Bybrua": {
        "ids": {
            "Mot nord": ["17949V320695"],
            "Mot sør": ["54184V320694"]
        },
        "description": "Bybrua - historisk broforbindelse over Strømsteinsundet",
        "opened": "Historisk"
    }
}

HUNDVAG_TUNNEL_IDS_UTEN_PÅRAMPE = [
    "10239V2725979",
    "92743V2726085",
]

# Month Names
MONTH_NAMES = [
    "Januar", "Februar", "Mars", "April", "Mai", "Juni",
    "Juli", "August", "September", "Oktober", "November", "Desember"
]

# Configuration
DEFAULT_YEARS = "2024,2025"
YEAR_RANGE = range(2019, 2026)
API_TIMEOUT = 15
API_MAX_RETRIES = 3
API_RETRY_DELAY = 1
API_CACHE_TTL = 24 * 3600

# GraphQL query templates
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

POINT_INFO_QUERY = """
query {{
  trafficRegistrationPoint(id: "{point_id}") {{
    id
    name
    location {{
      coordinates {{
        latLon {{
          lat
          lon
        }}
      }}
    }}
    operationalStatus
  }}
}}
"""

# Norske månedsnavn
NORWEGIAN_MONTH_NAMES = {
    1: "Januar", 2: "Februar", 3: "Mars", 4: "April", 
    5: "Mai", 6: "Juni", 7: "Juli", 8: "August",
    9: "September", 10: "Oktober", 11: "November", 12: "Desember"
}

# Session state initialization
def init_session_state():
    """Initialize session state variables"""
    if 'data_cache' not in st.session_state:
        st.session_state.data_cache = {}
    if 'export_data' not in st.session_state:
        st.session_state.export_data = None
    if 'comparison_history' not in st.session_state:
        st.session_state.comparison_history = []

@st.cache_data(ttl=API_CACHE_TTL, show_spinner=False)
def fetch_data(query: str) -> Optional[Dict]:
    """Fetch data from the API with improved error handling and retry logic."""
    for attempt in range(API_MAX_RETRIES):
        try:
            with st.spinner(f"Henter data... (forsøk {attempt + 1}/{API_MAX_RETRIES})"):
                response = requests.post(
                    URL, 
                    json={"query": query}, 
                    timeout=API_TIMEOUT
                )
                response.raise_for_status()
                
                data = response.json()
                
                # Check for GraphQL errors
                if "errors" in data:
                    logger.error(f"GraphQL errors: {data['errors']}")
                    st.error(f"GraphQL feil: {data['errors'][0]['message']}")
                    return None
                    
                return data
                
        except requests.Timeout:
            logger.warning(f"Timeout på forsøk {attempt + 1}")
            if attempt == API_MAX_RETRIES - 1:
                st.error("Forespørsel tok for lang tid. Prøv igjen senere.")
                return None
        except requests.RequestException as e:
            if attempt == API_MAX_RETRIES - 1:
                logger.error("API-forespørsel feilet: %s", str(e))
                st.error(f"Kunne ikke hente data: {str(e)}")
                return None
            logger.warning(f"Forsøk {attempt + 1} feilet, prøver igjen...")
            time.sleep(API_RETRY_DELAY * (attempt + 1))  # Exponential backoff

@st.cache_data(ttl=API_CACHE_TTL, show_spinner=False)
def fetch_batch_traffic_data(point_ids: List[str], year: int) -> Dict:
    """Fetch traffic data for multiple points with parallel processing."""
    if year < 2019:
        st.warning(f"Data er ikke tilgjengelig før 2019 (valgt år: {year})")
        return {}

    result = {}
    
    # Use ThreadPoolExecutor for parallel API calls
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_to_point = {}
        
        for point_id in point_ids:
            query = QUERY_TEMPLATE.format(point_id=point_id, year=year)
            future = executor.submit(fetch_data, query)
            future_to_point[future] = point_id
        
        for future in future_to_point:
            point_id = future_to_point[future]
            try:
                data = future.result()
                if data and "data" in data and data["data"]["trafficData"]:
                    monthly_data = data["data"]["trafficData"]["volume"]["average"]["daily"]["byMonth"]
                    if monthly_data:
                        result[point_id] = monthly_data
                    else:
                        logger.warning(f"No monthly data for point ID {point_id} in year {year}")
                else:
                    logger.warning(f"Failed to fetch data for point ID {point_id} in year {year}")
            except Exception as e:
                logger.error("Feil ved henting av data for punkt %s, år %s: %s", point_id, year, str(e))
                continue
                
    return result

@st.cache_data(ttl=API_CACHE_TTL, show_spinner=False)
def fetch_weekly_traffic_data(point_ids: List[str], year: int, week_numbers: List[int]) -> Dict:
    """Fetch traffic data for specific weeks with improved date handling."""
    if year < 2019:
        st.warning(f"Data er ikke tilgjengelig før 2019 (valgt år: {year})")
        return {}

    result = {}
    
    for week_num in week_numbers:
        try:
            # Calculate ISO week dates
            jan_1 = datetime(year, 1, 1)
            week_1_start = jan_1 - timedelta(days=jan_1.weekday())
            if week_1_start.year < year:
                week_1_start += timedelta(weeks=1)
            
            week_start = week_1_start + timedelta(weeks=week_num-1)
            week_end = week_start + timedelta(days=6)
            
            # Ensure dates are within the year
            if week_start.year != year or week_end.year != year:
                continue
                
            from_date = week_start.strftime("%Y-%m-%dT00:00:00+01:00")
            to_date = week_end.strftime("%Y-%m-%dT23:59:59+01:00")
            
            week_data = {}
            for point_id in point_ids:
                query = WEEKLY_QUERY_TEMPLATE.format(
                    point_id=point_id, 
                    from_date=from_date, 
                    to_date=to_date
                )
                
                data = fetch_data(query)
                if data and "data" in data and data["data"]["trafficData"]:
                    daily_data = data["data"]["trafficData"]["volume"]["byDay"]["edges"]
                    if daily_data:
                        total_volume = 0
                        valid_days = 0
                        for edge in daily_data:
                            volume = edge["node"]["total"]["volumeNumbers"]["volume"]
                            if volume is not None:
                                total_volume += volume
                                valid_days += 1
                        
                        if valid_days > 0:
                            week_average = total_volume / valid_days
                            week_data[point_id] = week_average
            
            if week_data:
                result[f"Uke {week_num}"] = week_data
                
        except Exception as e:
            logger.error("Feil ved henting av ukesdata for uke %s: %s", week_num, str(e))
            continue
            
    return result

def sum_traffic_data(traffic_data_dict: Dict) -> List[float]:
    """Sum traffic data from multiple points with confidence intervals."""
    monthly_sums = [0] * 12
    monthly_confidence = [{"lower": 0, "upper": 0} for _ in range(12)]
    
    for point_data in traffic_data_dict.values():
        for entry in point_data:
            month = entry["month"]
            try:
                volume = entry["total"]["volume"]["average"]
                if volume is not None:
                    monthly_sums[month - 1] += volume
                    
                    # Add confidence intervals if available
                    if "confidenceInterval" in entry["total"]["volume"]:
                        ci = entry["total"]["volume"]["confidenceInterval"]
                        if ci["lowerBound"] and ci["upperBound"]:
                            monthly_confidence[month - 1]["lower"] += ci["lowerBound"]
                            monthly_confidence[month - 1]["upper"] += ci["upperBound"]
                else:
                    logger.warning(f"Manglende volumdata for måned {month}")
            except (KeyError, TypeError) as e:
                logger.warning(f"Feil ved lesing av volumdata for måned {month}: {str(e)}")
                continue
                
    return monthly_sums, monthly_confidence

def sum_weekly_traffic_data(weekly_data_dict: Dict) -> Dict:
    """Sum weekly traffic data from multiple points."""
    week_sums = {}
    for week_name, point_data in weekly_data_dict.items():
        total_volume = sum(point_data.values()) if point_data else 0
        week_sums[week_name] = total_volume
    return week_sums

def format_number(x):
    """Format number with thousands separator and handle various types."""
    if pd.isna(x):
        return "N/A"
    elif isinstance(x, (int, float)):
        if x == int(x):  # If it's a whole number
            return f"{int(x):,}".replace(",", " ")
        else:
            return f"{x:,.1f}".replace(",", " ")
    elif isinstance(x, str):
        try:
            num = float(x)
            return format_number(num)
        except ValueError:
            return x
    else:
        return str(x)

def days_in_year(year: int) -> int:
    return 366 if calendar.isleap(year) else 365

def monthly_totals_from_monthly_averages(df: pd.DataFrame, year: int) -> pd.Series:
    """
    Konverterer månedsvis gjennomsnittlig døgntrafikk (ÅDT) til totale passeringer per måned.
    Forutsetter `Month`-kolonne og årskolonne som str (f.eks. "2025").
    """
    year_col = str(year)
    if df is None or df.empty or "Month" not in df.columns or year_col not in df.columns:
        return pd.Series([pd.NA] * (len(df) if df is not None else 0))

    totals = []
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
    """Returnerer tabell med totale passeringer per måned for valgte år."""
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
    """
    Lager månedstabell for totalsummer per gruppe, basert på rå per-punkt API-data.
    Returnerer (totals_df, coverage_df) med kolonner: Month, Month Name, <group...>
    coverage_df er snittdekning (%) per måned per gruppe.
    """
    point_to_group = {}
    for group, ids in point_ids_by_group.items():
        for pid in ids:
            point_to_group[pid] = group

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
            try:
                avg_daily = entry["total"]["volume"]["average"]
            except Exception:
                avg_daily = None
            if avg_daily is None:
                continue
            dim = calendar.monthrange(year, month)[1]
            totals[group][month] += float(avg_daily) * dim

            try:
                cov = entry["total"]["coverage"]["percentage"]
            except Exception:
                cov = None
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
            (sum(cov_lists[g][m]) / len(cov_lists[g][m])) if cov_lists[g][m] else np.nan
            for m in range(1, 13)
        ]
    coverage_df = add_month_names(coverage_df)
    return totals_df, coverage_df

def calculate_yearly_total_from_monthly_averages(df: pd.DataFrame, year: int) -> Tuple[float, int, int]:
    """
    Konverter månedsvis gjennomsnittlig døgntrafikk til totaltall.

    Returnerer (total, antall_måneder_med_data, antall_dager_dekket).
    """
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
        if pd.isna(avg_daily) or avg_daily is None:
            continue

        if not (1 <= month <= 12):
            continue

        dim = calendar.monthrange(year, month)[1]
        total += float(avg_daily) * dim
        months_present += 1
        days_covered += dim

    return total, months_present, days_covered

def calculate_growth_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate year-over-year growth rates."""
    growth_df = df.copy()
    year_columns = [col for col in df.columns if col not in ["Month", "Month Name", "Week", "Volume"]]
    
    if len(year_columns) >= 2:
        for i in range(1, len(year_columns)):
            prev_year = year_columns[i-1]
            curr_year = year_columns[i]
            growth_col = f"Vekst {prev_year}-{curr_year} (%)"
            
            growth_df[growth_col] = ((df[curr_year] - df[prev_year]) / df[prev_year] * 100).round(1)
    
    return growth_df

def calculate_seasonal_patterns(df: pd.DataFrame) -> Dict:
    """Calculate seasonal traffic patterns."""
    if "Month" not in df.columns:
        return {}
    
    patterns = {}
    year_columns = [col for col in df.columns if col not in ["Month", "Month Name"]]
    
    for year in year_columns:
        if year in df.columns:
            yearly_data = df[year].values
            if len(yearly_data) == 12:
                patterns[year] = {
                    "vinter_snitt": np.mean([yearly_data[11], yearly_data[0], yearly_data[1]]),  # Des, Jan, Feb
                    "vår_snitt": np.mean(yearly_data[2:5]),    # Mar, Apr, Mai
                    "sommer_snitt": np.mean(yearly_data[5:8]), # Jun, Jul, Aug
                    "høst_snitt": np.mean(yearly_data[8:11])   # Sep, Okt, Nov
                }
    
    return patterns

def export_to_excel(df: pd.DataFrame, filename: str) -> bytes:
    """Export DataFrame to Excel with formatting."""
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Write main data
        df.to_excel(writer, sheet_name='Trafikkdata', index=False)
        
        # Add growth rates if applicable
        if len([col for col in df.columns if col.isdigit()]) >= 2:
            growth_df = calculate_growth_rates(df)
            growth_df.to_excel(writer, sheet_name='Vekstrater', index=False)
        
        # Add seasonal patterns
        seasonal = calculate_seasonal_patterns(df)
        if seasonal:
            seasonal_df = pd.DataFrame(seasonal).T
            seasonal_df.to_excel(writer, sheet_name='Sesongmønstre')
    
    return output.getvalue()

def create_advanced_visualization(df: pd.DataFrame, point: str, chart_type: str = "line") -> go.Figure:
    """Create advanced visualizations with multiple chart types."""
    
    if chart_type == "line_with_confidence":
        # Line chart with confidence intervals (if available)
        fig = go.Figure()
        
        year_columns = [col for col in df.columns if col not in ["Month", "Month Name", "Week", "Volume"]]
        colors = px.colors.qualitative.Set1
        
        for i, year in enumerate(year_columns):
            fig.add_trace(go.Scatter(
                x=df["Month Name"] if "Month Name" in df.columns else df.index,
                y=df[year],
                mode='lines+markers',
                name=year,
                line=dict(color=colors[i % len(colors)], width=3),
                marker=dict(size=8)
            ))
        
        fig.update_layout(
            title=f"Trafikkutvikling for {point}",
            xaxis_title="Måned",
            yaxis_title="Gjennomsnittlig døgntrafikk",
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )
        
    elif chart_type == "heatmap":
        # Heatmap for seasonal patterns
        year_columns = [col for col in df.columns if col not in ["Month", "Month Name", "Week", "Volume"]]
        if len(year_columns) > 1 and "Month Name" in df.columns:
            heatmap_data = df[year_columns].T
            
            fig = go.Figure(data=go.Heatmap(
                z=heatmap_data.values,
                x=df["Month Name"],
                y=year_columns,
                colorscale='RdYlBu_r',
                hoverongaps=False
            ))
            
            fig.update_layout(
                title=f"Sesongmønster for {point}",
                xaxis_title="Måned",
                yaxis_title="År"
            )
        else:
            return create_advanced_visualization(df, point, "line")
    
    elif chart_type == "box":
        # Box plot for distribution analysis
        year_columns = [col for col in df.columns if col not in ["Month", "Month Name", "Week", "Volume"]]
        
        fig = go.Figure()
        
        for year in year_columns:
            fig.add_trace(go.Box(
                y=df[year],
                name=year,
                boxpoints='all',
                jitter=0.3,
                pointpos=-1.8
            ))
        
        fig.update_layout(
            title=f"Trafikkfordeling for {point}",
            yaxis_title="Gjennomsnittlig døgntrafikk",
            xaxis_title="År"
        )
    
    else:  # Default line chart
        df_melted = df.melt(
            id_vars=['Month', 'Month Name'] if 'Month Name' in df.columns else ['Week'],
            var_name='År',
            value_name='Trafikk'
        )
        
        fig = px.line(
            df_melted, 
            x='Month Name' if 'Month Name' in df_melted.columns else 'Week',
            y='Trafikk',
            color='År',
            title=f"Trafikkutvikling for {point}",
            labels={
                'Trafikk': 'Gjennomsnittlig døgntrafikk',
                'Month Name': 'Måned'
            }
        )
    
    return fig

def create_comparison_dashboard(df: pd.DataFrame, point: str):
    """Create a comprehensive comparison dashboard."""
    
    col1, col2 = st.columns(2)
    
    with col1:
        chart_type = st.selectbox(
            "Velg diagramtype",
            ["line", "heatmap", "box", "line_with_confidence"],
            format_func=lambda x: {
                "line": "Linjediagram",
                "heatmap": "Varmekart",
                "box": "Boksplot",
                "line_with_confidence": "Linje med konfidensintervall"
            }[x]
        )
    
    with col2:
        show_growth = st.checkbox("Vis vekstrater", value=False)
    
    # Main visualization
    fig = create_advanced_visualization(df, point, chart_type)
    st.plotly_chart(fig, use_container_width=True)
    
    # Growth rates if requested
    if show_growth:
        growth_df = calculate_growth_rates(df)
        growth_columns = [col for col in growth_df.columns if "Vekst" in col]
        
        if growth_columns:
            st.subheader("Vekstrater (år-til-år)")
            
            growth_melted = growth_df.melt(
                id_vars=['Month', 'Month Name'] if 'Month Name' in growth_df.columns else ['Week'],
                value_vars=growth_columns,
                var_name='Periode',
                value_name='Vekst (%)'
            )
            
            fig_growth = px.bar(
                growth_melted,
                x='Month Name' if 'Month Name' in growth_melted.columns else 'Week',
                y='Vekst (%)',
                color='Periode',
                title="År-til-år vekstrater"
            )
            
            # Add horizontal line at 0%
            fig_growth.add_hline(y=0, line_dash="dash", line_color="black")
            
            st.plotly_chart(fig_growth, use_container_width=True)

def process_data_for_years(point_ids: List[str], year_list: List[int]) -> pd.DataFrame:
    """Process data for multiple years with improved error handling."""
    data = {}
    confidence_data = {}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, year in enumerate(year_list):
        status_text.text(f"Henter data for {year}...")
        progress_bar.progress((i + 1) / len(year_list))
        
        traffic_data_dict = fetch_batch_traffic_data(point_ids, year)
        if traffic_data_dict:
            monthly_sums, monthly_conf = sum_traffic_data(traffic_data_dict)
            data[year] = monthly_sums
            confidence_data[year] = monthly_conf
        else:
            st.warning(f"Ingen komplette data for alle punkter i år {year}")

    status_text.empty()
    progress_bar.empty()
    
    df = pd.DataFrame({"Month": list(range(1, 13))})
    for year in year_list:
        if year in data:
            df[f"{year}"] = data[year]
    
    df = add_month_names(df)
    
    # Round numeric columns to integers
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].round(0).astype('Int64')  # Use nullable integer
    
    return df

def process_data_for_months(point_ids: List[str], year: int, months: List[int]) -> Optional[pd.DataFrame]:
    """Process data for selected months of a year."""
    with st.spinner(f"Henter data for {year}..."):
        traffic_data_dict = fetch_batch_traffic_data(point_ids, year)
        if traffic_data_dict:
            data, _ = sum_traffic_data(traffic_data_dict)
            df = pd.DataFrame({
                "Month": list(range(1, 13)),
                f"{year}": data
            })
            df = df[df['Month'].isin(months)]
            df = add_month_names(df)
            
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            df[numeric_columns] = df[numeric_columns].round(0).astype('Int64')
            
            return df
        else:
            st.warning(f"Ingen data tilgjengelig for år {year}")
            return None

def process_data_for_weeks(point_ids: List[str], year: int, weeks: List[int]) -> Optional[pd.DataFrame]:
    """Process data for selected weeks of a year."""
    with st.spinner(f"Henter ukesdata for {year}..."):
        weekly_data_dict = fetch_weekly_traffic_data(point_ids, year, weeks)
        if weekly_data_dict:
            weekly_sums = sum_weekly_traffic_data(weekly_data_dict)
            
            df = pd.DataFrame([
                {"Week": week_name, "Volume": volume} 
                for week_name, volume in weekly_sums.items()
            ])
            
            # Sort by week number
            df['Week_Num'] = df['Week'].str.extract(r'(\d+)').astype(int)
            df = df.sort_values('Week_Num').drop('Week_Num', axis=1).reset_index(drop=True)
            
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            df[numeric_columns] = df[numeric_columns].round(0).astype('Int64')
            
            return df
        else:
            st.warning(f"Ingen ukesdata tilgjengelig for år {year}")
            return None

def calculate_additional_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate comprehensive statistics for the dataset."""
    stats = {}
    
    if "Volume" in df.columns:  # Weekly data
        volume_data = df["Volume"].dropna()
        if len(volume_data) > 0:
            stats["Ukesdata"] = {
                "Høyeste uke": df.loc[volume_data.idxmax(), "Week"] if not volume_data.empty else "N/A",
                "Høyeste volum": volume_data.max(),
                "Laveste uke": df.loc[volume_data.idxmin(), "Week"] if not volume_data.empty else "N/A",
                "Laveste volum": volume_data.min(),
                "Volumspenn": volume_data.max() - volume_data.min(),
                "Variasjonskoeffisient (%)": (volume_data.std() / volume_data.mean() * 100).round(2),
                "Median": volume_data.median(),
                "Kvartil 1": volume_data.quantile(0.25),
                "Kvartil 3": volume_data.quantile(0.75)
            }
    else:  # Monthly/yearly data
        year_columns = [col for col in df.columns if col not in ["Month", "Month Name"]]
        for year in year_columns:
            year_data = df[year].dropna()
            if len(year_data) > 0:
                year_total, _, _ = calculate_yearly_total_from_monthly_averages(df, int(year)) if str(year).isdigit() else (np.nan, 0, 0)
                stats[year] = {
                    "Toppmåned": df.loc[year_data.idxmax(), "Month Name"] if "Month Name" in df.columns else "N/A",
                    "Toppvolum": year_data.max(),
                    "Laveste måned": df.loc[year_data.idxmin(), "Month Name"] if "Month Name" in df.columns else "N/A",
                    "Laveste volum": year_data.min(),
                    "Volumspenn": year_data.max() - year_data.min(),
                    "Variasjonskoeffisient (%)": (year_data.std() / year_data.mean() * 100).round(2),
                    "Årstrafikk": year_total,
                    "Median": year_data.median(),
                    "Kvartil 1": year_data.quantile(0.25),
                    "Kvartil 3": year_data.quantile(0.75)
                }
    
    return pd.DataFrame(stats).T

def add_month_names(df: pd.DataFrame) -> pd.DataFrame:
    """Add month names to the dataframe."""
    if "Month" in df.columns:
        df["Month Name"] = [MONTH_NAMES[i - 1] for i in df["Month"]]
        return df[
            ["Month", "Month Name"]
            + [col for col in df.columns if col not in ["Month", "Month Name"]]
        ]
    return df

def create_export_section(df: pd.DataFrame, point: str):
    """Create export functionality section."""
    st.subheader("📊 Eksporter data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📋 Kopier CSV til utklippstavle"):
            csv_string = df.to_csv(index=False, sep=';')
            st.code(csv_string, language='csv')
            st.success("CSV-data vist over - kopier manuelt")
    
    with col2:
        excel_data = export_to_excel(df, f"{point}_trafikkdata.xlsx")
        st.download_button(
            label="📁 Last ned Excel",
            data=excel_data,
            file_name=f"{point}_trafikkdata_{datetime.now().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    with col3:
        json_data = df.to_json(orient='records', indent=2)
        st.download_button(
            label="🔗 Last ned JSON",
            data=json_data,
            file_name=f"{point}_trafikkdata_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json"
        )

def create_comparison_report(df: pd.DataFrame, point: str) -> str:
    """Generate a comprehensive comparison report."""
    report = f"""
# Trafikkrapport for {point}
*Generert: {datetime.now().strftime('%d.%m.%Y %H:%M')}*

## Sammendrag
"""
    
    year_columns = [col for col in df.columns if col.isdigit()]
    if len(year_columns) >= 2:
        latest_year = max(year_columns, key=int)
        previous_year = str(int(latest_year) - 1)
        
        if previous_year in year_columns:
            latest_total, latest_months, _ = calculate_yearly_total_from_monthly_averages(df, int(latest_year))
            previous_total, _, _ = calculate_yearly_total_from_monthly_averages(df, int(previous_year))
            growth = ((latest_total - previous_total) / previous_total * 100) if previous_total else 0
            
            report += f"""
- **Totalt antall passeringer (Vegvesen-telling) {latest_year}**: {format_number(latest_total)} ({latest_months}/12 mnd)
- **Endring fra {previous_year}**: {growth:+.1f}%
- **Høyeste måned {latest_year}**: {df.loc[df[latest_year].idxmax(), 'Month Name']} ({format_number(df[latest_year].max())})
- **Laveste måned {latest_year}**: {df.loc[df[latest_year].idxmin(), 'Month Name']} ({format_number(df[latest_year].min())})

## Sesongvariasjoner
"""
            
            seasonal = calculate_seasonal_patterns(df)
            if latest_year in seasonal:
                s = seasonal[latest_year]
                report += f"""
- **Vinter** (des-feb): {format_number(s['vinter_snitt'])} gjennomsnitt
- **Vår** (mar-mai): {format_number(s['vår_snitt'])} gjennomsnitt  
- **Sommer** (jun-aug): {format_number(s['sommer_snitt'])} gjennomsnitt
- **Høst** (sep-nov): {format_number(s['høst_snitt'])} gjennomsnitt

"""
    
    return report

def main():
    """Enhanced main function with improved UI and functionality."""
    
    # Initialize session state
    init_session_state()
    
    # Page configuration
    st.set_page_config(
        page_title="Trafikkdata Visualisering - Ryfast",
        page_icon="🚗",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
    .warning-box {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>🚗 Trafikkdata Visualisering - Ryfast</h1>
        <p>Avansert analyse av trafikkmønstre for Ryfylketunnelen, Hundvågtunnelen og Bybrua</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar configuration
    st.sidebar.header("⚙️ Innstillinger")
    
    # Point selection with descriptions
    point_options = list(TRAFFIC_POINTS.keys())
    point_descriptions = [f"{point} - {TRAFFIC_POINTS[point]['description']}" for point in point_options]
    
    selected_index = st.sidebar.selectbox(
        "Velg målepunkt",
        range(len(point_options)),
        format_func=lambda x: point_descriptions[x],
        key="point_selector"
    )
    point = point_options[selected_index]
    
    # Display point information
    with st.sidebar.expander("ℹ️ Om valgt målepunkt"):
        st.write(f"**Beskrivelse:** {TRAFFIC_POINTS[point]['description']}")
        st.write(f"**Åpnet:** {TRAFFIC_POINTS[point]['opened']}")
        if point == "Ryfylketunnelen":
            st.write("**Lengde:** 14.4 km")
            st.write("**Maksimal dybde:** 292 meter under havnivå")
        elif point == "Hundvågtunnelen":
            st.write("**Lengde:** 5.7 km")
            st.write("**Del av:** Ryfast-prosjektet")

    comparison_mode = st.sidebar.radio(
        "Velg analysetype",
        ["Sammenlign år", "Sammenlign måneder", "Sammenlign uker"],
        key="comparison_mode_selector",
        help="Velg hvilken type sammenligning du ønsker å utføre"
    )

    # Advanced options
    with st.sidebar.expander("🔧 Avanserte innstillinger"):
        enable_confidence_intervals = st.checkbox("Vis konfidensintervaller", value=False)
        enable_caching = st.checkbox("Bruk hurtigbuffer", value=True)
        api_timeout = st.slider("API timeout (sekunder)", 10, 60, API_TIMEOUT)

    # Handle point IDs based on selection
    if point == "Ryfast (sum tunneler)":
        include_ramp = st.sidebar.checkbox(
            "Inkluder pårampe i Hundvågtunnelen",
            value=True,
            key="ryfast_include_ramp",
            help="Når av, summeres kun hovedløpene (uten pårampe) for Hundvågtunnelen.",
        )
        ryfylke_ids = TRAFFIC_POINTS["Ryfylketunnelen"]["ids"]
        hundvag_ids = (
            TRAFFIC_POINTS["Hundvågtunnelen"]["ids"]
            if include_ramp
            else HUNDVAG_TUNNEL_IDS_UTEN_PÅRAMPE
        )
        point_ids = ryfylke_ids + hundvag_ids
    elif point == "Ryfylketunnelen":
        point_ids = TRAFFIC_POINTS["Ryfylketunnelen"]["ids"]
    elif point == "Hundvågtunnelen":
        point_ids = TRAFFIC_POINTS["Hundvågtunnelen"]["ids"]
    else:  # Bybrua
        direction = st.sidebar.selectbox(
            "Velg retning",
            ["Begge retninger", "Mot nord", "Mot sør"],
            key="direction_selector"
        )

        if direction == "Begge retninger":
            point_ids = (
                TRAFFIC_POINTS["Bybrua"]["ids"]["Mot nord"]
                + TRAFFIC_POINTS["Bybrua"]["ids"]["Mot sør"]
            )
        else:
            point_ids = TRAFFIC_POINTS["Bybrua"]["ids"][direction]

    # Input configuration based on comparison mode
    if comparison_mode == "Sammenlign år":
        year_input = st.sidebar.text_input(
            "År som skal sammenlignes (kommaseparert)",
            DEFAULT_YEARS,
            key="year_input",
            help="Eksempel: 2022,2023,2024,2025"
        )
        try:
            year_list = [int(year.strip()) for year in year_input.split(",")]
            invalid_years = [year for year in year_list if year < 2019 or year > 2026]
            if invalid_years:
                st.sidebar.error(f"Ugyldige år: {', '.join(map(str, invalid_years))}")
                st.stop()
        except ValueError:
            st.sidebar.error("Ugyldig format. Bruk format: 2023,2024,2025")
            st.stop()
            
    elif comparison_mode == "Sammenlign måneder":
        year = st.sidebar.selectbox(
            "Velg år", 
            list(range(2019, 2026)), 
            index=6,  # Default to 2025
            key="year_selector_months"
        )
        
        quarter = st.sidebar.selectbox(
            "Hurtigvalg",
            ["Alle måneder", "Q1 (Jan-Mar)", "Q2 (Apr-Jun)", "Q3 (Jul-Sep)", "Q4 (Okt-Des)"],
            key="quarter_selector"
        )
        
        if quarter == "Alle måneder":
            default_months = list(range(1, 13))
        elif quarter == "Q1 (Jan-Mar)":
            default_months = [1, 2, 3]
        elif quarter == "Q2 (Apr-Jun)":
            default_months = [4, 5, 6]
        elif quarter == "Q3 (Jul-Sep)":
            default_months = [7, 8, 9]
        else:  # Q4
            default_months = [10, 11, 12]
            
        months = st.sidebar.multiselect(
            "Velg måneder",
            options=list(range(1, 13)),
            default=default_months,
            format_func=lambda x: NORWEGIAN_MONTH_NAMES[x],
            key="month_selector"
        )
        
        if not months:
            st.sidebar.warning("Velg minst én måned")
            st.stop()
        
    else:  # Sammenlign uker
        year = st.sidebar.selectbox(
            "Velg år", 
            list(range(2019, 2026)), 
            index=6,  # Default to 2025
            key="year_selector_weeks"
        )
        
        week_range = st.sidebar.selectbox(
            "Hurtigvalg",
            ["Egendefinert", "Første kvartal (1-13)", "Andre kvartal (14-26)", 
             "Tredje kvartal (27-39)", "Fjerde kvartal (40-52)"],
            key="week_range_selector"
        )
        
        if week_range == "Første kvartal (1-13)":
            default_weeks = list(range(1, 14))
        elif week_range == "Andre kvartal (14-26)":
            default_weeks = list(range(14, 27))
        elif week_range == "Tredje kvartal (27-39)":
            default_weeks = list(range(27, 40))
        elif week_range == "Fjerde kvartal (40-52)":
            default_weeks = list(range(40, 53))
        else:
            default_weeks = list(range(1, 11))
            
        weeks = st.sidebar.multiselect(
            "Velg uker",
            options=list(range(1, 53)),
            default=default_weeks,
            key="week_selector",
            help="ISO uke-nummerering. Uke 1 starter første mandag i januar."
        )
        
        if not weeks:
            st.sidebar.warning("Velg minst én uke")
            st.stop()

    # Action buttons
    col1, col2 = st.sidebar.columns(2)
    with col1:
        fetch_data_btn = st.button("📊 Analyser data", type="primary", key="fetch_button")
    with col2:
        clear_cache_btn = st.button("🗑️ Tøm cache", key="clear_cache_button")
    
    if clear_cache_btn:
        st.cache_data.clear()
        st.sidebar.success("Cache tømt!")

    # Main content area
    if fetch_data_btn:
        try:
            with st.spinner("🔄 Behandler data..."):
                if comparison_mode == "Sammenlign år":
                    df = process_data_for_years(point_ids, year_list)
                    title = f"Årlig sammenligning for {point}"
                    
                elif comparison_mode == "Sammenlign måneder":
                    df = process_data_for_months(point_ids, year, months)
                    title = f"Månedlig analyse for {point} i {year}"
                    
                else:  # Sammenlign uker
                    df = process_data_for_weeks(point_ids, year, weeks)
                    title = f"Ukentlig analyse for {point} i {year}"
                
                if df is None or df.empty:
                    st.error("❌ Ingen data tilgjengelig for valgte kriterier")
                    st.stop()

            # Create tabs for different views
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📈 Visualisering", "📊 Data", "🧮 Totaltall", "📋 Statistikk", "📄 Rapport"
            ])

            with tab1:
                st.subheader(title)
                create_comparison_dashboard(df, point)
                
                # Additional insights
                if comparison_mode == "Sammenlign år" and len(year_list) >= 2:
                    with st.expander("🔍 Innsikter og trender"):
                        seasonal_patterns = calculate_seasonal_patterns(df)
                        
                        if seasonal_patterns:
                            st.write("**Sesongmønstre:**")
                            seasonal_df = pd.DataFrame(seasonal_patterns).T
                            seasonal_df = seasonal_df.round(0).astype(int)
                            seasonal_df.columns = ["Vinter", "Vår", "Sommer", "Høst"]
                            st.dataframe(seasonal_df.map(format_number))
                            
                        # Growth analysis
                        growth_df = calculate_growth_rates(df)
                        growth_cols = [col for col in growth_df.columns if "Vekst" in col]
                        if growth_cols:
                            avg_growth = growth_df[growth_cols].mean().mean()
                            st.metric("Gjennomsnittlig årlig vekst", f"{avg_growth:.1f}%")

            with tab2:
                st.subheader("📊 Rådata")
                
                # Display formatted data
                formatted_df = df.copy()
                numeric_cols = formatted_df.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    formatted_df[col] = formatted_df[col].map(format_number)
                
                st.dataframe(
                    formatted_df,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Summary metrics
                if comparison_mode == "Sammenlign år":
                    year_columns = [col for col in df.columns if col.isdigit()]
                    if year_columns:
                        latest_year = max(year_columns, key=int)
                        latest_year_int = int(latest_year)
                        total, months_present, days_covered = calculate_yearly_total_from_monthly_averages(df, latest_year_int)
                        avg_per_day = (total / days_covered) if days_covered else None
                        full_year_estimate = (avg_per_day * days_in_year(latest_year_int)) if avg_per_day is not None else None
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(
                                f"Trafikk {latest_year} ({'hittil' if months_present < 12 else 'totalt'})",
                                format_number(total),
                                help="Beregnet som (månedsvis gj.snitt per døgn) × (antall dager i måned), summert."
                            )
                        with col2:
                            peak_month = df.loc[df[latest_year].idxmax(), 'Month Name']
                            peak_value = df[latest_year].max()
                            st.metric(
                                "Topptrafikk måned",
                                f"{peak_month}: {format_number(peak_value)}"
                            )
                        with col3:
                            if months_present < 12 and full_year_estimate is not None:
                                st.metric(
                                    "Estimert helår",
                                    format_number(full_year_estimate),
                                    help="Ekstrapolerer fra tilgjengelige måneder ved å bruke vektet snitt per dag."
                                )
                            else:
                                variation = (df[latest_year].std() / df[latest_year].mean() * 100)
                                st.metric(
                                    "Sesongvariasjon",
                                    f"{variation:.1f}%",
                                    help="Variasjonskoeffisient mellom måneder"
                                )

            with tab3:
                st.subheader("🧮 Totale passeringer (samlet trafikk)")

                if comparison_mode == "Sammenlign uker":
                    st.info(
                        "Ukesvisning viser per nå gjennomsnitt per døgn (sum av punkter). "
                        "Totale uketall krever at vi vet hvor mange gyldige dager som inngår i snittet."
                    )
                else:
                    years_for_totals = year_list if comparison_mode == "Sammenlign år" else [year]
                    totals_df = compute_monthly_totals_table(df, years_for_totals)

                    st.caption(
                        "Beregning: (månedsvis gjennomsnitt per døgn) × (antall dager i måneden), summert per måned."
                    )

                # Summary metrics for latest year
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
                            st.metric(
                                "Snitt per døgn (basert på tilgjengelige måneder)",
                                format_number(avg_per_day) if avg_per_day is not None else "N/A",
                            )
                        with m3:
                            if months_present < 12 and full_year_estimate is not None:
                                st.metric("Estimert helår", format_number(full_year_estimate))
                            else:
                                st.metric("Måneder med data", f"{months_present}/12")

                        if comparison_mode == "Sammenlign år" and len(year_cols) >= 2:
                            prev_year = sorted(year_cols)[-2]
                            prev_total, _, _ = calculate_yearly_total_from_monthly_averages(df, int(prev_year))
                            if prev_total:
                                st.metric(
                                    f"Endring vs {prev_year}",
                                    f"{((total - prev_total) / prev_total * 100):+.1f}%",
                                )

                # Plot totals by month/year
                    melt_cols = [str(y) for y in years_for_totals if str(y) in totals_df.columns]
                    if melt_cols:
                        melted = totals_df.melt(
                            id_vars=[c for c in ["Month", "Month Name"] if c in totals_df.columns],
                            value_vars=melt_cols,
                            var_name="År",
                            value_name="Passeringer",
                        )
                        fig_totals = px.bar(
                            melted,
                            x="Month Name" if "Month Name" in melted.columns else "Month",
                            y="Passeringer",
                            color="År",
                            barmode="group",
                            title="Totale passeringer per måned",
                        )
                        fig_totals.update_yaxes(tickformat=",")
                        st.plotly_chart(fig_totals, use_container_width=True)

                # Extra: breakdown when summing both tunnels
                    if point == "Ryfast (sum tunneler)":
                        with st.expander("🔎 Fordeling mellom tunneler (Ryfylketunnelen vs Hundvågtunnelen)"):
                            breakdown_year = st.selectbox(
                                "Velg år for fordeling",
                                options=years_for_totals,
                                index=len(years_for_totals) - 1,
                                key="ryfast_breakdown_year",
                            )
                            traffic_by_point = fetch_batch_traffic_data(point_ids, int(breakdown_year))
                            point_ids_by_group = {
                                "Ryfylketunnelen": TRAFFIC_POINTS["Ryfylketunnelen"]["ids"],
                                "Hundvågtunnelen": (
                                    TRAFFIC_POINTS["Hundvågtunnelen"]["ids"]
                                    if st.session_state.get("ryfast_include_ramp", True)
                                    else HUNDVAG_TUNNEL_IDS_UTEN_PÅRAMPE
                                ),
                            }

                            totals_by_group_df, coverage_by_group_df = aggregate_monthly_totals_by_group(
                                traffic_by_point, point_ids_by_group, int(breakdown_year)
                            )

                            melted_groups = totals_by_group_df.melt(
                                id_vars=["Month", "Month Name"],
                                value_vars=list(point_ids_by_group.keys()),
                                var_name="Tunnel",
                                value_name="Passeringer",
                            )
                            fig_stack = px.bar(
                                melted_groups,
                                x="Month Name",
                                y="Passeringer",
                                color="Tunnel",
                                title=f"Samlet trafikk {breakdown_year} (stacked per tunnel)",
                            )
                            fig_stack.update_yaxes(tickformat=",")
                            st.plotly_chart(fig_stack, use_container_width=True)

                            st.caption("Dekning (%) er snitt av rapportert dekning på målepunktene per måned.")
                            st.dataframe(
                                coverage_by_group_df.round(1),
                                use_container_width=True,
                                hide_index=True,
                            )

                # Show totals table
                    with st.expander("📋 Totaltabell"):
                        formatted_totals = totals_df.copy()
                        numeric_cols = formatted_totals.select_dtypes(include=[np.number]).columns
                        for col in numeric_cols:
                            formatted_totals[col] = formatted_totals[col].map(format_number)
                        st.dataframe(formatted_totals, use_container_width=True, hide_index=True)

            with tab4:
                st.subheader("📋 Detaljert statistikk")
                
                # Basic statistics
                st.write("**Grunnleggende statistikk:**")
                basic_stats = df.describe().round(1)
                basic_stats.index = [
                    "Antall observasjoner", "Gjennomsnitt", "Standardavvik", 
                    "Minimum", "25% kvartil", "Median (50%)", "75% kvartil", "Maksimum"
                ]
                st.dataframe(basic_stats.map(format_number), use_container_width=True)
                
                # Advanced statistics
                st.write("**Avansert statistikk:**")
                advanced_stats = calculate_additional_statistics(df)
                st.dataframe(advanced_stats.map(format_number), use_container_width=True)

            with tab5:
                st.subheader("📄 Automatisk rapport")
                
                report_text = create_comparison_report(df, point)
                st.markdown(report_text)
                
                # Export section
                create_export_section(df, point)
                
                # Save to comparison history
                comparison_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "point": point,
                    "mode": comparison_mode,
                    "parameters": {
                        "years": year_list if comparison_mode == "Sammenlign år" else [year],
                        "months": months if comparison_mode == "Sammenlign måneder" else None,
                        "weeks": weeks if comparison_mode == "Sammenlign uker" else None
                    }
                }
                st.session_state.comparison_history.append(comparison_entry)
                
                # Show comparison history
                if len(st.session_state.comparison_history) > 1:
                    with st.expander("📚 Sammenligningshistorikk"):
                        for i, entry in enumerate(reversed(st.session_state.comparison_history[-5:])):
                            st.write(f"**{i+1}.** {entry['point']} - {entry['mode']} "
                                   f"({entry['timestamp'][:19].replace('T', ' ')})")

        except Exception as e:
            st.error(f"❌ En feil oppstod: {str(e)}")
            logger.exception("Feil i hovedprosess")
            
            with st.expander("🔧 Feilsøkingsinformasjon"):
                st.code(f"""
Feiltype: {type(e).__name__}
Feilmelding: {str(e)}
Valgte innstillinger:
- Punkt: {point}
- Modus: {comparison_mode}
- API Timeout: {api_timeout}s
                """)

    # Footer with additional information
    st.markdown("---")
    
    # Information sections
    col1, col2 = st.columns(2)
    
    with col1:
        with st.expander("📅 Ryfast tidslinje"):
            st.markdown("""
            **2019:**
            - 30. desember: Ryfylketunnelen åpnet
            
            **2020:**
            - 22. april: Hundvågtunnelen åpnet
            - Oktober: Bom på Bybrua snudd
            
            **2021:**
            - Februar: Bompengeinnkreving startet
            
            **2022-2024:**
            - Regelmessige takstøkninger
            
            **2025:**
            - Fortsatt drift og datainnsamling
            """)
    
    with col2:
        with st.expander("ℹ️ Om dataene"):
            st.markdown("""
            **Datakilde:** Statens vegvesen trafikkdata API
            
            **Datatype:** Gjennomsnittlig døgntrafikk (ÅDT)
            
            **Oppdateringsfrekvens:** Daglig
            
            **Kvalitet:** Data inkluderer kvalitetsparametere
            
            **Dekningsgrad:** Varierer per målepunkt og tidsperiode
            
            **Beregninger:** Totaler er estimert basert på ÅDT × 365
            """)

    # Technical info in sidebar
    with st.sidebar.expander("🔧 Teknisk informasjon"):
        st.write(f"**API URL:** {URL}")
        st.write(f"**Cache TTL:** {API_CACHE_TTL/3600:.1f} timer")
        st.write(f"**Maks forsøk:** {API_MAX_RETRIES}")
        st.write(f"**Timeout:** {API_TIMEOUT}s")
        st.write(f"**Siste oppdatering:** {datetime.now().strftime('%H:%M:%S')}")

if __name__ == "__main__":
    main()
