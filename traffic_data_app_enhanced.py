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
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import threading

# Forbedret logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# API Constants - Forbedrede verdier
URL = "https://trafikkdata-api.atlas.vegvesen.no"

# Forbedrede timeout-innstillinger
API_TIMEOUT = 60  # Økt fra 15 til 60
API_MAX_RETRIES = 2  # Redusert for raskere feilhåndtering
API_RETRY_DELAY = 2
API_CACHE_TTL = 24 * 3600

# Nye konfigurasjon for avbrytelse
MAX_CONCURRENT_REQUESTS = 3  # Begrens samtidige requests
MAX_WEEKS_WITHOUT_WARNING = 5  # Advarsel for mange uker

# Traffic Point Constants (samme som før)
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

# Month Names
MONTH_NAMES = [
    "Januar", "Februar", "Mars", "April", "Mai", "Juni",
    "Juli", "August", "September", "Oktober", "November", "Desember"
]

DEFAULT_YEARS = "2024,2025"
YEAR_RANGE = range(2019, 2026)

# Optimaliserte GraphQL queries
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

# Optimalisert ukesdata query - henter kun nødvendige felt
WEEKLY_QUERY_TEMPLATE = """
query {{
  trafficData(trafficRegistrationPointId: "{point_id}") {{
    volume {{
      byDay(from: "{from_date}", to: "{to_date}") {{
        edges {{
          node {{
            from
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

NORWEGIAN_MONTH_NAMES = {
    1: "Januar", 2: "Februar", 3: "Mars", 4: "April", 
    5: "Mai", 6: "Juni", 7: "Juli", 8: "August",
    9: "September", 10: "Oktober", 11: "November", 12: "Desember"
}

# Forbedret session state med avbrytelse-funksjonalitet
def init_session_state():
    """Initialize enhanced session state with cancellation support"""
    if 'data_cache' not in st.session_state:
        st.session_state.data_cache = {}
    if 'export_data' not in st.session_state:
        st.session_state.export_data = None
    if 'comparison_history' not in st.session_state:
        st.session_state.comparison_history = []
    if 'cancel_requested' not in st.session_state:
        st.session_state.cancel_requested = False
    if 'current_operation' not in st.session_state:
        st.session_state.current_operation = None
    if 'operation_progress' not in st.session_state:
        st.session_state.operation_progress = {"current": 0, "total": 0, "status": ""}

# Forbedret fetch_data med timeout-debugging og avbrytelse
@st.cache_data(ttl=API_CACHE_TTL, show_spinner=False)
def fetch_data_enhanced(query: str, operation_name: str = "API-kall") -> Optional[Dict]:
    """Enhanced fetch with cancellation support and detailed logging."""
    
    # Sjekk om operasjon er avbrutt
    if st.session_state.get('cancel_requested', False):
        logger.info(f"Operation cancelled: {operation_name}")
        return None
    
    for attempt in range(API_MAX_RETRIES):
        try:
            # Progress feedback
            progress_text = f"{operation_name} (forsøk {attempt + 1}/{API_MAX_RETRIES})"
            if attempt > 0:
                progress_text += " - Retry etter feil"
            
            # Update operation status
            st.session_state.operation_progress["status"] = progress_text
            
            start_time = time.time()
            
            headers = {
                'Content-Type': 'application/json',
                'User-Agent': 'Ryfast-Enhanced/2.0',
                'Accept': 'application/json'
            }
            
            response = requests.post(
                URL, 
                json={"query": query}, 
                timeout=API_TIMEOUT,
                headers=headers
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            
            response.raise_for_status()
            data = response.json()
            
            # Detailed response logging
            logger.info(f"{operation_name} completed in {response_time:.1f}s")
            
            # Check for GraphQL errors
            if "errors" in data:
                error_msg = data['errors'][0]['message']
                logger.error(f"GraphQL error in {operation_name}: {error_msg}")
                st.error(f"GraphQL feil: {error_msg}")
                return None
            
            # Success feedback
            if response_time > 10:
                st.info(f"⏱️ {operation_name}: {response_time:.1f}s (treg respons)")
            
            return data
            
        except requests.Timeout:
            logger.warning(f"Timeout in {operation_name} attempt {attempt + 1} (>{API_TIMEOUT}s)")
            if attempt == API_MAX_RETRIES - 1:
                st.error(f"⏰ Timeout: {operation_name} tok mer enn {API_TIMEOUT} sekunder")
                st.warning("💡 Prøv å redusere datamengde eller sjekk nettverkstilkobling")
                return None
            else:
                st.warning(f"⏰ Timeout forsøk {attempt + 1}, prøver igjen...")
                time.sleep(API_RETRY_DELAY * (attempt + 1))
                
        except requests.RequestException as e:
            if attempt == API_MAX_RETRIES - 1:
                logger.error(f"Request failed for {operation_name}: {str(e)}")
                st.error(f"🔌 Nettverksfeil: {str(e)}")
                return None
            logger.warning(f"Attempt {attempt + 1} failed for {operation_name}, retrying...")
            time.sleep(API_RETRY_DELAY * (attempt + 1))
        
        # Sjekk avbrytelse mellom forsøk
        if st.session_state.get('cancel_requested', False):
            logger.info(f"Operation cancelled during retry: {operation_name}")
            return None

# Forbedret batch data fetching med progress tracking
@st.cache_data(ttl=API_CACHE_TTL, show_spinner=False)
def fetch_batch_traffic_data_enhanced(point_ids: List[str], year: int) -> Dict:
    """Enhanced batch fetch with progress tracking and cancellation."""
    if year < 2019:
        st.warning(f"Data er ikke tilgjengelig før 2019 (valgt år: {year})")
        return {}

    # Reset cancellation flag
    st.session_state.cancel_requested = False
    st.session_state.current_operation = f"Henter data for {year}"
    
    # Setup progress tracking
    total_points = len(point_ids)
    st.session_state.operation_progress = {
        "current": 0, 
        "total": total_points, 
        "status": "Starter datahenting..."
    }
    
    # Create progress UI
    progress_container = st.container()
    with progress_container:
        st.subheader(f"🔄 Henter data for {year}")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Cancel button
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("❌ Avbryt", key=f"cancel_{year}"):
                st.session_state.cancel_requested = True
                st.warning("Operasjon avbrutt av bruker")
                return {}
    
    result = {}
    
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS) as executor:
        # Submit all tasks
        future_to_point = {}
        for i, point_id in enumerate(point_ids):
            if st.session_state.get('cancel_requested', False):
                break
                
            query = QUERY_TEMPLATE.format(point_id=point_id, year=year)
            future = executor.submit(fetch_data_enhanced, query, f"Punkt {i+1}/{total_points}")
            future_to_point[future] = point_id
        
        # Process completed futures
        completed = 0
        for future in as_completed(future_to_point):
            if st.session_state.get('cancel_requested', False):
                # Cancel remaining futures
                for f in future_to_point:
                    f.cancel()
                break
                
            point_id = future_to_point[future]
            completed += 1
            
            # Update progress
            progress = completed / total_points
            progress_bar.progress(progress)
            status_text.text(f"Fullført: {completed}/{total_points} målepunkter")
            
            try:
                data = future.result()
                if data and "data" in data and data["data"]["trafficData"]:
                    monthly_data = data["data"]["trafficData"]["volume"]["average"]["daily"]["byMonth"]
                    if monthly_data:
                        result[point_id] = monthly_data
                        logger.info(f"Successfully fetched data for point {point_id}")
                    else:
                        logger.warning(f"No monthly data for point {point_id} in year {year}")
                else:
                    logger.warning(f"Failed to fetch data for point {point_id} in year {year}")
            except Exception as e:
                logger.error(f"Error processing data for point {point_id}: {str(e)}")
                st.warning(f"Feil ved prosessering av punkt {point_id}: {str(e)}")
    
    # Cleanup progress UI
    progress_container.empty()
    
    if not st.session_state.get('cancel_requested', False):
        st.success(f"✅ Hentet data for {len(result)}/{total_points} målepunkter")
    
    # Reset operation state
    st.session_state.current_operation = None
    st.session_state.cancel_requested = False
    
    return result

# Kraftig forbedret weekly data fetching
@st.cache_data(ttl=API_CACHE_TTL, show_spinner=False)
def fetch_weekly_traffic_data_enhanced(point_ids: List[str], year: int, week_numbers: List[int]) -> Dict:
    """Enhanced weekly fetch with smart batching and cancellation."""
    if year < 2019:
        st.warning(f"Data er ikke tilgjengelig før 2019 (valgt år: {year})")
        return {}
    
    num_weeks = len(week_numbers)
    num_points = len(point_ids)
    total_requests = num_weeks * num_points
    
    # Advarsel for mange requests
    if total_requests > 20:
        st.warning(f"⚠️ Mange API-kall ({total_requests}) kan ta {total_requests * 2}-{total_requests * 5} sekunder")
        if not st.button(f"🚀 Fortsett med {num_weeks} uker", key=f"continue_weeks"):
            st.info("💡 Reduser antall uker eller velg færre målepunkter for raskere resultat")
            return {}
    
    # Reset state
    st.session_state.cancel_requested = False
    st.session_state.current_operation = f"Henter ukesdata for {year}"
    
    # Progress tracking setup
    progress_container = st.container()
    with progress_container:
        st.subheader(f"📅 Henter ukesdata for {year}")
        
        # Main progress bar
        main_progress = st.progress(0)
        
        # Detailed status
        col1, col2 = st.columns(2)
        with col1:
            status_text = st.empty()
        with col2:
            eta_text = st.empty()
        
        # Performance metrics
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        with metrics_col1:
            requests_metric = st.empty()
        with metrics_col2:
            speed_metric = st.empty()
        with metrics_col3:
            success_metric = st.empty()
        
        # Cancel button
        if st.button("❌ Avbryt operasjon", key=f"cancel_weekly"):
            st.session_state.cancel_requested = True
            st.error("Operasjon avbrutt av bruker")
            return {}
    
    result = {}
    successful_requests = 0
    failed_requests = 0
    start_time = time.time()
    
    for week_idx, week_num in enumerate(week_numbers):
        if st.session_state.get('cancel_requested', False):
            break
            
        try:
            # Calculate ISO week dates
            jan_1 = datetime(year, 1, 1)
            week_1_start = jan_1 - timedelta(days=jan_1.weekday())
            if week_1_start.year < year:
                week_1_start += timedelta(weeks=1)
            
            week_start = week_1_start + timedelta(weeks=week_num-1)
            week_end = week_start + timedelta(days=6)
            
            # Validate dates
            if week_start.year != year or week_end.year != year:
                continue
                
            from_date = week_start.strftime("%Y-%m-%dT00:00:00+01:00")
            to_date = week_end.strftime("%Y-%m-%dT23:59:59+01:00")
            
            # Progress update
            overall_progress = week_idx / num_weeks
            main_progress.progress(overall_progress)
            status_text.text(f"Prosesserer uke {week_num} ({week_idx + 1}/{num_weeks})")
            
            # ETA calculation
            if week_idx > 0:
                elapsed = time.time() - start_time
                avg_time_per_week = elapsed / week_idx
                remaining_weeks = num_weeks - week_idx
                eta_seconds = remaining_weeks * avg_time_per_week
                eta_text.text(f"ETA: {eta_seconds:.0f}s")
            
            week_data = {}
            
            # Fetch data for all points in this week
            for point_idx, point_id in enumerate(point_ids):
                if st.session_state.get('cancel_requested', False):
                    break
                    
                query = WEEKLY_QUERY_TEMPLATE.format(
                    point_id=point_id, 
                    from_date=from_date, 
                    to_date=to_date
                )
                
                operation_name = f"Uke {week_num}, punkt {point_idx + 1}/{num_points}"
                data = fetch_data_enhanced(query, operation_name)
                
                if data and "data" in data and data["data"]["trafficData"]:
                    daily_data = data["data"]["trafficData"]["volume"]["byDay"]["edges"]
                    if daily_data:
                        total_volume = 0
                        valid_days = 0
                        for edge in daily_data:
                            volume_info = edge["node"]["total"]["volumeNumbers"]
                            if volume_info and volume_info["volume"] is not None:
                                total_volume += volume_info["volume"]
                                valid_days += 1
                        
                        if valid_days > 0:
                            week_average = total_volume / valid_days
                            week_data[point_id] = week_average
                            successful_requests += 1
                        else:
                            failed_requests += 1
                    else:
                        failed_requests += 1
                else:
                    failed_requests += 1
                
                # Update metrics
                total_requests_made = successful_requests + failed_requests
                requests_metric.metric("Requests", f"{total_requests_made}/{total_requests}")
                
                if total_requests_made > 0:
                    success_rate = (successful_requests / total_requests_made) * 100
                    success_metric.metric("Suksessrate", f"{success_rate:.0f}%")
                
                if week_idx > 0:
                    elapsed = time.time() - start_time
                    speed = total_requests_made / elapsed
                    speed_metric.metric("Hastighet", f"{speed:.1f} req/s")
                
                # Rate limiting - vær snill mot API
                time.sleep(0.3)
            
            if week_data:
                result[f"Uke {week_num}"] = week_data
                
        except Exception as e:
            logger.error(f"Error processing week {week_num}: {str(e)}")
            st.warning(f"Feil ved prosessering av uke {week_num}: {str(e)}")
            failed_requests += len(point_ids)
    
    # Cleanup and final status
    progress_container.empty()
    
    total_elapsed = time.time() - start_time
    
    if not st.session_state.get('cancel_requested', False):
        st.success(f"✅ Ukesdata hentet på {total_elapsed:.1f} sekunder")
        st.info(f"📊 {successful_requests} vellykkede, {failed_requests} feilede requests")
    else:
        st.warning(f"⚠️ Operasjon avbrutt etter {total_elapsed:.1f} sekunder")
    
    # Reset state
    st.session_state.current_operation = None
    st.session_state.cancel_requested = False
    
    return result

# De andre funksjonene forblir stort sett like, men med forbedret error handling
def sum_traffic_data(traffic_data_dict: Dict) -> List[float]:
    """Sum traffic data with enhanced error handling."""
    monthly_sums = [0] * 12
    monthly_confidence = [{"lower": 0, "upper": 0} for _ in range(12)]
    
    if not traffic_data_dict:
        logger.warning("No traffic data to sum")
        return monthly_sums, monthly_confidence
    
    for point_id, point_data in traffic_data_dict.items():
        try:
            for entry in point_data:
                month = entry["month"]
                volume = entry["total"]["volume"]["average"]
                if volume is not None and 1 <= month <= 12:
                    monthly_sums[month - 1] += volume
                    
                    # Add confidence intervals if available
                    if "confidenceInterval" in entry["total"]["volume"]:
                        ci = entry["total"]["volume"]["confidenceInterval"]
                        if ci and ci.get("lowerBound") and ci.get("upperBound"):
                            monthly_confidence[month - 1]["lower"] += ci["lowerBound"]
                            monthly_confidence[month - 1]["upper"] += ci["upperBound"]
                else:
                    logger.warning(f"Invalid data for month {month} in point {point_id}")
        except (KeyError, TypeError) as e:
            logger.warning(f"Error processing data for point {point_id}: {str(e)}")
            continue
                
    return monthly_sums, monthly_confidence

def sum_weekly_traffic_data(weekly_data_dict: Dict) -> Dict:
    """Sum weekly traffic data with error handling."""
    week_sums = {}
    
    if not weekly_data_dict:
        logger.warning("No weekly data to sum")
        return week_sums
    
    for week_name, point_data in weekly_data_dict.items():
        try:
            if point_data:
                total_volume = sum(volume for volume in point_data.values() if volume is not None)
                week_sums[week_name] = total_volume
            else:
                week_sums[week_name] = 0
        except (TypeError, AttributeError) as e:
            logger.warning(f"Error processing weekly data for {week_name}: {str(e)}")
            week_sums[week_name] = 0
    
    return week_sums

# Utility functions (samme som før men med bedre error handling)
def format_number(x):
    """Format number with thousands separator and handle various types."""
    if pd.isna(x) or x is None:
        return "N/A"
    elif isinstance(x, (int, float)):
        if np.isnan(x):
            return "N/A"
        if x == int(x):
            return f"{int(x):,}".replace(",", " ")
        else:
            return f"{x:,.1f}".replace(",", " ")
    elif isinstance(x, str):
        try:
            num = float(x)
            return format_number(num)
        except (ValueError, TypeError):
            return x
    else:
        return str(x)

def days_in_year(year: int) -> int:
    return 366 if calendar.isleap(year) else 365

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
    """Calculate growth rates with error handling."""
    try:
        growth_df = df.copy()
        year_columns = [col for col in df.columns if col not in ["Month", "Month Name", "Week", "Volume"]]
        
        if len(year_columns) >= 2:
            for i in range(1, len(year_columns)):
                prev_year = year_columns[i-1]
                curr_year = year_columns[i]
                growth_col = f"Vekst {prev_year}-{curr_year} (%)"
                
                # Avoid division by zero
                prev_data = df[prev_year].replace(0, np.nan)
                growth_df[growth_col] = ((df[curr_year] - prev_data) / prev_data * 100).round(1)
        
        return growth_df
    except Exception as e:
        logger.error(f"Error calculating growth rates: {str(e)}")
        return df

def calculate_seasonal_patterns(df: pd.DataFrame) -> Dict:
    """Calculate seasonal patterns with error handling."""
    try:
        if "Month" not in df.columns:
            return {}
        
        patterns = {}
        year_columns = [col for col in df.columns if col not in ["Month", "Month Name"]]
        
        for year in year_columns:
            if year in df.columns:
                yearly_data = df[year].dropna()
                if len(yearly_data) >= 12:
                    patterns[year] = {
                        "vinter_snitt": np.mean([yearly_data.iloc[11], yearly_data.iloc[0], yearly_data.iloc[1]]),
                        "vår_snitt": np.mean(yearly_data.iloc[2:5]),
                        "sommer_snitt": np.mean(yearly_data.iloc[5:8]),
                        "høst_snitt": np.mean(yearly_data.iloc[8:11])
                    }
        
        return patterns
    except Exception as e:
        logger.error(f"Error calculating seasonal patterns: {str(e)}")
        return {}

# Export functions with error handling
try:
    import openpyxl
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False
    logger.warning("openpyxl not available, Excel export disabled")

def export_to_excel(df: pd.DataFrame, filename: str) -> bytes:
    """Export DataFrame to Excel with enhanced error handling."""
    if not OPENPYXL_AVAILABLE:
        raise ImportError("openpyxl not available")
    
    try:
        output = io.BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Write main data
            df.to_excel(writer, sheet_name='Trafikkdata', index=False)
            
            # Add growth rates if applicable
            year_columns = [col for col in df.columns if col.isdigit()]
            if len(year_columns) >= 2:
                growth_df = calculate_growth_rates(df)
                growth_df.to_excel(writer, sheet_name='Vekstrater', index=False)
            
            # Add seasonal patterns
            seasonal = calculate_seasonal_patterns(df)
            if seasonal:
                seasonal_df = pd.DataFrame(seasonal).T
                seasonal_df.to_excel(writer, sheet_name='Sesongmønstre')
        
        return output.getvalue()
    except Exception as e:
        logger.error(f"Error creating Excel export: {str(e)}")
        raise

def export_to_csv_alternative(df: pd.DataFrame, filename: str) -> str:
    """Export DataFrame to CSV as alternative to Excel."""
    try:
        return df.to_csv(index=False, sep=';', encoding='utf-8')
    except Exception as e:
        logger.error(f"Error creating CSV export: {str(e)}")
        return f"Error creating CSV: {str(e)}"

# Visualisering functions (samme som før)
def create_advanced_visualization(df: pd.DataFrame, point: str, chart_type: str = "line") -> go.Figure:
    """Create advanced visualizations with error handling."""
    try:
        if chart_type == "line_with_confidence":
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
            if "Month Name" in df.columns:
                df_melted = df.melt(
                    id_vars=['Month', 'Month Name'],
                    var_name='År',
                    value_name='Trafikk'
                )
                x_col = 'Month Name'
            elif "Week" in df.columns:
                df_melted = df.melt(
                    id_vars=['Week'],
                    var_name='År', 
                    value_name='Trafikk'
                )
                x_col = 'Week'
            else:
                df_melted = df.melt(
                    var_name='År',
                    value_name='Trafikk'
                )
                x_col = df_melted.index
            
            fig = px.line(
                df_melted, 
                x=x_col,
                y='Trafikk',
                color='År',
                title=f"Trafikkutvikling for {point}",
                labels={
                    'Trafikk': 'Gjennomsnittlig døgntrafikk',
                    x_col: 'Periode'
                }
            )
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating visualization: {str(e)}")
        # Return simple fallback chart
        fig = go.Figure()
        fig.add_annotation(
            text=f"Feil ved generering av diagram: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False
        )
        return fig

def create_comparison_dashboard(df: pd.DataFrame, point: str):
    """Create enhanced comparison dashboard with error handling."""
    try:
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
            try:
                growth_df = calculate_growth_rates(df)
                growth_columns = [col for col in growth_df.columns if "Vekst" in col]
                
                if growth_columns:
                    st.subheader("Vekstrater (år-til-år)")
                    
                    if "Month Name" in growth_df.columns:
                        id_vars = ['Month', 'Month Name']
                        x_col = 'Month Name'
                    elif "Week" in growth_df.columns:
                        id_vars = ['Week']
                        x_col = 'Week'
                    else:
                        id_vars = []
                        x_col = growth_df.index
                    
                    growth_melted = growth_df.melt(
                        id_vars=id_vars,
                        value_vars=growth_columns,
                        var_name='Periode',
                        value_name='Vekst (%)'
                    )
                    
                    fig_growth = px.bar(
                        growth_melted,
                        x=x_col,
                        y='Vekst (%)',
                        color='Periode',
                        title="År-til-år vekstrater"
                    )
                    
                    # Add horizontal line at 0%
                    fig_growth.add_hline(y=0, line_dash="dash", line_color="black")
                    
                    st.plotly_chart(fig_growth, use_container_width=True)
            except Exception as e:
                st.error(f"Feil ved generering av vekstrater: {str(e)}")
    
    except Exception as e:
        st.error(f"Feil ved generering av dashboard: {str(e)}")

# Enhanced data processing functions
def process_data_for_years(point_ids: List[str], year_list: List[int]) -> pd.DataFrame:
    """Process data for multiple years with enhanced progress tracking."""
    data = {}
    confidence_data = {}
    
    # Enhanced progress tracking
    st.subheader("📊 Prosesserer årsdata")
    
    for i, year in enumerate(year_list):
        if st.session_state.get('cancel_requested', False):
            st.warning("Operasjon avbrutt")
            break
            
        traffic_data_dict = fetch_batch_traffic_data_enhanced(point_ids, year)
        if traffic_data_dict:
            monthly_sums, monthly_conf = sum_traffic_data(traffic_data_dict)
            data[year] = monthly_sums
            confidence_data[year] = monthly_conf
        else:
            st.warning(f"Ingen komplette data for alle punkter i år {year}")

    if not data:
        st.error("Ingen data ble hentet. Sjekk nettverkstilkobling og prøv igjen.")
        return pd.DataFrame()
    
    # Build DataFrame
    try:
        df = pd.DataFrame({"Month": list(range(1, 13))})
        for year in year_list:
            if year in data:
                df[f"{year}"] = data[year]
        
        df = add_month_names(df)
        
        # Round numeric columns to integers
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].round(0).astype('Int64')
        
        return df
    except Exception as e:
        st.error(f"Feil ved oppbygging av DataFrame: {str(e)}")
        return pd.DataFrame()

def process_data_for_months(point_ids: List[str], year: int, months: List[int]) -> Optional[pd.DataFrame]:
    """Process data for selected months with error handling."""
    try:
        traffic_data_dict = fetch_batch_traffic_data_enhanced(point_ids, year)
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
    except Exception as e:
        st.error(f"Feil ved prosessering av månedlige data: {str(e)}")
        return None

def process_data_for_weeks(point_ids: List[str], year: int, weeks: List[int]) -> Optional[pd.DataFrame]:
    """Process data for selected weeks with enhanced handling."""
    try:
        weekly_data_dict = fetch_weekly_traffic_data_enhanced(point_ids, year, weeks)
        if weekly_data_dict:
            weekly_sums = sum_weekly_traffic_data(weekly_data_dict)
            
            if not weekly_sums:
                st.warning("Ingen ukesdata ble hentet")
                return None
            
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
    except Exception as e:
        st.error(f"Feil ved prosessering av ukesdata: {str(e)}")
        return None

def calculate_additional_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate comprehensive statistics with error handling."""
    try:
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
                    "Variasjonskoeffisient (%)": (volume_data.std() / volume_data.mean() * 100).round(2) if volume_data.mean() > 0 else 0,
                    "Median": volume_data.median(),
                    "Kvartil 1": volume_data.quantile(0.25),
                    "Kvartil 3": volume_data.quantile(0.75)
                }
        else:  # Monthly/yearly data
            year_columns = [col for col in df.columns if col not in ["Month", "Month Name"]]
            for year in year_columns:
                year_data = df[year].dropna()
                if len(year_data) > 0:
                    year_total, _, _ = calculate_yearly_total_from_monthly_averages(df, int(year)) if str(year).isdigit() else (0, 0, 0)
                    stats[year] = {
                        "Toppmåned": df.loc[year_data.idxmax(), "Month Name"] if "Month Name" in df.columns and not year_data.empty else "N/A",
                        "Toppvolum": year_data.max(),
                        "Laveste måned": df.loc[year_data.idxmin(), "Month Name"] if "Month Name" in df.columns and not year_data.empty else "N/A",
                        "Laveste volum": year_data.min(),
                        "Volumspenn": year_data.max() - year_data.min(),
                        "Variasjonskoeffisient (%)": (year_data.std() / year_data.mean() * 100).round(2) if year_data.mean() > 0 else 0,
                        "Årstrafikk": year_total,
                        "Median": year_data.median(),
                        "Kvartil 1": year_data.quantile(0.25),
                        "Kvartil 3": year_data.quantile(0.75)
                    }
        
        return pd.DataFrame(stats).T
    except Exception as e:
        logger.error(f"Error calculating statistics: {str(e)}")
        return pd.DataFrame()

def add_month_names(df: pd.DataFrame) -> pd.DataFrame:
    """Add month names with error handling."""
    try:
        if "Month" in df.columns:
            df["Month Name"] = [MONTH_NAMES[i - 1] for i in df["Month"]]
            return df[
                ["Month", "Month Name"]
                + [col for col in df.columns if col not in ["Month", "Month Name"]]
            ]
        return df
    except Exception as e:
        logger.error(f"Error adding month names: {str(e)}")
        return df

def create_export_section(df: pd.DataFrame, point: str):
    """Create enhanced export functionality section."""
    try:
        st.subheader("📊 Eksporter data")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # CSV export (always available)
            csv_data = export_to_csv_alternative(df, f"{point}_trafikkdata.csv")
            st.download_button(
                label="📄 Last ned CSV",
                data=csv_data,
                file_name=f"{point}_trafikkdata_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        with col2:
            # Excel export (if available)
            if OPENPYXL_AVAILABLE:
                try:
                    excel_data = export_to_excel(df, f"{point}_trafikkdata.xlsx")
                    st.download_button(
                        label="📊 Last ned Excel",
                        data=excel_data,
                        file_name=f"{point}_trafikkdata_{datetime.now().strftime('%Y%m%d')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                except Exception as e:
                    st.error(f"Feil ved Excel-eksport: {str(e)}")
                    st.info("CSV-eksport er tilgjengelig som alternativ")
            else:
                st.info("📊 Excel-eksport ikke tilgjengelig (installer openpyxl)")
        
        with col3:
            # JSON export
            try:
                json_data = df.to_json(orient='records', indent=2)
                st.download_button(
                    label="🔗 Last ned JSON",
                    data=json_data,
                    file_name=f"{point}_trafikkdata_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json"
                )
            except Exception as e:
                st.error(f"Feil ved JSON-eksport: {str(e)}")
    
    except Exception as e:
        st.error(f"Feil ved opprettelse av eksport-seksjon: {str(e)}")

def create_comparison_report(df: pd.DataFrame, point: str) -> str:
    """Generate enhanced comparison report."""
    try:
        report = f"""
# 🚗 Trafikkrapport for {point}
*Generert: {datetime.now().strftime('%d.%m.%Y %H:%M')}*

## 📊 Sammendrag
"""
        
        year_columns = [col for col in df.columns if col.isdigit()]
        if len(year_columns) >= 2:
            latest_year = max(year_columns, key=int)
            previous_year = str(int(latest_year) - 1)
            
            if previous_year in year_columns:
                latest_total, latest_months, _ = calculate_yearly_total_from_monthly_averages(df, int(latest_year))
                previous_total, _, _ = calculate_yearly_total_from_monthly_averages(df, int(previous_year))
                
                if previous_total > 0:
                    growth = ((latest_total - previous_total) / previous_total * 100)
                    
                    report += f"""
- **Totalt antall passeringer (Vegvesen-telling) {latest_year}**: {format_number(latest_total)} ({latest_months}/12 mnd)
- **Endring fra {previous_year}**: {growth:+.1f}%
- **Høyeste måned {latest_year}**: {df.loc[df[latest_year].idxmax(), 'Month Name']} ({format_number(df[latest_year].max())})
- **Laveste måned {latest_year}**: {df.loc[df[latest_year].idxmin(), 'Month Name']} ({format_number(df[latest_year].min())})

## 🌡️ Sesongvariasjoner
"""
                    
                    seasonal = calculate_seasonal_patterns(df)
                    if latest_year in seasonal:
                        s = seasonal[latest_year]
                        report += f"""
- **Vinter** (des-feb): {format_number(s['vinter_snitt'])} gjennomsnitt
- **Vår** (mar-mai): {format_number(s['vår_snitt'])} gjennomsnitt  
- **Sommer** (jun-aug): {format_number(s['sommer_snitt'])} gjennomsnitt
- **Høst** (sep-nov): {format_number(s['høst_snitt'])} gjennomsnitt

## 🎯 Konklusjon

Rapporten viser {'vekst' if growth > 0 else 'nedgang'} i trafikken på {point}.
"""
        
        return report
    except Exception as e:
        logger.error(f"Error creating report: {str(e)}")
        return f"Feil ved generering av rapport: {str(e)}"

# Enhanced main function
def main():
    """Enhanced main function with comprehensive error handling."""
    
    # Initialize session state
    init_session_state()
    
    # Page configuration
    st.set_page_config(
        page_title="Ryfast Trafikkdata - Enhanced",
        page_icon="🚗",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for enhanced styling
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
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .cancel-button {
        background-color: #dc3545;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Enhanced header with status
    st.markdown("""
    <div class="main-header">
        <h1>🚗 Ryfast Trafikkdata - Enhanced Edition</h1>
        <p>Avansert analyse med progress tracking, timeout-håndtering og avbrytelse-funksjonalitet</p>
    </div>
    """, unsafe_allow_html=True)

    # Status indicator
    if st.session_state.get('current_operation'):
        st.info(f"🔄 Pågående operasjon: {st.session_state.current_operation}")
        if st.button("❌ Avbryt pågående operasjon"):
            st.session_state.cancel_requested = True
            st.session_state.current_operation = None
            st.rerun()

    # Sidebar configuration
    st.sidebar.header("⚙️ Forbedrede innstillinger")
    
    # Point selection
    point_options = list(TRAFFIC_POINTS.keys())
    point_descriptions = [f"{point} - {TRAFFIC_POINTS[point]['description']}" for point in point_options]
    
    selected_index = st.sidebar.selectbox(
        "Velg målepunkt",
        range(len(point_options)),
        format_func=lambda x: point_descriptions[x],
        key="point_selector"
    )
    point = point_options[selected_index]
    
    # Enhanced point information
    with st.sidebar.expander("ℹ️ Om valgt målepunkt"):
        st.write(f"**Beskrivelse:** {TRAFFIC_POINTS[point]['description']}")
        st.write(f"**Åpnet:** {TRAFFIC_POINTS[point]['opened']}")
        
        if point == "Ryfylketunnelen":
            st.write("**Lengde:** 14.4 km")
            st.write("**Maksimal dybde:** 292 meter under havnivå")
            st.write("**Status:** ✅ API tilgjengelig")
        elif point == "Hundvågtunnelen":
            st.write("**Lengde:** 5.7 km")
            st.write("**Del av:** Ryfast-prosjektet")
            st.write("**Status:** ✅ API tilgjengelig")

    comparison_mode = st.sidebar.radio(
        "Velg analysetype",
        ["Sammenlign år", "Sammenlign måneder", "Sammenlign uker"],
        key="comparison_mode_selector",
        help="Velg hvilken type sammenligning du ønsker å utføre"
    )

    # Enhanced advanced options
    with st.sidebar.expander("🔧 Avanserte innstillinger"):
        enable_confidence_intervals = st.checkbox("Vis konfidensintervaller", value=False)
        enable_caching = st.checkbox("Bruk hurtigbuffer", value=True)
        
        st.write("**API-innstillinger:**")
        st.write(f"• Timeout: {API_TIMEOUT}s")
        st.write(f"• Maks forsøk: {API_MAX_RETRIES}")
        st.write(f"• Samtidige requests: {MAX_CONCURRENT_REQUESTS}")
        
        if st.button("🧪 Test API-tilkobling"):
            test_query = """
            query {
              trafficRegistrationPoints(first: 1) {
                edges {
                  node {
                    id
                    name
                  }
                }
              }
            }
            """
            with st.spinner("Tester API..."):
                result = fetch_data_enhanced(test_query, "API-test")
                if result:
                    st.success("✅ API-tilkobling fungerer")
                else:
                    st.error("❌ API-tilkobling feilet")

    # Handle point IDs based on selection
    if point == "Ryfylketunnelen":
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
            point_ids = (TRAFFIC_POINTS["Bybrua"]["ids"]["Mot nord"] + 
                        TRAFFIC_POINTS["Bybrua"]["ids"]["Mot sør"])
        else:
            point_ids = TRAFFIC_POINTS["Bybrua"]["ids"][direction]

    # Enhanced input configuration
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
            
            # Estimate time for operation
            estimated_time = len(year_list) * len(point_ids) * 3
            if estimated_time > 30:
                st.sidebar.warning(f"⏰ Estimert tid: {estimated_time}s")
                
        except ValueError:
            st.sidebar.error("Ugyldig format. Bruk format: 2023,2024,2025")
            st.stop()
            
    elif comparison_mode == "Sammenlign måneder":
        year = st.sidebar.selectbox(
            "Velg år", 
            list(range(2019, 2026)), 
            index=6,
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
            index=6,
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
            default_weeks = list(range(1, 6))  # Safe default
            
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
        
        # Enhanced warnings for weekly data
        if len(weeks) > MAX_WEEKS_WITHOUT_WARNING:
            estimated_requests = len(weeks) * len(point_ids)
            estimated_time = estimated_requests * 2
            st.sidebar.warning(f"⚠️ Mange uker valgt: {len(weeks)}")
            st.sidebar.info(f"📊 Estimert {estimated_requests} API-kall (~{estimated_time}s)")

    # Enhanced action buttons
    col1, col2, col3 = st.sidebar.columns(3)
    with col1:
        fetch_data_btn = st.button("📊 Analyser", type="primary", key="fetch_button")
    with col2:
        clear_cache_btn = st.button("🗑️ Rens cache", key="clear_cache_button")
    with col3:
        if st.button("🔄 Reset", key="reset_button"):
            st.session_state.cancel_requested = False
            st.session_state.current_operation = None
            st.rerun()
    
    if clear_cache_btn:
        st.cache_data.clear()
        st.sidebar.success("Cache renset!")

    # System status
    with st.sidebar.expander("📊 Systemstatus"):
        st.write("**API Status:**")
        st.write(f"✅ Base URL: {URL}")
        st.write(f"⏱️ Timeout: {API_TIMEOUT}s")
        st.write(f"🔄 Cache TTL: {API_CACHE_TTL/3600:.1f}h")
        
        st.write("**Tilgjengelige eksporter:**")
        st.write(f"📄 CSV: ✅ Tilgjengelig")
        st.write(f"📊 Excel: {'✅ Tilgjengelig' if OPENPYXL_AVAILABLE else '❌ Ikke installert'}")
        st.write(f"🔗 JSON: ✅ Tilgjengelig")
        
        st.write(f"**Siste oppdatering:** {datetime.now().strftime('%H:%M:%S')}")

    # Main content area with enhanced error handling
    if fetch_data_btn:
        try:
            # Reset any previous cancellation
            st.session_state.cancel_requested = False
            
            with st.spinner("🔄 Forbereder dataanalyse..."):
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
                    st.info("💡 Prøv å:")
                    st.write("• Sjekk nettverkstilkobling")
                    st.write("• Velg færre datapunkter")
                    st.write("• Prøv et annet tidsrom")
                    st.stop()

            # Tabs (kun Vegvesen-tellestasjoner)
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📈 Visualisering", "📊 Data", "📋 Statistikk", "📄 Rapport", "🔧 Debug"
            ])

            with tab1:
                st.subheader(title)
                
                # Performance indicator
                if len(df) > 0:
                    data_points = len(df) * len([col for col in df.columns if col.isdigit() or col == "Volume"])
                    st.success(f"✅ {data_points} datapunkter hentet og prosessert")
                
                create_comparison_dashboard(df, point)
                
                # Enhanced insights
                if comparison_mode == "Sammenlign år" and len([col for col in df.columns if col.isdigit()]) >= 2:
                    with st.expander("🔍 Avanserte innsikter og trender"):
                        seasonal_patterns = calculate_seasonal_patterns(df)
                        
                        if seasonal_patterns:
                            st.write("**Sesongmønstre:**")
                            seasonal_df = pd.DataFrame(seasonal_patterns).T
                            seasonal_df = seasonal_df.round(0).astype(int)
                            seasonal_df.columns = ["Vinter", "Vår", "Sommer", "Høst"]
                            st.dataframe(seasonal_df.map(format_number))
                            
                            # Season comparison chart
                            fig_seasonal = px.bar(
                                seasonal_df.reset_index(),
                                x='index',
                                y=['Vinter', 'Vår', 'Sommer', 'Høst'],
                                title="Sesongmønstre per år",
                                labels={'index': 'År', 'value': 'Gjennomsnittlig trafikk'}
                            )
                            st.plotly_chart(fig_seasonal, use_container_width=True)
                            
                        # Growth analysis
                        growth_df = calculate_growth_rates(df)
                        growth_cols = [col for col in growth_df.columns if "Vekst" in col]
                        if growth_cols:
                            avg_growth = growth_df[growth_cols].mean().mean()
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Gjennomsnittlig årlig vekst", f"{avg_growth:.1f}%")
                            with col2:
                                max_growth = growth_df[growth_cols].max().max()
                                st.metric("Høyeste månedsøkning", f"{max_growth:.1f}%")
                            with col3:
                                min_growth = growth_df[growth_cols].min().min()
                                st.metric("Største månedsnedgang", f"{min_growth:.1f}%")

            with tab2:
                st.subheader("📊 Rådata og eksport")
                
                # Enhanced data display
                formatted_df = df.copy()
                numeric_cols = formatted_df.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    formatted_df[col] = formatted_df[col].map(format_number)
                
                # Data quality indicators
                if len(df) > 0:
                    null_count = df.isnull().sum().sum()
                    total_cells = df.shape[0] * df.shape[1]
                    data_quality = ((total_cells - null_count) / total_cells) * 100
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Datakvalitet", f"{data_quality:.1f}%")
                    with col2:
                        st.metric("Rader", f"{len(df):,}")
                    with col3:
                        st.metric("Kolonner", f"{len(df.columns):,}")
                
                st.dataframe(
                    formatted_df,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Enhanced summary metrics
                if comparison_mode == "Sammenlign år":
                    year_columns = [col for col in df.columns if col.isdigit()]
                    if year_columns:
                        latest_year = max(year_columns, key=int)
                        latest_year_int = int(latest_year)
                        total, months_present, days_covered = calculate_yearly_total_from_monthly_averages(df, latest_year_int)
                        avg_per_day = (total / days_covered) if days_covered else None
                        full_year_estimate = (avg_per_day * days_in_year(latest_year_int)) if avg_per_day is not None else None
                        
                        st.subheader("📈 Nøkkelstatistikk")
                        col1, col2, col3, col4 = st.columns(4)
                        
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
                                f"{peak_month}",
                                f"{format_number(peak_value)}"
                            )
                        with col3:
                            variation = (df[latest_year].std() / df[latest_year].mean() * 100)
                            st.metric(
                                "Sesongvariasjon",
                                f"{variation:.1f}%",
                                help="Variasjonskoeffisient mellom måneder"
                            )
                        with col4:
                            if months_present < 12 and full_year_estimate is not None:
                                st.metric(
                                    "Estimert helår",
                                    format_number(full_year_estimate),
                                    help="Ekstrapolerer fra tilgjengelige måneder ved å bruke vektet snitt per dag."
                                )
                            elif avg_per_day is not None:
                                st.metric(
                                    "Gj.snitt per dag",
                                    format_number(avg_per_day),
                                    help="Vektet snitt per dag basert på månedsdata."
                                )

            with tab3:
                st.subheader("📋 Detaljert statistikk og analyse")
                
                # Enhanced basic statistics
                st.write("**Grunnleggende statistikk:**")
                if len(df) > 0:
                    basic_stats = df.describe().round(1)
                    basic_stats.index = [
                        "Antall observasjoner", "Gjennomsnitt", "Standardavvik", 
                        "Minimum", "25% kvartil", "Median (50%)", "75% kvartil", "Maksimum"
                    ]
                    
                    # Add Norwegian number formatting
                    formatted_stats = basic_stats.copy()
                    for col in formatted_stats.columns:
                        if col not in ["Month", "Month Name", "Week"]:
                            formatted_stats[col] = formatted_stats[col].map(format_number)
                    
                    st.dataframe(formatted_stats, use_container_width=True)
                
                # Advanced statistics
                st.write("**Avansert statistikk:**")
                advanced_stats = calculate_additional_statistics(df)
                if not advanced_stats.empty:
                    formatted_advanced = advanced_stats.copy()
                    for col in formatted_advanced.columns:
                        if formatted_advanced[col].dtype in ['int64', 'float64']:
                            formatted_advanced[col] = formatted_advanced[col].map(format_number)
                    st.dataframe(formatted_advanced, use_container_width=True)
                
                # Data distribution analysis
                if comparison_mode == "Sammenlign år":
                    year_columns = [col for col in df.columns if col.isdigit()]
                    if len(year_columns) >= 2:
                        st.write("**Årlig sammenligning:**")
                        
                        comparison_stats = {}
                        for year in year_columns:
                            year_data = df[year].dropna()
                            if len(year_data) > 0:
                                year_total, months_present, _ = calculate_yearly_total_from_monthly_averages(df, int(year))
                                total_label = (
                                    f"{format_number(year_total)} ({months_present}/12 mnd)"
                                    if months_present < 12
                                    else format_number(year_total)
                                )
                                comparison_stats[year] = {
                                    "Årstrafikk": total_label,
                                    "Gjennomsnitt": format_number(year_data.mean()),
                                    "Median": format_number(year_data.median()),
                                    "Standardavvik": format_number(year_data.std()),
                                    "Min": format_number(year_data.min()),
                                    "Maks": format_number(year_data.max())
                                }
                        
                        comparison_df = pd.DataFrame(comparison_stats).T
                        st.dataframe(comparison_df, use_container_width=True)

            with tab4:
                st.subheader("📄 Automatisk rapport og eksport")
                
                # Enhanced report generation
                report_text = create_comparison_report(df, point)
                st.markdown(report_text)
                
                # Report download
                st.download_button(
                    label="📋 Last ned rapport (Markdown)",
                    data=report_text,
                    file_name=f"ryfast_rapport_{point}_{datetime.now().strftime('%Y%m%d')}.md",
                    mime="text/markdown"
                )
                
                # Enhanced export section
                create_export_section(df, point)
                
                # Comparison history
                comparison_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "point": point,
                    "mode": comparison_mode,
                    "parameters": {
                        "years": year_list if comparison_mode == "Sammenlign år" else [year],
                        "months": months if comparison_mode == "Sammenlign måneder" else None,
                        "weeks": weeks if comparison_mode == "Sammenlign uker" else None
                    },
                    "data_points": len(df) if not df.empty else 0
                }
                st.session_state.comparison_history.append(comparison_entry)
                
                # Enhanced comparison history
                if len(st.session_state.comparison_history) > 1:
                    with st.expander("📚 Sammenligningshistorikk"):
                        for i, entry in enumerate(reversed(st.session_state.comparison_history[-10:])):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.write(f"**{i+1}.** {entry['point']}")
                            with col2:
                                st.write(f"{entry['mode']}")
                            with col3:
                                timestamp = entry['timestamp'][:19].replace('T', ' ')
                                st.write(f"{timestamp} ({entry['data_points']} punkter)")

            with tab5:
                st.subheader("🔧 Debug og systeminfo")
                
                # System information
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**API-konfigurasjon:**")
                    st.code(f"""
URL: {URL}
Timeout: {API_TIMEOUT}s
Maks forsøk: {API_MAX_RETRIES}
Samtidig requests: {MAX_CONCURRENT_REQUESTS}
Cache TTL: {API_CACHE_TTL/3600:.1f}h
                    """)
                
                with col2:
                    st.write("**Valgte parametere:**")
                    st.code(f"""
Punkt: {point}
Punkt-IDer: {len(point_ids)} stk
Modus: {comparison_mode}
{'År: ' + str(year_list) if comparison_mode == 'Sammenlign år' else ''}
{'Måneder: ' + str(len(months)) if comparison_mode == 'Sammenlign måneder' else ''}
{'Uker: ' + str(len(weeks)) if comparison_mode == 'Sammenlign uker' else ''}
                    """)
                
                # Performance metrics
                if not df.empty:
                    st.write("**Ytelsesstatistikk:**")
                    data_size = df.memory_usage(deep=True).sum()
                    st.write(f"• DataFrame størrelse: {data_size:,} bytes")
                    st.write(f"• Antall rader: {len(df):,}")
                    st.write(f"• Antall kolonner: {len(df.columns):,}")
                    st.write(f"• Null-verdier: {df.isnull().sum().sum():,}")
                
                # Debug tools
                st.write("**Debug-verktøy:**")
                if st.button("🧪 Test enkelt API-kall"):
                    test_query = QUERY_TEMPLATE.format(point_id=point_ids[0], year=2024)
                    with st.spinner("Tester..."):
                        result = fetch_data_enhanced(test_query, "Debug-test")
                        if result:
                            st.success("✅ API-kall vellykket")
                            st.json(result)
                        else:
                            st.error("❌ API-kall feilet")
                
                if st.button("🗑️ Rens all cache"):
                    st.cache_data.clear()
                    st.success("All cache renset")
                
                if st.button("🔄 Reset session state"):
                    for key in list(st.session_state.keys()):
                        del st.session_state[key]
                    st.success("Session state tilbakestilt")
                    st.rerun()

        except Exception as e:
            st.error(f"❌ En kritisk feil oppstod: {str(e)}")
            logger.exception("Critical error in main process")
            
            with st.expander("🔧 Detaljert feilsøkingsinformasjon"):
                st.code(f"""
Feiltype: {type(e).__name__}
Feilmelding: {str(e)}
Tidspunkt: {datetime.now().isoformat()}

Valgte innstillinger:
- Punkt: {point}
- Modus: {comparison_mode}
- API Timeout: {API_TIMEOUT}s
- Cache aktivert: {enable_caching}

Systeminfo:
- Python-miljø: Streamlit
- Pandas versjon: {pd.__version__}
- Numpy versjon: {np.__version__}
- Requests tilgjengelig: ✅
- openpyxl tilgjengelig: {'✅' if OPENPYXL_AVAILABLE else '❌'}
                """)
                
                st.write("**Foreslåtte løsninger:**")
                st.write("1. Sjekk nettverkstilkobling")
                st.write("2. Prøv færre datapunkter")
                st.write("3. Rens cache og prøv igjen")
                st.write("4. Restart applikasjonen")

    # Enhanced footer with additional information
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
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
            - Fritaksordninger implementert
            
            **2025:**
            - Fortsatt drift og datainnsamling
            - Enhanced app version lansert
            """)
    
    with col2:
        with st.expander("ℹ️ Om dataene"):
            st.markdown("""
            **Datakilde:** Statens vegvesen trafikkdata API
            
            **Datatype:** Gjennomsnittlig døgntrafikk (ÅDT)
            
            **Oppdateringsfrekvens:** Daglig
            
            **Kvalitet:** Data inkluderer kvalitetsparametere og konfidensintervaller
            
            **Dekningsgrad:** Varierer per målepunkt og tidsperiode
            
            **Beregninger:** Totaler er estimert basert på ÅDT × 365
            
            **API-ytelse:** Optimalisert med parallell prosessering og caching
            """)
    
    with col3:
        with st.expander("🆕 Forbedringer i Enhanced Edition"):
            st.markdown("""
            **Nye funksjoner:**
            - ⏱️ Intelligent timeout-håndtering (60s)
            - 🚫 Avbrytelse av lange operasjoner
            - 📊 Detaljerte progress bars
            - 🔄 Parallell API-prosessering
            - 📈 Forbedrede visualiseringer
            - 🐛 Omfattende feilhåndtering
            - 💾 Optimalisert caching
            - 🔧 Debug-verktøy
            - 📊 Ytelsesstatistikk
            - 🎨 Forbedret brukergrensesnitt
            """)
