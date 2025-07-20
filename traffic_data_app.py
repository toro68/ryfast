import logging
import numpy as np
import requests
import pandas as pd
import plotly.express as px
import streamlit as st
import time
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# API Constants
URL = "https://trafikkdata-api.atlas.vegvesen.no"

# Traffic Point Constants
RYFYLKE_POINT_IDS = ["99040V2725982", "00911V2725983"]
HUNDVAG_POINT_IDS = ["10239V2725979", "62464V2725991", "92743V2726085"]
BYBRUA_POINT_IDS = {
    "Mot nord": ["17949V320695"],
    "Mot sør": ["54184V320694"]
}

# Month Names
MONTH_NAMES = [
    "Januar", "Februar", "Mars", "April", "Mai", "Juni",
    "Juli", "August", "September", "Oktober", "November", "Desember"
]

# Configuration
DEFAULT_YEARS = "2024"
YEAR_RANGE = range(2019, 2025)
API_TIMEOUT = 10

# Define GraphQL query templates
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
              }}
            }}
          }}
        }}
      }}
    }}
  }}
}}
"""

# Ny GraphQL query for ukesdata
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

# API constants
API_MAX_RETRIES = 3
API_RETRY_DELAY = 1
API_CACHE_TTL = 24 * 3600

# Ferde data for 2024
FERDE_DATA_2024 = {
    "Hundvågtunnelen": {
        "total_passeringer": {
            1: 184679, 2: 182709, 3: 199378, 4: 212489, 5: 236726, 
            6: 240047, 7: 233855, 8: 253300, 9: 226734, 10: 217541,
            11: 200790, 12: 189330
        },
        "fritakspasseringer": {
            1: 55214, 2: 56004, 3: 63135, 4: 66046, 5: 75436,
            6: 75886, 7: 75617, 8: 79373, 9: 69799, 10: 67367,
            11: 61442, 12: 60346
        }
    },
    "Ryfylketunnelen": {
        "total_passeringer": {
            1: 131708, 2: 133245, 3: 153886, 4: 158835, 5: 186254,
            6: 189346, 7: 196522, 8: 202069, 9: 172856, 10: 159849,
            11: 145479, 12: 136503
        },
        "fritakspasseringer": {
            1: 5360, 2: 4348, 3: 3833, 4: 5604, 5: 4915,
            6: 5374, 7: 4602, 8: 6224, 9: 5510, 10: 6480,
            11: 5385, 12: 3896
        }
    }
}

# Norske månedsnavn
NORWEGIAN_MONTH_NAMES = {
    1: "Januar", 2: "Februar", 3: "Mars", 4: "April", 
    5: "Mai", 6: "Juni", 7: "Juli", 8: "August",
    9: "September", 10: "Oktober", 11: "November", 12: "Desember"
}

@st.cache_data(ttl=API_CACHE_TTL, show_spinner=False)
def fetch_data(query: str) -> dict | None:
    """Fetch data from the API using the provided GraphQL query."""
    for attempt in range(API_MAX_RETRIES):
        try:
            response = requests.post(
                URL, 
                json={"query": query}, 
                timeout=API_TIMEOUT
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            if attempt == API_MAX_RETRIES - 1:
                logger.error("API-forespørsel feilet: %s", str(e))
                st.error(f"Kunne ikke hente data: {str(e)}")
                return None
            logger.warning(f"Forsøk {attempt + 1} feilet, prøver igjen...")
            time.sleep(API_RETRY_DELAY)

@st.cache_data(ttl=API_CACHE_TTL, show_spinner=False)
def fetch_batch_traffic_data(point_ids: list[str], year: int) -> dict:
    """Fetch traffic data for multiple points in a single batch request."""
    if year < 2019:
        st.warning(f"Data er ikke tilgjengelig før 2019 (valgt år: {year})")
        return {}

    result = {}
    for point_id in point_ids:
        query = QUERY_TEMPLATE.format(point_id=point_id, year=year)
            
        try:
            data = fetch_data(query)
            if data and "data" in data:
                monthly_data = data["data"]["trafficData"]["volume"]["average"]["daily"]["byMonth"]
                if monthly_data:
                    result[point_id] = monthly_data
                else:
                    logger.warning(f"No monthly data for point ID {point_id} in year {year}")
            else:
                logger.warning(f"Failed to fetch data for point ID {point_id} in year {year}")
        except requests.RequestException as e:
            logger.error("Feil ved henting av data for punkt %s, år %s: %s", point_id, year, str(e))
            continue
        except (KeyError, TypeError, ValueError) as e:
            logger.error("Feil ved behandling av data for punkt %s, år %s: %s", point_id, year, str(e))
            continue
        except Exception as e:
            logger.error("Uventet feil ved henting av data for punkt %s, år %s: %s", point_id, year, str(e))
            continue
            
    return result

@st.cache_data(ttl=API_CACHE_TTL, show_spinner=False)
def fetch_weekly_traffic_data(point_ids: list[str], year: int, week_numbers: list[int]) -> dict:
    """Fetch traffic data for specific weeks of a year."""
    if year < 2019:
        st.warning(f"Data er ikke tilgjengelig før 2019 (valgt år: {year})")
        return {}

    result = {}
    
    for week_num in week_numbers:
        # Beregn start og slutt dato for uken
        # Uke 1 starter første mandag i januar (eller 1. januar hvis det er mandag)
        jan_1 = datetime(year, 1, 1)
        days_to_monday = (7 - jan_1.weekday()) % 7
        if days_to_monday == 0 and jan_1.weekday() != 0:
            days_to_monday = 7
        
        first_monday = jan_1 + timedelta(days=days_to_monday)
        week_start = first_monday + timedelta(weeks=week_num-1)
        week_end = week_start + timedelta(days=6)
        
        # Sørg for at datoene er innenfor året
        if week_start.year != year:
            continue
        if week_end.year != year:
            week_end = datetime(year, 12, 31)
            
        from_date = week_start.strftime("%Y-%m-%dT00:00:00+01:00")
        to_date = week_end.strftime("%Y-%m-%dT23:59:59+01:00")
        
        week_data = {}
        for point_id in point_ids:
            query = WEEKLY_QUERY_TEMPLATE.format(
                point_id=point_id, 
                from_date=from_date, 
                to_date=to_date
            )
            
            try:
                data = fetch_data(query)
                if data and "data" in data and data["data"]["trafficData"]:
                    daily_data = data["data"]["trafficData"]["volume"]["byDay"]["edges"]
                    if daily_data:
                        # Beregn gjennomsnitt for uken
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
                        else:
                            logger.warning(f"Ingen gyldig data for punkt {point_id} i uke {week_num}")
                    else:
                        logger.warning(f"Ingen daglig data for punkt {point_id} i uke {week_num}")
                else:
                    logger.warning(f"Feilet å hente data for punkt {point_id} i uke {week_num}")
            except Exception as e:
                logger.error("Feil ved henting av ukesdata for punkt %s, uke %s: %s", point_id, week_num, str(e))
                continue
        
        if week_data:
            result[f"Uke {week_num}"] = week_data
            
    return result

def sum_traffic_data(traffic_data_dict: dict) -> list[float]:
    """Sum traffic data from multiple points."""
    monthly_sums = [0] * 12
    for point_data in traffic_data_dict.values():
        for entry in point_data:
            month = entry["month"]
            try:
                volume = entry["total"]["volume"]["average"]
                if volume is not None:
                    monthly_sums[month - 1] += volume
                else:
                    logger.warning(f"Manglende volumdata for måned {month}")
            except (KeyError, TypeError) as e:
                logger.warning(f"Feil ved lesing av volumdata for måned {month}: {str(e)}")
                continue
    return monthly_sums

def sum_weekly_traffic_data(weekly_data_dict: dict) -> dict:
    """Sum weekly traffic data from multiple points."""
    week_sums = {}
    for week_name, point_data in weekly_data_dict.items():
        total_volume = sum(point_data.values()) if point_data else 0
        week_sums[week_name] = total_volume
    return week_sums

def format_number(x):
    """Format number with thousands separator."""
    if isinstance(x, (int, float)):
        return f"{x:,}".replace(",", " ")
    elif isinstance(x, str):
        try:
            num = float(x)
            return f"{num:,}".replace(",", " ")
        except ValueError:
            return x
    else:
        return str(x)

def process_data_for_years(point_ids: list[str], year_list: list[int]) -> pd.DataFrame:
    """Process data for multiple years."""
    data = {}
    for year in year_list:
        with st.spinner(f"Fetching data for {year}..."):
            traffic_data_dict = fetch_batch_traffic_data(point_ids, year)
            if traffic_data_dict:
                data[year] = sum_traffic_data(traffic_data_dict)
            else:
                st.warning(f"No complete data for all points in year {year}")

    df = pd.DataFrame({"Month": list(range(1, 13))})
    for year in year_list:
        if year in data:
            df[f"{year}"] = data[year]
    df = add_month_names(df)
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].round(0).astype(int)
    
    return df

def process_data_for_months(point_ids, year, months):
    """Process data for selected months of a year."""
    with st.spinner(f"Henter data for {year}..."):
        traffic_data_dict = fetch_batch_traffic_data(point_ids, year)
        if traffic_data_dict:
            data = sum_traffic_data(traffic_data_dict)
            df = pd.DataFrame({
                "Month": list(range(1, 13)),
                f"{year}": data
            })
            df = df[df['Month'].isin(months)]
            df = add_month_names(df)
            
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            df[numeric_columns] = df[numeric_columns].round(0).astype(int)
            
            return df
        else:
            st.warning(f"Ingen data tilgjengelig for år {year}")
            return None

def process_data_for_weeks(point_ids, year, weeks):
    """Process data for selected weeks of a year."""
    with st.spinner(f"Henter ukesdata for {year}..."):
        weekly_data_dict = fetch_weekly_traffic_data(point_ids, year, weeks)
        if weekly_data_dict:
            weekly_sums = sum_weekly_traffic_data(weekly_data_dict)
            
            df = pd.DataFrame([
                {"Week": week_name, "Volume": volume} 
                for week_name, volume in weekly_sums.items()
            ])
            
            # Sorter etter ukenummer
            df['Week_Num'] = df['Week'].str.extract(r'(\d+)').astype(int)
            df = df.sort_values('Week_Num').drop('Week_Num', axis=1).reset_index(drop=True)
            
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            df[numeric_columns] = df[numeric_columns].round(0).astype(int)
            
            return df
        else:
            st.warning(f"Ingen ukesdata tilgjengelig for år {year}")
            return None

def calculate_additional_statistics(df):
    """Calculate additional statistics for the dataset."""
    year_columns = [col for col in df.columns if col not in ["Month", "Month Name", "Week", "Volume"]]
    stats = {}
    
    if "Volume" in df.columns:  # Weekly data
        volume_data = df["Volume"]
        stats["Ukesdata"] = {
            "Høyeste uke": df.loc[volume_data.idxmax(), "Week"],
            "Høyeste volum": volume_data.max(),
            "Laveste uke": df.loc[volume_data.idxmin(), "Week"],
            "Laveste volum": volume_data.min(),
            "Volumspenn": volume_data.max() - volume_data.min(),
            "Variasjonskoeffisient": (volume_data.std() / volume_data.mean() * 100),
        }
    else:  # Monthly/yearly data
        for year in year_columns:
            year_data = df[year]
            stats[year] = {
                "Peak Month": df.loc[year_data.idxmax(), "Month Name"],
                "Peak Volume": year_data.max(),
                "Lowest Month": df.loc[year_data.idxmin(), "Month Name"],
                "Lowest Volume": year_data.min(),
                "Volume Range": year_data.max() - year_data.min(),
                "Coefficient of Variation": (year_data.std() / year_data.mean() * 100),
            }
    
    return pd.DataFrame(stats).T

def add_month_names(df):
    """Add month names to the dataframe."""
    df["Month Name"] = [MONTH_NAMES[i - 1] for i in df["Month"]]
    return df[
        ["Month", "Month Name"]
        + [col for col in df.columns if col not in ["Month", "Month Name"]]
    ]

def create_year_visualization(df: pd.DataFrame, point: str) -> go.Figure:
    """Create visualization for yearly comparison."""
    df_melted = df.melt(
        id_vars=['Month', 'Month Name'],
        var_name='År',
        value_name='Trafikk'
    )
    
    fig = px.line(
        df_melted, 
        x='Month Name',
        y='Trafikk',
        color='År',
        title=f"Trafikkutvikling for {point}",
        labels={
            'Trafikk': 'Gjennomsnittlig døgntrafikk',
            'Month Name': 'Måned'
        }
    )
    
    return fig

def create_month_visualization(df, point, year):
    """Create visualization for a single year."""
    fig = px.bar(
        df,
        x="Month Name",
        y=str(year),
        title=f"Månedlig trafikkvolum for {point} i {year}"
    )
    fig.update_layout(
        xaxis_title="Måned", 
        yaxis_title="Antall passeringer", 
        bargap=0.2
    )
    return fig

def create_week_visualization(df, point, year):
    """Create visualization for weekly data."""
    fig = px.bar(
        df,
        x="Week",
        y="Volume",
        title=f"Ukentlig trafikkvolum for {point} i {year}"
    )
    fig.update_layout(
        xaxis_title="Uke", 
        yaxis_title="Gjennomsnittlig døgntrafikk", 
        bargap=0.2
    )
    # Roter x-akse etiketter for bedre lesbarhet
    fig.update_xaxes(tickangle=45)
    return fig

def analyze_toll_exemptions(df: pd.DataFrame, point: str, year: int) -> tuple[pd.DataFrame, dict] | None:
    """Analyser forskjellen mellom tellepunkt-data og bompengedata."""
    if year == 2024:
        if point not in FERDE_DATA_2024:
            return None
            
        ferde_data = FERDE_DATA_2024[point]
        
        analysis_df = pd.DataFrame({
            "Måned": MONTH_NAMES[:12],
            "Bompasseringer": [ferde_data["total_passeringer"][m] for m in range(1, 13)],
            "Herav fritakspasseringer": [ferde_data["fritakspasseringer"][m] for m in range(1, 13)]
        })
        
        analysis_df["Andel fritakspasseringer"] = (
            analysis_df["Herav fritakspasseringer"] / analysis_df["Bompasseringer"] * 100
        ).round(2)
        
        return analysis_df, {
            "total_passages": analysis_df["Bompasseringer"].sum(),
            "total_exemptions": analysis_df["Herav fritakspasseringer"].sum()
        }
    return None

def main():
    """Main function to run the Streamlit app."""
    st.title("Trafikkdata Visualisering")

    st.markdown("""
    Dette programmet viser trafikkdata for Ryfast-tunnelene og Bybrua. 
    Velg ønskede innstillinger i menyen til venstre.
    """)

    st.sidebar.header("Innstillinger")
    
    point = st.sidebar.selectbox(
        "Velg målepunkt",
        ["Ryfylketunnelen", "Hundvågtunnelen", "Bybrua"],
        key="point_selector"
    )

    comparison_mode = st.sidebar.radio(
        "Velg sammenligningstype",
        ["Sammenlign år", "Sammenlign måneder", "Sammenlign uker"],
        key="comparison_mode_selector"
    )

    if point == "Ryfylketunnelen":
        point_ids = RYFYLKE_POINT_IDS
    elif point == "Hundvågtunnelen":
        point_ids = HUNDVAG_POINT_IDS
    else:
        direction = st.sidebar.selectbox(
            "Velg retning",
            ["Begge retninger", "Mot nord", "Mot sør"],
            key="direction_selector"
        )
        
        if direction == "Begge retninger":
            point_ids = BYBRUA_POINT_IDS["Mot nord"] + BYBRUA_POINT_IDS["Mot sør"]
        else:
            point_ids = BYBRUA_POINT_IDS[direction]

    if comparison_mode == "Sammenlign år":
        year_input = st.sidebar.text_input(
            "Skriv inn år som skal sammenlignes, skilt med komma (f.eks. 2021,2022,2023,2024)",
            DEFAULT_YEARS,
            key="year_input"
        )
        try:
            year_list = [int(year.strip()) for year in year_input.split(",")]
            invalid_years = [year for year in year_list if year < 2019 or year > 2025]
            if invalid_years:
                st.warning(f"Følgende år er ikke tilgjengelige: {', '.join(map(str, invalid_years))}. "
                          "Velg år mellom 2019 og 2024.")
                return
        except ValueError:
            st.error("Ugyldig årsformat. Bruk kun tall skilt med komma.")
            return
    elif comparison_mode == "Sammenlign måneder":
        year = st.sidebar.selectbox(
            "Velg år", 
            list(range(2019, 2024)),
            key="year_selector"
        )
        months = st.sidebar.multiselect(
            "Velg måneder som skal sammenlignes",
            options=list(range(1, 13)),
            default=list(range(1, 13)),
            format_func=lambda x: NORWEGIAN_MONTH_NAMES[x],
            key="month_selector"
        )
    else:  # Sammenlign uker
        year = st.sidebar.selectbox(
            "Velg år", 
            list(range(2019, 2024)),
            key="year_selector_weeks"
        )
        
        # Lag en liste med ukenummer (1-52)
        all_weeks = list(range(1, 53))
        weeks = st.sidebar.multiselect(
            "Velg uker som skal sammenlignes",
            options=all_weeks,
            default=list(range(1, 11)),  # Standard: første 10 uker
            key="week_selector",
            help="Velg hvilke uker du vil sammenligne. Uke 1 starter første mandag i januar."
        )
        
        if not weeks:
            st.warning("Velg minst én uke for sammenligning.")
            return

    if st.sidebar.button("Hent og vis data", key="fetch_button"):
        try:
            if comparison_mode == "Sammenlign år":
                df = process_data_for_years(point_ids, year_list)
                fig = create_year_visualization(df, point)
                
                if 2024 in year_list and point in ["Ryfylketunnelen", "Hundvågtunnelen"]:
                    st.subheader("Analyse av fritakspasseringer")
                    exemption_df, totals = analyze_toll_exemptions(df, point, 2024)
                    if exemption_df is not None:
                        formatted_exemption_df = exemption_df.copy()
                        numeric_columns = [
                            "Bompasseringer", 
                            "Herav fritakspasseringer",
                            "Andel fritakspasseringer"
                        ]
                        for col in numeric_columns:
                            formatted_exemption_df[col] = formatted_exemption_df[col].map(format_number)
                        
                        st.dataframe(formatted_exemption_df, hide_index=True)
                        
                        total_passages = totals["total_passages"]
                        total_exemptions = totals["total_exemptions"]
                        exemption_percentage = (total_exemptions / total_passages * 100).round(1)
                        
                        st.markdown(f"""
                        ### Oppsummering for {point} i 2024
                        - Totalt antall bompasseringer: {format_number(total_passages)}
                        - Herav fritakspasseringer: {format_number(total_exemptions)} ({exemption_percentage}%)
                        """)
                        
                        fig_exemptions = px.line(
                            exemption_df,
                            x="Måned",
                            y="Andel fritakspasseringer",
                            title=f"Fritakspasseringer for {point}",
                            labels={
                                "Andel fritakspasseringer": "Prosent",
                                "Måned": "Måned"
                            }
                        )
                        st.plotly_chart(fig_exemptions)
                
            elif comparison_mode == "Sammenlign måneder":
                df = process_data_for_months(point_ids, year, months)
                if df is not None:
                    fig = create_month_visualization(df, point, year)
                else:
                    st.error("Ingen data tilgjengelig for valgte alternativer.")
                    return
                    
            else:  # Sammenlign uker
                df = process_data_for_weeks(point_ids, year, weeks)
                if df is not None:
                    fig = create_week_visualization(df, point, year)
                else:
                    st.error("Ingen ukesdata tilgjengelig for valgte alternativer.")
                    return

            st.subheader(f"Trafikkvolum for {point}")
            st.plotly_chart(fig)

            formatted_df = df.map(format_number)
            st.dataframe(formatted_df)

            st.subheader("Grunnleggende statistikk")
            statistics = df.describe().round(0).astype(int)
            st.write(statistics.map(format_number))

            st.subheader("Tilleggsstatistikk")
            additional_stats = calculate_additional_statistics(df)
            st.write(additional_stats.map(format_number))

        except requests.RequestException as e:
            st.error(f"Feil ved henting av data: {str(e)}")
            logger.exception("Feil oppstod ved henting av data")
        except ValueError as e:
            st.error(f"Feil ved behandling av data: {str(e)}")
            logger.exception("Feil oppstod ved behandling av data")
        except Exception as e:
            st.error(f"En uventet feil oppstod: {str(e)}")
            logger.exception("En uventet feil oppstod")

    # Add historical timeline
    st.subheader("Tidslinje for Ryfast")
    st.markdown("""
    ### 2019
    - **30. desember:** Ryfylketunnelen åpnet.

    ### 2020
    - **22. april:** Hundvågtunnelen og Eiganestunnelen åpnet.
    - **30. mars:** Planlagt start for bompengeinnkreving.
    - **Oktober:** Bommen på Bybrua ble snudd.

    ### 2021
    - **Februar:** Bompengeinnkreving startet. 
      Ryfast var gratis frem til dette tidspunktet.

    ### 2022
    - **1. juli:** Første takstøkning siden starten av bompengeinnkrevingen.

    ### 2023
    - **3. mai:** Takstøkning.

    ### 2024
    - **8. februar:** Takstøkning.
    - **1. juli:** Takstøkning.
    """)

    # Add explanations for statistical terms
    st.subheader("Forklaringer for statistikktermene")
    st.markdown("""
    - **count**: Antall ikke-NA/NaN (ikke-null) observasjoner.
    - **mean**: Gjennomsnittlig verdi av dataene.
    - **std**: Standardavvik, som måler spredningen i dataene.
    - **min**: Minimumsverdien i dataene.
    - **25%**: Første kvartil, verdien under hvilken 25% av dataene faller.
    - **50%**: Median, verdien som deler datasettet i to like store deler.
    - **75%**: Tredje kvartil, verdien under hvilken 75% av dataene faller.
    - **max**: Maksimumsverdien i dataene.
    """)

if __name__ == "__main__":
    main()