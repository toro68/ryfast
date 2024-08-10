import calendar
import logging
from fpdf import FPDF
import numpy as np
import requests
import pandas as pd
import plotly.express as px
import streamlit as st
import io

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Define the GraphQL query template
QUERY_TEMPLATE = """
{{
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

# Define traffic registration points
RYFYLKE_POINT_IDS = ["99040V2725982", "00911V2725983"]
HUNDVAG_POINT_IDS = ["10239V2725979", "62464V2725991", "92743V2726085"]
BYBRUA_POINT_IDS = ["17949V320695"]

# Define the API endpoint
URL = "https://trafikkdata-api.atlas.vegvesen.no"

def fetch_traffic_data(point_id, year):
    """Fetch traffic data for a given point ID and year."""
    query = QUERY_TEMPLATE.format(point_id=point_id, year=year)
    try:
        response = requests.post(URL, json={"query": query}, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        logger.error(f"API request failed for point ID {point_id} in year {year}: {str(e)}")
        return None

def process_traffic_data(traffic_data_dict):
    """Process the traffic data dictionary."""
    monthly_data = traffic_data_dict["data"]["trafficData"]["volume"]["average"]["daily"]["byMonth"]
    if monthly_data:
        return pd.DataFrame(monthly_data)
    else:
        logger.warning("No monthly data found for the given point ID and year.")
        return None

def fetch_and_process_data(point_ids, year):
    """Fetch and process data for multiple point IDs."""
    data = []
    for point_id in point_ids:
        traffic_data = fetch_traffic_data(point_id, year)
        if traffic_data:
            processed_data = process_traffic_data(traffic_data)
            if processed_data is not None:
                processed_data["Point ID"] = point_id
                data.append(processed_data)
    if data:
        return pd.concat(data)
    else:
        return None

def create_visualization(df, point, year, comparison_mode):
    """Create visualization based on the data and comparison mode."""
    if comparison_mode == "Compare Years":
        df_melted = df.melt(id_vars=["Month", "Point ID"], var_name="Year", value_name="Volume")
        fig = px.bar(
            df_melted,
            x="Month",
            y="Volume",
            color="Year",
            barmode="group",
            title=f"Monthly Traffic Volume for {point} Across Years",
            facet_col="Point ID",
            facet_col_wrap=2,
        )
        fig.update_xaxes(categoryorder="array", categoryarray=df["Month"])
    else:  # Compare Months
        fig = px.bar(
            df,
            x="Month",
            y=str(year),
            title=f"Monthly Traffic Volume for {point} in {year}",
            facet_col="Point ID",
            facet_col_wrap=2,
        )
    fig.update_layout(xaxis_title="Month", yaxis_title="Traffic Volume", bargap=0.2)
    return fig

def main():
    """Main function to run the Streamlit app."""
    st.title("Traffic Data Visualization")

    st.sidebar.header("Settings")
    point = st.sidebar.selectbox(
        "Select traffic registration point",
        ["Ryfylketunnelen", "Hundvågtunnelen", "Bybrua"],
    )

    comparison_mode = st.sidebar.radio(
        "Choose comparison mode", ["Compare Years", "Compare Months"]
    )

    if point == "Ryfylketunnelen":
        point_ids = RYFYLKE_POINT_IDS
    elif point == "Hundvågtunnelen":
        point_ids = HUNDVAG_POINT_IDS
    else:
        point_ids = BYBRUA_POINT_IDS

    if comparison_mode == "Compare Years":
        year_input = st.sidebar.text_input(
            "Enter years to compare, separated by commas (e.g., 2024,2023,2022)",
            "2024,2023,2022",
        )
        year_list = [int(year.strip()) for year in year_input.split(",")]
    else:  # Compare Months
        year = st.sidebar.selectbox("Select year", range(2022, 2025))

    if st.sidebar.button("Fetch and display data"):
        try:
            if comparison_mode == "Compare Years":
                df = pd.DataFrame()
                for year in year_list:
                    df_year = fetch_and_process_data(point_ids, year)
                    if df_year is not None:
                        df_year["Year"] = year
                        df = pd.concat([df, df_year])
                if not df.empty:
                    df = add_month_names(df)
                    fig = create_visualization(df, point, year, comparison_mode)
                    st.subheader(f"Traffic volume for {point}")
                    st.plotly_chart(fig)
                    st.dataframe(df)
                else:
                    st.warning("No data available for the selected years.")
            else:  # Compare Months
                df = fetch_and_process_data(point_ids, year)
                if df is not None:
                    df = add_month_names(df)
                    fig = create_visualization(df, point, year, comparison_mode)
                    st.subheader(f"Traffic volume for {point} in {year}")
                    st.plotly_chart(fig)
                    st.dataframe(df)
                else:
                    st.warning(f"No data available for {point} in {year}.")
        except requests.RequestException as e:
            st.error(f"Error fetching data: {str(e)}")
            logger.exception("Error occurred while fetching data")
        except ValueError as e:
            st.error(f"Error processing data: {str(e)}")
            logger.exception("Error occurred while processing data")
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")
            logger.exception("An unexpected error occurred")


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
    - **Februar:** Bompengeinnkreving startet. Ryfast var gratis frem til dette tidspunktet.

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


def add_month_names(df):
    """
    Add month names to the dataframe.

    Args:
    df (pandas.DataFrame): Input dataframe with 'Month' column.

    Returns:
    pandas.DataFrame: Dataframe with added 'Month Name' column.
    """
    month_names = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]
    df["Month Name"] = [month_names[i - 1] for i in df["Month"]]
    return df

if __name__ == "__main__":
    main()