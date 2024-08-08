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


def fetch_data(query):
    """Fetch data from the API using the provided GraphQL query."""
    try:
        response = requests.post(URL, json={"query": query}, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        logger.error("API request failed: %s", str(e))
        return None


def fetch_traffic_data(point_ids, year):
    """Fetch traffic data for given point IDs and year."""
    data_per_point = {}
    for point_id in point_ids:
        query = QUERY_TEMPLATE.format(point_id=point_id, year=year)
        data = fetch_data(query)
        if data and "data" in data:
            monthly_data = data["data"]["trafficData"]["volume"]["average"]["daily"][
                "byMonth"
            ]
            if monthly_data:
                data_per_point[point_id] = monthly_data
            else:
                logger.warning(
                    "No monthly data for point ID %s in year %s", point_id, year
                )
        else:
            logger.warning(
                "Failed to fetch data for point ID %s in year %s", point_id, year
            )
    return data_per_point


def sum_traffic_data(traffic_data_dict):
    """Sum traffic data from multiple points."""
    monthly_sums = [0] * 12
    for point_data in traffic_data_dict.values():
        for entry in point_data:
            month = entry["month"]
            volume = entry["total"]["volume"]["average"]
            monthly_sums[month - 1] += volume
    return monthly_sums


def process_data_for_points(point_ids, year_list):
    """Process data for given point IDs and years."""
    data = {}
    for year in year_list:
        with st.spinner(f"Fetching data for {year}..."):
            traffic_data_dict = fetch_traffic_data(point_ids, year)
            if traffic_data_dict:
                data[year] = sum_traffic_data(traffic_data_dict)
            else:
                st.warning(f"No complete data for all points in year {year}")

    months = list(range(1, 13))
    df = pd.DataFrame({"Month": months})
    for year in year_list:
        if year in data:
            df[f"Volume for {year}"] = data[year]

    df = df.round(0).astype(int)
    return df


def format_number(x):
    """
    Format number with thousands separator.

    Args:
    x: The value to format (can be a number or a string)

    Returns:
    str: Formatted string with space as thousands separator
    """
    if isinstance(x, (int, float)):
        return f"{x:,}".replace(",", " ")
    elif isinstance(x, str):
        try:
            num = float(x)
            return f"{num:,}".replace(",", " ")
        except ValueError:
            return x  # Return the original string if it can't be converted to a number
    else:
        return str(x)  # For any other type, convert to string


def create_pdf(df, statistics, point):
    """Create a PDF report of the traffic data."""
    buffer = io.BytesIO()
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Traffic Data Visualization", ln=True, align="C")
    pdf.cell(
        200,
        10,
        txt=f"Summed monthly average daily traffic volume for {point}",
        ln=True,
        align="C",
    )

    pdf.ln(10)
    pdf.cell(200, 10, txt="Data:", ln=True)
    for col in df.columns:
        pdf.cell(
            200, 10, txt=f"{col}: {', '.join(df[col].astype(str).tolist())}", ln=True
        )

    pdf.ln(10)
    pdf.cell(200, 10, txt="Statistics:", ln=True)
    for col in statistics.columns:
        pdf.cell(
            200,
            10,
            txt=f"{col}: {', '.join(statistics[col].astype(str).tolist())}",
            ln=True,
        )

    pdf.output(buffer)
    buffer.seek(0)
    return buffer.getvalue()  # Return PDF as bytes


def process_data_for_years(point_ids, year_list):
    """
    Process data for multiple years.
    
    Args:
    point_ids (list): List of traffic registration point IDs.
    year_list (list): List of years to process.
    
    Returns:
    pandas.DataFrame: Processed data for the specified years.
    """
    data = {}
    for year in year_list:
        with st.spinner(f"Fetching data for {year}..."):
            traffic_data_dict = fetch_traffic_data(point_ids, year)
            if traffic_data_dict:
                data[year] = sum_traffic_data(traffic_data_dict)
            else:
                st.warning(f"No complete data for all points in year {year}")

    df = pd.DataFrame({"Month": list(range(1, 13))})
    for year in year_list:
        if year in data:
            df[f"{year}"] = data[year]
    df = add_month_names(df)  # Add month names
    
    # Round numeric columns to integers
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].round(0).astype(int)
    
    return df

def process_data_for_months(point_ids, year, months):
    """
    Process data for selected months of a year.
    
    Args:
    point_ids (list): List of traffic registration point IDs.
    year (int): Year to process.
    months (list): List of months to process.
    
    Returns:
    pandas.DataFrame: Processed data for the specified months.
    """
    with st.spinner(f"Fetching data for {year}..."):
        traffic_data_dict = fetch_traffic_data(point_ids, year)
        if traffic_data_dict:
            data = sum_traffic_data(traffic_data_dict)
            df = pd.DataFrame({
                "Month": list(range(1, 13)),
                f"{year}": data
            })
            df = df[df['Month'].isin(months)]
            df = add_month_names(df)
            
            # Round numeric columns to integers
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            df[numeric_columns] = df[numeric_columns].round(0).astype(int)
            
            return df
        else:
            st.warning(f"No complete data for all points in year {year}")
            return None

def calculate_additional_statistics(df):
    """Calculate additional statistics for the dataset."""
    year_columns = [col for col in df.columns if col not in ["Month", "Month Name"]]
    stats = {}
    for year in year_columns:
        year_data = df[year]
        stats[year] = {
            "Peak Month": df.loc[year_data.idxmax(), "Month Name"],
            "Peak Volume": year_data.max(),
            "Lowest Month": df.loc[year_data.idxmin(), "Month Name"],
            "Lowest Volume": year_data.min(),
            "Volume Range": year_data.max() - year_data.min(),
            "Coefficient of Variation": year_data.std() / year_data.mean() * 100,  # as percentage
        }
    return pd.DataFrame(stats).T

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
    return df[
        ["Month", "Month Name"]
        + [col for col in df.columns if col not in ["Month", "Month Name"]]
    ]


def create_year_visualization(df, point):
    """
    Create visualization for multiple years.

    Args:
    df (pandas.DataFrame): Processed traffic data.
    point (str): Name of the traffic registration point.

    Returns:
    plotly.graph_objs._figure.Figure: The created visualization.
    """
    df_melted = df.melt(
        id_vars=["Month", "Month Name"], var_name="Year", value_name="Volume"
    )
    fig = px.bar(
        df_melted,
        x="Month Name",
        y="Volume",
        color="Year",
        barmode="group",
        title=f"Monthly Traffic Volume for {point} Across Years",
    )
    fig.update_xaxes(categoryorder="array", categoryarray=df["Month Name"])
    return fig


def create_month_visualization(df, point, year):
    """
    Create visualization for a single year.

    Args:
    df (pandas.DataFrame): Processed traffic data.
    point (str): Name of the traffic registration point.
    year (int): Year of the data.

    Returns:
    plotly.graph_objs._figure.Figure: The created visualization.
    """
    fig = px.bar(
        df,
        x="Month Name",
        y=str(year),
        title=f"Monthly Traffic Volume for {point} in {year}",
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
        months = st.sidebar.multiselect(
            "Select months to compare",
            options=list(range(1, 13)),
            default=list(range(1, 13)),
            format_func=lambda x: calendar.month_name[x],
        )

    if st.sidebar.button("Fetch and display data"):
        try:
            if comparison_mode == "Compare Years":
                df = process_data_for_years(point_ids, year_list)
                fig = create_year_visualization(df, point)
            else:
                df = process_data_for_months(point_ids, year, months)
                if df is not None:
                    fig = create_month_visualization(df, point, year)
                else:
                    st.error("No data available for the selected options.")
                    return

            st.subheader(f"Traffic volume for {point}")
            st.plotly_chart(fig)

            formatted_df = df.map(format_number)
            st.dataframe(formatted_df)

            st.subheader("Basic Statistics")
            statistics = df.describe().round(0).astype(int)
            st.write(statistics.map(format_number))

            st.subheader("Additional Statistics")
            additional_stats = calculate_additional_statistics(df)
            st.write(additional_stats.map(format_number))

            pdf_bytes = create_pdf(
                formatted_df, statistics.applymap(format_number), point
            )
            st.download_button(
                label="Download PDF Report",
                data=pdf_bytes,
                file_name="traffic_data_report.pdf",
                mime="application/pdf",
                key=f"download_button_{point}",
            )

        except requests.RequestException as e:
            st.error(f"Error fetching data: {str(e)}")
            logger.exception("Error occurred while fetching data")
        except ValueError as e:
            st.error(f"Error processing data: {str(e)}")
            logger.exception("Error occurred while processing data")
        except Exception as e:
            # We use a broad exception here to catch any unexpected errors
            # and provide a user-friendly error message while logging the details.
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

if __name__ == "__main__":
    main()
