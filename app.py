import pandas as pd
import streamlit as st

# Load datasets
results = pd.read_csv("merged_f1_data.csv")
drivers = pd.read_csv("drivers.csv")
constructors = pd.read_csv("constructors.csv")
races = pd.read_csv("races.csv")
circuits = pd.read_csv("circuits.csv")

# Merge descriptive info to results
df = results.merge(
    drivers[["driverId", "forename", "surname"]], on="driverId", how="left"
)
df = df.merge(
    constructors[["constructorId", "name"]],
    on="constructorId",
    how="left",
    suffixes=("", "_constructor"),
)
df = df.merge(
    races[["raceId", "name", "year", "circuitId"]],
    on="raceId",
    how="left",
    suffixes=("", "_race"),
)
df = df.merge(circuits[["circuitId", "country"]], on="circuitId", how="left")

# Prepare readable columns
df["driver_name"] = df["forename"] + " " + df["surname"]
df.rename(
    columns={"name_constructor": "constructor_name", "name_race": "race_name"},
    inplace=True,
)

# Sidebar filter: year
years = sorted(df["year"].unique())
selected_year = st.sidebar.selectbox("Select Year", years)

# Filter countries for the selected year dynamically
countries_in_year = df[df["year"] == selected_year]["country"].unique()
countries_in_year = sorted(countries_in_year)

selected_country = st.sidebar.selectbox("Select Country", countries_in_year)

# Filter by year and country
filtered_df = df[(df["year"] == selected_year) & (df["country"] == selected_country)]

# Display relevant columns WITHOUT circuit_name (as requested)
display_cols = [
    "year",
    "race_name",
    "country",
    "driver_name",
    "constructor_name",
    "position",
    "points",
    "fastestLapTime",
]
st.dataframe(filtered_df[display_cols].reset_index(drop=True))
