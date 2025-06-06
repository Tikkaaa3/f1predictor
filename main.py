import kagglehub
import pandas as pd
from kagglehub import KaggleDatasetAdapter

# ------------------ Load datasets ------------------
races = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "rohanrao/formula-1-world-championship-1950-2020",
    "races.csv",
)
results = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "rohanrao/formula-1-world-championship-1950-2020",
    "results.csv",
)
drivers = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "rohanrao/formula-1-world-championship-1950-2020",
    "drivers.csv",
)
constructors = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "rohanrao/formula-1-world-championship-1950-2020",
    "constructors.csv",
)
qualifying = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "rohanrao/formula-1-world-championship-1950-2020",
    "qualifying.csv",
)
circuits = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "rohanrao/formula-1-world-championship-1950-2020",
    "circuits.csv",
)

# ------------------ Merge base datasets ------------------
merged = results.merge(races, on="raceId", suffixes=("", "_race"))
merged = merged.merge(drivers, on="driverId", suffixes=("", "_driver"))
merged = merged.merge(constructors, on="constructorId", suffixes=("", "_constructor"))

# ------------------ Add qualifying position ------------------
qualifying_subset = qualifying[["raceId", "driverId", "position"]].rename(
    columns={"position": "position_qualifying"}
)
merged = merged.merge(qualifying_subset, on=["raceId", "driverId"], how="left")
merged["position_qualifying"] = merged["position_qualifying"].fillna(99).astype(int)


def classify_circuit_type(name):
    street_circuits = [
        "Monaco",
        "Singapore",
        "Baku",
        "Melbourne",
        "Montréal",
        "Azerbaijan",
    ]
    for street in street_circuits:
        if street.lower() in name.lower():
            return "street"
    return "permanent"


circuits["circuit_type"] = circuits["name"].apply(classify_circuit_type)

# Now merge
merged = merged.merge(
    circuits[["circuitId", "circuit_type"]], on="circuitId", how="left"
)

# ------------------ Add circuit info ------------------
# merged = merged.merge(circuits, on="circuitId", suffixes=("", "_circuit"))

# ------------------ Calculate team_points ------------------
results_team_points = results.merge(races[["raceId", "year", "date"]], on="raceId")
results_team_points["date"] = pd.to_datetime(results_team_points["date"])
results_team_points = results_team_points.sort_values(
    by=["constructorId", "year", "date"]
)

# Cumulative points for team up to and including each race
results_team_points["team_points"] = results_team_points.groupby(
    ["constructorId", "year"]
)["points"].cumsum()

# Extract last team points per race
team_points_per_race = results_team_points[
    ["raceId", "constructorId", "team_points"]
].drop_duplicates(subset=["raceId", "constructorId"], keep="last")

# Merge into main dataset
merged = merged.merge(team_points_per_race, on=["raceId", "constructorId"], how="left")

# ------------------ Clean unnecessary columns ------------------
drop_cols = [
    "url",
    "url_driver",
    "url_constructor",
    "url_circuit",
    "constructorId_qualifying",
    "qualifyId",
    "fp1_date",
    "fp1_time",
    "fp2_date",
    "fp2_time",
    "fp3_date",
    "fp3_time",
    "quali_date",
    "quali_time",
    "sprint_date",
    "sprint_time",
    "positionText",
    "number_driver",
    "number_qualifying",
    "q1",
    "q2",
    "q3",
    "circuitRef",
    "name_circuit",
    "location",
    "country",
    "lat",
    "lng",
    "alt",
    "driverRef",
    "code",
    "forename",
    "surname",
    "dob",
    "nationality",
    "constructorRef",
    "name_constructor",
    "nationality_constructor",
    "time",
    "time_race",
    "round",
]

df = merged.drop(columns=[col for col in drop_cols if col in merged.columns])

# ------------------ Export cleaned dataset ------------------
df.to_csv("merged_f1_data_with_qualifying_position_and_team_points.csv", index=False)

# Save raw files
races.to_csv("races.csv", index=False)
results.to_csv("results.csv", index=False)
drivers.to_csv("drivers.csv", index=False)
constructors.to_csv("constructors.csv", index=False)
qualifying.to_csv("qualifying.csv", index=False)
circuits.to_csv("circuits.csv", index=False)
