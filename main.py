import kagglehub
import pandas as pd
from kagglehub import KaggleDatasetAdapter

# Load datasets using kagglehub
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

# Merge datasets on keys
merged = results.merge(races, on="raceId", suffixes=("", "_race"))
merged = merged.merge(drivers, on="driverId", suffixes=("", "_driver"))
merged = merged.merge(constructors, on="constructorId", suffixes=("", "_constructor"))

# Merge qualifying position only (use left join to keep all results)
qualifying_subset = qualifying[["raceId", "driverId", "position"]].rename(
    columns={"position": "position_qualifying"}
)
merged = merged.merge(qualifying_subset, on=["raceId", "driverId"], how="left")

merged = merged.merge(circuits, on="circuitId", suffixes=("", "_circuit"))

# Handle missing qualifying positions by filling with a distinct number (e.g., 99)
merged["position_qualifying"] = merged["position_qualifying"].fillna(99).astype(int)

# Columns to drop to clean dataset
drop_cols = [
    # URLs
    "url",
    "url_driver",
    "url_constructor",
    "url_circuit",
    # Qualifying and session IDs not needed if focusing on race results
    "constructorId_qualifying",
    "qualifyId",
    # Qualifying session details (times/dates)
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
    # Driver-specific redundant or descriptive columns
    "positionText",  # usually redundant with numeric positions
    "number_driver",  # redundant with 'number'
    # Columns related to detailed qualifying times removed since we only keep position
    "number_qualifying",
    "q1",
    "q2",
    "q3",
    # Circuit redundant details
    "circuitRef",
    "name_circuit",
    "location",
    "country",
    "lat",
    "lng",
    "alt",
    # Driver personal info
    "driverRef",
    "code",
    "forename",
    "surname",
    "dob",
    "nationality",
    # Constructor personal info
    "constructorRef",
    "name_constructor",
    "nationality_constructor",
    # Race time details
    "time",
    "time_race",
    "round",
]

df = merged.drop(columns=[col for col in drop_cols if col in merged.columns])

# Export clean dataset for ML training
df.to_csv("merged_f1_data_with_qualifying_position.csv", index=False)

# Also save original raw datasets if needed
races.to_csv("races.csv", index=False)
results.to_csv("results.csv", index=False)
drivers.to_csv("drivers.csv", index=False)
constructors.to_csv("constructors.csv", index=False)
qualifying.to_csv("qualifying.csv", index=False)
circuits.to_csv("circuits.csv", index=False)
