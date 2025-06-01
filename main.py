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

# Now do the same merge steps as before:
merged = results.merge(races, on="raceId", suffixes=("", "_race"))
merged = merged.merge(drivers, on="driverId", suffixes=("", "_driver"))
merged = merged.merge(constructors, on="constructorId", suffixes=("", "_constructor"))
merged = merged.merge(
    qualifying, on=["raceId", "driverId"], how="left", suffixes=("", "_qualifying")
)
df = merged.merge(circuits, on="circuitId", suffixes=("", "_circuit"))

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
    # From race results (suggested from the dataset you shared)
    "number_qualifying",
    "position_qualifying",
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


df = df.drop(columns=[col for col in drop_cols if col in df.columns])

df.to_csv("merged_f1_data.csv", index=False)
races.to_csv("races.csv", index=False)
results.to_csv("results.csv", index=False)
drivers.to_csv("drivers.csv", index=False)
constructors.to_csv("constructors.csv", index=False)
qualifying.to_csv("qualifying.csv", index=False)
circuits.to_csv("circuits.csv", index=False)
