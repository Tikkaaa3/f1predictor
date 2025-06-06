import joblib
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb

# ------------------ Load Model and Encoders ------------------
model = xgb.XGBClassifier()
model.load_model("xgb_model.json")
le_dict = joblib.load("label_encoders.pkl")

# ------------------ Load Data ------------------
df = pd.read_csv("merged_f1_data_with_qualifying_position_and_team_points.csv")
drivers = pd.read_csv("drivers.csv")
constructors = pd.read_csv("constructors.csv")
races = pd.read_csv("races.csv")

# ------------------ Add Human-Readable Names ------------------
df = df.merge(drivers[["driverId", "forename", "surname"]], on="driverId", how="left")
df["driver_name"] = df["forename"] + " " + df["surname"]
df = df.merge(
    constructors[["constructorId", "name"]].rename(
        columns={"name": "constructor_name"}
    ),
    on="constructorId",
    how="left",
)
df = df.merge(
    races[["raceId", "name"]].rename(columns={"name": "race_name"}),
    on="raceId",
    how="left",
)

# ------------------ Sidebar Input ------------------
st.sidebar.title("🏁 F1 Podium Predictor Input")

selected_race = st.sidebar.selectbox("Race", sorted(df["race_name"].unique()))
selected_driver = st.sidebar.selectbox("Driver", sorted(df["driver_name"].unique()))
selected_constructor = st.sidebar.selectbox(
    "Constructor", sorted(df["constructor_name"].unique())
)
grid_position = st.sidebar.slider("Grid Position", 1, 20, 10)
fastest_lap = st.sidebar.number_input(
    "Fastest Lap Time (seconds)", min_value=60.0, max_value=120.0, value=90.0
)
rank = st.sidebar.slider("Rank", 1, 20, 5)
status_id = st.sidebar.number_input("Status ID", min_value=1, max_value=100, value=1)
qualifying_position = st.sidebar.slider("Qualifying Position", 1, 20, 10)
team_points = st.sidebar.number_input(
    "Team Points", min_value=0.0, max_value=800.0, value=100.0
)

# ------------------ Match IDs ------------------
race_id = df[df["race_name"] == selected_race]["raceId"].iloc[0]
driver_id = df[df["driver_name"] == selected_driver]["driverId"].iloc[0]
constructor_id = df[df["constructor_name"] == selected_constructor][
    "constructorId"
].iloc[0]

st.write(f"### Selected IDs")
st.write(f"Race ID: {race_id}")
st.write(f"Driver ID: {driver_id}")
st.write(f"Constructor ID: {constructor_id}")

# ------------------ Show Encoder Classes ------------------
st.write("### Label Encoder Classes Snapshot")
for key in ["raceId", "driverId", "constructorId", "statusId"]:
    st.write(f"{key}: {le_dict[key].classes_[:10]} ... (showing first 10)")

# ------------------ Encode Using LabelEncoders ------------------
try:
    encoded_race = le_dict["raceId"].transform([race_id])[0]
except ValueError:
    st.error(f"Race ID {race_id} not in label encoder classes.")
    encoded_race = -1

try:
    encoded_driver = le_dict["driverId"].transform([driver_id])[0]
except ValueError:
    st.error(f"Driver ID {driver_id} not in label encoder classes.")
    encoded_driver = -1

try:
    encoded_constructor = le_dict["constructorId"].transform([constructor_id])[0]
except ValueError:
    st.error(f"Constructor ID {constructor_id} not in label encoder classes.")
    encoded_constructor = -1

if status_id in le_dict["statusId"].classes_:
    encoded_status = le_dict["statusId"].transform([status_id])[0]
else:
    st.warning(f"Status ID {status_id} not found in encoder classes, defaulting to 0.")
    encoded_status = 0

encoded_input = {
    "raceId": encoded_race,
    "driverId": encoded_driver,
    "constructorId": encoded_constructor,
    "grid": grid_position,
    "fastestLapTime": fastest_lap,
    "rank": rank,
    "statusId": encoded_status,
    "position_qualifying": qualifying_position,
    "team_points": team_points,
}

# ------------------ Show Encoded Input ------------------
st.write("### Encoded Input Features for Prediction")
st.json(encoded_input)

# ------------------ Create DataFrame and Predict ------------------
features = [
    "raceId",
    "driverId",
    "constructorId",
    "grid",
    "fastestLapTime",
    "rank",
    "statusId",
    "position_qualifying",
    "team_points",
]

X_app = pd.DataFrame([encoded_input])[features]

proba = model.predict_proba(X_app)[0][1]
result = proba >= 0.5

# ------------------ Output Prediction and Probability ------------------
st.title("🏎️ Formula 1 Podium Predictor")
st.write("### Prediction Result")
st.write(f"Predicted probability of podium finish: {proba:.4f}")

if result:
    st.success(f"✅ Prediction: Podium Finish (Confidence: {proba*100:.2f}%)")
else:
    st.error(f"🚫 Prediction: Not a Podium Finish (Confidence: {proba*100:.2f}%)")

# ------------------ Optional: Show input DataFrame ------------------
if st.checkbox("Show raw input DataFrame"):
    st.dataframe(X_app)
