import joblib
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


# Helper function for safe label encoding
def safe_transform(le, value):
    if value in le.classes_:
        return le.transform([value])[0]
    else:
        return 0  # fallback index if unseen


# ------------------ Sidebar Input ------------------
st.sidebar.title("🏁 F1 Podium Predictor Input")

selected_race = st.sidebar.selectbox("Race", sorted(df["race_name"].unique()))
selected_driver = st.sidebar.selectbox("Driver", sorted(df["driver_name"].unique()))
selected_constructor = st.sidebar.selectbox(
    "Constructor", sorted(df["constructor_name"].unique())
)
fastest_lap = st.sidebar.number_input(
    "Fastest Lap Time (seconds)", min_value=60.0, max_value=120.0, value=90.0
)
qualifying_position = st.sidebar.slider("Qualifying Position", 1, 20, 10)
team_points = st.sidebar.number_input(
    "Team Points", min_value=0.0, max_value=800.0, value=100.0
)

# Assume statusId = 1 means "Finished Normally"
normal_status_id = 1

# ------------------ Match IDs ------------------
race_id = df[df["race_name"] == selected_race]["raceId"].iloc[0]
driver_id = df[df["driver_name"] == selected_driver]["driverId"].iloc[0]
constructor_id = df[df["constructor_name"] == selected_constructor][
    "constructorId"
].iloc[0]

# ------------------ Encode Using LabelEncoders ------------------
encoded_input = {
    "raceId": safe_transform(le_dict["raceId"], race_id),
    "driverId": safe_transform(le_dict["driverId"], driver_id),
    "constructorId": safe_transform(le_dict["constructorId"], constructor_id),
    # "grid": 10,  # fixed default grid position (since UI removed)
    "fastestLapTime": fastest_lap,
    # "rank": 10,  # fixed default rank (since UI removed)
    "statusId": safe_transform(le_dict["statusId"], normal_status_id),
    "position_qualifying": qualifying_position,
    "team_points": team_points,
}

# ------------------ Create DataFrame and Predict ------------------
features = [
    "raceId",
    "driverId",
    "constructorId",
    # "grid",
    "fastestLapTime",
    # "rank",
    "statusId",
    "position_qualifying",
    "team_points",
]

X_app = pd.DataFrame([encoded_input])[features]
proba = model.predict_proba(X_app)[0][1]
result = proba >= 0.5

# ------------------ UI Output ------------------
st.title("🏎️ Formula 1 Podium Predictor")
st.write("### Prediction Result")
if result:
    st.success(f"✅ Prediction: Podium Finish (Confidence: {proba*100:.2f}%)")
else:
    st.error(f"🚫 Prediction: Not a Podium Finish (Confidence: {proba*100:.2f}%)")

# ------------------ Debug: Show Input Data ------------------
if st.checkbox("Show input data"):
    st.dataframe(pd.DataFrame([encoded_input]))
