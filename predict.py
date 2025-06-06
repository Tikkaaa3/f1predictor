import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ------------------ Load Clean Dataset ------------------
df = pd.read_csv("merged_f1_data_with_qualifying_position_and_team_points.csv")

# ------------------ Clean Core Dataset ------------------
df["position"] = df["position"].replace("\\N", np.nan)
df = df[df["position"].notna()]
df["position"] = df["position"].astype(int)

# ------------------ Create Binary Target: Podium or Not ------------------
df["podium"] = (df["position"] <= 3).astype(int)


# ------------------ Convert Fastest Lap Time ------------------
def time_to_seconds(t):
    try:
        if ":" in str(t):
            m, s = t.split(":")
            return float(m) * 60 + float(s)
        return float(t)
    except:
        return None


if df["fastestLapTime"].dtype == object:
    df["fastestLapTime"] = df["fastestLapTime"].apply(time_to_seconds)

# ------------------ Fix and Clean Types ------------------
df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
df["team_points"] = pd.to_numeric(df["team_points"], errors="coerce")
df["position_qualifying"] = pd.to_numeric(df["position_qualifying"], errors="coerce")

# ------------------ Define Features ------------------
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

df = df[features + ["podium"]].dropna()

# ------------------ Encode Categorical Columns ------------------
le_dict = {}
for col in ["raceId", "driverId", "constructorId", "statusId"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

# ------------------ Prepare Inputs and Target ------------------
X = df[features]
y = df["podium"]

# ------------------ Train/Test Split ------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------ Train XGBoost Model ------------------
model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    verbosity=1,
)

model.fit(X_train, y_train)

# ------------------ Evaluate Model ------------------
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {acc:.4f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))
