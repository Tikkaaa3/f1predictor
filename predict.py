import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ------------------ Load Clean Dataset ------------------
df = pd.read_csv("merged_f1_data.csv")

# ------------------ Clean Core Dataset ------------------
df["position"] = df["position"].replace("\\N", np.nan)
df = df[df["position"].notna()]
df["position"] = df["position"].astype(int)


# ------------------ Map Finishing Position to Class ------------------
def position_to_class(pos):
    if pos <= 3:
        return 0  # Podium
    elif pos <= 10:
        return 1  # Points finish
    elif pos <= 15:
        return 2  # Finished no points
    else:
        return 3  # DNF / Low positions


df["position_class"] = df["position"].apply(position_to_class)


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
df["qualifying_position"] = pd.to_numeric(df["qualifying_position"], errors="coerce")

# ------------------ Drop Rows with Any Missing Data ------------------
features = [
    "raceId",
    "driverId",
    "constructorId",
    "grid",
    "fastestLapTime",
    "rank",
    "laps",
    "statusId",
    "qualifying_position",
    "team_points",
    "circuit_type",
]

df = df[features + ["position_class"]].dropna()

# ------------------ Encode Categorical Columns ------------------
le_dict = {}
for col in ["raceId", "driverId", "constructorId", "statusId"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

# Encode circuit_type (permanent/street)
df["circuit_type"] = LabelEncoder().fit_transform(df["circuit_type"])

# ------------------ Model Inputs ------------------
X = df[features]
y = df["position_class"]

# ------------------ Train/Test Split ------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------ Train XGBoost Model ------------------
model = xgb.XGBClassifier(
    objective="multi:softmax", num_class=y.nunique(), eval_metric="merror", verbosity=1
)

model.fit(X_train, y_train)

# ------------------ Evaluate ------------------
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\nAccuracy: {acc:.4f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))
