import matplotlib.pyplot as plt
import missingno as msno
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import train_test_split

# Load preprocessed dataset
df = pd.read_csv("merged_f1_data_with_qualifying_position_and_team_points.csv")


# --- Missing Value Visualization ---
print("Missing values per column:\n")
print(df.isnull().sum())
msno.matrix(df)
plt.title("Missing Value Matrix")
plt.tight_layout()
plt.savefig("output/MissingValue.png")
plt.close()


# --- Correlation Matrix ---
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig("output/CorrelationMatrix.png")
plt.close()

# --- Histograms of Key Features ---
df["position_qualifying"].hist(bins=20)
plt.title("Histogram of Qualifying Positions")
plt.xlabel("Qualifying Position")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("output/HistogramofQualifyingPositions.png")
plt.close()

# Optional: Histogram of other numeric feature
if "team_points" in df.columns:
    df["team_points"].hist(bins=20)
    plt.title("Histogram of Team Points")
    plt.xlabel("Team Points")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("output/TeamPoints.png")
    plt.close()

# --- ROC/AUC Evaluation on binary top 3 prediction ---
if "top3_finish" in df.columns:
    features = [
        col for col in df.columns if col not in ["position_order", "top3_finish"]
    ]
    X = df[features]
    y = df["top3_finish"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    model.fit(X_train, y_train)

    y_score = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic - Top 3 Finish Prediction")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("output/ReceiverOperatingCharactersitic.png")
    plt.close()
