# f1predictor

XGBoost:
Advantages of XGboost

    Scalable and efficient for large datasets with millions of records
    Supports parallel processing and GPU acceleration for faster training
    Offers customizable parameters and regularization for fine-tuning
    Includes feature importance analysis for better insights and selection
    Trusted by data scientists across multiple programming languages

Disadvantages of XGBoost

    XGBoost can be computationally intensive, making it less ideal for resource-constrained systems.
    It may be sensitive to noise or outliers, requiring careful data preprocessing.
    Prone to overfitting, especially on small datasets or with too many trees.
    Offers feature importance, but overall model interpretability is limited compared to simpler methods which is an issue in fields like healthcare or finance.

Two Main Pages:
1️⃣ Dashboard Page

    Team Scoreboard:

        Total wins per constructor (season / all-time)

        Championship points summary

        Visuals: bar charts, line trends over years

    Driver Scoreboard:

        Total wins per driver

        Podiums, pole positions

        Career progression charts

    Interactive filters:

        Season selector (to view specific years)

        Circuit selector (see wins by circuit)

        Driver / Team search

2️⃣ Race Simulator Page

    Inputs:

        Select Circuit & Year (optional pre-fill)

        Input all drivers with:

            Qualifying positions

            Teams

        Optional weather condition input

    Outputs:

        Predicted finishing order for all drivers (sorted list)

        Points earned per driver/team based on predicted results

        Podium visualization

        Driver-specific feature importance / SHAP explanation

    Extra:

        Compare simulation with real past race if data available

🔧 Technical Approach Summary

    XGBoost regression model predicting finishing positions per driver per race

    Dataset cleaned & enriched with driver/team/circuit features

    Streamlit multi-page app:

        Page 1: Stats & scoreboards with interactive charts

        Page 2: Race simulator with full race prediction & explanations
