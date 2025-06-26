"""
Evaluate Prophet models for each skill using a 6-month hold-out test set.
Computes one MAPE per skill and prints a summary table to console.
"""
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error
from pathlib import Path

# Configuration
from constants import SKILLS
DATA_DIR = Path("data")


def evaluate_skill(skill: str) -> float:
    """
    Load time-series data for a skill, train a Prophet model on all but the last 6 months,
    forecast for those 6 months, and return the MAPE.
    """
    # Load data
    path = DATA_DIR / f"{skill}.csv"
    df = pd.read_csv(path, parse_dates=["ds"])  # expects columns 'ds','y'
    df = df.sort_values("ds")

    # Define cutoff: last date minus 6 months
    last_date = df["ds"].max()
    cutoff = last_date - pd.DateOffset(months=6)

    # Split
    train_df = df[df["ds"] <= cutoff].copy()
    test_df  = df[df["ds"] >  cutoff].copy()

    if train_df.empty or test_df.empty:
        raise ValueError(f"Not enough data for skill '{skill}' to perform evaluation.")

    # Train model
    m = Prophet(yearly_seasonality=True)
    m.fit(train_df)

    # Forecast for test dates
    future = test_df[["ds"]].rename(columns={"ds": "ds"})
    forecast = m.predict(future)

    # Align predictions with actuals
    y_true = test_df["y"].values
    y_pred = forecast["yhat"].values

    # Compute MAPE
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return mape


def main():
    # Evaluate each skill and collect results
    results = []
    for skill in SKILLS:
        try:
            mape = evaluate_skill(skill)
            results.append({"skill": skill, "MAPE": mape})
            print(f"✅ {skill}: MAPE = {mape:.2%}")
        except Exception as e:
            print(f"❌ {skill}: evaluation error -> {e}")

    # Summary table
    if results:
        summary = pd.DataFrame(results)
        print("\n=== Summary of MAPE for Each Skill ===")
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
