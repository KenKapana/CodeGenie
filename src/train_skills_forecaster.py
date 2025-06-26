# train_skills_forecaster.py

import pandas as pd
import json
from prophet import Prophet
from prophet.serialize import model_to_json
from pathlib import Path

# Configuration
from constants import SKILLS
DATA_DIR = Path("data")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


def train_and_save_model(skill: str, df: pd.DataFrame):
    """
    Trains a Prophet model on the skill's time-series data.
    Expects a DataFrame with ['ds', 'y'] where ds is datetime and y is numeric.
    Saves the model to models/{skill}_model.json.
    """
    m = Prophet(yearly_seasonality=True)
    m.fit(df)
    out_path = MODEL_DIR / f"{skill}_model.json"
    with open(out_path, "w") as f:
        json.dump(model_to_json(m), f)
    print(f"✅ Trained and saved model for {skill} to {out_path}")


def main():
    for skill in SKILLS:
        file_path = DATA_DIR / f"{skill}.csv"
        if not file_path.exists():
            print(f"❌ Skipping {skill}, no data file found at {file_path}")
            continue

        df = pd.read_csv(file_path)
        if not {"ds", "y"}.issubset(df.columns):
            print(f"❌ Skipping {skill}, file does not have 'ds' and 'y' columns")
            continue

        df["ds"] = pd.to_datetime(df["ds"])
        df = df.sort_values("ds")

        train_and_save_model(skill, df)


if __name__ == "__main__":
    main()
