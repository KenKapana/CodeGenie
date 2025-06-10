# TODO: implement 1-year trend forecaster (e.g., Prophet)
from prophet import Prophet
import pandas as pd


def train_1yr_forecaster(df: pd.DataFrame) -> Prophet:
    # expects df with columns ["ds", "y"]
    m = Prophet(yearly_seasonality=True)
    m.fit(df)
    m.save("models/prophet_1yr.pkl")
    return m