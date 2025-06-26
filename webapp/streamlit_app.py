import os
import sys
import json
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error
from prophet import Prophet
from prophet.serialize import model_from_json

# allow imports from src/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.collector import get_adzuna_jobs, scrape_github_trends
from src.features import extract_skill_frequency
from src.recommender import best_framework

# Configuration
from src.constants import SKILLS
GITHUB_TOPICS = ["machine-learning", "data-science", "web-development", "python", "javascript", "project", "devops", "cloud-computing"]

st.set_page_config(page_title="PredictSkills", layout="wide")
st.title("🔍 PredictSkills: Top 5 Tech Skills & Evaluation")

# --- Prediction Section ---
st.header("🔍 Find Top Skills")
region = st.selectbox("Location", ["Toronto","Vancouver","Montreal","Calgary","Ottawa"])
major = st.selectbox("Major", ["Computer Science","Engineering","Business","Other"])
if st.button("Run Prediction"):
    jobs = get_adzuna_jobs(region)
    adzuna_freq = extract_skill_frequency(jobs, SKILLS)
    github_trends = scrape_github_trends(GITHUB_TOPICS)

    total = {s: adzuna_freq.get(s,0)*2 + github_trends.get(s,0) for s in set(SKILLS)|set(github_trends)}
    top5 = sorted(total.items(), key=lambda x: x[1], reverse=True)[:5]

    st.subheader("Top 5 Skills for Next Year")
    for i,(skill,score) in enumerate(top5,1):
        st.write(f"{i}. {skill.title()} — score {score}")

    st.subheader("🔎 Adzuna Skill Frequencies")
    st.bar_chart(pd.Series(adzuna_freq).sort_values(ascending=False).head(15))

    st.subheader("📈 GitHub Language Popularity")
    st.bar_chart(pd.Series(github_trends).sort_values(ascending=False).head(15))

# --- Evaluation Section ---
st.sidebar.header("📊 Evaluate Forecast Accuracy")
if st.sidebar.button("Evaluate SMAPE"):
    results = []
    data_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data")))

    def smape(y_true, y_pred):
        denom = (np.abs(y_true) + np.abs(y_pred)) / 2
        diff = np.abs(y_pred - y_true)
        smape_vals = np.where(denom == 0, 0, diff / denom)
        return np.mean(smape_vals)

    for skill in SKILLS:
        path = data_dir / f"{skill}.csv"
        df = pd.read_csv(path, parse_dates=["ds"] )
        df = df.sort_values("ds")
        last_date = df["ds"].max()
        cutoff = last_date - pd.DateOffset(months=6)

        train_df = df[df["ds"] <= cutoff]
        test_df  = df[df["ds"] >  cutoff]
        if train_df.empty or test_df.empty:
            st.write(f"❌ Not enough data for {skill}")
            continue

        m = Prophet(yearly_seasonality=True)
        m.fit(train_df)
        future = test_df[["ds"]]
        forecast = m.predict(future)

        y_true = test_df["y"].values
        y_pred = forecast["yhat"].values

        error = smape(y_true, y_pred)
        results.append({"Skill": skill.title(), "SMAPE": f"{error:.2%}"})
        st.write(f"{skill.title()}: SMAPE = {error:.2%}")

    if results:
        st.subheader("SMAPE Summary")
        st.table(pd.DataFrame(results))