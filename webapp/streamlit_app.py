import streamlit as st
import pandas as pd
from src.collector import get_adzuna_jobs, scrape_github_trends
from src.features import extract_skill_frequency
from src.recommender import best_framework

st.set_page_config(page_title="Skill Predictor", layout="wide")
st.title("🔍 Top 5 Tech Skills for Students")

region = st.selectbox("Where are you located?", ["Toronto","Vancouver","Montreal","Calgary","Ottawa"])
major = st.selectbox("Your major", ["Computer Science","Engineering","Business","Other"])
if st.button("Find Top Skills"):
    jobs = get_adzuna_jobs(region)
    adzuna_freq = extract_skill_frequency(jobs, SKILLS := ["python","javascript","react","sql","java","docker","aws"])
    github_trends = scrape_github_trends(GITHUB_TOPICS)

    # Combine scores
    total = {s: adzuna_freq.get(s,0)*2 + github_trends.get(s,0) for s in set(SKILLS) | set(github_trends)}
    top5 = sorted(total.items(), key=lambda x: x[1], reverse=True)[:5]

    st.subheader("Top 5 Skills to Learn in the Next Year")
    for i,(skill,score) in enumerate(top5,1):
        st.write(f"{i}. {skill.title()} — score {score}")

    st.subheader("🔎 Adzuna Skill Frequencies")
    st.bar_chart(pd.Series(adzuna_freq).sort_values(ascending=False).head(15))

    st.subheader("📈 GitHub Language Popularity")
    st.bar_chart(pd.Series(github_trends).sort_values(ascending=False).head(15))