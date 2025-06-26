# used by: 
#   train_skills_forecaster.py

#utilize: 
#   collector.py
#   features.py

import argparse
import pandas as pd
from pytrends.request import TrendReq
from pathlib import Path
from datetime import date

# Optional extras
from src.collector import get_adzuna_jobs, scrape_github_trends
from src.features import extract_skill_frequency

# Configuration
from constants import SKILLS
TIMEFRAME_DEFAULT = "today 5-y"  # last 5 years
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


def fetch_google_trends(skill: str, timeframe: str) -> pd.DataFrame:
    """
    Fetch weekly interest for a single skill from Google Trends.
    Returns a DataFrame with columns ['ds','y'].
    """
    pytrends = TrendReq()
    pytrends.build_payload([skill], timeframe=timeframe)
    df = pytrends.interest_over_time()
    if df.empty:
        return pd.DataFrame(columns=["ds", "y"])
    df = df.reset_index()[["date", skill]]
    df.columns = ["ds", "y"]
    df["ds"] = pd.to_datetime(df["ds"])
    return df


def append_extras(df: pd.DataFrame, skill: str, region: str, days: int) -> pd.DataFrame:
    """
    Append a data point for today's date using Adzuna and GitHub metrics.
    Weighted sum: y = 2*job_count + repo_lang_count
    """
    # Adzuna job descriptions
    jobs = get_adzuna_jobs(region=region, query=skill, days=days)
    adzuna_counts = extract_skill_frequency(jobs, [skill])
    job_count = adzuna_counts.get(skill, 0)
    # GitHub trends
    gh_counts = scrape_github_trends([skill])
    repo_count = gh_counts.get(skill, 0)
    # weighted score
    extra_y = job_count * 2 + repo_count

    today = pd.Timestamp(date.today())
    extra_row = pd.DataFrame([[today, extra_y]], columns=["ds", "y"])
    return pd.concat([df, extra_row], ignore_index=True)


def main():
    parser = argparse.ArgumentParser(
        description="Generate time series CSVs for skills."
    )
    parser.add_argument(
        "--timeframe", "-t", default=TIMEFRAME_DEFAULT,
        help="Google Trends timeframe string (e.g., 'today 5-y')"
    )
    parser.add_argument(
        "--include-extras", "-e", action="store_true",
        help="Append today's Adzuna and GitHub extra data point"
    )
    parser.add_argument(
        "--region", "-r", default="Toronto",
        help="Region for Adzuna job search (e.g., 'Toronto')"
    )
    parser.add_argument(
        "--days", "-d", type=int, default=30,
        help="Max days old for Adzuna jobs"
    )
    args = parser.parse_args()

    for skill in SKILLS:
        print(f"\nFetching Google Trends for '{skill}'...")
        df = fetch_google_trends(skill, args.timeframe)
        if df.empty:
            print(f"  ⚠️ No Google Trends data for {skill}.")
            continue

        if args.include_extras:
            print(f"  Adding Adzuna/GitHub extras for {skill}...")
            df = append_extras(df, skill, args.region, args.days)

        out_path = DATA_DIR / f"{skill}.csv"
        df.to_csv(out_path, index=False)
        print(f"  ✅ Saved {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
