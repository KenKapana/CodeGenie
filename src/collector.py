import requests
from collections import Counter
from dotenv import load_dotenv
import os

load_dotenv()
ADZUNA_APP_ID = os.getenv("ADZUNA_APP_ID")
ADZUNA_APP_KEY = os.getenv("ADZUNA_APP_KEY")
COUNTRY = "ca"
GITHUB_TOPICS = ["machine-learning", "data-science", "web-development", "python", "javascript"]


def get_adzuna_jobs(region: str, query: str = "developer", days: int = 30) -> list[str]:
    url = f"https://api.adzuna.com/v1/api/jobs/{COUNTRY}/search/1"
    params = {
        "app_id": ADZUNA_APP_ID,
        "app_key": ADZUNA_APP_KEY,
        "what": query,
        "where": region,
        "max_days_old": days,
        "results_per_page": 100,
    }
    res = requests.get(url, params=params)
    res.raise_for_status()
    return [job["description"] for job in res.json().get("results", [])]


def scrape_github_trends(topics: list[str]) -> dict[str, int]:
    headers = {"Accept": "application/vnd.github.v3+json"}
    counts = Counter()
    for topic in topics:
        url = f"https://api.github.com/search/repositories?q=topic:{topic}&sort=stars&order=desc"
        res = requests.get(url, headers=headers)
        if res.status_code == 200:
            for item in res.json().get("items", []):
                lang = item.get("language")
                if lang:
                    counts[lang.lower()] += 1
    return dict(counts)