# run local:
streamlit run webapp/streamlit_app.py

# venv:
activate in cmd: .venv\Scripts\activate.bat
deactivate in cmd: deactivate

# an optional extras feature to generate_skill_timeseries.py:

## CLI flags:

--include-extras and -e to append today’s Adzuna & GitHub-derived metric.

--region and --days to configure Adzuna parameters.

Weighted score for adzuna jobs: 
y = 2*job_count + repo_count for that extra data point.

# create time table for training:
python src/generate_timeseries.py
or 
python scripts/generate_skill_timeseries.py --include-extras --region Vancouver --days 14


# train model:
pythontrain_skills_forecaster.py

## training method
prioritized SMAPE to avoid data blowing up (handles 0 cases well)
hold-out split of 52 weeks
6 months test window

# What's next?
- Automated Retraining
Build a background job (e.g. via GitHub Actions or a cron) that re-pulls data weekly, retrains, and refreshes your visualizations automatically.
- Multi-source Trend Comparison
Plot GitHub vs. Google Trends vs. StackOverflow interest on one time-series chart so users see where demand is coming from.
- Hands-On Challenges
Embed mini coding problems or notebooks so users can test a skill right in the browser.
- Dockerization