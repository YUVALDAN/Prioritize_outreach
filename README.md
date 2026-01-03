# WellCo churn reduction – prioritized outreach

## What this is
This project builds a ranked list of members for proactive outreach to reduce churn, and selects an outreach size `n`.

Key deliverable:
- `data/outreach_prioritized_members.csv` with `member_id`, `prioritization_score` (uplift), and `rank`.

## Data
All input files are under `data/`:
- `data/train/`: `web_visits.csv`, `app_usage.csv`, `claims.csv`, `churn_labels.csv`
- `data/test/`: `test_web_visits.csv`, `test_app_usage.csv`, `test_claims.csv`, `test_churn_labels.csv`
- Schemas: `data/schema_*.md`
- Provided baselines: `data/auc_baseline_test.txt`, `data/classification_report_baseline_test.txt`

## Setup
Python 3.10+ recommended.

Install dependencies (typical):
```bash
pip install -U pandas numpy scikit-learn matplotlib seaborn jupyter
```

## Run
Open and run the main notebook:
- `notebooks/priorize_members.ipynb`

open `priorize_members.ipynb` and **Run All**.

The notebook writes:
- `data/outreach_prioritized_members.csv`

## Approach
1) **Sanity checks + cleaning**: validate shapes/nulls/date ranges and drop exact duplicate claim rows to avoid inflating count-based features.
2) **Feature engineering (member-level)**:
   - Tenure/lifecycle features from `signup_date`.
   - Engagement features from web/app (counts, active days, recency, simple rates).
   - Web content intent via topic shares: start from brief-driven seeds, expand keywords from train web text using TF‑IDF lift, then count topic visits per member.
   - Claims burden via claim counts/recency and a small set of ICD indicators (top 5 by absolute churn lift among sufficiently prevalent codes).
3) **Model 1 (churn-risk)**: churn prediction is evaluated with stratified K-fold CV and ROC-AUC (threshold-free, robust under class imbalance).
4) **Model 2 (uplift / who benefits from outreach)**:
   - Fit a propensity model `P(outreach|X)` to adjust for non-random outreach.
   - Fit separate churn models for control vs treated and define uplift as `P(churn|no outreach) - P(churn|outreach)`.
   - Rank members by uplift (tie-break by higher `P(churn|no outreach)`).
5) **Choosing outreach size `n`**:
   - Plot cumulative expected churn prevented vs `n`.
   - Select `n` based on the rolling marginal uplift (slope)

Thank you for your time