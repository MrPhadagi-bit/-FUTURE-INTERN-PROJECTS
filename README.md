# Future Intern ML project 

This repository combines three practical machine learning projects that solve real business workflow problems:

1. `superstore-sales-forecasting` forecasts future sales to support planning and inventory decisions.
2. `support-ticket-classification` classifies incoming tickets and predicts urgency for support teams.
3. `resume-screening-system` ranks candidates against a job description and highlights skill gaps.

## Repository Structure

```text
business-ml-portfolio/
  projects/
    superstore-sales-forecasting/
    support-ticket-classification/
    resume-screening-system/
```

## Included Projects

### 1. Superstore Sales Forecasting

This project was migrated from the earlier standalone `superstore-sales-forecasting` repository so it can live alongside the rest of your ML portfolio.

- Path: `projects/superstore-sales-forecasting`
- Main script: `python src/forecast_sales.py`
- Business value: supports forecasting, stock planning, and budget preparation

### 2. Support Ticket Classification and Prioritization

This NLP workflow reads support tickets, cleans text, converts it into TF-IDF features, predicts the ticket category, and predicts whether the issue is high, medium, or low priority.

- Path: `projects/support-ticket-classification`
- Main script: `python src/train_ticket_models.py`
- Business value: helps support teams route tickets faster and focus on urgent issues first

### 3. Resume and Candidate Screening System

This project ranks resumes against a target job description using TF-IDF similarity plus skill coverage analysis. It also shows which required skills are missing for each candidate.

- Path: `projects/resume-screening-system`
- Main script: `python src/rank_resumes.py`
- Business value: helps recruiters shortlist faster and explain candidate fit more clearly

## Setup

Create a virtual environment and install the shared dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run Each Project

```bash
cd projects/superstore-sales-forecasting
python src/forecast_sales.py

cd ..\support-ticket-classification
python src/train_ticket_models.py

cd ..\resume-screening-system
python src/rank_resumes.py
```

## Notes

- The support ticket and resume projects include sample datasets so the code runs out of the box.
- Both new projects are structured so you can replace the sample data with Kaggle datasets later without changing the pipeline design.
- GitHub CLI is not installed in this environment, so the repository has been prepared locally and is ready to push once you create a remote or install `gh`.
