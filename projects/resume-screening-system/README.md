# Resume and Candidate Screening System

This project ranks resumes against a target job description using a blend of NLP similarity and skill coverage. The goal is to help recruiters shortlist candidates faster while keeping the explanation business-friendly.

## Business Problem

Recruiters often receive many resumes for one role and need to answer three questions quickly:

- Which candidates fit the role best
- Why some candidates rank higher than others
- Which important skills are missing

## Features Implemented

- Resume text cleaning and preprocessing
- Job description parsing
- Skill extraction using a curated skill vocabulary
- Resume-to-role similarity scoring with TF-IDF and cosine similarity
- Candidate ranking
- Skill gap identification
- Candidate skill matrix output for easy review

## Data

The repository includes:

- `data/job_description.txt`
- `data/resumes/*.txt`

These are realistic sample files so the project runs locally without requiring a download. You can later replace them with:

- Kaggle job descriptions: <https://www.kaggle.com/datasets/PromptCloudHQ/us-jobs-on-monstercom>
- Real or anonymized resumes in plain text format

## Run

```bash
python src/rank_resumes.py
```

## Outputs

- `outputs/candidate_rankings.csv`
- `outputs/candidate_skill_matrix.csv`
- `outputs/ranking_summary.json`
- `outputs/ranking_summary.md`

## Scoring Logic

The final score combines:

- `70%` TF-IDF text similarity between the resume and the job description
- `30%` skill coverage based on matched required skills

This keeps the ranking easy to explain:

- high-ranking candidates use language similar to the job post
- high-ranking candidates also cover more required skills
- missing skills are listed clearly for recruiter review

## Business Explanation

- Resumes score higher when they match both the wording and the core skills in the job description.
- Skill gaps show exactly what a recruiter might want to probe during interviews.
- The system reduces manual screening time while making shortlist decisions more consistent.
