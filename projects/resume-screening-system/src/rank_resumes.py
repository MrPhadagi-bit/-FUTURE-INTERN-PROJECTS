from __future__ import annotations

import json
import os
import re
from pathlib import Path

import matplotlib
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


ROOT_DIR = Path(__file__).resolve().parents[1]
MPL_CONFIG_DIR = ROOT_DIR / ".matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = ROOT_DIR / "data"
RESUMES_DIR = DATA_DIR / "resumes"
OUTPUTS_DIR = ROOT_DIR / "outputs"

SKILL_VOCABULARY = [
    "python",
    "sql",
    "machine learning",
    "data analysis",
    "scikit-learn",
    "customer support",
    "ticket triage",
    "reporting",
    "dashboarding",
    "communication",
    "nlp",
    "power bi",
    "etl",
    "git",
    "a/b testing",
]


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s/-]", " ", text)
    tokens = [token for token in text.split() if token not in ENGLISH_STOP_WORDS]
    return " ".join(tokens)


def extract_skills(text: str, vocabulary: list[str]) -> list[str]:
    text = clean_text(text)
    found_skills = []
    for skill in vocabulary:
        normalized_skill = skill.lower()
        if normalized_skill in text:
            found_skills.append(skill)
    return found_skills


def extract_required_skills(job_description: str) -> list[str]:
    match = re.search(
        r"required skills:(.*?)(nice to have:|responsibilities:|$)",
        job_description,
        flags=re.IGNORECASE | re.DOTALL,
    )
    required_section = match.group(1) if match else job_description
    return extract_skills(required_section, SKILL_VOCABULARY)


def score_resumes(job_description: str, resumes: dict[str, str]) -> pd.DataFrame:
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    documents = [job_description, *resumes.values()]
    matrix = vectorizer.fit_transform(documents)

    job_vector = matrix[0:1]
    resume_vectors = matrix[1:]
    similarity_scores = cosine_similarity(job_vector, resume_vectors).flatten()

    required_skills = extract_required_skills(job_description)
    rows = []

    for (candidate_name, resume_text), similarity_score in zip(resumes.items(), similarity_scores):
        candidate_skills = extract_skills(resume_text, SKILL_VOCABULARY)
        matched_skills = sorted(set(required_skills).intersection(candidate_skills))
        missing_skills = sorted(set(required_skills) - set(candidate_skills))

        coverage = len(matched_skills) / len(required_skills) if required_skills else 0.0
        final_score = (0.7 * float(similarity_score)) + (0.3 * coverage)

        rows.append(
            {
                "candidate_name": candidate_name,
                "similarity_score": round(float(similarity_score), 4),
                "skill_coverage": round(float(coverage), 4),
                "final_score": round(float(final_score), 4),
                "matched_skills": ", ".join(matched_skills),
                "missing_skills": ", ".join(missing_skills),
            }
        )

    rankings = pd.DataFrame(rows).sort_values("final_score", ascending=False).reset_index(drop=True)
    rankings.index = rankings.index + 1
    rankings.index.name = "rank"
    return rankings


def build_skill_matrix(job_description: str, resumes: dict[str, str]) -> pd.DataFrame:
    required_skills = extract_required_skills(job_description)
    records = []

    for candidate_name, resume_text in resumes.items():
        candidate_skills = set(extract_skills(resume_text, SKILL_VOCABULARY))
        row = {"candidate_name": candidate_name}
        for skill in required_skills:
            row[skill] = int(skill in candidate_skills)
        records.append(row)

    return pd.DataFrame(records)


def save_rankings_plot(rankings: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_frame = rankings.sort_values("final_score", ascending=True)
    ax.barh(plot_frame["candidate_name"], plot_frame["final_score"], color="#2563eb")
    ax.set_title("Resume Ranking Scores")
    ax.set_xlabel("Final score")
    ax.set_ylabel("Candidate")

    for index, value in enumerate(plot_frame["final_score"]):
        ax.text(value + 0.005, index, f"{value:.3f}", va="center", fontsize=9)

    ax.set_xlim(0, max(plot_frame["final_score"]) + 0.1)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    job_description = (DATA_DIR / "job_description.txt").read_text(encoding="utf-8")
    resumes = {
        path.stem.replace("_", " ").title(): path.read_text(encoding="utf-8")
        for path in sorted(RESUMES_DIR.glob("*.txt"))
    }

    rankings = score_resumes(job_description, resumes)
    skill_matrix = build_skill_matrix(job_description, resumes)

    rankings.to_csv(OUTPUTS_DIR / "candidate_rankings.csv")
    skill_matrix.to_csv(OUTPUTS_DIR / "candidate_skill_matrix.csv", index=False)
    save_rankings_plot(rankings, OUTPUTS_DIR / "candidate_rankings.png")

    top_candidate = rankings.iloc[0]
    summary = {
        "job_title": "Machine Learning Support Analyst",
        "candidates_screened": int(len(rankings)),
        "top_candidate": top_candidate["candidate_name"],
        "top_candidate_score": float(top_candidate["final_score"]),
        "top_candidate_missing_skills": top_candidate["missing_skills"].split(", ")
        if top_candidate["missing_skills"]
        else [],
    }

    with (OUTPUTS_DIR / "ranking_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    markdown_lines = [
        "# Resume Screening Summary",
        "",
        f"- Job title: {summary['job_title']}",
        f"- Candidates screened: {summary['candidates_screened']}",
        f"- Top-ranked candidate: {summary['top_candidate']}",
        f"- Top score: {summary['top_candidate_score']:.4f}",
        "",
        "## Why the top candidate ranked first",
        "",
        "The top candidate combined strong text similarity with broad coverage of the required role skills.",
        "",
        "## Missing skills for the top candidate",
        "",
        top_candidate["missing_skills"] or "No missing required skills identified.",
    ]
    (OUTPUTS_DIR / "ranking_summary.md").write_text("\n".join(markdown_lines), encoding="utf-8")

    print("Saved candidate rankings, skill matrix, and summary files.")


if __name__ == "__main__":
    main()
