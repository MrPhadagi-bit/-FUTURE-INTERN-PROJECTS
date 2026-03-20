from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "support_tickets.csv"
MODELS_DIR = ROOT_DIR / "models"
OUTPUTS_DIR = ROOT_DIR / "outputs"

CATEGORY_SIGNALS = {
    "billing": ["billing", "invoice", "refund", "charged", "payment", "subscription", "renewal", "receipts"],
    "technical": ["crash", "error", "timeout", "api", "sync", "upload", "dashboard", "export", "blank screen"],
    "account": ["login", "password", "account", "role", "permissions", "profile", "access", "authentication"],
    "query": ["offer", "support", "demo", "documentation", "available", "clarify", "included", "template"],
}

PRIORITY_SIGNALS = {
    "high": ["urgent", "today", "blocked", "cannot", "failed", "suspended", "500", "outage", "disabled", "locked"],
    "medium": ["slow", "issue", "pending", "expired", "clarify", "confirm", "migration", "stopped"],
    "low": ["how", "can", "please", "explain", "update", "copy", "demo", "template"],
}


def normalize_text(text: str) -> str:
    raw_text = text.lower()
    text = raw_text
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = [token for token in text.split() if token not in ENGLISH_STOP_WORDS]
    engineered_tokens = []

    for signal_name, keywords in CATEGORY_SIGNALS.items():
        if any(keyword in raw_text for keyword in keywords):
            engineered_tokens.append(f"signal_category_{signal_name}")

    for signal_name, keywords in PRIORITY_SIGNALS.items():
        if any(keyword in raw_text for keyword in keywords):
            engineered_tokens.append(f"signal_priority_{signal_name}")

    return " ".join(tokens + engineered_tokens)


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=normalize_text,
                    ngram_range=(1, 2),
                    min_df=1,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LinearSVC(class_weight="balanced", random_state=42, dual="auto"),
            ),
        ]
    )


def evaluate_model(model: Pipeline, x_test: pd.Series, y_test: pd.Series) -> dict:
    predictions = model.predict(x_test)
    labels = sorted(y_test.unique())
    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_test, predictions, labels=labels)

    return {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "labels": labels,
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "predictions": predictions.tolist(),
    }


def save_confusion_matrix(labels: list[str], matrix: list[list[int]], output_path: Path) -> None:
    frame = pd.DataFrame(matrix, index=labels, columns=labels)
    frame.index.name = "actual"
    frame.to_csv(output_path)


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    tickets = pd.read_csv(DATA_PATH)

    category_train_df, category_test_df = train_test_split(
        tickets,
        test_size=0.25,
        random_state=42,
        stratify=tickets["category"],
    )
    x_category_train = category_train_df["ticket_text"]
    x_category_test = category_test_df["ticket_text"]
    y_category_train = category_train_df["category"]
    y_category_test = category_test_df["category"]

    priority_train_df, priority_test_df = train_test_split(
        tickets,
        test_size=0.25,
        random_state=42,
        stratify=tickets["priority"],
    )
    x_priority_train = priority_train_df["ticket_text"]
    x_priority_test = priority_test_df["ticket_text"]
    y_priority_train = priority_train_df["priority"]
    y_priority_test = priority_test_df["priority"]

    category_model = build_pipeline()
    category_model.fit(x_category_train, y_category_train)
    category_results = evaluate_model(category_model, x_category_test, y_category_test)
    save_confusion_matrix(
        category_results["labels"],
        category_results["confusion_matrix"],
        OUTPUTS_DIR / "category_confusion_matrix.csv",
    )
    joblib.dump(category_model, MODELS_DIR / "category_model.joblib")

    priority_model = build_pipeline()
    priority_model.fit(x_priority_train, y_priority_train)
    priority_results = evaluate_model(priority_model, x_priority_test, y_priority_test)
    save_confusion_matrix(
        priority_results["labels"],
        priority_results["confusion_matrix"],
        OUTPUTS_DIR / "priority_confusion_matrix.csv",
    )
    joblib.dump(priority_model, MODELS_DIR / "priority_model.joblib")

    sample_tickets = pd.DataFrame(
        {
            "ticket_text": [
                "Users cannot log in after a password reset and the whole account team is blocked.",
                "Please send a copy of our April invoice and update the billing contact.",
                "How do I add a viewer role for our external auditor?",
                "The dashboard is slow but still working for most users.",
            ]
        }
    )
    sample_tickets["predicted_category"] = category_model.predict(sample_tickets["ticket_text"])
    sample_tickets["predicted_priority"] = priority_model.predict(sample_tickets["ticket_text"])
    sample_tickets.to_csv(OUTPUTS_DIR / "sample_ticket_predictions.csv", index=False)

    metrics = {
        "dataset_rows": int(len(tickets)),
        "category_model": {
            "accuracy": category_results["accuracy"],
            "precision_weighted": round(
                float(category_results["classification_report"]["weighted avg"]["precision"]), 4
            ),
            "recall_weighted": round(
                float(category_results["classification_report"]["weighted avg"]["recall"]), 4
            ),
            "f1_weighted": round(
                float(category_results["classification_report"]["weighted avg"]["f1-score"]), 4
            ),
        },
        "priority_model": {
            "accuracy": priority_results["accuracy"],
            "precision_weighted": round(
                float(priority_results["classification_report"]["weighted avg"]["precision"]), 4
            ),
            "recall_weighted": round(
                float(priority_results["classification_report"]["weighted avg"]["recall"]), 4
            ),
            "f1_weighted": round(
                float(priority_results["classification_report"]["weighted avg"]["f1-score"]), 4
            ),
        },
    }

    with (OUTPUTS_DIR / "ticket_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    summary_lines = [
        "# Support Ticket Model Summary",
        "",
        f"- Dataset rows: {metrics['dataset_rows']}",
        f"- Category accuracy: {metrics['category_model']['accuracy']:.4f}",
        f"- Category weighted precision: {metrics['category_model']['precision_weighted']:.4f}",
        f"- Priority accuracy: {metrics['priority_model']['accuracy']:.4f}",
        f"- Priority weighted precision: {metrics['priority_model']['precision_weighted']:.4f}",
        "",
        "## Operational interpretation",
        "",
        "The category model is strong enough for a first-pass routing workflow, while the priority model should improve further with more historical tickets and richer urgency labels.",
    ]
    (OUTPUTS_DIR / "ticket_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    print("Saved ticket models, metrics, confusion matrices, and sample predictions.")


if __name__ == "__main__":
    main()
