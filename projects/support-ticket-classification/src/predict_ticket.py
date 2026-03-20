from __future__ import annotations

import argparse
from pathlib import Path

import joblib


ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models"


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict ticket category and priority.")
    parser.add_argument("ticket_text", help="Support ticket text to classify")
    args = parser.parse_args()

    category_model = joblib.load(MODELS_DIR / "category_model.joblib")
    priority_model = joblib.load(MODELS_DIR / "priority_model.joblib")

    category = category_model.predict([args.ticket_text])[0]
    priority = priority_model.predict([args.ticket_text])[0]

    print(f"Category: {category}")
    print(f"Priority: {priority}")


if __name__ == "__main__":
    main()
