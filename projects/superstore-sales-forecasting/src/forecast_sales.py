from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
MPL_CONFIG_DIR = ROOT_DIR / ".matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthBegin
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error


FEATURE_COLUMNS = [
    "time_idx",
    "year",
    "month",
    "quarter",
    "month_sin",
    "month_cos",
    "lag_1",
    "lag_3",
    "lag_6",
    "lag_12",
    "rolling_mean_3",
    "rolling_mean_6",
    "rolling_mean_12",
]


@dataclass
class CleaningSummary:
    source_file: str
    rows_loaded: int
    rows_after_cleaning: int
    rows_dropped: int
    start_date: str
    end_date: str
    months_covered: int
    unique_regions: int
    unique_categories: int


def resolve_project_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT_DIR / path


def to_snake_case(value: str) -> str:
    value = re.sub(r"[^0-9a-zA-Z]+", "_", value.strip())
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    return value.strip("_").lower()


def load_sales_data(csv_path: Path) -> tuple[pd.DataFrame, CleaningSummary]:
    raw_df = pd.read_csv(csv_path)
    raw_df.columns = [to_snake_case(column) for column in raw_df.columns]

    rows_loaded = len(raw_df)
    for column in ("order_date", "ship_date"):
        if column in raw_df.columns:
            raw_df[column] = pd.to_datetime(raw_df[column], errors="coerce")

    raw_df["sales"] = pd.to_numeric(raw_df["sales"], errors="coerce")
    for column in ("region", "category", "sub_category", "segment", "state"):
        if column in raw_df.columns:
            raw_df[column] = raw_df[column].fillna("Unknown")

    cleaned_df = raw_df.dropna(subset=["order_date", "sales"]).copy()
    cleaned_df = cleaned_df.sort_values("order_date")
    cleaned_df["order_month"] = cleaned_df["order_date"].dt.to_period("M").dt.to_timestamp()

    monthly_sales = (
        cleaned_df.groupby("order_month", as_index=False)["sales"]
        .sum()
        .sort_values("order_month")
    )

    full_month_range = pd.date_range(
        start=monthly_sales["order_month"].min(),
        end=monthly_sales["order_month"].max(),
        freq="MS",
    )

    monthly_sales = (
        monthly_sales.set_index("order_month")
        .reindex(full_month_range, fill_value=0.0)
        .rename_axis("order_month")
        .reset_index()
    )
    monthly_sales["sales"] = monthly_sales["sales"].astype(float).round(2)

    summary = CleaningSummary(
        source_file=str(csv_path.relative_to(ROOT_DIR)),
        rows_loaded=rows_loaded,
        rows_after_cleaning=len(cleaned_df),
        rows_dropped=rows_loaded - len(cleaned_df),
        start_date=monthly_sales["order_month"].min().strftime("%Y-%m-%d"),
        end_date=monthly_sales["order_month"].max().strftime("%Y-%m-%d"),
        months_covered=len(monthly_sales),
        unique_regions=cleaned_df["region"].nunique() if "region" in cleaned_df else 0,
        unique_categories=cleaned_df["category"].nunique() if "category" in cleaned_df else 0,
    )
    return monthly_sales, summary


def build_feature_row(history: list[float], current_date: pd.Timestamp, time_idx: int) -> dict[str, float]:
    if len(history) < 12:
        raise ValueError("At least 12 months of history are required to create forecasting features.")

    month = current_date.month
    return {
        "time_idx": float(time_idx),
        "year": float(current_date.year),
        "month": float(month),
        "quarter": float(current_date.quarter),
        "month_sin": float(np.sin(2 * np.pi * month / 12)),
        "month_cos": float(np.cos(2 * np.pi * month / 12)),
        "lag_1": float(history[-1]),
        "lag_3": float(history[-3]),
        "lag_6": float(history[-6]),
        "lag_12": float(history[-12]),
        "rolling_mean_3": float(np.mean(history[-3:])),
        "rolling_mean_6": float(np.mean(history[-6:])),
        "rolling_mean_12": float(np.mean(history[-12:])),
    }


def create_supervised_frame(monthly_sales: pd.DataFrame) -> pd.DataFrame:
    history = monthly_sales["sales"].tolist()
    rows: list[dict[str, float | pd.Timestamp]] = []

    for index in range(12, len(monthly_sales)):
        current_date = monthly_sales.loc[index, "order_month"]
        feature_row = build_feature_row(history[:index], current_date, index)
        feature_row["order_month"] = current_date
        feature_row["sales"] = float(history[index])
        rows.append(feature_row)

    return pd.DataFrame(rows)


def recursive_forecast(
    model: Ridge,
    history: list[float],
    forecast_dates: list[pd.Timestamp],
) -> np.ndarray:
    predictions: list[float] = []
    recursive_history = history.copy()

    for forecast_date in forecast_dates:
        feature_row = build_feature_row(
            recursive_history,
            pd.Timestamp(forecast_date),
            len(recursive_history),
        )
        prediction = float(model.predict(pd.DataFrame([feature_row])[FEATURE_COLUMNS])[0])
        prediction = max(0.0, prediction)
        predictions.append(prediction)
        recursive_history.append(prediction)

    return np.array(predictions)


def seasonal_naive_forecast(history: list[float], horizon: int) -> np.ndarray:
    predictions: list[float] = []
    recursive_history = history.copy()

    for _ in range(horizon):
        baseline = recursive_history[-12] if len(recursive_history) >= 12 else recursive_history[-1]
        predictions.append(float(baseline))
        recursive_history.append(float(baseline))

    return np.array(predictions)


def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    non_zero_mask = actual != 0
    mape = (
        float(np.mean(np.abs((actual[non_zero_mask] - predicted[non_zero_mask]) / actual[non_zero_mask])) * 100)
        if np.any(non_zero_mask)
        else 0.0
    )

    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(mean_squared_error(actual, predicted, squared=False)),
        "mape_pct": mape,
        "mean_bias": float(np.mean(predicted - actual)),
    }


def build_business_summary(
    summary: CleaningSummary,
    metrics: dict[str, float],
    baseline_metrics: dict[str, float],
    monthly_sales: pd.DataFrame,
    future_forecast: pd.DataFrame,
) -> str:
    recent_average = float(monthly_sales["sales"].tail(6).mean())
    forecast_average = float(future_forecast["forecast_sales"].mean())
    average_change_pct = ((forecast_average - recent_average) / recent_average) * 100 if recent_average else 0.0
    direction = "above" if average_change_pct >= 0 else "below"

    strongest_month = future_forecast.loc[future_forecast["forecast_sales"].idxmax()]
    softest_month = future_forecast.loc[future_forecast["forecast_sales"].idxmin()]
    seasonal_profile = monthly_sales.assign(month_name=monthly_sales["order_month"].dt.strftime("%B"))
    average_by_month = seasonal_profile.groupby("month_name")["sales"].mean().sort_values(ascending=False)

    lines = [
        "# Sales Forecast Summary",
        "",
        "## What the forecast means",
        "",
        f"- The dataset covers {summary.start_date} to {summary.end_date} across {summary.months_covered} monthly periods.",
        f"- After cleaning, the model used {summary.rows_after_cleaning:,} valid transactions from {summary.unique_regions} regions and {summary.unique_categories} product categories.",
        f"- On the final holdout window, the forecasting model reached an MAE of ${metrics['mae']:,.0f}, RMSE of ${metrics['rmse']:,.0f}, and MAPE of {metrics['mape_pct']:.1f}%.",
        f"- The model outperformed a seasonal naive baseline, which had an MAE of ${baseline_metrics['mae']:,.0f}.",
        f"- The next {len(future_forecast)} months are forecast at a total of ${future_forecast['forecast_sales'].sum():,.0f}, with an average month {abs(average_change_pct):.1f}% {direction} the latest six-month average.",
        f"- The strongest forecast month is {strongest_month['order_month'].strftime('%B %Y')} at ${strongest_month['forecast_sales']:,.0f}.",
        f"- The softest forecast month is {softest_month['order_month'].strftime('%B %Y')} at ${softest_month['forecast_sales']:,.0f}.",
        "",
        "## How a business can use this",
        "",
        f"- Inventory teams can stock more aggressively before {strongest_month['order_month'].strftime('%B')} and stay leaner near {softest_month['order_month'].strftime('%B')}.",
        "- Finance teams can use the projected revenue range to plan cash flow and purchasing schedules.",
        "- Staffing managers can align promotions, labor, and fulfillment capacity to expected demand instead of reacting late.",
        f"- Seasonality appears strongest in {average_by_month.index[0]}, so campaigns and replenishment plans should be locked in ahead of that month.",
    ]
    return "\n".join(lines)


def create_visualizations(
    monthly_sales: pd.DataFrame,
    evaluation_df: pd.DataFrame,
    future_forecast: pd.DataFrame,
    output_dir: Path,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    history_color = "#0F4C5C"
    forecast_color = "#E36414"
    baseline_color = "#9A031E"
    accent_color = "#2C7DA0"

    seasonality = (
        monthly_sales.assign(month=monthly_sales["order_month"].dt.strftime("%b"))
        .groupby("month", as_index=False)["sales"]
        .mean()
    )
    seasonality["month_order"] = pd.to_datetime(seasonality["month"], format="%b").dt.month
    seasonality = seasonality.sort_values("month_order")

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.patch.set_facecolor("#FAF7F2")

    axes[0, 0].plot(monthly_sales["order_month"], monthly_sales["sales"], color=history_color, linewidth=2)
    axes[0, 0].plot(
        monthly_sales["order_month"],
        monthly_sales["sales"].rolling(window=3).mean(),
        color=forecast_color,
        linewidth=2,
        linestyle="--",
    )
    axes[0, 0].set_title("Monthly Sales Trend")
    axes[0, 0].set_ylabel("Sales ($)")
    axes[0, 0].legend(["Monthly sales", "3-month rolling mean"], frameon=False)

    axes[0, 1].bar(seasonality["month"], seasonality["sales"], color=accent_color)
    axes[0, 1].set_title("Average Sales by Month")
    axes[0, 1].set_ylabel("Average sales ($)")
    axes[0, 1].tick_params(axis="x", rotation=45)

    axes[1, 0].plot(evaluation_df["order_month"], evaluation_df["actual_sales"], color=history_color, linewidth=2)
    axes[1, 0].plot(evaluation_df["order_month"], evaluation_df["predicted_sales"], color=forecast_color, linewidth=2)
    axes[1, 0].plot(
        evaluation_df["order_month"],
        evaluation_df["seasonal_baseline_sales"],
        color=baseline_color,
        linewidth=2,
        linestyle=":",
    )
    axes[1, 0].set_title("Holdout Performance")
    axes[1, 0].set_ylabel("Sales ($)")
    axes[1, 0].legend(["Actual", "Model prediction", "Seasonal baseline"], frameon=False)

    axes[1, 1].bar(
        future_forecast["order_month"].dt.strftime("%b %Y"),
        future_forecast["forecast_sales"],
        color=forecast_color,
    )
    axes[1, 1].set_title("Next 6-Month Forecast")
    axes[1, 1].set_ylabel("Forecast sales ($)")
    axes[1, 1].tick_params(axis="x", rotation=45)

    for axis in axes.flat:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_dir / "forecast_dashboard.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor("#FAF7F2")
    axis.plot(monthly_sales["order_month"], monthly_sales["sales"], color=history_color, linewidth=2.5, label="Historical sales")
    axis.plot(future_forecast["order_month"], future_forecast["forecast_sales"], color=forecast_color, linewidth=2.5, label="Forecast")
    axis.axvspan(
        future_forecast["order_month"].min() - MonthBegin(1),
        future_forecast["order_month"].max() + MonthBegin(1),
        color="#FEC89A",
        alpha=0.35,
        label="Forecast window",
    )
    axis.set_title("Sales Forecast Outlook")
    axis.set_ylabel("Sales ($)")
    axis.legend(frameon=False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_dir / "sales_forecast_outlook.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_forecast(
    input_path: Path,
    processed_dir: Path,
    output_dir: Path,
    forecast_horizon: int,
    holdout_months: int,
) -> None:
    monthly_sales, summary = load_sales_data(input_path)
    if len(monthly_sales) < 24:
        raise ValueError("The model needs at least 24 monthly observations to learn trend and seasonality.")

    supervised_df = create_supervised_frame(monthly_sales)
    if len(supervised_df) <= holdout_months:
        raise ValueError("Not enough monthly observations left after feature engineering for the requested holdout window.")

    train_df = supervised_df.iloc[:-holdout_months].copy()
    test_df = supervised_df.iloc[-holdout_months:].copy()

    model = Ridge(alpha=1.0)
    model.fit(train_df[FEATURE_COLUMNS], train_df["sales"])

    history_before_test = monthly_sales["sales"].iloc[:-holdout_months].tolist()
    holdout_dates = test_df["order_month"].tolist()
    holdout_predictions = recursive_forecast(model, history_before_test, holdout_dates)
    holdout_baseline = seasonal_naive_forecast(history_before_test, holdout_months)

    actual_holdout = test_df["sales"].to_numpy()
    metrics = calculate_metrics(actual_holdout, holdout_predictions)
    baseline_metrics = calculate_metrics(actual_holdout, holdout_baseline)

    evaluation_df = pd.DataFrame(
        {
            "order_month": holdout_dates,
            "actual_sales": actual_holdout,
            "predicted_sales": np.round(holdout_predictions, 2),
            "seasonal_baseline_sales": np.round(holdout_baseline, 2),
        }
    )
    evaluation_df["absolute_error"] = (evaluation_df["actual_sales"] - evaluation_df["predicted_sales"]).abs().round(2)
    evaluation_df["absolute_percentage_error"] = (
        np.where(
            evaluation_df["actual_sales"] == 0,
            0.0,
            evaluation_df["absolute_error"] / evaluation_df["actual_sales"] * 100,
        )
    ).round(2)

    final_model = Ridge(alpha=1.0)
    final_model.fit(supervised_df[FEATURE_COLUMNS], supervised_df["sales"])

    last_observed_month = monthly_sales["order_month"].max()
    future_dates = pd.date_range(
        start=last_observed_month + MonthBegin(1),
        periods=forecast_horizon,
        freq="MS",
    )
    future_predictions = recursive_forecast(final_model, monthly_sales["sales"].tolist(), future_dates.tolist())

    future_forecast = pd.DataFrame(
        {
            "order_month": future_dates,
            "forecast_sales": np.round(future_predictions, 2),
        }
    )

    processed_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    monthly_sales.to_csv(processed_dir / "monthly_sales.csv", index=False)
    evaluation_df.to_csv(output_dir / "holdout_predictions.csv", index=False)
    future_forecast.to_csv(output_dir / "future_forecast.csv", index=False)

    metrics_payload = {
        "model_metrics": {key: round(value, 4) for key, value in metrics.items()},
        "seasonal_baseline_metrics": {key: round(value, 4) for key, value in baseline_metrics.items()},
        "data_summary": asdict(summary),
    }
    (output_dir / "model_metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    summary_text = build_business_summary(summary, metrics, baseline_metrics, monthly_sales, future_forecast)
    (output_dir / "forecast_summary.md").write_text(summary_text, encoding="utf-8")

    create_visualizations(monthly_sales, evaluation_df, future_forecast, output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Forecast monthly Superstore sales using trend and seasonality features.")
    parser.add_argument("--input", default="data/raw/superstore.csv", help="Path to the raw sales CSV file.")
    parser.add_argument("--processed-dir", default="data/processed", help="Directory for cleaned monthly data.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for forecasts, plots, and metrics.")
    parser.add_argument("--forecast-horizon", type=int, default=6, help="How many future months to forecast.")
    parser.add_argument("--holdout-months", type=int, default=6, help="How many months to reserve for evaluation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_forecast(
        input_path=resolve_project_path(args.input),
        processed_dir=resolve_project_path(args.processed_dir),
        output_dir=resolve_project_path(args.output_dir),
        forecast_horizon=args.forecast_horizon,
        holdout_months=args.holdout_months,
    )


if __name__ == "__main__":
    main()
