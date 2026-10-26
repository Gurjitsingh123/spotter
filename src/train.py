from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from features import FeatureBuilder, daily_market_levels, find_bad_prices

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
REPORTS_DIR = PROJECT_DIR / "reports"
RANDOM_SEED = 42
MODEL_SETTINGS = dict(
    n_estimators=1500,
    learning_rate=0.03,
    num_leaves=63,
    min_child_samples=30,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    random_state=RANDOM_SEED,
    verbose=-1,
)
TEST_MONTHS = ["2025-07", "2025-08", "2025-09", "2025-10"]
UNSEEN_CITY_COUNT = 8


def train_model(training_loads: pd.DataFrame, market_by_day: pd.DataFrame):
    good_loads = training_loads[~find_bad_prices(training_loads)]
    builder = FeatureBuilder().fit(good_loads)
    features = builder.transform(good_loads, market_by_day)
    rate_per_mile = good_loads["posted_rate"].values / features["distance"].values
    model = lgb.LGBMRegressor(**MODEL_SETTINGS).fit(features, rate_per_mile)
    return builder, model


def predict_rates(builder, model, loads: pd.DataFrame, market_by_day: pd.DataFrame) -> np.ndarray:
    features = builder.transform(loads, market_by_day)
    return model.predict(features) * features["distance"].values


def score(actual: np.ndarray, predicted: np.ndarray, good_rows: np.ndarray) -> dict:
    errors = np.abs(predicted - actual)
    return {
        "MAE_all_rows": round(float(errors.mean()), 2),
        "MAE_clean_rows": round(float(errors[good_rows].mean()), 2),
        "MAPE_clean_rows_pct": round(float(np.mean(errors[good_rows] / actual[good_rows]) * 100), 2),
        "RMSE_clean_rows": round(float(np.sqrt(np.mean(errors[good_rows] ** 2))), 2),
        "rows": int(len(actual)),
    }


def test_month_by_month(history: pd.DataFrame, market_by_day: pd.DataFrame) -> list[dict]:
    results = []
    for month in TEST_MONTHS:
        month_start = pd.Timestamp(month + "-01")
        month_end = month_start + pd.offsets.MonthBegin(1)
        earlier_loads = history[history.date < month_start]
        month_loads = history[(history.date >= month_start) & (history.date < month_end)]
        builder, model = train_model(earlier_loads, market_by_day)
        predicted = predict_rates(builder, model, month_loads, market_by_day)
        result = score(month_loads.posted_rate.values, predicted, ~find_bad_prices(month_loads).values)
        result["test_month"] = month
        results.append(result)
        print(f"test month {month}: {result}")
    return results


def test_unseen_cities(history: pd.DataFrame, market_by_day: pd.DataFrame) -> dict:
    random = np.random.default_rng(RANDOM_SEED)
    all_cities = np.array(sorted(set(history.pickup) | set(history.delivery)))
    hidden_cities = set(random.choice(all_cities, size=UNSEEN_CITY_COUNT, replace=False))
    touches_hidden_city = history.pickup.isin(hidden_cities) | history.delivery.isin(hidden_cities)
    is_recent = history.date >= "2025-09-01"

    builder, model = train_model(history[~is_recent & ~touches_hidden_city], market_by_day)
    test_loads = history[is_recent & touches_hidden_city]
    predicted = predict_rates(builder, model, test_loads, market_by_day)
    result = score(test_loads.posted_rate.values, predicted, ~find_bad_prices(test_loads).values)
    result["held_out_cities"] = sorted(str(city) for city in hidden_cities)
    print("unseen cities:", result)
    return result


def save_report_charts(history: pd.DataFrame) -> None:
    good_loads = history[~find_bad_prices(history)].assign(rate_per_mile=lambda loads: loads.posted_rate / loads.distance)
    by_day = good_loads.groupby("date").agg(rate_per_mile=("rate_per_mile", "median"), market_index=("market_index", "mean"))

    figure, price_axis = plt.subplots(figsize=(10, 4), dpi=150)
    price_axis.plot(by_day.index, by_day.rate_per_mile, color="#064A56", lw=1.2, label="median $/mile (clean loads)")
    market_axis = price_axis.twinx()
    market_axis.plot(by_day.index, by_day.market_index, color="#C2410C", lw=1, alpha=0.7, label="market_index (daily mean)")
    price_axis.set_ylabel("$/mile")
    market_axis.set_ylabel("market_index")
    price_axis.set_title("Daily price per mile follows the market index", loc="left")
    figure.legend(loc="upper right")
    figure.tight_layout()
    figure.savefig(REPORTS_DIR / "rpm_vs_market_index.png")
    plt.close(figure)

    raw_rate_per_mile = history.posted_rate / history.distance
    figure, axis = plt.subplots(figsize=(8, 3.6), dpi=150)
    axis.hist(raw_rate_per_mile.clip(upper=8), bins=120, color="#064A56")
    axis.set_yscale("log")
    axis.set_xlabel("$/mile (clipped at 8)")
    axis.set_title("Rate per mile — the long tails are corrupt labels", loc="left")
    figure.tight_layout()
    figure.savefig(REPORTS_DIR / "rate_per_mile_distribution.png")
    plt.close(figure)


def main() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    history = pd.read_csv(DATA_DIR / "train_test.csv", parse_dates=["date"])
    validation = pd.read_csv(DATA_DIR / "validation.csv", parse_dates=["date"])
    december = pd.read_csv(DATA_DIR / "december_chart_inputs.csv")
    market_by_day = daily_market_levels(history, validation)

    report = {
        "data": {
            "train_rows": len(history),
            "train_dates": [str(history.date.min().date()), str(history.date.max().date())],
            "validation_rows": len(validation),
            "validation_dates": [str(validation.date.min().date()), str(validation.date.max().date())],
            "label_outliers_removed": int(find_bad_prices(history).sum()),
            "negative_weights_fixed": int((history.weight < 0).sum()),
            "missing_weight": int(history.weight.isna().sum()),
            "missing_market_index": int(history.market_index.isna().sum()),
        },
        "rolling_time_split": test_month_by_month(history, market_by_day),
        "unseen_city_split": test_unseen_cities(history, market_by_day),
    }

    builder, model = train_model(history, market_by_day)

    validation_rates = predict_rates(builder, model, validation, market_by_day)
    submission = pd.read_csv(DATA_DIR / "validation_predictions_template.csv")
    submission["predicted_rate"] = submission["load_id"].map(pd.Series(validation_rates, index=validation.load_id)).round(2)
    submission.to_csv(PROJECT_DIR / "validation_predictions.csv", index=False)

    december_loads = builder.add_city_coordinates(december)
    december_loads["market_index"] = np.nan
    december_loads["quote_signal"] = np.nan
    december["predicted_rate"] = np.round(predict_rates(builder, model, december_loads, market_by_day), 2)
    december.to_csv(DATA_DIR / "december_chart_inputs.csv", index=False)

    feature_names = builder.transform(history.head(5), market_by_day).columns
    importance = pd.Series(model.booster_.feature_importance("gain"), index=feature_names)
    report["feature_importance_gain_pct"] = (importance / importance.sum() * 100).round(2).sort_values(ascending=False).to_dict()
    report["december_predictions"] = (
        december[["date", "predicted_rate"]].assign(date=lambda rows: rows.date.astype(str)).to_dict("records")
    )
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(report, indent=2))

    save_report_charts(history)
    print("\nWrote validation_predictions.csv, data/december_chart_inputs.csv and reports/metrics.json")


if __name__ == "__main__":
    main()
